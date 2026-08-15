from __future__ import annotations

import pytest
from a2a.types import Task

from secure_mediation_agent.merchant.service import MerchantStartResult
from secure_mediation_agent.payment_profiles.a2a import payment_required_task
from secure_mediation_agent.workflow.controller import Identity, WorkflowController
from secure_mediation_agent.workflow.errors import DomainError
from secure_mediation_agent.workflow.models import MessagePart, WorkflowRequest


pytestmark = [pytest.mark.unit, pytest.mark.security]


IDENTITY = Identity("demo-tenant", "demo-customer")


def _create(controller: WorkflowController, suffix: str, *, paid: bool = True):
    return controller.create(
        WorkflowRequest(goal=f"acceptance {suffix}", paymentRequired=paid),
        identity=IDENTITY,
        session_id=f"session-{suffix}",
        context_id=f"context-{suffix}",
        idempotency_key=f"create-{suffix}",
    )


@pytest.mark.parametrize(
    "parts",
    [
        [MessagePart(kind="text", text=" 承認")],
        [MessagePart(kind="text", text="承認 ")],
        [MessagePart(kind="text", text="承認\n")],
        [MessagePart(kind="text", text="承認します")],
        [MessagePart(kind="text", text="yes")],
        [MessagePart(kind="text", text="承 認")],
        [MessagePart(kind="text", text="承認承認")],
        [MessagePart(kind="text", text="承認"), MessagePart(kind="text", text="承認")],
    ],
)
def test_all_non_exact_approval_variants_have_zero_business_effects(
    workflow_fixture, parts: list[MessagePart]
) -> None:
    controller = workflow_fixture["runtime"].controller
    view = _create(controller, f"variant-{abs(hash(tuple(item.text for item in parts)))}")
    with pytest.raises(DomainError) as raised:
        controller.message(
            view.workflow_id,
            parts,
            identity=IDENTITY,
            message_id=f"message-{view.workflow_id}",
            idempotency_key=f"invalid-{view.workflow_id}",
        )
    assert raised.value.code == "APPROVAL_EXACT_TOKEN_REQUIRED"
    assert workflow_fixture["repository"].counts(view.workflow_id) == {
        "planApprovals": 0,
        "paymentApprovals": 0,
        "paymentArtifacts": 0,
        "settlements": 0,
        "refunds": 0,
    }


def test_free_workflow_regression_has_no_payment_side_effects(workflow_fixture) -> None:
    controller = workflow_fixture["runtime"].controller
    view = _create(controller, "free", paid=False)
    completed = controller.message(
        view.workflow_id,
        [MessagePart(kind="text", text="承認")],
        identity=IDENTITY,
        message_id="free-approval",
        idempotency_key="free-approval-key",
    )
    assert completed.state == "completed"
    counts = workflow_fixture["repository"].counts(view.workflow_id)
    assert counts["paymentApprovals"] == 0
    assert counts["paymentArtifacts"] == 0
    assert counts["settlements"] == 0


class _DriftMerchant:
    def __init__(self, delegate, *, activation: str, requirements: dict[str, object]):
        self._delegate = delegate
        self._activation = activation
        self._requirements = requirements

    def agent_card(self):
        return self._delegate.agent_card()

    def start_task(self, **values):
        task: Task = payment_required_task(
            task_id=values["task_id"],
            context_id=values["context_id"],
            message_id=f"message:drift:{values['task_id']}",
            required=self._requirements,
            project={},
        )
        return MerchantStartResult(
            task=task,
            checkout_jwt="not-observed-before-drift-rejection",
            checkout_hash="not-observed-before-drift-rejection",
            requirements=self._requirements,
            activation_echo=self._activation,
            checkout_challenge="unused",
            payment_challenge="unused",
        )


@pytest.mark.parametrize("drift", ["activation", "constraints"])
def test_selected_profile_drift_requires_replan_before_payment_ui(
    workflow_fixture, drift: str
) -> None:
    base = workflow_fixture["runtime"].controller
    requirements = base.profile.build_required(amount=1250)
    activation = base.profile.extension_uri
    if drift == "activation":
        activation = "urn:wrong"
    else:
        requirements = {
            **requirements,
            "accepts": [{**requirements["accepts"][0], "maxAmountRequired": "1251"}],
        }
    controller = WorkflowController(
        workflow_fixture["repository"],
        workflow_fixture["keys"],
        merchant=_DriftMerchant(base.merchant, activation=activation, requirements=requirements),
    )
    view = _create(controller, f"drift-{drift}")
    replanned = controller.message(
        view.workflow_id,
        [MessagePart(kind="text", text="承認")],
        identity=IDENTITY,
        message_id=f"drift-{drift}",
        idempotency_key=f"drift-{drift}-key",
    )
    assert replanned.state == "replan_required"
    assert replanned.pending_approval is None
    counts = workflow_fixture["repository"].counts(view.workflow_id)
    assert counts["paymentApprovals"] == 0
    assert counts["paymentArtifacts"] == 0
    assert counts["settlements"] == 0
