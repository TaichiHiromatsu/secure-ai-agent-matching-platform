from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from secure_mediation_agent.workflow.controller import Identity
from secure_mediation_agent.workflow.errors import DomainError
from secure_mediation_agent.workflow.models import MessagePart, WorkflowRequest


pytestmark = pytest.mark.concurrency


def test_parallel_create_and_approval_have_one_business_effect(workflow_fixture) -> None:
    controller = workflow_fixture["runtime"].controller
    identity = Identity("demo-tenant", "demo-customer")

    def create():
        return controller.create(
            WorkflowRequest(goal="parallel demo"),
            identity=identity,
            session_id="parallel-session",
            context_id="parallel-context",
            idempotency_key="parallel-create-key",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        created = list(pool.map(lambda _: create(), range(2)))
    assert created[0].workflow_id == created[1].workflow_id

    def approve():
        return controller.message(
            created[0].workflow_id,
            [MessagePart(kind="text", text="承認")],
            identity=identity,
            message_id="parallel-plan-message",
            idempotency_key="parallel-plan-key",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        approved = list(pool.map(lambda _: approve(), range(2)))
    assert {item.state for item in approved} == {"payment_approval_required"}
    counts = controller.repository.counts(created[0].workflow_id)
    assert counts["planApprovals"] == 1
    assert counts["paymentApprovals"] == 0
    assert counts["settlements"] == 0


def test_stale_expected_version_has_zero_approval_effect(workflow_fixture) -> None:
    controller = workflow_fixture["runtime"].controller
    identity = Identity("demo-tenant", "demo-customer")
    view = controller.create(
        WorkflowRequest(goal="stale version"),
        identity=identity,
        session_id="stale-session",
        context_id="stale-context",
        idempotency_key="stale-create",
    )
    with pytest.raises(DomainError, match="changed concurrently"):
        controller.message(
            view.workflow_id,
            [MessagePart(kind="text", text="承認")],
            identity=identity,
            message_id="stale-plan-message",
            idempotency_key="stale-plan-key",
            expected_version=view.version + 1,
        )
    assert controller.repository.counts(view.workflow_id)["planApprovals"] == 0
