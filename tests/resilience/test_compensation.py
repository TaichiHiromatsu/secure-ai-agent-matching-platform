from __future__ import annotations

import pytest

from secure_mediation_agent.workflow.controller import Identity, WorkflowController
from secure_mediation_agent.workflow.errors import DomainError
from secure_mediation_agent.workflow.models import MessagePart, WorkflowRequest


pytestmark = [pytest.mark.integration, pytest.mark.restart]


def _run(controller: WorkflowController, *, suffix: str):
    identity = Identity("demo-tenant", "demo-customer")
    view = controller.create(
        WorkflowRequest(goal=f"fault demo {suffix}"),
        identity=identity,
        session_id=f"session-{suffix}",
        context_id=f"context-{suffix}",
        idempotency_key=f"create-{suffix}",
    )
    payment = controller.message(
        view.workflow_id,
        [MessagePart(kind="text", text="承認")],
        identity=identity,
        message_id=f"plan-{suffix}",
        idempotency_key=f"plan-{suffix}",
    )
    terminal = controller.message(
        view.workflow_id,
        [MessagePart(kind="text", text="承認")],
        identity=identity,
        message_id=f"payment-{suffix}",
        idempotency_key=f"payment-{suffix}",
    )
    return terminal


def test_definitive_settlement_failure_has_error_receipt_and_no_debit(workflow_fixture) -> None:
    controller = WorkflowController(
        workflow_fixture["repository"], workflow_fixture["keys"], rail_fault="failed"
    )
    failed = _run(controller, suffix="failed")
    assert failed.state == "payment_failed"
    assert controller.repository.rail_balance("demo-customer") == 100_000
    assert controller.repository.rail_balance("demo-merchant") == 0
    refs = controller.repository.artifact_refs(failed.workflow_id)
    assert [item["kind"] for item in refs].count("payment-receipt") == 1


def test_unknown_settlement_never_recharges_and_requires_reconciliation(workflow_fixture) -> None:
    controller = WorkflowController(
        workflow_fixture["repository"], workflow_fixture["keys"], rail_fault="unknown"
    )
    unknown = _run(controller, suffix="unknown")
    assert unknown.state == "reconciliation_required"
    assert controller.repository.counts(unknown.workflow_id)["settlements"] == 1
    assert controller.repository.rail_balance("demo-customer") == 100_000
    assert controller.repository.rail_balance("demo-merchant") == 0
    failed = controller.reconcile_unknown(
        unknown.workflow_id,
        operator_id="demo-operator",
        idempotency_key="reconcile-settlement-failed",
        observed_state="failed",
    )
    assert failed.state == "payment_failed"
    assert controller.repository.counts(unknown.workflow_id)["settlements"] == 1
    assert controller.repository.rail_balance("demo-customer") == 100_000


def test_late_settlement_is_refunded_without_a_second_charge(workflow_fixture) -> None:
    controller = WorkflowController(
        workflow_fixture["repository"], workflow_fixture["keys"], rail_fault="unknown"
    )
    unknown = _run(controller, suffix="late-settlement")
    required = controller.reconcile_unknown(
        unknown.workflow_id,
        operator_id="demo-operator",
        idempotency_key="reconcile-settlement-success",
        observed_state="settled",
    )
    assert required.state == "refund_required"
    assert controller.repository.counts(unknown.workflow_id)["settlements"] == 1
    assert controller.repository.counts(unknown.workflow_id)["refunds"] == 1
    assert controller.repository.rail_balance("demo-customer") == 98_750
    refunded = controller.execute_required_refund(
        unknown.workflow_id,
        operator_id="demo-operator",
        idempotency_key="late-settlement-refund",
    )
    assert refunded.state == "refunded"
    assert controller.repository.rail_balance("demo-customer") == 100_000


def test_commit_failure_appends_refund_and_preserves_original_evidence(workflow_fixture) -> None:
    controller = WorkflowController(
        workflow_fixture["repository"], workflow_fixture["keys"], commit_fault=True
    )
    required = _run(controller, suffix="refund")
    assert required.state == "refund_required"
    assert controller.repository.counts(required.workflow_id)["refunds"] == 1
    before = controller.repository.artifact_refs(required.workflow_id)
    before_digests = [(item["evidence_id"], item["evidence_digest"]) for item in before]
    assert controller.repository.rail_balance("demo-customer") == 98_750
    assert controller.repository.rail_balance("demo-merchant") == 1_250

    with pytest.raises(DomainError, match="Authorized demo operator"):
        controller.execute_required_refund(
            required.workflow_id,
            operator_id="attacker",
            idempotency_key="refund-op",
        )
    refunded = controller.execute_required_refund(
        required.workflow_id,
        operator_id="demo-operator",
        idempotency_key="refund-op",
    )
    assert refunded.state == "refunded"
    assert controller.repository.rail_balance("demo-customer") == 100_000
    assert controller.repository.rail_balance("demo-merchant") == 0
    repeated = controller.repository.refund_simulation(
        workflow_id=required.workflow_id,
        idempotency_key="refund-op",
    )
    assert repeated["applied"] == 1
    assert controller.repository.rail_balance("demo-customer") == 100_000
    after = controller.repository.artifact_refs(required.workflow_id)
    assert before_digests == [
        (item["evidence_id"], item["evidence_digest"]) for item in after
    ]


def test_unknown_refund_reconciles_same_external_id(workflow_fixture) -> None:
    controller = WorkflowController(
        workflow_fixture["repository"], workflow_fixture["keys"], commit_fault=True
    )
    required = _run(controller, suffix="refund-unknown")
    unknown = controller.execute_required_refund(
        required.workflow_id,
        operator_id="demo-operator",
        idempotency_key="refund-unknown-op",
        outcome="unknown",
    )
    assert unknown.state == "reconciliation_required"
    reconciled = controller.reconcile_unknown(
        required.workflow_id,
        operator_id="demo-operator",
        idempotency_key="reconcile-refund-success",
        observed_state="settled",
    )
    assert reconciled.state == "refunded"
    assert controller.repository.counts(required.workflow_id)["settlements"] == 1
    assert controller.repository.counts(required.workflow_id)["refunds"] == 1
    assert controller.repository.rail_balance("demo-customer") == 100_000
