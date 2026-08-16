from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from a2a.types import Message, Part, Role, Task, TaskState, TaskStatus, TextPart

from secure_mediation_agent.merchant.service import PaidBookingMerchant
from secure_mediation_agent.payment_bridge import (
    BridgeState,
    PaymentA2AOperation,
    PaymentBridge,
    PaymentSubmissionRejected,
)
from secure_mediation_agent.payment_profiles.a2a import (
    payment_message,
    payment_required_task,
)
from secure_mediation_agent.payment_profiles.registry import ProfileRegistry
from secure_mediation_agent.ap2.verification import b64url_sha256
from secure_mediation_agent.workflow.canonical import canonical_digest, sha256_digest
from secure_mediation_agent.workflow.errors import DomainError


OWNER = {
    "tenantId": "tenant-1",
    "subjectId": "subject-1",
    "sessionId": "adk-session-1",
    "contextId": "mediation-context-1",
    "mediationSessionId": "mediation-session-1",
}
PLAN = {
    "planId": "plan-1",
    "planVersion": 1,
    "planDigest": sha256_digest("approved-plan"),
    "approvalId": "plan-approval-1",
}
STEP = {
    "stepId": "step-paid-1",
    "canonicalAgentId": "agent-005",
    "agentCardDigest": sha256_digest("paid-booking-agent-card"),
    "rpcEndpoint": "http://127.0.0.1:8005/a2a",
    "skillId": "paid-booking",
}


class DeterministicExecutor:
    def __init__(self, reject_phase: str | None = None) -> None:
        self.reject_phase = reject_phase
        self.operations: list[PaymentA2AOperation] = []

    def execute(self, operation: PaymentA2AOperation) -> Task:
        self.operations.append(operation)
        if operation.phase == self.reject_phase:
            raise PaymentSubmissionRejected(f"rejected:{operation.phase}")
        state = (
            TaskState.working
            if operation.phase == "guarantee-submit"
            else TaskState.completed
        )
        response = Message(
            messageId=f"response:{operation.operation_id}",
            taskId=operation.task_id,
            contextId=operation.context_id,
            role=Role.agent,
            parts=[Part(root=TextPart(text=operation.phase))],
        )
        return Task(
            id=operation.task_id,
            contextId=operation.context_id,
            status=TaskStatus(state=state, message=response),
            history=[operation.message, response],
        )


def _requirement(workflow_fixture, *, task_id: str = "remote-task-1") -> dict:
    profile = ProfileRegistry.load(
        "x402-wire-simulation/1",
        simulation_key=workflow_fixture["keys"].simulation_signer,
    )
    required = profile.build_required(amount=1250)
    checkout_jwt = "private.checkout.jwt"
    return {
        "schemaVersion": "payment-requirement-snapshot/1",
        "taskId": task_id,
        "contextId": "remote-context-1",
        "orderId": "order-1",
        "quoteId": "quote-1",
        "paymentRequired": required,
        "requirementDigest": canonical_digest(required),
        "checkoutJwt": checkout_jwt,
        "checkoutHash": b64url_sha256(checkout_jwt),
        "amountMinor": 1250,
        "currency": "USD",
        "payee": "demo-merchant",
        "profileId": "x402-wire-simulation/1",
        "extensionUri": "urn:secure-a2a:extensions:x402-wire-simulation:v1",
        "checkoutAudience": "demo-merchant",
        "checkoutNonce": "checkout-nonce-1234567890",
        "paymentAudience": "demo-credential-provider",
        "paymentNonce": "payment-nonce-1234567890",
        "expiresAt": (datetime.now(UTC) + timedelta(hours=1))
        .isoformat()
        .replace("+00:00", "Z"),
    }


def _attached_bridge(workflow_fixture):
    bridge = PaymentBridge(
        workflow_fixture["repository"], workflow_fixture["keys"]
    )
    requirement = _requirement(workflow_fixture)
    attachment = bridge.attach(
        OWNER,
        PLAN,
        STEP,
        {
            "taskId": requirement["taskId"],
            "contextId": requirement["contextId"],
            "state": "input-required",
        },
        requirement,
    )
    return bridge, attachment, requirement


def _balance(repository, account_id: str) -> int:
    with repository._connect(repository.paths.marketplace) as conn:
        return conn.execute(
            "SELECT balance FROM rail_accounts_v2 WHERE account_id=? AND asset='USD'",
            (account_id,),
        ).fetchone()[0]


def test_exact_approval_guarantee_settlement_and_same_task_completion(
    workflow_fixture,
) -> None:
    bridge, attachment, requirement = _attached_bridge(workflow_fixture)

    replay = bridge.attach(
        OWNER,
        PLAN,
        STEP,
        {"taskId": "remote-task-1", "contextId": "remote-context-1"},
        requirement,
    )
    assert replay.continuation_id == attachment.continuation_id
    assert replay.created is False
    with pytest.raises(DomainError, match="exactly match"):
        bridge.approve(
            attachment.continuation_id,
            attachment.version,
            "承認します",
            owner=OWNER,
        )
    assert bridge.counts(attachment.continuation_id)["approvals"] == 0

    approval = bridge.approve(
        attachment.continuation_id,
        attachment.version,
        "承認",
        owner=OWNER,
    )
    executor = DeterministicExecutor()
    result = bridge.execute_approved_payment(
        f"payment-submit:{attachment.continuation_id}:1",
        attachment.continuation_id,
        approval.version,
        executor,
    )

    assert result.state == BridgeState.COMPLETED
    assert [operation.phase for operation in executor.operations] == [
        "guarantee-submit",
        "fulfillment-commit",
    ]
    assert all(operation.task_id == "remote-task-1" for operation in executor.operations)
    assert all(
        operation.context_id == "remote-context-1" for operation in executor.operations
    )
    counts = bridge.counts(attachment.continuation_id)
    assert counts == {
        "continuations": 1,
        "approvals": 1,
        "guarantees": 1,
        "settlements": 1,
        "refunds": 0,
    }
    assert _balance(workflow_fixture["repository"], "demo-customer") == 98_750
    assert _balance(workflow_fixture["repository"], "demo-merchant") == 1_250

    replay_result = bridge.execute_approved_payment(
        f"payment-submit:{attachment.continuation_id}:1",
        attachment.continuation_id,
        approval.version,
        DeterministicExecutor(),
    )
    assert replay_result.result_digest == result.result_digest
    assert bridge.counts(attachment.continuation_id) == counts


def test_pre_settlement_rejection_cancels_guarantee_without_moving_money(
    workflow_fixture,
) -> None:
    bridge, attachment, _ = _attached_bridge(workflow_fixture)
    approval = bridge.approve(
        attachment.continuation_id,
        attachment.version,
        "承認",
        owner=OWNER,
    )

    with pytest.raises(DomainError) as rejected:
        bridge.execute_approved_payment(
            f"payment-submit:{attachment.continuation_id}:1",
            attachment.continuation_id,
            approval.version,
            DeterministicExecutor("guarantee-submit"),
        )

    assert rejected.value.code == "PAYMENT_FAILED"
    status = bridge.status(attachment.continuation_id, owner=OWNER)
    assert status.state == BridgeState.GUARANTEE_CANCELLED
    assert bridge.counts(attachment.continuation_id)["settlements"] == 0
    assert _balance(workflow_fixture["repository"], "demo-customer") == 100_000
    assert _balance(workflow_fixture["repository"], "demo-merchant") == 0


def test_post_settlement_failure_requires_one_full_refund(workflow_fixture) -> None:
    bridge, attachment, _ = _attached_bridge(workflow_fixture)
    approval = bridge.approve(
        attachment.continuation_id,
        attachment.version,
        "承認",
        owner=OWNER,
    )
    with pytest.raises(DomainError) as failed:
        bridge.execute_approved_payment(
            f"payment-submit:{attachment.continuation_id}:1",
            attachment.continuation_id,
            approval.version,
            DeterministicExecutor("fulfillment-commit"),
        )
    assert failed.value.code == "REFUND_REQUIRED"
    status = bridge.status(attachment.continuation_id, owner=OWNER)
    assert status.state == BridgeState.REFUND_REQUIRED

    refunded = bridge.refund(
        f"refund:{attachment.continuation_id}:1",
        attachment.continuation_id,
        status.version,
    )
    replay = bridge.refund(
        f"refund:{attachment.continuation_id}:1",
        attachment.continuation_id,
        status.version,
    )
    assert replay.refund_id == refunded.refund_id
    assert bridge.counts(attachment.continuation_id)["refunds"] == 1
    assert _balance(workflow_fixture["repository"], "demo-customer") == 100_000
    assert _balance(workflow_fixture["repository"], "demo-merchant") == 0


def test_free_result_creates_no_payment_record(workflow_fixture) -> None:
    bridge = PaymentBridge(
        workflow_fixture["repository"], workflow_fixture["keys"]
    )
    with pytest.raises(DomainError) as error:
        bridge.attach(
            OWNER,
            PLAN,
            STEP,
            {"taskId": "free-task", "contextId": "free-context"},
            None,
        )
    assert error.value.code == "PAYMENT_NOT_REQUIRED"
    with workflow_fixture["repository"]._connect(
        workflow_fixture["paths"].marketplace
    ) as conn:
        assert conn.execute("SELECT COUNT(*) FROM payment_continuations_v3").fetchone()[0] == 0


def test_merchant_verifies_guarantee_before_fulfillment(workflow_fixture) -> None:
    repository = workflow_fixture["repository"]
    keys = workflow_fixture["keys"]
    profile = ProfileRegistry.load(
        "x402-wire-simulation/1", simulation_key=keys.simulation_signer
    )
    merchant = PaidBookingMerchant(repository, keys, profile)
    required = {
        **profile.build_required(amount=1250),
        "orderId": "merchant-order-1",
        "quoteId": "merchant-quote-1",
        "expiresAt": (datetime.now(UTC) + timedelta(minutes=10))
        .isoformat()
        .replace("+00:00", "Z"),
    }
    origin = payment_required_task(
        task_id="merchant-task-1",
        context_id="merchant-context-1",
        message_id="message:required:merchant-task-1",
        required=required,
        project={"orderId": "merchant-order-1", "quoteId": "merchant-quote-1"},
    )
    repository.save_merchant_origin(
        workflow_id="merchant-workflow-1",
        task_id="merchant-task-1",
        context_id="merchant-context-1",
        order_id="merchant-order-1",
        task=origin.model_dump(mode="json", by_alias=True, exclude_none=True),
        requirements_id="requirements:merchant-task-1",
        requirements=required,
        checkout_jwt="private.checkout.jwt",
        checkout_hash="private-checkout-hash",
        capability_id="capability:merchant-start-1",
    )
    now = int(datetime.now(UTC).timestamp())
    guarantee = profile.issue_guarantee(
        {
            "guaranteeId": "guarantee-merchant-1",
            "iss": "secure-mediator-payment-authority",
            "aud": "a2a-agent:agent-005",
            "operation": "merchant.fulfillment.guarantee",
            "taskId": "merchant-task-1",
            "contextId": "merchant-context-1",
            "orderId": "merchant-order-1",
            "quoteId": "merchant-quote-1",
            "amountMinor": 1250,
            "currency": "USD",
            "payee": "demo-merchant",
            "paymentMandateDigest": sha256_digest("payment-mandate"),
            "authorizationEnvelopeDigest": sha256_digest("authorization-envelope"),
            "settlementCommitmentId": "settlement-commitment-1",
            "jti": "guarantee-merchant-1",
            "iat": now,
            "nbf": now,
            "exp": now + 600,
        }
    )
    submission = profile.build_guarantee_submission(
        guarantee=guarantee,
        guarantee_digest=sha256_digest(guarantee),
        checkout_mandate_digest=sha256_digest("checkout-mandate"),
        payment_mandate_digest=sha256_digest("payment-mandate"),
        authorization_envelope_digest=sha256_digest("authorization-envelope"),
    )
    submitted = payment_message(
        task_id="merchant-task-1",
        context_id="merchant-context-1",
        message_id="message:guarantee:merchant-task-1",
        status="payment-submitted",
        payload=submission,
        project={"orderId": "merchant-order-1", "quoteId": "merchant-quote-1"},
    )
    tampered = submitted.model_copy(deep=True)
    tampered.metadata["x402.payment.payload"]["paymentGuarantee"] += "tamper"
    tampered.metadata["x402.payment.payload"]["paymentGuaranteeDigest"] = sha256_digest(
        tampered.metadata["x402.payment.payload"]["paymentGuarantee"]
    )
    with pytest.raises(DomainError) as invalid:
        merchant.accept_guarantee(message=tampered)
    assert invalid.value.code == "PAYMENT_GUARANTEE_INVALID"

    guaranteed = merchant.accept_guarantee(message=submitted)
    assert guaranteed.id == "merchant-task-1"
    assert guaranteed.status.state == TaskState.working
    assert guaranteed.status.message.metadata["x402.payment.status"] == "payment-guaranteed"

    receipt = profile.settle_receipt(attempt_id="settlement-merchant-1", success=True)
    commit_message = payment_message(
        task_id="merchant-task-1",
        context_id="merchant-context-1",
        message_id="message:commit:merchant-task-1",
        status="payment-settled",
        payload={
            "schemaVersion": "merchant-fulfillment-commit/1",
            "guaranteeId": "guarantee-merchant-1",
            "settlementId": "settlement-merchant-1",
            "settlementReceipt": receipt,
            "settlementReceiptDigest": canonical_digest(receipt),
        },
        project={
            "orderId": "merchant-order-1",
            "quoteId": "merchant-quote-1",
            "simulated": True,
        },
    )
    committed = merchant.commit_guaranteed_fulfillment(message=commit_message)
    assert committed.id == "merchant-task-1"
    assert committed.status.state == TaskState.completed
    replay = merchant.commit_guaranteed_fulfillment(message=commit_message)
    assert replay.status.state == TaskState.completed
    with repository._connect(repository.paths.merchant) as conn:
        row = conn.execute(
            "SELECT state FROM merchant_guarantees_v3 WHERE task_id='merchant-task-1'"
        ).fetchone()
    assert row["state"] == "fulfilled"
