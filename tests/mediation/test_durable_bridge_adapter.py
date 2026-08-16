from __future__ import annotations

import asyncio
import inspect
from datetime import timedelta

import pytest

from secure_mediation_agent.mediation.a2a_executor import A2AExecution, A2AOperation
from secure_mediation_agent.mediation.adapters import HttpxA2ATransport
from secure_mediation_agent.mediation.approval_targets import (
    build_payment_approval_target,
    build_plan_approval_target,
)
from secure_mediation_agent.mediation.canonical import canonical_digest
from secure_mediation_agent.mediation.errors import SecurityBlocked
from secure_mediation_agent.mediation.models import (
    A2AResponseEnvelope,
    GateDecision,
    MediationPlan,
    MediationStep,
    OwnerScope,
    PlanApproval,
    RemoteTaskSnapshot,
    SelectedAgentSnapshot,
    utc_now,
)
from secure_mediation_agent.mediation.payment_bridge_adapter import (
    DurablePaymentBridgeAdapter,
)
from secure_mediation_agent.merchant.service import PaidBookingMerchant
from secure_mediation_agent.payment_bridge import PaymentBridge
from secure_mediation_agent.payment_profiles.registry import ProfileRegistry


def test_approved_payment_tool_surface_accepts_only_server_lookup_keys():
    signature = inspect.signature(
        DurablePaymentBridgeAdapter.execute_approved_payment
    )
    assert list(signature.parameters) == [
        "self",
        "operation_id",
        "continuation_id",
        "expected_version",
    ]


def _selected() -> SelectedAgentSnapshot:
    wire = {
        "canonicalAgentId": "agent-005",
        "registryName": "paid_booking_agent",
        "a2aAgentName": "paid-booking-agent",
        "agentCardUrl": "http://127.0.0.1:8005/.well-known/agent-card.json",
        "rpcEndpoint": "http://127.0.0.1:8005/a2a",
        "a2aSkillId": "paid-booking",
        "trustScore": 95,
        "cardDigest": canonical_digest({"card": "paid"}),
        "paymentExtensionUris": (
            "urn:secure-a2a:extensions:x402-wire-simulation:v1",
        ),
    }
    return SelectedAgentSnapshot(**wire, snapshotDigest=canonical_digest(wire))


class _PaymentExecutor:
    def __init__(self, *, task_id: str, context_id: str, order_id: str, quote_id: str):
        self.task_id = task_id
        self.context_id = context_id
        self.order_id = order_id
        self.quote_id = quote_id
        self.actions: list[str] = []

    async def execute(self, operation: A2AOperation) -> A2AExecution:
        action = operation.request["params"]["action"]
        self.actions.append(action)
        state = (
            "working"
            if action == "merchant:payment-guarantee-submit"
            else "completed"
        )
        task = RemoteTaskSnapshot(
            taskId=self.task_id,
            contextId=self.context_id,
            state=state,
            taskDigest=canonical_digest({"action": action}),
            orderId=self.order_id,
            quoteId=self.quote_id,
            artifact={"booking": "confirmed"} if state == "completed" else None,
        )
        envelope = A2AResponseEnvelope(
            task=task,
            envelopeDigest=canonical_digest({"action": action, "state": state}),
        )
        pre, post = operation.gate_ids()
        return A2AExecution(
            operation=operation,
            response=envelope,
            preDecision=GateDecision(
                gateId=pre,
                decision="PASS",
                decisionDigest=canonical_digest({"gate": pre}),
            ),
            postDecision=GateDecision(
                gateId=post,
                decision="PASS",
                decisionDigest=canonical_digest({"gate": post}),
            ),
            eventOrder=(pre, post),
        )


def test_actual_payment_bridge_completes_via_action_mapped_adapter(workflow_fixture):
    repository = workflow_fixture["repository"]
    keys = workflow_fixture["keys"]
    profile = ProfileRegistry.load(
        "x402-wire-simulation/1", simulation_key=keys.simulation_signer
    )
    merchant = PaidBookingMerchant(repository, keys, profile)
    now = utc_now()
    issued_at = int(now.timestamp())
    started = merchant.start_task(
        workflow_id="mediation-session-1",
        plan_digest=canonical_digest({"plan": 1}),
        task_id="remote-task-1",
        order_id="order-1",
        context_id="remote-context-1",
        capability_id="start-capability-1",
        activation={profile.extension_uri},
        issued_at=issued_at,
        expires_at=issued_at + 600,
    )
    selected = _selected()
    operation = A2AOperation(
        operationId="start-1",
        kind="task-start",
        agent=selected,
        request={"jsonrpc": "2.0"},
        requestDigest=canonical_digest({"jsonrpc": "2.0"}),
        idempotencyKey="start-1",
    )
    result = {
        "task": started.task.model_dump(mode="json", by_alias=True, exclude_none=True),
        "privatePaymentMaterial": {
            "checkoutJwt": started.checkout_jwt,
            "checkoutHash": started.checkout_hash,
        },
    }
    envelope = HttpxA2ATransport._task_from_result(operation, result)
    requirement = envelope.task.payment_requirement
    assert requirement is not None

    owner = OwnerScope(
        subject="alice",
        tenantId="tenant-a",
        adkSessionId="adk-session-1",
        mediationSessionId="mediation-session-1",
    )
    step = MediationStep(
        stepId="step-1",
        ordinal=1,
        selectedAgent=selected,
        inputDigest=canonical_digest({"goal": "paid booking"}),
        goal="paid booking",
        paymentLimitMinor=5000,
        currency="USD",
    )
    plan = MediationPlan(
        planId="plan-1",
        planVersion=1,
        planDigest=canonical_digest({"plan": "approved"}),
        goalDigest=canonical_digest({"goal": "paid booking"}),
        owner=owner,
        steps=(step,),
        createdAt=now,
        expiresAt=now + timedelta(minutes=10),
    )
    approval = PlanApproval(
        approvalId="approval-plan-1",
        planId=plan.plan_id,
        planVersion=1,
        planDigest=plan.plan_digest,
        approvalTargetDigest=canonical_digest(build_plan_approval_target(plan)),
        nonce="plan-approval-nonce-1234",
        issuedAt=now,
    )
    executor = _PaymentExecutor(
        task_id=envelope.task.task_id,
        context_id=envelope.task.context_id,
        order_id=requirement.order_id,
        quote_id=requirement.quote_id,
    )
    bridge = DurablePaymentBridgeAdapter(
        PaymentBridge(repository, keys), executor=executor
    )
    owner_wire = {
        "tenantId": "tenant-a",
        "subjectId": "alice",
        "sessionId": "adk-session-1",
        "contextId": "mediation-session-1",
        "mediationSessionId": "mediation-session-1",
    }
    attached = bridge.attach(
        owner=owner_wire,
        approved_plan={"plan": plan, "approval": approval},
        step=step,
        remote_task=envelope.task,
        requirement={
            "requirement": requirement,
            "privatePaymentMaterial": envelope.private_payment_material,
        },
    )
    payment_target = build_payment_approval_target(
        plan_id=plan.plan_id,
        plan_version=plan.plan_version,
        plan_digest=plan.plan_digest,
        step_id=step.step_id,
        task_id=envelope.task.task_id,
        context_id=envelope.task.context_id,
        order_id=requirement.order_id,
        quote_id=requirement.quote_id,
        merchant=requirement.payee,
        amount_minor=requirement.amount_minor,
        currency=requirement.currency,
        profile_id=requirement.profile_id,
        expires_at=requirement.expires_at,
        payment_required=requirement.payment_required,
        requirement_digest=requirement.requirement_digest,
        checkout_digest=requirement.checkout_digest,
    )
    with pytest.raises(SecurityBlocked, match="terms changed"):
        bridge.approve(
            owner=owner_wire,
            continuation_id=attached.continuation_id,
            expected_version=attached.version,
            approval_text="承認",
            expected_approval_target_digest=canonical_digest(
                {"displayedTarget": "stale"}
            ),
        )
    assert PaymentBridge(repository, keys).counts(attached.continuation_id) == {
        "continuations": 1,
        "approvals": 0,
        "guarantees": 0,
        "settlements": 0,
        "refunds": 0,
    }

    approved = bridge.approve(
        owner=owner_wire,
        continuation_id=attached.continuation_id,
        expected_version=attached.version,
        approval_text="承認",
        expected_approval_target_digest=canonical_digest(payment_target),
    )
    persisted_approval = bridge.bridge._approval(attached.continuation_id)
    assert persisted_approval["display_digest"] == (
        payment_target.bridge_display_digest
    )
    completed = asyncio.run(
        bridge.execute_approved_payment(
            operation_id=f"payment-submit:{attached.continuation_id}:1",
            continuation_id=attached.continuation_id,
            expected_version=approved.version,
        )
    )

    assert completed.state == "same-task-completed"
    assert completed.remote_task.task_id == envelope.task.task_id
    assert completed.remote_task.context_id == envelope.task.context_id
    assert len(completed.a2a_executions) == 2
    assert executor.actions == [
        "merchant:payment-guarantee-submit",
        "merchant:guaranteed-fulfillment-commit",
    ]
    assert PaymentBridge(repository, keys).counts(attached.continuation_id) == {
        "continuations": 1,
        "approvals": 1,
        "guarantees": 1,
        "settlements": 1,
        "refunds": 0,
    }
