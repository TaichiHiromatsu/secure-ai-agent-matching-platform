from __future__ import annotations

import asyncio
import base64
import hashlib
from datetime import timedelta
from uuid import uuid4

import pytest

from secure_mediation_agent.mediation.a2a_executor import (
    A2AOperation,
    SharedA2AOperationExecutor,
)
from secure_mediation_agent.mediation.canonical import canonical_digest
from secure_mediation_agent.mediation.controller import MediationController
from secure_mediation_agent.mediation.errors import SecurityBlocked
from secure_mediation_agent.mediation.models import (
    A2AResponseEnvelope,
    BridgeApprovalResult,
    BridgeA2AExecutionSummary,
    BridgeAttachment,
    BridgeExecutionResult,
    GateDecision,
    MediationPlan,
    MediationState,
    MediationStep,
    OwnerScope,
    PaymentRequirementSnapshot,
    PrivatePaymentMaterial,
    RefundResult,
    RemoteTaskSnapshot,
    SelectedAgentSnapshot,
    SubjectScope,
    utc_now,
)
from secure_mediation_agent.mediation.store import InMemoryMediationStore


PAID_EXTENSION = "urn:secure-a2a:extensions:x402-wire-simulation:v1"


def _agent(*, paid: bool) -> SelectedAgentSnapshot:
    wire = {
        "canonicalAgentId": "agent-005" if paid else "agent-002",
        "registryName": "paid_booking_agent" if paid else "hotel_agent",
        "a2aAgentName": "paid-booking-agent" if paid else "hotel_agent",
        "agentCardUrl": (
            "http://127.0.0.1:8005/.well-known/agent-card.json"
            if paid
            else "http://127.0.0.1:8002/a2a/hotel_agent/.well-known/agent-card.json"
        ),
        "rpcEndpoint": (
            "http://127.0.0.1:8005/a2a"
            if paid
            else "http://127.0.0.1:8002/a2a/hotel_agent"
        ),
        "a2aSkillId": "paid-booking" if paid else "hotel_search",
        "trustScore": 95,
        "cardDigest": canonical_digest({"card": "paid" if paid else "free"}),
        "paymentExtensionUris": (PAID_EXTENSION,) if paid else (),
    }
    return SelectedAgentSnapshot(**wire, snapshotDigest=canonical_digest(wire))


class FakeMatcher:
    def __init__(self, selected: SelectedAgentSnapshot) -> None:
        self.selected = selected

    async def match(self, goal: str):
        return [self.selected]


class FakePlanner:
    async def create_plan(self, goal, owner: OwnerScope, candidates):
        now = utc_now()
        step = MediationStep(
            stepId="step-1",
            ordinal=1,
            selectedAgent=candidates[0],
            inputDigest=canonical_digest({"goal": goal}),
            goal=goal,
            paymentLimitMinor=5000,
            currency="USD" if candidates[0].canonical_agent_id == "agent-005" else "JPY",
        )
        return MediationPlan(
            planId=f"plan-{uuid4()}",
            planVersion=1,
            planDigest=canonical_digest(
                {
                    "owner": owner.model_dump(mode="json", by_alias=True),
                    "step": step.model_dump(mode="json", by_alias=True),
                }
            ),
            goalDigest=canonical_digest({"goal": goal}),
            owner=owner,
            steps=(step,),
            createdAt=now,
            expiresAt=now + timedelta(minutes=10),
        )


class RecordingHook:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def before(self, operation):
        self.events.append(f"before:{operation.kind}")

    async def after(self, operation, response):
        self.events.append(f"after:{operation.kind}")


class PassingGates:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def decide(self, gate_id, operation, response):
        self.events.append(gate_id)
        return GateDecision(
            gateId=gate_id,
            decision="PASS",
            decisionDigest=canonical_digest(
                {"gate": gate_id, "operation": operation.operation_id}
            ),
        )


class SequenceTransport:
    def __init__(self, responses: list[A2AResponseEnvelope]) -> None:
        self.responses = responses
        self.calls: list[A2AOperation] = []

    async def send(self, operation):
        self.calls.append(operation)
        return self.responses.pop(0)


class AcceptFinal:
    async def validate(self, session, result):
        return "ACCEPT"


class FakeBridge:
    def __init__(self) -> None:
        self.attach_calls = 0
        self.approve_calls = 0
        self.execute_calls = 0
        self.step = None
        self.remote = None
        self.executor = None
        self.refund_on_execute = False

    def attach(self, *, owner, approved_plan, step, remote_task, requirement):
        self.attach_calls += 1
        assert set(owner) == {
            "tenantId",
            "subjectId",
            "sessionId",
            "contextId",
            "mediationSessionId",
        }
        assert requirement["privatePaymentMaterial"].checkout_jwt == "secret-checkout"
        self.step = step
        self.remote = remote_task
        return BridgeAttachment(
            continuationId="continuation-1",
            paymentWorkflowId="payment-workflow-1",
            version=1,
        )

    def approve(
        self,
        *,
        owner,
        continuation_id,
        expected_version,
        approval_text,
        expected_approval_target_digest,
    ):
        self.approve_calls += 1
        assert approval_text == "承認"
        assert expected_version == 1
        assert expected_approval_target_digest.startswith("sha256:")
        return BridgeApprovalResult(
            continuationId=continuation_id,
            version=2,
            approvalDigest=canonical_digest({"approval": continuation_id}),
            state="PaymentApproved",
        )

    async def execute_approved_payment(
        self, *, operation_id, continuation_id, expected_version
    ):
        self.execute_calls += 1
        assert operation_id == "payment-submit:continuation-1:1"
        assert expected_version == 2
        request = {
            "jsonrpc": "2.0",
            "id": operation_id,
            "method": "message/send",
            "params": {"message": {"taskId": self.remote.task_id}},
        }
        execution = await self.executor.execute(
            A2AOperation(
                operationId=operation_id,
                kind="payment-submit",
                agent=self.step.selected_agent,
                request=request,
                requestDigest=canonical_digest(request),
                idempotencyKey=operation_id,
                taskId=self.remote.task_id,
                contextId=self.remote.context_id,
            )
        )
        remote = execution.response.task
        if self.refund_on_execute:
            return BridgeExecutionResult(
                continuationId=continuation_id,
                version=3,
                remoteTask=remote.model_copy(update={"state": "failed"}),
                result={"taskState": "failed", "refundEligible": True},
                state="refund-required",
            )
        return BridgeExecutionResult(
            continuationId=continuation_id,
            version=3,
            remoteTask=remote,
            result={
                "taskState": "completed",
                "refundEligible": False,
                "simulation": True,
            },
            state="same-task-completed",
            a2aExecutions=(
                BridgeA2AExecutionSummary(
                    operationId=operation_id,
                    taskDigest=remote.task_digest,
                    eventOrder=execution.event_order,
                ),
            ),
        )

    def refund(self, **kwargs):
        return RefundResult(
            refundId="refund-1",
            state="refunded",
            resultDigest=canonical_digest({"refund": 1}),
        )


def _free_envelope() -> A2AResponseEnvelope:
    task = RemoteTaskSnapshot(
        taskId="free-task-1",
        contextId="free-context-1",
        state="completed",
        taskDigest=canonical_digest({"task": "free"}),
        artifact={"answer": "free result"},
    )
    return A2AResponseEnvelope(
        task=task,
        envelopeDigest=canonical_digest(
            {"task": task.model_dump(mode="json", by_alias=True)}
        ),
    )


def _paid_envelopes() -> list[A2AResponseEnvelope]:
    checkout_jwt = "secret-checkout"
    checkout_hash = base64.urlsafe_b64encode(
        hashlib.sha256(checkout_jwt.encode()).digest()
    ).rstrip(b"=").decode()
    required = {
        "x402Version": 1,
        "accepts": [
            {
                "scheme": "exact-simulated",
                "network": "demo:local",
                "asset": "USD",
                "payTo": "merchant:demo-merchant",
                "maxAmountRequired": "1250",
            }
        ],
    }
    requirement = PaymentRequirementSnapshot(
        taskState="input-required",
        paymentStatus="payment-required",
        extensionUri=PAID_EXTENSION,
        profileId="x402-wire-simulation/1",
        orderId="order-1",
        quoteId="quote-1",
        amountMinor=1250,
        currency="USD",
        payee="demo-merchant",
        expiresAt=utc_now() + timedelta(minutes=10),
        requirementDigest=canonical_digest(required),
        checkoutDigest=canonical_digest(checkout_jwt),
        paymentRequired=required,
        checkoutAudience="demo-merchant",
        checkoutNonce="checkout-nonce-1234567890",
        paymentAudience="demo-credential-provider",
        paymentNonce="payment-nonce-1234567890",
    )
    initial = RemoteTaskSnapshot(
        taskId="paid-task-1",
        contextId="paid-context-1",
        state="input-required",
        taskDigest=canonical_digest({"task": "paid-initial"}),
        orderId="order-1",
        quoteId="quote-1",
        paymentRequirement=requirement,
    )
    completed = RemoteTaskSnapshot(
        taskId="paid-task-1",
        contextId="paid-context-1",
        state="completed",
        taskDigest=canonical_digest({"task": "paid-completed"}),
        orderId="order-1",
        quoteId="quote-1",
        artifact={"booking": "confirmed"},
    )
    return [
        A2AResponseEnvelope(
            task=initial,
            privatePaymentMaterial=PrivatePaymentMaterial(
                checkoutJwt=checkout_jwt, checkoutHash=checkout_hash
            ),
            envelopeDigest=canonical_digest({"envelope": "initial"}),
        ),
        A2AResponseEnvelope(
            task=completed,
            envelopeDigest=canonical_digest({"envelope": "completed"}),
        ),
    ]


def _controller(*, paid: bool):
    hook = RecordingHook()
    gates = PassingGates()
    transport = SequenceTransport(_paid_envelopes() if paid else [_free_envelope()])
    bridge = FakeBridge()
    executor = SharedA2AOperationExecutor(
        callback=hook, gates=gates, transport=transport
    )
    bridge.executor = executor
    controller = MediationController(
        store=InMemoryMediationStore(),
        matcher=FakeMatcher(_agent(paid=paid)),
        planner=FakePlanner(),
        executor=executor,
        gates=gates,
        payment_bridge=bridge,
        final_validator=AcceptFinal(),
    )
    return controller, hook, gates, transport, bridge


def _submit(controller, scope, text, request_id):
    return asyncio.run(
        controller.submit(
            scope=scope,
            parts=[{"kind": "text", "text": text}],
            request_id=request_id,
        )
    )


def test_free_path_requires_exact_plan_approval_and_never_calls_payment_bridge():
    controller, hook, gates, transport, bridge = _controller(paid=False)
    scope = SubjectScope(subject="alice", tenantId="tenant-a", adkSessionId="s-1")

    planned = _submit(controller, scope, "無料ホテルを検索", "request-1")
    assert planned.state == MediationState.WAITING_FOR_PLAN_APPROVAL
    assert transport.calls == []

    still_waiting = _submit(controller, scope, " 承認", "request-2")
    assert still_waiting.state == MediationState.WAITING_FOR_PLAN_APPROVAL
    assert transport.calls == []

    completed = _submit(controller, scope, "承認", "request-3")
    assert completed.state == MediationState.COMPLETED
    assert len(transport.calls) == 1
    assert bridge.attach_calls == bridge.approve_calls == bridge.execute_calls == 0
    assert hook.events == ["before:task-start", "after:task-start"]
    assert gates.events == ["PRE_A2A_START", "POST_A2A_RESPONSE"]
    assert completed.trace[-1].stage == "final-validation"
    assert completed.trace[-1].decision == "ACCEPT"


def test_paid_path_uses_same_executor_and_second_exact_approval():
    controller, hook, gates, transport, bridge = _controller(paid=True)
    scope = SubjectScope(subject="alice", tenantId="tenant-a", adkSessionId="s-2")

    planned = _submit(controller, scope, "有料予約", "paid-1")
    assert planned.state == MediationState.WAITING_FOR_PLAN_APPROVAL
    assert planned.approval_target is not None
    assert planned.approval_target.approval_kind == "plan"
    assert planned.approval_target.plan_id.startswith("plan-")
    assert planned.approval_target.plan_version == 1
    assert planned.approval_target.steps[0].goal == "有料予約"
    assert planned.approval_target.steps[0].agent.canonical_agent_id == "agent-005"
    assert planned.approval_target.steps[0].currency == "USD"
    assert planned.approval_target.steps[0].payment_limit_minor == 5000
    assert planned.approval_target.approval_token == "承認"
    assert planned.approval_target_digest == canonical_digest(
        planned.approval_target
    )
    waiting = _submit(controller, scope, "承認", "paid-2")
    assert waiting.state == MediationState.WAITING_FOR_PAYMENT_APPROVAL
    assert bridge.attach_calls == 1
    assert "1250 USD" in waiting.message
    assert "計画承認とは別" in waiting.message
    assert waiting.approval_target is not None
    assert waiting.approval_target.approval_kind == "payment"
    assert waiting.approval_target.distinct_from_plan_approval is True
    assert waiting.approval_target.product == "Demo paid booking"
    assert waiting.approval_target.payment_method == (
        "signed-simulated-payment-guarantee"
    )
    assert waiting.approval_target.scheme == "exact-simulated"
    assert waiting.approval_target.network == "demo:local"
    assert waiting.approval_target.asset == "USD"
    assert waiting.approval_target.step_ref == waiting.step_ref
    assert waiting.approval_target.task_ref == waiting.task_ref
    assert waiting.approval_target.bridge_display.amount_minor == 1250
    assert waiting.approval_target.bridge_display_digest == canonical_digest(
        waiting.approval_target.bridge_display
    )
    assert waiting.approval_target_digest == canonical_digest(
        waiting.approval_target
    )
    assert "secret-checkout" not in waiting.model_dump_json()

    not_approved = _submit(controller, scope, "承認\n", "paid-3")
    assert not_approved.state == MediationState.WAITING_FOR_PAYMENT_APPROVAL
    assert bridge.approve_calls == 0

    completed = _submit(controller, scope, "承認", "paid-4")
    assert completed.state == MediationState.COMPLETED
    assert bridge.approve_calls == bridge.execute_calls == 1
    assert [operation.kind for operation in transport.calls] == [
        "task-start",
        "payment-submit",
    ]
    assert hook.events == [
        "before:task-start",
        "after:task-start",
        "before:payment-submit",
        "after:payment-submit",
    ]
    assert gates.events == [
        "PRE_A2A_START",
        "POST_A2A_RESPONSE",
        "POST_PAYMENT_REQUIREMENT",
        "PRE_PAYMENT_SUBMIT",
        "POST_PAYMENT_RESULT",
    ]
    trace_wire = completed.model_dump_json()
    assert "secret-checkout" not in trace_wire
    assert "checkout-nonce-1234567890" not in trace_wire
    assert completed.trace[-1].stage == "final-validation"
    assert completed.trace[-1].decision == "ACCEPT"
    payment_stages = [
        event.stage
        for event in completed.trace
        if event.operation_id == "payment-submit:continuation-1:1"
    ]
    assert payment_stages == [
        "payment-submit-started",
        "legacy-callback-before",
        "PRE_PAYMENT_SUBMIT",
        "transport",
        "response-persisted",
        "legacy-callback-after",
        "POST_PAYMENT_RESULT",
        "payment-result-bound",
    ]


def test_changed_plan_invalidates_displayed_approval_before_remote_side_effects():
    controller, _, _, transport, bridge = _controller(paid=True)
    scope = SubjectScope(subject="alice", tenantId="tenant-a", adkSessionId="s-plan-mutation")
    planned = _submit(controller, scope, "有料予約", "mutation-plan-1")
    assert planned.approval_target_digest is not None

    session = controller.store.active_for(scope)
    assert session is not None
    expected_version = session.version
    changed_step = session.active_step.model_copy(update={"goal": "改変された予約"})
    session.plan = session.plan.model_copy(update={"steps": (changed_step,)})
    session.version += 1
    controller.store.compare_and_set(session, expected_version=expected_version)

    blocked = _submit(controller, scope, "承認", "mutation-plan-2")
    assert blocked.state == MediationState.BLOCKED
    assert transport.calls == []
    assert bridge.attach_calls == bridge.approve_calls == bridge.execute_calls == 0
    assert blocked.trace[-1].stage == "plan-approval-target-mismatch"


def test_changed_checkout_invalidates_payment_approval_before_bridge_side_effects():
    controller, _, _, transport, bridge = _controller(paid=True)
    scope = SubjectScope(subject="alice", tenantId="tenant-a", adkSessionId="s-checkout-mutation")
    _submit(controller, scope, "有料予約", "mutation-payment-1")
    waiting = _submit(controller, scope, "承認", "mutation-payment-2")
    assert waiting.state == MediationState.WAITING_FOR_PAYMENT_APPROVAL

    session = controller.store.active_for(scope)
    assert session is not None and session.continuation is not None
    expected_version = session.version
    changed_requirement = session.continuation.requirement.model_copy(
        update={"checkout_digest": canonical_digest({"checkout": "changed"})}
    )
    session.continuation = session.continuation.model_copy(
        update={"requirement": changed_requirement}
    )
    session.version += 1
    controller.store.compare_and_set(session, expected_version=expected_version)

    blocked = _submit(controller, scope, "承認", "mutation-payment-3")
    assert blocked.state == MediationState.BLOCKED
    assert len(transport.calls) == 1
    assert bridge.approve_calls == bridge.execute_calls == 0
    assert blocked.trace[-1].stage == "payment-approval-target-mismatch"


def test_store_owner_binding_prevents_cross_subject_session_access():
    controller, _, _, _, _ = _controller(paid=False)
    alice = SubjectScope(subject="alice", tenantId="tenant-a", adkSessionId="same")
    bob = SubjectScope(subject="bob", tenantId="tenant-a", adkSessionId="same")
    _submit(controller, alice, "alice goal", "alice-1")
    _submit(controller, bob, "bob goal", "bob-1")
    alice_session = controller.store.active_for(alice)
    assert alice_session is not None
    with pytest.raises(SecurityBlocked):
        controller.store.get(alice_session.owner.mediation_session_id, bob)


def test_settled_fulfillment_rejection_requires_exact_refund_approval():
    controller, _, _, _, bridge = _controller(paid=True)
    bridge.refund_on_execute = True
    scope = SubjectScope(subject="alice", tenantId="tenant-a", adkSessionId="s-r")
    _submit(controller, scope, "有料予約", "refund-1")
    _submit(controller, scope, "承認", "refund-2")
    pending = _submit(controller, scope, "承認", "refund-3")
    assert pending.state == MediationState.REFUND_PENDING

    unchanged = _submit(controller, scope, " 承認", "refund-4")
    assert unchanged.state == MediationState.REFUND_PENDING
    refunded = _submit(controller, scope, "承認", "refund-5")
    assert refunded.state == MediationState.REFUNDED


def test_idempotency_key_cannot_be_reused_with_different_content():
    controller, _, _, _, _ = _controller(paid=False)
    scope = SubjectScope(subject="alice", tenantId="tenant-a", adkSessionId="s-3")
    first = _submit(controller, scope, "goal", "same-id")
    replay = _submit(controller, scope, "goal", "same-id")
    assert replay == first
    with pytest.raises(Exception, match="different content"):
        _submit(controller, scope, "other", "same-id")
