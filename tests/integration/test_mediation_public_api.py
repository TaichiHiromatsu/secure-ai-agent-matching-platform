from __future__ import annotations

import asyncio
import base64
import hashlib
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai import types

import secure_mediation_agent.mediation.adk_adapter as adk_adapter_module

from secure_mediation_agent.mediation.a2a_executor import A2AExecution
from secure_mediation_agent.mediation.adk_adapter import SecureMediationAdapter
from secure_mediation_agent.mediation.authority import HttpMediationAuthority
from secure_mediation_agent.mediation.canonical import canonical_digest
from secure_mediation_agent.mediation.controller import MediationController
from secure_mediation_agent.mediation.errors import MediationError
from secure_mediation_agent.mediation.models import (
    A2AResponseEnvelope,
    BridgeApprovalResult,
    BridgeAttachment,
    BridgeExecutionResult,
    GateDecision,
    MediationPlan,
    MediationStep,
    OwnerScope,
    PaymentRequirementSnapshot,
    PrivatePaymentMaterial,
    RemoteTaskSnapshot,
    SelectedAgentSnapshot,
    SubjectScope,
    TextPart,
    utc_now,
)
from secure_mediation_agent.mediation.persistence import SqliteMediationStore
from secure_mediation_agent.mediation.store import InMemoryMediationStore
from secure_mediation_agent.workflow.api import create_app


PAID_EXTENSION = "urn:secure-a2a:extensions:x402-wire-simulation:v1"


def _agent(paid: bool) -> SelectedAgentSnapshot:
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


class StaticMatcher:
    def __init__(self, paid: bool) -> None:
        self.agent = _agent(paid)

    async def match(self, goal: str):
        return (self.agent,)


class StaticPlanner:
    async def create_plan(self, goal: str, owner: OwnerScope, candidates):
        now = utc_now()
        step = MediationStep(
            stepId="step-api-1",
            ordinal=1,
            selectedAgent=candidates[0],
            inputDigest=canonical_digest({"goal": goal}),
            goal=goal,
            paymentLimitMinor=5000,
            currency="USD" if candidates[0].canonical_agent_id == "agent-005" else "JPY",
        )
        plan_wire = {
            "owner": owner.model_dump(mode="json", by_alias=True),
            "step": step.model_dump(mode="json", by_alias=True),
        }
        return MediationPlan(
            planId="plan-api-1",
            planVersion=1,
            planDigest=canonical_digest(plan_wire),
            goalDigest=canonical_digest({"goal": goal}),
            owner=owner,
            steps=(step,),
            createdAt=now,
            expiresAt=now + timedelta(minutes=10),
        )


class PassingGate:
    async def decide(self, gate_id, operation, response):
        return GateDecision(
            gateId=gate_id,
            decision="PASS",
            decisionDigest=canonical_digest(
                {"gate": gate_id, "operation": operation.operation_id}
            ),
        )


class InitialExecutor:
    def __init__(self, response: A2AResponseEnvelope) -> None:
        self.response = response
        self.calls = 0

    async def execute(self, operation):
        self.calls += 1
        pre = await PassingGate().decide("PRE_A2A_START", operation, None)
        post = await PassingGate().decide(
            "POST_A2A_RESPONSE", operation, self.response.task
        )
        return A2AExecution(
            operation=operation,
            response=self.response,
            preDecision=pre,
            postDecision=post,
            eventOrder=(
                "legacy-callback-before",
                "PRE_A2A_START",
                "transport",
                "response-persisted",
                "legacy-callback-after",
                "POST_A2A_RESPONSE",
            ),
        )


class AcceptFinal:
    async def validate(self, session, result):
        return "ACCEPT"


class ApiPaymentBridge:
    def __init__(self, completed: RemoteTaskSnapshot) -> None:
        self.completed = completed
        self.attach_calls = 0
        self.approve_calls = 0
        self.execute_calls = 0

    def attach(self, **kwargs):
        self.attach_calls += 1
        return BridgeAttachment(
            continuationId="continuation-api-1",
            paymentWorkflowId="payment-workflow-api-1",
            version=1,
        )

    def approve(self, **kwargs):
        self.approve_calls += 1
        return BridgeApprovalResult(
            continuationId="continuation-api-1",
            version=2,
            approvalDigest=canonical_digest({"approval": "api"}),
            state="PaymentApproved",
        )

    def execute_approved_payment(self, **kwargs):
        self.execute_calls += 1
        return BridgeExecutionResult(
            continuationId="continuation-api-1",
            version=3,
            remoteTask=self.completed,
            result={
                "taskState": "completed",
                "refundEligible": False,
                "simulation": True,
            },
            state="same-task-completed",
        )

    def refund(self, **kwargs):
        raise AssertionError("refund is not part of the happy path")


def _free_envelope() -> A2AResponseEnvelope:
    task = RemoteTaskSnapshot(
        taskId="free-task-api-1",
        contextId="free-context-api-1",
        state="completed",
        taskDigest=canonical_digest({"task": "free-api"}),
        artifact={"answer": "free result"},
    )
    return A2AResponseEnvelope(
        task=task,
        envelopeDigest=canonical_digest({"envelope": "free-api"}),
    )


def _paid_envelope() -> tuple[A2AResponseEnvelope, RemoteTaskSnapshot]:
    checkout_jwt = "private-checkout-api"
    checkout_hash = base64.urlsafe_b64encode(
        hashlib.sha256(checkout_jwt.encode("utf-8")).digest()
    ).rstrip(b"=").decode("ascii")
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
        orderId="order-api-1",
        quoteId="quote-api-1",
        amountMinor=1250,
        currency="USD",
        payee="demo-merchant",
        expiresAt=utc_now() + timedelta(minutes=10),
        requirementDigest=canonical_digest(required),
        checkoutDigest=canonical_digest(checkout_jwt),
        paymentRequired=required,
        checkoutAudience="demo-merchant",
        checkoutNonce="checkout-nonce-api-123456",
        paymentAudience="demo-credential-provider",
        paymentNonce="payment-nonce-api-123456",
    )
    initial = RemoteTaskSnapshot(
        taskId="paid-task-api-1",
        contextId="paid-context-api-1",
        state="input-required",
        taskDigest=canonical_digest({"task": "paid-api-initial"}),
        orderId="order-api-1",
        quoteId="quote-api-1",
        paymentRequirement=requirement,
    )
    completed = RemoteTaskSnapshot(
        taskId="paid-task-api-1",
        contextId="paid-context-api-1",
        state="completed",
        taskDigest=canonical_digest({"task": "paid-api-completed"}),
        orderId="order-api-1",
        quoteId="quote-api-1",
        artifact={"booking": "confirmed"},
    )
    return (
        A2AResponseEnvelope(
            task=initial,
            privatePaymentMaterial=PrivatePaymentMaterial(
                checkoutJwt=checkout_jwt,
                checkoutHash=checkout_hash,
            ),
            envelopeDigest=canonical_digest({"envelope": "paid-api-initial"}),
        ),
        completed,
    )


def _controller(
    paid: bool, *, store=None
) -> tuple[MediationController, ApiPaymentBridge]:
    if paid:
        initial, completed = _paid_envelope()
    else:
        initial = _free_envelope()
        completed = initial.task
    bridge = ApiPaymentBridge(completed)
    gate = PassingGate()
    return (
        MediationController(
            store=store or InMemoryMediationStore(),
            matcher=StaticMatcher(paid),
            planner=StaticPlanner(),
            executor=InitialExecutor(initial),
            gates=gate,
            payment_bridge=bridge,
            final_validator=AcceptFinal(),
        ),
        bridge,
    )


def _durable_runtime(workflow_fixture, paid: bool, key: bytes):
    controller, bridge = _controller(
        paid,
        store=SqliteMediationStore(workflow_fixture["repository"], key),
    )
    runtime = replace(
        workflow_fixture["runtime"],
        mediation_controller=controller,
        public_route_probe=lambda: True,
    )
    return runtime, controller, bridge


def _turn(
    client: TestClient,
    assertion: str,
    request_id: str,
    text: str,
    expected_version: int | None,
):
    body = {
        "schemaVersion": "mediation-turn-request/1",
        "requestId": request_id,
        "message": {"parts": [{"kind": "text", "text": text}]},
        "selectionToken": None,
    }
    if expected_version is not None:
        body["expectedVersion"] = expected_version
    return client.post(
        "/v1/turns",
        headers={
            "X-Verified-Identity": assertion,
            "Idempotency-Key": request_id,
            "X-Request-ID": request_id,
        },
        json=body,
    )


@pytest.mark.parametrize("paid", [False, True], ids=["free", "paid"])
def test_public_turn_then_view_happy_paths(workflow_fixture, paid: bool) -> None:
    controller, bridge = _controller(paid)
    runtime = replace(
        workflow_fixture["runtime"],
        mediation_controller=controller,
        public_route_probe=lambda: True,
    )
    app = create_app(runtime)
    assertion = workflow_fixture["assertion"]
    with TestClient(app) as client:
        ready = client.get(
            "/ready", headers={"X-Verified-Identity": assertion}
        )
        assert ready.status_code == 200, ready.text
        assert ready.json()["checks"]["routeIsolation"] is True
        assert ready.json()["checks"]["mediationComposition"] is True

        planned = _turn(client, assertion, f"turn-{paid}-0001", "予約", None)
        assert planned.status_code == 200, planned.text
        planned_wire = planned.json()
        assert planned_wire["state"] == "WaitingForPlanApproval"
        assert planned_wire["view"]["state"] == "WaitingForPlanApproval"
        assert "workflowId" not in planned_wire

        viewed = client.get(
            "/v1/view", headers={"X-Verified-Identity": assertion}
        )
        assert viewed.status_code == 200
        assert viewed.json() == planned_wire["view"]

        first_approval = _turn(
            client,
            assertion,
            f"turn-{paid}-0002",
            "承認",
            planned_wire["version"],
        )
        assert first_approval.status_code == 200, first_approval.text
        current = first_approval.json()
        first_approval_replay = _turn(
            client,
            assertion,
            f"turn-{paid}-0002",
            "承認",
            planned_wire["version"],
        )
        assert first_approval_replay.status_code == 200
        assert first_approval_replay.json() == current
        if paid:
            assert current["state"] == "WaitingForPaymentApproval"
            completed_response = _turn(
                client,
                assertion,
                "turn-paid-0003",
                "承認",
                current["version"],
            )
            assert completed_response.status_code == 200, completed_response.text
            current = completed_response.json()

        assert current["state"] == "Completed"
        final_view = client.get(
            "/v1/view", headers={"X-Verified-Identity": assertion}
        )
        assert final_view.status_code == 200
        assert final_view.json() == current["view"]
        assert final_view.json()["pendingAction"]["kind"] == "none"

    if paid:
        assert (
            bridge.attach_calls,
            bridge.approve_calls,
            bridge.execute_calls,
        ) == (1, 1, 1)
    else:
        assert (
            bridge.attach_calls,
            bridge.approve_calls,
            bridge.execute_calls,
        ) == (0, 0, 0)


def test_turn_body_cannot_select_identity_or_workflow(workflow_fixture) -> None:
    controller, _ = _controller(False)
    runtime = replace(
        workflow_fixture["runtime"], mediation_controller=controller
    )
    with TestClient(create_app(runtime)) as client:
        response = client.post(
            "/v1/turns",
            headers={
                "X-Verified-Identity": workflow_fixture["assertion"],
                "Idempotency-Key": "selector-0001",
            },
            json={
                "schemaVersion": "mediation-turn-request/1",
                "requestId": "selector-0001",
                "workflowId": "attacker-selected",
                "subject": "attacker-selected",
                "message": {"parts": [{"kind": "text", "text": "予約"}]},
            },
        )
    assert response.status_code == 422
    assert controller.store.latest_for(
        SubjectScope(
            subject="test-user",
            tenantId="demo-tenant",
            adkSessionId=(
                "public-"
                + hashlib.sha256(b"demo-tenant\0test-user").hexdigest()[:32]
            ),
        )
    ) is None


@pytest.mark.parametrize("paid", [False, True], ids=["free", "paid"])
def test_adk_and_workflow_routes_share_one_authoritative_session(
    workflow_fixture, monkeypatch, paid: bool
) -> None:
    """Alternate the two public-compatible transports over one controller/store."""

    controller, bridge = _controller(paid)
    runtime = replace(
        workflow_fixture["runtime"],
        mediation_controller=controller,
        public_route_probe=lambda: True,
    )
    app = create_app(runtime)
    assertion = workflow_fixture["assertion"]
    authority = HttpMediationAuthority(
        "http://workflow-authority",
        transport=httpx.ASGITransport(app=app),
    )
    adapter = SecureMediationAdapter(
        name="payment_user_agent", authority=authority
    )
    scope = SubjectScope(
        subject="test-user",
        tenantId="demo-tenant",
        adkSessionId=(
            "public-"
            + hashlib.sha256(b"demo-tenant\0test-user").hexdigest()[:32]
        ),
    )

    monkeypatch.setenv("AP2_DEMO_KEY_DIR", "/test-keys")
    monkeypatch.setattr(
        adk_adapter_module,
        "load_role_key",
        lambda directory, role: workflow_fixture["keys"].service_auth,
    )
    context = SimpleNamespace(
        invocation_id=f"adk-{paid}-invocation-0001",
        session=SimpleNamespace(
            id="browser-session-1",
            user_id="test-user",
            state={adk_adapter_module.ADK_IDENTITY_STATE_KEY: assertion},
        ),
        user_content=types.Content(
            role="user", parts=[types.Part(text="予約")]
        ),
    )

    async def run_adk_turn():
        return [event async for event in adapter._run_async_impl(context)]

    # The actual ADK adapter creates the only mediation session in workflow.
    events = asyncio.run(run_adk_turn())
    assert len(events) == 1
    planned = asyncio.run(authority.view(assertion=assertion))
    assert planned is not None
    assert planned.state.value in events[0].content.parts[0].text
    assert planned.approval_target_digest in events[0].content.parts[0].text
    session = controller.store.latest_for(scope)
    assert session is not None
    mediation_session_id = session.owner.mediation_session_id
    with TestClient(app) as client:
        workflow_view = client.get(
            "/v1/view", headers={"X-Verified-Identity": assertion}
        )
        assert workflow_view.status_code == 200
        assert workflow_view.json() == planned.model_dump(
            mode="json", by_alias=True
        )

        # The workflow turn resumes the session created through ADK.
        first = _turn(
            client,
            assertion,
            f"workflow-{paid}-turn-0002",
            "承認",
            planned.version,
        )
        assert first.status_code == 200, first.text
        current = first.json()["view"]

    adk_view = asyncio.run(authority.view(assertion=assertion))
    assert adk_view is not None
    assert adk_view.model_dump(
        mode="json", by_alias=True
    ) == current
    assert adk_view.state.value in SecureMediationAdapter._reply(adk_view)
    assert controller.store.latest_for(scope).owner.mediation_session_id == (
        mediation_session_id
    )

    if paid:
        assert adk_view.state.value == "WaitingForPaymentApproval"
        assert adk_view.approval_target is not None
        assert "計画承認とは別" in SecureMediationAdapter._reply(adk_view)
        completed = asyncio.run(
            authority.turn(
                assertion=assertion,
                parts=(TextPart(text="承認"),),
                request_id="adk-paid-turn-0003",
                expected_version=adk_view.version,
            )
        )
    else:
        completed = adk_view

    assert completed.state.value == "Completed"
    with TestClient(app) as client:
        final_view = client.get(
            "/v1/view", headers={"X-Verified-Identity": assertion}
        )
        assert final_view.status_code == 200
        assert final_view.json() == completed.model_dump(
            mode="json", by_alias=True
        )
    assert controller.store.latest_for(scope).owner.mediation_session_id == (
        mediation_session_id
    )
    assert bridge.attach_calls == bridge.approve_calls == bridge.execute_calls == (
        1 if paid else 0
    )


@pytest.mark.restart
@pytest.mark.parametrize("paid", [False, True], ids=["free", "paid"])
def test_sqlite_restores_waiting_and_terminal_turns_across_controllers(
    workflow_fixture, paid: bool
) -> None:
    """A fresh controller can replay waiting results and continue the next turn."""

    key = b"mediation-restart-test-key-0001!"
    assertion = workflow_fixture["assertion"]

    runtime, _, _ = _durable_runtime(workflow_fixture, paid, key)
    with TestClient(create_app(runtime)) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200, ready.text
        ready_wire = ready.json()
        assert ready_wire["mediationStore"] == {
            "mode": "sqlite",
            "durabilityProfile": "local-durable",
            "schemaVersion": 4,
            "writable": True,
            "decryptable": True,
        }
        assert ready_wire["checks"]["mediationStoreMode"] is True
        assert ready_wire["checks"]["mediationStoreSchema"] is True
        assert ready_wire["checks"]["mediationStoreProbe"] is True
        planned_response = _turn(
            client, assertion, f"restart-{paid}-0001", "予約", None
        )
        assert planned_response.status_code == 200, planned_response.text
        planned = planned_response.json()
        assert planned["state"] == "WaitingForPlanApproval"
        assert planned["view"]["durabilityProfile"] == "local-durable"

    runtime, waiting_controller, _ = _durable_runtime(
        workflow_fixture, paid, key
    )
    with TestClient(create_app(runtime)) as client:
        restored = client.get(
            "/v1/view", headers={"X-Verified-Identity": assertion}
        )
        assert restored.status_code == 200
        assert restored.json() == planned["view"]
        waiting_response = _turn(
            client,
            assertion,
            f"restart-{paid}-0002",
            "確認中",
            planned["version"],
        )
        assert waiting_response.status_code == 200, waiting_response.text
        waiting = waiting_response.json()
        assert waiting["view"] == planned["view"]
        assert waiting_controller.executor.calls == 0

    runtime, replay_controller, _ = _durable_runtime(
        workflow_fixture, paid, key
    )
    with TestClient(create_app(runtime)) as client:
        waiting_replay = _turn(
            client,
            assertion,
            f"restart-{paid}-0002",
            "確認中",
            planned["version"],
        )
        assert waiting_replay.status_code == 200
        assert waiting_replay.json() == waiting
        assert replay_controller.executor.calls == 0

        digest_conflict = _turn(
            client,
            assertion,
            f"restart-{paid}-0002",
            "別の内容",
            planned["version"],
        )
        assert digest_conflict.status_code == 409
        assert digest_conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
        assert replay_controller.executor.calls == 0

        first_approval_response = _turn(
            client,
            assertion,
            f"restart-{paid}-0003",
            "承認",
            planned["version"],
        )
        assert first_approval_response.status_code == 200, first_approval_response.text
        first_approval = first_approval_response.json()

    if paid:
        assert first_approval["state"] == "WaitingForPaymentApproval"
        runtime, payment_controller, payment_bridge = _durable_runtime(
            workflow_fixture, paid, key
        )
        with TestClient(create_app(runtime)) as client:
            payment_view = client.get(
                "/v1/view", headers={"X-Verified-Identity": assertion}
            )
            assert payment_view.status_code == 200
            assert payment_view.json() == first_approval["view"]
            terminal_response = _turn(
                client,
                assertion,
                "restart-paid-0004",
                "承認",
                first_approval["version"],
            )
            assert terminal_response.status_code == 200, terminal_response.text
            terminal = terminal_response.json()
        assert payment_controller.executor.calls == 0
        assert payment_bridge.approve_calls == payment_bridge.execute_calls == 1
    else:
        terminal = first_approval

    assert terminal["state"] == "Completed"
    runtime, terminal_controller, terminal_bridge = _durable_runtime(
        workflow_fixture, paid, key
    )
    terminal_request_id = "restart-paid-0004" if paid else "restart-False-0003"
    terminal_expected_version = (
        first_approval["version"] if paid else planned["version"]
    )
    with TestClient(create_app(runtime)) as client:
        restored_terminal = client.get(
            "/v1/view", headers={"X-Verified-Identity": assertion}
        )
        assert restored_terminal.status_code == 200
        assert restored_terminal.json() == terminal["view"]
        terminal_replay = _turn(
            client,
            assertion,
            terminal_request_id,
            "承認",
            terminal_expected_version,
        )
        assert terminal_replay.status_code == 200
        assert terminal_replay.json() == terminal
    assert terminal_controller.executor.calls == 0
    assert terminal_bridge.approve_calls == terminal_bridge.execute_calls == 0


@pytest.mark.restart
def test_sqlite_completes_and_replays_a_blocked_terminal_turn(
    workflow_fixture,
) -> None:
    key = b"b" * 32
    assertion = workflow_fixture["assertion"]
    scope = SubjectScope(
        subject="test-user",
        tenantId="demo-tenant",
        adkSessionId=(
            "public-"
            + hashlib.sha256(b"demo-tenant\0test-user").hexdigest()[:32]
        ),
    )

    runtime, controller, _ = _durable_runtime(workflow_fixture, True, key)
    with TestClient(create_app(runtime)) as client:
        planned_response = _turn(
            client, assertion, "blocked-restart-0001", "有料予約", None
        )
        assert planned_response.status_code == 200, planned_response.text

    session = controller.store.active_for(scope)
    assert session is not None
    expected = session.version
    changed_step = session.active_step.model_copy(
        update={"goal": "承認表示後に改変された予約"}
    )
    session.plan = session.plan.model_copy(update={"steps": (changed_step,)})
    session.version += 1
    controller.store.compare_and_set(session, expected_version=expected)

    runtime, blocked_controller, blocked_bridge = _durable_runtime(
        workflow_fixture, True, key
    )
    with TestClient(create_app(runtime)) as client:
        blocked_response = _turn(
            client,
            assertion,
            "blocked-restart-0002",
            "承認",
            session.version,
        )
        assert blocked_response.status_code == 200, blocked_response.text
        blocked = blocked_response.json()
        assert blocked["state"] == "Blocked"
    assert blocked_controller.executor.calls == 0
    assert blocked_bridge.attach_calls == 0

    runtime, replay_controller, replay_bridge = _durable_runtime(
        workflow_fixture, True, key
    )
    with TestClient(create_app(runtime)) as client:
        replay = _turn(
            client,
            assertion,
            "blocked-restart-0002",
            "承認",
            session.version,
        )
        assert replay.status_code == 200
        assert replay.json() == blocked
    assert replay_controller.executor.calls == 0
    assert replay_bridge.attach_calls == 0


@pytest.mark.restart
def test_old_completed_replay_keeps_its_original_session_identity(
    workflow_fixture,
) -> None:
    key = b"c" * 32
    assertion = workflow_fixture["assertion"]
    runtime, _, _ = _durable_runtime(workflow_fixture, False, key)
    with TestClient(create_app(runtime)) as client:
        first_plan = _turn(
            client, assertion, "old-result-0001", "最初の予約", None
        )
        assert first_plan.status_code == 200, first_plan.text
        first_plan_wire = first_plan.json()
        first_terminal = _turn(
            client,
            assertion,
            "old-result-0002",
            "承認",
            first_plan_wire["version"],
        )
        assert first_terminal.status_code == 200, first_terminal.text
        first_terminal_wire = first_terminal.json()

        second_plan = _turn(
            client, assertion, "old-result-0003", "次の予約", None
        )
        assert second_plan.status_code == 200, second_plan.text
        second_plan_wire = second_plan.json()
        assert (
            second_plan_wire["mediationSessionId"]
            != first_terminal_wire["mediationSessionId"]
        )

        replay = _turn(
            client,
            assertion,
            "old-result-0002",
            "承認",
            first_plan_wire["version"],
        )
        assert replay.status_code == 200, replay.text
        assert replay.json() == first_terminal_wire


def test_readiness_fails_closed_when_sqlite_probe_fails(
    workflow_fixture, monkeypatch
) -> None:
    runtime, controller, _ = _durable_runtime(
        workflow_fixture, False, b"d" * 32
    )

    def failed_probe():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(controller.store, "readiness_probe", failed_probe)
    with TestClient(create_app(runtime)) as client:
        ready = client.get("/ready")

    assert ready.status_code == 503
    wire = ready.json()
    assert wire["checks"]["mediationStoreMode"] is True
    assert wire["checks"]["mediationStoreSchema"] is False
    assert wire["checks"]["mediationStoreProbe"] is False
    assert wire["mediationStore"] == {
        "mode": "sqlite",
        "durabilityProfile": "local-durable",
        "schemaVersion": None,
        "writable": False,
        "decryptable": False,
    }


def test_sqlite_reserves_request_before_matcher_is_called(
    workflow_fixture,
) -> None:
    store = SqliteMediationStore(workflow_fixture["repository"], b"e" * 32)
    controller, _ = _controller(False, store=store)
    scope = SubjectScope(
        subject="reservation-user",
        tenantId="demo-tenant",
        adkSessionId="reservation-session",
    )
    request_id = "reservation-order-0001"
    parts = (TextPart(text="予約"),)
    request_digest = controller.request_digest(
        scope=scope, parts=parts, expected_version=None
    )
    selected = _agent(False)

    class ReservationCheckingMatcher:
        async def match(self, goal: str):
            with pytest.raises(MediationError) as raised:
                store.load_request(scope, request_id, request_digest)
            assert raised.value.code == "MEDIATION_REQUEST_IN_PROGRESS"
            return (selected,)

    controller.matcher = ReservationCheckingMatcher()
    view = asyncio.run(
        controller.submit(
            scope=scope,
            parts=parts,
            request_id=request_id,
        )
    )
    assert view.state.value == "WaitingForPlanApproval"


def test_ephemeral_readiness_requires_exact_memory_profile(
    workflow_fixture,
) -> None:
    memory_controller, _ = _controller(False)
    memory_runtime = replace(
        workflow_fixture["runtime"],
        mediation_controller=memory_controller,
        public_route_probe=lambda: True,
        ephemeral_cloud_run_demo=True,
    )
    with TestClient(create_app(memory_runtime)) as client:
        memory_ready = client.get("/ready")
    assert memory_ready.status_code == 200, memory_ready.text
    memory_wire = memory_ready.json()
    assert memory_wire["mediationStore"] == {
        "mode": "memory",
        "durabilityProfile": "ephemeral-demo",
        "schemaVersion": None,
        "writable": True,
        "decryptable": True,
    }
    assert memory_wire["checks"]["mediationStoreMode"] is True
    assert memory_wire["checks"]["mediationStoreProfile"] is True
    assert memory_wire["checks"]["mediationStoreSchema"] is True
    assert memory_wire["checks"]["mediationStoreProbe"] is True

    sqlite_runtime, _, _ = _durable_runtime(
        workflow_fixture, False, b"f" * 32
    )
    sqlite_runtime = replace(sqlite_runtime, ephemeral_cloud_run_demo=True)
    with TestClient(create_app(sqlite_runtime)) as client:
        sqlite_ready = client.get("/ready")
    assert sqlite_ready.status_code == 503
    sqlite_checks = sqlite_ready.json()["checks"]
    assert sqlite_checks["mediationStoreMode"] is False
    assert sqlite_checks["mediationStoreProfile"] is False
    assert sqlite_checks["mediationStoreSchema"] is False
    assert sqlite_checks["mediationStoreProbe"] is True
