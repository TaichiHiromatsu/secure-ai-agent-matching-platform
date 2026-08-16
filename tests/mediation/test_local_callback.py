from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from secure_mediation_agent.mediation.a2a_executor import A2AOperation
from secure_mediation_agent.mediation.adapters import (
    LegacyCallbackHook,
    LocalDeterministicCallbackHook,
)
from secure_mediation_agent.mediation.canonical import canonical_digest
from secure_mediation_agent.mediation.composition import _configured_callback_hook
from secure_mediation_agent.mediation.errors import SecurityBlocked
from secure_mediation_agent.mediation.models import (
    RemoteTaskSnapshot,
    SelectedAgentSnapshot,
)


ROOT = Path(__file__).resolve().parents[2]


def _payment_operation() -> A2AOperation:
    agent_wire = {
        "canonicalAgentId": "agent-005",
        "registryName": "paid_booking_agent",
        "a2aAgentName": "paid-booking-agent",
        "agentCardUrl": "http://127.0.0.1:8005/.well-known/agent-card.json",
        "rpcEndpoint": "http://127.0.0.1:8005/a2a",
        "a2aSkillId": "paid-booking",
        "trustScore": 100,
        "cardDigest": canonical_digest({"card": "paid"}),
        "paymentExtensionUris": (
            "urn:secure-a2a:extensions:x402-wire-simulation:v1",
        ),
    }
    agent = SelectedAgentSnapshot(
        **agent_wire, snapshotDigest=canonical_digest(agent_wire)
    )
    request = {
        "jsonrpc": "2.0",
        "id": "payment-operation-1",
        "method": "message/send",
        "params": {
            "action": "merchant:guaranteed-fulfillment-commit",
            "taskId": "task-1",
            "message": {"taskId": "task-1"},
        },
    }
    return A2AOperation(
        operationId="payment-operation-1",
        kind="payment-submit",
        agent=agent,
        request=request,
        requestDigest=canonical_digest(request),
        idempotencyKey="payment-operation-1",
        taskId="task-1",
        contextId="context-1",
        authorization="signed-local-capability",
    )


def test_local_callback_validates_the_actual_remote_task_binding() -> None:
    hook = LocalDeterministicCallbackHook()
    operation = _payment_operation()
    matching = RemoteTaskSnapshot(
        taskId="task-1",
        contextId="context-1",
        state="completed",
        taskDigest=canonical_digest({"task": "matching"}),
    )
    asyncio.run(hook.before(operation))
    asyncio.run(hook.after(operation, matching))

    mismatched = matching.model_copy(update={"context_id": "other-context"})
    with pytest.raises(SecurityBlocked, match="Task binding mismatch"):
        asyncio.run(hook.after(operation, mismatched))


def test_deterministic_callback_mode_is_explicit_and_local_only(monkeypatch) -> None:
    monkeypatch.delenv("MEDIATION_CALLBACK_MODE", raising=False)
    assert isinstance(_configured_callback_hook(), LegacyCallbackHook)

    monkeypatch.setenv("MEDIATION_CALLBACK_MODE", "deterministic-local")
    monkeypatch.setenv("DEV_MODE", "true")
    monkeypatch.setenv("APP_ENV", "ephemeral-demo")
    with pytest.raises(RuntimeError, match="restricted"):
        _configured_callback_hook()

    monkeypatch.setenv("APP_ENV", "local")
    assert isinstance(_configured_callback_hook(), LocalDeterministicCallbackHook)


def test_deterministic_hotel_agent_mode_is_explicit_and_local_only(monkeypatch) -> None:
    import importlib.util

    server_path = ROOT / "external-agents/trusted-agents/a2a_server.py"
    spec = importlib.util.spec_from_file_location("local_a2a_server_test", server_path)
    assert spec is not None and spec.loader is not None
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)

    monkeypatch.setenv("MEDIATION_LOCAL_AGENT_MODE", "deterministic")
    monkeypatch.setenv("DEV_MODE", "true")
    monkeypatch.setenv("APP_ENV", "ephemeral-demo")
    with pytest.raises(RuntimeError, match="restricted"):
        server._deterministic_local_mode()

    monkeypatch.setenv("APP_ENV", "local")
    assert server._deterministic_local_mode() is True
