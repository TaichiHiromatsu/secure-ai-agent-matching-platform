from __future__ import annotations

import asyncio

import pytest

from secure_mediation_agent.mediation.a2a_executor import (
    A2AOperation,
    SharedA2AOperationExecutor,
)
from secure_mediation_agent.mediation.adapters import _validate_exact_destination
from secure_mediation_agent.mediation.canonical import canonical_digest
from secure_mediation_agent.mediation.errors import SecurityBlocked
from secure_mediation_agent.mediation.models import (
    A2AResponseEnvelope,
    GateDecision,
    RemoteTaskSnapshot,
    SelectedAgentSnapshot,
)


def _agent():
    wire = {
        "canonicalAgentId": "agent-002",
        "registryName": "hotel_agent",
        "a2aAgentName": "hotel_agent",
        "agentCardUrl": "http://127.0.0.1:8002/card",
        "rpcEndpoint": "http://127.0.0.1:8002/a2a/hotel_agent",
        "a2aSkillId": "hotel_search",
        "trustScore": 90,
        "cardDigest": canonical_digest({"card": 1}),
        "paymentExtensionUris": (),
    }
    return SelectedAgentSnapshot(**wire, snapshotDigest=canonical_digest(wire))


def _operation():
    request = {"jsonrpc": "2.0", "id": "op-1", "method": "message/send"}
    return A2AOperation(
        operationId="op-1",
        kind="task-start",
        agent=_agent(),
        request=request,
        requestDigest=canonical_digest(request),
        idempotencyKey="idem-1",
    )


class Harness:
    def __init__(self, fail=None):
        self.events = []
        self.fail = fail

    async def before(self, operation):
        self.events.append("before")
        if self.fail == "before":
            raise RuntimeError("no")

    async def after(self, operation, response):
        self.events.append("after")
        if self.fail == "after":
            raise RuntimeError("no")

    async def decide(self, gate_id, operation, response):
        self.events.append(gate_id)
        return GateDecision(
            gateId=gate_id,
            decision="PASS",
            decisionDigest=canonical_digest({"gate": gate_id}),
        )

    async def send(self, operation):
        self.events.append("transport")
        task = RemoteTaskSnapshot(
            taskId="task-1",
            contextId="context-1",
            state="completed",
            taskDigest=canonical_digest({"task": 1}),
        )
        return A2AResponseEnvelope(
            task=task, envelopeDigest=canonical_digest({"envelope": 1})
        )

    async def persist_response(self, operation, response):
        self.events.append("persist")


def _execute(harness):
    return asyncio.run(
        SharedA2AOperationExecutor(
            callback=harness,
            gates=harness,
            transport=harness,
            observer=harness,
        ).execute(_operation())
    )


def test_executor_has_one_enforced_callback_gate_transport_order():
    harness = Harness()
    result = _execute(harness)
    assert harness.events == [
        "before",
        "PRE_A2A_START",
        "transport",
        "persist",
        "after",
        "POST_A2A_RESPONSE",
    ]
    assert result.event_order == (
        "legacy-callback-before",
        "PRE_A2A_START",
        "transport",
        "response-persisted",
        "legacy-callback-after",
        "POST_A2A_RESPONSE",
    )


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        ("before", ["before"]),
        (
            "after",
            ["before", "PRE_A2A_START", "transport", "persist", "after"],
        ),
    ],
)
def test_callback_failure_is_fail_closed(phase, expected):
    harness = Harness(fail=phase)
    with pytest.raises(SecurityBlocked):
        _execute(harness)
    assert harness.events == expected


def test_transport_destination_is_exactly_allowlisted():
    _validate_exact_destination(
        "http://127.0.0.1:8005/.well-known/agent-card.json"
    )
    with pytest.raises(SecurityBlocked):
        _validate_exact_destination("https://attacker.example/a2a")
    with pytest.raises(SecurityBlocked):
        _validate_exact_destination("http://127.0.0.1:8005/a2a?next=internal")
