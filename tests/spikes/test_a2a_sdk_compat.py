"""Executable A2A SDK 0.3.19 wire and extension compatibility checks."""

from __future__ import annotations

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentSkill,
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)


pytestmark = [pytest.mark.spike, pytest.mark.contract_x402_simulation]

SIMULATION_URI = "urn:secure-a2a:extensions:x402-wire-simulation:v1"


def test_a2a_camel_case_models_and_dotted_metadata_round_trip() -> None:
    card = AgentCard(
        name="paid_booking_agent",
        description="AP2 demo Merchant; local simulation only.",
        url="http://127.0.0.1:8005/a2a",
        version="2.0.0-simulation",
        protocolVersion="0.3.0",
        capabilities=AgentCapabilities(
            extensions=[
                AgentExtension(
                    uri=SIMULATION_URI,
                    required=True,
                    params={"simulated": True, "conformance": "NOT_CONFORMANT"},
                )
            ]
        ),
        skills=[
            AgentSkill(
                id="paid-booking",
                name="Demo paid booking",
                description="One fixed local simulation product.",
                tags=["simulation"],
            )
        ],
        defaultInputModes=["text/plain", "application/json"],
        defaultOutputModes=["text/plain", "application/json"],
    )
    wire = card.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert wire["protocolVersion"] == "0.3.0"
    assert wire["capabilities"]["extensions"][0]["uri"] == SIMULATION_URI

    message = Message(
        messageId="message-required",
        contextId="context-spike",
        role=Role.agent,
        parts=[Part(root=TextPart(text="Payment approval is required."))],
        metadata={
            "x402.payment.status": "payment-required",
            "x402.payment.required": {
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
            },
        },
    )
    task = Task(
        id="task-spike",
        contextId="context-spike",
        status=TaskStatus(state=TaskState.input_required, message=message),
        artifacts=[
            Artifact(
                artifactId="artifact-spike",
                parts=[Part(root=TextPart(text="draft"))],
            )
        ],
    )
    task_wire = task.model_dump(mode="json", by_alias=True, exclude_none=True)
    restored = Task.model_validate(task_wire)
    assert restored.status.message is not None
    assert restored.status.message.metadata is not None
    assert restored.status.message.metadata["x402.payment.required"]["x402Version"] == 1
    assert "x402.payment" not in restored.status.message.metadata


def test_request_context_extension_activation_echo_and_reserved_task_id() -> None:
    call = ServerCallContext(requested_extensions={SIMULATION_URI})
    message = Message(
        messageId="message-start",
        contextId="context-spike",
        role=Role.user,
        parts=[Part(root=TextPart(text="start"))],
    )
    context = RequestContext(
        task_id="task-reserved",
        context_id="context-spike",
        call_context=call,
    )
    assert context.task_id == "task-reserved"
    assert context.context_id == "context-spike"
    assert context.requested_extensions == {SIMULATION_URI}
    context.add_activated_extension(SIMULATION_URI)
    assert call.activated_extensions == {SIMULATION_URI}
    assert message.task_id is None
