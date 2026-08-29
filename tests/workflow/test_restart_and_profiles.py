from __future__ import annotations

import pytest

from secure_mediation_agent.payment_profiles.registry import ProfileRegistry
from secure_mediation_agent.payment_profiles.x402_v01 import CANONICAL_X402_V01_URI
from secure_mediation_agent.workflow.controller import Identity, WorkflowController
from secure_mediation_agent.workflow.models import MessagePart, WorkflowRequest
from secure_mediation_agent.workflow.repository import WorkflowRepository


pytestmark = pytest.mark.integration


def test_restart_between_two_approvals_uses_authoritative_sqlite(workflow_fixture) -> None:
    identity = Identity("demo-tenant", "demo-customer")
    first_controller = workflow_fixture["runtime"].controller
    view = first_controller.create(
        WorkflowRequest(goal="restart demo"),
        identity=identity,
        session_id="restart-session",
        context_id="restart-context",
        idempotency_key="restart-create",
    )
    payment = first_controller.message(
        view.workflow_id,
        [MessagePart(kind="text", text="承認")],
        identity=identity,
        message_id="restart-plan",
        idempotency_key="restart-message-plan",
        expected_version=view.version,
    )
    restarted = WorkflowController(
        WorkflowRepository(workflow_fixture["paths"]), workflow_fixture["keys"]
    )
    completed = restarted.message(
        view.workflow_id,
        [MessagePart(kind="text", text="承認")],
        identity=identity,
        message_id="restart-payment",
        idempotency_key="restart-message-payment",
        expected_version=payment.version,
    )
    assert completed.state == "completed"
    assert restarted.repository.rail_balance("demo-customer") == 98_750
    assert restarted.repository.rail_balance("demo-merchant") == 1_250


def test_registry_has_no_canonical_x402_fallback(workflow_fixture) -> None:
    registry = ProfileRegistry.load(
        "x402-wire-simulation/1",
        simulation_key=workflow_fixture["keys"].simulation_signer,
    )
    assert registry.extension_uri != CANONICAL_X402_V01_URI
    with pytest.raises(RuntimeError, match="disabled"):
        ProfileRegistry.load(
            "x402-v0.1",
            simulation_key=workflow_fixture["keys"].simulation_signer,
        )
