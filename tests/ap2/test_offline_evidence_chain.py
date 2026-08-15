from __future__ import annotations

import os
import subprocess
import sys

import pytest

from secure_mediation_agent.ap2.evidence_verifier import verify_evidence_graph
from secure_mediation_agent.workflow.controller import Identity
from secure_mediation_agent.workflow.models import MessagePart, WorkflowRequest


pytestmark = pytest.mark.contract_ap2


def test_offline_verifier_imports_in_a_clean_process() -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [sys.executable, "-c", "from secure_mediation_agent.ap2.evidence_verifier import verify_evidence_graph"],
        check=True,
        env=environment,
    )


def test_completed_graph_verifies_offline_with_role_separated_keys(workflow_fixture) -> None:
    controller = workflow_fixture["runtime"].controller
    identity = Identity("demo-tenant", "demo-customer")
    view = controller.create(
        WorkflowRequest(goal="offline evidence"),
        identity=identity,
        session_id="offline-session",
        context_id="offline-context",
        idempotency_key="offline-create",
    )
    for name in ("plan", "payment"):
        view = controller.message(
            view.workflow_id,
            [MessagePart(kind="text", text="承認")],
            identity=identity,
            message_id=f"offline-{name}",
            idempotency_key=f"offline-{name}",
        )
    report = verify_evidence_graph(
        workflow_fixture["repository"], workflow_fixture["keys"], view.workflow_id
    )
    assert report["status"] == "PASS"
    assert "signature:checkout-mandate" in report["checked"]
    assert "signature:payment-mandate" in report["checked"]
    assert "trust-snapshots" in report["checked"]
    assert "NOT CONFORMANT" in report["x402"]
