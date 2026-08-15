from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx


USER_AGENT_DIR = Path(__file__).resolve().parents[2] / "user-agent"
if str(USER_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(USER_AGENT_DIR))

from payment_client import APPROVAL_WORD, run_interactive  # noqa: E402
from secure_mediation_agent.workflow.client import WorkflowClient  # noqa: E402


def _view(state: str, version: int) -> dict[str, object]:
    pending = "plan" if state == "plan_approval_required" else "payment" if state == "payment_approval_required" else None
    return {
        "workflowId": "workflow-demo",
        "state": state,
        "version": version,
        "pendingApproval": pending,
        "planId": "plan-demo",
        "planDigest": "sha256:" + "a" * 64,
        "renderedText": f"画面: {state}",
        "profile": "x402-wire-simulation/1",
        "ap2Label": "AP2 v0.2 Human Present demo",
        "x402Label": "x402 v0.1 wire-shape test fixture (NOT CONFORMANT)",
        "railLabel": "simulated; no real asset or on-chain transaction",
    }


def test_exact_japanese_approvals_use_one_workflow_api() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["cookie"] == "__Host-payment-session=firebase-session-token"
        body = json.loads(request.content)
        if request.url.path == "/mediation-api/v1/workflows":
            calls.append((request.url.path, body["request"]["goal"]))
            return httpx.Response(200, json=_view("plan_approval_required", 1))
        text = body["parts"][0]["text"]
        calls.append((request.url.path, text))
        state = "payment_approval_required" if len(calls) == 2 else "completed"
        return httpx.Response(200, json=_view(state, 4 if len(calls) == 2 else 12))

    output: list[str] = []
    result = run_interactive(
        WorkflowClient(
            "https://demo.example/mediation-api",
            session_cookie="firebase-session-token",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
        prompt="予約して",
        plan_approval=APPROVAL_WORD,
        payment_approval=APPROVAL_WORD,
        output_fn=output.append,
    )
    assert calls == [
        ("/mediation-api/v1/workflows", "予約して"),
        ("/mediation-api/v1/workflows/workflow-demo/messages", "承認"),
        ("/mediation-api/v1/workflows/workflow-demo/messages", "承認"),
    ]
    assert result is not None and result["state"] == "completed"
    assert len(output) == 3


def test_whitespace_is_not_normalized_into_plan_approval() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json=_view("plan_approval_required", 1))

    output: list[str] = []
    result = run_interactive(
        WorkflowClient(
            "http://mediator.test",
            identity_assertion="test-assertion",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
        prompt="予約して",
        plan_approval="承認 ",
        payment_approval=APPROVAL_WORD,
        output_fn=output.append,
    )
    assert result is None
    assert calls == ["/v1/workflows"]
    assert "完全一致" in output[-1]
