from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx


USER_AGENT_DIR = Path(__file__).resolve().parents[2] / "user-agent"
if str(USER_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(USER_AGENT_DIR))

from payment_client import (  # noqa: E402
    APPROVAL_WORD,
    PaymentMediatorClient,
    run_interactive,
)


def _pricing() -> dict[str, object]:
    return {
        "policyVersion": "zero-fee-v1",
        "merchandiseAmount": 1250,
        "customerSurcharge": 0,
        "collectionRailCost": 0,
        "customerTotal": 1250,
        "providerCommission": 0,
        "merchantPayableAmount": 1250,
        "payoutRailCost": 0,
        "asset": "USD",
        "currency": "USD",
        "network": "demo:local",
        "decimals": 2,
        "roundingRule": "minor-unit-exact",
        "calculatedAt": "2026-08-15T00:00:00Z",
    }


def _start_result() -> dict[str, object]:
    return {
        "id": "task-demo",
        "contextId": "context-demo",
        "status": {"state": "input-required"},
        "metadata": {
            "x402.payment": {
                "status": "payment-required",
                "requirement": {
                    "x402Version": 2,
                    "resource": {"url": "a2a://demo/order", "description": "demo"},
                    "accepts": [
                        {
                            "scheme": "exact-simulated",
                            "network": "demo:local",
                            "amount": "1250",
                            "asset": "USD",
                            "decimals": 2,
                            "payTo": "mediation-platform",
                            "maxTimeoutSeconds": 300,
                            "extra": {"quoteDigest": "sha256:" + "a" * 64},
                        }
                    ],
                },
            },
            "marketplace.payment": {
                "orderId": "order-demo",
                "merchantId": "demo-merchant",
                "quoteId": "quote-demo",
                "correlationId": "correlation-demo",
                "pricing": _pricing(),
                "trustedSurfaceInput": {
                    "checkoutJwt": "fixed.demo.checkout",
                    "audience": "mediation-platform",
                    "nonce": "nonce-demo",
                    "orderId": "order-demo",
                    "taskId": "task-demo",
                    "quoteId": "quote-demo",
                    "challengeId": "challenge-demo",
                },
            },
        },
    }


def _completed_result() -> dict[str, object]:
    return {
        "id": "task-demo",
        "contextId": "context-demo",
        "status": {"state": "completed"},
        "metadata": {
            "x402.payment": {"status": "payment-completed", "receipts": [{}, {}, {}]},
            "marketplace.payment": {"orderId": "order-demo"},
        },
    }


def test_exact_japanese_approval_submits_on_the_original_a2a_task() -> None:
    actions: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        message = body["params"]["message"]
        data = message["parts"][0]["data"]
        actions.append(data["action"])
        assert request.headers["X-A2A-Extensions"].endswith("marketplace:v1")
        assert "requestAuth" in data["request"]
        if data["action"] == "start_order":
            result = _start_result()
        else:
            assert data["orderId"] == "order-demo"
            assert message["taskId"] == "task-demo"
            assert message["contextId"] == "context-demo"
            assert data["request"]["paymentPayload"]["x402Version"] == 2
            result = _completed_result()
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    output: list[str] = []
    result = run_interactive(
        PaymentMediatorClient("http://mediator.test", client=http),
        prompt="予約して",
        approval=APPROVAL_WORD,
        output_fn=output.append,
    )

    assert actions == ["start_order", "submit_payment"]
    assert result is not None
    assert result["status"]["state"] == "completed"
    assert "支払総額: 1250 USD minor units" in output[0]
    assert "「承認」" in output[0]


def test_yes_is_not_an_approval_word_and_never_submits_payment() -> None:
    actions: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        data = body["params"]["message"]["parts"][0]["data"]
        actions.append(data["action"])
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": body["id"], "result": _start_result()},
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    output: list[str] = []
    result = run_interactive(
        PaymentMediatorClient("http://mediator.test", client=http),
        prompt="予約して",
        approval="yes",
        output_fn=output.append,
    )

    assert result is None
    assert actions == ["start_order"]
    assert output[-1] == "承認語が一致しないため、決済は実行しませんでした。"
