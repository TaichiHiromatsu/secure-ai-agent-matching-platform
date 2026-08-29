"""A2A wire 0.3 adapter for the project-local marketplace profile.

The payment domain intentionally does not depend on A2A SDK model classes.  This
module is the only place where JSON-RPC/A2A task shapes are assembled.
"""

from __future__ import annotations

from typing import Any


PROFILE = "urn:secure-a2a:extensions:ap2-x402-marketplace:v1"
WIRE_VERSION = "0.3.0"
SDK_VERSION = "0.3.19"


def mediation_agent_card(public_url: str) -> dict[str, Any]:
    return {
        "name": "Secure Mediation Marketplace Payment Agent",
        "description": (
            "Project-local AP2 v0.2 / x402 v2-shaped payment mediation demo. "
            "Simulation only; no real settlement or legal payment guarantee."
        ),
        "url": public_url.rstrip("/") + "/a2a",
        "version": "1.0.0",
        "protocolVersion": WIRE_VERSION,
        "preferredTransport": "JSONRPC",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
            "extensions": [
                {
                    "uri": PROFILE,
                    "required": True,
                    "params": {
                        "profile": PROFILE,
                        "simulated": True,
                        "sdkPackage": "a2a-sdk",
                        "sdkVersion": SDK_VERSION,
                        "wireProtocolVersion": WIRE_VERSION,
                        "roles": ["customer", "merchant", "operator"],
                        "upstream": {
                            "schemes": ["exact-simulated"],
                            "networks": ["demo:local"],
                            "assets": [{"asset": "USD", "decimals": 2}],
                            "payTo": ["mediation-platform"],
                        },
                        "merchantCredit": {
                            "schemes": ["platform-credit"],
                            "networks": ["demo:mediation-ledger"],
                        },
                    },
                }
            ],
        },
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "skills": [
            {
                "id": "marketplace_order",
                "name": "Marketplace order",
                "description": "Create/resume a paid order through the mediator",
                "tags": ["ap2", "x402", "marketplace", "simulation"],
                "examples": ["Book the demo service through the marketplace"],
            },
            {
                "id": "payout_status",
                "name": "Payout status",
                "description": "Merchant-scoped authoritative payout status query",
                "tags": ["marketplace", "payout"],
                "examples": ["Get payout status for this merchant"],
            },
        ],
    }


def payment_metadata(
    *,
    status: str,
    leg: str,
    order_id: str,
    merchant_id: str,
    quote_id: str,
    correlation_id: str,
    requirement: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    settlement: dict[str, Any] | None = None,
    ap2: dict[str, Any] | None = None,
    receipts: list[dict[str, Any]] | None = None,
    error: dict[str, Any] | None = None,
    **marketplace_ids: Any,
) -> dict[str, Any]:
    x402: dict[str, Any] = {
        "extension": PROFILE,
        "profile": PROFILE,
        "simulated": True,
        "status": status,
        "leg": leg,
        "receipts": receipts or [],
    }
    for key, value in (
        ("requirement", requirement),
        ("payload", payload),
        ("settlement", settlement),
        ("error", error),
    ):
        if value is not None:
            x402[key] = value
    market = {
        "orderId": order_id,
        "merchantId": merchant_id,
        "quoteId": quote_id,
        "correlationId": correlation_id,
    }
    market.update({key: value for key, value in marketplace_ids.items() if value is not None})
    metadata: dict[str, Any] = {"x402.payment": x402, "marketplace.payment": market}
    if ap2:
        metadata["ap2.payment"] = ap2
    return metadata


def task_result(
    *,
    task_id: str,
    context_id: str,
    state: str,
    metadata: dict[str, Any],
    message: str,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "contextId": context_id,
        "status": {
            "state": state,
            "message": {
                "messageId": f"message-{task_id}-{state}",
                "role": "agent",
                "parts": [{"kind": "text", "text": message}],
                "metadata": metadata,
            },
        },
        "metadata": metadata,
    }


def jsonrpc_result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def jsonrpc_error(request_id: Any, code: int, message: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message, "data": data},
    }
