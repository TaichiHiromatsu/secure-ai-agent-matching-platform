"""Vendor-neutral demo user agent client for the marketplace payment profile.

Natural-language intent is deliberately mapped to one fixed demo product.  All
pricing checks, Human Present approval construction, and signatures are handled
by deterministic code outside an LLM.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from secure_mediation_agent.payment_marketplace.auth import build_request_auth
from secure_mediation_agent.payment_marketplace.config import (
    CUSTOMER_KID,
    CUSTOMER_SUBJECT,
    PROFILE_URI,
)
from secure_mediation_agent.payment_marketplace.models import PricingBreakdown
from secure_mediation_agent.payment_marketplace.trusted_surface import TrustedSurface


APPROVAL_WORD = "承認"
DEFAULT_PROMPT = (
    "信頼済みの予約エージェントを使い、デモ予約を1件取得してください。"
    "支払総額が12.50 USDを超える場合は、承認前に止めてください。"
)


def _nonce(label: str) -> str:
    return f"{label}-{uuid.uuid4().hex}"


def _signed_body(path: str, body: dict[str, Any]) -> dict[str, Any]:
    return {
        **body,
        "requestAuth": build_request_auth(
            method="POST",
            path=path,
            body=body,
            subject=CUSTOMER_SUBJECT,
            role="customer",
            tenant_id=CUSTOMER_SUBJECT,
            kid=CUSTOMER_KID,
            nonce=_nonce("customer-agent"),
        ),
    }


class PaymentMediatorClient:
    """Small A2A client used by both the CLI and ADK Web demo agent."""

    def __init__(
        self,
        mediator_url: str,
        *,
        client: httpx.Client | None = None,
        max_customer_total: int = 1_250,
    ) -> None:
        self.mediator_url = mediator_url.rstrip("/")
        self.client = client or httpx.Client(timeout=15.0)
        self.max_customer_total = max_customer_total

    def _send(
        self,
        *,
        action: str,
        request: dict[str, Any],
        idempotency_key: str,
        task_id: str | None = None,
        context_id: str | None = None,
        order_id: str | None = None,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "messageId": f"message-{uuid.uuid4().hex}",
            "role": "user",
            "parts": [
                {
                    "kind": "data",
                    "data": {
                        "action": action,
                        "request": request,
                        **({"orderId": order_id} if order_id else {}),
                    },
                }
            ],
        }
        if task_id:
            message["taskId"] = task_id
        if context_id:
            message["contextId"] = context_id
        envelope = {
            "jsonrpc": "2.0",
            "id": f"rpc-{uuid.uuid4().hex}",
            "method": "message/send",
            "params": {"message": message},
        }
        response = self.client.post(
            f"{self.mediator_url}/a2a",
            json=envelope,
            headers={
                "Idempotency-Key": idempotency_key,
                "X-A2A-Extensions": PROFILE_URI,
            },
        )
        try:
            value = response.json()
        except Exception as exc:
            raise RuntimeError("仲介エージェントからJSON応答を受信できませんでした。") from exc
        if response.status_code >= 400:
            raise RuntimeError(f"仲介エージェントHTTPエラー: {response.status_code}")
        if not isinstance(value, dict) or "error" in value:
            detail = value.get("error", {}).get("data", {}) if isinstance(value, dict) else {}
            code = detail.get("code", "A2A_ERROR") if isinstance(detail, dict) else "A2A_ERROR"
            raise RuntimeError(f"仲介エージェントが決済要求を拒否しました: {code}")
        result = value.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("仲介エージェントのA2A応答が不正です。")
        return result

    def request_payment(self, prompt: str) -> dict[str, Any]:
        """Map the demo intent to a fixed product and request a payment task."""

        correlation_id = f"user-agent-{uuid.uuid4().hex}"
        body = {
            "productId": "demo-paid-booking",
            "quantity": 1,
            "correlationId": correlation_id,
        }
        task = self._send(
            action="start_order",
            request=_signed_body("/v1/orders", body),
            idempotency_key=f"user-agent-start-{uuid.uuid4().hex}",
        )
        metadata = task.get("metadata", {})
        x402 = metadata.get("x402.payment", {})
        marketplace = metadata.get("marketplace.payment", {})
        pricing = PricingBreakdown.model_validate(marketplace.get("pricing"))
        if pricing.customer_total > self.max_customer_total:
            raise RuntimeError(
                f"支払総額 {pricing.customer_total} minor units が上限 "
                f"{self.max_customer_total} minor units を超えています。"
            )
        trusted = marketplace.get("trustedSurfaceInput")
        requirement = x402.get("requirement")
        if not isinstance(trusted, dict) or not isinstance(requirement, dict):
            raise RuntimeError("仲介エージェントの承認材料が不足しています。")
        return {
            "prompt": prompt,
            "taskId": task["id"],
            "contextId": task["contextId"],
            "orderId": marketplace["orderId"],
            "merchantId": marketplace["merchantId"],
            "quoteId": marketplace["quoteId"],
            "pricing": pricing.model_dump(mode="json", by_alias=True),
            "requirement": requirement,
            "trustedSurfaceInput": trusted,
            "simulated": True,
        }

    def submit_approval(self, pending: dict[str, Any]) -> dict[str, Any]:
        """Build a closed mandate and submit it on the original A2A task."""

        trusted = pending["trustedSurfaceInput"]
        pricing = PricingBreakdown.model_validate(pending["pricing"])
        approval = TrustedSurface(clock=lambda: datetime.now(timezone.utc)).build_approval(
            checkout_jwt=trusted["checkoutJwt"],
            pricing=pricing,
            audience=trusted["audience"],
            nonce=trusted["nonce"],
            order_id=trusted["orderId"],
            task_id=trusted["taskId"],
            quote_id=trusted["quoteId"],
            challenge_id=trusted["challengeId"],
        ).model_dump(mode="json", by_alias=True)
        body = {
            "approval": approval,
            "paymentPayload": {
                "x402Version": 2,
                "accepted": pending["requirement"]["accepts"][0],
                "payload": {"authorization": approval["authorization"]},
            },
        }
        path = f"/v1/orders/{pending['orderId']}/payment"
        return self._send(
            action="submit_payment",
            request=_signed_body(path, body),
            idempotency_key=f"user-agent-payment-{uuid.uuid4().hex}",
            task_id=pending["taskId"],
            context_id=pending["contextId"],
            order_id=pending["orderId"],
        )


def format_payment_request(pending: dict[str, Any]) -> str:
    pricing = pending["pricing"]
    fields = (
        ("商品代金", "merchandiseAmount"),
        ("利用者手数料", "customerSurcharge"),
        ("回収レール費用", "collectionRailCost"),
        ("支払総額", "customerTotal"),
        ("事業者手数料", "providerCommission"),
        ("事業者未払金", "merchantPayableAmount"),
        ("送金レール費用", "payoutRailCost"),
    )
    lines = [
        "仲介エージェントから支払依頼を受信しました。",
        f"受取人: {pending['requirement']['accepts'][0]['payTo']}",
        f"事業者: {pending['merchantId']}",
    ]
    lines.extend(f"{label}: {pricing[key]} USD minor units" for label, key in fields)
    lines.extend(
        [
            "決済方式: exact-simulated / demo:local（実資産は移動しません）",
            f"支払う場合は「{APPROVAL_WORD}」と入力してください。",
        ]
    )
    return "\n".join(lines)


def format_completion(task: dict[str, Any]) -> str:
    metadata = task.get("metadata", {})
    market = metadata.get("marketplace.payment", {})
    receipts = metadata.get("x402.payment", {}).get("receipts", [])
    return "\n".join(
        [
            "承認を確認し、仲介エージェント経由の決済を完了しました。",
            f"注文ID: {market.get('orderId')}",
            f"状態: {task.get('status', {}).get('state')}",
            f"receipt数: {len(receipts)}",
            "事業者への支払いは未払金として計上され、後日精算されます。",
        ]
    )


def run_interactive(
    client: PaymentMediatorClient,
    *,
    prompt: str,
    approval: str | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> dict[str, Any] | None:
    pending = client.request_payment(prompt)
    output_fn(format_payment_request(pending))
    entered = approval if approval is not None else input_fn("入力: ")
    if entered.strip() != APPROVAL_WORD:
        output_fn("承認語が一致しないため、決済は実行しませんでした。")
        return None
    result = client.submit_approval(pending)
    output_fn(format_completion(result))
    return result
