#!/usr/bin/env python3
"""Run the deterministic marketplace payment demo against live services."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from secure_mediation_agent.payment_marketplace.auth import build_request_auth
from secure_mediation_agent.payment_marketplace.config import (
    CUSTOMER_KID,
    CUSTOMER_SUBJECT,
    MERCHANT_SUBJECT,
    OPERATOR_KID,
    OPERATOR_SUBJECT,
    PROFILE_URI,
)
from secure_mediation_agent.payment_marketplace.models import PricingBreakdown
from secure_mediation_agent.payment_marketplace.trusted_surface import TrustedSurface


def nonce(label: str) -> str:
    return f"{label}-{uuid.uuid4().hex}"


def signed_body(
    path: str,
    body: dict[str, Any],
    *,
    subject: str,
    role: str,
    tenant: str,
    kid: str,
) -> dict[str, Any]:
    return {
        **body,
        "requestAuth": build_request_auth(
            method="POST",
            path=path,
            body=body,
            subject=subject,
            role=role,
            tenant_id=tenant,
            kid=kid,
            nonce=nonce(role),
        ),
    }


class DemoRunner:
    def __init__(self, payment_url: str, merchant_url: str) -> None:
        self.payment_url = payment_url.rstrip("/")
        self.merchant_url = merchant_url.rstrip("/")
        self.client = httpx.Client(timeout=15.0)
        self.extension_headers = {"X-A2A-Extensions": PROFILE_URI}

    def post(
        self,
        url: str,
        body: dict[str, Any],
        *,
        key: str | None = None,
        extension: bool = False,
        expected_error_code: str | None = None,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if key:
            headers["Idempotency-Key"] = key
        if extension:
            headers.update(self.extension_headers)
        response = self.client.post(url, json=body, headers=headers)
        value = response.json()
        if not isinstance(value, dict):
            raise RuntimeError(f"invalid response from {url}")
        if response.status_code >= 400:
            if expected_error_code and value.get("code") == expected_error_code:
                return value
            raise RuntimeError(f"{response.status_code} {url}: {response.text}")
        return value

    def create_order(self, suffix: str) -> dict[str, Any]:
        body = {"productId": "demo-paid-booking", "quantity": 1, "correlationId": f"demo-{suffix}-{uuid.uuid4().hex}"}
        request = signed_body(
            "/v1/orders",
            body,
            subject=CUSTOMER_SUBJECT,
            role="customer",
            tenant=CUSTOMER_SUBJECT,
            kid=CUSTOMER_KID,
        )
        return self.post(
            f"{self.payment_url}/v1/orders",
            request,
            key=f"start-{suffix}-{uuid.uuid4().hex}",
            extension=True,
        )

    def pay(
        self,
        start: dict[str, Any],
        suffix: str,
        fault: str | None = None,
        expected_error_code: str | None = None,
    ) -> dict[str, Any]:
        trusted = start["trustedSurfaceInput"]
        surface = TrustedSurface(clock=lambda: datetime.now(timezone.utc))
        approval = surface.build_approval(
            checkout_jwt=trusted["checkoutJwt"],
            pricing=PricingBreakdown.model_validate(start["pricing"]),
            audience=trusted["audience"],
            nonce=trusted["nonce"],
            order_id=trusted["orderId"],
            task_id=trusted["taskId"],
            quote_id=trusted["quoteId"],
            challenge_id=trusted["challengeId"],
        ).model_dump(mode="json", by_alias=True)
        body: dict[str, Any] = {
            "approval": approval,
            "paymentPayload": {
                "x402Version": 2,
                "accepted": start["requirement"]["accepts"][0],
                "payload": {"authorization": approval["authorization"]},
            },
        }
        if fault:
            body["merchantFault"] = fault
        path = f"/v1/orders/{start['orderId']}/payment"
        request = signed_body(
            path,
            body,
            subject=CUSTOMER_SUBJECT,
            role="customer",
            tenant=CUSTOMER_SUBJECT,
            kid=CUSTOMER_KID,
        )
        return self.post(
            f"{self.payment_url}{path}",
            request,
            key=f"payment-{suffix}-{uuid.uuid4().hex}",
            extension=True,
            expected_error_code=expected_error_code,
        )

    def payout(self, suffix: str) -> dict[str, Any]:
        body = {"merchantId": MERCHANT_SUBJECT, "reason": "explicit demo payout"}
        request = signed_body(
            "/internal/v1/payouts",
            body,
            subject=OPERATOR_SUBJECT,
            role="operator",
            tenant=OPERATOR_SUBJECT,
            kid=OPERATOR_KID,
        )
        return self.post(
            f"{self.payment_url}/internal/v1/payouts",
            request,
            key=f"payout-{suffix}-{uuid.uuid4().hex}",
        )

    def refund(self, order_id: str, suffix: str) -> dict[str, Any]:
        path = f"/internal/v1/orders/{order_id}/refunds"
        body = {"reason": "merchant fulfillment failed in demo"}
        request = signed_body(
            path,
            body,
            subject=OPERATOR_SUBJECT,
            role="operator",
            tenant=OPERATOR_SUBJECT,
            kid=OPERATOR_KID,
        )
        return self.post(
            f"{self.payment_url}{path}",
            request,
            key=f"refund-{suffix}-{uuid.uuid4().hex}",
        )

    def reconcile(self, order_id: str) -> dict[str, Any]:
        path = f"/internal/v1/orders/{order_id}/reconcile"
        body = {"reason": "merchant timeout status reconciliation"}
        request = signed_body(
            path,
            body,
            subject=OPERATOR_SUBJECT,
            role="operator",
            tenant=OPERATOR_SUBJECT,
            kid=OPERATOR_KID,
        )
        return self.post(f"{self.payment_url}{path}", request)

    def merchant_payout_status(self, payout_id: str) -> dict[str, Any]:
        signed = self.post(
            f"{self.merchant_url}/v1/payout-status-requests",
            {"payoutId": payout_id, "correlationId": nonce("payout-status")},
        )
        return self.post(f"{self.payment_url}/v1/payouts/{payout_id}/status", signed)

    def run(self) -> dict[str, Any]:
        happy_start = self.create_order("happy")
        happy = self.pay(happy_start, "happy")
        if happy["state"] != "completed":
            raise RuntimeError("happy order did not complete")
        payout = self.payout("happy")
        payout_status = self.merchant_payout_status(payout["payoutId"])
        if payout_status["state"] != "paid":
            raise RuntimeError("merchant payout status is not paid")

        failed_start = self.create_order("failure")
        failed = self.pay(failed_start, "failure", fault="failure")
        if failed["state"] != "refund_required":
            raise RuntimeError("failed fulfillment did not require a refund")
        refund = self.refund(failed_start["orderId"], "failure")
        if refund["state"] != "settled":
            raise RuntimeError("refund did not settle")

        unknown_start = self.create_order("unknown")
        unknown = self.pay(
            unknown_start,
            "unknown",
            fault="timeout",
            expected_error_code="SETTLEMENT_UNKNOWN",
        )
        if unknown.get("retryable") is not True:
            raise RuntimeError("unknown fulfillment was not reported as retryable")
        reconciled = self.reconcile(unknown_start["orderId"])
        if reconciled["state"] != "completed":
            raise RuntimeError("unknown fulfillment did not reconcile to completed")
        return {
            "profile": PROFILE_URI,
            "simulated": True,
            "happyOrder": happy["orderId"],
            "happyState": happy["state"],
            "payoutId": payout["payoutId"],
            "payoutState": payout_status["state"],
            "failedOrder": failed["orderId"],
            "failedState": failed["state"],
            "refundId": refund["refundId"],
            "refundState": refund["state"],
            "reconciledOrder": reconciled["orderId"],
            "reconciledState": reconciled["state"],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payment-url", default="http://127.0.0.1:8004")
    parser.add_argument("--merchant-url", default="http://127.0.0.1:8005")
    args = parser.parse_args()
    result = DemoRunner(args.payment_url, args.merchant_url).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
