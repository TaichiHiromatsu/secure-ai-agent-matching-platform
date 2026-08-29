"""Contract tests for the independently path-loaded paid booking agent.

The HMAC strings below are the documented Appendix A test-only fixtures.  Product
code receives them through dependency injection and never embeds or returns them.
"""

from __future__ import annotations

import base64
import copy
import importlib.util
import itertools
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "external-agents" / "paid-booking-agent"
PACKAGE = "paid_booking_agent_path_test"
MERCHANT_TEST_KEY = b"test-only-demo-merchant-key-v1"
MEDIATOR_TEST_KEY = b"test-only-demo-mediator-key-v1"
FIXED_NOW = 1_800_000_000


def _load_module(name: str):
    qualified_name = f"{PACKAGE}.{name}"
    spec = importlib.util.spec_from_file_location(qualified_name, AGENT_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


namespace = types.ModuleType(PACKAGE)
namespace.__path__ = [str(AGENT_DIR)]
sys.modules[PACKAGE] = namespace
models = _load_module("models")
service_module = _load_module("service")
app_module = _load_module("app")


@pytest.fixture
def merchant_service(tmp_path):
    counter = itertools.count(1)
    service = service_module.MerchantService(
        merchant_key=MERCHANT_TEST_KEY,
        mediator_keys={models.MEDIATOR_KID: MEDIATOR_TEST_KEY},
        database_path=str(tmp_path / "merchant.sqlite3"),
        clock=lambda: FIXED_NOW,
        id_factory=lambda: f"{next(counter):04d}",
        nonce_factory=lambda: f"nonce-{next(counter):04d}",
        allow_test_faults=True,
    )
    yield service
    service.close()


@pytest.fixture
def client(merchant_service):
    with TestClient(app_module.create_app(merchant_service)) as test_client:
        yield test_client


def _quote(client: TestClient, suffix: str = "1") -> dict:
    response = client.post(
        "/v1/quotes",
        json={
            "orderId": f"order-{suffix}",
            "taskId": f"task-{suffix}",
            "correlationId": f"corr-{suffix}",
            "productId": "demo-paid-booking",
            "quantity": 1,
            "audience": "mediation-platform",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _guarantee(quote: dict, suffix: str = "1") -> dict:
    requirement = quote["requirement"]
    quote_claims = requirement["quote"]
    unsigned_claims = {
        "kind": "payment-guarantee",
        "profile": models.PROFILE,
        "simulated": True,
        "guaranteeId": f"guarantee-{suffix}",
        "merchantQuoteRequirementDigest": quote["quoteDigest"],
        "orderId": quote_claims["orderId"],
        "taskId": quote_claims["taskId"],
        "quoteId": quote_claims["quoteId"],
        "merchantId": models.MERCHANT_ID,
        "upstreamX402ReceiptDigest": "sha256:" + "1" * 64,
        "upstreamAp2ReceiptDigest": "sha256:" + "2" * 64,
        "payableJournalTransactionId": f"journal-{suffix}",
        "payableEntryId": f"payable-{suffix}",
        "payableAmount": requirement["accepts"][0]["amount"],
        "commission": "0",
        "currency": "USD",
        "payoutTermsVersion": "manual-payout-v1",
        "iat": FIXED_NOW,
        "exp": FIXED_NOW + 300,
    }
    signed_claims = service_module.sign_document(
        unsigned_claims,
        kid=models.MEDIATOR_KID,
        key=MEDIATOR_TEST_KEY,
    )
    return {
        "paymentPayload": {
            "x402Version": 2,
            "accepted": copy.deepcopy(requirement["accepts"][0]),
            "payload": signed_claims,
        },
        "correlationId": f"corr-{suffix}",
    }


def _decode_segment(value: str) -> dict:
    padded = value + "=" * (-len(value) % 4)
    return json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))


def test_health_ready_and_agent_card_are_payment_aware_without_keys(client):
    assert client.get("/health").json() == {
        "status": "ok",
        "service": "paid-booking-agent",
        "simulated": True,
    }
    assert client.get("/ready").status_code == 200

    card_response = client.get("/.well-known/agent-card.json")
    assert card_response.status_code == 200
    card = card_response.json()
    assert card["protocolVersion"] == "0.3.0"
    assert card["url"].endswith("/a2a")
    extension = card["capabilities"]["extensions"][0]
    assert extension["uri"] == models.PROFILE
    assert extension["required"] is True
    assert extension["params"]["sdkVersion"] == "0.3.19"
    assert extension["params"]["merchantCredit"]["schemes"] == ["platform-credit"]
    assert {skill["id"] for skill in card["skills"]} == {
        "paid_booking",
        "fulfillment_status",
        "payout_status",
    }
    serialized = card_response.text
    assert MERCHANT_TEST_KEY.decode() not in serialized
    assert MEDIATOR_TEST_KEY.decode() not in serialized


def test_a2a_quote_adapter_returns_signed_merchant_requirement(client):
    request = {
        "orderId": "order-a2a",
        "taskId": "task-a2a",
        "correlationId": "corr-a2a",
        "productId": "demo-paid-booking",
        "quantity": 1,
        "audience": "mediation-platform",
    }
    response = client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "id": "rpc-a2a-quote",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "message-a2a-quote",
                    "role": "user",
                    "parts": [
                        {"kind": "data", "data": {"action": "quote", "request": request}}
                    ],
                }
            },
        },
    )
    assert response.status_code == 200, response.text
    envelope = response.json()
    assert envelope["jsonrpc"] == "2.0"
    assert envelope["id"] == "rpc-a2a-quote"
    result = envelope["result"]
    assert result["requirement"]["quote"]["audience"] == "mediation-platform"
    service_module.verify_document(
        result["requirement"],
        keys={models.MERCHANT_KID: MERCHANT_TEST_KEY},
        expected_kid=models.MERCHANT_KID,
    )


def test_quote_is_signed_fixed_v2_requirement_with_exact_checkout_jwt(client):
    quote = _quote(client)
    requirement = quote["requirement"]
    service_module.verify_document(
        requirement,
        keys={models.MERCHANT_KID: MERCHANT_TEST_KEY},
        expected_kid=models.MERCHANT_KID,
    )

    assert requirement["x402Version"] == 2
    assert requirement["profile"] == models.PROFILE
    assert requirement["simulated"] is True
    accepted = requirement["accepts"]
    assert len(accepted) == 1
    assert accepted[0] == {
        "scheme": "platform-credit",
        "network": "demo:mediation-ledger",
        "amount": "1250",
        "asset": "USD",
        "decimals": 2,
        "payTo": "demo-merchant",
        "maxTimeoutSeconds": 300,
        "extra": {
            "profile": models.PROFILE,
            "simulated": True,
            "quoteId": requirement["quote"]["quoteId"],
            "orderId": "order-1",
            "merchantId": "demo-merchant",
            "pricingPolicyVersion": "zero-fee-v1",
            "fulfillmentTermsDigest": models.FULFILLMENT_TERMS_DIGEST,
        },
    }

    checkout_jwt = quote["checkoutJwt"]
    assert checkout_jwt == requirement["quote"]["checkoutJwt"]
    header, claims, signature = checkout_jwt.split(".")
    assert _decode_segment(header) == {
        "alg": "HS256",
        "kid": "demo-merchant-hmac-v1",
        "typ": "JWT",
    }
    checkout_claims = _decode_segment(claims)
    assert checkout_claims["orderId"] == "order-1"
    assert checkout_claims["product"]["merchandiseAmount"] == "1250"
    assert signature

    # Quote creation is deterministic/idempotent for a given order and input.
    assert _quote(client) == quote
    conflict = client.post(
        "/v1/quotes",
        json={
            "orderId": "order-1",
            "taskId": "different-task",
            "correlationId": "corr-1",
        },
    )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "IDEMPOTENCY_CONFLICT"


def test_valid_guarantee_fulfils_once_and_returns_signed_receipt(client, merchant_service):
    quote = _quote(client)
    guarantee = _guarantee(quote)

    first = client.post("/v1/fulfillments", json=guarantee)
    assert first.status_code == 200, first.text
    first_body = first.json()
    assert first_body["state"] == "fulfilled"
    assert first_body["idempotent"] is False
    assert first_body["receipt"]["receiptType"] == "merchant-order"
    service_module.verify_document(
        first_body["receipt"],
        keys={models.MERCHANT_KID: MERCHANT_TEST_KEY},
        expected_kid=models.MERCHANT_KID,
    )

    retry = client.post("/v1/fulfillments", json=guarantee)
    assert retry.status_code == 200
    retry_body = retry.json()
    assert retry_body["idempotent"] is True
    assert retry_body["receipt"] == first_body["receipt"]
    assert merchant_service.count_fulfillments() == 1

    status = client.get("/v1/fulfillments/order-1/guarantee-1")
    assert status.status_code == 200
    assert status.json()["fulfillmentId"] == first_body["fulfillmentId"]


def test_tamper_and_customer_proof_fail_closed_without_reflection(client, merchant_service):
    quote = _quote(client)
    guarantee = _guarantee(quote)
    guarantee["paymentPayload"]["accepted"]["amount"] = "1251"
    tampered = client.post("/v1/fulfillments", json=guarantee)
    assert tampered.status_code == 400
    assert tampered.json()["code"] == "QUOTE_MISMATCH"
    assert merchant_service.count_fulfillments() == 0

    raw_secret = "raw-customer-proof-must-not-return"
    rejected = client.post(
        "/v1/fulfillments",
        json={**_guarantee(quote), "customerProof": raw_secret},
    )
    assert rejected.status_code == 422
    assert rejected.json()["code"] == "INVALID_SCHEMA"
    assert raw_secret not in rejected.text
    assert merchant_service.count_fulfillments() == 0

    invalid_signature = _guarantee(quote)
    invalid_signature["paymentPayload"]["payload"]["payableEntryId"] = "tampered"
    rejected_signature = client.post("/v1/fulfillments", json=invalid_signature)
    assert rejected_signature.status_code == 400
    assert rejected_signature.json()["code"] == "INVALID_SIGNATURE"
    assert merchant_service.count_fulfillments() == 0


def test_failure_and_timeout_fixtures_are_queryable_and_idempotent(client, merchant_service):
    failed_quote = _quote(client, "failure")
    failed_guarantee = _guarantee(failed_quote, "failure")
    failed = client.post(
        "/v1/fulfillments",
        json=failed_guarantee,
        headers={"X-Demo-Test-Fault": "failure"},
    )
    assert failed.status_code == 200
    assert failed.json()["state"] == "failed"
    assert failed.json()["receipt"]["status"] == "failed"

    timeout_quote = _quote(client, "timeout")
    timeout_guarantee = _guarantee(timeout_quote, "timeout")
    timeout = client.post(
        "/v1/fulfillments",
        json=timeout_guarantee,
        headers={"X-Demo-Test-Fault": "timeout"},
    )
    assert timeout.status_code == 504
    assert timeout.json()["code"] == "SETTLEMENT_UNKNOWN"
    assert timeout.json()["retryable"] is True

    status = client.get("/v1/fulfillments/order-timeout/guarantee-timeout")
    assert status.status_code == 200
    assert status.json()["state"] == "fulfilled"

    retry = client.post("/v1/fulfillments", json=timeout_guarantee)
    assert retry.status_code == 200
    assert retry.json()["idempotent"] is True
    assert merchant_service.count_fulfillments() == 2


def test_payout_status_request_is_merchant_signed_and_tenant_bound(client):
    response = client.post(
        "/v1/payout-status-requests",
        json={"payoutId": "payout-123", "correlationId": "corr-payout"},
    )
    assert response.status_code == 200
    signed_request = response.json()
    assert signed_request["skill"] == "payout_status"
    assert signed_request["path"] == "/v1/payouts/payout-123"
    assert signed_request["issuer"] == "demo-merchant"
    assert signed_request["audience"] == "mediation-platform"
    assert signed_request["actor"] == {
        "role": "merchant",
        "merchantId": "demo-merchant",
    }
    service_module.verify_document(
        signed_request,
        keys={models.MERCHANT_KID: MERCHANT_TEST_KEY},
        expected_kid=models.MERCHANT_KID,
    )


def test_missing_runtime_keys_is_not_ready_without_exposing_configuration(monkeypatch):
    monkeypatch.delenv("PAYMENT_DEMO_MERCHANT_HMAC_KEY", raising=False)
    monkeypatch.delenv("PAYMENT_DEMO_MEDIATOR_HMAC_KEY", raising=False)
    with TestClient(app_module.create_app()) as unconfigured:
        response = unconfigured.get("/ready")
        assert response.status_code == 503
        assert response.json()["status"] == "not-ready"
        assert "key" not in response.text.lower()
