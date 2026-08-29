from __future__ import annotations

import base64
import itertools
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from secure_mediation_agent.payment_marketplace.api import create_app
from secure_mediation_agent.payment_marketplace.auth import build_request_auth
from secure_mediation_agent.payment_marketplace.canonical import (
    canonical_bytes,
    digest_object,
    with_signature,
)
from secure_mediation_agent.payment_marketplace.config import (
    CUSTOMER_KID,
    CUSTOMER_SUBJECT,
    MERCHANT_KID,
    MERCHANT_SUBJECT,
    OPERATOR_KID,
    OPERATOR_SUBJECT,
    PROFILE_URI,
)
from secure_mediation_agent.payment_marketplace.ledger import Ledger
from secure_mediation_agent.payment_marketplace.merchant_client import (
    MerchantClientError,
    MerchantTimeout,
)
from secure_mediation_agent.payment_marketplace.rail import LocalPaymentRail
from secure_mediation_agent.payment_marketplace.service import MarketplaceError, MarketplaceService
from secure_mediation_agent.payment_marketplace.store import MarketplaceStore
from secure_mediation_agent.payment_marketplace.trusted_surface import TrustedSurface


NOW = 1_800_000_000
EXTENSION_HEADERS = {"X-A2A-Extensions": PROFILE_URI}


class StubMerchant:
    def __init__(self) -> None:
        self.fulfillment_count = 0
        self.last_receipt: dict | None = None

    def create_quote(self, request: dict) -> dict:
        quote_id = f"quote:{request['orderId']}"
        checkout_jwt = "eyJhbGciOiJIUzI1NiJ9.eyJkZW1vIjp0cnVlfQ.signature"
        requirement = with_signature(
            {
                "x402Version": 2,
                "profile": PROFILE_URI,
                "simulated": True,
                "resource": {
                    "url": f"a2a://demo-merchant/orders/{request['orderId']}",
                    "description": "demo booking",
                    "mimeType": "application/json",
                },
                "accepts": [
                    {
                        "scheme": "platform-credit",
                        "network": "demo:mediation-ledger",
                        "amount": "1250",
                        "asset": "USD",
                        "decimals": 2,
                        "payTo": MERCHANT_SUBJECT,
                        "maxTimeoutSeconds": 300,
                        "extra": {
                            "profile": PROFILE_URI,
                            "simulated": True,
                            "quoteId": quote_id,
                            "orderId": request["orderId"],
                            "merchantId": MERCHANT_SUBJECT,
                            "pricingPolicyVersion": "zero-fee-v1",
                            "fulfillmentTermsDigest": "sha256:" + "a" * 64,
                        },
                    }
                ],
                "quote": {
                    "issuer": MERCHANT_SUBJECT,
                    "audience": "mediation-platform",
                    "orderId": request["orderId"],
                    "taskId": request["taskId"],
                    "quoteId": quote_id,
                    "merchantId": MERCHANT_SUBJECT,
                    "pricingPolicyVersion": "zero-fee-v1",
                    "checkoutJwt": checkout_jwt,
                    "iat": NOW,
                    "exp": NOW + 300,
                },
            },
            kid=MERCHANT_KID,
        )
        return {
            "requirement": requirement,
            "checkoutJwt": checkout_jwt,
            "quoteDigest": digest_object(requirement),
        }

    def fulfill(self, request: dict) -> dict:
        guarantee = request["paymentPayload"]["payload"]
        fault = request.get("_testFault")
        self.fulfillment_count += 1
        state = "failed" if fault == "failure" else "fulfilled"
        receipt = with_signature(
            {
                "receiptType": "merchant-order",
                "receiptId": f"merchant-receipt:{guarantee['orderId']}",
                "profile": PROFILE_URI,
                "simulated": True,
                "status": state,
                "issuedAt": NOW,
                "issuer": MERCHANT_SUBJECT,
                "subject": "mediation-platform",
                "orderId": guarantee["orderId"],
                "quoteId": guarantee["quoteId"],
                "guaranteeId": guarantee["guaranteeId"],
                "fulfillmentId": f"fulfillment:{guarantee['orderId']}",
                "relatedDigests": {"guarantee": digest_object(request["paymentPayload"])},
            },
            kid=CUSTOMER_KID if fault == "wrong-signer" else MERCHANT_KID,
        )
        self.last_receipt = receipt
        if fault == "timeout":
            raise MerchantTimeout("simulated timeout")
        return {
            "state": state,
            "fulfillmentId": f"fulfillment:{guarantee['orderId']}",
            "receipt": receipt,
            "idempotent": False,
        }

    def fulfillment_status(self, order_id: str, guarantee_id: str) -> dict:
        assert self.last_receipt is not None
        return {
            "orderId": order_id,
            "guaranteeId": guarantee_id,
            "state": self.last_receipt["status"],
            "fulfillmentId": self.last_receipt["fulfillmentId"],
            "guaranteeDigest": self.last_receipt["relatedDigests"]["guarantee"],
            "receipt": self.last_receipt,
        }


@pytest.fixture
def service(tmp_path):
    counter = itertools.count(1)
    store = MarketplaceStore(tmp_path / "business.db", tmp_path / "evidence.db")
    ledger = Ledger(store)
    rail = LocalPaymentRail(store, allow_test_faults=True)
    result = MarketplaceService(
        store,
        ledger,
        rail,
        StubMerchant(),
        id_factory=lambda prefix: f"{prefix}-{next(counter):04d}",
        clock=lambda: NOW,
    )
    result.seed_demo_onboarding("http://127.0.0.1:8005")
    return result


def _approval(start: dict) -> dict:
    surface = TrustedSurface(clock=lambda: datetime.fromtimestamp(NOW, tz=timezone.utc))
    data = start["trustedSurfaceInput"]
    approval = surface.build_approval(
        checkout_jwt=data["checkoutJwt"],
        pricing=__import__(
            "secure_mediation_agent.payment_marketplace.models", fromlist=["PricingBreakdown"]
        ).PricingBreakdown.model_validate(start["pricing"]),
        audience=data["audience"],
        nonce=data["nonce"],
        order_id=data["orderId"],
        task_id=data["taskId"],
        quote_id=data["quoteId"],
        challenge_id=data["challengeId"],
    )
    return approval.model_dump(mode="json", by_alias=True)


def _payment_request(start: dict, *, fault: str | None = None) -> dict:
    result = {
        "approval": _approval(start),
        "paymentPayload": {
            "x402Version": 2,
            "accepted": start["requirement"]["accepts"][0],
            "payload": {"authorizationDigest": digest_object(_approval(start)["authorization"])},
        },
    }
    if fault:
        result["merchantFault"] = fault
    return result


def _restart_service(service, *, merchant=None) -> MarketplaceService:
    store = MarketplaceStore(service.store.business_db, service.store.evidence_db)
    return MarketplaceService(
        store,
        Ledger(store),
        LocalPaymentRail(store, allow_test_faults=True),
        merchant or service.merchant,
        id_factory=lambda prefix: f"{prefix}-restart",
        clock=lambda: NOW,
    )


def test_marketplace_happy_path_and_deferred_payout(service):
    start = service.start_order({}, idempotency_key="start-1")
    assert start["state"] == "payment_required"
    assert start["requirement"]["accepts"][0]["payTo"] == "mediation-platform"

    payment = _payment_request(start)
    completed = service.submit_payment(start["orderId"], payment, idempotency_key="pay-1")
    assert completed["state"] == "completed"
    assert completed["payable"]["state"] == "eligible"
    assert completed["guarantee"]["state"] == "accepted"
    assert len(completed["receipts"]) == 3
    assert service.rail.get_balance("demo-customer") == 98_750
    assert service.rail.get_balance("mediation-platform") == 1_250
    assert service.ledger.account_balance("simulated_cash") == 1_250

    payout = service.create_payout(
        merchant_id=MERCHANT_SUBJECT,
        idempotency_key="payout-1",
        actor_id=OPERATOR_SUBJECT,
        reason="manual demo payout",
    )
    assert payout["state"] == "paid"
    assert payout["netAmount"] == 1_250
    assert service.rail.get_balance("mediation-platform") == 0
    assert service.rail.get_balance("demo-merchant") == 1_250
    assert service.ledger.account_balance("simulated_cash") == 0
    assert service.ledger.all_journals_balanced()
    assert service.create_payout(
        merchant_id=MERCHANT_SUBJECT,
        idempotency_key="payout-1",
        actor_id=OPERATOR_SUBJECT,
        reason="manual demo payout",
    ) == payout


def test_completed_payment_retry_returns_cached_result_without_side_effect(service):
    start = service.start_order({}, idempotency_key="start-idempotent")
    assert service.start_order({}, idempotency_key="start-idempotent") == start
    payment = _payment_request(start)
    first = service.submit_payment(start["orderId"], payment, idempotency_key="pay-idempotent")
    second = service.submit_payment(start["orderId"], payment, idempotency_key="pay-idempotent")
    assert second == first
    assert service.merchant.fulfillment_count == 1
    assert len(service.store.fetch_business("SELECT * FROM rail_operations")) == 1


def test_fulfillment_failure_requires_and_completes_full_refund(service):
    start = service.start_order({}, idempotency_key="start-failure")
    payment = _payment_request(start)
    result = service.submit_payment(
        start["orderId"], payment, idempotency_key="pay-failure", merchant_fault="failure"
    )
    assert result["state"] == "refund_required"

    refund = service.refund_order(
        start["orderId"],
        idempotency_key="refund-1",
        actor_id=OPERATOR_SUBJECT,
        reason="merchant fulfillment failed",
    )
    assert refund["state"] == "settled"
    assert service.order_status(start["orderId"], customer_id=CUSTOMER_SUBJECT)["state"] == "refunded"
    assert service.rail.get_balance("demo-customer") == 100_000
    assert service.rail.get_balance("mediation-platform") == 0
    assert service.ledger.account_balance("simulated_cash") == 0
    assert service.refund_order(
        start["orderId"],
        idempotency_key="refund-1",
        actor_id=OPERATOR_SUBJECT,
        reason="merchant fulfillment failed",
    ) == refund


def test_timeout_is_reconciled_by_status_without_second_charge(service):
    start = service.start_order({}, idempotency_key="start-timeout")
    payment = _payment_request(start)
    with pytest.raises(MarketplaceError) as ambiguous:
        service.submit_payment(
            start["orderId"],
            payment,
            idempotency_key="pay-timeout",
            merchant_fault="timeout",
        )
    assert ambiguous.value.code == "SETTLEMENT_UNKNOWN"
    before = service.rail.get_balance("mediation-platform")
    with pytest.raises(MarketplaceError) as retrying:
        service.submit_payment(
            start["orderId"], payment, idempotency_key="pay-timeout"
        )
    assert retrying.value.code == "SETTLEMENT_UNKNOWN"
    assert service.merchant.fulfillment_count == 1

    reconciled = service.reconcile_order(
        start["orderId"],
        actor_id=OPERATOR_SUBJECT,
        reason="authoritative merchant status",
    )

    assert reconciled["state"] == "completed"
    assert reconciled["fulfillment"]["state"] == "fulfilled"
    assert service.rail.get_balance("mediation-platform") == before == 1_250
    assert len(service.store.fetch_business("SELECT * FROM rail_operations")) == 1
    assert (
        service.submit_payment(start["orderId"], payment, idempotency_key="pay-timeout")
        == reconciled
    )


def test_refund_settled_ledger_failure_reconciles_without_second_transfer(service, monkeypatch):
    start = service.start_order({}, idempotency_key="start-refund-ledger")
    service.submit_payment(
        start["orderId"],
        _payment_request(start),
        idempotency_key="pay-refund-ledger",
        merchant_fault="failure",
    )
    original = service.ledger.post_refund
    attempts = 0

    def fail_once(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected refund ledger failure")
        return original(**kwargs)

    monkeypatch.setattr(service.ledger, "post_refund", fail_once)
    with pytest.raises(MarketplaceError) as ambiguous:
        service.refund_order(
            start["orderId"],
            idempotency_key="refund-ledger",
            actor_id=OPERATOR_SUBJECT,
            reason="merchant failed",
        )
    assert ambiguous.value.code == "LEDGER_POST_FAILED"
    assert service.rail.get_balance("demo-customer") == 100_000

    reconciled = service.reconcile_order(
        start["orderId"], actor_id=OPERATOR_SUBJECT, reason="post refund journal"
    )
    assert reconciled["state"] == "settled"
    assert service.order_status(start["orderId"], customer_id=CUSTOMER_SUBJECT)["state"] == "refunded"
    assert len(service.store.fetch_business("SELECT * FROM rail_operations WHERE kind='refund'")) == 1


def test_crash_after_refund_transfer_reconciles_from_refunding(service, monkeypatch):
    start = service.start_order({}, idempotency_key="start-refund-crash")
    service.submit_payment(
        start["orderId"],
        _payment_request(start),
        idempotency_key="pay-refund-crash",
        merchant_fault="failure",
    )
    original = service.rail.refund

    def crash_after_transfer(**kwargs):
        original(**kwargs)
        raise RuntimeError("simulated process crash after refund")

    monkeypatch.setattr(service.rail, "refund", crash_after_transfer)
    with pytest.raises(RuntimeError, match="process crash"):
        service.refund_order(
            start["orderId"],
            idempotency_key="refund-crash",
            actor_id=OPERATOR_SUBJECT,
            reason="merchant failed",
        )
    monkeypatch.setattr(service.rail, "refund", original)
    assert service.order_status(start["orderId"], customer_id=CUSTOMER_SUBJECT)["state"] == "refunding"

    reconciled = service.reconcile_order(
        start["orderId"], actor_id=OPERATOR_SUBJECT, reason="resume settled refund"
    )
    assert reconciled["state"] == "settled"
    assert service.order_status(start["orderId"], customer_id=CUSTOMER_SUBJECT)["state"] == "refunded"


def test_payout_settled_ledger_failure_reconciles_without_second_transfer(service, monkeypatch):
    start = service.start_order({}, idempotency_key="start-payout-ledger")
    service.submit_payment(
        start["orderId"], _payment_request(start), idempotency_key="pay-payout-ledger"
    )
    original = service.ledger.post_payout
    attempts = 0

    def fail_once(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected payout ledger failure")
        return original(**kwargs)

    monkeypatch.setattr(service.ledger, "post_payout", fail_once)
    with pytest.raises(MarketplaceError) as ambiguous:
        service.create_payout(
            merchant_id=MERCHANT_SUBJECT,
            idempotency_key="payout-ledger",
            actor_id=OPERATOR_SUBJECT,
            reason="manual payout",
        )
    assert ambiguous.value.code == "LEDGER_POST_FAILED"
    payout = service.store.fetch_business("SELECT * FROM payouts")[0]
    assert service.rail.get_balance("demo-merchant") == 1_250

    reconciled = service.reconcile_payout(
        payout["payout_id"], actor_id=OPERATOR_SUBJECT, reason="post payout journal"
    )
    assert reconciled["state"] == "paid"
    assert service.rail.get_balance("demo-merchant") == 1_250
    assert len(service.store.fetch_business("SELECT * FROM rail_operations WHERE kind='payout'")) == 1
    assert service.ledger.all_journals_balanced()


def test_crash_after_payout_transfer_reconciles_from_settling(service, monkeypatch):
    start = service.start_order({}, idempotency_key="start-payout-crash")
    service.submit_payment(
        start["orderId"], _payment_request(start), idempotency_key="pay-payout-crash"
    )
    original = service.rail.payout

    def crash_after_transfer(**kwargs):
        original(**kwargs)
        raise RuntimeError("simulated process crash after payout")

    monkeypatch.setattr(service.rail, "payout", crash_after_transfer)
    with pytest.raises(RuntimeError, match="process crash"):
        service.create_payout(
            merchant_id=MERCHANT_SUBJECT,
            idempotency_key="payout-crash",
            actor_id=OPERATOR_SUBJECT,
            reason="manual payout",
        )
    monkeypatch.setattr(service.rail, "payout", original)
    payout = service.store.fetch_business("SELECT * FROM payouts")[0]
    assert payout["state"] == "settling"

    reconciled = service.reconcile_payout(
        payout["payout_id"], actor_id=OPERATOR_SUBJECT, reason="resume settled payout"
    )
    assert reconciled["state"] == "paid"
    assert service.rail.get_balance("demo-merchant") == 1_250


def test_tampered_approval_and_replayed_nonce_fail_closed(service):
    start = service.start_order({}, idempotency_key="start-tamper")
    payment = _payment_request(start)
    payment["approval"]["paymentMandate"]["payment_amount"]["amount"] = 1
    with pytest.raises(MarketplaceError) as tampered:
        service.submit_payment(start["orderId"], payment, idempotency_key="pay-tamper")
    assert tampered.value.code == "INVALID_SIGNATURE"
    assert service.rail.get_balance("demo-customer") == 100_000


def test_approval_is_bound_to_exact_merchant_checkout(service):
    start = service.start_order({}, idempotency_key="start-checkout-binding")
    trusted = start["trustedSurfaceInput"]
    forged = TrustedSurface(
        clock=lambda: datetime.fromtimestamp(NOW, tz=timezone.utc)
    ).build_approval(
        checkout_jwt="attacker.controlled.checkout.jwt",
        pricing=__import__(
            "secure_mediation_agent.payment_marketplace.models",
            fromlist=["PricingBreakdown"],
        ).PricingBreakdown.model_validate(start["pricing"]),
        audience=trusted["audience"],
        nonce=trusted["nonce"],
        order_id=trusted["orderId"],
        task_id=trusted["taskId"],
        quote_id=trusted["quoteId"],
        challenge_id=trusted["challengeId"],
    ).model_dump(mode="json", by_alias=True)
    payment = _payment_request(start)
    payment["approval"] = forged

    with pytest.raises(MarketplaceError) as rejected:
        service.submit_payment(
            start["orderId"], payment, idempotency_key="pay-checkout-binding"
        )
    assert rejected.value.code == "QUOTE_MISMATCH"
    assert service.rail.get_balance("demo-customer") == 100_000


def test_suspended_merchant_is_rejected_before_order_or_quote(service):
    with service.store.business_transaction() as connection:
        connection.execute(
            "UPDATE merchant_onboarding SET status='suspended' WHERE merchant_id=?",
            (MERCHANT_SUBJECT,),
        )
    with pytest.raises(MarketplaceError) as rejected:
        service.start_order({}, idempotency_key="start-suspended")
    assert rejected.value.code == "MERCHANT_SUSPENDED"
    assert service.store.fetch_business("SELECT * FROM orders") == []


def test_wrong_merchant_receipt_key_is_rejected(service):
    start = service.start_order({}, idempotency_key="start-wrong-signer")
    with pytest.raises(MarketplaceError) as rejected:
        service.submit_payment(
            start["orderId"],
            _payment_request(start),
            idempotency_key="pay-wrong-signer",
            merchant_fault="wrong-signer",
        )
    assert rejected.value.code == "INVALID_SIGNATURE"
    assert service.order_status(start["orderId"], customer_id=CUSTOMER_SUBJECT)["state"] == "refund_required"


def test_charge_settled_ledger_failure_reconciles_without_second_charge(service, monkeypatch):
    start = service.start_order({}, idempotency_key="start-ledger-recovery")
    original = service.ledger.post_charge
    attempts = 0

    def fail_once(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected ledger failure")
        return original(**kwargs)

    monkeypatch.setattr(service.ledger, "post_charge", fail_once)
    with pytest.raises(MarketplaceError) as ambiguous:
        service.submit_payment(
            start["orderId"],
            _payment_request(start),
            idempotency_key="pay-ledger-recovery",
        )
    assert ambiguous.value.code == "LEDGER_POST_FAILED"
    assert service.rail.get_balance("mediation-platform") == 1_250
    assert service.store.fetch_business("SELECT * FROM journal_transactions") == []

    completed = service.reconcile_order(
        start["orderId"], actor_id=OPERATOR_SUBJECT, reason="post settled charge journal"
    )
    assert completed["state"] == "completed"
    assert service.rail.get_balance("mediation-platform") == 1_250
    assert len(service.store.fetch_business("SELECT * FROM rail_operations")) == 1
    assert service.ledger.all_journals_balanced()


def test_crash_after_charge_transfer_reconciles_from_charge_processing(service, monkeypatch):
    start = service.start_order({}, idempotency_key="start-charge-crash")
    original = service.rail.settle_charge

    def crash_after_transfer(**kwargs):
        original(**kwargs)
        raise RuntimeError("simulated process crash after charge")

    monkeypatch.setattr(service.rail, "settle_charge", crash_after_transfer)
    with pytest.raises(RuntimeError, match="process crash"):
        service.submit_payment(
            start["orderId"], _payment_request(start), idempotency_key="pay-charge-crash"
        )
    monkeypatch.setattr(service.rail, "settle_charge", original)
    assert service.order_status(start["orderId"], customer_id=CUSTOMER_SUBJECT)["state"] == "charge_processing"

    completed = service.reconcile_order(
        start["orderId"], actor_id=OPERATOR_SUBJECT, reason="resume settled charge"
    )
    assert completed["state"] == "completed"
    assert len(service.store.fetch_business("SELECT * FROM rail_operations WHERE kind='charge'")) == 1


def test_nonce_and_charge_processing_commit_roll_back_together(service, monkeypatch):
    start = service.start_order({}, idempotency_key="start-nonce-atomic")
    payment = _payment_request(start)

    def crash_inside_transaction(step: str):
        assert step == "nonce-consumed"
        raise RuntimeError("simulated crash after nonce insert")

    monkeypatch.setattr(service, "_after_atomic_recovery_step", crash_inside_transaction)
    with pytest.raises(RuntimeError, match="nonce insert"):
        service.submit_payment(
            start["orderId"], payment, idempotency_key="pay-nonce-atomic"
        )

    assert service.order_status(start["orderId"], customer_id=CUSTOMER_SUBJECT)["state"] == "payment_required"
    assert service.store.fetch_business("SELECT * FROM used_nonces") == []
    assert service.store.fetch_business("SELECT * FROM rail_operations") == []

    monkeypatch.setattr(service, "_after_atomic_recovery_step", lambda step: None)
    completed = service.submit_payment(
        start["orderId"], payment, idempotency_key="pay-nonce-atomic"
    )
    assert completed["state"] == "completed"
    assert len(service.store.fetch_business("SELECT * FROM used_nonces")) == 1
    assert len(service.store.fetch_business("SELECT * FROM rail_operations WHERE kind='charge'")) == 1


def test_restart_resumes_charge_when_operation_was_not_created(service, monkeypatch):
    start = service.start_order({}, idempotency_key="start-charge-before-rail")

    def crash_before_rail(**kwargs):
        raise RuntimeError("simulated crash before rail operation")

    monkeypatch.setattr(service.rail, "settle_charge", crash_before_rail)
    with pytest.raises(RuntimeError, match="before rail"):
        service.submit_payment(
            start["orderId"],
            _payment_request(start),
            idempotency_key="pay-charge-before-rail",
        )
    assert service.order_status(start["orderId"], customer_id=CUSTOMER_SUBJECT)["state"] == "charge_processing"
    assert service.store.fetch_business("SELECT * FROM rail_operations") == []
    charge = service.store.fetch_business("SELECT * FROM charges WHERE order_id=?", (start["orderId"],))[0]

    restarted = _restart_service(service)
    completed = restarted.reconcile_order(
        start["orderId"], actor_id=OPERATOR_SUBJECT, reason="resume missing charge operation"
    )
    assert completed["state"] == "completed"
    operations = restarted.store.fetch_business("SELECT * FROM rail_operations WHERE kind='charge'")
    assert [item["operation_id"] for item in operations] == [charge["operation_id"]]
    assert operations[0]["idempotency_key"] == f"charge:{charge['idempotency_key']}"


def test_restart_resumes_payable_posted_without_new_charge(service, monkeypatch):
    start = service.start_order({}, idempotency_key="start-payable-posted")
    original = service._load_or_create_guarantee

    def crash_before_guarantee(**kwargs):
        raise RuntimeError("simulated crash after payable")

    monkeypatch.setattr(service, "_load_or_create_guarantee", crash_before_guarantee)
    with pytest.raises(RuntimeError, match="after payable"):
        service.submit_payment(
            start["orderId"],
            _payment_request(start),
            idempotency_key="pay-payable-posted",
        )
    assert service.order_status(start["orderId"], customer_id=CUSTOMER_SUBJECT)["state"] == "payable_posted"
    assert len(service.store.fetch_business("SELECT * FROM rail_operations WHERE kind='charge'")) == 1
    assert service.store.fetch_business("SELECT * FROM guarantees") == []
    monkeypatch.setattr(service, "_load_or_create_guarantee", original)

    restarted = _restart_service(service)
    completed = restarted.reconcile_order(
        start["orderId"], actor_id=OPERATOR_SUBJECT, reason="resume payable posted"
    )
    assert completed["state"] == "completed"
    assert len(restarted.store.fetch_business("SELECT * FROM rail_operations WHERE kind='charge'")) == 1
    assert len(restarted.store.fetch_business("SELECT * FROM journal_transactions WHERE event_type='charge'")) == 1
    assert len(restarted.store.fetch_business("SELECT * FROM guarantees")) == 1


def test_restart_recovers_guarantee_row_from_exact_evidence(service, monkeypatch):
    start = service.start_order({}, idempotency_key="start-guarantee-evidence")

    def crash_after_evidence(step: str):
        if step == "guarantee-evidence-committed":
            raise RuntimeError("simulated crash after guarantee evidence")

    monkeypatch.setattr(service, "_after_atomic_recovery_step", crash_after_evidence)
    with pytest.raises(RuntimeError, match="guarantee evidence"):
        service.submit_payment(
            start["orderId"],
            _payment_request(start),
            idempotency_key="pay-guarantee-evidence",
        )
    assert service.order_status(start["orderId"], customer_id=CUSTOMER_SUBJECT)["state"] == "payable_posted"
    assert service.store.fetch_business("SELECT * FROM guarantees") == []
    evidence_id = f"evidence:guarantee:{start['orderId']}"
    exact_before = service.store.read_evidence(
        evidence_id, actor_id=OPERATOR_SUBJECT, actor_role="operator"
    )

    restarted = _restart_service(service)
    completed = restarted.reconcile_order(
        start["orderId"], actor_id=OPERATOR_SUBJECT, reason="recover evidence-first guarantee"
    )
    assert completed["state"] == "completed"
    persisted = restarted.store.fetch_business("SELECT * FROM guarantees")[0]
    assert persisted["evidence_id"] == evidence_id
    assert restarted.store.read_evidence(
        evidence_id, actor_id=OPERATOR_SUBJECT, actor_role="operator"
    ) == exact_before


def test_restart_completes_fulfilling_state_without_recharge(service, monkeypatch):
    start = service.start_order({}, idempotency_key="start-fulfilling-crash")

    def crash_while_fulfilling(step: str):
        if step == "fulfilling-persisted":
            raise RuntimeError("simulated crash while fulfilling")

    monkeypatch.setattr(service, "_after_atomic_recovery_step", crash_while_fulfilling)
    with pytest.raises(RuntimeError, match="while fulfilling"):
        service.submit_payment(
            start["orderId"],
            _payment_request(start),
            idempotency_key="pay-fulfilling-crash",
        )
    assert service.order_status(start["orderId"], customer_id=CUSTOMER_SUBJECT)["state"] == "fulfilling"
    assert len(service.store.fetch_business("SELECT * FROM fulfillments")) == 1

    restarted = _restart_service(service, merchant=service.merchant)
    completed = restarted.reconcile_order(
        start["orderId"], actor_id=OPERATOR_SUBJECT, reason="resume fulfilling"
    )
    assert completed["state"] == "completed"
    assert len(restarted.store.fetch_business("SELECT * FROM rail_operations WHERE kind='charge'")) == 1
    assert len(restarted.store.fetch_business("SELECT * FROM fulfillments")) == 1


def test_missing_merchant_status_redelivers_identical_guarantee_evidence(service, monkeypatch):
    start = service.start_order({}, idempotency_key="start-guarantee-redelivery")
    transmitted: list[bytes] = []
    original_fulfill = service.merchant.fulfill
    first = True

    def timeout_without_merchant_record(request: dict):
        nonlocal first
        transmitted.append(canonical_bytes(request["paymentPayload"]))
        if first:
            first = False
            raise MerchantTimeout("delivery status unknown")
        return original_fulfill(request)

    def status_not_found(order_id: str, guarantee_id: str):
        raise MerchantClientError("FULFILLMENT_NOT_FOUND")

    monkeypatch.setattr(service.merchant, "fulfill", timeout_without_merchant_record)
    monkeypatch.setattr(service.merchant, "fulfillment_status", status_not_found)
    with pytest.raises(MarketplaceError) as unknown:
        service.submit_payment(
            start["orderId"],
            _payment_request(start),
            idempotency_key="pay-guarantee-redelivery",
        )
    assert unknown.value.code == "SETTLEMENT_UNKNOWN"
    guarantee = service.store.fetch_business("SELECT * FROM guarantees")[0]
    exact_evidence = service.store.read_evidence(
        guarantee["evidence_id"], actor_id=OPERATOR_SUBJECT, actor_role="operator"
    )

    restarted = _restart_service(service, merchant=service.merchant)
    completed = restarted.reconcile_order(
        start["orderId"], actor_id=OPERATOR_SUBJECT, reason="merchant status absent"
    )
    assert completed["state"] == "completed"
    assert transmitted == [exact_evidence, exact_evidence]
    assert len(restarted.store.fetch_business("SELECT * FROM guarantees")) == 1


def test_restart_resumes_refund_record_with_no_rail_operation(service, monkeypatch):
    start = service.start_order({}, idempotency_key="start-refund-before-rail")
    service.submit_payment(
        start["orderId"],
        _payment_request(start),
        idempotency_key="pay-refund-before-rail",
        merchant_fault="failure",
    )

    def crash_before_refund(**kwargs):
        raise RuntimeError("simulated crash before refund operation")

    monkeypatch.setattr(service.rail, "refund", crash_before_refund)
    with pytest.raises(RuntimeError, match="before refund"):
        service.refund_order(
            start["orderId"],
            idempotency_key="refund-before-rail",
            actor_id=OPERATOR_SUBJECT,
            reason="merchant failure",
        )
    refund = service.store.fetch_business("SELECT * FROM refunds")[0]
    assert service.store.fetch_business("SELECT * FROM rail_operations WHERE kind='refund'") == []

    restarted = _restart_service(service)
    completed = restarted.refund_order(
        start["orderId"],
        idempotency_key="refund-before-rail",
        actor_id=OPERATOR_SUBJECT,
        reason="merchant failure",
    )
    assert completed["state"] == "settled"
    operations = restarted.store.fetch_business("SELECT * FROM rail_operations WHERE kind='refund'")
    assert [item["operation_id"] for item in operations] == [refund["operation_id"]]
    assert operations[0]["idempotency_key"] == "refund:refund-before-rail"


def test_restart_resumes_payout_record_with_no_rail_operation(service, monkeypatch):
    start = service.start_order({}, idempotency_key="start-payout-before-rail")
    service.submit_payment(
        start["orderId"],
        _payment_request(start),
        idempotency_key="pay-payout-before-rail",
    )

    def crash_before_payout(**kwargs):
        raise RuntimeError("simulated crash before payout operation")

    monkeypatch.setattr(service.rail, "payout", crash_before_payout)
    with pytest.raises(RuntimeError, match="before payout"):
        service.create_payout(
            merchant_id=MERCHANT_SUBJECT,
            idempotency_key="payout-before-rail",
            actor_id=OPERATOR_SUBJECT,
            reason="manual payout",
        )
    payout = service.store.fetch_business("SELECT * FROM payouts")[0]
    assert service.store.fetch_business("SELECT * FROM rail_operations WHERE kind='payout'") == []

    restarted = _restart_service(service)
    completed = restarted.create_payout(
        merchant_id=MERCHANT_SUBJECT,
        idempotency_key="payout-before-rail",
        actor_id=OPERATOR_SUBJECT,
        reason="manual payout",
    )
    assert completed["state"] == "paid"
    operations = restarted.store.fetch_business("SELECT * FROM rail_operations WHERE kind='payout'")
    assert [item["operation_id"] for item in operations] == [payout["operation_id"]]
    assert operations[0]["idempotency_key"] == "payout:payout-before-rail"


def _signed_body(path: str, body: dict, *, subject: str, role: str, tenant: str, kid: str, nonce: str) -> dict:
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
            nonce=nonce,
            timestamp=NOW,
        ),
    }


def test_http_and_a2a_contract(service, monkeypatch):
    monkeypatch.setattr("secure_mediation_agent.payment_marketplace.auth.time.time", lambda: NOW)
    with TestClient(create_app(service)) as client:
        card = client.get("/.well-known/agent-card.json").json()
        assert card["protocolVersion"] == "0.3.0"
        assert card["capabilities"]["extensions"][0]["params"]["sdkVersion"] == "0.3.19"
        assert client.post("/v1/orders", json={}).status_code == 422

        start_body: dict = {}
        start_request = _signed_body(
            "/v1/orders",
            start_body,
            subject=CUSTOMER_SUBJECT,
            role="customer",
            tenant=CUSTOMER_SUBJECT,
            kid=CUSTOMER_KID,
            nonce="http-start",
        )
        started_response = client.post(
            "/v1/orders",
            json=start_request,
            headers={**EXTENSION_HEADERS, "Idempotency-Key": "http-start-1"},
        )
        assert started_response.status_code == 200, started_response.text
        started = started_response.json()

        payment_body = _payment_request(started)
        payment_request = _signed_body(
            f"/v1/orders/{started['orderId']}/payment",
            payment_body,
            subject=CUSTOMER_SUBJECT,
            role="customer",
            tenant=CUSTOMER_SUBJECT,
            kid=CUSTOMER_KID,
            nonce="http-payment",
        )
        paid_response = client.post(
            f"/v1/orders/{started['orderId']}/payment",
            json=payment_request,
            headers={**EXTENSION_HEADERS, "Idempotency-Key": "http-pay-1"},
        )
        assert paid_response.status_code == 200, paid_response.text
        assert paid_response.json()["state"] == "completed"

        payout_body = {"merchantId": MERCHANT_SUBJECT, "reason": "container demo"}
        payout_request = _signed_body(
            "/internal/v1/payouts",
            payout_body,
            subject=OPERATOR_SUBJECT,
            role="operator",
            tenant=OPERATOR_SUBJECT,
            kid=OPERATOR_KID,
            nonce="http-payout",
        )
        payout_response = client.post(
            "/internal/v1/payouts",
            json=payout_request,
            headers={"Idempotency-Key": "http-payout-1"},
        )
        assert payout_response.status_code == 200, payout_response.text
        assert payout_response.json()["state"] == "paid"

        a2a_body = {}
        a2a_request = _signed_body(
            "/v1/orders",
            a2a_body,
            subject=CUSTOMER_SUBJECT,
            role="customer",
            tenant=CUSTOMER_SUBJECT,
            kid=CUSTOMER_KID,
            nonce="a2a-start",
        )
        envelope = {
            "jsonrpc": "2.0",
            "id": "rpc-1",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "m-1",
                    "role": "user",
                    "parts": [{"kind": "data", "data": {"action": "start_order", "request": a2a_request}}],
                }
            },
        }
        a2a = client.post(
            "/a2a",
            json=envelope,
            headers={**EXTENSION_HEADERS, "Idempotency-Key": "a2a-start-1"},
        )
        assert a2a.status_code == 200
        assert a2a.json()["result"]["status"]["state"] == "input-required"
        assert a2a.json()["result"]["metadata"]["x402.payment"]["status"] == "payment-required"
