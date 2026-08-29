"""Deterministic marketplace payment use cases.

The service implements one real-time customer-to-marketplace simulation charge,
posts the merchant payable, issues a signed platform-credit guarantee, and then
invokes the paid merchant.  Merchant payout is a separate operator lifecycle.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from .canonical import canonical_bytes, digest_object, verify_payload_signature, with_signature
from .config import (
    CUSTOMER_SUBJECT,
    MEDIATOR_KID,
    MEDIATOR_SUBJECT,
    MERCHANT_KID,
    MERCHANT_SUBJECT,
    PRICING_POLICY_VERSION,
    PROFILE_URI,
)
from .ledger import Ledger
from .merchant_client import MerchantClient, MerchantClientError, MerchantTimeout
from .models import (
    CheckoutMandate,
    PaymentAcceptance,
    PaymentAcceptanceExtra,
    PaymentRequired,
    PaymentResource,
    PricingBreakdown,
    TrustedSurfaceApproval,
    calculate_zero_fee_pricing,
)
from .rail import LocalPaymentRail
from .store import MarketplaceStore, compact_json, utc_now
from .trusted_surface import verify_approval


class MarketplaceError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        retryable: bool = False,
        correlation_id: str = "payment-marketplace",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.correlation_id = correlation_id

    def envelope(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "correlationId": self.correlation_id,
        }


IdFactory = Callable[[str], str]
EpochClock = Callable[[], int]


def default_id_factory(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def epoch_now() -> int:
    return int(time.time())


@dataclass(frozen=True)
class MarketplaceSettings:
    merchant_id: str = MERCHANT_SUBJECT
    customer_id: str = CUSTOMER_SUBJECT
    marketplace_id: str = MEDIATOR_SUBJECT
    guarantee_ttl_seconds: int = 300
    schema_version: int = 1


class MarketplaceService:
    def __init__(
        self,
        store: MarketplaceStore,
        ledger: Ledger,
        rail: LocalPaymentRail,
        merchant: MerchantClient,
        *,
        settings: MarketplaceSettings | None = None,
        id_factory: IdFactory = default_id_factory,
        clock: EpochClock = epoch_now,
    ) -> None:
        self.store = store
        self.ledger = ledger
        self.rail = rail
        self.merchant = merchant
        self.settings = settings or MarketplaceSettings()
        self.id_factory = id_factory
        self.clock = clock

    # -- setup/readiness -----------------------------------------------------

    def seed_demo_onboarding(self, endpoint: str) -> None:
        now = utc_now()
        with self.store.business_transaction() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO merchant_onboarding
                   (merchant_id, version, status, key_id, endpoint, agreement_version,
                    pricing_policy_version, payout_destination, valid_from, valid_to,
                    schema_version, created_at)
                   VALUES (?, 'demo-onboarding-v1', 'active', ?, ?, 'demo-agreement-v1',
                           ?, 'demo-merchant', ?, NULL, ?, ?)""",
                (
                    self.settings.merchant_id,
                    MERCHANT_KID,
                    endpoint,
                    PRICING_POLICY_VERSION,
                    now,
                    self.settings.schema_version,
                    now,
                ),
            )

    def ready(self) -> tuple[bool, dict[str, Any]]:
        versions = self.store.schema_versions()
        try:
            onboarding = self._require_active_merchant(self.settings.merchant_id)
        except MarketplaceError:
            onboarding = None
        reconciliation = self.rail.reconcile_platform_cash(self.ledger)
        result = {
            "profile": PROFILE_URI,
            "simulated": True,
            "schemaVersions": versions,
            "merchantOnboarded": onboarding is not None,
            "railLedgerReconciliation": reconciliation,
            "testFaultsEnabled": self.rail.allow_test_faults,
        }
        return bool(
            onboarding
            and reconciliation["balanced"]
            and all(value == 1 for value in versions.values())
        ), result

    # -- order ---------------------------------------------------------------

    def start_order(self, request: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        if request.get("productId", "demo-paid-booking") != "demo-paid-booking" or request.get("quantity", 1) != 1:
            raise MarketplaceError("INVALID_SCHEMA", "Only the deterministic demo product is available.")
        self._require_active_merchant(self.settings.merchant_id)
        request_hash = digest_object(request)
        idem = self.store.begin_idempotency(
            "order-start", self.settings.customer_id, idempotency_key, request_hash
        )
        cached = self._cached_idempotent_response(idem)
        if cached is not None:
            return cached
        order_id = self.id_factory("order")
        task_id = str(request.get("taskId") or self.id_factory("task"))
        context_id = str(request.get("contextId") or self.id_factory("context"))
        correlation_id = str(request.get("correlationId") or self.id_factory("corr"))
        self.store.save_task(
            task_id,
            context_id,
            "working",
            actor_id=self.settings.customer_id,
            tenant_id=self.settings.customer_id,
            metadata={"profile": PROFILE_URI, "simulated": True},
            expected_version=0,
        )
        self.store.create_order(
            order_id,
            task_id,
            context_id,
            self.settings.customer_id,
            self.settings.merchant_id,
            correlation_id=correlation_id,
        )
        quote_response = self.merchant.create_quote(
            {
                "orderId": order_id,
                "taskId": task_id,
                "correlationId": correlation_id,
                "productId": "demo-paid-booking",
                "quantity": 1,
                "audience": self.settings.marketplace_id,
            }
        )
        requirement, quote, quote_digest = self._verify_quote(
            quote_response, order_id=order_id, task_id=task_id
        )
        checkout_jwt = quote["checkoutJwt"]
        quote_id = quote["quoteId"]
        amount = int(requirement["accepts"][0]["amount"])
        calculated_at = datetime.fromtimestamp(self.clock(), tz=timezone.utc)
        pricing = calculate_zero_fee_pricing(amount, calculated_at=calculated_at)
        challenge_id = self.id_factory("challenge")
        nonce = self.id_factory("nonce")

        quote_evidence_id = self.id_factory("evidence-quote")
        quote_intent = self.store.put_evidence(
            intent_id=f"quote:{quote_id}",
            evidence_id=quote_evidence_id,
            tenant_type="merchant",
            tenant_id=self.settings.merchant_id,
            kind="merchant-quote",
            exact_bytes=canonical_bytes(requirement),
            kid=MERCHANT_KID,
        )
        if quote_intent["state"] != "committed":
            raise MarketplaceError("LEDGER_POST_FAILED", "Quote evidence is not durable.", retryable=True)

        upstream = PaymentRequired(
            resource=PaymentResource(
                url=f"a2a://{self.settings.marketplace_id}/orders/{order_id}",
                description="simulated marketplace order",
            ),
            accepts=[
                PaymentAcceptance(
                    scheme="exact-simulated",
                    network="demo:local",
                    amount=str(pricing.customer_total),
                    asset="USD",
                    decimals=2,
                    payTo=self.settings.marketplace_id,
                    maxTimeoutSeconds=300,
                    extra=PaymentAcceptanceExtra(quoteDigest=quote_digest),
                )
            ],
        ).model_dump(mode="json", by_alias=True, exclude_none=True)
        checkout_mandate = CheckoutMandate(
            checkout_jwt=checkout_jwt,
            checkout_hash=self._checkout_hash(checkout_jwt),
            iat=self.clock(),
            exp=min(int(quote["exp"]), self.clock() + 300),
        ).model_dump(mode="json", by_alias=True)

        now = utc_now()
        with self.store.business_transaction() as conn:
            conn.execute(
                """INSERT INTO merchant_quotes
                   (quote_id, order_id, merchant_id, requirement_digest, evidence_id,
                    merchandise_amount, policy_version, state, iat, exp, schema_version, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'verified', ?, ?, ?, ?)""",
                (
                    quote_id,
                    order_id,
                    self.settings.merchant_id,
                    quote_digest,
                    quote_evidence_id,
                    amount,
                    PRICING_POLICY_VERSION,
                    str(quote["iat"]),
                    str(quote["exp"]),
                    self.settings.schema_version,
                    now,
                ),
            )
            pricing_wire = pricing.model_dump(mode="json", by_alias=True)
            conn.execute(
                """INSERT INTO pricing
                   (order_id, policy_version, merchandise_amount, customer_surcharge,
                    collection_rail_cost, customer_total, provider_commission,
                    merchant_payable_amount, payout_rail_cost, asset, decimals, network,
                    rounding_rule, calculated_at, schema_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    order_id,
                    pricing.policy_version,
                    pricing.merchandise_amount,
                    pricing.customer_surcharge,
                    pricing.collection_rail_cost,
                    pricing.customer_total,
                    pricing.provider_commission,
                    pricing.merchant_payable_amount,
                    pricing.payout_rail_cost,
                    pricing.asset,
                    pricing.decimals,
                    pricing.network,
                    pricing.rounding_rule,
                    pricing_wire["calculatedAt"],
                    self.settings.schema_version,
                ),
            )
            charge_id = self.id_factory("charge")
            conn.execute(
                """INSERT INTO charges
                   (charge_id, order_id, challenge_id, payer_id, pay_to, amount, asset,
                    nonce, state, idempotency_key, schema_version, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'USD', ?, 'required', ?, ?, ?, ?)""",
                (
                    charge_id,
                    order_id,
                    challenge_id,
                    self.settings.customer_id,
                    self.settings.marketplace_id,
                    pricing.customer_total,
                    nonce,
                    idempotency_key,
                    self.settings.schema_version,
                    now,
                    now,
                ),
            )
            conn.execute("UPDATE orders SET quote_id=? WHERE order_id=?", (quote_id, order_id))
        order = self.store.get_order(order_id)
        assert order is not None
        self.store.update_order_state(
            order_id,
            "awaiting_quote",
            "payment_required",
            actor_id=self.settings.marketplace_id,
            reason="merchant-quote-verified",
            expected_version=int(order["version"]),
        )
        response = {
            "orderId": order_id,
            "taskId": task_id,
            "contextId": context_id,
            "correlationId": correlation_id,
            "state": "payment_required",
            "quoteId": quote_id,
            "merchantId": self.settings.merchant_id,
            "challengeId": challenge_id,
            "nonce": nonce,
            "requirement": upstream,
            "ap2": {"checkoutMandate": checkout_mandate},
            "pricing": pricing.model_dump(mode="json", by_alias=True),
            "trustedSurfaceInput": {
                "checkoutJwt": checkout_jwt,
                "audience": self.settings.marketplace_id,
                "nonce": nonce,
                "orderId": order_id,
                "taskId": task_id,
                "quoteId": quote_id,
                "challengeId": challenge_id,
            },
            "profile": PROFILE_URI,
            "simulated": True,
        }
        self.store.save_task(
            task_id,
            context_id,
            "input-required",
            actor_id=self.settings.customer_id,
            tenant_id=self.settings.customer_id,
            metadata={"orderId": order_id, "paymentStatus": "payment-required"},
            response=response,
            expected_version=1,
        )
        self.store.complete_idempotency(
            "order-start", self.settings.customer_id, idempotency_key, request_hash, response
        )
        return response

    def submit_payment(
        self,
        order_id: str,
        request: dict[str, Any],
        *,
        idempotency_key: str,
        merchant_fault: str | None = None,
    ) -> dict[str, Any]:
        request_hash = digest_object(request)
        idem = self.store.begin_idempotency(
            "payment-submit", self.settings.customer_id, idempotency_key, request_hash
        )
        order = self._owned_order(order_id, self.settings.customer_id)
        if idem["status"] == "hit" and idem["response"] is not None:
            return idem["response"]
        # A crash before the atomic nonce/state commit leaves the order at
        # payment_required.  The same normalized request may safely continue;
        # later states must be resumed through reconciliation instead.
        if idem["status"] == "hit" and order["state"] != "payment_required":
            raise MarketplaceError(
                "SETTLEMENT_UNKNOWN",
                "The payment operation is already durable and requires reconciliation.",
                status_code=409,
                retryable=True,
            )
        if order["state"] != "payment_required":
            raise MarketplaceError("INVALID_STATE_TRANSITION", "Order is not awaiting payment.")
        try:
            approval = TrustedSurfaceApproval.model_validate(request["approval"])
            verify_approval(approval)
        except Exception as exc:
            raise MarketplaceError("INVALID_SIGNATURE", "Closed mandate authorization is invalid.") from exc
        charge = self._one("SELECT * FROM charges WHERE order_id=?", (order_id,))
        quote = self._one("SELECT * FROM merchant_quotes WHERE order_id=?", (order_id,))
        pricing = self._one("SELECT * FROM pricing WHERE order_id=?", (order_id,))
        quote_requirement = self._quote_requirement(quote["evidence_id"])
        merchant_checkout = quote_requirement["quote"]["checkoutJwt"]
        authorization = approval.authorization
        now_epoch = self.clock()
        checks = (
            (authorization.order_id == order_id, "QUOTE_MISMATCH"),
            (authorization.task_id == order["task_id"], "QUOTE_MISMATCH"),
            (authorization.quote_id == order["quote_id"], "QUOTE_MISMATCH"),
            (authorization.challenge_id == charge["challenge_id"], "QUOTE_MISMATCH"),
            (authorization.nonce == charge["nonce"], "REPLAY_DETECTED"),
            (authorization.audience == self.settings.marketplace_id, "AUDIENCE_MISMATCH"),
            (approval.payment_mandate.payment_amount.amount == pricing["customer_total"], "AMOUNT_MISMATCH"),
            (approval.payment_mandate.payee.id == self.settings.marketplace_id, "PAYEE_MISMATCH"),
            (approval.checkout_mandate.checkout_jwt == merchant_checkout, "QUOTE_MISMATCH"),
            (approval.display.checkout_jwt == merchant_checkout, "QUOTE_MISMATCH"),
            (approval.checkout_mandate.checkout_hash == self._checkout_hash(merchant_checkout), "QUOTE_MISMATCH"),
            (approval.payment_mandate.transaction_id == self._checkout_hash(merchant_checkout), "QUOTE_MISMATCH"),
            (approval.payment_mandate.payment_instrument.id == self.settings.customer_id, "PAYEE_MISMATCH"),
            (approval.payment_mandate.payment_instrument.type == "simulation", "PAYEE_MISMATCH"),
            (approval.display.payment_instrument == approval.payment_mandate.payment_instrument, "QUOTE_MISMATCH"),
            (approval.display.pricing.model_dump(mode="json", by_alias=True) == self._pricing_public(pricing), "AMOUNT_MISMATCH"),
            (approval.checkout_mandate.iat <= now_epoch < approval.checkout_mandate.exp, "EXPIRED"),
            (approval.payment_mandate.iat <= now_epoch < approval.payment_mandate.exp, "EXPIRED"),
            (now_epoch < int(quote["exp"]), "EXPIRED"),
            (authorization.iat <= now_epoch < authorization.exp, "EXPIRED"),
        )
        for valid, code in checks:
            if not valid:
                raise MarketplaceError(code, "Payment proof binding is invalid.")
        payment_payload = request.get("paymentPayload", {})
        if payment_payload.get("x402Version") != 2 or not isinstance(payment_payload.get("payload"), dict):
            raise MarketplaceError("INVALID_SCHEMA", "x402 payment payload is invalid.")
        accepted = payment_payload.get("accepted")
        expected_acceptance = self._upstream_acceptance(order_id, quote["requirement_digest"], pricing)
        if accepted != expected_acceptance:
            raise MarketplaceError("AMOUNT_MISMATCH", "Selected x402 acceptance does not match the challenge.")

        proof_digest = digest_object(request)
        payment_payload_digest = digest_object(payment_payload)
        payment_mandate_digest = digest_object(approval.payment_mandate)
        operation_id = f"rail-charge:{charge['charge_id']}"
        proof_evidence = self.store.put_evidence(
            intent_id=f"proof:{charge['charge_id']}",
            evidence_id=f"evidence-proof:{charge['charge_id']}",
            tenant_type="customer",
            tenant_id=self.settings.customer_id,
            kind="ap2-x402-proof",
            exact_bytes=canonical_bytes(request),
            kid=authorization.kid,
        )
        if proof_evidence["state"] != "committed":
            raise MarketplaceError("LEDGER_POST_FAILED", "Payment evidence is not durable.", retryable=True)
        self._persist_verified_payment(
            order=order,
            charge=charge,
            issuer=authorization.subject,
            nonce=authorization.nonce,
            proof_digest=proof_digest,
            operation_id=operation_id,
            idempotency_key=idempotency_key,
        )
        rail_result = self.rail.settle_charge(
            operation_id=operation_id,
            source_id=charge["charge_id"],
            amount=int(pricing["customer_total"]),
            idempotency_key=f"charge:{idempotency_key}",
        )
        if rail_result["state"] == "unknown":
            self._update_charge(charge["charge_id"], "unknown")
            self._transition_order(
                order_id,
                "charge_processing",
                "reconciliation_required",
                "charge-result-unknown",
                recovery_kind="charge",
                operation_id=operation_id,
            )
            raise MarketplaceError("SETTLEMENT_UNKNOWN", "Charge result is unknown.", retryable=True)
        if rail_result["state"] != "settled":
            self._update_charge(charge["charge_id"], "failed")
            self._transition_order(order_id, "charge_processing", "failed", "charge-failed")
            code = rail_result.get("error_code") or "INTERNAL_ERROR"
            raise MarketplaceError(code, "Simulated charge failed.")
        self._update_charge(charge["charge_id"], "settled")
        response = self._post_charge_and_fulfill(
            order=self._owned_order(order_id, self.settings.customer_id),
            charge=charge,
            quote=quote,
            pricing=pricing,
            rail_result=rail_result,
            proof_digest=proof_digest,
            payment_payload_digest=payment_payload_digest,
            payment_mandate_digest=payment_mandate_digest,
            idempotency_key=idempotency_key,
            merchant_fault=merchant_fault,
        )
        self.store.complete_idempotency(
            "payment-submit", self.settings.customer_id, idempotency_key, request_hash, response
        )
        return response

    def _persist_verified_payment(
        self,
        *,
        order: dict[str, Any],
        charge: dict[str, Any],
        issuer: str,
        nonce: str,
        proof_digest: str,
        operation_id: str,
        idempotency_key: str,
    ) -> None:
        """Atomically consume proof nonce and make the operation resumable.

        There must never be a committed used nonce while the order still says
        payment_required.  The hook is deliberately a no-op and exists only so
        crash tests can raise inside this transaction.
        """

        now = utc_now()
        with self.store.business_transaction() as conn:
            current_order = conn.execute(
                "SELECT * FROM orders WHERE order_id=?", (order["order_id"],)
            ).fetchone()
            current_charge = conn.execute(
                "SELECT * FROM charges WHERE charge_id=?", (charge["charge_id"],)
            ).fetchone()
            if current_order is None or current_charge is None:
                raise MarketplaceError("INTERNAL_ERROR", "Payment aggregate is missing.")
            existing_nonce = conn.execute(
                "SELECT * FROM used_nonces WHERE issuer=? AND nonce=?", (issuer, nonce)
            ).fetchone()
            if existing_nonce is not None:
                same_use = (
                    existing_nonce["digest"] == proof_digest
                    and existing_nonce["order_id"] == order["order_id"]
                    and existing_nonce["task_id"] == order["task_id"]
                    and existing_nonce["operation"] == "upstream-charge"
                )
                if not same_use:
                    raise MarketplaceError("REPLAY_DETECTED", "Payment nonce was already used.")
            elif current_order["state"] == "payment_required":
                try:
                    conn.execute(
                        """INSERT INTO used_nonces
                           (issuer, nonce, digest, order_id, task_id, operation, consumed_at)
                           VALUES (?, ?, ?, ?, ?, 'upstream-charge', ?)""",
                        (issuer, nonce, proof_digest, order["order_id"], order["task_id"], now),
                    )
                except sqlite3.IntegrityError as exc:
                    raise MarketplaceError("REPLAY_DETECTED", "Payment nonce was already used.") from exc
            else:
                raise MarketplaceError(
                    "INVALID_STATE_TRANSITION", "Payment verification is not resumable from this state."
                )

            self._after_atomic_recovery_step("nonce-consumed")
            if current_order["state"] == "payment_required":
                next_version = int(current_order["version"]) + 1
                changed = conn.execute(
                    """UPDATE orders SET state='charge_processing', version=?, recovery_kind='charge',
                       authoritative_operation_id=?, updated_at=?
                       WHERE order_id=? AND state='payment_required' AND version=?""",
                    (
                        next_version,
                        operation_id,
                        now,
                        order["order_id"],
                        current_order["version"],
                    ),
                ).rowcount
                if changed != 1:
                    raise MarketplaceError("INVALID_STATE_TRANSITION", "Payment order changed concurrently.")
                conn.execute(
                    """INSERT INTO state_events
                       (aggregate_type, aggregate_id, from_state, to_state, actor_id,
                        reason, sequence, created_at)
                       VALUES ('order', ?, 'payment_required', 'charge_processing', ?,
                               'proof-verified', ?, ?)""",
                    (order["order_id"], self.settings.marketplace_id, next_version, now),
                )
            conn.execute(
                """UPDATE charges SET state='verified', proof_digest=?, operation_id=?,
                   idempotency_key=?, version=version+1, updated_at=? WHERE charge_id=?""",
                (proof_digest, operation_id, idempotency_key, now, charge["charge_id"]),
            )

    def _after_atomic_recovery_step(self, step: str) -> None:
        """No-op injection point used by crash/rollback tests only."""

        return None

    def _post_charge_and_fulfill(
        self,
        *,
        order: dict[str, Any],
        charge: dict[str, Any],
        quote: dict[str, Any],
        pricing: dict[str, Any],
        rail_result: dict[str, Any],
        proof_digest: str,
        payment_payload_digest: str,
        payment_mandate_digest: str,
        idempotency_key: str,
        merchant_fault: str | None,
        from_order_state: str = "charge_processing",
    ) -> dict[str, Any]:
        order_id = order["order_id"]
        journal_id = f"journal-charge:{charge['charge_id']}"
        payable_id = f"payable:{order_id}"
        try:
            self.ledger.post_charge(
                journal_id=journal_id,
                charge_id=charge["charge_id"],
                order_id=order_id,
                payable_id=payable_id,
                merchant_id=self.settings.merchant_id,
                amount=int(pricing["merchant_payable_amount"]),
                idempotency_key=f"ledger:{charge['charge_id']}",
            )
        except Exception as exc:
            current = self._one("SELECT * FROM orders WHERE order_id=?", (order_id,))
            if current["state"] == from_order_state:
                self._transition_order(
                    order_id,
                    from_order_state,
                    "reconciliation_required",
                    "charge-settled-ledger-unposted",
                    recovery_kind="charge-ledger",
                    operation_id=rail_result["operation_id"],
                )
            raise MarketplaceError(
                "LEDGER_POST_FAILED",
                "Charge settled but its balanced journal requires reconciliation.",
                retryable=True,
            ) from exc
        x402_receipt_id = f"receipt-x402:{charge['charge_id']}"
        ap2_receipt_id = f"receipt-ap2:{charge['charge_id']}"
        settlement_reference = rail_result["operation_id"]
        rail_receipt_digest = digest_object(rail_result["receipt"])
        settlement_time = rail_result["receipt"]["issuedAt"]
        x402_receipt = with_signature(
            {
                "receiptType": "x402-settlement",
                "receiptId": x402_receipt_id,
                "profile": PROFILE_URI,
                "simulated": True,
                "status": "settled",
                "issuedAt": settlement_time,
                "issuer": self.settings.marketplace_id,
                "subject": self.settings.customer_id,
                "orderId": order_id,
                "settlementReference": settlement_reference,
                "relatedReceiptId": ap2_receipt_id,
                "relatedDigests": {"paymentPayload": payment_payload_digest, "railReceipt": rail_receipt_digest},
            },
            kid=MEDIATOR_KID,
        )
        ap2_receipt = with_signature(
            {
                "receiptType": "ap2-payment",
                "receiptId": ap2_receipt_id,
                "profile": PROFILE_URI,
                "simulated": True,
                "status": "settled",
                "issuedAt": settlement_time,
                "issuer": self.settings.marketplace_id,
                "subject": self.settings.customer_id,
                "orderId": order_id,
                "settlementReference": settlement_reference,
                "relatedReceiptId": x402_receipt_id,
                "relatedDigests": {
                    "paymentMandate": payment_mandate_digest,
                    "authorization": proof_digest,
                },
            },
            kid=MEDIATOR_KID,
        )
        for receipt in (x402_receipt, ap2_receipt):
            self.store.put_evidence(
                intent_id=f"receipt:{receipt['receiptId']}",
                evidence_id=f"evidence:{receipt['receiptId']}",
                tenant_type="customer",
                tenant_id=self.settings.customer_id,
                kind=receipt["receiptType"],
                exact_bytes=canonical_bytes(receipt),
                kid=MEDIATOR_KID,
            )
        with self.store.business_transaction() as conn:
            conn.execute(
                """UPDATE charges SET journal_id=?, settlement_receipt_id=?, ap2_receipt_id=?,
                   updated_at=? WHERE charge_id=?""",
                (journal_id, x402_receipt_id, ap2_receipt_id, utc_now(), charge["charge_id"]),
            )
        current_order = self._one("SELECT * FROM orders WHERE order_id=?", (order_id,))
        if current_order["state"] == from_order_state and from_order_state != "payable_posted":
            self._transition_order(
                order_id, from_order_state, "payable_posted", "balanced-payable-posted"
            )
        elif current_order["state"] != "payable_posted":
            raise MarketplaceError(
                "INVALID_STATE_TRANSITION", "Posted payable cannot be resumed from this order state."
            )

        try:
            self._require_active_merchant(self.settings.merchant_id)
        except MarketplaceError:
            self._mark_refund_required(order_id, payable_id, "merchant-onboarding-gate-failed")
            raise

        requirement = self._quote_requirement(quote["evidence_id"])
        accepted = requirement["accepts"][0]
        guarantee_id = f"guarantee:{order_id}"
        evidence_id = f"evidence:{guarantee_id}"
        guarantee_payload, guarantee_digest = self._load_or_create_guarantee(
            evidence_id=evidence_id,
            guarantee_id=guarantee_id,
            accepted=accepted,
            order=order,
            quote=quote,
            pricing=pricing,
            journal_id=journal_id,
            x402_receipt=x402_receipt,
            ap2_receipt=ap2_receipt,
        )
        self._after_atomic_recovery_step("guarantee-evidence-committed")
        guarantee_claims = guarantee_payload["payload"]
        issued = int(guarantee_claims["iat"])
        expires = int(guarantee_claims["exp"])
        now = utc_now()
        with self.store.business_transaction() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO guarantees
                   (guarantee_id, order_id, payable_id, state, evidence_id, digest, iat,
                    exp, schema_version, created_at, updated_at)
                   VALUES (?, ?, ?, 'issued', ?, ?, ?, ?, ?, ?, ?)""",
                (
                    guarantee_id,
                    order_id,
                    payable_id,
                    evidence_id,
                    guarantee_digest,
                    str(issued),
                    str(expires),
                    self.settings.schema_version,
                    now,
                    now,
                ),
            )
            conn.execute(
                "UPDATE payables SET state='guaranteed', guarantee_id=?, version=version+1, updated_at=? WHERE payable_id=? AND state='open'",
                (guarantee_id, now, payable_id),
            )
            persisted = conn.execute(
                "SELECT evidence_id, digest FROM guarantees WHERE guarantee_id=?", (guarantee_id,)
            ).fetchone()
            if (
                persisted is None
                or persisted["evidence_id"] != evidence_id
                or persisted["digest"] != guarantee_digest
            ):
                raise MarketplaceError("GUARANTEE_INVALID", "Persisted guarantee binding conflicts.")
        current_order = self._one("SELECT * FROM orders WHERE order_id=?", (order_id,))
        if current_order["state"] == "payable_posted":
            self._transition_order(order_id, "payable_posted", "guarantee_issued", "guarantee-signed")
        elif current_order["state"] not in {"guarantee_issued", "reconciliation_required", "fulfilling"}:
            raise MarketplaceError(
                "INVALID_STATE_TRANSITION", "Guarantee cannot be delivered from this order state."
            )

        fulfillment_request = {
            "paymentPayload": guarantee_payload,
            "correlationId": order["correlation_id"],
        }
        if merchant_fault:
            fulfillment_request["_testFault"] = merchant_fault
        try:
            fulfillment = self.merchant.fulfill(fulfillment_request)
        except MerchantTimeout as exc:
            with self.store.business_transaction() as conn:
                conn.execute(
                    "UPDATE guarantees SET state='delivery_unknown', version=version+1, updated_at=? WHERE guarantee_id=?",
                    (utc_now(), guarantee_id),
                )
            self._transition_order(
                order_id,
                "guarantee_issued",
                "reconciliation_required",
                "merchant-result-unknown",
                recovery_kind="fulfillment",
                operation_id=guarantee_id,
            )
            raise MarketplaceError("SETTLEMENT_UNKNOWN", str(exc), retryable=True) from exc
        except MerchantClientError as exc:
            self._mark_refund_required(order_id, payable_id, "guarantee-delivery-failed")
            raise MarketplaceError("GUARANTEE_INVALID", "Merchant rejected the guarantee.") from exc

        receipt = fulfillment.get("receipt")
        try:
            if not isinstance(receipt, dict):
                raise ValueError("receipt missing")
            verify_payload_signature(receipt, expected_kid=MERCHANT_KID)
            fulfillment_id = str(fulfillment["fulfillmentId"])
            fulfillment_state = str(fulfillment["state"])
            if (
                receipt.get("receiptType") != "merchant-order"
                or receipt.get("issuer") != self.settings.merchant_id
                or receipt.get("subject") != self.settings.marketplace_id
                or receipt.get("orderId") != order_id
                or receipt.get("quoteId") != order["quote_id"]
                or receipt.get("guaranteeId") != guarantee_id
                or receipt.get("fulfillmentId") != fulfillment_id
                or receipt.get("status") != fulfillment_state
                or receipt.get("relatedDigests", {}).get("guarantee") != guarantee_digest
            ):
                raise ValueError("receipt binding mismatch")
        except Exception as exc:
            self._mark_refund_required(order_id, payable_id, "merchant-receipt-invalid")
            raise MarketplaceError("INVALID_SIGNATURE", "Merchant order receipt is invalid.") from exc
        receipt_digest = digest_object(receipt)
        self.store.put_evidence(
            intent_id=f"receipt:{receipt['receiptId']}",
            evidence_id=f"evidence:{receipt['receiptId']}",
            tenant_type="merchant",
            tenant_id=self.settings.merchant_id,
            kind="merchant-order",
            exact_bytes=canonical_bytes(receipt),
            kid=MERCHANT_KID,
        )
        with self.store.business_transaction() as conn:
            conn.execute(
                """INSERT INTO fulfillments
                   (fulfillment_id, order_id, guarantee_id, merchant_id, state, receipt_id,
                    receipt_digest, attempt, schema_version, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (
                    fulfillment_id,
                    order_id,
                    guarantee_id,
                    self.settings.merchant_id,
                    fulfillment_state,
                    receipt["receiptId"],
                    receipt_digest,
                    self.settings.schema_version,
                    utc_now(),
                    utc_now(),
                ),
            )
            conn.execute(
                "UPDATE guarantees SET state='accepted', version=version+1, updated_at=? WHERE guarantee_id=?",
                (utc_now(), guarantee_id),
            )
        if fulfillment_state != "fulfilled":
            self._mark_refund_required(order_id, payable_id, "merchant-fulfillment-failed")
            return self.order_status(order_id, customer_id=self.settings.customer_id)
        with self.store.business_transaction() as conn:
            conn.execute(
                "UPDATE payables SET state='eligible', available_at=?, version=version+1, updated_at=? WHERE payable_id=? AND state='guaranteed'",
                (utc_now(), utc_now(), payable_id),
            )
        self._transition_order(order_id, "guarantee_issued", "fulfilling", "guarantee-accepted")
        self._after_atomic_recovery_step("fulfilling-persisted")
        self._transition_order(order_id, "fulfilling", "completed", "merchant-receipt-verified")
        response = {
            "orderId": order_id,
            "taskId": order["task_id"],
            "contextId": order["context_id"],
            "correlationId": order["correlation_id"],
            "state": "completed",
            "profile": PROFILE_URI,
            "simulated": True,
            "pricing": self._pricing_public(pricing),
            "payable": {"payableId": payable_id, "amount": pricing["merchant_payable_amount"], "state": "eligible"},
            "guarantee": {"guaranteeId": guarantee_id, "digest": guarantee_digest, "state": "accepted"},
            "fulfillment": {"fulfillmentId": fulfillment_id, "state": "fulfilled", "receiptDigest": receipt_digest},
            "receipts": [x402_receipt, ap2_receipt, receipt],
        }
        self.store.save_task(
            order["task_id"],
            order["context_id"],
            "completed",
            actor_id=self.settings.customer_id,
            tenant_id=self.settings.customer_id,
            metadata={"orderId": order_id, "paymentStatus": "payment-completed"},
            response=response,
        )
        return response

    def order_status(self, order_id: str, *, customer_id: str) -> dict[str, Any]:
        order = self._owned_order(order_id, customer_id)
        pricing = self._maybe_one("SELECT * FROM pricing WHERE order_id=?", (order_id,))
        charge = self._maybe_one("SELECT * FROM charges WHERE order_id=?", (order_id,))
        payable = self._maybe_one("SELECT * FROM payables WHERE order_id=?", (order_id,))
        guarantee = self._maybe_one("SELECT * FROM guarantees WHERE order_id=?", (order_id,))
        fulfillment = self._maybe_one("SELECT * FROM fulfillments WHERE order_id=?", (order_id,))
        refunds = self.store.fetch_business("SELECT * FROM refunds WHERE order_id=? ORDER BY created_at", (order_id,))
        return {
            "orderId": order_id,
            "taskId": order["task_id"],
            "contextId": order["context_id"],
            "state": order["state"],
            "merchantId": order["merchant_id"],
            "quoteId": order["quote_id"],
            "correlationId": order["correlation_id"],
            "profile": PROFILE_URI,
            "simulated": True,
            "pricing": self._pricing_public(pricing) if pricing else None,
            "charge": self._public_row(charge, {"proof_digest", "idempotency_key"}),
            "payable": self._public_row(payable, {"journal_id"}),
            "guarantee": self._public_row(guarantee, {"evidence_id"}),
            "fulfillment": self._public_row(fulfillment, set()),
            "refunds": [self._public_row(row, {"idempotency_key", "request_hash"}) for row in refunds],
        }

    def reconcile_order(self, order_id: str, *, actor_id: str, reason: str) -> dict[str, Any]:
        """Resolve an ambiguous merchant delivery by querying authoritative status.

        A reconciliation never creates another customer charge or another guarantee.
        The merchant is queried using the identifiers already persisted before the
        original delivery attempt, and only its signed receipt can advance state.
        """

        order = self._one("SELECT * FROM orders WHERE order_id=?", (order_id,))
        if order["state"] in {"charge_processing", "payable_posted"}:
            return self._reconcile_charge_ledger(order, actor_id=actor_id)
        if order["state"] == "refund_required":
            refund = self._maybe_one(
                "SELECT * FROM refunds WHERE order_id=? ORDER BY created_at DESC LIMIT 1",
                (order_id,),
            )
            if refund is not None and refund["state"] not in {"settled", "failed"}:
                self._transition_order(
                    order_id,
                    "refund_required",
                    "refunding",
                    reason,
                    actor_id=actor_id,
                )
                return self._reconcile_refund(refund, actor_id=actor_id)
        if order["state"] == "refunding":
            refund = self._one(
                "SELECT * FROM refunds WHERE order_id=? ORDER BY created_at DESC LIMIT 1",
                (order_id,),
            )
            return self._reconcile_refund(refund, actor_id=actor_id)
        if order["state"] not in {
            "reconciliation_required",
            "guarantee_issued",
            "fulfilling",
        }:
            raise MarketplaceError(
                "INVALID_STATE_TRANSITION", "Order does not require reconciliation."
            )
        if order["recovery_kind"] in {"charge", "charge-ledger"}:
            return self._reconcile_charge_ledger(order, actor_id=actor_id)
        if order["recovery_kind"] == "refund":
            refund = self._one(
                "SELECT * FROM refunds WHERE order_id=? ORDER BY created_at DESC LIMIT 1",
                (order_id,),
            )
            return self._reconcile_refund(refund, actor_id=actor_id)
        if (
            order["state"] not in {"guarantee_issued", "fulfilling"}
            and order["recovery_kind"] != "fulfillment"
        ):
            raise MarketplaceError(
                "INVALID_STATE_TRANSITION", "Order recovery kind is unsupported."
            )
        guarantee = self._one("SELECT * FROM guarantees WHERE order_id=?", (order_id,))
        payable = self._one("SELECT * FROM payables WHERE order_id=?", (order_id,))
        try:
            fulfillment = self.merchant.fulfillment_status(order_id, guarantee["guarantee_id"])
        except MerchantTimeout as exc:
            raise MarketplaceError(
                "SETTLEMENT_UNKNOWN",
                "Merchant fulfillment status is unavailable; do not redeliver blindly.",
                retryable=True,
            ) from exc
        except MerchantClientError as exc:
            # A conclusive missing status means the original delivery had no
            # durable merchant-side effect.  Redeliver the immutable evidence,
            # not a re-signed or reconstructed guarantee.
            status_error = str(exc).strip().upper().replace(" ", "_")
            if status_error not in {"INVALID_SCHEMA", "FULFILLMENT_NOT_FOUND", "NOT_FOUND"}:
                raise MarketplaceError(
                    "SETTLEMENT_UNKNOWN",
                    "Merchant fulfillment remains unknown; retry status reconciliation.",
                    retryable=True,
                ) from exc
            fulfillment = self._redeliver_persisted_guarantee(order, guarantee)

        receipt = fulfillment.get("receipt")
        try:
            if not isinstance(receipt, dict):
                raise ValueError("receipt missing")
            verify_payload_signature(receipt, expected_kid=MERCHANT_KID)
            fulfillment_id = str(fulfillment["fulfillmentId"])
            state = str(fulfillment["state"])
            if (
                fulfillment.get("orderId") != order_id
                or fulfillment.get("guaranteeId") != guarantee["guarantee_id"]
                or fulfillment.get("guaranteeDigest") != guarantee["digest"]
                or receipt.get("receiptType") != "merchant-order"
                or receipt.get("issuer") != self.settings.merchant_id
                or receipt.get("subject") != self.settings.marketplace_id
                or receipt.get("orderId") != order_id
                or receipt.get("quoteId") != order["quote_id"]
                or receipt.get("guaranteeId") != guarantee["guarantee_id"]
                or receipt.get("fulfillmentId") != fulfillment_id
                or receipt.get("status") != state
                or receipt.get("relatedDigests", {}).get("guarantee") != guarantee["digest"]
            ):
                raise ValueError("fulfillment status binding mismatch")
        except Exception as exc:
            raise MarketplaceError(
                "INVALID_SIGNATURE", "Merchant reconciliation receipt is invalid."
            ) from exc

        receipt_digest = digest_object(receipt)
        evidence = self.store.put_evidence(
            intent_id=f"receipt:{receipt['receiptId']}",
            evidence_id=f"evidence:{receipt['receiptId']}",
            tenant_type="merchant",
            tenant_id=self.settings.merchant_id,
            kind="merchant-order",
            exact_bytes=canonical_bytes(receipt),
            kid=MERCHANT_KID,
        )
        if evidence["state"] != "committed":
            raise MarketplaceError(
                "LEDGER_POST_FAILED", "Reconciliation evidence is not durable.", retryable=True
            )
        now = utc_now()
        with self.store.business_transaction() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO fulfillments
                   (fulfillment_id, order_id, guarantee_id, merchant_id, state, receipt_id,
                    receipt_digest, attempt, schema_version, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (
                    fulfillment_id,
                    order_id,
                    guarantee["guarantee_id"],
                    self.settings.merchant_id,
                    state,
                    receipt["receiptId"],
                    receipt_digest,
                    self.settings.schema_version,
                    now,
                    now,
                ),
            )
            conn.execute(
                "UPDATE guarantees SET state='accepted', version=version+1, updated_at=? WHERE guarantee_id=?",
                (now, guarantee["guarantee_id"]),
            )

        if state != "fulfilled":
            self._mark_refund_required(order_id, payable["payable_id"], "reconciled-fulfillment-failed")
            return self.order_status(order_id, customer_id=self.settings.customer_id)

        with self.store.business_transaction() as conn:
            conn.execute(
                """UPDATE payables SET state='eligible', available_at=?, version=version+1,
                   updated_at=? WHERE payable_id=? AND state='guaranteed'""",
                (now, now, payable["payable_id"]),
            )
        current_order = self._one("SELECT * FROM orders WHERE order_id=?", (order_id,))
        if current_order["state"] != "fulfilling":
            self._transition_order(
                order_id,
                current_order["state"],
                "fulfilling",
                reason,
                actor_id=actor_id,
            )
        self._transition_order(
            order_id,
            "fulfilling",
            "completed",
            "merchant-status-reconciled",
            actor_id=actor_id,
        )
        response = self.order_status(order_id, customer_id=self.settings.customer_id)
        self.store.save_task(
            order["task_id"],
            order["context_id"],
            "completed",
            actor_id=self.settings.customer_id,
            tenant_id=self.settings.customer_id,
            metadata={"orderId": order_id, "paymentStatus": "payment-completed"},
            response=response,
        )
        charge = self._one("SELECT * FROM charges WHERE order_id=?", (order_id,))
        self._complete_payment_idempotency(charge, response)
        return response

    def _redeliver_persisted_guarantee(
        self, order: dict[str, Any], guarantee: dict[str, Any]
    ) -> dict[str, Any]:
        raw = self.store.read_evidence(
            guarantee["evidence_id"],
            actor_id=self.settings.marketplace_id,
            actor_role="operator",
        )
        try:
            payload = json.loads(raw)
            if canonical_bytes(payload) != raw or digest_object(payload) != guarantee["digest"]:
                raise ValueError("immutable guarantee mismatch")
            verify_payload_signature(payload["payload"], expected_kid=MEDIATOR_KID)
        except Exception as exc:
            raise MarketplaceError(
                "GUARANTEE_INVALID", "Persisted guarantee evidence is invalid."
            ) from exc
        try:
            result = self.merchant.fulfill(
                {"paymentPayload": payload, "correlationId": order["correlation_id"]}
            )
        except MerchantClientError as exc:
            raise MarketplaceError(
                "SETTLEMENT_UNKNOWN",
                "Guarantee redelivery remains unknown; retry status reconciliation.",
                retryable=True,
            ) from exc
        # Normalize the fulfill response to the authoritative status shape used
        # by the common receipt validator below.
        normalized = dict(result)
        normalized.setdefault("orderId", order["order_id"])
        normalized.setdefault("guaranteeId", guarantee["guarantee_id"])
        normalized.setdefault("guaranteeDigest", guarantee["digest"])
        return normalized

    def _reconcile_charge_ledger(
        self, order: dict[str, Any], *, actor_id: str
    ) -> dict[str, Any]:
        order_id = order["order_id"]
        charge = self._one("SELECT * FROM charges WHERE order_id=?", (order_id,))
        quote = self._one("SELECT * FROM merchant_quotes WHERE order_id=?", (order_id,))
        pricing = self._one("SELECT * FROM pricing WHERE order_id=?", (order_id,))
        operation_id = str(order["authoritative_operation_id"] or charge["operation_id"] or "")
        rail_result = self.rail.get_operation(operation_id)
        if rail_result is None:
            # The operation identifiers were committed before the rail call.  A
            # crash in that gap is resumed with exactly those identifiers; a new
            # operation or idempotency key is never allocated.
            rail_result = self.rail.settle_charge(
                operation_id=operation_id,
                source_id=charge["charge_id"],
                amount=int(pricing["customer_total"]),
                idempotency_key=f"charge:{charge['idempotency_key']}",
            )
        if rail_result["state"] == "unknown":
            raise MarketplaceError(
                "SETTLEMENT_UNKNOWN", "Charge result is still unknown.", retryable=True
            )
        if rail_result["state"] != "settled":
            self._update_charge(charge["charge_id"], "failed")
            self._transition_order(
                order_id,
                order["state"],
                "failed",
                "charge-reconciled-not-settled",
                actor_id=actor_id,
            )
            raise MarketplaceError("INTERNAL_ERROR", "Charge did not settle.")
        self._update_charge(charge["charge_id"], "settled")
        proof_bytes = self.store.read_evidence(
            f"evidence-proof:{charge['charge_id']}",
            actor_id=actor_id,
            actor_role="operator",
        )
        proof = json.loads(proof_bytes)
        approval = TrustedSurfaceApproval.model_validate(proof["approval"])
        response = self._post_charge_and_fulfill(
            order=order,
            charge=charge,
            quote=quote,
            pricing=pricing,
            rail_result=rail_result,
            proof_digest=str(charge["proof_digest"]),
            payment_payload_digest=digest_object(proof["paymentPayload"]),
            payment_mandate_digest=digest_object(approval.payment_mandate),
            idempotency_key=str(charge["idempotency_key"]),
            merchant_fault=None,
            from_order_state=order["state"],
        )
        self._complete_payment_idempotency(charge, response)
        return response

    # -- refund --------------------------------------------------------------

    def refund_order(
        self, order_id: str, *, idempotency_key: str, actor_id: str, reason: str
    ) -> dict[str, Any]:
        order = self._one("SELECT * FROM orders WHERE order_id=?", (order_id,))
        payable = self._one("SELECT * FROM payables WHERE order_id=?", (order_id,))
        charge = self._one("SELECT * FROM charges WHERE order_id=?", (order_id,))
        request = {"orderId": order_id, "reason": reason, "amount": payable["amount"]}
        request_hash = digest_object(request)
        idem = self.store.begin_idempotency("refund", actor_id, idempotency_key, request_hash)
        if idem["status"] == "hit" and idem["response"] is not None:
            return idem["response"]
        existing = self._maybe_one(
            "SELECT * FROM refunds WHERE order_id=? AND idempotency_key=?",
            (order_id, idempotency_key),
        )
        if idem["status"] == "hit":
            if existing is None:
                raise MarketplaceError(
                    "REFUND_UNKNOWN", "Refund creation is incomplete.", retryable=True
                )
            if order["state"] == "refund_required":
                self._transition_order(
                    order_id, "refund_required", "refunding", reason, actor_id=actor_id
                )
            return self._reconcile_refund(existing, actor_id=actor_id)
        if order["state"] != "refund_required":
            raise MarketplaceError("INVALID_STATE_TRANSITION", "Order is not refund-required.")
        if payable["state"] == "paid":
            raise MarketplaceError("INVALID_STATE_TRANSITION", "Post-payout refunds are out of MVP scope.")
        refund_id = self.id_factory("refund")
        operation_id = f"rail-refund:{refund_id}"
        now = utc_now()
        with self.store.business_transaction() as conn:
            conn.execute(
                """INSERT INTO refunds
                   (refund_id, order_id, charge_id, payable_id, responsibility, reason,
                    amount, asset, state, rail_state, ledger_state, operation_id,
                    idempotency_key, request_hash, schema_version, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'merchant', ?, ?, 'USD', 'settling', 'settling',
                           'pending', ?, ?, ?, ?, ?, ?)""",
                (
                    refund_id,
                    order_id,
                    charge["charge_id"],
                    payable["payable_id"],
                    reason,
                    payable["amount"],
                    operation_id,
                    idempotency_key,
                    request_hash,
                    self.settings.schema_version,
                    now,
                    now,
                ),
            )
        self._transition_order(order_id, "refund_required", "refunding", reason, actor_id=actor_id)
        refund = self._one("SELECT * FROM refunds WHERE refund_id=?", (refund_id,))
        return self._reconcile_refund(refund, actor_id=actor_id)

    def _reconcile_refund(
        self, refund: dict[str, Any], *, actor_id: str
    ) -> dict[str, Any]:
        rail_result = self.rail.get_operation(str(refund["operation_id"]))
        if rail_result is None:
            rail_result = self.rail.refund(
                operation_id=str(refund["operation_id"]),
                source_id=refund["refund_id"],
                amount=int(refund["amount"]),
                idempotency_key=f"refund:{refund['idempotency_key']}",
            )
        if rail_result["state"] == "unknown":
            with self.store.business_transaction() as conn:
                conn.execute(
                    "UPDATE refunds SET state='unknown', rail_state='unknown', updated_at=? WHERE refund_id=?",
                    (utc_now(), refund["refund_id"]),
                )
            order = self._one("SELECT * FROM orders WHERE order_id=?", (refund["order_id"],))
            if order["state"] == "refunding":
                self._transition_order(
                    refund["order_id"],
                    "refunding",
                    "reconciliation_required",
                    "refund-result-unknown",
                    actor_id=actor_id,
                    recovery_kind="refund",
                    operation_id=refund["operation_id"],
                )
            raise MarketplaceError("REFUND_UNKNOWN", "Refund result is still unknown.", retryable=True)
        if rail_result["state"] != "settled":
            with self.store.business_transaction() as conn:
                conn.execute(
                    "UPDATE refunds SET state='failed', rail_state='failed', updated_at=? WHERE refund_id=?",
                    (utc_now(), refund["refund_id"]),
                )
            order = self._one("SELECT * FROM orders WHERE order_id=?", (refund["order_id"],))
            if order["state"] in {"refunding", "reconciliation_required"}:
                self._transition_order(
                    refund["order_id"],
                    order["state"],
                    "refund_required",
                    "refund-not-settled",
                    actor_id=actor_id,
                )
            raise MarketplaceError("INTERNAL_ERROR", "Refund did not settle.")
        order = self._one("SELECT state FROM orders WHERE order_id=?", (refund["order_id"],))
        return self._complete_refund(
            refund,
            rail_result,
            actor_id=actor_id,
            from_order_state=order["state"],
        )

    def _complete_refund(
        self,
        refund: dict[str, Any],
        rail_result: dict[str, Any],
        *,
        actor_id: str,
        from_order_state: str,
    ) -> dict[str, Any]:
        refund_id = refund["refund_id"]
        order_id = refund["order_id"]
        payable = self._one("SELECT * FROM payables WHERE payable_id=?", (refund["payable_id"],))
        journal_id = f"journal-refund:{refund_id}"
        try:
            self.ledger.post_refund(
                journal_id=journal_id,
                refund_id=refund_id,
                merchant_id=self.settings.merchant_id,
                amount=int(refund["amount"]),
                payable_id=payable["payable_id"],
                idempotency_key=f"ledger:{refund_id}",
            )
        except Exception as exc:
            with self.store.business_transaction() as conn:
                conn.execute(
                    """UPDATE refunds SET state='reconciliation_required', rail_state='settled',
                       ledger_state='failed', updated_at=? WHERE refund_id=?""",
                    (utc_now(), refund_id),
                )
            if from_order_state != "reconciliation_required":
                self._transition_order(
                    order_id,
                    from_order_state,
                    "reconciliation_required",
                    "refund-settled-ledger-unposted",
                    actor_id=actor_id,
                    recovery_kind="refund",
                    operation_id=refund["operation_id"],
                )
            raise MarketplaceError(
                "LEDGER_POST_FAILED", "Refund journal requires reconciliation.", retryable=True
            ) from exc
        receipt = with_signature(
            {
                "receiptType": "refund",
                "receiptId": f"receipt:{refund_id}",
                "profile": PROFILE_URI,
                "simulated": True,
                "status": "settled",
                "issuedAt": rail_result["receipt"]["issuedAt"],
                "issuer": self.settings.marketplace_id,
                "subject": self.settings.customer_id,
                "orderId": order_id,
                "references": {
                    "chargeId": refund["charge_id"],
                    "payableId": refund["payable_id"],
                    "journalId": journal_id,
                },
                "relatedDigests": {"railReceipt": digest_object(rail_result["receipt"]), "journal": digest_object(self.ledger.get_journal(journal_id))},
            },
            kid=MEDIATOR_KID,
        )
        self.store.put_evidence(
            intent_id=f"receipt:{refund_id}",
            evidence_id=f"evidence:receipt:{refund_id}",
            tenant_type="customer",
            tenant_id=self.settings.customer_id,
            kind="refund",
            exact_bytes=canonical_bytes(receipt),
            kid=MEDIATOR_KID,
        )
        with self.store.business_transaction() as conn:
            conn.execute(
                """UPDATE refunds SET state='settled', rail_state='settled', ledger_state='settled',
                   journal_id=?, receipt_id=?, updated_at=? WHERE refund_id=?""",
                (journal_id, receipt["receiptId"], utc_now(), refund_id),
            )
        self._transition_order(
            order_id, from_order_state, "refunded", "refund-settled", actor_id=actor_id
        )
        response = {"refundId": refund_id, "orderId": order_id, "state": "settled", "receipt": receipt, "simulated": True}
        self.store.complete_idempotency(
            "refund", actor_id, refund["idempotency_key"], refund["request_hash"], response
        )
        return response

    # -- payout --------------------------------------------------------------

    def create_payout(
        self, *, merchant_id: str, idempotency_key: str, actor_id: str, reason: str
    ) -> dict[str, Any]:
        self._require_active_merchant(merchant_id)
        existing = self._maybe_one(
            "SELECT * FROM payouts WHERE merchant_id=? AND idempotency_key=?",
            (merchant_id, idempotency_key),
        )
        if existing is not None:
            snapshot = json.loads(existing["eligibility_json"])
            retry_body = {
                "merchantId": merchant_id,
                "payableIds": snapshot["payableIds"],
                "reason": reason,
            }
            idem = self.store.begin_idempotency(
                "payout", actor_id, idempotency_key, digest_object(retry_body)
            )
            if idem["status"] == "hit" and idem["response"] is not None:
                return idem["response"]
            return self.reconcile_payout(
                existing["payout_id"], actor_id=actor_id, reason=reason
            )
        payables = self.store.fetch_business(
            "SELECT * FROM payables WHERE merchant_id=? AND state='eligible' ORDER BY created_at",
            (merchant_id,),
        )
        if not payables:
            raise MarketplaceError("PAYOUT_NOT_ELIGIBLE", "No eligible payable exists.")
        body = {"merchantId": merchant_id, "payableIds": [item["payable_id"] for item in payables], "reason": reason}
        request_hash = digest_object(body)
        idem = self.store.begin_idempotency("payout", actor_id, idempotency_key, request_hash)
        cached = self._cached_idempotent_response(idem)
        if cached is not None:
            return cached
        payout_id = self.id_factory("payout")
        gross = sum(int(item["amount"]) for item in payables)
        operation_id = f"rail-payout:{payout_id}"
        now = utc_now()
        with self.store.business_transaction() as conn:
            conn.execute(
                """INSERT INTO payouts
                   (payout_id, merchant_id, state, gross_amount, commission_amount,
                    rail_cost, net_amount, asset, eligibility_json, operation_id,
                    idempotency_key, request_hash, attempt, schema_version, created_at, updated_at)
                   VALUES (?, ?, 'settling', ?, 0, 0, ?, 'USD', ?, ?, ?, ?, 1, ?, ?, ?)""",
                (
                    payout_id,
                    merchant_id,
                    gross,
                    gross,
                    compact_json({"payableIds": body["payableIds"], "reason": reason, "capturedAt": now}),
                    operation_id,
                    idempotency_key,
                    request_hash,
                    self.settings.schema_version,
                    now,
                    now,
                ),
            )
            for payable in payables:
                conn.execute(
                    "INSERT INTO payout_items(payout_id,payable_id,amount,state,created_at) VALUES (?, ?, ?, 'claimed', ?)",
                    (payout_id, payable["payable_id"], payable["amount"], now),
                )
                conn.execute(
                    "UPDATE payables SET state='included', payout_id=?, version=version+1, updated_at=? WHERE payable_id=? AND state='eligible'",
                    (payout_id, now, payable["payable_id"]),
                )
        rail_result = self.rail.payout(
            operation_id=operation_id,
            source_id=payout_id,
            amount=gross,
            idempotency_key=f"payout:{idempotency_key}",
        )
        if rail_result["state"] != "settled":
            state = "unknown" if rail_result["state"] == "unknown" else "failed"
            with self.store.business_transaction() as conn:
                conn.execute("UPDATE payouts SET state=?, updated_at=? WHERE payout_id=?", (state, utc_now(), payout_id))
            if state == "failed":
                self._release_failed_payout(payout_id)
            raise MarketplaceError(
                "PAYOUT_UNKNOWN" if state == "unknown" else "INTERNAL_ERROR",
                "Payout did not settle.",
                retryable=state == "unknown",
            )
        payout = self._one("SELECT * FROM payouts WHERE payout_id=?", (payout_id,))
        return self._complete_payout(payout, rail_result, actor_id=actor_id)

    def reconcile_payout(self, payout_id: str, *, actor_id: str, reason: str) -> dict[str, Any]:
        payout = self._one("SELECT * FROM payouts WHERE payout_id=?", (payout_id,))
        if payout["state"] not in {"created", "settling", "unknown", "reconciliation_required"}:
            raise MarketplaceError("INVALID_STATE_TRANSITION", "Payout does not require reconciliation.")
        self._require_active_merchant(payout["merchant_id"])
        rail_result = self.rail.get_operation(str(payout["operation_id"]))
        if rail_result is None:
            rail_result = self.rail.payout(
                operation_id=str(payout["operation_id"]),
                source_id=payout["payout_id"],
                amount=int(payout["net_amount"]),
                idempotency_key=f"payout:{payout['idempotency_key']}",
            )
        if rail_result["state"] == "unknown":
            raise MarketplaceError("PAYOUT_UNKNOWN", "Payout result is still unknown.", retryable=True)
        if rail_result["state"] != "settled":
            with self.store.business_transaction() as conn:
                conn.execute(
                    "UPDATE payouts SET state='failed', updated_at=? WHERE payout_id=?",
                    (utc_now(), payout_id),
                )
            self._release_failed_payout(payout_id)
            raise MarketplaceError("INTERNAL_ERROR", "Payout did not settle.")
        return self._complete_payout(payout, rail_result, actor_id=actor_id)

    def _complete_payout(
        self, payout: dict[str, Any], rail_result: dict[str, Any], *, actor_id: str
    ) -> dict[str, Any]:
        payout_id = payout["payout_id"]
        merchant_id = payout["merchant_id"]
        gross = int(payout["gross_amount"])
        items = self.store.fetch_business(
            "SELECT payable_id, amount FROM payout_items WHERE payout_id=? ORDER BY payable_id",
            (payout_id,),
        )
        payable_ids = tuple(item["payable_id"] for item in items)
        journal_id = f"journal-payout:{payout_id}"
        try:
            self.ledger.post_payout(
                journal_id=journal_id,
                payout_id=payout_id,
                merchant_id=merchant_id,
                amount=gross,
                payable_ids=payable_ids,
                idempotency_key=f"ledger:{payout_id}",
            )
        except Exception as exc:
            with self.store.business_transaction() as conn:
                conn.execute(
                    """UPDATE payouts SET state='reconciliation_required',
                       updated_at=? WHERE payout_id=?""",
                    (utc_now(), payout_id),
                )
            raise MarketplaceError(
                "LEDGER_POST_FAILED", "Payout journal requires reconciliation.", retryable=True
            ) from exc
        receipt = with_signature(
            {
                "receiptType": "payout",
                "receiptId": f"receipt:{payout_id}",
                "profile": PROFILE_URI,
                "simulated": True,
                "status": "paid",
                "issuedAt": rail_result["receipt"]["issuedAt"],
                "issuer": self.settings.marketplace_id,
                "subject": merchant_id,
                "orderId": "payout-batch",
                "references": {"payoutId": payout_id, "payableIds": list(payable_ids), "journalId": journal_id},
                "relatedDigests": {"railReceipt": digest_object(rail_result["receipt"]), "journal": digest_object(self.ledger.get_journal(journal_id))},
            },
            kid=MEDIATOR_KID,
        )
        self.store.put_evidence(
            intent_id=f"receipt:{payout_id}",
            evidence_id=f"evidence:receipt:{payout_id}",
            tenant_type="merchant",
            tenant_id=merchant_id,
            kind="payout",
            exact_bytes=canonical_bytes(receipt),
            kid=MEDIATOR_KID,
        )
        with self.store.business_transaction() as conn:
            conn.execute(
                "UPDATE payouts SET state='paid', journal_id=?, receipt_id=?, updated_at=? WHERE payout_id=?",
                (journal_id, receipt["receiptId"], utc_now(), payout_id),
            )
        response = {
            "payoutId": payout_id,
            "merchantId": merchant_id,
            "state": "paid",
            "grossAmount": gross,
            "commissionAmount": 0,
            "railCost": 0,
            "netAmount": gross,
            "payableIds": list(payable_ids),
            "receipt": receipt,
            "simulated": True,
        }
        self.store.complete_idempotency(
            "payout", actor_id, payout["idempotency_key"], payout["request_hash"], response
        )
        return response

    def _release_failed_payout(self, payout_id: str) -> None:
        now = utc_now()
        with self.store.business_transaction() as conn:
            conn.execute(
                """UPDATE payables SET state='eligible', payout_id=NULL,
                   version=version+1, updated_at=?
                   WHERE payout_id=? AND state='included'""",
                (now, payout_id),
            )
            conn.execute(
                "UPDATE payout_items SET state='released' WHERE payout_id=? AND state='claimed'",
                (payout_id,),
            )

    def payout_status(self, payout_id: str, *, merchant_id: str) -> dict[str, Any]:
        payout = self._maybe_one(
            "SELECT * FROM payouts WHERE payout_id=? AND merchant_id=?", (payout_id, merchant_id)
        )
        if payout is None:
            raise MarketplaceError("FORBIDDEN", "Payout is unavailable for this merchant.", status_code=403)
        items = self.store.fetch_business("SELECT payable_id, amount, state FROM payout_items WHERE payout_id=?", (payout_id,))
        return {
            "payoutId": payout_id,
            "merchantId": merchant_id,
            "state": payout["state"],
            "grossAmount": payout["gross_amount"],
            "commissionAmount": payout["commission_amount"],
            "railCost": payout["rail_cost"],
            "netAmount": payout["net_amount"],
            "receiptId": payout["receipt_id"],
            "items": items,
            "profile": PROFILE_URI,
            "simulated": True,
        }

    # -- helpers -------------------------------------------------------------

    def _load_or_create_guarantee(
        self,
        *,
        evidence_id: str,
        guarantee_id: str,
        accepted: dict[str, Any],
        order: dict[str, Any],
        quote: dict[str, Any],
        pricing: dict[str, Any],
        journal_id: str,
        x402_receipt: dict[str, Any],
        ap2_receipt: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        """Return the one immutable guarantee, recovering evidence-first crashes."""

        metadata = self.store.get_evidence_metadata(evidence_id)
        if metadata is not None:
            raw = self.store.read_evidence(
                evidence_id,
                actor_id=self.settings.marketplace_id,
                actor_role="operator",
            )
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError) as exc:
                raise MarketplaceError("GUARANTEE_INVALID", "Stored guarantee is invalid.") from exc
            if canonical_bytes(payload) != raw:
                raise MarketplaceError("GUARANTEE_INVALID", "Stored guarantee bytes are not canonical.")
        else:
            issued = self.clock()
            claims = with_signature(
                {
                    "kind": "payment-guarantee",
                    "profile": PROFILE_URI,
                    "simulated": True,
                    "guaranteeId": guarantee_id,
                    "merchantQuoteRequirementDigest": quote["requirement_digest"],
                    "orderId": order["order_id"],
                    "taskId": order["task_id"],
                    "quoteId": order["quote_id"],
                    "merchantId": self.settings.merchant_id,
                    "upstreamX402ReceiptDigest": digest_object(x402_receipt),
                    "upstreamAp2ReceiptDigest": digest_object(ap2_receipt),
                    "payableJournalTransactionId": journal_id,
                    "payableEntryId": f"{journal_id}:entry:2",
                    "payableAmount": str(pricing["merchant_payable_amount"]),
                    "commission": "0",
                    "currency": "USD",
                    "payoutTermsVersion": "manual-payout-v1",
                    "iat": issued,
                    "exp": issued + self.settings.guarantee_ttl_seconds,
                },
                kid=MEDIATOR_KID,
            )
            payload = {"x402Version": 2, "accepted": accepted, "payload": claims}
            evidence = self.store.put_evidence(
                intent_id=f"guarantee:{guarantee_id}",
                evidence_id=evidence_id,
                tenant_type="merchant",
                tenant_id=self.settings.merchant_id,
                kind="marketplace-guarantee",
                exact_bytes=canonical_bytes(payload),
                kid=MEDIATOR_KID,
            )
            if evidence["state"] != "committed":
                raise MarketplaceError(
                    "LEDGER_POST_FAILED", "Guarantee evidence is not durable.", retryable=True
                )

        digest = digest_object(payload)
        claims = payload.get("payload") if isinstance(payload, dict) else None
        try:
            if payload.get("x402Version") != 2 or payload.get("accepted") != accepted:
                raise ValueError("accepted binding mismatch")
            if not isinstance(claims, dict):
                raise ValueError("claims missing")
            verify_payload_signature(claims, expected_kid=MEDIATOR_KID)
            if (
                claims.get("guaranteeId") != guarantee_id
                or claims.get("orderId") != order["order_id"]
                or claims.get("taskId") != order["task_id"]
                or claims.get("quoteId") != order["quote_id"]
                or claims.get("merchantId") != self.settings.merchant_id
                or claims.get("merchantQuoteRequirementDigest") != quote["requirement_digest"]
                or claims.get("payableJournalTransactionId") != journal_id
                or claims.get("payableAmount") != str(pricing["merchant_payable_amount"])
            ):
                raise ValueError("guarantee binding mismatch")
            if metadata is not None and metadata["digest"] != f"sha256:{__import__('hashlib').sha256(canonical_bytes(payload)).hexdigest()}":
                raise ValueError("evidence digest mismatch")
        except Exception as exc:
            raise MarketplaceError("GUARANTEE_INVALID", "Stored guarantee binding is invalid.") from exc
        return payload, digest

    @staticmethod
    def _checkout_hash(checkout_jwt: str) -> str:
        from .canonical import checkout_hash

        return checkout_hash(checkout_jwt)

    def _verify_quote(
        self, response: dict[str, Any], *, order_id: str, task_id: str
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        requirement = response.get("requirement")
        if not isinstance(requirement, dict):
            raise MarketplaceError("INVALID_SCHEMA", "Merchant quote requirement is missing.")
        try:
            verify_payload_signature(requirement, expected_kid=MERCHANT_KID)
        except Exception as exc:
            raise MarketplaceError("INVALID_SIGNATURE", "Merchant quote signature is invalid.") from exc
        quote = requirement.get("quote")
        accepts = requirement.get("accepts")
        if not isinstance(quote, dict) or not isinstance(accepts, list) or len(accepts) != 1:
            raise MarketplaceError("INVALID_SCHEMA", "Merchant quote shape is invalid.")
        accepted = accepts[0]
        checks = (
            requirement.get("x402Version") == 2,
            requirement.get("profile") == PROFILE_URI,
            requirement.get("simulated") is True,
            quote.get("issuer") == self.settings.merchant_id,
            quote.get("audience") == self.settings.marketplace_id,
            quote.get("orderId") == order_id,
            quote.get("taskId") == task_id,
            quote.get("merchantId") == self.settings.merchant_id,
            quote.get("pricingPolicyVersion") == PRICING_POLICY_VERSION,
            accepted.get("scheme") == "platform-credit",
            accepted.get("network") == "demo:mediation-ledger",
            accepted.get("payTo") == self.settings.merchant_id,
            accepted.get("asset") == "USD",
            accepted.get("decimals") == 2,
            int(quote.get("iat", 0)) <= self.clock() < int(quote.get("exp", 0)),
            response.get("checkoutJwt") == quote.get("checkoutJwt"),
        )
        if not all(checks):
            raise MarketplaceError("QUOTE_MISMATCH", "Merchant quote binding is invalid.")
        digest = digest_object(requirement)
        if response.get("quoteDigest") != digest:
            raise MarketplaceError("QUOTE_MISMATCH", "Merchant quote digest is invalid.")
        return requirement, quote, digest

    def _quote_requirement(self, evidence_id: str) -> dict[str, Any]:
        raw = self.store.read_evidence(
            evidence_id,
            actor_id=self.settings.marketplace_id,
            actor_role="operator",
        )
        return json.loads(raw)

    def _upstream_acceptance(
        self, order_id: str, quote_digest: str, pricing: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "scheme": "exact-simulated",
            "network": "demo:local",
            "amount": str(pricing["customer_total"]),
            "asset": "USD",
            "decimals": 2,
            "payTo": self.settings.marketplace_id,
            "maxTimeoutSeconds": 300,
            "extra": {"profile": PROFILE_URI, "simulated": True, "quoteDigest": quote_digest},
        }

    def _owned_order(self, order_id: str, customer_id: str) -> dict[str, Any]:
        order = self._maybe_one(
            "SELECT * FROM orders WHERE order_id=? AND customer_id=?", (order_id, customer_id)
        )
        if order is None:
            raise MarketplaceError("FORBIDDEN", "Order is unavailable for this customer.", status_code=403)
        return order

    def _transition_order(
        self,
        order_id: str,
        from_state: str,
        to_state: str,
        reason: str,
        *,
        actor_id: str | None = None,
        recovery_kind: str | None = None,
        operation_id: str | None = None,
    ) -> None:
        order = self._one("SELECT * FROM orders WHERE order_id=?", (order_id,))
        if order["state"] != from_state:
            raise MarketplaceError("INVALID_STATE_TRANSITION", f"Expected {from_state}, got {order['state']}.")
        self.store.update_order_state(
            order_id,
            from_state,
            to_state,
            actor_id=actor_id or self.settings.marketplace_id,
            reason=reason,
            expected_version=int(order["version"]),
            recovery_kind=recovery_kind,
            authoritative_operation_id=operation_id,
        )

    def _update_charge(
        self,
        charge_id: str,
        state: str,
        *,
        proof_digest: str | None = None,
        operation_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        with self.store.business_transaction() as conn:
            conn.execute(
                """UPDATE charges SET state=?, proof_digest=COALESCE(?,proof_digest),
                   operation_id=COALESCE(?,operation_id),
                   idempotency_key=COALESCE(?,idempotency_key),
                   version=version+1, updated_at=?
                   WHERE charge_id=?""",
                (state, proof_digest, operation_id, idempotency_key, utc_now(), charge_id),
            )

    def _mark_refund_required(self, order_id: str, payable_id: str, reason: str) -> None:
        order = self._one("SELECT * FROM orders WHERE order_id=?", (order_id,))
        if order["state"] in {"payable_posted", "guarantee_issued", "fulfilling", "reconciliation_required"}:
            self._transition_order(order_id, order["state"], "refund_required", reason)
        with self.store.business_transaction() as conn:
            conn.execute(
                "UPDATE payables SET state='reversing', version=version+1, updated_at=? WHERE payable_id=? AND state IN ('open','guaranteed','eligible')",
                (utc_now(), payable_id),
            )

    def _one(self, query: str, params: tuple[Any, ...]) -> dict[str, Any]:
        value = self._maybe_one(query, params)
        if value is None:
            raise MarketplaceError("INTERNAL_ERROR", "Required payment state is missing.", status_code=500)
        return value

    def _maybe_one(self, query: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        rows = self.store.fetch_business(query, params)
        return rows[0] if rows else None

    def _require_active_merchant(self, merchant_id: str) -> dict[str, Any]:
        onboarding = self._maybe_one(
            "SELECT * FROM merchant_onboarding WHERE merchant_id=?", (merchant_id,)
        )
        if onboarding is None:
            raise MarketplaceError(
                "MERCHANT_NOT_ONBOARDED", "Merchant is not onboarded.", status_code=403
            )
        now = utc_now()
        valid = (
            merchant_id == self.settings.merchant_id
            and onboarding["status"] == "active"
            and onboarding["key_id"] == MERCHANT_KID
            and onboarding["agreement_version"] == "demo-agreement-v1"
            and onboarding["pricing_policy_version"] == PRICING_POLICY_VERSION
            and onboarding["payout_destination"] == "demo-merchant"
            and bool(onboarding["endpoint"])
            and onboarding["valid_from"] <= now
            and (onboarding["valid_to"] is None or now < onboarding["valid_to"])
            and int(onboarding["schema_version"]) == self.settings.schema_version
        )
        if not valid:
            raise MarketplaceError(
                "MERCHANT_SUSPENDED",
                "Merchant onboarding is inactive or no longer valid.",
                status_code=403,
            )
        return onboarding

    def _complete_payment_idempotency(
        self, charge: dict[str, Any], response: dict[str, Any]
    ) -> None:
        records = self.store.fetch_business(
            """SELECT request_hash FROM idempotency_records
               WHERE scope='payment-submit' AND actor_id=? AND idempotency_key=?""",
            (self.settings.customer_id, charge["idempotency_key"]),
        )
        if records:
            self.store.complete_idempotency(
                "payment-submit",
                self.settings.customer_id,
                charge["idempotency_key"],
                records[0]["request_hash"],
                response,
            )

    @staticmethod
    def _cached_idempotent_response(record: dict[str, Any]) -> dict[str, Any] | None:
        if record["status"] != "hit":
            return None
        if record["response"] is not None:
            return record["response"]
        raise MarketplaceError(
            "SETTLEMENT_UNKNOWN",
            "The original idempotent operation is still in progress or requires reconciliation.",
            status_code=409,
            retryable=True,
        )

    @staticmethod
    def _pricing_public(pricing: dict[str, Any]) -> dict[str, Any]:
        return {
            "policyVersion": pricing["policy_version"],
            "merchandiseAmount": pricing["merchandise_amount"],
            "customerSurcharge": pricing["customer_surcharge"],
            "collectionRailCost": pricing["collection_rail_cost"],
            "customerTotal": pricing["customer_total"],
            "providerCommission": pricing["provider_commission"],
            "merchantPayableAmount": pricing["merchant_payable_amount"],
            "payoutRailCost": pricing["payout_rail_cost"],
            "asset": pricing["asset"],
            "currency": pricing["asset"],
            "decimals": pricing["decimals"],
            "network": pricing["network"],
            "roundingRule": pricing["rounding_rule"],
            "calculatedAt": pricing["calculated_at"],
        }

    @staticmethod
    def _public_row(row: dict[str, Any] | None, hidden: set[str]) -> dict[str, Any] | None:
        if row is None:
            return None
        return {key: value for key, value in row.items() if key not in hidden and not key.endswith("_json")}
