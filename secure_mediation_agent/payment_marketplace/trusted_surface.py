"""Deterministic Human Present Trusted Surface fixture.

This module builds AP2 v0.2-shaped closed mandate claims and signs a separate
project-local authorization envelope.  The AP2 objects themselves deliberately
contain no project signer, x402 network, nonce, or order binding fields.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from .canonical import (
    InvalidSignatureError,
    checkout_hash,
    digest_object,
    sign_payload,
    verify_payload_signature,
)
from .config import (
    ASSET,
    CUSTOMER_KID,
    CUSTOMER_SUBJECT,
    DECIMALS,
    UPSTREAM_NETWORK,
    subject_for_kid,
)
from .models import (
    CheckoutMandate,
    PaymentAmount,
    PaymentInstrument,
    PaymentMandate,
    PaymentPayee,
    PricingBreakdown,
    ProjectAuthorization,
    TrustedSurfaceApproval,
    TrustedSurfaceDisplay,
)


Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TrustedSurface:
    """Build deterministic Human Present approval fixtures outside the LLM."""

    def __init__(
        self,
        *,
        subject: str = CUSTOMER_SUBJECT,
        kid: str = CUSTOMER_KID,
        clock: Clock = utc_now,
        default_ttl_seconds: int = 300,
        instrument: PaymentInstrument | None = None,
    ) -> None:
        if not isinstance(default_ttl_seconds, int) or isinstance(
            default_ttl_seconds, bool
        ):
            raise TypeError("default_ttl_seconds must be an integer")
        if default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be positive")
        if subject_for_kid(kid) != subject:
            raise ValueError("subject does not match the fixed test key")

        self._subject = subject
        self._kid = kid
        self._clock = clock
        self._default_ttl_seconds = default_ttl_seconds
        self._instrument = instrument or PaymentInstrument(
            id=CUSTOMER_SUBJECT,
            type="simulation",
            description="Demo customer balance",
        )

    def build_approval(
        self,
        *,
        checkout_jwt: str,
        pricing: PricingBreakdown,
        audience: str,
        nonce: str,
        order_id: str,
        task_id: str,
        quote_id: str,
        challenge_id: str,
        ttl_seconds: int | None = None,
    ) -> TrustedSurfaceApproval:
        """Display checkout/totals and return signed closed-mandate approval."""

        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Trusted Surface clock must be timezone-aware")
        issued_at = int(now.timestamp())
        ttl = self._default_ttl_seconds if ttl_seconds is None else ttl_seconds
        if not isinstance(ttl, int) or isinstance(ttl, bool):
            raise TypeError("ttl_seconds must be an integer")
        if ttl <= 0:
            raise ValueError("ttl_seconds must be positive")
        expires_at = issued_at + ttl

        exact_checkout_hash = checkout_hash(checkout_jwt)
        checkout_mandate = CheckoutMandate(
            checkout_jwt=checkout_jwt,
            checkout_hash=exact_checkout_hash,
            iat=issued_at,
            exp=expires_at,
        )
        payee = PaymentPayee()
        payment_mandate = PaymentMandate(
            transaction_id=exact_checkout_hash,
            payee=payee,
            payment_amount=PaymentAmount(
                amount=pricing.customer_total,
                currency="USD",
            ),
            payment_instrument=self._instrument,
            iat=issued_at,
            exp=expires_at,
        )

        checkout_digest = digest_object(checkout_mandate)
        payment_digest = digest_object(payment_mandate)
        unsigned_authorization = ProjectAuthorization(
            subject=self._subject,
            kid=self._kid,
            audience=audience,
            nonce=nonce,
            orderId=order_id,
            taskId=task_id,
            quoteId=quote_id,
            challengeId=challenge_id,
            checkoutMandateDigest=checkout_digest,
            paymentMandateDigest=payment_digest,
            asset=ASSET,
            network=UPSTREAM_NETWORK,
            decimals=DECIMALS,
            iat=issued_at,
            exp=expires_at,
        )
        signature = sign_payload(unsigned_authorization, kid=self._kid)
        authorization = ProjectAuthorization.model_validate(
            {
                **unsigned_authorization.wire_dict(),
                "signature": signature.wire_dict(),
            }
        )

        return TrustedSurfaceApproval(
            display=TrustedSurfaceDisplay(
                checkout_jwt=checkout_jwt,
                pricing=pricing,
                payee=payee,
                payment_instrument=self._instrument,
            ),
            checkoutMandate=checkout_mandate,
            paymentMandate=payment_mandate,
            authorization=authorization,
        )


def verify_approval(approval: TrustedSurfaceApproval) -> None:
    """Verify exact checkout binding, mandate digests, and outer authorization."""

    expected_hash = checkout_hash(approval.checkout_mandate.checkout_jwt)
    if approval.checkout_mandate.checkout_hash != expected_hash:
        raise InvalidSignatureError("checkout hash does not match exact checkout_jwt")
    if approval.payment_mandate.transaction_id != expected_hash:
        raise InvalidSignatureError("payment transaction_id is not checkout-bound")
    if (
        approval.payment_mandate.payment_amount.amount
        != approval.display.pricing.customer_total
    ):
        raise InvalidSignatureError("payment amount does not match displayed total")
    if approval.authorization.subject != subject_for_kid(approval.authorization.kid):
        raise InvalidSignatureError("authorization subject does not match kid")
    if approval.authorization.subject != CUSTOMER_SUBJECT or approval.authorization.kid != CUSTOMER_KID:
        raise InvalidSignatureError("authorization is not the fixed Human Present customer")
    if approval.authorization.checkout_mandate_digest != digest_object(
        approval.checkout_mandate
    ):
        raise InvalidSignatureError("checkout mandate digest mismatch")
    if approval.authorization.payment_mandate_digest != digest_object(
        approval.payment_mandate
    ):
        raise InvalidSignatureError("payment mandate digest mismatch")
    verify_payload_signature(approval.authorization, expected_kid=CUSTOMER_KID)


def build_human_present_approval(
    *,
    checkout_jwt: str,
    pricing: PricingBreakdown,
    audience: str,
    nonce: str,
    order_id: str,
    task_id: str,
    quote_id: str,
    challenge_id: str,
    clock: Clock = utc_now,
    ttl_seconds: int = 300,
) -> TrustedSurfaceApproval:
    """Convenience wrapper for the fixed demo customer Trusted Surface."""

    return TrustedSurface(clock=clock, default_ttl_seconds=ttl_seconds).build_approval(
        checkout_jwt=checkout_jwt,
        pricing=pricing,
        audience=audience,
        nonce=nonce,
        order_id=order_id,
        task_id=task_id,
        quote_id=quote_id,
        challenge_id=challenge_id,
    )
