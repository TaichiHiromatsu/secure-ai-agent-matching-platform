"""Non-agentic AP2 Human Present closed-mandate issuer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ap2.sdk.generated.checkout_mandate import CheckoutMandate
from ap2.sdk.generated.open_checkout_mandate import OpenCheckoutMandate
from ap2.sdk.generated.open_payment_mandate import OpenPaymentMandate
from ap2.sdk.generated.payment_mandate import PaymentMandate
from ap2.sdk.generated.types.amount import Amount
from ap2.sdk.generated.types.merchant import Merchant
from ap2.sdk.generated.types.payment_instrument import PaymentInstrument
from ap2.sdk.mandate import MandateClient

from .keys import DemoKeySet
from .verification import b64url_sha256


@dataclass(frozen=True, slots=True)
class MandatePresentations:
    checkout: str
    payment: str
    checkout_hash: str


class TrustedSurface:
    """Typed trust component; never exposed as an ADK tool."""

    def __init__(self, keys: DemoKeySet) -> None:
        self._keys = keys
        self._client = MandateClient()

    @staticmethod
    def _cnf(key) -> dict[str, Any]:
        return {"jwk": json.loads(key.export_public())}

    def issue_closed_mandates(
        self,
        *,
        checkout_jwt: str,
        merchant_id: str,
        merchant_name: str,
        amount: int,
        currency: str,
        instrument_id: str,
        checkout_audience: str,
        checkout_nonce: str,
        payment_audience: str,
        payment_nonce: str,
        issued_at: int,
        expires_at: int,
    ) -> MandatePresentations:
        if merchant_id != "demo-merchant" or amount != 1250 or currency != "USD":
            raise ValueError("Trusted Surface typed release policy mismatch")
        checkout_hash = b64url_sha256(checkout_jwt)
        checkout_root = self._client.create(
            [
                OpenCheckoutMandate(
                    constraints=[],
                    cnf=self._cnf(self._keys.trusted_surface),
                    iat=issued_at,
                    exp=expires_at,
                )
            ],
            self._keys.user_root,
        )
        checkout = self._client.present(
            self._keys.trusted_surface,
            checkout_root,
            [
                CheckoutMandate(
                    checkout_jwt=checkout_jwt,
                    checkout_hash=checkout_hash,
                    iat=issued_at,
                    exp=expires_at,
                )
            ],
            aud=checkout_audience,
            nonce=checkout_nonce,
        )
        payment_root = self._client.create(
            [
                OpenPaymentMandate(
                    constraints=[],
                    cnf=self._cnf(self._keys.trusted_surface),
                    iat=issued_at,
                    exp=expires_at,
                )
            ],
            self._keys.user_root,
        )
        payment = self._client.present(
            self._keys.trusted_surface,
            payment_root,
            [
                PaymentMandate(
                    transaction_id=checkout_hash,
                    payee=Merchant(id=merchant_id, name=merchant_name),
                    payment_amount=Amount(amount=amount, currency=currency),
                    payment_instrument=PaymentInstrument(
                        id=instrument_id, type="demo-card"
                    ),
                    iat=issued_at,
                    exp=expires_at,
                )
            ],
            aud=payment_audience,
            nonce=payment_nonce,
        )
        return MandatePresentations(
            checkout=checkout,
            payment=payment,
            checkout_hash=checkout_hash,
        )
