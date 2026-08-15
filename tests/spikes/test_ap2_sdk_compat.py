"""Executable compatibility gate for the pinned AP2 v0.2 SDK.

These tests intentionally exercise only the Human Present terminal closed-
mandate path used by the approved simulation release.  The pinned upstream
suite currently has two failures in *intermediate* ``kb+sd-jwt+kb`` audience
and nonce checks; this release never emits an intermediate delegation hop.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time

import pytest
from ap2.sdk.generated.checkout_mandate import CheckoutMandate
from ap2.sdk.generated.checkout_receipt import (
    CheckoutReceipt,
    CheckoutReceiptError,
    CheckoutReceiptSuccess,
)
from ap2.sdk.generated.open_checkout_mandate import OpenCheckoutMandate
from ap2.sdk.generated.open_payment_mandate import OpenPaymentMandate
from ap2.sdk.generated.payment_mandate import PaymentMandate
from ap2.sdk.generated.payment_receipt import (
    PaymentReceipt,
    PaymentReceiptError,
    PaymentReceiptSuccess,
)
from ap2.sdk.generated.types.amount import Amount
from ap2.sdk.generated.types.merchant import Merchant
from ap2.sdk.generated.types.payment_instrument import PaymentInstrument
from ap2.sdk.jwt_helper import create_jwt, verify_jwt
from ap2.sdk.mandate import MandateClient
from jwcrypto.jwk import JWK


pytestmark = [pytest.mark.spike, pytest.mark.contract_ap2]


def _key(kid: str) -> JWK:
    key = JWK.generate(kty="EC", crv="P-256")
    value = json.loads(key.export())
    value["kid"] = kid
    return JWK.from_json(json.dumps(value))


def _public(key: JWK) -> JWK:
    return JWK.from_json(key.export_public())


def _cnf(key: JWK) -> dict[str, object]:
    return {"jwk": json.loads(key.export_public())}


def _b64sha256(value: str) -> str:
    return base64.urlsafe_b64encode(
        hashlib.sha256(value.encode("utf-8")).digest()
    ).rstrip(b"=").decode("ascii")


def _provider(root_key: JWK):
    public = _public(root_key)
    return lambda _token: public


def _terminal_presentations() -> tuple[dict[str, str], dict[str, JWK]]:
    client = MandateClient()
    root_key = _key("demo-user-root-1")
    holder_key = _key("demo-trusted-surface-1")
    merchant_key = _key("demo-merchant-checkout-1")
    now = int(time.time())
    checkout_jwt = create_jwt(
        {"alg": "ES256", "kid": "demo-merchant-checkout-1", "typ": "JWT"},
        {
            "iss": "demo-merchant",
            "aud": "secure-mediation-workflow",
            "jti": "checkout-spike",
            "checkoutNonce": "fresh-entropy-for-spike-only",
            "orderId": "order-spike",
            "taskId": "task-spike",
            "productId": "demo-paid-booking",
            "quantity": 1,
            "amount": 1250,
            "currency": "USD",
            "iat": now,
            "exp": now + 300,
        },
        merchant_key,
    )
    checkout_hash = _b64sha256(checkout_jwt)

    checkout_root = client.create(
        [OpenCheckoutMandate(constraints=[], cnf=_cnf(holder_key))], root_key
    )
    checkout = client.present(
        holder_key,
        checkout_root,
        [
            CheckoutMandate(
                checkout_jwt=checkout_jwt,
                checkout_hash=checkout_hash,
                iat=now,
                exp=now + 300,
            )
        ],
        aud="demo-merchant",
        nonce="checkout-challenge",
    )

    payment_root = client.create(
        [OpenPaymentMandate(constraints=[], cnf=_cnf(holder_key))], root_key
    )
    payment = client.present(
        holder_key,
        payment_root,
        [
            PaymentMandate(
                transaction_id=checkout_hash,
                payee=Merchant(id="demo-merchant", name="Demo Merchant"),
                payment_amount=Amount(amount=1250, currency="USD"),
                payment_instrument=PaymentInstrument(
                    id="demo-instrument-1", type="demo-card"
                ),
                iat=now,
                exp=now + 300,
            )
        ],
        aud="demo-credential-provider",
        nonce="payment-challenge",
    )
    return (
        {
            "checkout": checkout,
            "payment": payment,
            "checkout_hash": checkout_hash,
        },
        {"root": root_key, "holder": holder_key, "merchant": merchant_key},
    )


def _tamper_compact_jwt(token: str) -> str:
    chain, leaf = token.rsplit("~~", 1)
    jwt, *disclosures = leaf.split("~")
    header, payload, signature = jwt.split(".")
    changed = ("A" if signature[0] != "A" else "B") + signature[1:]
    return f"{chain}~~{header}.{payload}.{changed}~{'~'.join(disclosures)}"


def test_terminal_closed_mandates_verify_and_bind_checkout_hash() -> None:
    values, keys = _terminal_presentations()
    client = MandateClient()
    checkout_payloads = client.verify(
        values["checkout"],
        _provider(keys["root"]),
        expected_aud="demo-merchant",
        expected_nonce="checkout-challenge",
    )
    payment_payloads = client.verify(
        values["payment"],
        _provider(keys["root"]),
        expected_aud="demo-credential-provider",
        expected_nonce="payment-challenge",
    )
    assert checkout_payloads[-1]["vct"] == "mandate.checkout.1"
    assert checkout_payloads[-1]["checkout_hash"] == values["checkout_hash"]
    assert payment_payloads[-1]["vct"] == "mandate.payment.1"
    assert payment_payloads[-1]["transaction_id"] == values["checkout_hash"]


@pytest.mark.parametrize(
    ("kind", "audience", "nonce"),
    [
        ("checkout", "wrong-merchant", "checkout-challenge"),
        ("checkout", "demo-merchant", "wrong-nonce"),
        ("payment", "wrong-cp", "payment-challenge"),
        ("payment", "demo-credential-provider", "wrong-nonce"),
    ],
)
def test_terminal_closed_mandate_rejects_audience_or_nonce_tamper(
    kind: str, audience: str, nonce: str
) -> None:
    values, keys = _terminal_presentations()
    with pytest.raises(ValueError, match="mismatch"):
        MandateClient().verify(
            values[kind],
            _provider(keys["root"]),
            expected_aud=audience,
            expected_nonce=nonce,
        )


def test_terminal_closed_mandate_rejects_signature_and_root_issuer_tamper() -> None:
    values, keys = _terminal_presentations()
    client = MandateClient()
    with pytest.raises(Exception):
        client.verify(
            _tamper_compact_jwt(values["payment"]),
            _provider(keys["root"]),
            expected_aud="demo-credential-provider",
            expected_nonce="payment-challenge",
        )
    with pytest.raises(Exception):
        client.verify(
            values["payment"],
            _provider(_key("wrong-user-root")),
            expected_aud="demo-credential-provider",
            expected_nonce="payment-challenge",
        )


@pytest.mark.parametrize("status", ["Success", "Error"])
def test_checkout_and_payment_receipt_variants_verify_and_bind_reference(
    status: str,
) -> None:
    values, _ = _terminal_presentations()
    client = MandateClient()
    checkout_reference = _b64sha256(
        client.get_closed_mandate_jwt(values["checkout"])
    )
    payment_reference = _b64sha256(
        client.get_closed_mandate_jwt(values["payment"])
    )
    merchant_key = _key("merchant-receipt-1")
    mpp_key = _key("mpp-receipt-1")
    now = int(time.time())
    if status == "Success":
        checkout_model = CheckoutReceipt(
            root=CheckoutReceiptSuccess(
                status="Success",
                iss="demo-merchant",
                iat=now,
                reference=checkout_reference,
                order_id="order-spike",
            )
        )
        payment_model = PaymentReceipt(
            root=PaymentReceiptSuccess(
                status="Success",
                iss="demo-mpp",
                iat=now,
                reference=payment_reference,
                payment_id="payment-spike",
                psp_confirmation_id="sim:payment-spike",
                network_confirmation_id="sim:payment-spike",
            )
        )
    else:
        checkout_model = CheckoutReceipt(
            root=CheckoutReceiptError(
                status="Error",
                iss="demo-merchant",
                iat=now,
                reference=checkout_reference,
                error="invalid_mandate",
                error_description="Checkout mandate rejected.",
            )
        )
        payment_model = PaymentReceipt(
            root=PaymentReceiptError(
                status="Error",
                iss="demo-mpp",
                iat=now,
                reference=payment_reference,
                error="invalid_credential",
                error_description="Payment credential rejected.",
                payment_id="payment-spike",
            )
        )

    checkout_jwt = create_jwt(
        {"alg": "ES256", "kid": "merchant-receipt-1"},
        checkout_model.model_dump(mode="json"),
        merchant_key,
    )
    payment_jwt = create_jwt(
        {"alg": "ES256", "kid": "mpp-receipt-1"},
        payment_model.model_dump(mode="json"),
        mpp_key,
    )
    checkout_verified = CheckoutReceipt.model_validate(
        verify_jwt(checkout_jwt, _public(merchant_key))
    )
    payment_verified = PaymentReceipt.model_validate(
        verify_jwt(payment_jwt, _public(mpp_key))
    )
    assert checkout_verified.root.reference == checkout_reference
    assert payment_verified.root.reference == payment_reference
    with pytest.raises(Exception):
        verify_jwt(checkout_jwt, _public(mpp_key))
    with pytest.raises(Exception):
        verify_jwt(payment_jwt, _public(merchant_key))
    assert checkout_verified.root.reference != payment_reference
    assert payment_verified.root.reference != checkout_reference
