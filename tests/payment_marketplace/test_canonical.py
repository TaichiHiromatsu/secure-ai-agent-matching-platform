from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from secure_mediation_agent.payment_marketplace.canonical import (
    CanonicalizationError,
    DuplicateKeyError,
    InvalidSignatureError,
    base64url_decode,
    base64url_encode,
    canonical_bytes,
    canonical_json,
    checkout_hash,
    digest_object,
    loads_strict,
    sign_payload,
    verify_payload_signature,
    with_signature,
)
from secure_mediation_agent.payment_marketplace.config import (
    CUSTOMER_KID,
    MEDIATOR_KID,
    PROFILE_URI,
    UnknownKeyIdError,
    public_test_key_registry,
)
from secure_mediation_agent.payment_marketplace.models import (
    CheckoutMandate,
    ErrorCode,
    PricingBreakdown,
    ReceiptEnvelope,
    StableError,
    calculate_zero_fee_pricing,
)
from secure_mediation_agent.payment_marketplace.trusted_surface import (
    TrustedSurface,
    verify_approval,
)


FIXED_NOW = datetime(2026, 8, 15, 3, 4, 5, tzinfo=timezone.utc)


def test_canonical_json_sorts_recursively_and_preserves_unicode() -> None:
    left = {"z": [{"β": 2, "a": 1}], "message": "決済"}
    right = loads_strict(
        b'{ "message" : "\xe6\xb1\xba\xe6\xb8\x88", "z" : [ { "a":1,"\xce\xb2":2 } ] }'
    )

    expected = '{"message":"決済","z":[{"a":1,"β":2}]}'
    assert canonical_json(left) == expected
    assert canonical_bytes(right) == expected.encode("utf-8")
    assert digest_object(left) == digest_object(right)


@pytest.mark.parametrize(
    "raw",
    [
        '{"amount":1.0}',
        '{"amount":NaN}',
        '{"amount":Infinity}',
        '{"amount":-Infinity}',
    ],
)
def test_strict_json_parser_rejects_float_and_non_finite(raw: str) -> None:
    with pytest.raises(CanonicalizationError):
        loads_strict(raw)


def test_strict_json_parser_rejects_duplicate_keys() -> None:
    with pytest.raises(DuplicateKeyError):
        loads_strict('{"nonce":"first","nonce":"second"}')


@pytest.mark.parametrize(
    "value",
    [
        {"amount": 1.0},
        {"amount": float("nan")},
        {"amount": float("inf")},
        {"value": object()},
        {1: "non-string-key"},
        ("tuple-is-not-json-array",),
    ],
)
def test_canonicalizer_rejects_unsupported_python_values(value: object) -> None:
    with pytest.raises(CanonicalizationError):
        canonical_bytes(value)


def test_base64url_is_unpadded_and_strict() -> None:
    encoded = base64url_encode(b"\xff\x00checkout")
    assert "=" not in encoded
    assert base64url_decode(encoded) == b"\xff\x00checkout"

    with pytest.raises(CanonicalizationError):
        base64url_decode(encoded + "=")


def test_checkout_hash_uses_exact_field_value_bytes() -> None:
    compact = "eyJhbGciOiJFUzI1NiJ9.eyJvcmRlciI6IjEyMyJ9.signature"
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(compact.encode("utf-8")).digest()
    ).rstrip(b"=").decode("ascii")

    assert checkout_hash(compact) == expected
    assert len(expected) == 43
    assert checkout_hash(compact + " ") != expected


def test_sign_verify_tamper_and_top_level_signature_exclusion() -> None:
    payload = {"subject": "demo-customer", "amount": 1250, "nested": {"b": 2}}
    signature = sign_payload(payload, kid=CUSTOMER_KID)
    verify_payload_signature(payload, signature)

    signed = with_signature(payload, kid=CUSTOMER_KID)
    verify_payload_signature(signed)
    assert signed["signature"]["alg"] == "HS256"
    assert signed["signature"]["kid"] == CUSTOMER_KID

    # Re-signing ignores an existing top-level signature, not nested evidence.
    assert sign_payload(signed, kid=CUSTOMER_KID) == signature

    tampered = {**signed, "amount": 1251}
    with pytest.raises(InvalidSignatureError):
        verify_payload_signature(tampered)


def test_unknown_kid_fails_closed() -> None:
    with pytest.raises(UnknownKeyIdError):
        sign_payload({"value": 1}, kid="unknown-kid")

    with pytest.raises(UnknownKeyIdError):
        verify_payload_signature(
            {"value": 1},
            {"alg": "HS256", "kid": "unknown-kid", "value": "AA"},
        )


def test_public_key_metadata_and_repr_do_not_expose_test_key_material() -> None:
    serialized = repr(public_test_key_registry())
    assert "test-only-demo" not in serialized
    assert CUSTOMER_KID in serialized


def test_stable_error_accepts_known_wire_code_and_rejects_unknown_or_wrong_retryability() -> None:
    error = StableError(
        code="SETTLEMENT_UNKNOWN",
        message="Settlement result is not yet authoritative",
        retryable=True,
        correlationId="corr-001",
    )
    assert error.code is ErrorCode.SETTLEMENT_UNKNOWN

    with pytest.raises(ValidationError):
        StableError(
            code="NOT_A_CODE",
            message="bad",
            retryable=False,
            correlationId="corr-002",
        )

    with pytest.raises(ValidationError):
        StableError(
            code="SETTLEMENT_UNKNOWN",
            message="bad retry flag",
            retryable=False,
            correlationId="corr-003",
        )


def test_zero_fee_pricing_is_strict_integer_minor_units() -> None:
    pricing = calculate_zero_fee_pricing(1250, calculated_at=FIXED_NOW)

    assert pricing.merchandise_amount == 1250
    assert pricing.customer_surcharge == 0
    assert pricing.collection_rail_cost == 0
    assert pricing.customer_total == 1250
    assert pricing.provider_commission == 0
    assert pricing.payout_rail_cost == 0
    assert pricing.merchant_payable_amount == 1250
    assert pricing.currency == "USD"
    assert pricing.decimals == 2

    with pytest.raises(ValidationError):
        calculate_zero_fee_pricing(12.5, calculated_at=FIXED_NOW)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        PricingBreakdown(
            merchandiseAmount=1250,
            customerTotal=1251,
            merchantPayableAmount=1250,
            calculatedAt=FIXED_NOW,
        )


def test_ap2_models_reject_unknown_fields_and_wrong_vct() -> None:
    digest = checkout_hash("a.b.c")

    with pytest.raises(ValidationError):
        CheckoutMandate(
            vct="mandate.checkout.open.1",
            checkout_jwt="a.b.c",
            checkout_hash=digest,
            iat=1,
            exp=2,
        )

    with pytest.raises(ValidationError):
        CheckoutMandate.model_validate(
            {
                "vct": "mandate.checkout.1",
                "checkout_jwt": "a.b.c",
                "checkout_hash": digest,
                "iat": 1,
                "exp": 2,
                "signer": {"kid": CUSTOMER_KID},
            }
        )


def test_separate_receipts_cross_reference_stable_ids_without_cyclic_hashes() -> None:
    payment_payload_digest = digest_object({"payload": "proof"})
    payment_mandate_digest = digest_object({"vct": "mandate.payment.1"})

    x402_unsigned = ReceiptEnvelope(
        receiptType="x402-settlement",
        receiptId="receipt-x402-001",
        status="success",
        issuedAt=FIXED_NOW,
        issuer="mediation-platform",
        subject="demo-customer",
        orderId="order-001",
        relatedDigests={"paymentPayload": payment_payload_digest},
        references={
            "settlementReference": "settlement-001",
            "relatedAp2ReceiptId": "receipt-ap2-001",
        },
    )
    ap2_unsigned = ReceiptEnvelope(
        receiptType="ap2-payment",
        receiptId="receipt-ap2-001",
        status="success",
        issuedAt=FIXED_NOW,
        issuer="mediation-platform",
        subject="demo-customer",
        orderId="order-001",
        relatedDigests={"paymentMandate": payment_mandate_digest},
        references={
            "settlementReference": "settlement-001",
            "relatedX402ReceiptId": "receipt-x402-001",
        },
    )

    x402_signed = ReceiptEnvelope.model_validate(
        {
            **x402_unsigned.wire_dict(),
            "signature": sign_payload(x402_unsigned, kid=MEDIATOR_KID).wire_dict(),
        }
    )
    ap2_signed = ReceiptEnvelope.model_validate(
        {
            **ap2_unsigned.wire_dict(),
            "signature": sign_payload(ap2_unsigned, kid=MEDIATOR_KID).wire_dict(),
        }
    )

    verify_payload_signature(x402_signed)
    verify_payload_signature(ap2_signed)
    assert x402_signed.receipt_id != ap2_signed.receipt_id
    assert x402_signed.references["relatedAp2ReceiptId"] == ap2_signed.receipt_id
    assert ap2_signed.references["relatedX402ReceiptId"] == x402_signed.receipt_id

def test_trusted_surface_builds_official_ap2_shape_and_outer_authorization() -> None:
    pricing = calculate_zero_fee_pricing(1250, calculated_at=FIXED_NOW)
    surface = TrustedSurface(clock=lambda: FIXED_NOW)
    approval = surface.build_approval(
        checkout_jwt="header.payload.merchant-signature",
        pricing=pricing,
        audience="mediation-platform",
        nonce="nonce-001",
        order_id="order-001",
        task_id="task-001",
        quote_id="quote-001",
        challenge_id="challenge-001",
    )

    expected_hash = checkout_hash("header.payload.merchant-signature")
    checkout_wire = approval.checkout_mandate.wire_dict()
    payment_wire = approval.payment_mandate.wire_dict()
    authorization_wire = approval.authorization.wire_dict()

    assert checkout_wire == {
        "vct": "mandate.checkout.1",
        "checkout_jwt": "header.payload.merchant-signature",
        "checkout_hash": expected_hash,
        "iat": int(FIXED_NOW.timestamp()),
        "exp": int(FIXED_NOW.timestamp()) + 300,
    }
    assert payment_wire["vct"] == "mandate.payment.1"
    assert payment_wire["transaction_id"] == expected_hash
    assert payment_wire["payee"] == {
        "id": "mediation-platform",
        "name": "Secure Mediation Marketplace",
    }
    assert payment_wire["payment_amount"] == {"amount": 1250, "currency": "USD"}
    assert payment_wire["payment_instrument"] == {
        "id": "demo-customer",
        "type": "simulation",
        "description": "Demo customer balance",
    }
    assert "signer" not in checkout_wire
    assert "signer" not in payment_wire
    assert "nonce" not in payment_wire
    assert "network" not in payment_wire

    assert authorization_wire["profile"] == PROFILE_URI
    assert authorization_wire["orderId"] == "order-001"
    assert authorization_wire["checkoutMandateDigest"] == digest_object(
        approval.checkout_mandate
    )
    assert authorization_wire["paymentMandateDigest"] == digest_object(
        approval.payment_mandate
    )
    assert approval.display.checkout_jwt == "header.payload.merchant-signature"
    verify_approval(approval)


def test_trusted_surface_verifier_detects_exact_checkout_tampering() -> None:
    pricing = calculate_zero_fee_pricing(500, calculated_at=FIXED_NOW)
    approval = TrustedSurface(clock=lambda: FIXED_NOW).build_approval(
        checkout_jwt="header.payload.signature",
        pricing=pricing,
        audience="mediation-platform",
        nonce="nonce-002",
        order_id="order-002",
        task_id="task-002",
        quote_id="quote-002",
        challenge_id="challenge-002",
    )
    tampered_wire = approval.wire_dict()
    tampered_wire["checkoutMandate"]["checkout_jwt"] += " "
    tampered = type(approval).model_validate(tampered_wire)

    with pytest.raises(InvalidSignatureError):
        verify_approval(tampered)
