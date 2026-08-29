"""Canonical AP2 Checkout and Payment Receipt factory."""

from __future__ import annotations

from ap2.sdk.generated.checkout_receipt import (
    CheckoutReceipt,
    CheckoutReceiptError,
    CheckoutReceiptSuccess,
)
from ap2.sdk.generated.payment_receipt import (
    PaymentReceipt,
    PaymentReceiptError,
    PaymentReceiptSuccess,
)
from ap2.sdk.jwt_helper import create_jwt, verify_jwt
from jwcrypto.jwk import JWK


class Ap2ReceiptFactory:
    @staticmethod
    def checkout(
        *,
        key: JWK,
        reference: str,
        issued_at: int,
        order_id: str,
        success: bool,
        error: str = "invalid_mandate",
        error_description: str = "Checkout mandate rejected.",
    ) -> str:
        if success:
            root = CheckoutReceiptSuccess(
                status="Success",
                iss="demo-merchant",
                iat=issued_at,
                reference=reference,
                order_id=order_id,
            )
        else:
            root = CheckoutReceiptError(
                status="Error",
                iss="demo-merchant",
                iat=issued_at,
                reference=reference,
                error=error,
                error_description=error_description,
            )
        receipt = CheckoutReceipt(root=root)
        return create_jwt(
            {"alg": "ES256", "kid": key.get("kid"), "typ": "JWT"},
            receipt.model_dump(mode="json"),
            key,
        )

    @staticmethod
    def payment(
        *,
        key: JWK,
        reference: str,
        issued_at: int,
        payment_id: str,
        simulation_reference: str,
        success: bool,
        error: str = "invalid_credential",
        error_description: str = "Payment authorization rejected.",
    ) -> str:
        if success:
            root = PaymentReceiptSuccess(
                status="Success",
                iss="demo-mpp",
                iat=issued_at,
                reference=reference,
                payment_id=payment_id,
                psp_confirmation_id=simulation_reference,
                network_confirmation_id=simulation_reference,
            )
        else:
            root = PaymentReceiptError(
                status="Error",
                iss="demo-mpp",
                iat=issued_at,
                reference=reference,
                error=error,
                error_description=error_description,
                payment_id=payment_id,
            )
        receipt = PaymentReceipt(root=root)
        return create_jwt(
            {"alg": "ES256", "kid": key.get("kid"), "typ": "JWT"},
            receipt.model_dump(mode="json"),
            key,
        )

    @staticmethod
    def verify_checkout(token: str, key: JWK, reference: str) -> CheckoutReceipt:
        receipt = CheckoutReceipt.model_validate(verify_jwt(token, key))
        if receipt.root.iss != "demo-merchant" or receipt.root.reference != reference:
            raise ValueError("Checkout Receipt issuer/reference mismatch")
        return receipt

    @staticmethod
    def verify_payment(token: str, key: JWK, reference: str) -> PaymentReceipt:
        receipt = PaymentReceipt.model_validate(verify_jwt(token, key))
        if receipt.root.iss != "demo-mpp" or receipt.root.reference != reference:
            raise ValueError("Payment Receipt issuer/reference mismatch")
        return receipt
