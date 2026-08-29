"""Project-local scoped payment credential bound to AP2 and profile evidence."""

from __future__ import annotations

from typing import Any

from ap2.sdk.jwt_helper import create_jwt, verify_jwt
from jwcrypto.jwk import JWK

from secure_mediation_agent.workflow.canonical import sha256_digest

from .keys import DemoKeySet, public_key
from .verification import verify_terminal_presentation


class CredentialProvider:
    issuer = "demo-credential-provider"

    def __init__(self, keys: DemoKeySet) -> None:
        self._keys = keys

    def verify_payment_mandate(
        self,
        presentation: str,
        *,
        nonce: str,
        checkout_hash: str,
        amount: int,
    ) -> dict[str, Any]:
        payload = verify_terminal_presentation(
            presentation,
            root_key=self._keys.user_root,
            audience=self.issuer,
            nonce=nonce,
            expected_vct="mandate.payment.1",
        )
        if payload.get("transaction_id") != checkout_hash:
            raise ValueError("Payment Mandate checkout binding mismatch")
        if payload.get("payee", {}).get("id") != "demo-merchant":
            raise ValueError("Payment Mandate payee mismatch")
        if payload.get("payment_amount") != {"amount": amount, "currency": "USD"}:
            raise ValueError("Payment Mandate amount mismatch")
        return payload

    def issue(
        self,
        *,
        credential_id: str,
        workflow_id: str,
        plan_digest: str,
        task_id: str,
        checkout_hash: str,
        payment_mandate: str,
        requirements_digest: str,
        payload_digest: str,
        nonce: str,
        issued_at: int,
        expires_at: int,
    ) -> str:
        claims = {
            "typ": "secure-payment-credential+jwt",
            "profile": "secure-mediation-credential/1",
            "jti": credential_id,
            "iss": self.issuer,
            "aud": ["merchant:demo-merchant", "demo-mpp"],
            "workflowId": workflow_id,
            "planDigest": plan_digest,
            "taskId": task_id,
            "checkoutHash": checkout_hash,
            "paymentMandateDigest": sha256_digest(payment_mandate),
            "requirementsDigest": requirements_digest,
            "payloadDigest": payload_digest,
            "payeeId": "demo-merchant",
            "amount": 1250,
            "currency": "USD",
            "instrumentId": "demo-instrument-1",
            "settlementTarget": "demo-merchant",
            "nonce": nonce,
            "iat": issued_at,
            "exp": expires_at,
        }
        return create_jwt(
            {
                "alg": "ES256",
                "kid": self._keys.credential_provider.get("kid"),
                "typ": "JWT",
            },
            claims,
            self._keys.credential_provider,
        )

    def verify(self, credential: str, *, task_id: str, payload_digest: str) -> dict[str, Any]:
        claims = verify_jwt(credential, public_key(self._keys.credential_provider))
        if claims.get("iss") != self.issuer or task_id != claims.get("taskId"):
            raise ValueError("credential issuer or task mismatch")
        if claims.get("payloadDigest") != payload_digest:
            raise ValueError("credential payload binding mismatch")
        return claims
