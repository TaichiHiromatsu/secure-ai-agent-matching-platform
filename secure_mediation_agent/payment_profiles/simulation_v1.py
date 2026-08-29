"""Project-local x402 wire-shape simulation profile.

This module never declares the canonical x402 extension URI and never returns
a blockchain transaction hash or wallet signature.
"""

from __future__ import annotations

import json
from typing import Any

from ap2.sdk.jwt_helper import create_jwt, verify_jwt
from jwcrypto.jwk import JWK

from secure_mediation_agent.workflow.canonical import canonical_digest

from .base import ProfileReadiness


class SimulationV1Profile:
    profile_id = "x402-wire-simulation/1"
    extension_uri = "urn:secure-a2a:extensions:x402-wire-simulation:v1"
    rail_mode = "simulated"
    conformance_label = "x402 v0.1 wire-shape test fixture (NOT CONFORMANT)"
    scheme = "exact-simulated"
    network = "demo:local"
    asset = "USD"
    pay_to = "merchant:demo-merchant"

    def __init__(self, signing_key: JWK, *, kid: str = "demo-simulation-signer-1") -> None:
        self._signing_key = signing_key
        self.kid = kid

    def build_required(self, *, amount: int) -> dict[str, Any]:
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError("simulation amount must be a non-negative integer")
        return {
            "x402Version": 1,
            "accepts": [
                {
                    "scheme": self.scheme,
                    "network": self.network,
                    "asset": self.asset,
                    "payTo": self.pay_to,
                    "maxAmountRequired": str(amount),
                }
            ],
        }

    def build_proof(self, authorization: dict[str, Any]) -> str:
        payload = {
            **authorization,
            "typ": "secure-simulation-authorization/1",
            "profile": self.profile_id,
            "simulated": True,
            "walletSigned": False,
        }
        return create_jwt(
            {"alg": "ES256", "kid": self.kid, "typ": "JWT"},
            payload,
            self._signing_key,
        )

    def verify_proof(self, proof: str, public_key: JWK) -> dict[str, Any]:
        payload = verify_jwt(proof, public_key)
        if payload.get("profile") != self.profile_id:
            raise ValueError("simulation proof profile mismatch")
        if payload.get("simulated") is not True or payload.get("walletSigned") is not False:
            raise ValueError("simulation proof classification mismatch")
        return payload

    def build_submission(self, *, proof: str) -> dict[str, Any]:
        return {
            "x402Version": 1,
            "network": self.network,
            "scheme": self.scheme,
            "payload": {"simulationAuthorization": proof},
        }

    def issue_guarantee(self, claims: dict[str, Any]) -> str:
        """Issue a demo commitment without claiming debit or settlement."""

        required = {
            "guaranteeId",
            "iss",
            "aud",
            "operation",
            "taskId",
            "contextId",
            "orderId",
            "quoteId",
            "amountMinor",
            "currency",
            "payee",
            "paymentMandateDigest",
            "authorizationEnvelopeDigest",
            "settlementCommitmentId",
            "jti",
            "iat",
            "nbf",
            "exp",
        }
        if set(claims) != required:
            raise ValueError("simulation guarantee claim set mismatch")
        if (
            claims["iss"] != "secure-mediator-payment-authority"
            or claims["aud"] != "a2a-agent:agent-005"
            or claims["operation"] != "merchant.fulfillment.guarantee"
            or claims["amountMinor"] != 1250
            or claims["currency"] != self.asset
            or claims["payee"] != "demo-merchant"
            or claims["exp"] <= claims["iat"]
            or claims["nbf"] != claims["iat"]
        ):
            raise ValueError("simulation guarantee policy mismatch")
        payload = {
            "schemaVersion": "signed-payment-guarantee/1",
            "profileId": self.profile_id,
            "simulated": True,
            "state": "GUARANTEED",
            **claims,
        }
        return create_jwt(
            {"alg": "ES256", "kid": self.kid, "typ": "signed-payment-guarantee+jwt"},
            payload,
            self._signing_key,
        )

    def verify_guarantee(
        self,
        token: str,
        public_key: JWK,
        *,
        expected: dict[str, Any],
    ) -> dict[str, Any]:
        payload = verify_jwt(token, public_key)
        for name, value in {
            "schemaVersion": "signed-payment-guarantee/1",
            "profileId": self.profile_id,
            "simulated": True,
            "state": "GUARANTEED",
            "iss": "secure-mediator-payment-authority",
            "aud": "a2a-agent:agent-005",
            "operation": "merchant.fulfillment.guarantee",
            **expected,
        }.items():
            if payload.get(name) != value:
                raise ValueError(f"simulation guarantee mismatch: {name}")
        forbidden = {
            "settled",
            "debited",
            "credited",
            "transaction",
            "transactionHash",
            "onChain",
        }
        if forbidden.intersection(payload):
            raise ValueError("simulation guarantee contains settlement claim")
        if not isinstance(payload.get("iat"), int) or payload.get("exp", 0) <= payload["iat"]:
            raise ValueError("simulation guarantee lifetime is invalid")
        return payload

    def build_guarantee_submission(
        self,
        *,
        guarantee: str,
        guarantee_digest: str,
        checkout_mandate_digest: str,
        payment_mandate_digest: str,
        authorization_envelope_digest: str,
    ) -> dict[str, Any]:
        return {
            "schemaVersion": "merchant-payment-guarantee-submission/1",
            "profileId": self.profile_id,
            "paymentGuarantee": guarantee,
            "paymentGuaranteeDigest": guarantee_digest,
            "ap2Evidence": {
                "checkoutMandateDigest": checkout_mandate_digest,
                "paymentMandateDigest": payment_mandate_digest,
                "authorizationEnvelopeDigest": authorization_envelope_digest,
            },
        }

    def validate_activation(self, requested: set[str]) -> None:
        if requested != {self.extension_uri}:
            raise ValueError("simulation activation mismatch")

    def settle_receipt(
        self,
        *,
        attempt_id: str,
        success: bool,
        error_reason: str | None = None,
    ) -> dict[str, Any]:
        if success:
            return {
                "success": True,
                "network": self.network,
                "transaction": f"sim:{attempt_id}",
                "simulated": True,
            }
        return {
            "success": False,
            "network": self.network,
            "errorReason": error_reason or "SETTLEMENT_FAILED",
            "simulated": True,
        }

    def requirements_digest(self, required: dict[str, Any]) -> str:
        return canonical_digest(required)

    def readiness(self) -> ProfileReadiness:
        return ProfileReadiness(
            ready=True,
            profile_id=self.profile_id,
            rail_mode="simulated",
            checks={
                "officialProfileEnablement": "NOT RUN",
                "declarationActivation": "PASS (project-local simulation URI)",
                "wireMetadata": "PASS",
                "taskCorrelation": "PASS",
                "receiptHistory": "PASS",
                "walletFacilitatorVerify": "NOT RUN",
                "onChainSettle": "NOT RUN",
            },
        )

    def public_key(self) -> JWK:
        return JWK.from_json(self._signing_key.export_public())
