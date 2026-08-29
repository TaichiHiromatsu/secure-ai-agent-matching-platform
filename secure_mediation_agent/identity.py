"""Verified demo identity assertions shared by auth, workflow API, and adapters."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any

from ap2.sdk.jwt_helper import create_jwt, verify_jwt
from jwcrypto.jwk import JWK

from secure_mediation_agent.ap2.keys import public_key


IDENTITY_AUDIENCE = "secure-mediation-workflow-api"
IDENTITY_ISSUER = "secure-mediation-auth"
DEMO_TENANT_ID = "demo-tenant"
DEMO_CUSTOMER_ID = "demo-customer"
# Private ADK session-state key.  The authenticated edge bridge injects a
# freshly signed assertion on every run and removes this value from responses.
# Browser-provided values under this key are never authoritative.
ADK_IDENTITY_STATE_KEY = "_secureVerifiedIdentityAssertion"


@dataclass(frozen=True, slots=True)
class VerifiedIdentity:
    tenant_id: str
    customer_id: str
    subject: str
    assertion_id: str


def issue_identity_assertion(
    key: JWK,
    *,
    subject: str,
    tenant_id: str = DEMO_TENANT_ID,
    customer_id: str = DEMO_CUSTOMER_ID,
    now: int | None = None,
    lifetime_seconds: int = 60,
) -> str:
    issued_at = int(time.time()) if now is None else now
    claims: dict[str, Any] = {
        "typ": "secure-verified-identity+jwt",
        "iss": IDENTITY_ISSUER,
        "aud": IDENTITY_AUDIENCE,
        "sub": subject,
        "tenantId": tenant_id,
        "customerId": customer_id,
        "jti": f"identity:{secrets.token_urlsafe(24)}",
        "iat": issued_at,
        "exp": issued_at + lifetime_seconds,
    }
    return create_jwt(
        {"alg": "ES256", "kid": key.get("kid"), "typ": "JWT"},
        claims,
        key,
    )


def verify_identity_assertion(
    token: str,
    key: JWK,
    *,
    now: int | None = None,
) -> VerifiedIdentity:
    claims = verify_jwt(token, public_key(key))
    current = int(time.time()) if now is None else now
    if claims.get("typ") != "secure-verified-identity+jwt":
        raise ValueError("identity assertion type mismatch")
    if claims.get("iss") != IDENTITY_ISSUER or claims.get("aud") != IDENTITY_AUDIENCE:
        raise ValueError("identity assertion issuer or audience mismatch")
    if not isinstance(claims.get("iat"), int) or not isinstance(claims.get("exp"), int):
        raise ValueError("identity assertion time claims missing")
    if current < claims["iat"] - 30 or current > claims["exp"]:
        raise ValueError("identity assertion expired or not yet valid")
    expected = {
        "sub": str,
        "tenantId": str,
        "customerId": str,
        "jti": str,
    }
    for name, expected_type in expected.items():
        if not isinstance(claims.get(name), expected_type) or not claims[name]:
            raise ValueError(f"identity assertion {name} missing")
    # This release has one pre-authenticated demo identity. Unknown Firebase
    # subjects are never allowed to choose tenant/customer IDs via headers/body.
    if claims["tenantId"] != DEMO_TENANT_ID or claims["customerId"] != DEMO_CUSTOMER_ID:
        raise ValueError("identity is not mapped to the approved demo principal")
    return VerifiedIdentity(
        tenant_id=claims["tenantId"],
        customer_id=claims["customerId"],
        subject=claims["sub"],
        assertion_id=claims["jti"],
    )
