"""Independent deterministic verification helpers around the pinned SDK."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from ap2.sdk.jwt_helper import verify_jwt
from ap2.sdk.mandate import MandateClient
from jwcrypto.jwk import JWK


def b64url_sha256(value: bytes | str) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode("ascii")


def public_provider(root_key: JWK):
    public = JWK.from_json(root_key.export_public())
    return lambda _token: public


def verify_terminal_presentation(
    token: str,
    *,
    root_key: JWK,
    audience: str,
    nonce: str,
    expected_vct: str,
) -> dict[str, Any]:
    payloads = MandateClient().verify(
        token,
        public_provider(root_key),
        expected_aud=audience,
        expected_nonce=nonce,
    )
    if not isinstance(payloads, list) or not payloads:
        raise ValueError("AP2 presentation did not contain an effective mandate")
    leaf = payloads[-1]
    if leaf.get("vct") != expected_vct:
        raise ValueError("AP2 mandate vct mismatch")
    return leaf


def closed_reference(presentation: str) -> str:
    leaf = MandateClient().get_closed_mandate_jwt(presentation)
    return b64url_sha256(leaf)


def verify_role_jwt(
    token: str,
    *,
    public_key: JWK,
    expected_issuer: str,
    expected_kid: str,
) -> dict[str, Any]:
    protected = json.loads(
        base64.urlsafe_b64decode(token.split(".", 1)[0] + "==")
    )
    if protected.get("alg") != "ES256" or protected.get("kid") != expected_kid:
        raise ValueError("signed object algorithm or kid mismatch")
    payload = verify_jwt(token, public_key)
    if payload.get("iss") != expected_issuer:
        raise ValueError("signed object issuer mismatch")
    return payload
