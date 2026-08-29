"""Project-local signed request authentication for the deterministic demo."""

from __future__ import annotations

import time
from typing import Any

from .canonical import digest_object, verify_payload_signature, with_signature
from .config import PROFILE_URI, subject_for_kid


class RequestAuthenticationError(ValueError):
    pass


def build_request_auth(
    *,
    method: str,
    path: str,
    body: dict[str, Any],
    subject: str,
    role: str,
    tenant_id: str,
    kid: str,
    nonce: str,
    timestamp: int | None = None,
) -> dict[str, Any]:
    unsigned = {
        "profile": PROFILE_URI,
        "simulated": True,
        "method": method.upper(),
        "path": path,
        "bodyDigest": digest_object(body),
        "actor": {"subject": subject, "role": role, "tenantId": tenant_id},
        "nonce": nonce,
        "timestamp": int(time.time()) if timestamp is None else timestamp,
    }
    return with_signature(unsigned, kid=kid)


def verify_request_auth(
    auth: dict[str, Any],
    *,
    method: str,
    path: str,
    body: dict[str, Any],
    expected_role: str,
    expected_tenant: str | None = None,
    now: int | None = None,
    max_skew_seconds: int = 300,
) -> dict[str, str]:
    if not isinstance(auth, dict):
        raise RequestAuthenticationError("request authentication is required")
    required = {"profile", "simulated", "method", "path", "bodyDigest", "actor", "nonce", "timestamp", "signature"}
    if set(auth) != required:
        raise RequestAuthenticationError("request authentication shape is invalid")
    if auth["profile"] != PROFILE_URI or auth["simulated"] is not True:
        raise RequestAuthenticationError("request profile is unsupported")
    if auth["method"] != method.upper() or auth["path"] != path:
        raise RequestAuthenticationError("request method/path binding is invalid")
    if auth["bodyDigest"] != digest_object(body):
        raise RequestAuthenticationError("request body binding is invalid")
    actor = auth["actor"]
    if not isinstance(actor, dict) or set(actor) != {"subject", "role", "tenantId"}:
        raise RequestAuthenticationError("request actor shape is invalid")
    if actor["role"] != expected_role:
        raise RequestAuthenticationError("request actor role is forbidden")
    if expected_tenant is not None and actor["tenantId"] != expected_tenant:
        raise RequestAuthenticationError("request tenant is forbidden")
    timestamp = auth["timestamp"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        raise RequestAuthenticationError("request timestamp is invalid")
    current = int(time.time()) if now is None else now
    if abs(current - timestamp) > max_skew_seconds:
        raise RequestAuthenticationError("request authentication expired")
    signature = auth["signature"]
    kid = signature.get("kid") if isinstance(signature, dict) else None
    if not isinstance(kid, str) or subject_for_kid(kid) != actor["subject"]:
        raise RequestAuthenticationError("request key is not bound to actor")
    try:
        verify_payload_signature(auth)
    except Exception as exc:
        raise RequestAuthenticationError("request signature is invalid") from exc
    nonce = auth["nonce"]
    if not isinstance(nonce, str) or not nonce:
        raise RequestAuthenticationError("request nonce is invalid")
    return {
        "subject": str(actor["subject"]),
        "role": str(actor["role"]),
        "tenant_id": str(actor["tenantId"]),
        "kid": kid,
        "nonce": nonce,
    }
