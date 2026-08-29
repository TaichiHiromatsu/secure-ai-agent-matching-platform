"""Firebase login sessions and short-lived signed upstream identities."""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
import google.auth.transport.requests
import google.oauth2.id_token
from pydantic import BaseModel, ConfigDict, StrictStr

from secure_mediation_agent.ap2.keys import load_role_key
from secure_mediation_agent.identity import (
    ADK_IDENTITY_STATE_KEY,
    VerifiedIdentity,
    issue_identity_assertion,
    verify_identity_assertion,
)


app = FastAPI()
FIREBASE_PROJECT_ID = "mediation-a2a-platform"
SESSION_COOKIE = "__Host-payment-session"
CSRF_COOKIE = "__Host-payment-csrf"
FIREBASE_CONFIG_FILE = Path(__file__).with_name("firebase-config.json")
DEV_MODE = os.environ.get("DEV_MODE", "false").lower() == "true"
if DEV_MODE and os.environ.get("APP_ENV") != "local":
    raise RuntimeError("DEV_MODE=true is permitted only when APP_ENV=local")

http_request = google.auth.transport.requests.Request()
ADK_UPSTREAM = os.environ.get("PAYMENT_ADK_INTERNAL_URL", "http://127.0.0.1:8000")
WORKFLOW_UPSTREAM = os.environ.get(
    "PAYMENT_WORKFLOW_INTERNAL_URL", "http://127.0.0.1:8004"
)
_SAFE_ADK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]{0,127}$")
_IDENTITY_SELECTOR_KEYS = {
    "verifiedidentity",
    "verifiedidentityassertion",
    "secureverifiedidentityassertion",
    "subject",
    "subjectid",
    "tenant",
    "tenantid",
    "userid",
    "customerid",
    "sessionid",
    "adksessionid",
    "mediationsessionid",
    "workflowid",
}
_IDENTITY_SELECTOR_HEADERS = (
    "x-subject",
    "x-subject-id",
    "x-tenant",
    "x-tenant-id",
    "x-user-id",
    "x-customer-id",
    "x-session-id",
    "x-adk-session-id",
    "x-mediation-session-id",
    "x-workflow-id",
    "user-id",
    "subject",
    "tenant-id",
)


def _identity_response(subject: str) -> Response:
    key_dir = os.environ.get("AP2_DEMO_KEY_DIR")
    if not key_dir:
        return Response(status_code=503)
    assertion = issue_identity_assertion(
        load_role_key(key_dir, "service_auth"), subject=subject
    )
    return Response(status_code=200, headers={"X-Verified-Identity": assertion})


def _request_origin(request: Request) -> str:
    configured = os.environ.get("AUTH_ALLOWED_ORIGIN")
    if configured:
        return configured.rstrip("/")
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0]
    host = request.headers.get("x-forwarded-host", request.headers.get("host", ""))
    return f"{scheme}://{host}".rstrip("/")


def _require_same_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if not origin or origin.rstrip("/") != _request_origin(request):
        raise HTTPException(status_code=403, detail="same-origin request required")


def _require_csrf(request: Request) -> None:
    cookie = request.cookies.get(CSRF_COOKIE) or request.headers.get(
        "x-verified-csrf-cookie"
    )
    header = request.headers.get("x-csrf-token")
    if not cookie or not header or not secrets.compare_digest(cookie, header):
        raise HTTPException(status_code=403, detail="CSRF validation failed")


def _require_public_mutation(request: Request) -> None:
    """Require an auth-request-confirmed CSRF check on public bridge writes."""

    _require_same_origin(request)
    if request.headers.get("x-verified-csrf") != "1":
        # Direct ASGI tests and any future non-nginx caller must still present
        # the actual double-submit cookie rather than bypassing the check.
        _require_csrf(request)


def _verified_firebase_claims(token: str) -> dict[str, Any]:
    claims = google.oauth2.id_token.verify_firebase_token(
        token, http_request, audience=FIREBASE_PROJECT_ID
    )
    expected_issuer = f"https://securetoken.google.com/{FIREBASE_PROJECT_ID}"
    subject = claims.get("sub")
    if (
        claims.get("aud") != FIREBASE_PROJECT_ID
        or claims.get("iss") != expected_issuer
        or not isinstance(subject, str)
        or not subject
    ):
        raise ValueError("Firebase project, issuer, or subject mismatch")
    return claims


class SessionRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    id_token: StrictStr


def _csrf_response(content: dict[str, object]) -> Response:
    token = secrets.token_urlsafe(32)
    response = JSONResponse({**content, "csrfToken": token})
    response.set_cookie(
        CSRF_COOKIE,
        token,
        secure=True,
        httponly=False,
        samesite="strict",
        path="/",
        max_age=3600,
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/auth/csrf")
async def csrf_token() -> Response:
    return _csrf_response({})


@app.get("/auth/browser-bootstrap")
async def browser_bootstrap(request: Request) -> Response:
    """Give the authenticated browser its fixed subject and CSRF token."""

    _reject_query(request)
    identity, _ = _adk_identity(request)
    return _csrf_response({"subject": identity.subject})


@app.get("/auth/firebase-config")
async def firebase_config() -> dict[str, Any]:
    """Serve Firebase's non-secret public browser configuration same-origin."""

    try:
        config = json.loads(FIREBASE_CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail="Firebase config unavailable") from error
    if config.get("projectId") != FIREBASE_PROJECT_ID:
        raise HTTPException(status_code=503, detail="Firebase project mismatch")
    return config


@app.get("/auth/deployment")
async def deployment_mode() -> dict[str, object]:
    ephemeral = os.environ.get("EPHEMERAL_CLOUD_RUN_DEMO") == "true"
    content: dict[str, object] = {
        "ephemeral": ephemeral,
        "notice": (
            "EPHEMERAL DEMO: state and keys may reset on restart" if ephemeral else ""
        ),
        "officialX402": "NOT RUN",
        "onChainSettlement": "NOT RUN",
    }
    if ephemeral:
        content.update(
            {
                "target": "ephemeral-cloud-run-demo",
                "durability": "NOT PROVIDED",
            }
        )
    return content


@app.post("/auth/session")
async def create_session(body: SessionRequest, request: Request) -> Response:
    """Exchange an ID token for a server-owned HttpOnly session cookie."""

    _require_same_origin(request)
    _require_csrf(request)
    try:
        claims = _verified_firebase_claims(body.id_token)
    except Exception as error:
        raise HTTPException(status_code=401, detail="invalid Firebase ID token") from error
    expires = int(claims.get("exp", int(time.time()) + 300))
    max_age = max(1, min(3600, expires - int(time.time())))
    response = JSONResponse({"authenticated": True})
    response.set_cookie(
        SESSION_COOKIE,
        body.id_token,
        secure=True,
        httponly=True,
        samesite="strict",
        path="/",
        max_age=max_age,
    )
    return response


@app.post("/auth/logout")
async def logout(request: Request) -> Response:
    _require_same_origin(request)
    _require_csrf(request)
    response = Response(status_code=204)
    response.delete_cookie(
        SESSION_COOKIE, secure=True, httponly=True, samesite="strict", path="/"
    )
    return response


@app.get("/auth/verify")
async def verify_token(request: Request) -> Response:
    """Verify the server-set Firebase cookie for nginx ``auth_request``."""

    unsafe = request.headers.get("x-original-method", "GET").upper() in {
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
    }
    if unsafe:
        _require_same_origin(request)
        _require_csrf(request)
    if DEV_MODE:
        response = _identity_response("demo-local-user")
        if unsafe:
            response.headers["X-CSRF-Validated"] = "1"
        return response
    session_cookie = request.cookies.get(SESSION_COOKIE)
    if not session_cookie:
        return Response(status_code=401)
    try:
        claims = _verified_firebase_claims(session_cookie)
        response = _identity_response(str(claims["sub"]))
        if unsafe:
            response.headers["X-CSRF-Validated"] = "1"
        return response
    except Exception:
        return Response(status_code=401)


def _state_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _reject_identity_selectors(value: object) -> None:
    """Reject identity/owner selectors at any client-controlled nesting level."""

    if not isinstance(value, dict):
        raise HTTPException(status_code=422, detail="ADK state must be an object")
    for key, item in value.items():
        if _state_key(key) in _IDENTITY_SELECTOR_KEYS:
            raise HTTPException(
                status_code=403,
                detail="identity selectors are controlled by the authenticated server",
            )
        if isinstance(item, dict):
            _reject_identity_selectors(item)
        elif isinstance(item, list):
            for nested in item:
                if isinstance(nested, dict):
                    _reject_identity_selectors(nested)


def _reject_untrusted_selector_tokens(value: object) -> None:
    """Reject selector-shaped inputs before a downstream DTO can return 422."""

    if not isinstance(value, dict):
        return
    for key, item in value.items():
        normalized = _state_key(key)
        if normalized.endswith("selector") or (
            normalized == "selectiontoken" and item is not None
        ):
            raise HTTPException(
                status_code=403,
                detail="public selector inputs are controlled by the authenticated server",
            )
        if isinstance(item, dict):
            _reject_untrusted_selector_tokens(item)
        elif isinstance(item, list):
            for nested in item:
                if isinstance(nested, dict):
                    _reject_untrusted_selector_tokens(nested)


def _reject_query(request: Request) -> None:
    if request.url.query:
        raise HTTPException(
            status_code=403,
            detail="public identity and workflow selectors are not accepted in query parameters",
        )


def _reject_selector_headers(request: Request) -> None:
    if any(request.headers.get(name) for name in _IDENTITY_SELECTOR_HEADERS):
        raise HTTPException(
            status_code=403,
            detail="client identity selector headers are not accepted",
        )


def _sanitize_adk_value(value: object) -> object:
    """Remove internal identity material from every ADK response shape."""

    if isinstance(value, dict):
        return {
            key: _sanitize_adk_value(item)
            for key, item in value.items()
            if _state_key(key)
            not in {
                "verifiedidentity",
                "verifiedidentityassertion",
                "secureverifiedidentityassertion",
            }
        }
    if isinstance(value, list):
        return [_sanitize_adk_value(item) for item in value]
    return value


def _adk_identity(request: Request) -> tuple[VerifiedIdentity, str]:
    assertion = request.headers.get("X-Verified-Identity")
    key_dir = os.environ.get("AP2_DEMO_KEY_DIR")
    if not assertion or not key_dir:
        raise HTTPException(status_code=401, detail="verified identity is required")
    try:
        identity = verify_identity_assertion(
            assertion, load_role_key(key_dir, "service_auth")
        )
    except Exception as error:
        raise HTTPException(status_code=401, detail="verified identity is invalid") from error
    return identity, assertion


async def _json_body(request: Request, *, optional: bool = False) -> dict[str, Any]:
    raw = await request.body()
    if not raw and optional:
        return {}
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=422, detail="valid JSON body is required") from error
    if not isinstance(value, dict):
        raise HTTPException(status_code=422, detail="JSON body must be an object")
    return value


async def _adk_request(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    query: str = "",
    accept: str | None = None,
    assertion: str,
) -> Response:
    transport = getattr(app.state, "adk_transport", None)
    url = f"{ADK_UPSTREAM.rstrip('/')}{path}"
    if query:
        url = f"{url}?{query}"
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(300.0),
        follow_redirects=False,
        trust_env=False,
        transport=transport,
    ) as client:
        request_kwargs: dict[str, Any] = {
            "headers": {
                "Accept": accept or "application/json",
                "Content-Type": "application/json",
            },
        }
        if payload is not None:
            request_kwargs["json"] = payload
        upstream = await client.request(method, url, **request_kwargs)
    content_type = upstream.headers.get("content-type", "application/json")
    content: bytes
    try:
        sanitized = _sanitize_adk_value(upstream.json())
        content = json.dumps(
            sanitized, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    except (ValueError, UnicodeDecodeError):
        # SSE is line-oriented JSON.  Removing the exact assertion is a final
        # non-disclosure guard even if a future ADK version changes its event
        # envelope.
        content = upstream.content.replace(assertion.encode("utf-8"), b"[redacted]")
    return Response(
        status_code=upstream.status_code,
        content=content,
        media_type=content_type.split(";", 1)[0],
    )


async def _workflow_request(
    request: Request,
    path: str,
    *,
    assertion: str,
    payload: dict[str, Any] | None = None,
) -> Response:
    transport = getattr(app.state, "workflow_transport", None)
    headers = {
        "Accept": request.headers.get("accept", "application/json"),
        "X-Verified-Identity": assertion,
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"
    for name in ("idempotency-key", "x-request-id"):
        value = request.headers.get(name)
        if value:
            headers[name] = value
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(300.0),
        follow_redirects=False,
        trust_env=False,
        transport=transport,
    ) as client:
        kwargs: dict[str, Any] = {"headers": headers}
        if payload is not None:
            kwargs["json"] = payload
        upstream = await client.request(
            request.method,
            f"{WORKFLOW_UPSTREAM.rstrip('/')}{path}",
            **kwargs,
        )
    response_headers: dict[str, str] = {}
    if "content-type" in upstream.headers:
        response_headers["Content-Type"] = upstream.headers["content-type"]
    return Response(
        status_code=upstream.status_code,
        content=upstream.content,
        headers=response_headers,
    )


def _safe_adk_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ADK_ID.fullmatch(value):
        raise HTTPException(status_code=422, detail=f"invalid {name}")
    return value


@app.api_route(
    "/apps/payment_user_agent/users/{requested_user_id}/sessions",
    methods=["GET", "POST"],
)
@app.api_route(
    "/apps/payment_user_agent/users/{requested_user_id}/sessions/{session_tail:path}",
    methods=["GET", "POST", "DELETE"],
)
async def authenticated_adk_sessions(
    request: Request,
    requested_user_id: str,
    session_tail: str = "",
) -> Response:
    """Bind every public ADK session operation to the verified auth subject."""

    identity, assertion = _adk_identity(request)
    _reject_selector_headers(request)
    _reject_query(request)
    _safe_adk_id(requested_user_id, "user ID")
    if requested_user_id != identity.subject:
        raise HTTPException(
            status_code=403,
            detail="session user does not match the authenticated subject",
        )
    tail_parts = [part for part in session_tail.split("/") if part]
    if any(not _SAFE_ADK_ID.fullmatch(part) for part in tail_parts):
        raise HTTPException(status_code=404, detail="session route not found")

    upstream_user = quote(identity.subject, safe="")
    base_path = f"/apps/payment_user_agent/users/{upstream_user}/sessions"
    if request.method == "POST":
        _require_public_mutation(request)
        body = await _json_body(request, optional=True)
        if set(body) - {"sessionId", "state", "events"}:
            raise HTTPException(status_code=422, detail="unsupported session field")
        if body.get("events") not in (None, []):
            raise HTTPException(
                status_code=403,
                detail="browser-provided initial events are not allowed",
            )
        state = body.get("state") or {}
        _reject_identity_selectors(state)
        session_id = tail_parts[0] if tail_parts else body.get("sessionId")
        if session_id is not None:
            session_id = _safe_adk_id(session_id, "session ID")
        if len(tail_parts) > 1:
            raise HTTPException(status_code=404, detail="session route not found")
        body = {
            **({"sessionId": session_id} if session_id else {}),
            "state": {**state, ADK_IDENTITY_STATE_KEY: assertion},
        }
        # Always use the current create-session endpoint so identity state is
        # injected consistently even when the browser uses the deprecated ID
        # path.
        return await _adk_request(
            "POST", base_path, payload=body, assertion=assertion
        )

    if request.method == "DELETE":
        _require_public_mutation(request)

    upstream_path = base_path
    if tail_parts:
        upstream_path += "/" + "/".join(quote(part, safe="") for part in tail_parts)
    return await _adk_request(
        request.method,
        upstream_path,
        query=request.url.query,
        accept=request.headers.get("accept"),
        assertion=assertion,
    )


async def _authenticated_adk_run(request: Request, *, sse: bool) -> Response:
    identity, assertion = _adk_identity(request)
    _reject_selector_headers(request)
    _reject_query(request)
    _require_public_mutation(request)
    body = await _json_body(request)
    _reject_untrusted_selector_tokens(body)
    allowed = {
        "appName",
        "userId",
        "sessionId",
        "newMessage",
        "streaming",
        "stateDelta",
        "invocationId",
    }
    if set(body) - allowed:
        raise HTTPException(status_code=422, detail="unsupported ADK run field")
    if body.get("appName") != "payment_user_agent":
        raise HTTPException(status_code=403, detail="only payment_user_agent is public")
    if body.get("userId") != identity.subject:
        raise HTTPException(
            status_code=403,
            detail="run user does not match the authenticated subject",
        )
    _safe_adk_id(body.get("sessionId"), "session ID")
    state_delta = body.get("stateDelta") or {}
    _reject_identity_selectors(state_delta)
    body["stateDelta"] = {
        **state_delta,
        ADK_IDENTITY_STATE_KEY: assertion,
    }
    return await _adk_request(
        "POST",
        "/run_sse" if sse else "/run",
        payload=body,
        accept="text/event-stream" if sse else request.headers.get("accept"),
        assertion=assertion,
    )


@app.post("/run")
async def authenticated_adk_run(request: Request) -> Response:
    return await _authenticated_adk_run(request, sse=False)


@app.post("/run_sse")
async def authenticated_adk_run_sse(request: Request) -> Response:
    return await _authenticated_adk_run(request, sse=True)


@app.api_route(
    "/mediation-api/{workflow_path:path}", methods=["GET", "POST", "DELETE"]
)
async def authenticated_workflow_api(request: Request, workflow_path: str) -> Response:
    """Validate the public workflow boundary before the loopback API."""

    identity, assertion = _adk_identity(request)
    del identity  # The signed assertion is the sole owner input to workflow_api.
    _reject_selector_headers(request)
    _reject_query(request)
    routes = {
        ("GET", "ready"): "/ready",
        ("GET", "v1/view"): "/v1/view",
        ("POST", "v1/turns"): "/v1/turns",
    }
    path = routes.get((request.method, workflow_path))
    if path is None:
        raise HTTPException(status_code=403, detail="workflow selector path is not public")
    payload: dict[str, Any] | None = None
    if request.method != "GET":
        _require_public_mutation(request)
        payload = await _json_body(request)
        _reject_untrusted_selector_tokens(payload)
        _reject_identity_selectors(payload)
    return await _workflow_request(
        request,
        path,
        assertion=assertion,
        payload=payload,
    )


@app.get("/auth/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
