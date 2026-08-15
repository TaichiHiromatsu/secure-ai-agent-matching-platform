"""Firebase login sessions and short-lived signed upstream identities."""

from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
import google.auth.transport.requests
import google.oauth2.id_token
from pydantic import BaseModel, ConfigDict, StrictStr

from secure_mediation_agent.ap2.keys import load_role_key
from secure_mediation_agent.identity import issue_identity_assertion


app = FastAPI()
FIREBASE_PROJECT_ID = "mediation-a2a-platform"
SESSION_COOKIE = "__Host-payment-session"
CSRF_COOKIE = "__Host-payment-csrf"
FIREBASE_CONFIG_FILE = Path(__file__).with_name("firebase-config.json")
DEV_MODE = os.environ.get("DEV_MODE", "false").lower() == "true"
if DEV_MODE and os.environ.get("APP_ENV") != "local":
    raise RuntimeError("DEV_MODE=true is permitted only when APP_ENV=local")

http_request = google.auth.transport.requests.Request()


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
    cookie = request.cookies.get(CSRF_COOKIE)
    header = request.headers.get("x-csrf-token")
    if not cookie or not header or not secrets.compare_digest(cookie, header):
        raise HTTPException(status_code=403, detail="CSRF validation failed")


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


class InternalIdentityRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    subject: StrictStr


class SessionRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    id_token: StrictStr


@app.get("/auth/csrf")
async def csrf_token() -> Response:
    token = secrets.token_urlsafe(32)
    response = JSONResponse({"csrfToken": token})
    response.set_cookie(
        CSRF_COOKIE,
        token,
        secure=True,
        httponly=False,
        samesite="strict",
        path="/",
        max_age=600,
    )
    return response


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


@app.post("/auth/internal/identity")
async def internal_identity(body: InternalIdentityRequest, request: Request):
    """Issue an adapter assertion on the loopback-only auth service port."""

    if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
        return Response(status_code=404)
    key_dir = os.environ.get("AP2_DEMO_KEY_DIR")
    if not key_dir or not body.subject:
        return Response(status_code=503)
    assertion = issue_identity_assertion(
        load_role_key(key_dir, "service_auth"), subject=body.subject
    )
    return {"assertion": assertion}


@app.get("/auth/verify")
async def verify_token(request: Request) -> Response:
    """Verify the server-set Firebase cookie for nginx ``auth_request``."""

    if DEV_MODE:
        return _identity_response("demo-local-user")
    session_cookie = request.cookies.get(SESSION_COOKIE)
    if not session_cookie:
        return Response(status_code=401)
    try:
        claims = _verified_firebase_claims(session_cookie)
        return _identity_response(str(claims["sub"]))
    except Exception:
        return Response(status_code=401)


@app.get("/auth/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
