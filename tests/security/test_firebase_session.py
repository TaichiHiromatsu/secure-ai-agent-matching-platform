from __future__ import annotations

import time

from fastapi import Response
from fastapi.testclient import TestClient
import pytest

import deploy.auth.verify as auth


pytestmark = pytest.mark.security


def _claims() -> dict[str, object]:
    return {
        "sub": "firebase-user",
        "aud": "mediation-a2a-platform",
        "iss": "https://securetoken.google.com/mediation-a2a-platform",
        "exp": int(time.time()) + 900,
    }


def test_same_origin_session_exchange_sets_host_only_http_only_cookie(monkeypatch) -> None:
    monkeypatch.setattr(auth, "_verified_firebase_claims", lambda token: _claims())
    with TestClient(auth.app, base_url="https://demo.example") as client:
        csrf = client.get("/auth/csrf").json()["csrfToken"]
        response = client.post(
            "/auth/session",
            headers={"Origin": "https://demo.example", "X-CSRF-Token": csrf},
            json={"id_token": "firebase-id-token"},
        )
    assert response.status_code == 200
    cookie = response.headers["set-cookie"]
    assert "__Host-payment-session=" in cookie
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=strict" in cookie


def test_session_exchange_rejects_cross_origin_and_missing_csrf(monkeypatch) -> None:
    monkeypatch.setattr(auth, "_verified_firebase_claims", lambda token: _claims())
    with TestClient(auth.app, base_url="https://demo.example") as client:
        csrf = client.get("/auth/csrf").json()["csrfToken"]
        cross_origin = client.post(
            "/auth/session",
            headers={"Origin": "https://evil.example", "X-CSRF-Token": csrf},
            json={"id_token": "token"},
        )
        missing_csrf = client.post(
            "/auth/session",
            headers={"Origin": "https://demo.example"},
            json={"id_token": "token"},
        )
    assert cross_origin.status_code == 403
    assert missing_csrf.status_code == 403


def test_auth_request_reverifies_cookie_and_logout_clears_it(monkeypatch) -> None:
    calls: list[str] = []

    def verify(token: str) -> dict[str, object]:
        calls.append(token)
        return _claims()

    monkeypatch.setattr(auth, "_verified_firebase_claims", verify)
    monkeypatch.setattr(auth, "_identity_response", lambda subject: Response(status_code=200))
    with TestClient(auth.app, base_url="https://demo.example") as client:
        csrf = client.get("/auth/csrf").json()["csrfToken"]
        client.cookies.set(auth.SESSION_COOKIE, "server-cookie")
        verified = client.get("/auth/verify")
        logout = client.post(
            "/auth/logout",
            headers={"Origin": "https://demo.example", "X-CSRF-Token": csrf},
        )
    assert verified.status_code == 200
    assert calls == ["server-cookie"]
    assert logout.status_code == 204
    assert "Max-Age=0" in logout.headers["set-cookie"]


def test_auth_request_validates_origin_and_csrf_for_original_mutation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(auth, "_verified_firebase_claims", lambda token: _claims())
    monkeypatch.setattr(
        auth,
        "_identity_response",
        lambda subject: Response(
            status_code=200,
            headers={"X-Verified-Identity": f"signed:{subject}"},
        ),
    )
    with TestClient(auth.app, base_url="https://demo.example") as client:
        csrf = client.get("/auth/csrf").json()["csrfToken"]
        client.cookies.set(auth.SESSION_COOKIE, "server-cookie")
        valid = client.get(
            "/auth/verify",
            headers={
                "X-Original-Method": "POST",
                "Origin": "https://demo.example",
                "X-CSRF-Token": csrf,
            },
        )
        missing = client.get(
            "/auth/verify",
            headers={
                "X-Original-Method": "POST",
                "Origin": "https://demo.example",
            },
        )
        wrong_origin = client.get(
            "/auth/verify",
            headers={
                "X-Original-Method": "DELETE",
                "Origin": "https://evil.example",
                "X-CSRF-Token": csrf,
            },
        )
    assert valid.status_code == 200
    assert valid.headers["x-csrf-validated"] == "1"
    assert (missing.status_code, wrong_origin.status_code) == (403, 403)


def test_firebase_verifier_binds_project_issuer_and_subject(monkeypatch) -> None:
    bad = _claims()
    bad["aud"] = "another-project"
    monkeypatch.setattr(
        auth.google.oauth2.id_token,
        "verify_firebase_token",
        lambda *args, **kwargs: bad,
    )
    with pytest.raises(ValueError, match="mismatch"):
        auth._verified_firebase_claims("token")
