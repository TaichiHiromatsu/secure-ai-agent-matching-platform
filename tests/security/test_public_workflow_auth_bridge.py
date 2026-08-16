from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient
import pytest

import deploy.auth.verify as auth
from secure_mediation_agent.ap2.keys import DemoKeySet
from secure_mediation_agent.identity import issue_identity_assertion


pytestmark = pytest.mark.security


@pytest.fixture
def workflow_bridge(monkeypatch):
    keys = DemoKeySet.generate_for_test()
    assertion = issue_identity_assertion(
        keys.service_auth, subject="demo-local-user"
    )
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.method == "GET" and request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.method == "GET" and request.url.path == "/v1/view":
            return httpx.Response(200, json=None)
        if request.method == "POST" and request.url.path == "/v1/turns":
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "schemaVersion": "mediation-turn-response/1",
                    "requestId": body["requestId"],
                },
            )
        raise AssertionError(
            f"unexpected workflow request: {request.method} {request.url}"
        )

    monkeypatch.setenv("AP2_DEMO_KEY_DIR", "/test-keys")
    monkeypatch.setattr(
        auth, "load_role_key", lambda directory, role: keys.service_auth
    )
    auth.app.state.workflow_transport = httpx.MockTransport(handler)
    client = TestClient(auth.app, base_url="https://demo.example")
    client.cookies.set(auth.CSRF_COOKIE, "csrf-token")
    try:
        yield client, assertion, observed
    finally:
        auth.app.state.workflow_transport = None


def _headers(assertion: str) -> dict[str, str]:
    return {
        "X-Verified-Identity": assertion,
        "Origin": "https://demo.example",
        "X-CSRF-Token": "csrf-token",
        "Idempotency-Key": "turn-request-0001",
        "X-Request-ID": "turn-request-0001",
    }


def _turn_body() -> dict[str, object]:
    return {
        "schemaVersion": "mediation-turn-request/1",
        "requestId": "turn-request-0001",
        "message": {"parts": [{"kind": "text", "text": "hello"}]},
        "selectionToken": None,
    }


def test_workflow_bridge_forwards_only_signed_identity_and_allowed_headers(
    workflow_bridge,
):
    client, assertion, observed = workflow_bridge
    ready = client.get(
        "/mediation-api/ready", headers={"X-Verified-Identity": assertion}
    )
    turn = client.post(
        "/mediation-api/v1/turns",
        headers=_headers(assertion),
        json=_turn_body(),
    )
    view = client.get(
        "/mediation-api/v1/view",
        headers={"X-Verified-Identity": assertion},
    )

    assert (ready.status_code, turn.status_code, view.status_code) == (200, 200, 200)
    assert [request.url.path for request in observed] == [
        "/ready",
        "/v1/turns",
        "/v1/view",
    ]
    for request in observed:
        assert request.headers["x-verified-identity"] == assertion
        assert "cookie" not in request.headers


@pytest.mark.parametrize(
    "path,method,body,extra_headers",
    [
        (
            "/mediation-api/v1/turns",
            "POST",
            {**_turn_body(), "subject": "victim-user"},
            {},
        ),
        (
            "/mediation-api/v1/turns",
            "POST",
            {
                **_turn_body(),
                "message": {
                    "parts": [
                        {
                            "kind": "text",
                            "text": "hello",
                            "tenantId": "victim-tenant",
                        }
                    ]
                },
            },
            {},
        ),
        (
            "/mediation-api/v1/turns",
            "POST",
            {**_turn_body(), "unknownSelector": "victim-user"},
            {},
        ),
        (
            "/mediation-api/v1/turns",
            "POST",
            {**_turn_body(), "selectionToken": "attacker-selected-token"},
            {},
        ),
        (
            "/mediation-api/v1/view?unknownSelector=victim-user",
            "GET",
            None,
            {},
        ),
        (
            "/mediation-api/v1/view/victim-user",
            "GET",
            None,
            {},
        ),
        (
            "/mediation-api/v1/view",
            "GET",
            None,
            {"X-Subject": "victim-user"},
        ),
    ],
)
def test_workflow_selectors_are_rejected_before_proxy(
    workflow_bridge, path, method, body, extra_headers
):
    client, assertion, observed = workflow_bridge
    headers = {"X-Verified-Identity": assertion, **extra_headers}
    if method == "POST":
        headers.update(_headers(assertion))
    response = client.request(method, path, headers=headers, json=body)
    assert response.status_code == 403
    assert observed == []


def test_workflow_mutation_requires_origin_and_csrf_before_proxy(
    workflow_bridge,
):
    client, assertion, observed = workflow_bridge
    no_csrf = client.post(
        "/mediation-api/v1/turns",
        headers={
            "X-Verified-Identity": assertion,
            "Idempotency-Key": "turn-request-0001",
        },
        json=_turn_body(),
    )
    wrong_origin = client.post(
        "/mediation-api/v1/turns",
        headers={**_headers(assertion), "Origin": "https://evil.example"},
        json=_turn_body(),
    )
    assert (no_csrf.status_code, wrong_origin.status_code) == (403, 403)
    assert observed == []


def test_authenticated_browser_bootstrap_issues_fixed_subject_and_csrf(
    workflow_bridge,
):
    client, assertion, _ = workflow_bridge
    response = client.get(
        "/auth/browser-bootstrap",
        headers={"X-Verified-Identity": assertion},
    )
    assert response.status_code == 200
    assert response.json()["subject"] == "demo-local-user"
    assert response.json()["csrfToken"]
    assert auth.CSRF_COOKIE in response.headers["set-cookie"]
    assert response.headers["cache-control"] == "no-store"
