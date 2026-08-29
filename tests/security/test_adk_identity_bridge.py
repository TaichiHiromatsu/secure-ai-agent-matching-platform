from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient
import pytest

import deploy.auth.verify as auth
from secure_mediation_agent.ap2.keys import DemoKeySet
from secure_mediation_agent.identity import (
    ADK_IDENTITY_STATE_KEY,
    issue_identity_assertion,
    verify_identity_assertion,
)


pytestmark = pytest.mark.security


@pytest.fixture
def bridge(monkeypatch):
    keys = DemoKeySet.generate_for_test()
    assertion = issue_identity_assertion(keys.service_auth, subject="demo-local-user")
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        body = json.loads(request.content) if request.content else {}
        if request.url.path.endswith("/sessions") and request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "id": body["sessionId"],
                    "appName": "payment_user_agent",
                    "userId": request.url.path.split("/users/", 1)[1].split("/", 1)[0],
                    "state": body["state"],
                    "events": [],
                },
            )
        if request.url.path == "/run":
            return httpx.Response(
                200,
                json=[
                    {
                        "content": {"role": "model", "parts": [{"text": "ok"}]},
                        "actions": {"stateDelta": body["stateDelta"]},
                    }
                ],
            )
        raise AssertionError(f"unexpected upstream request: {request.method} {request.url}")

    monkeypatch.setenv("AP2_DEMO_KEY_DIR", "/test-keys")
    monkeypatch.setattr(auth, "load_role_key", lambda directory, role: keys.service_auth)
    client = TestClient(auth.app, base_url="https://demo.example")
    client.cookies.set(auth.CSRF_COOKIE, "csrf-token")
    auth.app.state.adk_transport = httpx.MockTransport(handler)
    try:
        yield client, assertion, keys, observed
    finally:
        auth.app.state.adk_transport = None


def _mutation_headers(assertion: str) -> dict[str, str]:
    return {
        "X-Verified-Identity": assertion,
        "Origin": "https://demo.example",
        "X-CSRF-Token": "csrf-token",
    }


def test_bridge_binds_session_and_run_to_signed_subject_without_leaking_assertion(bridge):
    client, assertion, keys, observed = bridge
    headers = _mutation_headers(assertion)

    created = client.post(
        "/apps/payment_user_agent/users/demo-local-user/sessions",
        headers=headers,
        json={"sessionId": "session-1", "state": {"theme": "dark"}},
    )
    assert created.status_code == 200
    assert created.json()["userId"] == "demo-local-user"
    assert created.json()["state"] == {"theme": "dark"}

    upstream_create = json.loads(observed[0].content)
    injected = upstream_create["state"][ADK_IDENTITY_STATE_KEY]
    verified = verify_identity_assertion(injected, keys.service_auth)
    assert verified.subject == "demo-local-user"
    assert "/users/demo-local-user/sessions" in observed[0].url.path

    run = client.post(
        "/run",
        headers=headers,
        json={
            "appName": "payment_user_agent",
            "userId": "demo-local-user",
            "sessionId": "session-1",
            "newMessage": {"role": "user", "parts": [{"text": "hello"}]},
        },
    )
    assert run.status_code == 200
    assert ADK_IDENTITY_STATE_KEY not in run.text
    assert assertion not in run.text
    upstream_run = json.loads(observed[1].content)
    assert upstream_run["userId"] == "demo-local-user"
    assert upstream_run["stateDelta"][ADK_IDENTITY_STATE_KEY] == assertion
    assert all("cookie" not in request.headers for request in observed)


@pytest.mark.parametrize(
    "path,payload",
    [
        (
            "/apps/payment_user_agent/users/victim-user/sessions",
            {"sessionId": "session-victim", "state": {}},
        ),
        (
            "/run",
            {
                "appName": "payment_user_agent",
                "userId": "victim-user",
                "sessionId": "session-1",
                "newMessage": {"role": "user", "parts": [{"text": "hello"}]},
            },
        ),
        (
            "/apps/payment_user_agent/users/demo-local-user/sessions",
            {
                "sessionId": "session-2",
                "state": {
                    "preferences": {
                        "verifiedIdentity": {
                            "subject": "victim-user",
                            "tenantId": "victim-tenant",
                        }
                    }
                },
            },
        ),
        (
            "/run",
            {
                "appName": "payment_user_agent",
                "userId": "demo-local-user",
                "sessionId": "session-1",
                "newMessage": {"role": "user", "parts": [{"text": "hello"}]},
                "stateDelta": {"nested": {"tenantId": "victim-tenant"}},
            },
        ),
    ],
)
def test_bridge_rejects_browser_identity_state(bridge, path, payload):
    client, assertion, _, observed = bridge
    before = len(observed)
    response = client.post(
        path,
        headers=_mutation_headers(assertion),
        json=payload,
    )
    assert response.status_code == 403
    assert len(observed) == before


def test_direct_bridge_and_unsigned_identity_fail_closed(bridge):
    client, _, _, observed = bridge
    missing = client.post(
        "/apps/payment_user_agent/users/demo-local-user/sessions",
        json={"sessionId": "session-3", "state": {}},
    )
    invalid = client.post(
        "/run",
        headers={"X-Verified-Identity": "not-a-signed-assertion"},
        json={
            "appName": "payment_user_agent",
            "userId": "demo-local-user",
            "sessionId": "session-3",
            "newMessage": {"role": "user", "parts": [{"text": "hello"}]},
        },
    )
    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert observed == []


@pytest.mark.parametrize(
    "headers",
    [
        {"X-Verified-Identity": "ASSERTION"},
        {
            "X-Verified-Identity": "ASSERTION",
            "Origin": "https://evil.example",
            "X-CSRF-Token": "csrf-token",
        },
    ],
)
def test_mutation_requires_exact_origin_and_csrf_before_upstream(
    bridge, headers
):
    client, assertion, _, observed = bridge
    headers = {
        name: assertion if value == "ASSERTION" else value
        for name, value in headers.items()
    }
    response = client.post(
        "/apps/payment_user_agent/users/demo-local-user/sessions",
        headers=headers,
        json={"sessionId": "csrf-session", "state": {}},
    )
    assert response.status_code == 403
    assert observed == []


def test_query_selector_is_rejected_before_upstream(bridge):
    client, assertion, _, observed = bridge
    response = client.get(
        "/apps/payment_user_agent/users/demo-local-user/sessions?subject=victim",
        headers={"X-Verified-Identity": assertion},
    )
    assert response.status_code == 403
    assert observed == []


@pytest.mark.parametrize(
    "selector_input",
    [
        {"unknownSelector": "victim-user"},
        {"selectionToken": "attacker-selected-token"},
    ],
)
def test_run_rejects_unknown_selector_inputs_before_upstream(
    bridge, selector_input
):
    client, assertion, _, observed = bridge
    response = client.post(
        "/run",
        headers=_mutation_headers(assertion),
        json={
            "appName": "payment_user_agent",
            "userId": "demo-local-user",
            "sessionId": "session-1",
            "newMessage": {"role": "user", "parts": [{"text": "hello"}]},
            **selector_input,
        },
    )
    assert response.status_code == 403
    assert observed == []
