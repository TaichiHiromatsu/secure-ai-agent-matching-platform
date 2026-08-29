from __future__ import annotations

import json
from pathlib import Path

import pytest

from secure_mediation_agent.payment_profiles.x402_v01 import OfficialX402V01Profile


pytestmark = pytest.mark.security
ROOT = Path(__file__).resolve().parents[2]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_llm_facing_payment_app_is_thin_keyless_and_repository_free() -> None:
    internal_source = _text("secure_mediation_agent/agent.py")
    public_source = _text("payment_user_agent/agent.py")
    for source in (internal_source, public_source):
        assert "ap2.keys" not in source
        assert "WorkflowRepository" not in source
        assert "payment_marketplace" not in source
        assert "PaymentMediatorClient" not in source

    assert "SecureMediationAdapter" in internal_source
    assert 'name="secure_mediator"' in internal_source
    assert 'name="payment_user_agent"' not in internal_source
    assert "SecureMediationAdapter" in public_source
    assert 'name="payment_user_agent"' in public_source
    assert "PaymentWorkflowAdapter" not in public_source


def test_nginx_has_exact_authenticated_surfaces_and_strips_identity() -> None:
    source = _text("deploy/nginx.conf")
    assert "location = /mediation-api/ready" in source
    assert "location = /mediation-api/v1/turns" in source
    assert "location = /mediation-api/v1/view" in source
    assert "location = /list-apps" in source
    assert "location = /dev-ui/" in source
    assert "location = /run" in source
    assert "location = /run_sse" in source
    assert "^/apps/payment_user_agent/users/" in source
    assert "^/(?:main|polyfills|chunk|styles)-[A-Z0-9]+" in source
    assert "^/dev-ui/(?:main|polyfills|chunk|styles)-[A-Z0-9]+" in source
    assert "^/dev-ui/assets/(?:audio-processor" in source
    assert "location = /adk_favicon.svg" in source
    for asset in ("audio-processor", "ADK-512-color", "config/runtime-config"):
        assert asset in source
    assert "auth_request_set $verified_identity $upstream_http_x_verified_identity" in source
    assert "proxy_set_header X-Verified-Identity $verified_identity" in source
    assert "$has_client_identity_selector" in source
    assert "proxy_set_header X-Original-Method $request_method" in source
    assert "proxy_set_header X-Verified-CSRF $verified_csrf" in source
    assert "proxy_set_header X-Verified-CSRF-Cookie $csrf_cookie_value" in source
    assert source.count("proxy_pass_request_headers off;") >= 12
    for path in ("store", "api", "ws", "a2a", "v1", "internal", "payment", "paid-agent"):
        assert f"location = /{path} {{ return 404; }}" in source
        assert f"location ^~ /{path}/ {{ return 404; }}" in source
    assert "proxy_set_header Upgrade" not in source
    assert "trusted_agent_store" not in source
    assert "external_agents" not in source
    assert "payment_marketplace" not in source

    for marker in (
        'location ~ "^/apps/payment_user_agent/users/',
        "location = /run {",
        "location = /run_sse {",
    ):
        block = source.split(marker, maxsplit=1)[1].split(
            "\n        location", maxsplit=1
        )[0]
        assert "auth_request_set $verified_identity" in block
        assert "auth_request_set $verified_csrf" in block
        assert "proxy_set_header X-Verified-Identity $verified_identity" in block
        assert "proxy_set_header Cookie" not in block
        assert "proxy_pass http://auth_service" in block

    public_upstreams = source.split(
        "# Sole authenticated ADK application", maxsplit=1
    )[1]
    assert "proxy_set_header Cookie" not in public_upstreams
    assert source.count("proxy_set_header Cookie $http_cookie;") == 1
    assert "proxy_pass http://auth_service/mediation-api/ready" in source
    assert "proxy_pass http://auth_service/mediation-api/v1/turns" in source
    assert "proxy_pass http://auth_service/mediation-api/v1/view" in source
    assert "sub_filter '<head>'" in source

    auth = _text("deploy/auth/verify.py")
    assert "/auth/internal/identity" not in auth
    assert "ADK_IDENTITY_STATE_KEY" in auth
    assert "verify_identity_assertion" in auth
    assert "_require_public_mutation" in auth
    assert "_reject_identity_selectors" in auth
    assert "/auth/browser-bootstrap" in auth


def test_only_payment_user_agent_is_copied_into_adk_discovery() -> None:
    dockerfile = _text("Dockerfile")
    supervisor = _text("deploy/supervisord.conf")
    assert "COPY payment_user_agent /app/payment-apps/payment_user_agent" in dockerfile
    assert "COPY secure_mediation_agent /app/internal/secure_mediation_agent" in dockerfile
    assert "adk web /app/payment-apps" in supervisor
    assert "adk web /app/internal" not in supervisor
    cli = _text("user-agent/payment_cli.py")
    assert "127.0.0.1:8004" not in cli
    assert "/mediation-api" in cli


def test_supervisor_keeps_every_backend_on_loopback_or_without_listener() -> None:
    supervisor = _text("deploy/supervisord.conf")
    for port in range(8000, 8006):
        assert f"127.0.0.1 --port {port}" in supervisor or f"127.0.0.1:{port}" in supervisor
    assert "--host 0.0.0.0" not in supervisor
    worker = supervisor.split("[program:workflow_outbox_worker]", maxsplit=1)[1]
    assert "--port" not in worker


def test_public_app_metadata_and_registry_match_release_runtime() -> None:
    public_card = json.loads(_text("payment_user_agent/agent.json"))
    internal_card = json.loads(_text("secure_mediation_agent/agent.json"))
    assert public_card["name"] == "payment_user_agent"
    assert public_card["capabilities"]["deploymentDurability"] == "NOT PROVIDED"
    assert public_card["capabilities"]["ephemeralCloudRunDemo"] is True
    assert public_card["capabilities"]["officialX402"] is False
    assert internal_card["name"] == "secure_mediation_workflow"
    assert internal_card["capabilities"]["deploymentDurability"] == "NOT PROVIDED"

    registry = {
        item["id"]: item
        for item in json.loads(_text("trusted_agent_store/data/agents/registered-agents.json"))
    }
    free = registry["agent-002"]
    assert free["name"] == "hotel_agent"
    assert free["agent_card_url"] == (
        "http://127.0.0.1:8002/a2a/hotel_agent/.well-known/agent-card.json"
    )
    assert free["endpoint_url"] == "http://127.0.0.1:8002/a2a/hotel_agent"
    assert free.get("capabilities", {}) == {}
    assert {skill["id"] for skill in free["skills"]} >= {"hotel_search", "hotel_booking"}

    paid = registry["agent-005"]
    extension = paid["capabilities"]["extensions"][0]
    # The registry keeps its underscore alias, while the published A2A Card
    # uses the protocol-facing hyphenated name.
    assert paid["name"] == "paid_booking_agent"
    assert paid["agent_card_url"] == "http://127.0.0.1:8005/.well-known/agent-card.json"
    assert paid["endpoint_url"] == "http://127.0.0.1:8005/a2a"
    assert extension["uri"] == "urn:secure-a2a:extensions:x402-wire-simulation:v1"
    assert extension["params"]["profile"] == "x402-wire-simulation/1"
    assert extension["params"]["canonicalAgentId"] == "agent-005"
    assert extension["params"]["cardName"] == "paid-booking-agent"
    assert extension["params"]["identifierMappingVersion"] == "paid-booking-identifiers/v1"
    assert {skill["id"] for skill in paid["skills"]} == {"paid_booking"}


def test_login_uses_server_owned_session_and_payment_root_redirect() -> None:
    login = _text("deploy/auth/login.html")
    auth = _text("deploy/auth/verify.py")
    assert "document.cookie" not in login
    assert "'/auth/session'" in login
    assert "'/?app=payment_user_agent'" in login
    assert "httponly=True" in auth
    assert 'samesite="strict"' in auth
    assert 'FIREBASE_PROJECT_ID = "mediation-a2a-platform"' in auth
    assert "verify_firebase_token" in auth
    client = _text("secure_mediation_agent/workflow/client.py")
    verifier = _text("scripts/verify_ap2_x402_runtime.py")
    assert 'SESSION_COOKIE_NAME = "__Host-payment-session"' in client
    assert 'SESSION_COOKIE_NAME = "__Host-payment-session"' in verifier
    assert 'f"session={self.session_cookie}"' not in client

    browser_bootstrap = _text("deploy/auth/csrf-bootstrap.js")
    assert "document.cookie" not in browser_bootstrap
    assert "'/auth/browser-bootstrap'" in browser_bootstrap
    assert "X-CSRF-Token" in browser_bootstrap
    assert "value.userId = subject" in browser_bootstrap
    dockerfile = _text("Dockerfile")
    assert "COPY deploy/auth/csrf-bootstrap.js" in dockerfile


def test_current_cloud_run_paid_deployment_is_hard_blocked() -> None:
    lines = _text("deploy/deploy-cloudrun.sh").splitlines()
    exit_index = next(index for index, line in enumerate(lines) if line.strip() == "exit 2")
    deploy_index = next(index for index, line in enumerate(lines) if "gcloud run deploy" in line)
    assert exit_index < deploy_index


def test_payment_demo_deployments_pin_the_ephemeral_memory_store_profile() -> None:
    expected = (
        "EPHEMERAL_CLOUD_RUN_DEMO=true,MEDIATION_STORE_MODE=memory,"
        "APP_ENV=ephemeral-demo,DEV_MODE=false,"
        "GOOGLE_GENAI_USE_VERTEXAI=true,"
        "GOOGLE_CLOUD_PROJECT=gen-lang-client-0585901015,"
        "GOOGLE_CLOUD_LOCATION=global"
    )
    for path in (
        "deploy/deploy-payment-demo-cloudrun.sh",
        "deploy/update-payment-demo-cloudrun.sh",
    ):
        source = _text(path)
        assert f'readonly DEPLOY_ENV_VARS="{expected}"' in source
    update = _text("deploy/update-payment-demo-cloudrun.sh")
    assert 'env("EPHEMERAL_CLOUD_RUN_DEMO") == ["true"]' in update
    assert 'env("MEDIATION_STORE_MODE") == ["memory"]' in update
    assert 'env("GOOGLE_GENAI_USE_VERTEXAI") == ["true"]' in update
    assert 'env("GOOGLE_CLOUD_PROJECT") == ["gen-lang-client-0585901015"]' in update
    assert 'env("GOOGLE_CLOUD_LOCATION") == ["global"]' in update
    assert "| length) == 7" in update
    assert ".readiness.checks.vertexAdcConfiguration" in update
    assert '.publicDurabilityProfile == "ephemeral-demo"' in update
    assert 'mode:"memory", durabilityProfile:"ephemeral-demo"' in update


def test_cloud_judge_uses_stable_vertex_allowlisted_model() -> None:
    source = _text("secure_mediation_agent/security/custom_judge.py")
    assert "model='gemini-2.5-flash'" in source
    assert "gemini-3-flash-preview" not in source


def test_official_x402_adapter_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="NOT READY"):
        OfficialX402V01Profile()
