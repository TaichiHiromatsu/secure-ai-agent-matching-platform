from __future__ import annotations

from pathlib import Path

import pytest

from secure_mediation_agent.payment_profiles.x402_v01 import OfficialX402V01Profile


pytestmark = pytest.mark.security
ROOT = Path(__file__).resolve().parents[2]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_llm_facing_payment_app_is_thin_keyless_and_repository_free() -> None:
    source = _text("secure_mediation_agent/agent.py")
    public_source = _text("payment_user_agent/agent.py")
    assert "ap2.keys" not in source
    assert "WorkflowRepository" not in source
    assert "payment_marketplace" not in source
    assert "PaymentMediatorClient" not in source
    assert "BaseAgent" in source
    assert "root_agent" not in source
    assert 'name="payment_user_agent"' in public_source
    assert "PaymentWorkflowAdapter" in public_source


def test_nginx_has_one_authenticated_payment_surface_and_strips_identity() -> None:
    source = _text("deploy/nginx.conf")
    assert "location /mediation-api/" in source
    assert "auth_request_set $verified_identity $upstream_http_x_verified_identity" in source
    assert "proxy_set_header X-Verified-Identity $verified_identity" in source
    assert "location ^~ /payment/ { return 404; }" in source
    assert "location ^~ /paid-agent/ { return 404; }" in source
    assert "payment_marketplace" not in source


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


def test_current_cloud_run_paid_deployment_is_hard_blocked() -> None:
    lines = _text("deploy/deploy-cloudrun.sh").splitlines()
    exit_index = next(index for index, line in enumerate(lines) if line.strip() == "exit 2")
    deploy_index = next(index for index, line in enumerate(lines) if "gcloud run deploy" in line)
    assert exit_index < deploy_index


def test_official_x402_adapter_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="NOT READY"):
        OfficialX402V01Profile()
