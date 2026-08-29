from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from payment_user_agent.agent import root_agent
from secure_mediation_agent.ap2.keys import ROLE_KIDS, generate_key
from secure_mediation_agent.identity import (
    ADK_IDENTITY_STATE_KEY,
    issue_identity_assertion,
)
from secure_mediation_agent.mediation.adk_adapter import SecureMediationAdapter
from secure_mediation_agent.mediation.authority import HttpMediationAuthority
from secure_mediation_agent.mediation.errors import MediationError


def _context(*, assertion=None, state=None, user_id="alice", session_id="session-1"):
    session_state = dict(state or {})
    if assertion is not None:
        session_state[ADK_IDENTITY_STATE_KEY] = assertion
    return SimpleNamespace(
        session=SimpleNamespace(
            id=session_id,
            user_id=user_id,
            state=session_state,
        )
    )


@pytest.fixture
def signed_identity(tmp_path, monkeypatch):
    key = generate_key(ROLE_KIDS["service_auth"])
    key_path = tmp_path / "service_auth.jwk"
    key_path.write_text(key.export(private_key=True), encoding="utf-8")
    os.chmod(key_path, 0o600)
    monkeypatch.setenv("AP2_DEMO_KEY_DIR", str(tmp_path))
    return issue_identity_assertion(key, subject="alice"), key


def test_public_root_is_secure_mediation_session_router():
    assert isinstance(root_agent, SecureMediationAdapter)
    assert root_agent.name == "payment_user_agent"
    assert isinstance(root_agent._resolved_authority(), HttpMediationAuthority)
    assert not hasattr(root_agent, "_controller")


def test_adapter_accepts_only_signed_ingress_identity(signed_identity):
    assertion, _ = signed_identity
    scope = SecureMediationAdapter._scope(
        _context(assertion=assertion)
    )
    assert scope.key == ("alice", "demo-tenant", "session-1")

    with pytest.raises(MediationError, match="検証済み"):
        SecureMediationAdapter._scope(_context())
    with pytest.raises(MediationError, match="検証済み"):
        SecureMediationAdapter._scope(
            _context(assertion=assertion, user_id="mallory")
        )
    with pytest.raises(MediationError, match="検証済み"):
        SecureMediationAdapter._scope(
            _context(
                state={
                    "verifiedIdentity": {
                        "subject": "alice",
                        "tenantId": "demo-tenant",
                    }
                }
            )
        )


def test_adapter_rejects_forged_signed_identity(signed_identity):
    _, key = signed_identity
    other_key = generate_key(ROLE_KIDS["service_auth"])
    forged = issue_identity_assertion(other_key, subject="alice")
    with pytest.raises(MediationError, match="一致"):
        SecureMediationAdapter._scope(_context(assertion=forged))


def test_adapter_displays_exact_canonical_approval_target():
    display = SecureMediationAdapter._approval_display(
        SimpleNamespace(
            approval_target={
                "z": "last",
                "a": {"amount": "100", "currency": "JPY"},
            },
            approval_target_digest="sha256:target",
        )
    )

    assert (
        '承認対象 (canonical JSON): {"a":{"amount":"100",'
        '"currency":"JPY"},"z":"last"}'
    ) in display
    assert "承認対象digest: sha256:target" in display
    assert "メッセージ全体を完全一致「承認」として送信してください。" in display


def test_adapter_warns_only_for_ephemeral_demo_views():
    base = {
        "message": "承認内容を確認してください。",
        "state": SimpleNamespace(value="WaitingForPlanApproval"),
        "plan_ref": "sha256:plan",
        "step_ref": "sha256:step",
        "task_ref": None,
        "approval_target": None,
        "approval_target_digest": None,
    }
    ephemeral = SecureMediationAdapter._reply(
        SimpleNamespace(**base, durability_profile="ephemeral-demo")
    )
    durable = SecureMediationAdapter._reply(
        SimpleNamespace(**base, durability_profile="local-durable")
    )

    notice = "デモ環境: 再起動すると進行中の状態は失われます（耐久性保証なし）。"
    assert ephemeral.startswith(notice)
    assert notice not in durable
