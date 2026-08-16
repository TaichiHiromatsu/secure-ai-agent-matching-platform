from __future__ import annotations

import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest

from secure_mediation_agent.mediation.canonical import canonical_bytes, canonical_digest
from secure_mediation_agent.mediation.errors import MediationError, SecurityBlocked
from secure_mediation_agent.mediation.models import (
    MediationContinuation,
    MediationPlan,
    MediationSession,
    MediationState,
    MediationStep,
    OwnerScope,
    PaymentRequirementSnapshot,
    PendingAction,
    RemoteTaskSnapshot,
    SelectedAgentSnapshot,
    SubjectScope,
    utc_now,
)
from secure_mediation_agent.mediation.persistence import (
    SqliteMediationStore,
    _session_projection,
    load_mediation_store_key,
)
from secure_mediation_agent.mediation.persistence_models import build_local_durable_view
from secure_mediation_agent.workflow.migrations import DatabasePaths
from secure_mediation_agent.workflow.repository import WorkflowRepository


def _repository(tmp_path: Path) -> WorkflowRepository:
    return WorkflowRepository(
        DatabasePaths.resolve(
            tmp_path / "data" / "marketplace.db",
            tmp_path / "data" / "paid-agent.db",
            tmp_path / "evidence" / "evidence.db",
        )
    )


def _scope(label: str = "one") -> SubjectScope:
    return SubjectScope(
        subject=f"subject-{label}",
        tenantId="tenant-a",
        adkSessionId=f"adk-{label}",
    )


def _session(
    scope: SubjectScope,
    *,
    state: MediationState = MediationState.WAITING_FOR_PLAN_APPROVAL,
    paid: bool = False,
    mediation_id: str | None = None,
) -> MediationSession:
    owner = OwnerScope(
        subject=scope.subject,
        tenantId=scope.tenant_id,
        adkSessionId=scope.adk_session_id,
        mediationSessionId=mediation_id or f"mediation-{uuid4()}",
    )
    selected = SelectedAgentSnapshot(
        canonicalAgentId="agent-paid" if paid else "agent-free",
        registryName="paid_booking_agent" if paid else "hotel_agent",
        a2aAgentName="paid-booking-agent" if paid else "hotel-agent",
        agentCardUrl="http://127.0.0.1:8005/.well-known/agent-card.json",
        rpcEndpoint="http://127.0.0.1:8005/a2a",
        a2aSkillId="paid-booking" if paid else "hotel-search",
        trustScore=95,
        cardDigest=canonical_digest({"card": paid}),
        snapshotDigest=canonical_digest({"snapshot": paid}),
        paymentExtensionUris=(
            ("urn:secure-a2a:extensions:x402-wire-simulation:v1",)
            if paid
            else ()
        ),
    )
    now = utc_now()
    step = MediationStep(
        stepId="step-1",
        ordinal=1,
        selectedAgent=selected,
        inputDigest=canonical_digest({"goal": "book"}),
        goal="book",
        paymentLimitMinor=5000,
        currency="USD",
    )
    plan = MediationPlan(
        planId=f"plan-{uuid4()}",
        planVersion=1,
        planDigest=canonical_digest(
            {"owner": owner.model_dump(mode="json", by_alias=True), "step": "step-1"}
        ),
        goalDigest=canonical_digest({"goal": "book"}),
        owner=owner,
        steps=(step,),
        createdAt=now,
        expiresAt=now + timedelta(minutes=10),
    )
    pending = {
        MediationState.WAITING_FOR_PLAN_APPROVAL: PendingAction(
            kind="approve-plan", targetRef=plan.plan_id
        ),
        MediationState.WAITING_FOR_PAYMENT_APPROVAL: PendingAction(
            kind="approve-payment", targetRef="continuation-1"
        ),
        MediationState.REFUND_PENDING: PendingAction(
            kind="request-refund", targetRef="continuation-1"
        ),
    }.get(state, PendingAction(kind="none"))
    continuation = _continuation(owner, plan) if paid else None
    result = (
        {"refundId": "refund-1", "resultDigest": canonical_digest({"refund": 1})}
        if state == MediationState.REFUNDED
        else ({"taskState": "completed", "simulation": True} if state == MediationState.COMPLETED else None)
    )
    session = MediationSession(
        owner=owner,
        goal="book",
        state=state,
        version=0,
        plan=plan,
        continuation=continuation,
        result=result,
        pendingAction=pending,
    )
    if state in {
        MediationState.WAITING_FOR_PLAN_APPROVAL,
        MediationState.WAITING_FOR_PAYMENT_APPROVAL,
    }:
        target = build_local_durable_view(session).approval_target
        assert target is not None
        session.approval_target_digest = canonical_digest(target)
    return session


def _continuation(owner: OwnerScope, plan: MediationPlan) -> MediationContinuation:
    payment_required = {
        "x402Version": 1,
        "accepts": [
            {
                "scheme": "exact-simulated",
                "network": "demo:local",
                "asset": "USD",
                "payTo": "merchant:demo-merchant",
                "maxAmountRequired": "1250",
            }
        ],
    }
    requirement = PaymentRequirementSnapshot(
        taskState="input-required",
        paymentStatus="payment-required",
        extensionUri="urn:secure-a2a:extensions:x402-wire-simulation:v1",
        profileId="x402-wire-simulation/1",
        orderId="order-1",
        quoteId="quote-1",
        amountMinor=1250,
        currency="USD",
        payee="demo-merchant",
        expiresAt=utc_now() + timedelta(minutes=10),
        requirementDigest=canonical_digest(payment_required),
        checkoutDigest=canonical_digest("secret-checkout-jwt"),
        paymentRequired=payment_required,
        checkoutAudience="demo-merchant",
        checkoutNonce="secret-checkout-nonce-1234",
        paymentAudience="demo-credential-provider",
        paymentNonce="secret-payment-nonce-12345",
    )
    remote = RemoteTaskSnapshot(
        taskId="task-1",
        contextId="context-1",
        state="input-required",
        taskDigest=canonical_digest({"task": 1}),
        orderId="order-1",
        quoteId="quote-1",
        paymentRequirement=requirement,
        artifact={"credential": "secret-artifact-credential"},
    )
    return MediationContinuation(
        continuationId="continuation-1",
        paymentWorkflowId="payment-workflow-1",
        owner=owner,
        planId=plan.plan_id,
        planVersion=plan.plan_version,
        planDigest=plan.plan_digest,
        stepId="step-1",
        remoteTask=remote,
        requirement=requirement,
        version=1,
    )


@pytest.mark.parametrize(
    ("state", "paid"),
    [
        (MediationState.WAITING_FOR_PLAN_APPROVAL, False),
        (MediationState.WAITING_FOR_PAYMENT_APPROVAL, True),
        (MediationState.REFUND_PENDING, True),
        (MediationState.COMPLETED, False),
        (MediationState.REFUNDED, True),
    ],
)
@pytest.mark.restart
def test_five_stable_states_restart_with_exact_view_and_binding(
    tmp_path: Path, state: MediationState, paid: bool
) -> None:
    repository = _repository(tmp_path)
    key = bytes(range(32))
    scope = _scope(state.value)
    original = _session(scope, state=state, paid=paid)
    first = SqliteMediationStore(repository, key)
    first.save_new(original)

    restarted = SqliteMediationStore(repository, key)
    restored = restarted.get(original.owner.mediation_session_id, scope)
    expected_projection = _session_projection(original)
    assert restored.state == state
    assert restored.version == 0
    assert restored.owner == original.owner
    assert restored.plan.plan_digest == original.plan.plan_digest
    assert _session_projection(restored) == expected_projection
    assert build_local_durable_view(restored) == build_local_durable_view(original)
    assert restarted.latest_for(scope) == restored
    if state in {
        MediationState.COMPLETED,
        MediationState.REFUNDED,
    }:
        assert restarted.active_for(scope) is None
    else:
        assert restarted.active_for(scope) == restored
    if paid:
        assert restored.continuation is not None
        assert restored.continuation.requirement.checkout_nonce == "persisted-redacted-nonce"
        assert restored.continuation.requirement.payment_nonce == "persisted-redacted-nonce"
    with sqlite3.connect(repository.paths.marketplace) as conn:
        digest = conn.execute(
            "SELECT session_digest FROM mediation_sessions_v4 WHERE scope_key=?",
            (restarted._scope_key(scope),),
        ).fetchone()[0]
    assert digest == canonical_digest(expected_projection)


@pytest.mark.security
def test_owner_idor_and_tamper_fail_closed(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    key = b"k" * 32
    alice = _scope("alice")
    bob = _scope("bob")
    session = _session(alice)
    store = SqliteMediationStore(repository, key)
    store.save_new(session)
    with pytest.raises(SecurityBlocked, match="not available"):
        store.get(session.owner.mediation_session_id, bob)

    scope_key = store._scope_key(alice)
    with sqlite3.connect(repository.paths.marketplace) as conn:
        row = conn.execute(
            "SELECT session_ciphertext FROM mediation_sessions_v4 "
            "WHERE scope_key=? AND mediation_session_id=?",
            (scope_key, session.owner.mediation_session_id),
        ).fetchone()
        damaged = bytearray(row[0])
        damaged[-1] ^= 1
        conn.execute(
            "UPDATE mediation_sessions_v4 SET session_ciphertext=? "
            "WHERE scope_key=? AND mediation_session_id=?",
            (bytes(damaged), scope_key, session.owner.mediation_session_id),
        )
    with pytest.raises(MediationError) as caught:
        store.get(session.owner.mediation_session_id, alice)
    assert caught.value.code == "MEDIATION_STORE_INTEGRITY"


@pytest.mark.security
def test_secret_free_projection_and_ciphertext(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    scope = _scope("secrets")
    session = _session(scope, state=MediationState.REFUND_PENDING, paid=True)
    session.result = {
        "credential": "secret-result-credential",
        "accessToken": "header.payload.signature",
        "refreshToken": "refresh-secret",
        "proof": "proof-secret",
        "safe": "visible-result",
    }
    projection = canonical_bytes(_session_projection(session))
    for secret in (
        b"secret-checkout-nonce-1234",
        b"secret-payment-nonce-12345",
        b"secret-artifact-credential",
        b"secret-result-credential",
        b"header.payload.signature",
        b"refresh-secret",
        b"proof-secret",
        b"accessToken",
        b"refreshToken",
        b"proof",
    ):
        assert secret not in projection
    assert _session_projection(session)["result"] == {
        "schemaVersion": "mediation-safe-result/1",
        "sourceDigest": canonical_digest(session.result),
    }
    store = SqliteMediationStore(repository, b"s" * 32)
    store.save_new(session)
    with sqlite3.connect(repository.paths.marketplace) as conn:
        row = conn.execute(
            "SELECT session_ciphertext,view_ciphertext FROM mediation_sessions_v4 "
            "WHERE scope_key=?",
            (store._scope_key(scope),),
        ).fetchone()
    persisted_blobs = bytes(row[0]) + bytes(row[1])
    assert b"visible-result" not in persisted_blobs
    assert b"subject-secrets" not in repository.paths.marketplace.read_bytes()


@pytest.mark.concurrency
def test_request_reservation_exact_replay_conflict_and_processing(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    store = SqliteMediationStore(repository, b"r" * 32)
    scope = _scope("request")
    session = _session(scope)
    digest = canonical_digest({"request": 1})
    assert store.reserve_request(scope, "request-1", digest).status == "reserved"
    with pytest.raises(MediationError) as processing:
        store.reserve_request(scope, "request-1", digest)
    assert processing.value.code == "MEDIATION_REQUEST_IN_PROGRESS"
    store.save_new(session)
    view = build_local_durable_view(session)
    store.complete_request(scope, "request-1", digest, session, view)

    replay = SqliteMediationStore(repository, b"r" * 32).reserve_request(
        scope, "request-1", digest
    )
    assert replay.status == "completed"
    assert replay.mediation_session_id == session.owner.mediation_session_id
    assert replay.result_version == session.version
    assert replay.view == view
    with pytest.raises(MediationError) as conflict:
        store.reserve_request(scope, "request-1", canonical_digest({"request": 2}))
    assert conflict.value.code == "IDEMPOTENCY_CONFLICT"
    failed_digest = canonical_digest({"request": "safe-failure"})
    store.reserve_request(scope, "request-2", failed_digest)
    store.fail_request(scope, "request-2", failed_digest)
    with pytest.raises(MediationError) as failed:
        store.reserve_request(scope, "request-2", failed_digest)
    assert failed.value.code == "MEDIATION_REQUEST_FAILED"


@pytest.mark.security
def test_request_ciphertext_cannot_be_transplanted_to_other_owner_or_request(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    store = SqliteMediationStore(repository, b"b" * 32)
    alice = _scope("cipher-alice")
    bob = _scope("cipher-bob")
    digest = canonical_digest({"same": "body"})
    session = _session(alice)
    store.reserve_request(alice, "alice-request", digest)
    store.save_new(session)
    store.complete_request(
        alice,
        "alice-request",
        digest,
        session,
        build_local_durable_view(session),
    )
    store.reserve_request(bob, "bob-request", digest)
    with sqlite3.connect(repository.paths.marketplace) as conn:
        source = conn.execute(
            "SELECT mediation_session_id,result_version,result_view_schema_version,"
            "result_key_version,result_view_nonce,result_view_ciphertext,result_view_digest "
            "FROM mediation_requests_v4 WHERE scope_key=? AND request_id=?",
            (store._scope_key(alice), "alice-request"),
        ).fetchone()
        conn.execute(
            "UPDATE mediation_requests_v4 SET status='completed',mediation_session_id=?,"
            "result_version=?,result_view_schema_version=?,result_key_version=?,"
            "result_view_nonce=?,result_view_ciphertext=?,result_view_digest=? "
            "WHERE scope_key=? AND request_id=?",
            (*source, store._scope_key(bob), "bob-request"),
        )
    with pytest.raises(MediationError) as transplanted:
        store.load_request(bob, "bob-request", digest)
    assert transplanted.value.code == "MEDIATION_STORE_INTEGRITY"


@pytest.mark.concurrency
def test_two_store_cas_has_exactly_one_winner(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    key = b"c" * 32
    scope = _scope("cas")
    initial = _session(scope)
    SqliteMediationStore(repository, key).save_new(initial)
    barrier = Barrier(2)

    def transition(state: MediationState) -> str:
        local = SqliteMediationStore(repository, key)
        current = local.get(initial.owner.mediation_session_id, scope)
        current.state = state
        current.pending_action = PendingAction(kind="none")
        current.approval_target_digest = None
        current.version += 1
        barrier.wait()
        try:
            local.compare_and_set(current, expected_version=0)
            return "won"
        except MediationError as error:
            assert error.code == "STATE_TRANSITION_CONFLICT"
            return "lost"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                transition,
                (MediationState.COMPLETED, MediationState.CANCELLED),
            )
        )
    assert sorted(results) == ["lost", "won"]
    final = SqliteMediationStore(repository, key).latest_for(scope)
    assert final is not None and final.version == 1


@pytest.mark.restart
def test_active_unique_latest_and_readiness_probe(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    store = SqliteMediationStore(repository, b"p" * 32)
    scope = _scope("latest")
    active = _session(scope)
    store.save_new(active)
    with pytest.raises(MediationError) as duplicate:
        store.save_new(_session(scope))
    assert duplicate.value.code == "ACTIVE_MEDIATION_EXISTS"
    active.state = MediationState.COMPLETED
    active.pending_action = PendingAction(kind="none")
    active.approval_target_digest = None
    active.version = 1
    store.compare_and_set(active, expected_version=0)
    newer = _session(scope)
    store.save_new(newer)
    assert store.active_for(scope).owner == newer.owner
    assert store.latest_for(scope).owner == newer.owner
    readiness = store.readiness_probe()
    assert readiness.ready
    assert readiness.schema_version == 4
    assert readiness.durability_profile == "local-durable"


@pytest.mark.security
def test_key_loader_requires_raw_private_32_byte_file(tmp_path: Path) -> None:
    path = tmp_path / "mediation-store.key"
    path.write_bytes(bytes(range(32)))
    path.chmod(0o600)
    assert load_mediation_store_key(path) == bytes(range(32))
    path.chmod(0o644)
    with pytest.raises(RuntimeError, match="permissions"):
        load_mediation_store_key(path)
    path.chmod(0o600)
    path.write_bytes(b"short")
    with pytest.raises(RuntimeError, match="32 raw bytes"):
        load_mediation_store_key(path)


@pytest.mark.security
@pytest.mark.restart
def test_key_provisioner_creates_and_preserves_private_store_key(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "provision_ap2_demo_keys.py"
    target = tmp_path / "keys"
    subprocess.run([sys.executable, str(script), str(target)], check=True)
    path = target / "mediation-store.key"
    first = path.read_bytes()
    assert len(first) == 32
    assert path.stat().st_mode & 0o777 == 0o600
    subprocess.run([sys.executable, str(script), str(target)], check=True)
    assert path.read_bytes() == first


@pytest.mark.security
@pytest.mark.restart
def test_key_sentinel_rejects_wrong_key_tamper_and_pre_sentinel_v4(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    correct = b"y" * 32
    store = SqliteMediationStore(repository, correct)
    assert SqliteMediationStore(repository, correct).readiness_probe().ready
    assert len(store._scope_key(_scope("namespace"))) == 64
    assert store._scope_key(_scope("namespace")) != "__mediation_store_sentinel_v4__"
    with pytest.raises((MediationError, RuntimeError)):
        SqliteMediationStore(repository, b"z" * 32)

    with sqlite3.connect(repository.paths.marketplace) as conn:
        row = conn.execute(
            "SELECT result_view_ciphertext FROM mediation_requests_v4 "
            "WHERE scope_key='__mediation_store_sentinel_v4__' "
            "AND request_id='__key_check__'"
        ).fetchone()
        damaged = bytearray(row[0])
        damaged[0] ^= 1
        conn.execute(
            "UPDATE mediation_requests_v4 SET result_view_ciphertext=? "
            "WHERE scope_key='__mediation_store_sentinel_v4__' "
            "AND request_id='__key_check__'",
            (bytes(damaged),),
        )
    with pytest.raises(MediationError):
        store.readiness_probe()
    with pytest.raises(MediationError):
        SqliteMediationStore(repository, correct)

    reset_repository = _repository(tmp_path / "pre-sentinel")
    legacy = SqliteMediationStore(reset_repository, correct)
    legacy.save_new(_session(_scope("pre-sentinel")))
    with sqlite3.connect(reset_repository.paths.marketplace) as conn:
        conn.execute(
            "DELETE FROM mediation_requests_v4 "
            "WHERE scope_key='__mediation_store_sentinel_v4__'"
        )
    with pytest.raises(RuntimeError, match="explicit reset or migration"):
        SqliteMediationStore(reset_repository, correct)
