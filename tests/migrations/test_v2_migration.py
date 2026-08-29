from __future__ import annotations

import sqlite3
import json
import subprocess
import sys
from pathlib import Path

import pytest

from secure_mediation_agent.workflow.migrations import (
    DatabasePaths,
    backup_once,
    migrate,
    verify,
)


pytestmark = pytest.mark.migration


def _paths(tmp_path: Path) -> DatabasePaths:
    return DatabasePaths.resolve(
        tmp_path / "data" / "marketplace.db",
        tmp_path / "data" / "paid-agent.db",
        tmp_path / "evidence" / "evidence.db",
    )


def test_empty_migration_is_reapplicable_and_verified(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    expected = {"marketplace": 4, "merchant": 4, "evidence": 4}
    assert migrate(paths) == expected
    assert migrate(paths) == expected
    assert verify(paths) == expected
    with sqlite3.connect(paths.marketplace) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {
        "payment_continuations_v3",
        "payment_bridge_approvals_v3",
        "payment_guarantees_v3",
        "payment_bridge_outbox_v3",
        "payment_bridge_settlements_v3",
        "payment_bridge_refunds_v3",
        "mediation_sessions_v4",
        "mediation_requests_v4",
    } <= tables
    with sqlite3.connect(paths.merchant) as conn:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='merchant_guarantees_v3'"
        ).fetchone() == ("merchant_guarantees_v3",)
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(merchant_guarantees_v3)")
        }
    assert {"settlement_id", "settlement_receipt_digest"} <= columns


def test_existing_evidence_bytes_survive_additive_migration_and_backup(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    with sqlite3.connect(paths.evidence) as conn:
        conn.execute(
            "CREATE TABLE evidence(evidence_id TEXT PRIMARY KEY,tenant_type TEXT NOT NULL,tenant_id TEXT NOT NULL,kind TEXT NOT NULL,exact_bytes BLOB NOT NULL,digest TEXT NOT NULL,kid TEXT,schema_version INTEGER NOT NULL,created_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO evidence VALUES('legacy','tenant','demo-tenant','legacy',X'000102','sha256:legacy',NULL,1,'2026-01-01T00:00:00Z')"
        )
    manifest = backup_once(paths, tmp_path / "backups")
    assert backup_once(paths, tmp_path / "backups") == manifest
    assert migrate(paths) == {"marketplace": 4, "merchant": 4, "evidence": 4}
    assert migrate(paths) == {"marketplace": 4, "merchant": 4, "evidence": 4}
    with sqlite3.connect(paths.evidence) as conn:
        row = conn.execute(
            "SELECT exact_bytes,media_type,profile_id FROM evidence WHERE evidence_id='legacy'"
        ).fetchone()
    assert row == (b"\x00\x01\x02", None, None)


def test_populated_v3_is_upgraded_to_v4_without_reset(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    migrate(paths)
    with sqlite3.connect(paths.marketplace) as conn:
        conn.execute(
            "INSERT INTO payment_continuations_v3("
            "continuation_id,payment_workflow_id,tenant_id,subject_id,session_id,"
            "context_id,mediation_session_id,plan_id,plan_version,plan_digest,"
            "plan_approval_id,step_id,canonical_agent_id,agent_card_digest,rpc_endpoint,"
            "task_id,task_context_id,order_id,quote_id,requirement_json,requirement_digest,"
            "checkout_jwt,checkout_hash,amount_minor,currency,payee,profile_id,expires_at,"
            "attach_digest,state,created_at,updated_at) "
            "VALUES(:continuation_id,:payment_workflow_id,:tenant_id,:subject_id,:session_id,"
            ":context_id,:mediation_session_id,:plan_id,:plan_version,:plan_digest,"
            ":plan_approval_id,:step_id,:canonical_agent_id,:agent_card_digest,:rpc_endpoint,"
            ":task_id,:task_context_id,:order_id,:quote_id,:requirement_json,"
            ":requirement_digest,:checkout_jwt,:checkout_hash,:amount_minor,:currency,:payee,"
            ":profile_id,:expires_at,:attach_digest,:state,:created_at,:updated_at)",
            {
                "continuation_id": "continuation-v3",
                "payment_workflow_id": "payment-workflow-v3",
                "tenant_id": "tenant-v3",
                "subject_id": "subject-v3",
                "session_id": "session-v3",
                "context_id": "context-v3",
                "mediation_session_id": "mediation-v3",
                "plan_id": "plan-v3",
                "plan_version": 1,
                "plan_digest": "sha256:plan-v3",
                "plan_approval_id": "approval-v3",
                "step_id": "step-v3",
                "canonical_agent_id": "agent-v3",
                "agent_card_digest": "sha256:card-v3",
                "rpc_endpoint": "http://127.0.0.1:8005/a2a",
                "task_id": "task-v3",
                "task_context_id": "task-context-v3",
                "order_id": "order-v3",
                "quote_id": "quote-v3",
                "requirement_json": "{}",
                "requirement_digest": "sha256:requirement-v3",
                "checkout_jwt": "legacy-secret-stays-in-authoritative-v3-only",
                "checkout_hash": "checkout-hash-v3",
                "amount_minor": 1250,
                "currency": "USD",
                "payee": "demo-merchant",
                "profile_id": "x402-wire-simulation/1",
                "expires_at": "2027-01-01T00:00:00Z",
                "attach_digest": "sha256:attach-v3",
                "state": "waiting_for_payment_approval",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
        )
        conn.execute("DROP TABLE mediation_requests_v4")
        conn.execute("DROP TABLE mediation_sessions_v4")
    for path in (paths.marketplace, paths.merchant, paths.evidence):
        with sqlite3.connect(path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version,applied_at,checksum) "
                "VALUES(3,'2026-01-01T00:00:00Z','v3-populated')"
            )
            conn.execute("DELETE FROM schema_migrations WHERE version=4")

    expected = {"marketplace": 4, "merchant": 4, "evidence": 4}
    assert migrate(paths) == expected
    assert migrate(paths) == expected
    assert verify(paths) == expected
    with sqlite3.connect(paths.marketplace) as conn:
        assert conn.execute(
            "SELECT continuation_id,checkout_jwt,state FROM payment_continuations_v3 "
            "WHERE continuation_id='continuation-v3'"
        ).fetchone() == (
            "continuation-v3",
            "legacy-secret-stays-in-authoritative-v3-only",
            "waiting_for_payment_approval",
        )
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert conn.execute("SELECT COUNT(*) FROM mediation_sessions_v4").fetchone() == (0,)


def test_database_path_contract_rejects_implicit_business_db(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="filenames"):
        DatabasePaths.resolve(
            tmp_path / "business.db",
            tmp_path / "paid-agent.db",
            tmp_path / "evidence.db",
        )


def test_sanitized_v1_three_database_cutover_verify_and_restore(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    fixtures = (
        (paths.marketplace, "legacy_orders", "order-v1"),
        (paths.merchant, "legacy_tasks", "task-v1"),
        (paths.evidence, "legacy_evidence", "evidence-v1"),
    )
    for path, table, value in fixtures:
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY,applied_at TEXT NOT NULL)"
            )
            conn.execute("INSERT INTO schema_migrations VALUES(1,'2025-01-01T00:00:00Z')")
            conn.execute(f"CREATE TABLE {table}(id TEXT PRIMARY KEY,payload TEXT NOT NULL)")
            conn.execute(f"INSERT INTO {table} VALUES('legacy',?)", (value,))
    script = Path(__file__).resolve().parents[2] / "scripts" / "migrate_ap2_x402_v2.py"
    common = [
        "--marketplace",
        str(paths.marketplace),
        "--merchant",
        str(paths.merchant),
        "--evidence",
        str(paths.evidence),
    ]
    applied = subprocess.run(
        [sys.executable, str(script), "apply", *common, "--backup-dir", str(tmp_path / "backups")],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(applied.stdout)
    manifest = payload["backupManifest"]
    assert payload["schemas"] == {"marketplace": 4, "merchant": 4, "evidence": 4}
    subprocess.run(
        [sys.executable, str(script), "verify", *common],
        check=True,
        capture_output=True,
        text=True,
    )
    for path, table, value in fixtures:
        with sqlite3.connect(path) as conn:
            assert conn.execute(f"SELECT payload FROM {table} WHERE id='legacy'").fetchone() == (value,)
    with sqlite3.connect(paths.marketplace) as conn:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='payment_continuations_v3'"
        ).fetchone() == ("payment_continuations_v3",)
    with sqlite3.connect(paths.merchant) as conn:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='merchant_guarantees_v3'"
        ).fetchone() == ("merchant_guarantees_v3",)

    restored = subprocess.run(
        [
            sys.executable,
            str(script),
            "restore-pre-cutover",
            *common,
            "--manifest",
            manifest,
            "--confirm",
            "RESTORE-PRE-CUTOVER",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(restored.stdout)["status"] == "restored-pre-cutover"
    for path, table, value in fixtures:
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone() == (1,)
            assert conn.execute(f"SELECT payload FROM {table} WHERE id='legacy'").fetchone() == (value,)
