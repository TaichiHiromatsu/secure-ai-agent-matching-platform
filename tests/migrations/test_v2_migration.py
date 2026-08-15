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
    assert migrate(paths) == {"marketplace": 2, "merchant": 2, "evidence": 2}
    assert migrate(paths) == {"marketplace": 2, "merchant": 2, "evidence": 2}
    assert verify(paths) == {"marketplace": 2, "merchant": 2, "evidence": 2}


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
    migrate(paths)
    with sqlite3.connect(paths.evidence) as conn:
        row = conn.execute(
            "SELECT exact_bytes,media_type,profile_id FROM evidence WHERE evidence_id='legacy'"
        ).fetchone()
    assert row == (b"\x00\x01\x02", None, None)


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
    assert payload["schemas"] == {"marketplace": 2, "merchant": 2, "evidence": 2}
    subprocess.run(
        [sys.executable, str(script), "verify", *common],
        check=True,
        capture_output=True,
        text=True,
    )
    for path, table, value in fixtures:
        with sqlite3.connect(path) as conn:
            assert conn.execute(f"SELECT payload FROM {table} WHERE id='legacy'").fetchone() == (value,)

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
