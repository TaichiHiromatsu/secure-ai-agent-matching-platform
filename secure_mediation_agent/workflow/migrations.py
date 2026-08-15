"""Additive schema-v2 migration for the three explicit SQLite authorities."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _ensure_migration_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(schema_migrations)")}
    if "checksum" not in columns:
        conn.execute("ALTER TABLE schema_migrations ADD COLUMN checksum TEXT")


MARKETPLACE_SCHEMA_V2 = r"""
CREATE TABLE IF NOT EXISTS workflows (
    workflow_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    context_id TEXT NOT NULL,
    request_json TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    state TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1 CHECK(version > 0),
    active_plan_id TEXT,
    plan_digest TEXT,
    selected_profile TEXT NOT NULL,
    merchant_task_id TEXT,
    order_id TEXT,
    payment_approval_id TEXT,
    last_error_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_workflow_active_session
ON workflows(tenant_id, session_id, context_id)
WHERE state NOT IN ('completed','payment_failed','refunded','cancelled','expired');

CREATE TABLE IF NOT EXISTS plan_snapshots (
    plan_id TEXT NOT NULL,
    plan_version INTEGER NOT NULL,
    workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
    schema_version TEXT NOT NULL,
    canonicalization TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    plan_digest TEXT NOT NULL UNIQUE,
    evidence_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    PRIMARY KEY(plan_id, plan_version)
);
CREATE TRIGGER IF NOT EXISTS plan_snapshot_immutable_update
BEFORE UPDATE ON plan_snapshots BEGIN SELECT RAISE(ABORT, 'plan snapshot is immutable'); END;
CREATE TRIGGER IF NOT EXISTS plan_snapshot_immutable_delete
BEFORE DELETE ON plan_snapshots BEGIN SELECT RAISE(ABORT, 'plan snapshot is immutable'); END;

CREATE TABLE IF NOT EXISTS plan_approvals (
    approval_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL UNIQUE REFERENCES workflows(workflow_id),
    plan_id TEXT NOT NULL,
    plan_version INTEGER NOT NULL,
    plan_digest TEXT NOT NULL,
    intent TEXT NOT NULL CHECK(intent='approve-plan'),
    nonce TEXT NOT NULL UNIQUE,
    issuer TEXT NOT NULL,
    audience TEXT NOT NULL,
    status TEXT NOT NULL,
    authorization_evidence_id TEXT NOT NULL,
    authorization_digest TEXT NOT NULL,
    approved_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS plan_approval_immutable_update
BEFORE UPDATE ON plan_approvals BEGIN SELECT RAISE(ABORT, 'plan approval is immutable'); END;

CREATE TABLE IF NOT EXISTS payment_approvals (
    payment_approval_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL UNIQUE REFERENCES workflows(workflow_id),
    task_id TEXT NOT NULL,
    checkout_hash TEXT NOT NULL,
    intent TEXT NOT NULL CHECK(intent='approve-payment'),
    nonce TEXT NOT NULL UNIQUE,
    display_digest TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status='approved'),
    approved_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS payment_approval_immutable_update
BEFORE UPDATE ON payment_approvals BEGIN SELECT RAISE(ABORT, 'payment approval is immutable'); END;

CREATE TABLE IF NOT EXISTS downstream_capabilities (
    capability_id TEXT PRIMARY KEY,
    approval_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
    plan_digest TEXT NOT NULL,
    order_id TEXT,
    task_id TEXT,
    audience TEXT NOT NULL,
    operation TEXT NOT NULL,
    nonce TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('issued','consumed','invalidated')),
    request_hash TEXT,
    evidence_id TEXT NOT NULL,
    evidence_digest TEXT NOT NULL,
    iat INTEGER NOT NULL,
    exp INTEGER NOT NULL,
    consumed_at TEXT,
    UNIQUE(audience, operation, nonce)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_capability_business_effect
ON downstream_capabilities(workflow_id, COALESCE(task_id,''), audience, operation)
WHERE status IN ('issued','consumed');

CREATE TABLE IF NOT EXISTS used_nonces_v2 (
    issuer TEXT NOT NULL,
    scope TEXT NOT NULL,
    nonce TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    task_id TEXT,
    request_hash TEXT NOT NULL,
    consumed_at TEXT NOT NULL,
    PRIMARY KEY(issuer, scope, nonce)
);

CREATE TABLE IF NOT EXISTS merchant_task_mirrors (
    task_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL UNIQUE REFERENCES workflows(workflow_id),
    context_id TEXT NOT NULL,
    merchant_id TEXT NOT NULL,
    order_id TEXT NOT NULL UNIQUE,
    profile_id TEXT NOT NULL,
    observed_state TEXT NOT NULL,
    observed_version INTEGER NOT NULL,
    task_evidence_id TEXT NOT NULL,
    task_evidence_digest TEXT NOT NULL,
    agent_card_digest TEXT NOT NULL,
    onboarding_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS payment_requirements (
    requirements_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL UNIQUE REFERENCES merchant_task_mirrors(task_id),
    profile_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    evidence_digest TEXT NOT NULL,
    checkout_hash TEXT NOT NULL,
    capability_id TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at TEXT
);

CREATE TABLE IF NOT EXISTS payment_artifacts (
    artifact_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
    task_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    evidence_id TEXT NOT NULL UNIQUE,
    evidence_digest TEXT NOT NULL,
    issuer TEXT NOT NULL,
    kid TEXT NOT NULL,
    trust_snapshot_id TEXT,
    reference_digest TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(workflow_id, kind)
);
CREATE TRIGGER IF NOT EXISTS payment_artifact_immutable_update
BEFORE UPDATE ON payment_artifacts BEGIN SELECT RAISE(ABORT, 'payment artifact is immutable'); END;
CREATE TRIGGER IF NOT EXISTS payment_artifact_immutable_delete
BEFORE DELETE ON payment_artifacts BEGIN SELECT RAISE(ABORT, 'payment artifact is immutable'); END;

CREATE TABLE IF NOT EXISTS settlement_attempts (
    attempt_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal > 0),
    profile_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    external_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('pending','settled','failed','unknown')),
    network TEXT NOT NULL,
    transaction_ref TEXT,
    receipt_evidence_id TEXT,
    receipt_evidence_digest TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    UNIQUE(task_id, ordinal),
    UNIQUE(profile_id, external_id),
    UNIQUE(idempotency_key)
);
CREATE TABLE IF NOT EXISTS settlement_attempt_events (
    event_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL REFERENCES settlement_attempts(attempt_id),
    seq INTEGER NOT NULL,
    observed_state TEXT NOT NULL,
    network TEXT NOT NULL,
    transaction_ref TEXT,
    error_code TEXT,
    evidence_id TEXT,
    evidence_digest TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(attempt_id, seq)
);

CREATE TABLE IF NOT EXISTS profile_receipts (
    receipt_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL REFERENCES settlement_attempts(attempt_id),
    ordinal INTEGER NOT NULL,
    success INTEGER NOT NULL CHECK(success IN (0,1)),
    network TEXT NOT NULL,
    transaction_ref TEXT,
    error_code TEXT,
    evidence_id TEXT NOT NULL,
    evidence_digest TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(task_id, ordinal)
);
CREATE TRIGGER IF NOT EXISTS profile_receipt_immutable_update
BEFORE UPDATE ON profile_receipts BEGIN SELECT RAISE(ABORT, 'receipt history is append-only'); END;
CREATE TRIGGER IF NOT EXISTS profile_receipt_immutable_delete
BEFORE DELETE ON profile_receipts BEGIN SELECT RAISE(ABORT, 'receipt history is append-only'); END;

CREATE TABLE IF NOT EXISTS fulfillment_operations (
    operation_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    phase TEXT NOT NULL CHECK(phase IN ('prepare','commit')),
    state TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    external_id TEXT NOT NULL,
    artifact_evidence_id TEXT,
    artifact_evidence_digest TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(task_id, phase)
);

CREATE TABLE IF NOT EXISTS refunds_v2 (
    refund_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
    attempt_id TEXT NOT NULL REFERENCES settlement_attempts(attempt_id),
    original_payment_id TEXT NOT NULL,
    amount INTEGER NOT NULL CHECK(amount >= 0),
    currency TEXT NOT NULL,
    reason TEXT NOT NULL,
    provider_ref TEXT NOT NULL,
    state TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reconciliation_actions (
    action_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    actor_role TEXT NOT NULL,
    reason TEXT NOT NULL,
    external_id TEXT NOT NULL,
    observed_state TEXT NOT NULL,
    evidence_digest TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS idempotency_records_v2 (
    tenant_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    idem_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    result_type TEXT,
    result_id TEXT,
    response_json TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    PRIMARY KEY(tenant_id, actor_id, operation, idem_key)
);

CREATE TABLE IF NOT EXISTS workflow_events (
    event_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
    seq INTEGER NOT NULL,
    actor_id TEXT NOT NULL,
    actor_role TEXT NOT NULL,
    operation TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT NOT NULL,
    approval_intent TEXT,
    idempotency_result TEXT,
    error_code TEXT,
    related_digest TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(workflow_id, seq)
);
CREATE TRIGGER IF NOT EXISTS workflow_event_immutable_update
BEFORE UPDATE ON workflow_events BEGIN SELECT RAISE(ABORT, 'workflow event is append-only'); END;
CREATE TRIGGER IF NOT EXISTS workflow_event_immutable_delete
BEFORE DELETE ON workflow_events BEGIN SELECT RAISE(ABORT, 'workflow event is append-only'); END;

CREATE TABLE IF NOT EXISTS outbox (
    outbox_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
    event_type TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','leased','done','failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    available_at TEXT NOT NULL,
    lease_owner TEXT,
    lease_until TEXT,
    last_error_code TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(event_type, operation_id)
);
CREATE INDEX IF NOT EXISTS ix_outbox_available
ON outbox(status, available_at, lease_until);

CREATE TABLE IF NOT EXISTS worker_heartbeats (
    worker_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('starting','running','stopping')),
    last_operation_id TEXT,
    last_error_code TEXT
);

CREATE TABLE IF NOT EXISTS evidence_intents_v2 (
    intent_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL UNIQUE,
    expected_digest TEXT NOT NULL,
    kind TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('pending','committed','failed')),
    created_at TEXT NOT NULL,
    committed_at TEXT
);

CREATE TABLE IF NOT EXISTS trust_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    issuer TEXT NOT NULL,
    kid TEXT NOT NULL,
    jwks_evidence_id TEXT NOT NULL,
    jwks_evidence_digest TEXT NOT NULL,
    onboarding_version TEXT NOT NULL,
    valid_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(issuer, kid, onboarding_version)
);

CREATE TABLE IF NOT EXISTS rail_accounts_v2 (
    account_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    balance INTEGER NOT NULL CHECK(balance >= 0),
    version INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(account_id, asset)
);
CREATE TABLE IF NOT EXISTS rail_operations_v2 (
    operation_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('settle','refund')),
    source_id TEXT NOT NULL,
    payer TEXT NOT NULL,
    payee TEXT NOT NULL,
    amount INTEGER NOT NULL CHECK(amount >= 0),
    asset TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('settled','failed','unknown')),
    applied INTEGER NOT NULL CHECK(applied IN (0,1)),
    idempotency_key TEXT NOT NULL UNIQUE,
    request_digest TEXT NOT NULL,
    external_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


EVIDENCE_SCHEMA_V2 = r"""
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    tenant_type TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    exact_bytes BLOB NOT NULL,
    digest TEXT NOT NULL,
    kid TEXT,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_access_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    evidence_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    actor_role TEXT NOT NULL,
    allowed INTEGER NOT NULL CHECK(allowed IN (0,1)),
    created_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS evidence_immutable_update
BEFORE UPDATE ON evidence BEGIN SELECT RAISE(ABORT, 'evidence is immutable'); END;
CREATE TRIGGER IF NOT EXISTS evidence_immutable_delete
BEFORE DELETE ON evidence BEGIN SELECT RAISE(ABORT, 'evidence is immutable'); END;
"""


MERCHANT_SCHEMA_V2 = r"""
CREATE TABLE IF NOT EXISTS merchant_tasks_v2 (
    task_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL UNIQUE,
    context_id TEXT NOT NULL,
    order_id TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL,
    task_json TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS merchant_messages_v2 (
    message_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES merchant_tasks_v2(task_id),
    context_id TEXT NOT NULL,
    status TEXT NOT NULL,
    message_json TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS merchant_requirements_v2 (
    requirements_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL UNIQUE REFERENCES merchant_tasks_v2(task_id),
    requirements_json TEXT NOT NULL,
    requirements_digest TEXT NOT NULL,
    checkout_jwt TEXT NOT NULL,
    checkout_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS merchant_operations_v2 (
    operation_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES merchant_tasks_v2(task_id),
    phase TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    state TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(task_id, phase)
);
CREATE TABLE IF NOT EXISTS merchant_receipt_history_v2 (
    receipt_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES merchant_tasks_v2(task_id),
    ordinal INTEGER NOT NULL,
    receipt_json TEXT NOT NULL,
    receipt_digest TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(task_id, ordinal)
);
CREATE TABLE IF NOT EXISTS merchant_capability_consumptions_v2 (
    capability_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    consumed_at TEXT NOT NULL,
    UNIQUE(task_id, operation)
);
"""


@dataclass(frozen=True, slots=True)
class DatabasePaths:
    marketplace: Path
    merchant: Path
    evidence: Path

    @classmethod
    def resolve(
        cls, marketplace: str | Path, merchant: str | Path, evidence: str | Path
    ) -> "DatabasePaths":
        values = [Path(value).expanduser().resolve() for value in (marketplace, merchant, evidence)]
        expected = ("marketplace.db", "paid-agent.db", "evidence.db")
        if tuple(path.name for path in values) != expected:
            raise ValueError(f"database filenames must be exactly {expected}")
        for path in values:
            path.parent.mkdir(parents=True, exist_ok=True)
        return cls(*values)


def _add_evidence_columns(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(evidence)")}
    for name in ("media_type", "profile_id", "retention_class"):
        if name not in columns:
            conn.execute(f"ALTER TABLE evidence ADD COLUMN {name} TEXT")


def migrate(paths: DatabasePaths) -> dict[str, int]:
    schemas = (
        (paths.marketplace, MARKETPLACE_SCHEMA_V2),
        (paths.merchant, MERCHANT_SCHEMA_V2),
        (paths.evidence, EVIDENCE_SCHEMA_V2),
    )
    checksum = hashlib.sha256("\n".join(schema for _, schema in schemas).encode()).hexdigest()
    for path, schema in schemas:
        with _connect(path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            _ensure_migration_table(conn)
            conn.executescript(schema)
            if path == paths.evidence:
                _add_evidence_columns(conn)
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version,applied_at,checksum) VALUES(?,?,?)",
                (SCHEMA_VERSION, utc_now(), checksum),
            )
            conn.commit()
    return verify(paths)


def verify(paths: DatabasePaths) -> dict[str, int]:
    result: dict[str, int] = {}
    for label, path in (("marketplace", paths.marketplace), ("merchant", paths.merchant), ("evidence", paths.evidence)):
        with _connect(path) as conn:
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError(f"{label} database integrity check failed")
            if conn.execute("PRAGMA foreign_key_check").fetchall():
                raise RuntimeError(f"{label} database foreign key check failed")
            version = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
            if int(version or 0) != SCHEMA_VERSION:
                raise RuntimeError(f"{label} schema version mismatch")
            result[label] = int(version)
    return result


def backup_once(paths: DatabasePaths, directory: str | Path) -> Path:
    target = Path(directory).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    inventory = []
    for path in (paths.marketplace, paths.merchant, paths.evidence):
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "absent"
        inventory.append({"path": str(path), "sha256": digest})
    migration_id = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()[:16]
    manifest_path = target / f"pre-v2-{migration_id}.json"
    if manifest_path.exists():
        return manifest_path
    backups = []
    for path in (paths.marketplace, paths.merchant, paths.evidence):
        if not path.exists():
            backups.append({"source": str(path), "backup": None, "sha256": "absent"})
            continue
        destination = target / f"{path.name}.pre-v2-{migration_id}"
        shutil.copy2(path, destination)
        with destination.open("rb") as handle:
            os.fsync(handle.fileno())
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        backups.append({"source": str(path), "backup": str(destination), "sha256": digest})
    payload = {
        "migrationId": migration_id,
        "phase": "backed_up",
        "createdAt": utc_now(),
        "backups": backups,
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(manifest_path)
    directory_fd = os.open(target, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return manifest_path
