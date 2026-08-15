"""SQLite persistence for the project-local marketplace payment demo.

The module intentionally exposes dictionaries and primitive values so the store can
be used before (and independently from) the payment domain models.  Signed bytes are
kept in a physically separate evidence database.  A durable business-database intent
bridges the two SQLite transactions; callers must not advance a payment aggregate
until the intent reports ``committed``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


BUSINESS_SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 1


class StoreError(RuntimeError):
    """Base error for deterministic persistence failures."""


class IdempotencyConflict(StoreError):
    """The same idempotency key was reused with different normalized input."""


class ReplayDetected(StoreError):
    """A nonce was already consumed."""


class ConcurrentUpdate(StoreError):
    """An optimistic aggregate update lost a race."""


class EvidenceAccessDenied(StoreError):
    """The actor cannot read exact evidence bytes."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _decode_json(value: str | None) -> Any:
    return None if value is None else json.loads(value)


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


class MarketplaceStore:
    """File-backed SQLite repositories and migration owner."""

    def __init__(
        self,
        business_db: str | Path,
        evidence_db: str | Path,
        *,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        self.business_db = str(business_db)
        self.evidence_db = str(evidence_db)
        self.busy_timeout_ms = busy_timeout_ms
        for path in (Path(self.business_db), Path(self.evidence_db)):
            if str(path) != ":memory:":
                path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    def _connect(self, path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(path, timeout=self.busy_timeout_ms / 1000)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_ms)}")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = FULL")
        return conn

    @contextmanager
    def business_transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        conn = self._connect(self.business_db)
        try:
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def evidence_transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        conn = self._connect(self.evidence_db)
        try:
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def migrate(self) -> None:
        with self._connect(self.business_db) as conn:
            conn.executescript(_BUSINESS_SCHEMA_V1)
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (BUSINESS_SCHEMA_VERSION, utc_now()),
            )
        with self._connect(self.evidence_db) as conn:
            conn.executescript(_EVIDENCE_SCHEMA_V1)
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (EVIDENCE_SCHEMA_VERSION, utc_now()),
            )

    def schema_versions(self) -> dict[str, int]:
        with self._connect(self.business_db) as conn:
            business = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
        with self._connect(self.evidence_db) as conn:
            evidence = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
        return {"business": int(business or 0), "evidence": int(evidence or 0)}

    # ---- A2A task and marketplace aggregate ---------------------------------

    def save_task(
        self,
        task_id: str,
        context_id: str,
        state: str,
        *,
        actor_id: str,
        tenant_id: str,
        metadata: Mapping[str, Any] | None = None,
        response: Mapping[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.business_transaction() as conn:
            current = conn.execute("SELECT version FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            if current is None:
                if expected_version not in (None, 0):
                    raise ConcurrentUpdate(f"task {task_id} does not exist")
                conn.execute(
                    """INSERT INTO tasks
                       (task_id, context_id, state, actor_id, tenant_id, metadata_json,
                        response_json, version, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                    (
                        task_id,
                        context_id,
                        state,
                        actor_id,
                        tenant_id,
                        compact_json(metadata or {}),
                        compact_json(response) if response is not None else None,
                        now,
                        now,
                    ),
                )
            else:
                version = int(current[0])
                if expected_version is not None and version != expected_version:
                    raise ConcurrentUpdate(f"task {task_id} expected version {expected_version}, got {version}")
                changed = conn.execute(
                    """UPDATE tasks SET context_id=?, state=?, actor_id=?, tenant_id=?,
                       metadata_json=?, response_json=?, version=version+1, updated_at=?
                       WHERE task_id=? AND version=?""",
                    (
                        context_id,
                        state,
                        actor_id,
                        tenant_id,
                        compact_json(metadata or {}),
                        compact_json(response) if response is not None else None,
                        now,
                        task_id,
                        version,
                    ),
                ).rowcount
                if changed != 1:
                    raise ConcurrentUpdate(f"task {task_id} changed concurrently")
        return self.get_task(task_id)  # type: ignore[return-value]

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect(self.business_db) as conn:
            result = _row(conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone())
        if result:
            result["metadata"] = _decode_json(result.pop("metadata_json"))
            result["response"] = _decode_json(result.pop("response_json"))
        return result

    def create_order(
        self,
        order_id: str,
        task_id: str,
        context_id: str,
        customer_id: str,
        merchant_id: str,
        *,
        state: str = "awaiting_quote",
        correlation_id: str | None = None,
        schema_version: int = 1,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.business_transaction() as conn:
            conn.execute(
                """INSERT INTO orders
                   (order_id, task_id, context_id, customer_id, merchant_id, state,
                    correlation_id, version, schema_version, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (
                    order_id,
                    task_id,
                    context_id,
                    customer_id,
                    merchant_id,
                    state,
                    correlation_id or order_id,
                    schema_version,
                    now,
                    now,
                ),
            )
            self._append_state_event(
                conn, "order", order_id, None, state, customer_id, "order-created", 1, now
            )
        return self.get_order(order_id)  # type: ignore[return-value]

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        with self._connect(self.business_db) as conn:
            return _row(conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone())

    def update_order_state(
        self,
        order_id: str,
        from_state: str,
        to_state: str,
        *,
        actor_id: str,
        reason: str,
        expected_version: int,
        recovery_kind: str | None = None,
        authoritative_operation_id: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.business_transaction() as conn:
            changed = conn.execute(
                """UPDATE orders SET state=?, version=version+1, recovery_kind=?,
                   authoritative_operation_id=?, updated_at=?
                   WHERE order_id=? AND state=? AND version=?""",
                (
                    to_state,
                    recovery_kind,
                    authoritative_operation_id,
                    now,
                    order_id,
                    from_state,
                    expected_version,
                ),
            ).rowcount
            if changed != 1:
                raise ConcurrentUpdate(f"order {order_id} state/version changed")
            self._append_state_event(
                conn,
                "order",
                order_id,
                from_state,
                to_state,
                actor_id,
                reason,
                expected_version + 1,
                now,
            )
        return self.get_order(order_id)  # type: ignore[return-value]

    def list_state_events(self, aggregate_type: str, aggregate_id: str) -> list[dict[str, Any]]:
        with self._connect(self.business_db) as conn:
            return [
                dict(row)
                for row in conn.execute(
                    """SELECT * FROM state_events WHERE aggregate_type=? AND aggregate_id=?
                       ORDER BY sequence""",
                    (aggregate_type, aggregate_id),
                ).fetchall()
            ]

    @staticmethod
    def _append_state_event(
        conn: sqlite3.Connection,
        aggregate_type: str,
        aggregate_id: str,
        from_state: str | None,
        to_state: str,
        actor_id: str,
        reason: str,
        sequence: int,
        timestamp: str,
    ) -> None:
        conn.execute(
            """INSERT INTO state_events
               (aggregate_type, aggregate_id, from_state, to_state, actor_id,
                reason, sequence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (aggregate_type, aggregate_id, from_state, to_state, actor_id, reason, sequence, timestamp),
        )

    # ---- replay and idempotency ---------------------------------------------

    def begin_idempotency(
        self, scope: str, actor_id: str, key: str, request_hash: str
    ) -> dict[str, Any]:
        with self.business_transaction() as conn:
            existing = conn.execute(
                """SELECT request_hash, state, response_json FROM idempotency_records
                   WHERE scope=? AND actor_id=? AND idempotency_key=?""",
                (scope, actor_id, key),
            ).fetchone()
            if existing:
                if existing["request_hash"] != request_hash:
                    raise IdempotencyConflict(f"idempotency conflict in {scope}")
                return {
                    "status": "hit",
                    "state": existing["state"],
                    "response": _decode_json(existing["response_json"]),
                }
            now = utc_now()
            conn.execute(
                """INSERT INTO idempotency_records
                   (scope, actor_id, idempotency_key, request_hash, state, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'processing', ?, ?)""",
                (scope, actor_id, key, request_hash, now, now),
            )
            return {"status": "new", "state": "processing", "response": None}

    def complete_idempotency(
        self,
        scope: str,
        actor_id: str,
        key: str,
        request_hash: str,
        response: Mapping[str, Any],
    ) -> None:
        with self.business_transaction() as conn:
            row = conn.execute(
                """SELECT request_hash FROM idempotency_records
                   WHERE scope=? AND actor_id=? AND idempotency_key=?""",
                (scope, actor_id, key),
            ).fetchone()
            if row is None or row["request_hash"] != request_hash:
                raise IdempotencyConflict(f"missing or conflicting idempotency record in {scope}")
            conn.execute(
                """UPDATE idempotency_records SET state='completed', response_json=?, updated_at=?
                   WHERE scope=? AND actor_id=? AND idempotency_key=?""",
                (compact_json(response), utc_now(), scope, actor_id, key),
            )

    def consume_nonce(
        self,
        issuer: str,
        nonce: str,
        digest: str,
        *,
        order_id: str,
        task_id: str,
        operation: str,
    ) -> None:
        try:
            with self.business_transaction() as conn:
                conn.execute(
                    """INSERT INTO used_nonces
                       (issuer, nonce, digest, order_id, task_id, operation, consumed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (issuer, nonce, digest, order_id, task_id, operation, utc_now()),
                )
        except sqlite3.IntegrityError as exc:
            raise ReplayDetected(f"nonce already used by {issuer}") from exc

    # ---- evidence saga -------------------------------------------------------

    def put_evidence(
        self,
        *,
        intent_id: str,
        evidence_id: str,
        tenant_type: str,
        tenant_id: str,
        kind: str,
        exact_bytes: bytes | str,
        kid: str | None = None,
        schema_version: int = 1,
    ) -> dict[str, Any]:
        payload = exact_bytes.encode("utf-8") if isinstance(exact_bytes, str) else bytes(exact_bytes)
        digest = sha256_digest(payload)
        now = utc_now()
        with self.business_transaction() as conn:
            existing = conn.execute(
                "SELECT * FROM evidence_intents WHERE intent_id=?", (intent_id,)
            ).fetchone()
            if existing:
                if existing["evidence_id"] != evidence_id or existing["digest"] != digest:
                    raise IdempotencyConflict("evidence intent reused with different bytes")
            else:
                conn.execute(
                    """INSERT INTO evidence_intents
                       (intent_id, evidence_id, tenant_type, tenant_id, kind, digest, state,
                        schema_version, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)""",
                    (
                        intent_id,
                        evidence_id,
                        tenant_type,
                        tenant_id,
                        kind,
                        digest,
                        schema_version,
                        now,
                        now,
                    ),
                )
        with self.evidence_transaction() as conn:
            existing = conn.execute("SELECT digest FROM evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
            if existing and existing["digest"] != digest:
                raise IdempotencyConflict("evidence ID reused with different bytes")
            conn.execute(
                """INSERT OR IGNORE INTO evidence
                   (evidence_id, tenant_type, tenant_id, kind, exact_bytes, digest, kid,
                    schema_version, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (evidence_id, tenant_type, tenant_id, kind, payload, digest, kid, schema_version, now),
            )
        return self.reconcile_evidence_intent(intent_id)

    def reconcile_evidence_intent(self, intent_id: str) -> dict[str, Any]:
        with self._connect(self.business_db) as conn:
            intent = conn.execute("SELECT * FROM evidence_intents WHERE intent_id=?", (intent_id,)).fetchone()
        if intent is None:
            raise StoreError(f"unknown evidence intent {intent_id}")
        with self._connect(self.evidence_db) as conn:
            evidence = conn.execute(
                "SELECT evidence_id, digest FROM evidence WHERE evidence_id=?", (intent["evidence_id"],)
            ).fetchone()
        state = "committed" if evidence and evidence["digest"] == intent["digest"] else "pending"
        error = None if state == "committed" else "EVIDENCE_NOT_DURABLE"
        with self.business_transaction() as conn:
            conn.execute(
                "UPDATE evidence_intents SET state=?, last_error=?, updated_at=? WHERE intent_id=?",
                (state, error, utc_now(), intent_id),
            )
            result = conn.execute("SELECT * FROM evidence_intents WHERE intent_id=?", (intent_id,)).fetchone()
        return dict(result)

    def get_evidence_metadata(self, evidence_id: str) -> dict[str, Any] | None:
        with self._connect(self.evidence_db) as conn:
            return _row(
                conn.execute(
                    """SELECT evidence_id, tenant_type, tenant_id, kind, digest, kid,
                       schema_version, created_at FROM evidence WHERE evidence_id=?""",
                    (evidence_id,),
                ).fetchone()
            )

    def read_evidence(
        self,
        evidence_id: str,
        *,
        actor_id: str,
        actor_role: str,
        tenant_type: str | None = None,
        tenant_id: str | None = None,
    ) -> bytes:
        payload: bytes | None = None
        allowed = False
        with self.evidence_transaction() as conn:
            evidence = conn.execute("SELECT * FROM evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
            allowed = bool(
                evidence
                and (
                    actor_role == "operator"
                    or (
                        actor_role in {"customer", "merchant"}
                        and tenant_type == evidence["tenant_type"]
                        and tenant_id == evidence["tenant_id"]
                    )
                )
            )
            conn.execute(
                """INSERT INTO evidence_access_events
                   (evidence_id, actor_id, actor_role, allowed, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (evidence_id, actor_id, actor_role, int(allowed), utc_now()),
            )
            if allowed:
                payload = bytes(evidence["exact_bytes"])
        # The audit row must survive a denied read, so raise after commit.
        if not allowed or payload is None:
            raise EvidenceAccessDenied("evidence access denied")
        return payload

    # ---- small inspection helpers used by deterministic services/tests ------

    def fetch_business(self, query: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        if not query.lstrip().upper().startswith("SELECT"):
            raise StoreError("fetch_business only permits SELECT")
        with self._connect(self.business_db) as conn:
            return [dict(row) for row in conn.execute(query, parameters).fetchall()]


_BUSINESS_SCHEMA_V1 = r"""
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    context_id TEXT NOT NULL,
    state TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    response_json TEXT,
    version INTEGER NOT NULL DEFAULT 1 CHECK(version > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL UNIQUE REFERENCES tasks(task_id),
    context_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    merchant_id TEXT NOT NULL,
    quote_id TEXT,
    state TEXT NOT NULL,
    correlation_id TEXT NOT NULL,
    recovery_kind TEXT,
    authoritative_operation_id TEXT,
    version INTEGER NOT NULL DEFAULT 1 CHECK(version > 0),
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS merchant_quotes (
    quote_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(order_id),
    merchant_id TEXT NOT NULL,
    requirement_digest TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    merchandise_amount INTEGER NOT NULL CHECK(merchandise_amount >= 0),
    policy_version TEXT NOT NULL,
    state TEXT NOT NULL,
    iat TEXT NOT NULL,
    exp TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pricing (
    order_id TEXT PRIMARY KEY REFERENCES orders(order_id),
    policy_version TEXT NOT NULL,
    merchandise_amount INTEGER NOT NULL CHECK(merchandise_amount >= 0),
    customer_surcharge INTEGER NOT NULL CHECK(customer_surcharge >= 0),
    collection_rail_cost INTEGER NOT NULL CHECK(collection_rail_cost >= 0),
    customer_total INTEGER NOT NULL CHECK(customer_total >= 0),
    provider_commission INTEGER NOT NULL CHECK(provider_commission >= 0),
    merchant_payable_amount INTEGER NOT NULL CHECK(merchant_payable_amount >= 0),
    payout_rail_cost INTEGER NOT NULL CHECK(payout_rail_cost >= 0),
    asset TEXT NOT NULL,
    decimals INTEGER NOT NULL,
    network TEXT NOT NULL,
    rounding_rule TEXT NOT NULL,
    calculated_at TEXT NOT NULL,
    schema_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS charges (
    charge_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL UNIQUE REFERENCES orders(order_id),
    challenge_id TEXT NOT NULL UNIQUE,
    payer_id TEXT NOT NULL,
    pay_to TEXT NOT NULL,
    amount INTEGER NOT NULL CHECK(amount >= 0),
    asset TEXT NOT NULL,
    nonce TEXT NOT NULL,
    state TEXT NOT NULL,
    operation_id TEXT UNIQUE,
    proof_digest TEXT,
    settlement_receipt_id TEXT,
    ap2_receipt_id TEXT,
    journal_id TEXT,
    idempotency_key TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS payables (
    payable_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL UNIQUE REFERENCES orders(order_id),
    charge_id TEXT NOT NULL UNIQUE REFERENCES charges(charge_id),
    merchant_id TEXT NOT NULL,
    amount INTEGER NOT NULL CHECK(amount >= 0),
    asset TEXT NOT NULL,
    state TEXT NOT NULL,
    journal_id TEXT NOT NULL,
    guarantee_id TEXT,
    available_at TEXT,
    payout_id TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS guarantees (
    guarantee_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL UNIQUE REFERENCES orders(order_id),
    payable_id TEXT NOT NULL UNIQUE REFERENCES payables(payable_id),
    state TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    digest TEXT NOT NULL,
    iat TEXT NOT NULL,
    exp TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fulfillments (
    fulfillment_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL UNIQUE REFERENCES orders(order_id),
    guarantee_id TEXT NOT NULL UNIQUE REFERENCES guarantees(guarantee_id),
    merchant_id TEXT NOT NULL,
    state TEXT NOT NULL,
    receipt_id TEXT,
    receipt_digest TEXT,
    attempt INTEGER NOT NULL DEFAULT 0,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS refunds (
    refund_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(order_id),
    charge_id TEXT NOT NULL REFERENCES charges(charge_id),
    payable_id TEXT NOT NULL REFERENCES payables(payable_id),
    responsibility TEXT NOT NULL,
    reason TEXT NOT NULL,
    amount INTEGER NOT NULL CHECK(amount >= 0),
    asset TEXT NOT NULL,
    state TEXT NOT NULL,
    rail_state TEXT NOT NULL,
    ledger_state TEXT NOT NULL,
    operation_id TEXT UNIQUE,
    journal_id TEXT,
    receipt_id TEXT,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(order_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS payouts (
    payout_id TEXT PRIMARY KEY,
    merchant_id TEXT NOT NULL,
    state TEXT NOT NULL,
    gross_amount INTEGER NOT NULL CHECK(gross_amount >= 0),
    commission_amount INTEGER NOT NULL CHECK(commission_amount >= 0),
    rail_cost INTEGER NOT NULL CHECK(rail_cost >= 0),
    net_amount INTEGER NOT NULL CHECK(net_amount >= 0),
    asset TEXT NOT NULL,
    eligibility_json TEXT NOT NULL,
    operation_id TEXT UNIQUE,
    journal_id TEXT,
    receipt_id TEXT,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 1,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(merchant_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS payout_items (
    payout_id TEXT NOT NULL REFERENCES payouts(payout_id),
    payable_id TEXT NOT NULL REFERENCES payables(payable_id),
    amount INTEGER NOT NULL CHECK(amount >= 0),
    state TEXT NOT NULL DEFAULT 'claimed',
    created_at TEXT NOT NULL,
    PRIMARY KEY(payout_id, payable_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_payout_item_active_payable
ON payout_items(payable_id) WHERE state <> 'released';

CREATE TABLE IF NOT EXISTS journal_transactions (
    journal_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    currency TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('draft', 'posted')),
    content_hash TEXT NOT NULL,
    idempotency_key TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    posted_at TEXT,
    UNIQUE(event_type, source_id)
);

CREATE TABLE IF NOT EXISTS journal_entries (
    entry_id TEXT PRIMARY KEY,
    journal_id TEXT NOT NULL REFERENCES journal_transactions(journal_id),
    account TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('debit', 'credit')),
    amount INTEGER NOT NULL CHECK(amount >= 0),
    currency TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    source_event TEXT NOT NULL,
    idempotency_key TEXT,
    related_entry_id TEXT REFERENCES journal_entries(entry_id),
    schema_version INTEGER NOT NULL
);

CREATE TRIGGER IF NOT EXISTS journal_entry_currency_guard
BEFORE INSERT ON journal_entries
BEGIN
    SELECT CASE WHEN NEW.currency <> (SELECT currency FROM journal_transactions WHERE journal_id=NEW.journal_id)
        THEN RAISE(ABORT, 'journal currency mismatch') END;
END;

CREATE TRIGGER IF NOT EXISTS journal_balance_guard
BEFORE UPDATE OF state ON journal_transactions
WHEN NEW.state = 'posted'
BEGIN
    SELECT CASE WHEN
        COALESCE((SELECT SUM(amount) FROM journal_entries WHERE journal_id=NEW.journal_id AND side='debit'), 0)
        <>
        COALESCE((SELECT SUM(amount) FROM journal_entries WHERE journal_id=NEW.journal_id AND side='credit'), 0)
        OR NOT EXISTS (SELECT 1 FROM journal_entries WHERE journal_id=NEW.journal_id)
        THEN RAISE(ABORT, 'unbalanced journal') END;
END;

CREATE TRIGGER IF NOT EXISTS posted_journal_immutable
BEFORE UPDATE ON journal_transactions
WHEN OLD.state = 'posted'
BEGIN SELECT RAISE(ABORT, 'posted journal is immutable'); END;
CREATE TRIGGER IF NOT EXISTS posted_journal_no_delete
BEFORE DELETE ON journal_transactions
WHEN OLD.state = 'posted'
BEGIN SELECT RAISE(ABORT, 'posted journal is immutable'); END;
CREATE TRIGGER IF NOT EXISTS posted_entry_immutable_update
BEFORE UPDATE ON journal_entries
WHEN (SELECT state FROM journal_transactions WHERE journal_id=OLD.journal_id) = 'posted'
BEGIN SELECT RAISE(ABORT, 'posted journal entry is immutable'); END;
CREATE TRIGGER IF NOT EXISTS posted_entry_immutable_insert
BEFORE INSERT ON journal_entries
WHEN (SELECT state FROM journal_transactions WHERE journal_id=NEW.journal_id) = 'posted'
BEGIN SELECT RAISE(ABORT, 'posted journal entry is immutable'); END;
CREATE TRIGGER IF NOT EXISTS posted_entry_immutable_delete
BEFORE DELETE ON journal_entries
WHEN (SELECT state FROM journal_transactions WHERE journal_id=OLD.journal_id) = 'posted'
BEGIN SELECT RAISE(ABORT, 'posted journal entry is immutable'); END;

CREATE TABLE IF NOT EXISTS rail_accounts (
    account_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    balance INTEGER NOT NULL CHECK(balance >= 0),
    version INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(account_id, asset)
);

CREATE TABLE IF NOT EXISTS rail_operations (
    operation_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('charge', 'refund', 'payout')),
    source_id TEXT NOT NULL,
    from_account TEXT NOT NULL,
    to_account TEXT NOT NULL,
    asset TEXT NOT NULL,
    amount INTEGER NOT NULL CHECK(amount >= 0),
    state TEXT NOT NULL CHECK(state IN ('settled', 'failed', 'unknown')),
    applied INTEGER NOT NULL DEFAULT 0 CHECK(applied IN (0, 1)),
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    error_code TEXT,
    receipt_json TEXT,
    attempt INTEGER NOT NULL DEFAULT 1,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(kind, idempotency_key)
);

CREATE TABLE IF NOT EXISTS idempotency_records (
    scope TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    state TEXT NOT NULL,
    response_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(scope, actor_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS used_nonces (
    issuer TEXT NOT NULL,
    nonce TEXT NOT NULL,
    digest TEXT NOT NULL,
    order_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    consumed_at TEXT NOT NULL,
    PRIMARY KEY(issuer, nonce)
);

CREATE TABLE IF NOT EXISTS state_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    aggregate_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(aggregate_type, aggregate_id, sequence)
);

CREATE TABLE IF NOT EXISTS evidence_intents (
    intent_id TEXT PRIMARY KEY,
    evidence_id TEXT NOT NULL UNIQUE,
    tenant_type TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    digest TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('pending', 'committed')),
    last_error TEXT,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS merchant_onboarding (
    merchant_id TEXT NOT NULL,
    version TEXT NOT NULL,
    status TEXT NOT NULL,
    key_id TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    agreement_version TEXT NOT NULL,
    pricing_policy_version TEXT NOT NULL,
    payout_destination TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(merchant_id, version)
);

CREATE INDEX IF NOT EXISTS ix_orders_state ON orders(state);
CREATE INDEX IF NOT EXISTS ix_payables_merchant_state ON payables(merchant_id, state);
CREATE INDEX IF NOT EXISTS ix_rail_operations_state ON rail_operations(state);
CREATE INDEX IF NOT EXISTS ix_evidence_intents_state ON evidence_intents(state);
"""


_EVIDENCE_SCHEMA_V1 = r"""
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

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
    allowed INTEGER NOT NULL CHECK(allowed IN (0, 1)),
    created_at TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS evidence_immutable_update
BEFORE UPDATE ON evidence BEGIN SELECT RAISE(ABORT, 'evidence is immutable'); END;
CREATE TRIGGER IF NOT EXISTS evidence_immutable_delete
BEFORE DELETE ON evidence BEGIN SELECT RAISE(ABORT, 'evidence is immutable'); END;

CREATE INDEX IF NOT EXISTS ix_evidence_tenant ON evidence(tenant_type, tenant_id, kind);
"""
