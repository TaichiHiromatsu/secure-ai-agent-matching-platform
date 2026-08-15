"""SQLite repository for the authoritative workflow and exact evidence."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from .canonical import canonical_json, sha256_digest
from .errors import DomainError
from .failpoints import crash_after
from .migrations import DatabasePaths, migrate
from .models import PlanSnapshot, WorkflowState


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _row(value: sqlite3.Row | None) -> dict[str, Any] | None:
    return None if value is None else dict(value)


class WorkflowRepository:
    def __init__(self, paths: DatabasePaths, *, migrate_on_start: bool = True) -> None:
        self.paths = paths
        if migrate_on_start:
            migrate(paths)
        self._ensure_accounts()

    @classmethod
    def open(
        cls,
        marketplace: str | Path,
        merchant: str | Path,
        evidence: str | Path,
    ) -> "WorkflowRepository":
        return cls(DatabasePaths.resolve(marketplace, merchant, evidence))

    def _connect(self, path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect(self.paths.marketplace)
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def merchant_transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect(self.paths.merchant)
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def evidence_transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect(self.paths.evidence)
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_accounts(self) -> None:
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO rail_accounts_v2(account_id,asset,balance,updated_at) VALUES(?,?,?,?)",
                ("demo-customer", "USD", 100_000, now),
            )
            conn.execute(
                "INSERT OR IGNORE INTO rail_accounts_v2(account_id,asset,balance,updated_at) VALUES(?,?,?,?)",
                ("demo-merchant", "USD", 0, now),
            )

    def put_evidence(
        self,
        *,
        workflow_id: str,
        evidence_id: str,
        tenant_id: str,
        kind: str,
        exact_bytes: bytes | str,
        kid: str | None,
        media_type: str,
        profile_id: str,
        retention_class: str = "payment-evidence",
    ) -> str:
        payload = exact_bytes.encode("utf-8") if isinstance(exact_bytes, str) else exact_bytes
        digest = sha256_digest(payload)
        intent_id = f"intent:{evidence_id}"
        now = utc_now()
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT expected_digest,state FROM evidence_intents_v2 WHERE evidence_id=?",
                (evidence_id,),
            ).fetchone()
            if existing and existing["expected_digest"] != digest:
                raise DomainError("IDEMPOTENCY_CONFLICT", "Evidence ID was reused.", workflow_id)
            conn.execute(
                "INSERT OR IGNORE INTO evidence_intents_v2"
                "(intent_id,workflow_id,evidence_id,expected_digest,kind,state,created_at)"
                " VALUES(?,?,?,?,?,'pending',?)",
                (intent_id, workflow_id, evidence_id, digest, kind, now),
            )
        with self.evidence_transaction() as conn:
            existing = conn.execute(
                "SELECT digest FROM evidence WHERE evidence_id=?", (evidence_id,)
            ).fetchone()
            if existing and existing["digest"] != digest:
                raise DomainError("IDEMPOTENCY_CONFLICT", "Evidence bytes changed.", workflow_id)
            conn.execute(
                "INSERT OR IGNORE INTO evidence"
                "(evidence_id,tenant_type,tenant_id,kind,exact_bytes,digest,kid,schema_version,created_at,media_type,profile_id,retention_class)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    evidence_id,
                    "tenant",
                    tenant_id,
                    kind,
                    payload,
                    digest,
                    kid,
                    2,
                    now,
                    media_type,
                    profile_id,
                    retention_class,
                ),
            )
        with self.transaction() as conn:
            conn.execute(
                "UPDATE evidence_intents_v2 SET state='committed',committed_at=? "
                "WHERE evidence_id=? AND expected_digest=?",
                (utc_now(), evidence_id, digest),
            )
        return digest

    def read_evidence(
        self,
        evidence_id: str,
        *,
        actor_id: str,
        actor_role: str,
        tenant_id: str,
    ) -> bytes:
        with self.evidence_transaction() as conn:
            row = conn.execute("SELECT * FROM evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
            allowed = bool(row and (actor_role == "operator" or row["tenant_id"] == tenant_id))
            conn.execute(
                "INSERT INTO evidence_access_events(evidence_id,actor_id,actor_role,allowed,created_at) VALUES(?,?,?,?,?)",
                (evidence_id, actor_id, actor_role, int(allowed), utc_now()),
            )
            if not allowed:
                raise DomainError("TENANT_BINDING_MISMATCH", "Evidence access denied.", evidence_id)
            return bytes(row["exact_bytes"])

    def create_workflow(
        self,
        *,
        workflow_id: str,
        tenant_id: str,
        customer_id: str,
        session_id: str,
        context_id: str,
        request: dict[str, Any],
        request_digest: str,
        plan: PlanSnapshot,
        plan_digest: str,
        plan_evidence_id: str,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO workflows(workflow_id,tenant_id,customer_id,session_id,context_id,request_json,request_digest,state,version,active_plan_id,plan_digest,selected_profile,created_at,updated_at)"
                " VALUES(?,?,?,?,?,?,?,?,1,?,?,?,?,?)",
                (
                    workflow_id,
                    tenant_id,
                    customer_id,
                    session_id,
                    context_id,
                    canonical_json(request),
                    request_digest,
                    WorkflowState.PLAN_APPROVAL_REQUIRED,
                    plan.plan_id,
                    plan_digest,
                    "x402-wire-simulation/1",
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO plan_snapshots(plan_id,plan_version,workflow_id,schema_version,canonicalization,request_digest,plan_digest,evidence_id,created_at,expires_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    plan.plan_id,
                    plan.plan_version,
                    workflow_id,
                    plan.schema_version,
                    plan.canonicalization,
                    request_digest,
                    plan_digest,
                    plan_evidence_id,
                    plan.created_at,
                    plan.expires_at,
                ),
            )
            self._event(
                conn,
                workflow_id,
                actor_id=customer_id,
                actor_role="customer",
                operation="request-plan",
                from_state=None,
                to_state=WorkflowState.PLAN_APPROVAL_REQUIRED,
                related_digest=plan_digest,
            )
        result = self.get_workflow(workflow_id)
        crash_after(f"state:{WorkflowState.PLAN_APPROVAL_REQUIRED}")
        return result

    def get_workflow(self, workflow_id: str) -> dict[str, Any]:
        with self._connect(self.paths.marketplace) as conn:
            row = _row(conn.execute("SELECT * FROM workflows WHERE workflow_id=?", (workflow_id,)).fetchone())
        if row is None:
            raise KeyError(workflow_id)
        row["request"] = json.loads(row.pop("request_json"))
        return row

    def active_workflow(self, tenant_id: str, session_id: str, context_id: str) -> dict[str, Any] | None:
        with self._connect(self.paths.marketplace) as conn:
            row = _row(
                conn.execute(
                    "SELECT * FROM workflows WHERE tenant_id=? AND session_id=? AND context_id=? "
                    "AND state NOT IN ('completed','payment_failed','refunded','cancelled','expired')",
                    (tenant_id, session_id, context_id),
                ).fetchone()
            )
        if row:
            row["request"] = json.loads(row.pop("request_json"))
        return row

    def get_plan_bytes(self, workflow: dict[str, Any]) -> bytes:
        with self._connect(self.paths.marketplace) as conn:
            row = conn.execute(
                "SELECT evidence_id FROM plan_snapshots WHERE workflow_id=? AND plan_id=?",
                (workflow["workflow_id"], workflow["active_plan_id"]),
            ).fetchone()
        return self.read_evidence(
            row["evidence_id"],
            actor_id=workflow["customer_id"],
            actor_role="customer",
            tenant_id=workflow["tenant_id"],
        )

    def transition(
        self,
        workflow_id: str,
        *,
        expected_state: WorkflowState | str,
        to_state: WorkflowState | str,
        actor_id: str,
        actor_role: str,
        operation: str,
        expected_version: int | None = None,
        approval_intent: str | None = None,
        related_digest: str | None = None,
        error_code: str | None = None,
        updates: dict[str, Any] | None = None,
        outbox: tuple[str, str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        with self.transaction() as conn:
            row = conn.execute("SELECT state,version FROM workflows WHERE workflow_id=?", (workflow_id,)).fetchone()
            if row is None:
                raise KeyError(workflow_id)
            if row["state"] != str(expected_state) or (
                expected_version is not None and row["version"] != expected_version
            ):
                raise DomainError(
                    "STATE_TRANSITION_CONFLICT",
                    "Workflow state changed concurrently.",
                    workflow_id,
                    current_state=row["state"],
                )
            assignments = ["state=?", "version=version+1", "updated_at=?"]
            values: list[Any] = [str(to_state), utc_now()]
            allowed = {"merchant_task_id", "order_id", "payment_approval_id", "last_error_code"}
            for key, value in (updates or {}).items():
                if key not in allowed:
                    raise ValueError(f"unsupported workflow update field {key}")
                assignments.append(f"{key}=?")
                values.append(value)
            values.extend([workflow_id, row["version"]])
            changed = conn.execute(
                f"UPDATE workflows SET {','.join(assignments)} WHERE workflow_id=? AND version=?",
                values,
            ).rowcount
            if changed != 1:
                raise DomainError("STATE_TRANSITION_CONFLICT", "Workflow changed concurrently.", workflow_id)
            self._event(
                conn,
                workflow_id,
                actor_id=actor_id,
                actor_role=actor_role,
                operation=operation,
                from_state=row["state"],
                to_state=str(to_state),
                approval_intent=approval_intent,
                related_digest=related_digest,
                error_code=error_code,
            )
            if outbox:
                event_type, operation_id, payload = outbox
                payload_json = canonical_json(payload)
                conn.execute(
                    "INSERT OR IGNORE INTO outbox(outbox_id,workflow_id,event_type,operation_id,payload_json,payload_digest,status,available_at,created_at)"
                    " VALUES(?,?,?,?,?,?,'pending',?,?)",
                    (
                        f"outbox:{operation_id}",
                        workflow_id,
                        event_type,
                        operation_id,
                        payload_json,
                        sha256_digest(payload_json),
                        utc_now(),
                        utc_now(),
                    ),
                )
        result = self.get_workflow(workflow_id)
        crash_after(f"state:{to_state}")
        return result

    def heartbeat_worker(
        self,
        worker_id: str,
        *,
        status: str = "running",
        operation_id: str | None = None,
        error_code: str | None = None,
    ) -> None:
        """Publish a durable liveness record used by fail-closed readiness."""

        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO worker_heartbeats"
                "(worker_id,started_at,last_seen_at,status,last_operation_id,last_error_code) "
                "VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(worker_id) DO UPDATE SET "
                "last_seen_at=excluded.last_seen_at,status=excluded.status,"
                "last_operation_id=excluded.last_operation_id,last_error_code=excluded.last_error_code",
                (worker_id, now, now, status, operation_id, error_code),
            )

    def lease_outbox(
        self,
        worker_id: str,
        *,
        operation_id: str | None = None,
        lease_seconds: int = 30,
    ) -> dict[str, Any] | None:
        """Atomically lease one due row, including an abandoned expired lease."""

        now = datetime.now(UTC)
        now_text = now.isoformat().replace("+00:00", "Z")
        lease_until = (now + timedelta(seconds=lease_seconds)).isoformat().replace(
            "+00:00", "Z"
        )
        with self.transaction() as conn:
            clauses = [
                "available_at<=?",
                "(status='pending' OR (status='leased' AND lease_until<=?))",
            ]
            values: list[Any] = [now_text, now_text]
            if operation_id is not None:
                clauses.append("operation_id=?")
                values.append(operation_id)
            row = conn.execute(
                "SELECT * FROM outbox WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at,outbox_id LIMIT 1",
                values,
            ).fetchone()
            if row is None:
                return None
            changed = conn.execute(
                "UPDATE outbox SET status='leased',lease_owner=?,lease_until=?,attempts=attempts+1 "
                "WHERE outbox_id=? AND (status='pending' OR (status='leased' AND lease_until<=?))",
                (worker_id, lease_until, row["outbox_id"], now_text),
            ).rowcount
            if changed != 1:
                return None
            leased = dict(
                conn.execute(
                    "SELECT * FROM outbox WHERE outbox_id=?", (row["outbox_id"],)
                ).fetchone()
            )
        leased["payload"] = json.loads(leased.pop("payload_json"))
        return leased

    def complete_outbox(self, outbox_id: str, worker_id: str) -> None:
        with self.transaction() as conn:
            changed = conn.execute(
                "UPDATE outbox SET status='done',completed_at=?,lease_owner=NULL,"
                "lease_until=NULL,last_error_code=NULL WHERE outbox_id=? "
                "AND status='leased' AND lease_owner=?",
                (utc_now(), outbox_id, worker_id),
            ).rowcount
            if changed != 1:
                row = conn.execute(
                    "SELECT status FROM outbox WHERE outbox_id=?", (outbox_id,)
                ).fetchone()
                if row and row["status"] == "done":
                    return
                raise DomainError(
                    "OUTBOX_LEASE_LOST", "Outbox lease is no longer owned.", outbox_id
                )

    def retry_outbox(
        self,
        outbox_id: str,
        worker_id: str,
        *,
        error_code: str,
        delay_seconds: int = 1,
        terminal: bool = False,
    ) -> None:
        available_at = (
            datetime.now(UTC) + timedelta(seconds=max(0, delay_seconds))
        ).isoformat().replace("+00:00", "Z")
        with self.transaction() as conn:
            conn.execute(
                "UPDATE outbox SET status=?,available_at=?,lease_owner=NULL,lease_until=NULL,"
                "last_error_code=? WHERE outbox_id=? AND status='leased' AND lease_owner=?",
                (
                    "failed" if terminal else "pending",
                    available_at,
                    error_code,
                    outbox_id,
                    worker_id,
                ),
            )

    def outbox_row(self, operation_id: str) -> dict[str, Any] | None:
        with self._connect(self.paths.marketplace) as conn:
            row = conn.execute(
                "SELECT * FROM outbox WHERE operation_id=?", (operation_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def recoverable_workflow(self) -> dict[str, Any] | None:
        """Find a transient aggregate that lost its in-process continuation."""

        states = (
            WorkflowState.PLAN_APPROVED,
            WorkflowState.FREE_EXECUTING,
            WorkflowState.FINAL_VALIDATING,
        )
        with self._connect(self.paths.marketplace) as conn:
            row = conn.execute(
                "SELECT * FROM workflows WHERE state IN (?,?,?) ORDER BY updated_at LIMIT 1",
                tuple(str(state) for state in states),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["request"] = json.loads(result.pop("request_json"))
        return result

    def outbox_health(self, *, heartbeat_max_age_seconds: int = 15) -> dict[str, Any]:
        now = datetime.now(UTC)
        cutoff = (now - timedelta(seconds=heartbeat_max_age_seconds)).isoformat().replace(
            "+00:00", "Z"
        )
        now_text = now.isoformat().replace("+00:00", "Z")
        with self._connect(self.paths.marketplace) as conn:
            live_workers = conn.execute(
                "SELECT COUNT(*) FROM worker_heartbeats "
                "WHERE status='running' AND last_seen_at>=?",
                (cutoff,),
            ).fetchone()[0]
            stale_leases = conn.execute(
                "SELECT COUNT(*) FROM outbox WHERE status='leased' AND lease_until<=?",
                (now_text,),
            ).fetchone()[0]
            failed = conn.execute(
                "SELECT COUNT(*) FROM outbox WHERE status='failed'"
            ).fetchone()[0]
            overdue = conn.execute(
                "SELECT COUNT(*) FROM outbox WHERE status='pending' AND available_at<=?",
                (now_text,),
            ).fetchone()[0]
        return {
            "liveWorkers": int(live_workers),
            "staleLeases": int(stale_leases),
            "failed": int(failed),
            "overdue": int(overdue),
        }

    def evidence_intent_health(self) -> dict[str, int]:
        with self._connect(self.paths.marketplace) as conn:
            pending = conn.execute(
                "SELECT COUNT(*) FROM evidence_intents_v2 WHERE state='pending'"
            ).fetchone()[0]
            failed = conn.execute(
                "SELECT COUNT(*) FROM evidence_intents_v2 WHERE state='failed'"
            ).fetchone()[0]
        return {"pending": int(pending), "failed": int(failed)}

    def reconcile_evidence_intents(self) -> dict[str, int]:
        """Close cross-database intents after a crash between evidence and ack."""

        with self._connect(self.paths.marketplace) as conn:
            pending = [
                dict(row)
                for row in conn.execute(
                    "SELECT evidence_id,expected_digest FROM evidence_intents_v2 "
                    "WHERE state='pending'"
                )
            ]
        resolved: list[tuple[str, str]] = []
        corrupt: list[str] = []
        with self._connect(self.paths.evidence) as conn:
            for intent in pending:
                evidence = conn.execute(
                    "SELECT digest FROM evidence WHERE evidence_id=?",
                    (intent["evidence_id"],),
                ).fetchone()
                if evidence is None:
                    continue
                if evidence["digest"] == intent["expected_digest"]:
                    resolved.append((utc_now(), intent["evidence_id"]))
                else:
                    corrupt.append(intent["evidence_id"])
        if resolved or corrupt:
            with self.transaction() as conn:
                conn.executemany(
                    "UPDATE evidence_intents_v2 SET state='committed',committed_at=? "
                    "WHERE evidence_id=? AND state='pending'",
                    resolved,
                )
                conn.executemany(
                    "UPDATE evidence_intents_v2 SET state='failed' "
                    "WHERE evidence_id=? AND state='pending'",
                    [(item,) for item in corrupt],
                )
        return {
            "resolved": len(resolved),
            "missing": len(pending) - len(resolved) - len(corrupt),
            "corrupt": len(corrupt),
        }

    def trust_health(self) -> dict[str, int]:
        with self._connect(self.paths.marketplace) as conn:
            missing = conn.execute(
                "SELECT COUNT(*) FROM payment_artifacts a "
                "LEFT JOIN trust_snapshots s ON s.snapshot_id=a.trust_snapshot_id "
                "WHERE a.trust_snapshot_id IS NULL OR s.snapshot_id IS NULL "
                "OR s.kid<>a.kid"
            ).fetchone()[0]
            snapshots = [
                dict(row)
                for row in conn.execute(
                    "SELECT jwks_evidence_id,jwks_evidence_digest FROM trust_snapshots"
                )
            ]
        corrupt = 0
        with self._connect(self.paths.evidence) as conn:
            for snapshot in snapshots:
                evidence = conn.execute(
                    "SELECT digest FROM evidence WHERE evidence_id=?",
                    (snapshot["jwks_evidence_id"],),
                ).fetchone()
                if (
                    evidence is None
                    or evidence["digest"] != snapshot["jwks_evidence_digest"]
                ):
                    corrupt += 1
        return {"missing": int(missing), "corrupt": corrupt}

    def _event(
        self,
        conn: sqlite3.Connection,
        workflow_id: str,
        *,
        actor_id: str,
        actor_role: str,
        operation: str,
        from_state: str | None,
        to_state: str,
        approval_intent: str | None = None,
        related_digest: str | None = None,
        error_code: str | None = None,
    ) -> None:
        seq = conn.execute(
            "SELECT COALESCE(MAX(seq),0)+1 FROM workflow_events WHERE workflow_id=?",
            (workflow_id,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO workflow_events(event_id,workflow_id,seq,actor_id,actor_role,operation,from_state,to_state,approval_intent,error_code,related_digest,created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"event:{workflow_id}:{seq}",
                workflow_id,
                seq,
                actor_id,
                actor_role,
                operation,
                from_state,
                to_state,
                approval_intent,
                error_code,
                related_digest,
                utc_now(),
            ),
        )

    def begin_idempotency(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        operation: str,
        key: str,
        request_hash: str,
    ) -> dict[str, Any] | None:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM idempotency_records_v2 WHERE tenant_id=? AND actor_id=? AND operation=? AND idem_key=?",
                (tenant_id, actor_id, operation, key),
            ).fetchone()
            if row:
                if row["request_hash"] != request_hash:
                    raise DomainError("IDEMPOTENCY_CONFLICT", "Idempotency key input changed.", key)
                if row["status"] == "processing":
                    return {"_idempotencyStatus": "processing"}
                return json.loads(row["response_json"]) if row["response_json"] else None
            conn.execute(
                "INSERT INTO idempotency_records_v2(tenant_id,actor_id,operation,idem_key,request_hash,status,created_at,expires_at)"
                " VALUES(?,?,?,?,?,'processing',?,?)",
                (
                    tenant_id,
                    actor_id,
                    operation,
                    key,
                    request_hash,
                    utc_now(),
                    (datetime.now(UTC) + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
                ),
            )
        return None

    def abandon_idempotency(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        operation: str,
        key: str,
        request_hash: str,
    ) -> None:
        """Release a failed pre-effect request so an exact retry can be evaluated."""

        with self.transaction() as conn:
            conn.execute(
                "DELETE FROM idempotency_records_v2 WHERE tenant_id=? AND actor_id=? "
                "AND operation=? AND idem_key=? AND request_hash=? AND status='processing'",
                (tenant_id, actor_id, operation, key, request_hash),
            )

    def complete_idempotency(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        operation: str,
        key: str,
        request_hash: str,
        response: dict[str, Any],
        result_id: str,
    ) -> None:
        with self.transaction() as conn:
            changed = conn.execute(
                "UPDATE idempotency_records_v2 SET status='completed',result_type='workflow',result_id=?,response_json=? "
                "WHERE tenant_id=? AND actor_id=? AND operation=? AND idem_key=? AND request_hash=?",
                (
                    result_id,
                    canonical_json(response),
                    tenant_id,
                    actor_id,
                    operation,
                    key,
                    request_hash,
                ),
            ).rowcount
            if changed != 1:
                raise DomainError("IDEMPOTENCY_CONFLICT", "Idempotency record is unavailable.", key)

    def record_plan_approval(
        self,
        *,
        workflow_id: str,
        approval_id: str,
        nonce: str,
        authorization_evidence_id: str,
        authorization_digest: str,
        approved_at: str,
        expires_at: str,
    ) -> None:
        workflow = self.get_workflow(workflow_id)
        with self.transaction() as conn:
            row = conn.execute("SELECT state,version FROM workflows WHERE workflow_id=?", (workflow_id,)).fetchone()
            if row["state"] != WorkflowState.PLAN_APPROVAL_REQUIRED:
                raise DomainError("APPROVAL_NOT_PENDING", "Plan approval is not pending.", workflow_id, row["state"])
            try:
                conn.execute(
                    "INSERT INTO used_nonces_v2(issuer,scope,nonce,workflow_id,request_hash,consumed_at) VALUES(?,?,?,?,?,?)",
                    ("secure-mediation-plan-authority", "approve-plan", nonce, workflow_id, workflow["plan_digest"], approved_at),
                )
                conn.execute(
                    "INSERT INTO plan_approvals(approval_id,workflow_id,plan_id,plan_version,plan_digest,intent,nonce,issuer,audience,status,authorization_evidence_id,authorization_digest,approved_at,expires_at)"
                    " VALUES(?,?,?,?,?,'approve-plan',?,'secure-mediation-plan-authority','secure-mediation-workflow','approved',?,?,?,?)",
                    (
                        approval_id,
                        workflow_id,
                        workflow["active_plan_id"],
                        1,
                        workflow["plan_digest"],
                        nonce,
                        authorization_evidence_id,
                        authorization_digest,
                        approved_at,
                        expires_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DomainError("REPLAY_DETECTED", "Plan approval was already consumed.", workflow_id) from exc

    def record_payment_approval(
        self,
        *,
        workflow_id: str,
        approval_id: str,
        task_id: str,
        checkout_hash: str,
        nonce: str,
        display_digest: str,
        approved_at: str,
        expires_at: str,
    ) -> None:
        with self.transaction() as conn:
            row = conn.execute("SELECT state FROM workflows WHERE workflow_id=?", (workflow_id,)).fetchone()
            if row is None or row["state"] != WorkflowState.PAYMENT_APPROVAL_REQUIRED:
                raise DomainError("APPROVAL_NOT_PENDING", "Payment approval is not pending.", workflow_id, None if row is None else row["state"])
            try:
                conn.execute(
                    "INSERT INTO used_nonces_v2(issuer,scope,nonce,workflow_id,task_id,request_hash,consumed_at) VALUES(?,?,?,?,?,?,?)",
                    ("demo-trusted-surface", "approve-payment", nonce, workflow_id, task_id, display_digest, approved_at),
                )
                conn.execute(
                    "INSERT INTO payment_approvals(payment_approval_id,workflow_id,task_id,checkout_hash,intent,nonce,display_digest,status,approved_at,expires_at)"
                    " VALUES(?,?,?,?,'approve-payment',?,?,'approved',?,?)",
                    (approval_id, workflow_id, task_id, checkout_hash, nonce, display_digest, approved_at, expires_at),
                )
            except sqlite3.IntegrityError as exc:
                raise DomainError("REPLAY_DETECTED", "Payment approval was already consumed.", workflow_id) from exc

    def insert_capability(
        self,
        *,
        capability_id: str,
        approval_id: str,
        workflow_id: str,
        plan_digest: str,
        order_id: str | None,
        task_id: str | None,
        audience: str,
        operation: str,
        nonce: str,
        evidence_id: str,
        evidence_digest: str,
        issued_at: int,
        expires_at: int,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO downstream_capabilities(capability_id,approval_id,workflow_id,plan_digest,order_id,task_id,audience,operation,nonce,status,evidence_id,evidence_digest,iat,exp)"
                " VALUES(?,?,?,?,?,?,?,?,?,'issued',?,?,?,?)",
                (
                    capability_id,
                    approval_id,
                    workflow_id,
                    plan_digest,
                    order_id,
                    task_id,
                    audience,
                    operation,
                    nonce,
                    evidence_id,
                    evidence_digest,
                    issued_at,
                    expires_at,
                ),
            )

    def capability_for_operation(
        self, workflow_id: str, operation: str
    ) -> tuple[dict[str, Any], str] | None:
        """Return the immutable capability and exact token for worker replay."""

        with self._connect(self.paths.marketplace) as conn:
            row = conn.execute(
                "SELECT * FROM downstream_capabilities WHERE workflow_id=? AND operation=?",
                (workflow_id, operation),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        with self._connect(self.paths.evidence) as conn:
            evidence = conn.execute(
                "SELECT exact_bytes,digest FROM evidence WHERE evidence_id=?",
                (item["evidence_id"],),
            ).fetchone()
        if evidence is None or evidence["digest"] != item["evidence_digest"]:
            raise DomainError(
                "EVIDENCE_INTEGRITY_FAILED",
                "Capability evidence is missing or changed.",
                item["capability_id"],
            )
        return item, bytes(evidence["exact_bytes"]).decode("utf-8")

    def consume_capability(self, capability_id: str, request_hash: str) -> None:
        with self.transaction() as conn:
            changed = conn.execute(
                "UPDATE downstream_capabilities SET status='consumed',request_hash=?,consumed_at=? "
                "WHERE capability_id=? AND status='issued'",
                (request_hash, utc_now(), capability_id),
            ).rowcount
            if changed != 1:
                row = conn.execute(
                    "SELECT request_hash,status FROM downstream_capabilities WHERE capability_id=?",
                    (capability_id,),
                ).fetchone()
                if row and row["status"] == "consumed" and row["request_hash"] == request_hash:
                    return
                raise DomainError("REPLAY_DETECTED", "Capability is no longer usable.", capability_id)

    def invalidate_capability(self, capability_id: str) -> None:
        """Revoke a durable capability without deleting its audit record."""

        with self.transaction() as conn:
            changed = conn.execute(
                "UPDATE downstream_capabilities SET status='invalidated' "
                "WHERE capability_id=? AND status IN ('issued','consumed')",
                (capability_id,),
            ).rowcount
            if changed != 1:
                raise DomainError(
                    "CAPABILITY_INVALID", "Capability cannot be invalidated.", capability_id
                )

    def capability_record(self, capability_id: str) -> dict[str, Any] | None:
        with self._connect(self.paths.marketplace) as conn:
            return _row(
                conn.execute(
                    "SELECT capability_id,status,workflow_id,task_id,order_id,audience,operation,exp "
                    "FROM downstream_capabilities WHERE capability_id=?",
                    (capability_id,),
                ).fetchone()
            )

    def save_merchant_task(
        self,
        *,
        workflow_id: str,
        task_id: str,
        context_id: str,
        order_id: str,
        task_json: str,
        task_digest: str,
        task_evidence_id: str,
        requirements_id: str,
        requirements_json: str,
        requirements_digest: str,
        requirements_evidence_id: str,
        checkout_jwt: str,
        checkout_hash: str,
        capability_id: str,
        expires_at: str,
        agent_card_digest: str,
    ) -> None:
        now = utc_now()
        with self.merchant_transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO merchant_tasks_v2(task_id,workflow_id,context_id,order_id,state,task_json,created_at,updated_at) VALUES(?,?,?,?,'input-required',?,?,?)",
                (task_id, workflow_id, context_id, order_id, task_json, now, now),
            )
            conn.execute(
                "INSERT OR IGNORE INTO merchant_requirements_v2(requirements_id,task_id,requirements_json,requirements_digest,checkout_jwt,checkout_hash,created_at) VALUES(?,?,?,?,?,?,?)",
                (requirements_id, task_id, requirements_json, requirements_digest, checkout_jwt, checkout_hash, now),
            )
            conn.execute(
                "INSERT OR IGNORE INTO merchant_capability_consumptions_v2(capability_id,task_id,operation,request_digest,consumed_at) VALUES(?,?,'merchant-task:start',?,?)",
                (capability_id, task_id, task_digest, now),
            )
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO merchant_task_mirrors(task_id,workflow_id,context_id,merchant_id,order_id,profile_id,observed_state,observed_version,task_evidence_id,task_evidence_digest,agent_card_digest,onboarding_version,created_at,updated_at)"
                " VALUES(?,?,?,? ,?,'x402-wire-simulation/1','input-required',1,?,?,?,'simulation-v1',?,?)",
                (task_id, workflow_id, context_id, "demo-merchant", order_id, task_evidence_id, task_digest, agent_card_digest, now, now),
            )
            conn.execute(
                "INSERT OR IGNORE INTO payment_requirements(requirements_id,task_id,profile_id,evidence_id,evidence_digest,checkout_hash,capability_id,expires_at)"
                " VALUES(?,?,'x402-wire-simulation/1',?,?,?,?,?)",
                (requirements_id, task_id, requirements_evidence_id, requirements_digest, checkout_hash, capability_id, expires_at),
            )

    def save_merchant_origin(
        self,
        *,
        workflow_id: str,
        task_id: str,
        context_id: str,
        order_id: str,
        task: dict[str, Any],
        requirements_id: str,
        requirements: dict[str, Any],
        checkout_jwt: str,
        checkout_hash: str,
        capability_id: str,
    ) -> None:
        """Persist the Merchant-owned A2A Task before returning over HTTP."""

        now = utc_now()
        task_json = canonical_json(task)
        requirements_json = canonical_json(requirements)
        with self.merchant_transaction() as conn:
            existing = conn.execute(
                "SELECT task_json FROM merchant_tasks_v2 WHERE task_id=?", (task_id,)
            ).fetchone()
            if existing and existing["task_json"] != task_json:
                raise DomainError(
                    "IDEMPOTENCY_CONFLICT", "Merchant Task replay changed.", task_id
                )
            conn.execute(
                "INSERT OR IGNORE INTO merchant_tasks_v2"
                "(task_id,workflow_id,context_id,order_id,state,task_json,created_at,updated_at) "
                "VALUES(?,?,?,?,'input-required',?,?,?)",
                (task_id, workflow_id, context_id, order_id, task_json, now, now),
            )
            conn.execute(
                "INSERT OR IGNORE INTO merchant_requirements_v2"
                "(requirements_id,task_id,requirements_json,requirements_digest,checkout_jwt,checkout_hash,created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    requirements_id,
                    task_id,
                    requirements_json,
                    sha256_digest(requirements_json),
                    checkout_jwt,
                    checkout_hash,
                    now,
                ),
            )
            conn.execute(
                "INSERT OR IGNORE INTO merchant_capability_consumptions_v2"
                "(capability_id,task_id,operation,request_digest,consumed_at) "
                "VALUES(?,?,'merchant-task:start',?,?)",
                (capability_id, task_id, sha256_digest(task_json), now),
            )

    def merchant_task(self, task_id: str) -> dict[str, Any]:
        with self._connect(self.paths.merchant) as conn:
            task = _row(conn.execute("SELECT * FROM merchant_tasks_v2 WHERE task_id=?", (task_id,)).fetchone())
            requirement = _row(conn.execute("SELECT * FROM merchant_requirements_v2 WHERE task_id=?", (task_id,)).fetchone())
        with self._connect(self.paths.marketplace) as conn:
            mediation_requirement = _row(
                conn.execute(
                    "SELECT evidence_id,evidence_digest,expires_at,used_at FROM payment_requirements WHERE task_id=?",
                    (task_id,),
                ).fetchone()
            )
        if task is None or requirement is None:
            raise KeyError(task_id)
        task["task"] = json.loads(task.pop("task_json"))
        requirement["requirements"] = json.loads(requirement.pop("requirements_json"))
        if mediation_requirement:
            requirement.update(mediation_requirement)
        task["requirement"] = requirement
        return task

    def original_merchant_task(
        self, workflow_id: str, *, tenant_id: str, actor_id: str
    ) -> dict[str, Any]:
        """Read the immutable Merchant task snapshot used for authorization."""

        with self._connect(self.paths.marketplace) as conn:
            row = conn.execute(
                "SELECT task_evidence_id FROM merchant_task_mirrors WHERE workflow_id=?",
                (workflow_id,),
            ).fetchone()
        if row is None:
            raise KeyError(workflow_id)
        exact = self.read_evidence(
            row["task_evidence_id"],
            actor_id=actor_id,
            actor_role="customer",
            tenant_id=tenant_id,
        )
        return json.loads(exact)

    def append_merchant_message(
        self,
        *,
        message_id: str,
        task_id: str,
        context_id: str,
        status: str,
        message: dict[str, Any],
    ) -> None:
        wire = canonical_json(message)
        digest = sha256_digest(wire)
        with self.merchant_transaction() as conn:
            existing = conn.execute("SELECT request_digest FROM merchant_messages_v2 WHERE message_id=?", (message_id,)).fetchone()
            if existing and existing["request_digest"] != digest:
                raise DomainError("IDEMPOTENCY_CONFLICT", "Merchant message changed.", message_id)
            conn.execute(
                "INSERT OR IGNORE INTO merchant_messages_v2(message_id,task_id,context_id,status,message_json,request_digest,created_at) VALUES(?,?,?,?,?,?,?)",
                (message_id, task_id, context_id, status, wire, digest, utc_now()),
            )

    def save_fulfillment(
        self,
        *,
        operation_id: str,
        task_id: str,
        phase: str,
        request_digest: str,
        state: str,
        result: dict[str, Any],
    ) -> None:
        now = utc_now()
        with self.merchant_transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO merchant_operations_v2(operation_id,task_id,phase,request_digest,state,result_json,created_at) VALUES(?,?,?,?,?,?,?)",
                (operation_id, task_id, phase, request_digest, state, canonical_json(result), now),
            )
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO fulfillment_operations(operation_id,task_id,phase,state,request_digest,external_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (operation_id, task_id, phase, state, request_digest, operation_id, now, now),
            )

    def settle_simulation(
        self,
        *,
        attempt_id: str,
        task_id: str,
        idempotency_key: str,
        amount: int,
        request_digest: str,
        receipt: dict[str, Any],
        receipt_evidence_id: str,
        receipt_evidence_digest: str,
        unknown: bool = False,
        fail: bool = False,
    ) -> dict[str, Any]:
        external_id = f"sim:{attempt_id}"
        now = utc_now()
        with self.transaction() as conn:
            existing = conn.execute("SELECT * FROM settlement_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if existing:
                if existing["request_digest"] != request_digest:
                    raise DomainError("IDEMPOTENCY_CONFLICT", "Settlement input changed.", attempt_id)
                return dict(existing)
            state = "unknown" if unknown else "failed" if fail else "settled"
            applied = 0
            if state == "settled":
                debited = conn.execute(
                    "UPDATE rail_accounts_v2 SET balance=balance-?,version=version+1,updated_at=? WHERE account_id='demo-customer' AND asset='USD' AND balance>=?",
                    (amount, now, amount),
                ).rowcount
                if debited != 1:
                    state = "failed"
                else:
                    conn.execute(
                        "UPDATE rail_accounts_v2 SET balance=balance+?,version=version+1,updated_at=? WHERE account_id='demo-merchant' AND asset='USD'",
                        (amount, now),
                    )
                    applied = 1
            conn.execute(
                "INSERT INTO rail_operations_v2(operation_id,kind,source_id,payer,payee,amount,asset,state,applied,idempotency_key,request_digest,external_id,created_at,updated_at)"
                " VALUES(?,'settle',?,'demo-customer','demo-merchant',?,'USD',?,?,?,?,?,?,?)",
                (attempt_id, task_id, amount, state, applied, idempotency_key, request_digest, external_id, now, now),
            )
            ordinal = conn.execute("SELECT COALESCE(MAX(ordinal),0)+1 FROM settlement_attempts WHERE task_id=?", (task_id,)).fetchone()[0]
            conn.execute(
                "INSERT INTO settlement_attempts(attempt_id,task_id,ordinal,profile_id,idempotency_key,request_digest,external_id,state,network,transaction_ref,receipt_evidence_id,receipt_evidence_digest,created_at,resolved_at)"
                " VALUES(?,?,?,'x402-wire-simulation/1',?,?,?,?,'demo:local',?,?,?,?,?)",
                (attempt_id, task_id, ordinal, idempotency_key, request_digest, external_id, state, external_id if state == "settled" else None, receipt_evidence_id, receipt_evidence_digest, now, now if state != "unknown" else None),
            )
            conn.execute(
                "INSERT INTO settlement_attempt_events(event_id,attempt_id,seq,observed_state,network,transaction_ref,error_code,evidence_id,evidence_digest,created_at) VALUES(?,?,1,?,'demo:local',?,?,?,?,?)",
                (f"attempt-event:{attempt_id}:1", attempt_id, state, external_id if state == "settled" else None, None if state == "settled" else "SETTLEMENT_FAILED", receipt_evidence_id, receipt_evidence_digest, now),
            )
            conn.execute(
                "INSERT INTO profile_receipts(receipt_id,task_id,attempt_id,ordinal,success,network,transaction_ref,error_code,evidence_id,evidence_digest,created_at)"
                " VALUES(?,?,?,?,?,'demo:local',?,?,?,?,?)",
                (f"profile-receipt:{attempt_id}", task_id, attempt_id, ordinal, int(state == "settled"), external_id if state == "settled" else None, None if state == "settled" else "SETTLEMENT_FAILED", receipt_evidence_id, receipt_evidence_digest, now),
            )
            result = conn.execute("SELECT * FROM settlement_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        return dict(result)

    def profile_receipts(self, task_id: str) -> list[dict[str, Any]]:
        with self._connect(self.paths.marketplace) as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM profile_receipts WHERE task_id=? ORDER BY ordinal", (task_id,))]

    def latest_settlement(self, task_id: str) -> dict[str, Any]:
        with self._connect(self.paths.marketplace) as conn:
            row = _row(
                conn.execute(
                    "SELECT * FROM settlement_attempts WHERE task_id=? ORDER BY ordinal DESC LIMIT 1",
                    (task_id,),
                ).fetchone()
            )
        if row is None:
            raise KeyError(task_id)
        return row

    def require_refund(
        self,
        *,
        refund_id: str,
        workflow_id: str,
        attempt_id: str,
        original_payment_id: str,
        reason: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO refunds_v2(refund_id,workflow_id,attempt_id,original_payment_id,amount,currency,reason,provider_ref,state,idempotency_key,created_at,updated_at) "
                "VALUES(?,?,?,?,1250,'USD',?,?,'required',?,?,?)",
                (
                    refund_id,
                    workflow_id,
                    attempt_id,
                    original_payment_id,
                    reason,
                    f"sim-refund:{refund_id}",
                    idempotency_key,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM refunds_v2 WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
        return dict(row)

    def refund_simulation(
        self,
        *,
        workflow_id: str,
        idempotency_key: str,
        outcome: str = "settled",
    ) -> dict[str, Any]:
        if outcome not in {"settled", "failed", "unknown"}:
            raise ValueError("invalid refund outcome")
        now = utc_now()
        with self.transaction() as conn:
            refund = conn.execute(
                "SELECT * FROM refunds_v2 WHERE workflow_id=? ORDER BY created_at DESC LIMIT 1",
                (workflow_id,),
            ).fetchone()
            if refund is None:
                raise KeyError(workflow_id)
            existing = conn.execute(
                "SELECT * FROM rail_operations_v2 WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                if existing["source_id"] != refund["refund_id"]:
                    raise DomainError("IDEMPOTENCY_CONFLICT", "Refund key input changed.", workflow_id)
                return dict(existing)
            applied = 0
            if outcome == "settled":
                debited = conn.execute(
                    "UPDATE rail_accounts_v2 SET balance=balance-1250,version=version+1,updated_at=? "
                    "WHERE account_id='demo-merchant' AND asset='USD' AND balance>=1250",
                    (now,),
                ).rowcount
                if debited == 1:
                    conn.execute(
                        "UPDATE rail_accounts_v2 SET balance=balance+1250,version=version+1,updated_at=? "
                        "WHERE account_id='demo-customer' AND asset='USD'",
                        (now,),
                    )
                    applied = 1
                else:
                    outcome = "failed"
            operation_id = f"rail-refund:{refund['refund_id']}"
            external_id = refund["provider_ref"]
            request_digest = sha256_digest(
                canonical_json(
                    {
                        "refundId": refund["refund_id"],
                        "originalPaymentId": refund["original_payment_id"],
                        "amount": 1250,
                        "currency": "USD",
                    }
                )
            )
            conn.execute(
                "INSERT INTO rail_operations_v2(operation_id,kind,source_id,payer,payee,amount,asset,state,applied,idempotency_key,request_digest,external_id,created_at,updated_at) "
                "VALUES(?,'refund',?,'demo-merchant','demo-customer',1250,'USD',?,?,?,?,?,?,?)",
                (
                    operation_id,
                    refund["refund_id"],
                    outcome,
                    applied,
                    idempotency_key,
                    request_digest,
                    external_id,
                    now,
                    now,
                ),
            )
            conn.execute(
                "UPDATE refunds_v2 SET state=?,updated_at=? WHERE refund_id=? AND state='required'",
                ("refunded" if outcome == "settled" else outcome, now, refund["refund_id"]),
            )
            result = conn.execute(
                "SELECT * FROM rail_operations_v2 WHERE operation_id=?", (operation_id,)
            ).fetchone()
        return dict(result)

    def unknown_rail_operation(self, workflow_id: str) -> dict[str, Any]:
        with self._connect(self.paths.marketplace) as conn:
            workflow = conn.execute(
                "SELECT merchant_task_id FROM workflows WHERE workflow_id=?", (workflow_id,)
            ).fetchone()
            if workflow is None:
                raise KeyError(workflow_id)
            row = conn.execute(
                "SELECT * FROM rail_operations_v2 WHERE state='unknown' AND "
                "(source_id=? OR source_id IN (SELECT refund_id FROM refunds_v2 WHERE workflow_id=?)) "
                "ORDER BY created_at DESC LIMIT 1",
                (workflow["merchant_task_id"], workflow_id),
            ).fetchone()
        if row is None:
            raise DomainError(
                "RECONCILIATION_REQUIRED", "No unresolved simulation operation exists.", workflow_id
            )
        return dict(row)

    def reconcile_simulation_operation(
        self,
        *,
        workflow_id: str,
        operator_id: str,
        idempotency_key: str,
        observed_state: str,
        evidence_id: str,
        evidence_digest: str,
    ) -> dict[str, Any]:
        if observed_state not in {"settled", "failed"}:
            raise ValueError("reconciliation observation must be definitive")
        now = utc_now()
        with self.transaction() as conn:
            existing_action = conn.execute(
                "SELECT * FROM reconciliation_actions WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if existing_action:
                operation = conn.execute(
                    "SELECT * FROM rail_operations_v2 WHERE external_id=?",
                    (existing_action["external_id"],),
                ).fetchone()
                return dict(operation)
            workflow = conn.execute(
                "SELECT merchant_task_id FROM workflows WHERE workflow_id=?", (workflow_id,)
            ).fetchone()
            operation = conn.execute(
                "SELECT * FROM rail_operations_v2 WHERE state='unknown' AND "
                "(source_id=? OR source_id IN (SELECT refund_id FROM refunds_v2 WHERE workflow_id=?)) "
                "ORDER BY created_at DESC LIMIT 1",
                (workflow["merchant_task_id"], workflow_id),
            ).fetchone()
            if operation is None:
                raise DomainError(
                    "RECONCILIATION_REQUIRED", "No unresolved simulation operation exists.", workflow_id
                )
            applied = 0
            if observed_state == "settled" and operation["kind"] == "settle":
                debited = conn.execute(
                    "UPDATE rail_accounts_v2 SET balance=balance-1250,version=version+1,updated_at=? "
                    "WHERE account_id='demo-customer' AND asset='USD' AND balance>=1250",
                    (now,),
                ).rowcount
                if debited != 1:
                    observed_state = "failed"
                else:
                    conn.execute(
                        "UPDATE rail_accounts_v2 SET balance=balance+1250,version=version+1,updated_at=? "
                        "WHERE account_id='demo-merchant' AND asset='USD'",
                        (now,),
                    )
                    applied = 1
            elif observed_state == "settled" and operation["kind"] == "refund":
                debited = conn.execute(
                    "UPDATE rail_accounts_v2 SET balance=balance-1250,version=version+1,updated_at=? "
                    "WHERE account_id='demo-merchant' AND asset='USD' AND balance>=1250",
                    (now,),
                ).rowcount
                if debited != 1:
                    observed_state = "failed"
                else:
                    conn.execute(
                        "UPDATE rail_accounts_v2 SET balance=balance+1250,version=version+1,updated_at=? "
                        "WHERE account_id='demo-customer' AND asset='USD'",
                        (now,),
                    )
                    applied = 1
            conn.execute(
                "UPDATE rail_operations_v2 SET state=?,applied=?,updated_at=? WHERE operation_id=? AND state='unknown'",
                (observed_state, applied, now, operation["operation_id"]),
            )
            if operation["kind"] == "settle":
                conn.execute(
                    "UPDATE settlement_attempts SET state=?,transaction_ref=?,receipt_evidence_id=?,receipt_evidence_digest=?,resolved_at=? WHERE attempt_id=? AND state='unknown'",
                    (
                        observed_state,
                        operation["external_id"] if observed_state == "settled" else None,
                        evidence_id,
                        evidence_digest,
                        now,
                        operation["operation_id"],
                    ),
                )
                seq = conn.execute(
                    "SELECT COALESCE(MAX(seq),0)+1 FROM settlement_attempt_events WHERE attempt_id=?",
                    (operation["operation_id"],),
                ).fetchone()[0]
                conn.execute(
                    "INSERT INTO settlement_attempt_events(event_id,attempt_id,seq,observed_state,network,transaction_ref,error_code,evidence_id,evidence_digest,created_at) "
                    "VALUES(?,?,?,?,'demo:local',?,?,?,?,?)",
                    (
                        f"attempt-event:{operation['operation_id']}:{seq}",
                        operation["operation_id"],
                        seq,
                        observed_state,
                        operation["external_id"] if observed_state == "settled" else None,
                        None if observed_state == "settled" else "SETTLEMENT_FAILED",
                        evidence_id,
                        evidence_digest,
                        now,
                    ),
                )
                ordinal = conn.execute(
                    "SELECT COALESCE(MAX(ordinal),0)+1 FROM profile_receipts WHERE task_id=?",
                    (workflow["merchant_task_id"],),
                ).fetchone()[0]
                conn.execute(
                    "INSERT INTO profile_receipts(receipt_id,task_id,attempt_id,ordinal,success,network,transaction_ref,error_code,evidence_id,evidence_digest,created_at) "
                    "VALUES(?,?,?,?,?,'demo:local',?,?,?,?,?)",
                    (
                        f"profile-receipt:{operation['operation_id']}:{ordinal}",
                        workflow["merchant_task_id"],
                        operation["operation_id"],
                        ordinal,
                        int(observed_state == "settled"),
                        operation["external_id"] if observed_state == "settled" else None,
                        None if observed_state == "settled" else "SETTLEMENT_FAILED",
                        evidence_id,
                        evidence_digest,
                        now,
                    ),
                )
            else:
                conn.execute(
                    "UPDATE refunds_v2 SET state=?,updated_at=? WHERE refund_id=? AND state='unknown'",
                    ("refunded" if observed_state == "settled" else "required", now, operation["source_id"]),
                )
            conn.execute(
                "INSERT INTO reconciliation_actions(action_id,workflow_id,target_type,target_id,actor_id,actor_role,reason,external_id,observed_state,evidence_digest,idempotency_key,created_at) "
                "VALUES(?,?,?,?,?,'operator','authoritative-simulation-query',?,?,?,?,?)",
                (
                    f"reconcile:{operation['operation_id']}:{observed_state}",
                    workflow_id,
                    operation["kind"],
                    operation["source_id"],
                    operator_id,
                    operation["external_id"],
                    observed_state,
                    evidence_digest,
                    idempotency_key,
                    now,
                ),
            )
            result = conn.execute(
                "SELECT * FROM rail_operations_v2 WHERE operation_id=?",
                (operation["operation_id"],),
            ).fetchone()
        return dict(result)

    def complete_merchant_task(self, task_id: str, task: dict[str, Any]) -> None:
        with self.merchant_transaction() as conn:
            row = conn.execute("SELECT version FROM merchant_tasks_v2 WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(task_id)
            conn.execute(
                "UPDATE merchant_tasks_v2 SET state='completed',task_json=?,version=version+1,updated_at=? WHERE task_id=? AND version=?",
                (canonical_json(task), utc_now(), task_id, row["version"]),
            )
        with self.transaction() as conn:
            conn.execute(
                "UPDATE merchant_task_mirrors SET observed_state='completed',observed_version=observed_version+1,updated_at=? WHERE task_id=?",
                (utc_now(), task_id),
            )

    def insert_artifact_reference(
        self,
        *,
        artifact_id: str,
        workflow_id: str,
        task_id: str,
        kind: str,
        evidence_id: str,
        evidence_digest: str,
        issuer: str,
        kid: str,
        trust_snapshot_id: str,
        reference_digest: str | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO payment_artifacts(artifact_id,workflow_id,task_id,kind,evidence_id,evidence_digest,issuer,kid,trust_snapshot_id,reference_digest,created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (artifact_id, workflow_id, task_id, kind, evidence_id, evidence_digest, issuer, kid, trust_snapshot_id, reference_digest, utc_now()),
            )

    def ensure_trust_snapshot(
        self,
        *,
        snapshot_id: str,
        issuer: str,
        kid: str,
        evidence_id: str,
        evidence_digest: str,
    ) -> str:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO trust_snapshots(snapshot_id,issuer,kid,jwks_evidence_id,jwks_evidence_digest,onboarding_version,valid_at,created_at) "
                "VALUES(?,?,?,?,?,'demo-es256-v1',?,?)",
                (snapshot_id, issuer, kid, evidence_id, evidence_digest, utc_now(), utc_now()),
            )
            row = conn.execute(
                "SELECT snapshot_id,jwks_evidence_digest FROM trust_snapshots WHERE issuer=? AND kid=? AND onboarding_version='demo-es256-v1'",
                (issuer, kid),
            ).fetchone()
            if row["jwks_evidence_digest"] != evidence_digest:
                raise DomainError("IDEMPOTENCY_CONFLICT", "Trust snapshot changed.", snapshot_id)
        return str(row["snapshot_id"])

    def artifact_refs(self, workflow_id: str) -> list[dict[str, Any]]:
        with self._connect(self.paths.marketplace) as conn:
            return [dict(row) for row in conn.execute(
                "SELECT artifact_id,kind,evidence_id,evidence_digest,issuer,kid,trust_snapshot_id,reference_digest FROM payment_artifacts WHERE workflow_id=? ORDER BY created_at,artifact_id",
                (workflow_id,),
            )]

    def artifact_exact(self, workflow_id: str, kind: str) -> tuple[dict[str, Any], bytes] | None:
        with self._connect(self.paths.marketplace) as conn:
            row = conn.execute(
                "SELECT * FROM payment_artifacts WHERE workflow_id=? AND kind=?",
                (workflow_id, kind),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        with self._connect(self.paths.evidence) as conn:
            evidence = conn.execute(
                "SELECT exact_bytes,digest FROM evidence WHERE evidence_id=?",
                (item["evidence_id"],),
            ).fetchone()
        if evidence is None or evidence["digest"] != item["evidence_digest"]:
            raise DomainError(
                "EVIDENCE_INTEGRITY_FAILED",
                "Payment artifact evidence is missing or changed.",
                item["artifact_id"],
            )
        return item, bytes(evidence["exact_bytes"])

    def counts(self, workflow_id: str) -> dict[str, int]:
        with self._connect(self.paths.marketplace) as conn:
            task = conn.execute("SELECT merchant_task_id FROM workflows WHERE workflow_id=?", (workflow_id,)).fetchone()
            task_id = task[0] if task else None
            return {
                "planApprovals": conn.execute("SELECT COUNT(*) FROM plan_approvals WHERE workflow_id=?", (workflow_id,)).fetchone()[0],
                "paymentApprovals": conn.execute("SELECT COUNT(*) FROM payment_approvals WHERE workflow_id=?", (workflow_id,)).fetchone()[0],
                "paymentArtifacts": conn.execute("SELECT COUNT(*) FROM payment_artifacts WHERE workflow_id=?", (workflow_id,)).fetchone()[0],
                "settlements": conn.execute("SELECT COUNT(*) FROM settlement_attempts WHERE task_id=?", (task_id,)).fetchone()[0] if task_id else 0,
                "refunds": conn.execute("SELECT COUNT(*) FROM refunds_v2 WHERE workflow_id=?", (workflow_id,)).fetchone()[0],
            }

    def events(self, workflow_id: str) -> list[dict[str, Any]]:
        with self._connect(self.paths.marketplace) as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM workflow_events WHERE workflow_id=? ORDER BY seq", (workflow_id,))]

    def plan_approval(self, workflow_id: str) -> dict[str, Any]:
        with self._connect(self.paths.marketplace) as conn:
            row = _row(conn.execute("SELECT * FROM plan_approvals WHERE workflow_id=?", (workflow_id,)).fetchone())
        if row is None:
            raise KeyError(workflow_id)
        return row

    def payment_approval(self, workflow_id: str) -> dict[str, Any]:
        with self._connect(self.paths.marketplace) as conn:
            row = _row(conn.execute("SELECT * FROM payment_approvals WHERE workflow_id=?", (workflow_id,)).fetchone())
        if row is None:
            raise KeyError(workflow_id)
        return row

    def rail_balance(self, account_id: str) -> int:
        with self._connect(self.paths.marketplace) as conn:
            row = conn.execute("SELECT balance FROM rail_accounts_v2 WHERE account_id=? AND asset='USD'", (account_id,)).fetchone()
            return int(row[0])
