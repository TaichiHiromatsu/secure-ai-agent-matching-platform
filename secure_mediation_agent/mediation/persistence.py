"""Encrypted, owner-bound SQLite persistence for the mediation state machine."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import ValidationError

from secure_mediation_agent.workflow.migrations import SCHEMA_VERSION, utc_now
from secure_mediation_agent.workflow.repository import WorkflowRepository

from .canonical import canonical_bytes, canonical_digest
from .errors import MediationError, SecurityBlocked
from .models import ACTIVE_STATES, MediationPublicView, MediationSession, SubjectScope
from .persistence_models import (
    RequestReservation,
    StoreReadiness,
    build_local_durable_view,
)


MEDIATION_STORE_KEY_FILE_ENV = "MEDIATION_STORE_KEY_FILE"
DEFAULT_MEDIATION_STORE_KEY_FILE = Path("/run/secrets/ap2-demo/mediation-store.key")
KEY_VERSION = 1
SESSION_SCHEMA_VERSION = "mediation-session-projection/1"
VIEW_SCHEMA_VERSION = "mediation-public-view/1"
_NONCE_SIZE = 12
_HKDF_SALT = b"enterprise-a2a-pf/mediation-store/v1"
_REDACTED_AUDIENCE = "persisted-redacted"
_REDACTED_NONCE = "persisted-redacted-nonce"
_SAFE_RESULT_SCHEMA = "mediation-safe-result/1"
_SENTINEL_SCOPE_KEY = "__mediation_store_sentinel_v4__"
_SENTINEL_REQUEST_ID = "__key_check__"
_SENTINEL_SCHEMA = "mediation-store-sentinel/1"
_SENTINEL_SESSION_ID = "mediation-store-sentinel"


def load_mediation_store_key(path: str | Path) -> bytes:
    """Load one dedicated raw 32-byte key from a private regular file."""

    source = Path(path).expanduser()
    if source.is_symlink():
        raise RuntimeError("mediation store key must not be a symbolic link")
    try:
        metadata = source.stat()
        value = source.read_bytes()
    except OSError as error:
        raise RuntimeError("mediation store key is not readable") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("mediation store key must be a regular file")
    if metadata.st_mode & 0o077:
        raise RuntimeError("mediation store key permissions expose it to other users")
    if len(value) != 32:
        raise RuntimeError("mediation store key must contain exactly 32 raw bytes")
    return value


def resolve_mediation_store_key() -> bytes:
    configured = os.environ.get(MEDIATION_STORE_KEY_FILE_ENV)
    return load_mediation_store_key(configured or DEFAULT_MEDIATION_STORE_KEY_FILE)


def _derived_key(master_key: bytes, info: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_HKDF_SALT,
        info=info,
    ).derive(master_key)


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _safe_result(value: Any) -> dict[str, Any]:
    """Project arbitrary remote output into a small, typed, secret-free allowlist."""

    already_safe = (
        isinstance(value, dict)
        and value.get("schemaVersion") == _SAFE_RESULT_SCHEMA
        and _is_digest(value.get("sourceDigest"))
        and set(value).issubset(
            {
                "schemaVersion",
                "sourceDigest",
                "refundEligible",
                "taskState",
                "simulation",
                "refundState",
                "refundResultDigest",
            }
        )
    )
    source_digest = value["sourceDigest"] if already_safe else canonical_digest(value)
    source = value if isinstance(value, dict) else {}
    output: dict[str, Any] = {
        "schemaVersion": _SAFE_RESULT_SCHEMA,
        "sourceDigest": source_digest,
    }
    if isinstance(source.get("refundEligible"), bool):
        output["refundEligible"] = source["refundEligible"]
    if source.get("taskState") in {"completed", "failed"}:
        output["taskState"] = source["taskState"]
    if source.get("simulation") is True:
        output["simulation"] = True
    if source.get("refundState") == "refunded":
        output["refundState"] = "refunded"
    if _is_digest(source.get("refundResultDigest")):
        output["refundResultDigest"] = source["refundResultDigest"]
    return output


def _safe_payment_required(requirement: dict[str, Any]) -> dict[str, Any]:
    raw = requirement.get("paymentRequired")
    accepts = raw.get("accepts") if isinstance(raw, dict) else None
    accept = accepts[0] if isinstance(accepts, list) and len(accepts) == 1 else None
    amount = requirement.get("amountMinor")
    expected = {
        "scheme": "exact-simulated",
        "network": "demo:local",
        "asset": "USD",
        "payTo": "merchant:demo-merchant",
        "maxAmountRequired": str(amount),
    }
    if not isinstance(raw, dict) or raw.get("x402Version") != 1 or accept != expected:
        raise MediationError(
            "MEDIATION_STORE_INTEGRITY",
            "The persisted payment requirement is not an allowed public projection.",
        )
    return {"x402Version": 1, "accepts": [expected]}


def _project_requirement(requirement: dict[str, Any]) -> None:
    requirement["paymentRequired"] = _safe_payment_required(requirement)
    for key in (
        "checkoutAudience",
        "checkoutNonce",
        "paymentAudience",
        "paymentNonce",
    ):
        requirement.pop(key, None)


def _session_projection(session: MediationSession) -> dict[str, Any]:
    payload = session.model_dump(mode="json", by_alias=True, exclude_none=True)
    continuation = payload.get("continuation")
    if isinstance(continuation, dict):
        requirement = continuation.get("requirement")
        if not isinstance(requirement, dict):
            raise MediationError(
                "MEDIATION_STORE_INTEGRITY",
                "The mediation continuation is incomplete.",
            )
        _project_requirement(requirement)
        remote = continuation.get("remoteTask")
        if isinstance(remote, dict):
            remote_requirement = remote.get("paymentRequirement")
            if isinstance(remote_requirement, dict):
                _project_requirement(remote_requirement)
            artifact = remote.get("artifact")
            if artifact is not None:
                already_projected = (
                    isinstance(artifact, dict)
                    and set(artifact) == {"artifactDigest"}
                    and isinstance(artifact["artifactDigest"], str)
                    and artifact["artifactDigest"].startswith("sha256:")
                    and len(artifact["artifactDigest"]) == 71
                )
                if not already_projected:
                    remote["artifact"] = {
                        "artifactDigest": canonical_digest(artifact)
                    }
    if payload.get("result") is not None:
        payload["result"] = _safe_result(payload["result"])
    return payload


def _hydrate_projection(payload: dict[str, Any]) -> MediationSession:
    hydrated = json.loads(json.dumps(payload, ensure_ascii=False))
    continuation = hydrated.get("continuation")
    if isinstance(continuation, dict):
        requirements = [continuation.get("requirement")]
        remote = continuation.get("remoteTask")
        if isinstance(remote, dict):
            requirements.append(remote.get("paymentRequirement"))
        for requirement in requirements:
            if not isinstance(requirement, dict):
                continue
            requirement["checkoutAudience"] = _REDACTED_AUDIENCE
            requirement["checkoutNonce"] = _REDACTED_NONCE
            requirement["paymentAudience"] = _REDACTED_AUDIENCE
            requirement["paymentNonce"] = _REDACTED_NONCE
    return MediationSession.model_validate_json(
        json.dumps(hydrated, ensure_ascii=False, separators=(",", ":"))
    )


class SqliteMediationStore:
    """WAL-backed same-host durable store with encrypted owner-bound rows."""

    kind = "sqlite"
    durability_profile = "local-durable"

    def __init__(
        self,
        repository: WorkflowRepository,
        master_key: bytes | str | Path,
    ) -> None:
        key = (
            bytes(master_key)
            if isinstance(master_key, (bytes, bytearray))
            else load_mediation_store_key(master_key)
        )
        if len(key) != 32:
            raise RuntimeError("mediation store master key must be exactly 32 bytes")
        self.repository = repository
        self.path = repository.paths.marketplace
        self._index_key = _derived_key(key, b"scope-index")
        self._session_cipher = AESGCM(_derived_key(key, b"session-aead"))
        self._view_cipher = AESGCM(_derived_key(key, b"view-aead"))
        self._verify_schema()
        self._ensure_sentinel()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _verify_schema(self) -> None:
        try:
            with self._connect() as conn:
                version = conn.execute(
                    "SELECT MAX(version) FROM schema_migrations"
                ).fetchone()[0]
                tables = {
                    row["name"]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
        except sqlite3.Error as error:
            raise RuntimeError("mediation store schema is unavailable") from error
        if int(version or 0) != SCHEMA_VERSION or not {
            "mediation_sessions_v4",
            "mediation_requests_v4",
        }.issubset(tables):
            raise RuntimeError("mediation store schema version mismatch")

    def _scope_key(self, scope: SubjectScope) -> str:
        identity = {
            "adkSessionId": scope.adk_session_id,
            "subject": scope.subject,
            "tenantId": scope.tenant_id,
        }
        return hmac.new(
            self._index_key, canonical_bytes(identity), hashlib.sha256
        ).hexdigest()

    @staticmethod
    def _session_aad(session_id: str, version: int, schema: str) -> bytes:
        return canonical_bytes(
            {
                "kind": "mediation-session",
                "mediationSessionId": session_id,
                "schemaVersion": schema,
                "version": version,
            }
        )

    @staticmethod
    def _view_aad(session_id: str, version: int, schema: str) -> bytes:
        return canonical_bytes(
            {
                "kind": "mediation-public-view",
                "mediationSessionId": session_id,
                "schemaVersion": schema,
                "version": version,
            }
        )

    @staticmethod
    def _request_view_aad(
        scope_key: str,
        request_id: str,
        request_digest: str,
        session_id: str,
        version: int,
        schema: str,
    ) -> bytes:
        return canonical_bytes(
            {
                "kind": "mediation-request-result",
                "scopeKey": scope_key,
                "requestId": request_id,
                "requestDigest": request_digest,
                "mediationSessionId": session_id,
                "schemaVersion": schema,
                "version": version,
            }
        )

    @staticmethod
    def _seal(cipher: AESGCM, plaintext: bytes, aad: bytes) -> tuple[bytes, bytes]:
        nonce = os.urandom(_NONCE_SIZE)
        return nonce, cipher.encrypt(nonce, plaintext, aad)

    @staticmethod
    def _open(
        cipher: AESGCM, nonce: bytes, ciphertext: bytes, aad: bytes
    ) -> bytes:
        if len(nonce) != _NONCE_SIZE:
            raise MediationError(
                "MEDIATION_STORE_INTEGRITY", "The durable mediation record is invalid."
            )
        try:
            return cipher.decrypt(nonce, ciphertext, aad)
        except (InvalidTag, ValueError) as error:
            raise MediationError(
                "MEDIATION_STORE_INTEGRITY", "The durable mediation record is invalid."
            ) from error

    def _sentinel_aad(self) -> bytes:
        return self._request_view_aad(
            _SENTINEL_SCOPE_KEY,
            _SENTINEL_REQUEST_ID,
            canonical_digest({"schemaVersion": _SENTINEL_SCHEMA}),
            _SENTINEL_SESSION_ID,
            0,
            _SENTINEL_SCHEMA,
        )

    def _verify_sentinel_row(self, row: sqlite3.Row) -> None:
        expected_digest = canonical_digest({"schemaVersion": _SENTINEL_SCHEMA})
        if (
            row["scope_key"] != _SENTINEL_SCOPE_KEY
            or row["request_id"] != _SENTINEL_REQUEST_ID
            or row["request_digest"] != expected_digest
            or row["status"] != "completed"
            or row["mediation_session_id"] != _SENTINEL_SESSION_ID
            or row["result_version"] != 0
            or row["result_view_schema_version"] != _SENTINEL_SCHEMA
            or row["result_key_version"] != KEY_VERSION
        ):
            raise RuntimeError("mediation store key sentinel is invalid")
        plaintext = self._open(
            self._view_cipher,
            bytes(row["result_view_nonce"]),
            bytes(row["result_view_ciphertext"]),
            self._sentinel_aad(),
        )
        try:
            payload = json.loads(plaintext)
        except (json.JSONDecodeError, TypeError) as error:
            raise RuntimeError("mediation store key sentinel is invalid") from error
        if (
            payload != {"schemaVersion": _SENTINEL_SCHEMA}
            or canonical_digest(payload) != row["result_view_digest"]
        ):
            raise RuntimeError("mediation store key sentinel is invalid")

    def _ensure_sentinel(self) -> None:
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM mediation_requests_v4 WHERE scope_key=? AND request_id=?",
                (_SENTINEL_SCOPE_KEY, _SENTINEL_REQUEST_ID),
            ).fetchone()
            if row is not None:
                self._verify_sentinel_row(row)
                return
            session_count = conn.execute(
                "SELECT COUNT(*) FROM mediation_sessions_v4 WHERE scope_key<>?",
                (_SENTINEL_SCOPE_KEY,),
            ).fetchone()[0]
            request_count = conn.execute(
                "SELECT COUNT(*) FROM mediation_requests_v4 WHERE scope_key<>?",
                (_SENTINEL_SCOPE_KEY,),
            ).fetchone()[0]
            if session_count or request_count:
                raise RuntimeError(
                    "pre-sentinel mediation v4 data requires explicit reset or migration"
                )
            payload = {"schemaVersion": _SENTINEL_SCHEMA}
            plaintext = canonical_bytes(payload)
            nonce, ciphertext = self._seal(
                self._view_cipher, plaintext, self._sentinel_aad()
            )
            now = utc_now()
            conn.execute(
                "INSERT INTO mediation_requests_v4(scope_key,request_id,request_digest,"
                "status,mediation_session_id,result_version,result_view_schema_version,"
                "result_key_version,result_view_nonce,result_view_ciphertext,"
                "result_view_digest,created_at,updated_at) "
                "VALUES(?,?,?,'completed',?,?,?,?,?,?,?,?,?)",
                (
                    _SENTINEL_SCOPE_KEY,
                    _SENTINEL_REQUEST_ID,
                    canonical_digest(payload),
                    _SENTINEL_SESSION_ID,
                    0,
                    _SENTINEL_SCHEMA,
                    KEY_VERSION,
                    nonce,
                    ciphertext,
                    canonical_digest(payload),
                    now,
                    now,
                ),
            )

    def _verify_sentinel(self) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM mediation_requests_v4 WHERE scope_key=? AND request_id=?",
                (_SENTINEL_SCOPE_KEY, _SENTINEL_REQUEST_ID),
            ).fetchone()
        if row is None:
            raise RuntimeError("mediation store key sentinel is missing")
        self._verify_sentinel_row(row)

    def _prepare_session(self, session: MediationSession) -> dict[str, Any]:
        projection = _session_projection(session)
        session_plaintext = canonical_bytes(projection)
        persisted = _hydrate_projection(projection)
        view = build_local_durable_view(persisted)
        view_plaintext = canonical_bytes(view)
        session_id = persisted.owner.mediation_session_id
        session_nonce, session_ciphertext = self._seal(
            self._session_cipher,
            session_plaintext,
            self._session_aad(session_id, persisted.version, SESSION_SCHEMA_VERSION),
        )
        view_nonce, view_ciphertext = self._seal(
            self._view_cipher,
            view_plaintext,
            self._view_aad(session_id, persisted.version, VIEW_SCHEMA_VERSION),
        )
        return {
            "session": persisted,
            "session_nonce": session_nonce,
            "session_ciphertext": session_ciphertext,
            "session_digest": canonical_digest(projection),
            "view_nonce": view_nonce,
            "view_ciphertext": view_ciphertext,
            "view_digest": canonical_digest(view),
        }

    def _decode_session(self, row: sqlite3.Row, scope: SubjectScope) -> MediationSession:
        if row["key_version"] != KEY_VERSION:
            raise MediationError(
                "MEDIATION_STORE_INTEGRITY", "The durable mediation key version is unsupported."
            )
        session_id = row["mediation_session_id"]
        version = row["version"]
        schema = row["session_schema_version"]
        if schema != SESSION_SCHEMA_VERSION:
            raise MediationError(
                "MEDIATION_STORE_INTEGRITY", "The durable mediation schema is unsupported."
            )
        plaintext = self._open(
            self._session_cipher,
            bytes(row["session_nonce"]),
            bytes(row["session_ciphertext"]),
            self._session_aad(session_id, version, schema),
        )
        try:
            payload = json.loads(plaintext)
            if canonical_digest(payload) != row["session_digest"]:
                raise ValueError("session digest mismatch")
            session = _hydrate_projection(payload)
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as error:
            raise MediationError(
                "MEDIATION_STORE_INTEGRITY", "The durable mediation record is invalid."
            ) from error
        if (
            session.owner.subject_scope != scope
            or not hmac.compare_digest(row["scope_key"], self._scope_key(scope))
            or session.owner.mediation_session_id != session_id
            or session.state.value != row["state"]
            or session.version != version
            or session.plan.plan_digest != row["plan_digest"]
            or session.approval_target_digest != row["approval_target_digest"]
        ):
            raise MediationError(
                "MEDIATION_STORE_INTEGRITY", "The durable mediation binding is invalid."
            )
        self._decode_session_view(row, session)
        return session

    def _decode_session_view(
        self, row: sqlite3.Row, session: MediationSession
    ) -> MediationPublicView:
        schema = row["view_schema_version"]
        if schema != VIEW_SCHEMA_VERSION:
            raise MediationError(
                "MEDIATION_STORE_INTEGRITY", "The durable public view schema is unsupported."
            )
        plaintext = self._open(
            self._view_cipher,
            bytes(row["view_nonce"]),
            bytes(row["view_ciphertext"]),
            self._view_aad(
                session.owner.mediation_session_id, session.version, schema
            ),
        )
        try:
            payload = json.loads(plaintext)
            view = MediationPublicView.model_validate_json(plaintext)
            if canonical_digest(payload) != row["view_digest"]:
                raise ValueError("view digest mismatch")
            if canonical_bytes(view) != canonical_bytes(build_local_durable_view(session)):
                raise ValueError("view/session mismatch")
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as error:
            raise MediationError(
                "MEDIATION_STORE_INTEGRITY", "The durable public view is invalid."
            ) from error
        return view

    @staticmethod
    def _copy(session: MediationSession) -> MediationSession:
        return session.model_copy(deep=True)

    def active_for(self, scope: SubjectScope) -> MediationSession | None:
        scope_key = self._scope_key(scope)
        states = tuple(state.value for state in ACTIVE_STATES)
        placeholders = ",".join("?" for _ in states)
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM mediation_sessions_v4 WHERE scope_key=? "
                f"AND state IN ({placeholders}) ORDER BY updated_at DESC,row_id DESC LIMIT 1",
                (scope_key, *states),
            ).fetchone()
        return None if row is None else self._decode_session(row, scope)

    def latest_for(self, scope: SubjectScope) -> MediationSession | None:
        scope_key = self._scope_key(scope)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM mediation_sessions_v4 WHERE scope_key=? "
                "ORDER BY updated_at DESC,row_id DESC LIMIT 1",
                (scope_key,),
            ).fetchone()
        return None if row is None else self._decode_session(row, scope)

    def get(self, mediation_session_id: str, scope: SubjectScope) -> MediationSession:
        scope_key = self._scope_key(scope)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM mediation_sessions_v4 "
                "WHERE scope_key=? AND mediation_session_id=?",
                (scope_key, mediation_session_id),
            ).fetchone()
        if row is None:
            raise SecurityBlocked(
                "MEDIATION_NOT_FOUND", "The active mediation session is not available."
            )
        return self._decode_session(row, scope)

    def save_new(self, session: MediationSession) -> None:
        if session.version != 0:
            raise MediationError(
                "INVALID_VERSION_INCREMENT", "A new mediation session must start at version zero."
            )
        prepared = self._prepare_session(session)
        persisted = prepared["session"]
        scope_key = self._scope_key(persisted.owner.subject_scope)
        now = utc_now()
        try:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT INTO mediation_sessions_v4("
                    "mediation_session_id,scope_key,state,version,plan_digest,"
                    "approval_target_digest,session_schema_version,key_version,"
                    "session_nonce,session_ciphertext,session_digest,view_schema_version,"
                    "view_nonce,view_ciphertext,view_digest,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        persisted.owner.mediation_session_id,
                        scope_key,
                        persisted.state.value,
                        persisted.version,
                        persisted.plan.plan_digest,
                        persisted.approval_target_digest,
                        SESSION_SCHEMA_VERSION,
                        KEY_VERSION,
                        prepared["session_nonce"],
                        prepared["session_ciphertext"],
                        prepared["session_digest"],
                        VIEW_SCHEMA_VERSION,
                        prepared["view_nonce"],
                        prepared["view_ciphertext"],
                        prepared["view_digest"],
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as error:
            code = (
                "ACTIVE_MEDIATION_EXISTS"
                if "scope_key" in str(error)
                else "MEDIATION_ID_CONFLICT"
            )
            message = (
                "This session already has an active mediation request."
                if code == "ACTIVE_MEDIATION_EXISTS"
                else "Mediation session identifier conflict."
            )
            raise MediationError(code, message) from error

    def compare_and_set(
        self, session: MediationSession, *, expected_version: int
    ) -> MediationSession:
        if session.version != expected_version + 1:
            raise MediationError(
                "INVALID_VERSION_INCREMENT",
                "The mediation session version increment is invalid.",
            )
        prepared = self._prepare_session(session)
        persisted = prepared["session"]
        scope = persisted.owner.subject_scope
        scope_key = self._scope_key(scope)
        session_id = persisted.owner.mediation_session_id
        try:
            with self._transaction() as conn:
                current = conn.execute(
                    "SELECT * FROM mediation_sessions_v4 "
                    "WHERE scope_key=? AND mediation_session_id=?",
                    (scope_key, session_id),
                ).fetchone()
                if current is None:
                    raise SecurityBlocked(
                        "MEDIATION_NOT_FOUND",
                        "The active mediation session is not available.",
                    )
                self._decode_session(current, scope)
                result = conn.execute(
                    "UPDATE mediation_sessions_v4 SET state=?,version=?,plan_digest=?,"
                    "approval_target_digest=?,session_schema_version=?,key_version=?,"
                    "session_nonce=?,session_ciphertext=?,session_digest=?,"
                    "view_schema_version=?,view_nonce=?,view_ciphertext=?,view_digest=?,"
                    "updated_at=? WHERE scope_key=? AND mediation_session_id=? "
                    "AND state=? AND version=?",
                    (
                        persisted.state.value,
                        persisted.version,
                        persisted.plan.plan_digest,
                        persisted.approval_target_digest,
                        SESSION_SCHEMA_VERSION,
                        KEY_VERSION,
                        prepared["session_nonce"],
                        prepared["session_ciphertext"],
                        prepared["session_digest"],
                        VIEW_SCHEMA_VERSION,
                        prepared["view_nonce"],
                        prepared["view_ciphertext"],
                        prepared["view_digest"],
                        utc_now(),
                        scope_key,
                        session_id,
                        current["state"],
                        expected_version,
                    ),
                )
                if result.rowcount != 1:
                    raise MediationError(
                        "STATE_TRANSITION_CONFLICT",
                        "The mediation session changed; refresh before retrying.",
                    )
        except sqlite3.IntegrityError as error:
            raise MediationError(
                "ACTIVE_MEDIATION_EXISTS",
                "This session already has an active mediation request.",
            ) from error
        return self._copy(persisted)

    def _decode_request(self, row: sqlite3.Row) -> RequestReservation:
        status = row["status"]
        if status == "processing":
            raise MediationError(
                "MEDIATION_REQUEST_IN_PROGRESS",
                "This request is already being processed.",
            )
        if status == "failed":
            raise MediationError(
                "MEDIATION_REQUEST_FAILED",
                "This request previously failed before a safe result was committed.",
            )
        if status != "completed" or row["result_key_version"] != KEY_VERSION:
            raise MediationError(
                "MEDIATION_STORE_INTEGRITY", "The durable request record is invalid."
            )
        session_id = row["mediation_session_id"]
        version = row["result_version"]
        schema = row["result_view_schema_version"]
        if not isinstance(session_id, str) or not isinstance(version, int) or schema != VIEW_SCHEMA_VERSION:
            raise MediationError(
                "MEDIATION_STORE_INTEGRITY", "The durable request result is invalid."
            )
        plaintext = self._open(
            self._view_cipher,
            bytes(row["result_view_nonce"]),
            bytes(row["result_view_ciphertext"]),
            self._request_view_aad(
                row["scope_key"],
                row["request_id"],
                row["request_digest"],
                session_id,
                version,
                schema,
            ),
        )
        try:
            payload = json.loads(plaintext)
            view = MediationPublicView.model_validate_json(plaintext)
            if canonical_digest(payload) != row["result_view_digest"] or view.version != version:
                raise ValueError("request result mismatch")
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as error:
            raise MediationError(
                "MEDIATION_STORE_INTEGRITY", "The durable request result is invalid."
            ) from error
        return RequestReservation(
            status="completed",
            mediation_session_id=session_id,
            result_version=version,
            view=view,
        )

    def load_request(
        self, scope: SubjectScope, request_id: str, request_digest: str
    ) -> RequestReservation | None:
        scope_key = self._scope_key(scope)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM mediation_requests_v4 WHERE scope_key=? AND request_id=?",
                (scope_key, request_id),
            ).fetchone()
        if row is None:
            return None
        if not hmac.compare_digest(row["request_digest"], request_digest):
            raise MediationError(
                "IDEMPOTENCY_CONFLICT",
                "The request identifier was reused with different content.",
            )
        return self._decode_request(row)

    def reserve_request(
        self,
        scope: SubjectScope,
        request_id: str,
        request_digest: str,
        expected_version: int | None = None,
    ) -> RequestReservation:
        scope_key = self._scope_key(scope)
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM mediation_requests_v4 WHERE scope_key=? AND request_id=?",
                (scope_key, request_id),
            ).fetchone()
            if row is not None:
                if not hmac.compare_digest(row["request_digest"], request_digest):
                    raise MediationError(
                        "IDEMPOTENCY_CONFLICT",
                        "The request identifier was reused with different content.",
                    )
                return self._decode_request(row)
            if expected_version is not None:
                states = tuple(state.value for state in ACTIVE_STATES)
                placeholders = ",".join("?" for _ in states)
                active = conn.execute(
                    f"SELECT version FROM mediation_sessions_v4 WHERE scope_key=? "
                    f"AND state IN ({placeholders}) ORDER BY updated_at DESC,row_id DESC LIMIT 1",
                    (scope_key, *states),
                ).fetchone()
                if active is None or active["version"] != expected_version:
                    raise MediationError(
                        "STATE_TRANSITION_CONFLICT",
                        "The mediation session changed; refresh before retrying.",
                    )
            now = utc_now()
            conn.execute(
                "INSERT INTO mediation_requests_v4(scope_key,request_id,request_digest,"
                "status,created_at,updated_at) VALUES(?,?,?,'processing',?,?)",
                (scope_key, request_id, request_digest, now, now),
            )
        return RequestReservation(status="reserved")

    def complete_request(
        self,
        scope: SubjectScope,
        request_id: str,
        request_digest: str,
        session: MediationSession,
        view: MediationPublicView,
    ) -> None:
        if session.owner.subject_scope != scope or view.version != session.version:
            raise MediationError(
                "MEDIATION_STORE_INTEGRITY", "The request result owner or version is invalid."
            )
        expected_view = build_local_durable_view(_hydrate_projection(_session_projection(session)))
        if canonical_bytes(view) != canonical_bytes(expected_view):
            raise MediationError(
                "MEDIATION_STORE_INTEGRITY", "The request public result is invalid."
            )
        scope_key = self._scope_key(scope)
        plaintext = canonical_bytes(view)
        nonce, ciphertext = self._seal(
            self._view_cipher,
            plaintext,
            self._request_view_aad(
                scope_key,
                request_id,
                request_digest,
                session.owner.mediation_session_id,
                session.version,
                VIEW_SCHEMA_VERSION,
            ),
        )
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM mediation_requests_v4 WHERE scope_key=? AND request_id=?",
                (scope_key, request_id),
            ).fetchone()
            if row is None:
                raise MediationError(
                    "MEDIATION_REQUEST_NOT_RESERVED", "The request was not reserved."
                )
            if not hmac.compare_digest(row["request_digest"], request_digest):
                raise MediationError(
                    "IDEMPOTENCY_CONFLICT",
                    "The request identifier was reused with different content.",
                )
            if row["status"] == "completed":
                saved = self._decode_request(row)
                if (
                    saved.mediation_session_id != session.owner.mediation_session_id
                    or saved.result_version != session.version
                    or canonical_bytes(saved.view) != plaintext
                ):
                    raise MediationError(
                        "MEDIATION_STORE_INTEGRITY", "The request result changed after completion."
                    )
                return
            if row["status"] != "processing":
                raise MediationError(
                    "MEDIATION_REQUEST_FAILED", "The request cannot be completed."
                )
            persisted = conn.execute(
                "SELECT 1 FROM mediation_sessions_v4 WHERE scope_key=? "
                "AND mediation_session_id=? AND version=?",
                (scope_key, session.owner.mediation_session_id, session.version),
            ).fetchone()
            if persisted is None:
                raise MediationError(
                    "MEDIATION_STORE_INTEGRITY", "The completed session was not durably committed."
                )
            result = conn.execute(
                "UPDATE mediation_requests_v4 SET status='completed',mediation_session_id=?,"
                "result_version=?,result_view_schema_version=?,result_key_version=?,"
                "result_view_nonce=?,result_view_ciphertext=?,result_view_digest=?,updated_at=? "
                "WHERE scope_key=? AND request_id=? AND request_digest=? AND status='processing'",
                (
                    session.owner.mediation_session_id,
                    session.version,
                    VIEW_SCHEMA_VERSION,
                    KEY_VERSION,
                    nonce,
                    ciphertext,
                    canonical_digest(view),
                    utc_now(),
                    scope_key,
                    request_id,
                    request_digest,
                ),
            )
            if result.rowcount != 1:
                raise MediationError(
                    "STATE_TRANSITION_CONFLICT", "The request completion lost its CAS."
                )

    def fail_request(
        self, scope: SubjectScope, request_id: str, request_digest: str
    ) -> None:
        scope_key = self._scope_key(scope)
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT request_digest,status FROM mediation_requests_v4 "
                "WHERE scope_key=? AND request_id=?",
                (scope_key, request_id),
            ).fetchone()
            if row is None:
                return
            if not hmac.compare_digest(row["request_digest"], request_digest):
                raise MediationError(
                    "IDEMPOTENCY_CONFLICT",
                    "The request identifier was reused with different content.",
                )
            if row["status"] == "processing":
                conn.execute(
                    "UPDATE mediation_requests_v4 SET status='failed',updated_at=? "
                    "WHERE scope_key=? AND request_id=? AND request_digest=? "
                    "AND status='processing'",
                    (utc_now(), scope_key, request_id, request_digest),
                )

    # Compatibility methods for callers migrating from the pre-reservation port.
    def idempotent_result(
        self, scope: SubjectScope, request_id: str, request_digest: str
    ) -> MediationSession | None:
        saved = self.load_request(scope, request_id, request_digest)
        if saved is None:
            return None
        assert saved.mediation_session_id is not None
        return self.get(saved.mediation_session_id, scope)

    def remember_result(
        self,
        scope: SubjectScope,
        request_id: str,
        request_digest: str,
        session: MediationSession,
    ) -> None:
        if self.load_request(scope, request_id, request_digest) is None:
            self.reserve_request(scope, request_id, request_digest)
        self.complete_request(
            scope,
            request_id,
            request_digest,
            session,
            build_local_durable_view(session),
        )

    def readiness_probe(self) -> StoreReadiness:
        self._verify_sentinel()
        probe_id = f"probe-{uuid4()}"
        scope = SubjectScope(
            subject=probe_id, tenantId="readiness", adkSessionId="readiness"
        )
        scope_key = self._scope_key(scope)
        payload = canonical_bytes({"probeId": probe_id, "schemaVersion": SCHEMA_VERSION})
        aad = self._view_aad(probe_id, 0, "mediation-readiness/1")
        nonce, ciphertext = self._seal(self._view_cipher, payload, aad)
        decryptable = hmac.compare_digest(
            payload, self._open(self._view_cipher, nonce, ciphertext, aad)
        )
        with self._transaction() as conn:
            now = utc_now()
            conn.execute(
                "INSERT INTO mediation_requests_v4(scope_key,request_id,request_digest,"
                "status,created_at,updated_at) VALUES(?,?,?,'processing',?,?)",
                (
                    scope_key,
                    probe_id,
                    f"sha256:{hashlib.sha256(payload).hexdigest()}",
                    now,
                    now,
                ),
            )
            removed = conn.execute(
                "DELETE FROM mediation_requests_v4 WHERE scope_key=? AND request_id=?",
                (scope_key, probe_id),
            ).rowcount
            if removed != 1:
                raise RuntimeError("mediation store readiness write probe failed")
        return StoreReadiness(
            kind="sqlite",
            durability_profile="local-durable",
            schema_version=4,
            writable=True,
            decryptable=decryptable,
        )

    def probe(self) -> StoreReadiness:
        return self.readiness_probe()
