"""Thread-safe ephemeral store implementing the mediation repository port."""

from __future__ import annotations

from threading import RLock

from .errors import MediationError, SecurityBlocked
from .models import ACTIVE_STATES, MediationSession, SubjectScope


class InMemoryMediationStore:
    """Release-1 fallback store; production persistence can implement the same port."""

    def __init__(self) -> None:
        self._sessions: dict[str, MediationSession] = {}
        self._active: dict[tuple[str, str, str], str] = {}
        self._latest: dict[tuple[str, str, str], str] = {}
        self._requests: dict[
            tuple[str, str, str, str], tuple[str, str]
        ] = {}
        self._lock = RLock()

    @staticmethod
    def _copy(session: MediationSession) -> MediationSession:
        return session.model_copy(deep=True)

    def active_for(self, scope: SubjectScope) -> MediationSession | None:
        with self._lock:
            session_id = self._active.get(scope.key)
            if not session_id:
                return None
            session = self._sessions[session_id]
            if session.state not in ACTIVE_STATES:
                return None
            return self._copy(session)

    def latest_for(self, scope: SubjectScope) -> MediationSession | None:
        with self._lock:
            session_id = self._latest.get(scope.key)
            if not session_id:
                return None
            return self._copy(self._sessions[session_id])

    def get(self, mediation_session_id: str, scope: SubjectScope) -> MediationSession:
        with self._lock:
            session = self._sessions.get(mediation_session_id)
            if session is None or session.owner.subject_scope != scope:
                raise SecurityBlocked(
                    "MEDIATION_NOT_FOUND",
                    "The active mediation session is not available.",
                )
            return self._copy(session)

    def save_new(self, session: MediationSession) -> None:
        with self._lock:
            scope = session.owner.subject_scope
            existing_id = self._active.get(scope.key)
            if existing_id:
                existing = self._sessions[existing_id]
                if existing.state in ACTIVE_STATES:
                    raise MediationError(
                        "ACTIVE_MEDIATION_EXISTS",
                        "This session already has an active mediation request.",
                    )
            mediation_id = session.owner.mediation_session_id
            if mediation_id in self._sessions:
                raise MediationError(
                    "MEDIATION_ID_CONFLICT", "Mediation session identifier conflict."
                )
            self._sessions[mediation_id] = self._copy(session)
            self._active[scope.key] = mediation_id
            self._latest[scope.key] = mediation_id

    def compare_and_set(
        self,
        session: MediationSession,
        *,
        expected_version: int,
    ) -> MediationSession:
        with self._lock:
            mediation_id = session.owner.mediation_session_id
            current = self._sessions.get(mediation_id)
            if current is None or current.owner != session.owner:
                raise SecurityBlocked(
                    "MEDIATION_NOT_FOUND",
                    "The active mediation session is not available.",
                )
            if current.version != expected_version:
                raise MediationError(
                    "STATE_TRANSITION_CONFLICT",
                    "The mediation session changed; refresh before retrying.",
                )
            if session.version != expected_version + 1:
                raise MediationError(
                    "INVALID_VERSION_INCREMENT",
                    "The mediation session version increment is invalid.",
                )
            self._sessions[mediation_id] = self._copy(session)
            self._latest[session.owner.subject_scope.key] = mediation_id
            if session.state in ACTIVE_STATES:
                self._active[session.owner.subject_scope.key] = mediation_id
            else:
                self._active.pop(session.owner.subject_scope.key, None)
            return self._copy(session)

    def idempotent_result(
        self, scope: SubjectScope, request_id: str, request_digest: str
    ) -> MediationSession | None:
        with self._lock:
            record = self._requests.get((*scope.key, request_id))
            if record is None:
                return None
            saved_digest, mediation_id = record
            if saved_digest != request_digest:
                raise MediationError(
                    "IDEMPOTENCY_CONFLICT",
                    "The request identifier was reused with different content.",
                )
            return self._copy(self._sessions[mediation_id])

    def remember_result(
        self,
        scope: SubjectScope,
        request_id: str,
        request_digest: str,
        session: MediationSession,
    ) -> None:
        with self._lock:
            key = (*scope.key, request_id)
            existing = self._requests.get(key)
            if existing and existing[0] != request_digest:
                raise MediationError(
                    "IDEMPOTENCY_CONFLICT",
                    "The request identifier was reused with different content.",
                )
            self._requests[key] = (
                request_digest,
                session.owner.mediation_session_id,
            )
