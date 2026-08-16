"""Fail-closed, process-local Merchant fault control for local release tests."""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from threading import RLock


_OPERATION_PATTERN = re.compile(
    r"^fulfillment-commit:continuation:[0-9a-f]{32}:1$"
)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class FulfillmentFaultTarget:
    order_id: str
    task_id: str
    operation_id: str

    def validate(self) -> None:
        if not self.order_id or len(self.order_id) > 256:
            raise ValueError("fault target orderId is invalid")
        if not self.task_id or len(self.task_id) > 256:
            raise ValueError("fault target taskId is invalid")
        if not _OPERATION_PATTERN.fullmatch(self.operation_id):
            raise ValueError("fault target operationId is invalid")

    def public(self) -> dict[str, str]:
        return {
            "orderId": self.order_id,
            "taskId": self.task_id,
            "operationId": self.operation_id,
        }


@dataclass(frozen=True, slots=True)
class FaultAuditEvent:
    sequence: int
    event: str
    occurred_at: str
    target: dict[str, str]

    def public(self) -> dict[str, object]:
        value = asdict(self)
        return {
            "sequence": value["sequence"],
            "event": value["event"],
            "occurredAt": value["occurred_at"],
            "target": value["target"],
        }


class MerchantTestFaults:
    """Own one exact one-shot fault without touching SQLite schema or triggers."""

    def __init__(self, secret: str) -> None:
        if len(secret) < 32:
            raise ValueError("MEDIATION_TEST_FAULT_SECRET must be at least 32 characters")
        self._secret_digest = hashlib.sha256(secret.encode("utf-8")).digest()
        self._target: FulfillmentFaultTarget | None = None
        self._state = "idle"
        self._audit: list[FaultAuditEvent] = []
        self._lock = RLock()

    def authorized(self, supplied: str | None) -> bool:
        if supplied is None:
            return False
        candidate = hashlib.sha256(supplied.encode("utf-8")).digest()
        return secrets.compare_digest(candidate, self._secret_digest)

    def _record(self, event: str, target: FulfillmentFaultTarget) -> None:
        self._audit.append(
            FaultAuditEvent(
                sequence=len(self._audit) + 1,
                event=event,
                occurred_at=_now(),
                target=target.public(),
            )
        )

    def arm(self, target: FulfillmentFaultTarget) -> bool:
        target.validate()
        with self._lock:
            if self._state == "armed":
                if self._target == target:
                    self._record("arm-replayed", target)
                    return False
                raise RuntimeError("a different Merchant test fault is already armed")
            self._target = target
            self._state = "armed"
            self._record("armed", target)
            return True

    def consume_if_exact(self, target: FulfillmentFaultTarget) -> bool:
        target.validate()
        with self._lock:
            if self._state != "armed" or self._target is None:
                return False
            if self._target != target:
                self._record("target-mismatch", target)
                return False
            self._state = "consumed"
            self._record("consumed", target)
            return True

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "status": self._state,
                "target": self._target.public() if self._target else None,
                "audit": [event.public() for event in self._audit],
            }
