"""Stable domain errors shared by API, CLI, and A2A adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_RETRYABLE = {
    "STATE_TRANSITION_CONFLICT",
    "RECONCILIATION_REQUIRED",
}

_HTTP_STATUS = {
    "APPROVAL_EXACT_TOKEN_REQUIRED": 409,
    "APPROVAL_NOT_PENDING": 409,
    "PLAN_APPROVAL_REQUIRED": 403,
    "PLAN_APPROVAL_INVALID": 403,
    "PLAN_APPROVAL_EXPIRED": 410,
    "PLAN_BINDING_MISMATCH": 409,
    "PLAN_CONSTRAINT_VIOLATION": 422,
    "PAYMENT_APPROVAL_REQUIRED": 403,
    "PAYMENT_APPROVAL_EXPIRED": 410,
    "AP2_CHECKOUT_INVALID": 422,
    "AP2_MANDATE_INVALID": 422,
    "AP2_CREDENTIAL_INVALID": 422,
    "AP2_CONSTRAINT_UNRESOLVED": 422,
    "X402_EXTENSION_REQUIRED": 400,
    "X402_ACTIVATION_MISMATCH": 409,
    "X402_TASK_CORRELATION_MISMATCH": 409,
    "X402_REQUIREMENTS_MISMATCH": 422,
    "X402_PAYMENT_PAYLOAD_INVALID": 422,
    "PAYMENT_FAILED": 422,
    "REPLAY_DETECTED": 409,
    "IDEMPOTENCY_CONFLICT": 409,
    "STATE_TRANSITION_CONFLICT": 409,
    "TENANT_BINDING_MISMATCH": 403,
    "RECONCILIATION_REQUIRED": 409,
    "UNSUPPORTED_LEGACY_PROFILE": 422,
    "SERVICE_NOT_READY": 503,
}


@dataclass(slots=True)
class DomainError(RuntimeError):
    code: str
    message: str
    correlation_id: str
    current_state: str | None = None
    expected_action: str | None = None

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)

    @property
    def retryable(self) -> bool:
        return self.code in _RETRYABLE

    @property
    def http_status(self) -> int:
        return _HTTP_STATUS.get(self.code, 400)

    def envelope(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "correlationId": self.correlation_id,
        }
        if self.current_state is not None:
            result["currentState"] = self.current_state
        if self.expected_action is not None:
            result["expectedAction"] = self.expected_action
        return result
