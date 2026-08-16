"""Stable, non-secret errors for the deterministic mediation layer."""

from __future__ import annotations


class MediationError(RuntimeError):
    def __init__(self, code: str, message: str, *, review: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.review = review


class SecurityBlocked(MediationError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, review=False)


class ReviewRequired(MediationError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, review=True)


class DefinitiveA2ARejection(SecurityBlocked):
    """A request proven rejected before the remote side effect occurred."""
