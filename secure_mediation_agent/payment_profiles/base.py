"""Typed profile boundaries; official on-chain execution is intentionally absent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True, slots=True)
class ProfileReadiness:
    ready: bool
    profile_id: str
    rail_mode: Literal["simulated", "on-chain"]
    checks: dict[str, str]


class PaymentProfile(Protocol):
    profile_id: str
    extension_uri: str
    rail_mode: Literal["simulated", "on-chain"]
    conformance_label: str

    def build_required(self, *, amount: int) -> dict[str, Any]: ...
    def build_submission(self, *, proof: str) -> dict[str, Any]: ...
    def validate_activation(self, requested: set[str]) -> None: ...
    def readiness(self) -> ProfileReadiness: ...
