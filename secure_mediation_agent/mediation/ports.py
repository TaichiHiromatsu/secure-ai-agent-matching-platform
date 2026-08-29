"""Ports keeping the mediation controller independent from payment persistence."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from .models import (
    A2AResponseEnvelope,
    BridgeApprovalResult,
    BridgeAttachment,
    BridgeExecutionResult,
    GateDecision,
    MediationContinuation,
    MediationPlan,
    MediationSession,
    OwnerScope,
    RefundResult,
    RemoteTaskSnapshot,
    SelectedAgentSnapshot,
    SubjectScope,
)
from .persistence_models import RequestReservation, StoreReadiness


T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]


class MatcherPort(Protocol):
    async def match(self, goal: str) -> Sequence[SelectedAgentSnapshot]: ...


class PlannerPort(Protocol):
    async def create_plan(
        self,
        goal: str,
        owner: OwnerScope,
        candidates: Sequence[SelectedAgentSnapshot],
    ) -> MediationPlan: ...


class LegacySecurityHookPort(Protocol):
    async def before(self, operation: "A2AOperation") -> None: ...

    async def after(
        self, operation: "A2AOperation", response: RemoteTaskSnapshot
    ) -> None: ...


class StableGatePort(Protocol):
    async def decide(
        self,
        gate_id: str,
        operation: "A2AOperation",
        response: RemoteTaskSnapshot | None,
    ) -> GateDecision: ...


class A2ATransportPort(Protocol):
    async def send(self, operation: "A2AOperation") -> A2AResponseEnvelope: ...


class OperationObserverPort(Protocol):
    async def persist_response(
        self, operation: "A2AOperation", response: RemoteTaskSnapshot
    ) -> None: ...


class A2AExecutorPort(Protocol):
    async def execute(self, operation: "A2AOperation") -> "A2AExecution": ...


class PaymentBridgePort(Protocol):
    def attach(
        self,
        *,
        owner: Any,
        approved_plan: Any,
        step: Any,
        remote_task: Any,
        requirement: Any,
    ) -> MaybeAwaitable[BridgeAttachment | Any]: ...

    def approve(
        self,
        *,
        owner: Any,
        continuation_id: str,
        expected_version: int,
        approval_text: str,
        expected_approval_target_digest: str,
    ) -> MaybeAwaitable[BridgeApprovalResult | Any]: ...

    def execute_approved_payment(
        self,
        *,
        operation_id: str,
        continuation_id: str,
        expected_version: int,
    ) -> MaybeAwaitable[BridgeExecutionResult | Any]: ...

    def refund(
        self,
        *,
        owner: Any,
        operation_id: str,
        continuation_id: str,
        expected_version: int,
    ) -> MaybeAwaitable[RefundResult | Any]: ...


class FinalValidationPort(Protocol):
    async def validate(
        self, session: MediationSession, result: dict[str, Any]
    ) -> str: ...


class MediationStorePort(Protocol):
    def active_for(self, scope: SubjectScope) -> MediationSession | None: ...

    def latest_for(self, scope: SubjectScope) -> MediationSession | None: ...

    def get(self, mediation_session_id: str, scope: SubjectScope) -> MediationSession: ...

    def save_new(self, session: MediationSession) -> None: ...

    def compare_and_set(
        self,
        session: MediationSession,
        *,
        expected_version: int,
    ) -> MediationSession: ...

    def reserve_request(
        self,
        scope: SubjectScope,
        request_id: str,
        request_digest: str,
        expected_version: int | None = None,
    ) -> RequestReservation: ...

    def complete_request(
        self,
        scope: SubjectScope,
        request_id: str,
        request_digest: str,
        session: MediationSession,
        view: "MediationPublicView",
    ) -> None: ...

    def fail_request(
        self,
        scope: SubjectScope,
        request_id: str,
        request_digest: str,
    ) -> None: ...

    def load_request(
        self,
        scope: SubjectScope,
        request_id: str,
        request_digest: str,
    ) -> RequestReservation | None: ...

    def readiness_probe(self) -> StoreReadiness: ...

    def idempotent_result(
        self, scope: SubjectScope, request_id: str, request_digest: str
    ) -> MediationSession | None: ...

    def remember_result(
        self,
        scope: SubjectScope,
        request_id: str,
        request_digest: str,
        session: MediationSession,
    ) -> None: ...


if TYPE_CHECKING:
    from .a2a_executor import A2AExecution, A2AOperation
    from .models import MediationPublicView
