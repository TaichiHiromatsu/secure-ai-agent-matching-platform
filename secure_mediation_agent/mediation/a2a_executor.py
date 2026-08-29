"""One shared enforcement path for initial and same-Task A2A operations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, StrictStr

from .errors import DefinitiveA2ARejection, ReviewRequired, SecurityBlocked
from .models import (
    A2AResponseEnvelope,
    FrozenModel,
    GateDecision,
    RemoteTaskSnapshot,
    SelectedAgentSnapshot,
)
from .ports import (
    A2ATransportPort,
    LegacySecurityHookPort,
    OperationObserverPort,
    StableGatePort,
)


class A2AOperation(FrozenModel):
    operation_id: StrictStr = Field(alias="operationId", min_length=1)
    kind: Literal["task-start", "payment-submit"]
    agent: SelectedAgentSnapshot
    request: dict[str, Any]
    request_digest: StrictStr = Field(alias="requestDigest", pattern=r"^sha256:[0-9a-f]{64}$")
    idempotency_key: StrictStr = Field(alias="idempotencyKey", min_length=1)
    task_id: StrictStr | None = Field(default=None, alias="taskId")
    context_id: StrictStr | None = Field(default=None, alias="contextId")
    authorization: StrictStr | None = Field(default=None, repr=False)

    def gate_ids(self) -> tuple[str, str]:
        if self.kind == "task-start":
            return ("PRE_A2A_START", "POST_A2A_RESPONSE")
        return ("PRE_PAYMENT_SUBMIT", "POST_PAYMENT_RESULT")


class A2AExecution(FrozenModel):
    operation: A2AOperation
    response: A2AResponseEnvelope
    pre_decision: GateDecision = Field(alias="preDecision")
    post_decision: GateDecision = Field(alias="postDecision")
    event_order: tuple[StrictStr, ...] = Field(alias="eventOrder")


class NullOperationObserver:
    async def persist_response(
        self, operation: A2AOperation, response: RemoteTaskSnapshot
    ) -> None:
        return None


class SharedA2AOperationExecutor:
    """Forces callback-before/gate/transport/persist/callback-after/gate order."""

    def __init__(
        self,
        *,
        callback: LegacySecurityHookPort,
        gates: StableGatePort,
        transport: A2ATransportPort,
        observer: OperationObserverPort | None = None,
        authorizer: Any | None = None,
    ) -> None:
        self.callback = callback
        self.gates = gates
        self.transport = transport
        self.observer = observer or NullOperationObserver()
        self.authorizer = authorizer

    @staticmethod
    def _require_pass(decision: GateDecision) -> None:
        if decision.decision == "BLOCK":
            raise SecurityBlocked(
                f"{decision.gate_id}_BLOCKED", f"{decision.gate_id} blocked the A2A operation."
            )
        if decision.decision == "REVIEW":
            raise ReviewRequired(
                "A2A_GATE_REVIEW", f"{decision.gate_id} requires review."
            )

    async def execute(self, operation: A2AOperation) -> A2AExecution:
        if self.authorizer is not None:
            operation = await self.authorizer.authorize(operation)
        pre_gate, post_gate = operation.gate_ids()
        order: list[str] = []

        try:
            await self.callback.before(operation)
        except Exception as error:
            raise SecurityBlocked(
                "LEGACY_CALLBACK_BEFORE_FAILED",
                "The A2A request was blocked by the security callback.",
            ) from error
        order.append("legacy-callback-before")

        pre = await self.gates.decide(pre_gate, operation, None)
        order.append(pre_gate)
        self._require_pass(pre)

        try:
            envelope = await self.transport.send(operation)
        except TimeoutError as error:
            raise ReviewRequired(
                "A2A_RESULT_UNKNOWN",
                "The remote operation result is unknown; no new operation was started.",
            ) from error
        except DefinitiveA2ARejection:
            raise
        except SecurityBlocked:
            raise
        except Exception as error:
            raise ReviewRequired(
                "A2A_RESULT_UNKNOWN",
                "The remote operation result is unknown; no new operation was started.",
            ) from error
        order.append("transport")

        response = envelope.task
        await self.observer.persist_response(operation, response)
        order.append("response-persisted")

        try:
            await self.callback.after(operation, response)
        except Exception as error:
            raise SecurityBlocked(
                "LEGACY_CALLBACK_AFTER_FAILED",
                "The A2A response was blocked by the security callback.",
            ) from error
        order.append("legacy-callback-after")

        post = await self.gates.decide(post_gate, operation, response)
        order.append(post_gate)
        self._require_pass(post)

        if operation.kind == "payment-submit":
            if response.task_id != operation.task_id or response.context_id != operation.context_id:
                raise SecurityBlocked(
                    "REMOTE_TASK_BINDING_MISMATCH",
                    "The payment result did not belong to the approved remote Task.",
                )

        return A2AExecution(
            operation=operation,
            response=envelope,
            preDecision=pre,
            postDecision=post,
            eventOrder=tuple(order),
        )
