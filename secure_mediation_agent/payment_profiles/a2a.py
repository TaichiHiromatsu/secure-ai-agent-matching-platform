"""A2A 0.3.19 model builders for the project-local simulation fixture."""

from __future__ import annotations

from typing import Any

from a2a.types import Message, Part, Role, Task, TaskState, TaskStatus, TextPart

from secure_mediation_agent.workflow.errors import DomainError


PAYMENT_STATUS = "x402.payment.status"
PAYMENT_REQUIRED = "x402.payment.required"
PAYMENT_PAYLOAD = "x402.payment.payload"
PAYMENT_RECEIPTS = "x402.payment.receipts"
PROJECT_METADATA = "io.github.taichihiromatsu.secure-mediation.v1"


def require_activation(
    requested: set[str], expected_uri: str, *, correlation_id: str
) -> str:
    if not requested:
        raise DomainError(
            "X402_EXTENSION_REQUIRED",
            "The selected payment profile activation is required.",
            correlation_id,
        )
    if requested != {expected_uri}:
        raise DomainError(
            "X402_ACTIVATION_MISMATCH",
            "The selected payment profile activation does not match.",
            correlation_id,
        )
    return expected_uri


def payment_required_task(
    *,
    task_id: str,
    context_id: str,
    message_id: str,
    required: dict[str, Any],
    project: dict[str, Any],
) -> Task:
    message = Message(
        messageId=message_id,
        contextId=context_id,
        taskId=task_id,
        role=Role.agent,
        parts=[Part(root=TextPart(text="Payment approval is required."))],
        metadata={
            PAYMENT_STATUS: "payment-required",
            PAYMENT_REQUIRED: required,
            PROJECT_METADATA: project,
        },
    )
    return Task(
        id=task_id,
        contextId=context_id,
        status=TaskStatus(state=TaskState.input_required, message=message),
        history=[message],
    )


def payment_message(
    *,
    task_id: str,
    context_id: str,
    message_id: str,
    status: str,
    payload: dict[str, Any] | None = None,
    project: dict[str, Any] | None = None,
) -> Message:
    metadata: dict[str, Any] = {PAYMENT_STATUS: status}
    if payload is not None:
        metadata[PAYMENT_PAYLOAD] = payload
    if project is not None:
        metadata[PROJECT_METADATA] = project
    return Message(
        messageId=message_id,
        taskId=task_id,
        contextId=context_id,
        role=Role.user,
        parts=[Part(root=TextPart(text="Payment authorization submitted."))],
        metadata=metadata,
    )


def final_task_metadata(
    *, status: str, receipts: list[dict[str, Any]], error: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        PAYMENT_STATUS: status,
        PAYMENT_RECEIPTS: receipts,
    }
    if error is not None:
        result["x402.payment.error"] = error
    return result
