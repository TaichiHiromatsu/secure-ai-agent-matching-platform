"""Typed records and deterministic public projections for mediation persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from secure_mediation_agent.demo_catalog import (
    REQUIREMENT_SCHEMA_VERSION,
    demo_scenario,
    validate_confirmation,
    validate_payment_requirement,
)

from .approval_targets import build_plan_approval_target
from .canonical import safe_ref
from .errors import MediationError
from .models import (
    MediationPublicView,
    MediationSession,
    MediationState,
    PaymentApprovalTarget,
    PlanApprovalTarget,
)


@dataclass(frozen=True, slots=True)
class RequestReservation:
    status: Literal["reserved", "completed"]
    mediation_session_id: str | None = None
    result_version: int | None = None
    view: MediationPublicView | None = None

    def __post_init__(self) -> None:
        completed = self.status == "completed"
        fields_present = (
            self.mediation_session_id is not None
            and self.result_version is not None
            and self.view is not None
        )
        if completed != fields_present:
            raise ValueError("completed reservations require the exact persisted result")


@dataclass(frozen=True, slots=True)
class StoreReadiness:
    kind: Literal["sqlite"]
    durability_profile: Literal["local-durable"]
    schema_version: Literal[4]
    writable: bool
    decryptable: bool

    @property
    def ready(self) -> bool:
        return self.writable and self.decryptable


_MESSAGES = {
    MediationState.WAITING_FOR_PLAN_APPROVAL: (
        "以下の承認対象は実行する計画そのものです。全項目を確認し、"
        "実行する場合のみメッセージ全体を「承認」として送信してください。"
    ),
    MediationState.EXECUTING: "承認済み計画を実行中です。",
    MediationState.WAITING_FOR_PAYMENT_APPROVAL: (
        "これは計画承認とは別の支払い承認です。以下の承認対象を確認し、"
        "支払う場合のみメッセージ全体を「承認」として送信してください。"
    ),
    MediationState.PAYMENT_APPROVED: "支払い承認を記録しました。",
    MediationState.RESUMING_A2A: "承認済み支払いを送信中です。",
    MediationState.COMPLETED: "安全性確認を完了し、処理が完了しました。",
    MediationState.BLOCKED: "安全性確認により処理を停止しました。",
    MediationState.REVIEW_REQUIRED: "結果を確定できないため、手動確認が必要です。",
    MediationState.CANCELLED: "処理はキャンセルされました。",
    MediationState.REFUND_PENDING: (
        "返金対象を確認しました。返金する場合は、メッセージ全体を「承認」として送信してください。"
    ),
    MediationState.REFUND_SUBMITTING: "返金処理中です。",
    MediationState.REFUNDED: "返金処理が完了しました。",
}


def paid_payment_approval_message(requirement: object) -> str:
    """Render only catalog-owned scenario terms; never echo remote free text."""

    payment_required = requirement.payment_required
    schema_version = payment_required.get("schemaVersion")
    if schema_version is None:
        return (
            "これは計画承認とは別の支払い承認です。"
            f"Demo paid booking / {requirement.amount_minor} minor units "
            f"{requirement.currency} / {requirement.payee} / "
            f"期限 {requirement.expires_at.isoformat()}。"
            "これはupgrade前から保存されている汎用simulation条件です。"
            "以下の承認対象を確認し、支払う場合のみメッセージ全体を"
            "「承認」として送信してください。"
        )
    if schema_version != REQUIREMENT_SCHEMA_VERSION:
        raise MediationError(
            "MEDIATION_STORE_INTEGRITY",
            "The payment approval requirement schema is unsupported.",
        )
    try:
        validate_payment_requirement(payment_required)
    except ValueError as error:
        raise MediationError(
            "MEDIATION_STORE_INTEGRITY",
            "The payment approval scenario is invalid.",
        ) from error
    scenario = demo_scenario()
    decimals = scenario["arrangementFee"]["decimals"]
    amount = f"{requirement.amount_minor / (10**decimals):.{decimals}f}"
    return (
        "これは計画承認とは別の支払い承認です。"
        f"{scenario['service']} / {scenario['hotel']} / "
        f"{scenario['dates']['checkIn']}〜{scenario['dates']['checkOut']} / {scenario['guests']}名 / "
        f"予約手配サービス料 {amount} {requirement.currency}（宿泊代を含まない） / "
        f"{requirement.payee} / 期限 {requirement.expires_at.isoformat()}。"
        "シミュレーションであり、実予約・実hold・実送金・法的保証はありません。"
        "以下の承認対象を確認し、支払う場合のみメッセージ全体を"
        "「承認」として送信してください。"
    )


def paid_completion_message(session: MediationSession) -> str | None:
    """Project a strict catalog-only result after ACCEPT and same-Task completion."""

    if (
        session.state != MediationState.COMPLETED
        or session.active_step.selected_agent.canonical_agent_id != "agent-005"
        or session.continuation is None
        or not isinstance(session.result, dict)
        or not any(
            event.stage == "final-validation" and event.decision == "ACCEPT"
            for event in session.trace
        )
    ):
        return None
    artifact = session.result.get("artifact")
    parts = artifact.get("parts") if isinstance(artifact, dict) else None
    data = (
        parts[0].get("data")
        if isinstance(parts, list)
        and len(parts) == 1
        and isinstance(parts[0], dict)
        and parts[0].get("kind") == "data"
        else None
    )
    try:
        validate_confirmation(
            data, remote_task_id=session.continuation.remote_task.task_id
        )
    except (TypeError, ValueError):
        return None
    return (
        "デモ予約確認（シミュレーション）を発行しました。"
        f" {data['hotel']} / {data['dates']['checkIn']}〜{data['dates']['checkOut']} / "
        f"{data['guests']}名 / 参照番号 {data['confirmationReference']}。"
        " SIMULATED / NOT A REAL BOOKING。実予約・実送金はありません。"
    )


def build_local_durable_view(session: MediationSession) -> MediationPublicView:
    """Build the exact restart-safe public view without consulting mutable services."""

    continuation = session.continuation
    message = _MESSAGES[session.state]
    approval_target: PlanApprovalTarget | PaymentApprovalTarget | None = None
    if session.state == MediationState.WAITING_FOR_PLAN_APPROVAL:
        approval_target = build_plan_approval_target(session.plan)
    if session.state == MediationState.WAITING_FOR_PAYMENT_APPROVAL and continuation:
        # Local import avoids coupling persistence records to controller construction.
        from .approval_targets import build_payment_approval_target

        requirement = continuation.requirement
        remote = continuation.remote_task
        approval_target = build_payment_approval_target(
            plan_id=continuation.plan_id,
            plan_version=continuation.plan_version,
            plan_digest=continuation.plan_digest,
            step_id=continuation.step_id,
            task_id=remote.task_id,
            context_id=remote.context_id,
            order_id=requirement.order_id,
            quote_id=requirement.quote_id,
            merchant=requirement.payee,
            amount_minor=requirement.amount_minor,
            currency=requirement.currency,
            profile_id=requirement.profile_id,
            expires_at=requirement.expires_at,
            payment_required=requirement.payment_required,
            requirement_digest=requirement.requirement_digest,
            checkout_digest=requirement.checkout_digest,
        )
        message = paid_payment_approval_message(requirement)
    completion_message = paid_completion_message(session)
    if completion_message is not None:
        message = completion_message
    return MediationPublicView(
        state=session.state,
        version=session.version,
        message=message,
        agentLabel=session.active_step.selected_agent.registry_name,
        planRef=safe_ref(session.plan.plan_digest),
        stepRef=safe_ref(session.active_step.step_id),
        taskRef=(
            safe_ref(continuation.remote_task.task_id)
            if continuation is not None
            else None
        ),
        approvalTarget=approval_target,
        approvalTargetDigest=session.approval_target_digest,
        pendingAction=session.pending_action,
        trace=tuple(session.trace),
        durabilityProfile="local-durable",
    )
