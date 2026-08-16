"""Strict domain and wire-neutral types for the Release-1 mediation core."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        strict=True,
        validate_assignment=True,
    )


class FrozenModel(StrictModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        strict=True,
        frozen=True,
    )


class MediationState(StrEnum):
    WAITING_FOR_PLAN_APPROVAL = "WaitingForPlanApproval"
    EXECUTING = "Executing"
    WAITING_FOR_PAYMENT_APPROVAL = "WaitingForPaymentApproval"
    PAYMENT_APPROVED = "PaymentApproved"
    RESUMING_A2A = "ResumingA2A"
    COMPLETED = "Completed"
    BLOCKED = "Blocked"
    REVIEW_REQUIRED = "ReviewRequired"
    CANCELLED = "Cancelled"
    REFUND_PENDING = "RefundPending"
    REFUND_SUBMITTING = "RefundSubmitting"
    REFUNDED = "Refunded"


ACTIVE_STATES = frozenset(
    {
        MediationState.WAITING_FOR_PLAN_APPROVAL,
        MediationState.EXECUTING,
        MediationState.WAITING_FOR_PAYMENT_APPROVAL,
        MediationState.PAYMENT_APPROVED,
        MediationState.RESUMING_A2A,
        MediationState.REFUND_PENDING,
        MediationState.REFUND_SUBMITTING,
        MediationState.REVIEW_REQUIRED,
    }
)


class SubjectScope(FrozenModel):
    """Identity verified by the ingress, without a client-selected workflow ID."""

    subject: StrictStr = Field(min_length=1, max_length=256)
    tenant_id: StrictStr = Field(alias="tenantId", min_length=1, max_length=128)
    adk_session_id: StrictStr = Field(alias="adkSessionId", min_length=1, max_length=256)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.tenant_id, self.adk_session_id)


class OwnerScope(FrozenModel):
    subject: StrictStr = Field(min_length=1, max_length=256)
    tenant_id: StrictStr = Field(alias="tenantId", min_length=1, max_length=128)
    adk_session_id: StrictStr = Field(alias="adkSessionId", min_length=1, max_length=256)
    mediation_session_id: StrictStr = Field(
        alias="mediationSessionId", min_length=1, max_length=256
    )

    @property
    def subject_scope(self) -> SubjectScope:
        return SubjectScope(
            subject=self.subject,
            tenantId=self.tenant_id,
            adkSessionId=self.adk_session_id,
        )


class TextPart(FrozenModel):
    kind: Literal["text"] = "text"
    text: StrictStr = Field(min_length=1, max_length=65_536)


class SelectedAgentSnapshot(FrozenModel):
    canonical_agent_id: StrictStr = Field(alias="canonicalAgentId", min_length=1)
    registry_name: StrictStr = Field(alias="registryName", min_length=1)
    a2a_agent_name: StrictStr = Field(alias="a2aAgentName", min_length=1)
    agent_card_url: StrictStr = Field(alias="agentCardUrl", min_length=1)
    rpc_endpoint: StrictStr = Field(alias="rpcEndpoint", min_length=1)
    a2a_skill_id: StrictStr = Field(alias="a2aSkillId", min_length=1)
    trust_score: StrictInt = Field(alias="trustScore", ge=0, le=100)
    card_digest: StrictStr = Field(alias="cardDigest", pattern=SHA256_PATTERN)
    snapshot_digest: StrictStr = Field(alias="snapshotDigest", pattern=SHA256_PATTERN)
    payment_extension_uris: tuple[StrictStr, ...] = Field(
        default_factory=tuple, alias="paymentExtensionUris"
    )


class MediationStep(FrozenModel):
    step_id: StrictStr = Field(alias="stepId", min_length=1)
    ordinal: StrictInt = Field(ge=1)
    selected_agent: SelectedAgentSnapshot = Field(alias="selectedAgent")
    input_digest: StrictStr = Field(alias="inputDigest", pattern=SHA256_PATTERN)
    goal: StrictStr = Field(min_length=1, max_length=65_536)
    payment_limit_minor: StrictInt = Field(alias="paymentLimitMinor", ge=0)
    currency: StrictStr = Field(min_length=3, max_length=12)


class MediationPlan(FrozenModel):
    plan_id: StrictStr = Field(alias="planId", min_length=1)
    plan_version: Literal[1] = Field(default=1, alias="planVersion")
    plan_digest: StrictStr = Field(alias="planDigest", pattern=SHA256_PATTERN)
    goal_digest: StrictStr = Field(alias="goalDigest", pattern=SHA256_PATTERN)
    owner: OwnerScope
    steps: tuple[MediationStep, ...] = Field(min_length=1)
    created_at: datetime = Field(alias="createdAt")
    expires_at: datetime = Field(alias="expiresAt")

    @model_validator(mode="after")
    def validate_steps(self) -> "MediationPlan":
        ordinals = [step.ordinal for step in self.steps]
        ids = [step.step_id for step in self.steps]
        if ordinals != list(range(1, len(self.steps) + 1)):
            raise ValueError("plan step ordinals must be contiguous")
        if len(ids) != len(set(ids)):
            raise ValueError("plan step IDs must be unique")
        if self.expires_at <= self.created_at:
            raise ValueError("plan expiry must follow creation")
        return self


class PlanApprovalAgentTarget(FrozenModel):
    canonical_agent_id: StrictStr = Field(alias="canonicalAgentId", min_length=1)
    registry_name: StrictStr = Field(alias="registryName", min_length=1)
    a2a_agent_name: StrictStr = Field(alias="a2aAgentName", min_length=1)
    skill_id: StrictStr = Field(alias="skillId", min_length=1)
    rpc_endpoint: StrictStr = Field(alias="rpcEndpoint", min_length=1)
    card_digest: StrictStr = Field(alias="cardDigest", pattern=SHA256_PATTERN)
    snapshot_digest: StrictStr = Field(alias="snapshotDigest", pattern=SHA256_PATTERN)


class PlanApprovalStepTarget(FrozenModel):
    step_id: StrictStr = Field(alias="stepId", min_length=1)
    ordinal: StrictInt = Field(ge=1)
    agent: PlanApprovalAgentTarget
    goal: StrictStr = Field(min_length=1, max_length=65_536)
    conditions: tuple[StrictStr, ...] = Field(min_length=1)
    currency: StrictStr = Field(min_length=3, max_length=12)
    payment_limit_minor: StrictInt = Field(alias="paymentLimitMinor", ge=0)


class PlanApprovalTarget(FrozenModel):
    """Exact canonical object rendered before the first approval."""

    schema_version: Literal["plan-approval-target/1"] = Field(
        default="plan-approval-target/1", alias="schemaVersion"
    )
    approval_kind: Literal["plan"] = Field(default="plan", alias="approvalKind")
    plan_id: StrictStr = Field(alias="planId", min_length=1)
    plan_version: Literal[1] = Field(alias="planVersion")
    plan_digest: StrictStr = Field(alias="planDigest", pattern=SHA256_PATTERN)
    steps: tuple[PlanApprovalStepTarget, ...] = Field(min_length=1)
    expires_at: datetime = Field(alias="expiresAt")
    approval_token: Literal["承認"] = Field(default="承認", alias="approvalToken")


class BridgePaymentDisplay(FrozenModel):
    """The exact object hashed by the durable bridge as ``displayDigest``."""

    task_id: StrictStr = Field(alias="taskId", min_length=1)
    context_id: StrictStr = Field(alias="contextId", min_length=1)
    order_id: StrictStr = Field(alias="orderId", min_length=1)
    quote_id: StrictStr = Field(alias="quoteId", min_length=1)
    merchant: StrictStr = Field(min_length=1)
    amount_minor: StrictInt = Field(alias="amountMinor", gt=0)
    currency: StrictStr = Field(min_length=3, max_length=12)
    profile_id: StrictStr = Field(alias="profileId", min_length=1)
    simulated: Literal[True] = True
    state: Literal["WAITING_FOR_PAYMENT_APPROVAL"] = (
        "WAITING_FOR_PAYMENT_APPROVAL"
    )


class PaymentApprovalTarget(FrozenModel):
    """Exact canonical object rendered before the distinct payment approval."""

    schema_version: Literal["payment-approval-target/1"] = Field(
        default="payment-approval-target/1", alias="schemaVersion"
    )
    approval_kind: Literal["payment"] = Field(default="payment", alias="approvalKind")
    distinct_from_plan_approval: Literal[True] = Field(
        default=True, alias="distinctFromPlanApproval"
    )
    plan_id: StrictStr = Field(alias="planId", min_length=1)
    plan_version: Literal[1] = Field(alias="planVersion")
    plan_digest: StrictStr = Field(alias="planDigest", pattern=SHA256_PATTERN)
    bridge_display: BridgePaymentDisplay = Field(alias="bridgeDisplay")
    bridge_display_digest: StrictStr = Field(
        alias="bridgeDisplayDigest", pattern=SHA256_PATTERN
    )
    product: StrictStr = Field(min_length=1, max_length=256)
    expires_at: datetime = Field(alias="expiresAt")
    payment_method: StrictStr = Field(alias="paymentMethod", min_length=1)
    scheme: StrictStr = Field(min_length=1)
    network: StrictStr = Field(min_length=1)
    asset: StrictStr = Field(min_length=1)
    step_ref: StrictStr = Field(alias="stepRef", min_length=1)
    task_ref: StrictStr = Field(alias="taskRef", min_length=1)
    requirement_digest: StrictStr = Field(
        alias="requirementDigest", pattern=SHA256_PATTERN
    )
    checkout_digest: StrictStr = Field(alias="checkoutDigest", pattern=SHA256_PATTERN)
    approval_token: Literal["承認"] = Field(default="承認", alias="approvalToken")


class PlanApproval(FrozenModel):
    approval_id: StrictStr = Field(alias="approvalId", min_length=1)
    plan_id: StrictStr = Field(alias="planId", min_length=1)
    plan_version: Literal[1] = Field(alias="planVersion")
    plan_digest: StrictStr = Field(alias="planDigest", pattern=SHA256_PATTERN)
    approval_target_digest: StrictStr = Field(
        alias="approvalTargetDigest", pattern=SHA256_PATTERN
    )
    nonce: StrictStr = Field(min_length=16)
    issued_at: datetime = Field(alias="issuedAt")


class PaymentRequirementSnapshot(FrozenModel):
    task_state: Literal["input-required"] = Field(alias="taskState")
    payment_status: Literal["payment-required"] = Field(alias="paymentStatus")
    extension_uri: StrictStr = Field(alias="extensionUri", min_length=1)
    profile_id: StrictStr = Field(alias="profileId", min_length=1)
    order_id: StrictStr = Field(alias="orderId", min_length=1)
    quote_id: StrictStr = Field(alias="quoteId", min_length=1)
    amount_minor: StrictInt = Field(alias="amountMinor", gt=0)
    currency: StrictStr = Field(min_length=3, max_length=12)
    payee: StrictStr = Field(min_length=1, max_length=256)
    expires_at: datetime = Field(alias="expiresAt")
    requirement_digest: StrictStr = Field(
        alias="requirementDigest", pattern=SHA256_PATTERN
    )
    checkout_digest: StrictStr = Field(alias="checkoutDigest", pattern=SHA256_PATTERN)
    payment_required: dict[str, Any] = Field(alias="paymentRequired")
    checkout_audience: StrictStr = Field(alias="checkoutAudience", min_length=1)
    checkout_nonce: StrictStr = Field(alias="checkoutNonce", min_length=16)
    payment_audience: StrictStr = Field(alias="paymentAudience", min_length=1)
    payment_nonce: StrictStr = Field(alias="paymentNonce", min_length=16)


class PrivatePaymentMaterial(FrozenModel):
    """Secret A2A result material that may cross only into the payment bridge."""

    checkout_jwt: StrictStr = Field(alias="checkoutJwt", min_length=1, repr=False)
    checkout_hash: StrictStr = Field(alias="checkoutHash", min_length=16, repr=False)


class A2AResponseEnvelope(FrozenModel):
    """Typed internal result envelope; public Task data stays separate from secrets."""

    task: "RemoteTaskSnapshot"
    private_payment_material: PrivatePaymentMaterial | None = Field(
        default=None, alias="privatePaymentMaterial", repr=False
    )
    envelope_digest: StrictStr = Field(alias="envelopeDigest", pattern=SHA256_PATTERN)


class RemoteTaskSnapshot(FrozenModel):
    task_id: StrictStr = Field(alias="taskId", min_length=1)
    context_id: StrictStr = Field(alias="contextId", min_length=1)
    state: Literal["completed", "input-required", "working", "failed"]
    task_digest: StrictStr = Field(alias="taskDigest", pattern=SHA256_PATTERN)
    order_id: StrictStr | None = Field(default=None, alias="orderId")
    quote_id: StrictStr | None = Field(default=None, alias="quoteId")
    payment_requirement: PaymentRequirementSnapshot | None = Field(
        default=None, alias="paymentRequirement"
    )
    artifact: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_payment_branch(self) -> "RemoteTaskSnapshot":
        if self.payment_requirement is None:
            return self
        requirement = self.payment_requirement
        if self.state != "input-required":
            raise ValueError("payment requirement requires input-required Task")
        if requirement.task_state != self.state:
            raise ValueError("payment requirement Task state mismatch")
        if not self.order_id or not self.quote_id:
            raise ValueError("paid Task requires Merchant order and quote IDs")
        if requirement.order_id != self.order_id or requirement.quote_id != self.quote_id:
            raise ValueError("payment requirement order/quote mismatch")
        return self


class MediationContinuation(FrozenModel):
    continuation_id: StrictStr = Field(alias="continuationId", min_length=1)
    payment_workflow_id: StrictStr = Field(alias="paymentWorkflowId", min_length=1)
    owner: OwnerScope
    plan_id: StrictStr = Field(alias="planId", min_length=1)
    plan_version: Literal[1] = Field(alias="planVersion")
    plan_digest: StrictStr = Field(alias="planDigest", pattern=SHA256_PATTERN)
    step_id: StrictStr = Field(alias="stepId", min_length=1)
    remote_task: RemoteTaskSnapshot = Field(alias="remoteTask")
    requirement: PaymentRequirementSnapshot
    version: StrictInt = Field(ge=0)


class GateDecision(FrozenModel):
    gate_id: Literal[
        "PRE_A2A_START",
        "POST_A2A_RESPONSE",
        "POST_PAYMENT_REQUIREMENT",
        "PRE_PAYMENT_SUBMIT",
        "POST_PAYMENT_RESULT",
    ] = Field(alias="gateId")
    decision: Literal["PASS", "BLOCK", "REVIEW"]
    decision_digest: StrictStr = Field(alias="decisionDigest", pattern=SHA256_PATTERN)


class TraceEvent(FrozenModel):
    sequence: StrictInt = Field(ge=1)
    stage: StrictStr = Field(min_length=1, max_length=80)
    component_id: StrictStr = Field(alias="componentId", min_length=1, max_length=160)
    layer: Literal[
        "controller",
        "matcher",
        "planner",
        "callback-hook",
        "deterministic-validator",
        "payment-bridge",
        "final-validator",
    ]
    operation_id: StrictStr = Field(alias="operationId", min_length=1, max_length=256)
    decision: StrictStr = Field(min_length=1, max_length=80)
    safe_ref: StrictStr | None = Field(default=None, alias="safeRef", max_length=80)
    occurred_at: datetime = Field(default_factory=utc_now, alias="occurredAt")


class PendingAction(FrozenModel):
    kind: Literal[
        "approve-plan",
        "approve-payment",
        "execute-approved-payment",
        "request-refund",
        "wait",
        "none",
    ]
    target_ref: StrictStr | None = Field(default=None, alias="targetRef")


class MediationSession(StrictModel):
    schema_version: Literal["mediation-session/1"] = Field(
        default="mediation-session/1", alias="schemaVersion"
    )
    owner: OwnerScope
    goal: StrictStr = Field(min_length=1, max_length=65_536)
    state: MediationState
    version: StrictInt = Field(ge=0)
    plan: MediationPlan
    approval_target_digest: StrictStr | None = Field(
        default=None, alias="approvalTargetDigest", pattern=SHA256_PATTERN
    )
    plan_approval: PlanApproval | None = Field(default=None, alias="planApproval")
    active_step_ordinal: StrictInt = Field(default=1, alias="activeStepOrdinal", ge=1)
    continuation: MediationContinuation | None = None
    result: dict[str, Any] | None = None
    pending_action: PendingAction = Field(alias="pendingAction")
    trace: list[TraceEvent] = Field(default_factory=list)

    @property
    def active_step(self) -> MediationStep:
        return self.plan.steps[self.active_step_ordinal - 1]


class MediationPublicView(FrozenModel):
    schema_version: Literal["mediation-public-view/1"] = Field(
        default="mediation-public-view/1", alias="schemaVersion"
    )
    state: MediationState
    version: StrictInt
    message: StrictStr
    agent_label: StrictStr | None = Field(default=None, alias="agentLabel")
    plan_ref: StrictStr | None = Field(default=None, alias="planRef")
    step_ref: StrictStr | None = Field(default=None, alias="stepRef")
    task_ref: StrictStr | None = Field(default=None, alias="taskRef")
    approval_target: PlanApprovalTarget | PaymentApprovalTarget | None = Field(
        default=None, alias="approvalTarget"
    )
    approval_target_digest: StrictStr | None = Field(
        default=None, alias="approvalTargetDigest", pattern=SHA256_PATTERN
    )
    pending_action: PendingAction = Field(alias="pendingAction")
    trace: tuple[TraceEvent, ...]
    durability_profile: Literal["local-durable", "ephemeral-demo"] = Field(
        alias="durabilityProfile"
    )
    simulation: Literal[True] = True
    conformance: Literal["NOT CONFORMANT"] = "NOT CONFORMANT"


class BridgeAttachment(FrozenModel):
    continuation_id: StrictStr = Field(alias="continuationId", min_length=1)
    payment_workflow_id: StrictStr = Field(alias="paymentWorkflowId", min_length=1)
    version: StrictInt = Field(ge=0)


class BridgeApprovalResult(FrozenModel):
    continuation_id: StrictStr = Field(alias="continuationId", min_length=1)
    version: StrictInt = Field(ge=0)
    approval_digest: StrictStr = Field(alias="approvalDigest", pattern=SHA256_PATTERN)
    state: Literal["PaymentApproved"] = "PaymentApproved"


class BridgeExecutionResult(FrozenModel):
    continuation_id: StrictStr = Field(alias="continuationId", min_length=1)
    version: StrictInt = Field(ge=0)
    remote_task: RemoteTaskSnapshot = Field(alias="remoteTask")
    result: dict[str, Any]
    state: Literal[
        "same-task-working",
        "same-task-completed",
        "refund-required",
        "blocked",
        "review-required",
    ]
    a2a_executions: tuple["BridgeA2AExecutionSummary", ...] = Field(
        default_factory=tuple, alias="a2aExecutions"
    )


class BridgeA2AExecutionSummary(FrozenModel):
    operation_id: StrictStr = Field(alias="operationId", min_length=1)
    task_digest: StrictStr = Field(alias="taskDigest", pattern=SHA256_PATTERN)
    event_order: tuple[StrictStr, ...] = Field(alias="eventOrder")


class RefundResult(FrozenModel):
    refund_id: StrictStr = Field(alias="refundId", min_length=1)
    state: Literal["refunded", "rejected", "unknown"]
    result_digest: StrictStr = Field(alias="resultDigest", pattern=SHA256_PATTERN)
