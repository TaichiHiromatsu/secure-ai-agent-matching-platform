"""Strict immutable models for the approved single-product workflow."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        populate_by_name=True,
    )


class WorkflowState(StrEnum):
    REQUEST_RECEIVED = "request_received"
    PLANNING = "planning"
    PLAN_APPROVAL_REQUIRED = "plan_approval_required"
    PLAN_APPROVED = "plan_approved"
    FREE_EXECUTING = "free_executing"
    FINAL_VALIDATING = "final_validating"
    MERCHANT_TASK_STARTING = "merchant_task_starting"
    PAYMENT_APPROVAL_REQUIRED = "payment_approval_required"
    PAYMENT_AUTHORIZING = "payment_authorizing"
    PAYMENT_APPROVED = "payment_approved"
    PAYMENT_SUBMITTED = "payment_submitted"
    PAYMENT_VERIFYING = "payment_verifying"
    FULFILLMENT_PREPARING = "fulfillment_preparing"
    PAYMENT_SETTLING = "payment_settling"
    FULFILLMENT_COMMITTING = "fulfillment_committing"
    COMPLETED = "completed"
    REPLAN_REQUIRED = "replan_required"
    PAYMENT_FAILED = "payment_failed"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    REFUND_REQUIRED = "refund_required"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_STATES = frozenset(
    {
        WorkflowState.COMPLETED,
        WorkflowState.PAYMENT_FAILED,
        WorkflowState.REFUNDED,
        WorkflowState.CANCELLED,
        WorkflowState.EXPIRED,
    }
)


class MessagePart(FrozenModel):
    kind: Literal["text"]
    text: StrictStr


class WorkflowRequest(FrozenModel):
    schema_version: Literal["secure-mediation-request/1"] = Field(
        default="secure-mediation-request/1", alias="schemaVersion"
    )
    goal: StrictStr = Field(min_length=1)
    product_id: Literal["demo-paid-booking"] = Field(
        default="demo-paid-booking", alias="productId"
    )
    quantity: Literal[1] = 1
    maximum_customer_total: Annotated[StrictInt, Field(ge=0)] = Field(
        default=1250, alias="maximumCustomerTotal"
    )
    currency: Literal["USD"] = "USD"
    decimals: Literal[2] = 2
    fee_policy_version: Literal["zero-fee-v1"] = Field(
        default="zero-fee-v1", alias="feePolicyVersion"
    )
    requested_profile: Literal["x402-wire-simulation/1"] = Field(
        default="x402-wire-simulation/1", alias="requestedProfile"
    )
    payment_required: bool = Field(default=True, alias="paymentRequired")


class PriceBreakdown(FrozenModel):
    merchandise_amount: Annotated[StrictInt, Field(ge=0)] = Field(
        alias="merchandiseAmount"
    )
    customer_surcharge: Literal[0] = Field(default=0, alias="customerSurcharge")
    collection_rail_cost: Literal[0] = Field(default=0, alias="collectionRailCost")
    customer_total: Annotated[StrictInt, Field(ge=0)] = Field(alias="customerTotal")
    provider_commission: Literal[0] = Field(default=0, alias="providerCommission")
    merchant_payable_amount: Annotated[StrictInt, Field(ge=0)] = Field(
        alias="merchantPayableAmount"
    )
    payout_rail_cost: Literal[0] = Field(default=0, alias="payoutRailCost")

    @model_validator(mode="after")
    def zero_fee(self) -> "PriceBreakdown":
        if not (
            self.merchandise_amount
            == self.customer_total
            == self.merchant_payable_amount
        ):
            raise ValueError("zero-fee-v1 amount equality failed")
        return self


class SelectedAgent(FrozenModel):
    agent_id: Literal["paid-booking-agent"] = Field(alias="agentId")
    agent_card_digest: StrictStr = Field(alias="agentCardDigest", pattern=r"^sha256:[0-9a-f]{64}$")
    endpoint: StrictStr
    onboarding_version: Literal["simulation-v1"] = Field(alias="onboardingVersion")
    trust_key_set_version: Literal["demo-es256-v1"] = Field(alias="trustKeySetVersion")


class Merchant(FrozenModel):
    id: Literal["demo-merchant"] = "demo-merchant"
    name: Literal["Demo Merchant"] = "Demo Merchant"
    website: StrictStr = "http://127.0.0.1:8005"
    payee_id: Literal["demo-merchant"] = Field(default="demo-merchant", alias="payeeId")


class PlanStep(FrozenModel):
    step_id: Literal["step-1"] = Field(alias="stepId")
    agent_id: Literal["paid-booking-agent"] = Field(alias="agentId")
    skill_id: Literal["paid-booking"] = Field(alias="skillId")
    payment_required: bool = Field(alias="paymentRequired")
    input_digest: StrictStr = Field(alias="inputDigest", pattern=r"^sha256:[0-9a-f]{64}$")


class AllowedPayment(FrozenModel):
    profile: Literal["x402-wire-simulation/1"]
    extension_uri: Literal[
        "urn:secure-a2a:extensions:x402-wire-simulation:v1"
    ] = Field(alias="extensionUri")
    schemes: tuple[Literal["exact-simulated"], ...]
    networks: tuple[Literal["demo:local"], ...]
    assets: tuple[Literal["USD"], ...]
    rail_mode: Literal["simulated"] = Field(alias="railMode")


class PlanSnapshot(FrozenModel):
    schema_version: Literal["secure-mediation-plan/1"] = Field(alias="schemaVersion")
    canonicalization: Literal["RFC8785"]
    plan_id: StrictStr = Field(alias="planId", min_length=16)
    plan_version: Literal[1] = Field(alias="planVersion")
    tenant_id: StrictStr = Field(alias="tenantId", min_length=1)
    customer_id: StrictStr = Field(alias="customerId", min_length=1)
    session_id: StrictStr = Field(alias="sessionId", min_length=1)
    context_id: StrictStr = Field(alias="contextId", min_length=1)
    request_digest: StrictStr = Field(alias="requestDigest", pattern=r"^sha256:[0-9a-f]{64}$")
    request: dict[str, Any]
    selected_agent: SelectedAgent = Field(alias="selectedAgent")
    merchant: Merchant
    skill_id: Literal["paid-booking"] = Field(alias="skillId")
    product_id: Literal["demo-paid-booking"] = Field(alias="productId")
    quantity: Literal[1]
    steps: tuple[PlanStep, ...]
    maximum_customer_total: Annotated[StrictInt, Field(ge=0)] = Field(alias="maximumCustomerTotal")
    currency: Literal["USD"]
    decimals: Literal[2]
    fee_policy_version: Literal["zero-fee-v1"] = Field(alias="feePolicyVersion")
    allowed_payment: AllowedPayment = Field(alias="allowedPayment")
    fulfillment_constraints: dict[str, Any] = Field(alias="fulfillmentConstraints")
    created_at: StrictStr = Field(alias="createdAt")
    expires_at: StrictStr = Field(alias="expiresAt")


class PublicWorkflowView(FrozenModel):
    workflow_id: StrictStr = Field(alias="workflowId")
    state: WorkflowState
    version: StrictInt
    pending_approval: Literal["plan", "payment"] | None = Field(
        default=None, alias="pendingApproval"
    )
    plan_id: StrictStr | None = Field(default=None, alias="planId")
    plan_digest: StrictStr | None = Field(default=None, alias="planDigest")
    order_id: StrictStr | None = Field(default=None, alias="orderId")
    task_id: StrictStr | None = Field(default=None, alias="taskId")
    rendered_text: StrictStr = Field(alias="renderedText")
    profile: Literal["x402-wire-simulation/1"]
    ap2_label: Literal["AP2 v0.2 Human Present demo"] = Field(alias="ap2Label")
    x402_label: Literal[
        "x402 v0.1 wire-shape test fixture (NOT CONFORMANT)"
    ] = Field(alias="x402Label")
    rail_label: Literal["simulated; no real asset or on-chain transaction"] = Field(
        alias="railLabel"
    )
    evidence: dict[str, Any] | None = None
