"""Deterministic payment port attached to an approved mediation step.

The bridge never plans, selects an Agent, or accepts payment terms from an
LLM/tool argument.  It binds an already approved step to one existing remote
A2A Task and exposes only state-gated operations to the mediation controller.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Callable, Literal, Protocol

from a2a.types import Message, Task, TaskState
from ap2.sdk.jwt_helper import create_jwt
from pydantic import Field, StrictInt, StrictStr, model_validator

from secure_mediation_agent.ap2.credential_provider import CredentialProvider
from secure_mediation_agent.ap2.keys import DemoKeySet
from secure_mediation_agent.ap2.trusted_surface import TrustedSurface
from secure_mediation_agent.ap2.verification import b64url_sha256
from secure_mediation_agent.payment_profiles.a2a import payment_message
from secure_mediation_agent.payment_profiles.registry import ProfileRegistry
from secure_mediation_agent.workflow.canonical import (
    canonical_bytes,
    canonical_digest,
    canonical_json,
    sha256_digest,
)
from secure_mediation_agent.workflow.errors import DomainError
from secure_mediation_agent.workflow.models import FrozenModel
from secure_mediation_agent.workflow.repository import WorkflowRepository, utc_now


APPROVAL_TOKEN = "承認"
PAID_AGENT_ID = "agent-005"
PAID_AGENT_NAME = "paid-booking-agent"
PAID_SKILL_ID = "paid-booking"
PAID_RPC_ENDPOINT = "http://127.0.0.1:8005/a2a"
SIMULATION_PROFILE = "x402-wire-simulation/1"
SIMULATION_EXTENSION = "urn:secure-a2a:extensions:x402-wire-simulation:v1"


def _stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}:{uuid.uuid5(uuid.NAMESPACE_URL, '/'.join(parts)).hex}"


def _utc_now() -> datetime:
    return datetime.now(UTC)


class BridgeState(StrEnum):
    WAITING_FOR_PAYMENT_APPROVAL = "waiting_for_payment_approval"
    PAYMENT_APPROVED = "payment_approved"
    GUARANTEED = "guaranteed"
    PAYMENT_SUBMITTED = "payment_submitted"
    SETTLED = "settled"
    FULFILLMENT_COMMITTING = "fulfillment_committing"
    COMPLETED = "completed"
    GUARANTEE_CANCELLED = "guarantee_cancelled"
    REVIEW_REQUIRED = "review_required"
    REFUND_REQUIRED = "refund_required"
    REFUNDED = "refunded"


class OwnerRef(FrozenModel):
    tenant_id: StrictStr = Field(alias="tenantId", min_length=1)
    subject_id: StrictStr = Field(alias="subjectId", min_length=1)
    session_id: StrictStr = Field(alias="sessionId", min_length=1)
    context_id: StrictStr = Field(alias="contextId", min_length=1)
    mediation_session_id: StrictStr = Field(alias="mediationSessionId", min_length=1)


class ApprovedPlanRef(FrozenModel):
    plan_id: StrictStr = Field(alias="planId", min_length=1)
    plan_version: StrictInt = Field(alias="planVersion", ge=1)
    plan_digest: StrictStr = Field(alias="planDigest", pattern=r"^sha256:[0-9a-f]{64}$")
    approval_id: StrictStr = Field(alias="approvalId", min_length=1)


class ApprovedStepRef(FrozenModel):
    step_id: StrictStr = Field(alias="stepId", min_length=1)
    canonical_agent_id: Literal["agent-005"] = Field(alias="canonicalAgentId")
    agent_card_digest: StrictStr = Field(
        alias="agentCardDigest", pattern=r"^sha256:[0-9a-f]{64}$"
    )
    rpc_endpoint: Literal["http://127.0.0.1:8005/a2a"] = Field(alias="rpcEndpoint")
    skill_id: Literal["paid-booking"] = Field(alias="skillId")


class RemoteTaskRef(FrozenModel):
    task_id: StrictStr = Field(alias="taskId", min_length=1)
    context_id: StrictStr = Field(alias="contextId", min_length=1)
    state: Literal["input-required"] = "input-required"


class PaymentRequirementRef(FrozenModel):
    schema_version: Literal["payment-requirement-snapshot/1"] = Field(
        default="payment-requirement-snapshot/1", alias="schemaVersion"
    )
    task_id: StrictStr = Field(alias="taskId", min_length=1)
    context_id: StrictStr = Field(alias="contextId", min_length=1)
    order_id: StrictStr = Field(alias="orderId", min_length=1)
    quote_id: StrictStr = Field(alias="quoteId", min_length=1)
    payment_required: dict[str, Any] = Field(alias="paymentRequired")
    requirement_digest: StrictStr = Field(
        alias="requirementDigest", pattern=r"^sha256:[0-9a-f]{64}$"
    )
    checkout_jwt: StrictStr = Field(alias="checkoutJwt", min_length=1)
    checkout_hash: StrictStr = Field(alias="checkoutHash", min_length=16)
    amount_minor: Literal[1250] = Field(alias="amountMinor")
    currency: Literal["USD"] = "USD"
    payee: Literal["demo-merchant"] = "demo-merchant"
    profile_id: Literal["x402-wire-simulation/1"] = Field(alias="profileId")
    extension_uri: Literal[
        "urn:secure-a2a:extensions:x402-wire-simulation:v1"
    ] = Field(alias="extensionUri")
    checkout_audience: Literal["demo-merchant"] = Field(alias="checkoutAudience")
    checkout_nonce: StrictStr = Field(alias="checkoutNonce", min_length=16)
    payment_audience: Literal["demo-credential-provider"] = Field(alias="paymentAudience")
    payment_nonce: StrictStr = Field(alias="paymentNonce", min_length=16)
    expires_at: StrictStr = Field(alias="expiresAt", min_length=1)

    @model_validator(mode="after")
    def exact_profile_and_digest(self) -> "PaymentRequirementRef":
        if canonical_digest(self.payment_required) != self.requirement_digest:
            raise ValueError("payment requirement digest mismatch")
        if b64url_sha256(self.checkout_jwt) != self.checkout_hash:
            raise ValueError("Checkout hash mismatch")
        accepts = self.payment_required.get("accepts")
        expected = {
            "scheme": "exact-simulated",
            "network": "demo:local",
            "asset": "USD",
            "payTo": "merchant:demo-merchant",
            "maxAmountRequired": "1250",
        }
        if self.payment_required.get("x402Version") != 1 or accepts != [expected]:
            raise ValueError("unsupported or malformed payment profile")
        try:
            expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("payment requirement expiry is invalid") from error
        if expiry.tzinfo is None:
            raise ValueError("payment requirement expiry must include a timezone")
        return self


class BridgeAttachment(FrozenModel):
    continuation_id: StrictStr = Field(alias="continuationId")
    payment_workflow_id: StrictStr = Field(alias="paymentWorkflowId")
    state: BridgeState
    version: StrictInt
    task_id: StrictStr = Field(alias="taskId")
    context_id: StrictStr = Field(alias="contextId")
    order_id: StrictStr = Field(alias="orderId")
    quote_id: StrictStr = Field(alias="quoteId")
    requirement_digest: StrictStr = Field(alias="requirementDigest")
    checkout_digest: StrictStr = Field(alias="checkoutDigest")
    created: bool


class PaymentApprovalResult(FrozenModel):
    continuation_id: StrictStr = Field(alias="continuationId")
    payment_approval_id: StrictStr = Field(alias="paymentApprovalId")
    state: BridgeState
    version: StrictInt
    approval_digest: StrictStr = Field(alias="approvalDigest")
    checkout_mandate_digest: StrictStr = Field(alias="checkoutMandateDigest")
    payment_mandate_digest: StrictStr = Field(alias="paymentMandateDigest")
    expires_at: StrictStr = Field(alias="expiresAt")


class BridgeExecutionResult(FrozenModel):
    continuation_id: StrictStr = Field(alias="continuationId")
    operation_id: StrictStr = Field(alias="operationId")
    state: BridgeState
    version: StrictInt
    task_id: StrictStr = Field(alias="taskId")
    context_id: StrictStr = Field(alias="contextId")
    guarantee_id: StrictStr | None = Field(default=None, alias="guaranteeId")
    guarantee_digest: StrictStr | None = Field(default=None, alias="guaranteeDigest")
    settlement_id: StrictStr | None = Field(default=None, alias="settlementId")
    settlement_receipt_digest: StrictStr | None = Field(
        default=None, alias="settlementReceiptDigest"
    )
    result_digest: StrictStr | None = Field(default=None, alias="resultDigest")


class RefundResult(FrozenModel):
    continuation_id: StrictStr = Field(alias="continuationId")
    refund_id: StrictStr = Field(alias="refundId")
    original_settlement_id: StrictStr = Field(alias="originalSettlementId")
    amount_minor: Literal[1250] = Field(alias="amountMinor")
    currency: Literal["USD"] = "USD"
    state: Literal["refunded"] = "refunded"
    result_digest: StrictStr = Field(alias="resultDigest")
    version: StrictInt


@dataclass(frozen=True, slots=True)
class PaymentA2AOperation:
    operation_id: str
    phase: Literal["guarantee-submit", "fulfillment-commit"]
    canonical_agent_id: Literal["agent-005"]
    rpc_endpoint: Literal["http://127.0.0.1:8005/a2a"]
    task_id: str
    context_id: str
    order_id: str
    quote_id: str
    message: Message


class PaymentA2AExecutor(Protocol):
    def execute(self, operation: PaymentA2AOperation) -> Task: ...


class PaymentSubmissionRejected(RuntimeError):
    """Authoritative rejection proving that the remote effect did not occur."""


class PaymentResultUnknown(RuntimeError):
    """Transport outcome is unknown; callers must not create another payment."""


class PaymentBridge:
    def __init__(
        self,
        repository: WorkflowRepository,
        keys: DemoKeySet,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.repository = repository
        self.keys = keys
        self.clock = clock
        self.profile = ProfileRegistry.load(
            SIMULATION_PROFILE, simulation_key=keys.simulation_signer
        )
        self.trusted_surface = TrustedSurface(keys)
        self.credential_provider = CredentialProvider(keys)

    @staticmethod
    def _owner_tuple(owner: OwnerRef) -> tuple[str, str, str, str, str]:
        return (
            owner.tenant_id,
            owner.subject_id,
            owner.session_id,
            owner.context_id,
            owner.mediation_session_id,
        )

    def attach(
        self,
        owner: OwnerRef | dict[str, Any],
        approved_plan: ApprovedPlanRef | dict[str, Any],
        step: ApprovedStepRef | dict[str, Any],
        remote_task: RemoteTaskRef | dict[str, Any],
        requirement: PaymentRequirementRef | dict[str, Any] | None,
    ) -> BridgeAttachment:
        if requirement is None:
            raise DomainError(
                "PAYMENT_NOT_REQUIRED",
                "Free A2A results are not attached to the payment bridge.",
                "payment-bridge",
            )
        owner = OwnerRef.model_validate(owner)
        approved_plan = ApprovedPlanRef.model_validate(approved_plan)
        step = ApprovedStepRef.model_validate(step)
        remote_task = RemoteTaskRef.model_validate(remote_task)
        requirement = PaymentRequirementRef.model_validate(requirement)
        if (remote_task.task_id, remote_task.context_id) != (
            requirement.task_id,
            requirement.context_id,
        ):
            raise DomainError(
                "X402_TASK_CORRELATION_MISMATCH",
                "Payment requirement does not belong to the approved remote Task.",
                remote_task.task_id,
            )
        expires = datetime.fromisoformat(requirement.expires_at.replace("Z", "+00:00"))
        if self.clock() >= expires:
            raise DomainError(
                "PAYMENT_APPROVAL_EXPIRED", "Payment requirement expired.", remote_task.task_id
            )
        continuation_id = _stable_id(
            "continuation",
            owner.tenant_id,
            owner.subject_id,
            owner.session_id,
            owner.mediation_session_id,
            approved_plan.plan_id,
            str(approved_plan.plan_version),
            step.step_id,
        )
        payment_workflow_id = _stable_id("payment-workflow", continuation_id)
        attach_input = {
            "owner": owner.model_dump(mode="json", by_alias=True),
            "approvedPlan": approved_plan.model_dump(mode="json", by_alias=True),
            "step": step.model_dump(mode="json", by_alias=True),
            "remoteTask": remote_task.model_dump(mode="json", by_alias=True),
            "requirement": requirement.model_dump(mode="json", by_alias=True),
        }
        attach_digest = canonical_digest(attach_input)
        now = utc_now()
        values = (
            continuation_id,
            payment_workflow_id,
            owner.tenant_id,
            owner.subject_id,
            owner.session_id,
            owner.context_id,
            owner.mediation_session_id,
            approved_plan.plan_id,
            approved_plan.plan_version,
            approved_plan.plan_digest,
            approved_plan.approval_id,
            step.step_id,
            step.canonical_agent_id,
            step.agent_card_digest,
            step.rpc_endpoint,
            remote_task.task_id,
            remote_task.context_id,
            requirement.order_id,
            requirement.quote_id,
            canonical_json(requirement.model_dump(mode="json", by_alias=True)),
            requirement.requirement_digest,
            requirement.checkout_jwt,
            requirement.checkout_hash,
            requirement.amount_minor,
            requirement.currency,
            requirement.payee,
            requirement.profile_id,
            requirement.expires_at,
            attach_digest,
            now,
            now,
        )
        created = False
        try:
            with self.repository.transaction() as conn:
                existing = conn.execute(
                    "SELECT attach_digest FROM payment_continuations_v3 WHERE continuation_id=?",
                    (continuation_id,),
                ).fetchone()
                if existing is not None:
                    if existing["attach_digest"] != attach_digest:
                        raise DomainError(
                            "IDEMPOTENCY_CONFLICT",
                            "Continuation attach input changed.",
                            continuation_id,
                        )
                else:
                    conn.execute(
                        "INSERT INTO payment_continuations_v3(continuation_id,payment_workflow_id,"
                        "tenant_id,subject_id,session_id,context_id,mediation_session_id,plan_id,"
                        "plan_version,plan_digest,plan_approval_id,step_id,canonical_agent_id,"
                        "agent_card_digest,rpc_endpoint,task_id,task_context_id,order_id,quote_id,"
                        "requirement_json,requirement_digest,checkout_jwt,checkout_hash,amount_minor,"
                        "currency,payee,profile_id,expires_at,attach_digest,state,version,created_at,updated_at)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
                        "'waiting_for_payment_approval',1,?,?)",
                        values,
                    )
                    created = True
        except sqlite3.IntegrityError as error:
            raise DomainError(
                "IDEMPOTENCY_CONFLICT",
                "A different continuation already owns this Task or step.",
                continuation_id,
            ) from error
        return self._attachment(self._get(continuation_id), created=created)

    def approve(
        self,
        continuation_id: str,
        expected_version: int,
        approval_text: str,
        *,
        owner: OwnerRef | dict[str, Any],
    ) -> PaymentApprovalResult:
        owner = OwnerRef.model_validate(owner)
        if approval_text != APPROVAL_TOKEN:
            raise DomainError(
                "APPROVAL_EXACT_TOKEN_REQUIRED",
                "Payment approval must exactly match the required token.",
                continuation_id,
            )
        row = self._get(continuation_id)
        self._require_owner(row, owner)
        self._require_state_version(
            row, BridgeState.WAITING_FOR_PAYMENT_APPROVAL, expected_version
        )
        requirement = PaymentRequirementRef.model_validate(row["requirement"])
        now = self.clock()
        requirement_expiry = datetime.fromisoformat(
            requirement.expires_at.replace("Z", "+00:00")
        )
        if now >= requirement_expiry:
            raise DomainError(
                "PAYMENT_APPROVAL_EXPIRED", "Payment approval window expired.", continuation_id
            )
        expires = min(now + timedelta(minutes=10), requirement_expiry)
        issued_at = int(now.timestamp())
        expires_at = int(expires.timestamp())
        approval_id = _stable_id("payment-approval", continuation_id)
        display = {
            "taskId": row["task_id"],
            "contextId": row["task_context_id"],
            "orderId": row["order_id"],
            "quoteId": row["quote_id"],
            "merchant": row["payee"],
            "amountMinor": row["amount_minor"],
            "currency": row["currency"],
            "profileId": row["profile_id"],
            "simulated": True,
            "state": "WAITING_FOR_PAYMENT_APPROVAL",
        }
        display_digest = canonical_digest(display)
        mandates = self.trusted_surface.issue_closed_mandates(
            checkout_jwt=requirement.checkout_jwt,
            merchant_id=requirement.payee,
            merchant_name="Demo Merchant",
            amount=requirement.amount_minor,
            currency=requirement.currency,
            instrument_id="demo-instrument-1",
            checkout_audience=requirement.checkout_audience,
            checkout_nonce=requirement.checkout_nonce,
            payment_audience=requirement.payment_audience,
            payment_nonce=requirement.payment_nonce,
            issued_at=issued_at,
            expires_at=expires_at,
        )
        self.credential_provider.verify_payment_mandate(
            mandates.payment,
            nonce=requirement.payment_nonce,
            checkout_hash=requirement.checkout_hash,
            amount=requirement.amount_minor,
        )
        checkout_evidence_id, checkout_digest = self._put_evidence(
            row, "checkout-mandate", mandates.checkout, self.keys.user_root.get("kid")
        )
        payment_evidence_id, payment_digest = self._put_evidence(
            row, "payment-mandate", mandates.payment, self.keys.user_root.get("kid")
        )
        approval_message_digest = sha256_digest(approval_text)
        owner_digest = canonical_digest(owner)
        nonce = _stable_id("payment-approval-nonce", continuation_id)
        approval_digest = canonical_digest(
            {
                "approvalId": approval_id,
                "ownerDigest": owner_digest,
                "displayDigest": display_digest,
                "approvalMessageDigest": approval_message_digest,
                "checkoutMandateDigest": checkout_digest,
                "paymentMandateDigest": payment_digest,
                "nonce": nonce,
                "issuedAt": issued_at,
                "expiresAt": expires_at,
            }
        )
        with self.repository.transaction() as conn:
            current = conn.execute(
                "SELECT state,version,tenant_id,subject_id,session_id,context_id,"
                "mediation_session_id "
                "FROM payment_continuations_v3 WHERE continuation_id=?",
                (continuation_id,),
            ).fetchone()
            if current is None:
                raise KeyError(continuation_id)
            if tuple(
                current[name]
                for name in (
                    "tenant_id",
                    "subject_id",
                    "session_id",
                    "context_id",
                    "mediation_session_id",
                )
            ) != self._owner_tuple(owner):
                raise DomainError(
                    "TENANT_BINDING_MISMATCH", "Payment owner changed.", continuation_id
                )
            if current["state"] != BridgeState.WAITING_FOR_PAYMENT_APPROVAL or current["version"] != expected_version:
                raise DomainError(
                    "STATE_TRANSITION_CONFLICT", "Payment approval lost CAS.", continuation_id,
                    current_state=current["state"],
                )
            conn.execute(
                "INSERT INTO payment_bridge_approvals_v3(approval_id,continuation_id,owner_digest,"
                "display_digest,approval_message_digest,nonce,checkout_mandate_evidence_id,"
                "checkout_mandate_digest,payment_mandate_evidence_id,payment_mandate_digest,"
                "approved_at,expires_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    approval_id, continuation_id, owner_digest, display_digest,
                    approval_message_digest, nonce, checkout_evidence_id, checkout_digest,
                    payment_evidence_id, payment_digest, now.isoformat().replace("+00:00", "Z"),
                    expires.isoformat().replace("+00:00", "Z"),
                ),
            )
            conn.execute(
                "UPDATE payment_continuations_v3 SET state='payment_approved',version=version+1,"
                "payment_approval_id=?,payment_approval_digest=?,checkout_mandate_digest=?,"
                "payment_mandate_digest=?,updated_at=? WHERE continuation_id=? AND version=?",
                (
                    approval_id, approval_digest, checkout_digest, payment_digest, utc_now(),
                    continuation_id, expected_version,
                ),
            )
        current = self._get(continuation_id)
        return PaymentApprovalResult(
            continuationId=continuation_id,
            paymentApprovalId=approval_id,
            state=BridgeState(current["state"]),
            version=current["version"],
            approvalDigest=approval_digest,
            checkoutMandateDigest=checkout_digest,
            paymentMandateDigest=payment_digest,
            expiresAt=expires.isoformat().replace("+00:00", "Z"),
        )

    def execute_approved_payment(
        self,
        operation_id: str,
        continuation_id: str,
        expected_version: int,
        executor: PaymentA2AExecutor,
    ) -> BridgeExecutionResult:
        expected_operation = f"payment-submit:{continuation_id}:1"
        if operation_id != expected_operation:
            raise DomainError(
                "IDEMPOTENCY_CONFLICT", "Payment operation ID is not server-derived.", continuation_id
            )
        row = self._get(continuation_id)
        if row["state"] == BridgeState.COMPLETED:
            return self._execution_result(row, operation_id)
        self._require_state_version(row, BridgeState.PAYMENT_APPROVED, expected_version)
        approval = self._approval(continuation_id)
        approval_expiry = datetime.fromisoformat(
            approval["expires_at"].replace("Z", "+00:00")
        )
        if self.clock() >= approval_expiry:
            raise DomainError(
                "PAYMENT_APPROVAL_EXPIRED",
                "The exact payment approval expired before execution.",
                continuation_id,
            )
        checkout_mandate = self.repository.read_evidence(
            approval["checkout_mandate_evidence_id"],
            actor_id=row["subject_id"], actor_role="customer", tenant_id=row["tenant_id"],
        ).decode("utf-8")
        payment_mandate = self.repository.read_evidence(
            approval["payment_mandate_evidence_id"],
            actor_id=row["subject_id"], actor_role="customer", tenant_id=row["tenant_id"],
        ).decode("utf-8")
        requirement = PaymentRequirementRef.model_validate(row["requirement"])
        self.credential_provider.verify_payment_mandate(
            payment_mandate,
            nonce=requirement.payment_nonce,
            checkout_hash=requirement.checkout_hash,
            amount=requirement.amount_minor,
        )
        now = self.clock()
        issued_at = int(now.timestamp())
        expires_at = min(
            issued_at + 600,
            int(datetime.fromisoformat(requirement.expires_at.replace("Z", "+00:00")).timestamp()),
        )
        authorization_id = _stable_id("payment-authorization-envelope", continuation_id)
        authorization_claims = {
            "schemaVersion": "pre-payment-authorization-envelope/1",
            "jti": authorization_id,
            "iss": "secure-mediator-plan-authority",
            "aud": "secure-mediator-payment-authority",
            "continuationId": continuation_id,
            "paymentWorkflowId": row["payment_workflow_id"],
            "planId": row["plan_id"],
            "planVersion": row["plan_version"],
            "planDigest": row["plan_digest"],
            "stepId": row["step_id"],
            "taskId": row["task_id"],
            "contextId": row["task_context_id"],
            "orderId": row["order_id"],
            "quoteId": row["quote_id"],
            "paymentApprovalId": row["payment_approval_id"],
            "paymentApprovalDigest": row["payment_approval_digest"],
            "checkoutMandateDigest": sha256_digest(checkout_mandate),
            "paymentMandateDigest": sha256_digest(payment_mandate),
            "requirementsDigest": row["requirement_digest"],
            "amountMinor": row["amount_minor"],
            "currency": row["currency"],
            "payee": row["payee"],
            "iat": issued_at,
            "exp": expires_at,
        }
        authorization_envelope = create_jwt(
            {"alg": "ES256", "kid": self.keys.plan_authority.get("kid"), "typ": "JWT"},
            authorization_claims,
            self.keys.plan_authority,
        )
        authorization_evidence_id, authorization_digest = self._put_evidence(
            row,
            "pre-payment-authorization-envelope",
            authorization_envelope,
            self.keys.plan_authority.get("kid"),
        )
        guarantee_id = _stable_id("payment-guarantee", continuation_id)
        settlement_commitment_id = _stable_id("settlement-commitment", continuation_id)
        guarantee = self.profile.issue_guarantee(
            {
                "guaranteeId": guarantee_id,
                "iss": "secure-mediator-payment-authority",
                "aud": "a2a-agent:agent-005",
                "operation": "merchant.fulfillment.guarantee",
                "taskId": row["task_id"],
                "contextId": row["task_context_id"],
                "orderId": row["order_id"],
                "quoteId": row["quote_id"],
                "amountMinor": row["amount_minor"],
                "currency": row["currency"],
                "payee": row["payee"],
                "paymentMandateDigest": sha256_digest(payment_mandate),
                "authorizationEnvelopeDigest": authorization_digest,
                "settlementCommitmentId": settlement_commitment_id,
                "jti": guarantee_id,
                "iat": issued_at,
                "nbf": issued_at,
                "exp": expires_at,
            }
        )
        guarantee_evidence_id, guarantee_digest = self._put_evidence(
            row, "signed-payment-guarantee", guarantee, self.keys.simulation_signer.get("kid")
        )
        payload = self.profile.build_guarantee_submission(
            guarantee=guarantee,
            guarantee_digest=guarantee_digest,
            checkout_mandate_digest=sha256_digest(checkout_mandate),
            payment_mandate_digest=sha256_digest(payment_mandate),
            authorization_envelope_digest=authorization_digest,
        )
        message = payment_message(
            task_id=row["task_id"],
            context_id=row["task_context_id"],
            message_id=f"message:{operation_id}",
            status="payment-submitted",
            payload=payload,
            project={
                "canonicalAgentId": row["canonical_agent_id"],
                "orderId": row["order_id"],
                "quoteId": row["quote_id"],
                "profileId": row["profile_id"],
                "paymentGuaranteeDigest": guarantee_digest,
                "simulated": True,
            },
        )
        message_digest = canonical_digest(
            message.model_dump(mode="json", by_alias=True, exclude_none=True)
        )
        self._record_guarantee(
            row=row,
            expected_version=expected_version,
            guarantee_id=guarantee_id,
            guarantee_digest=guarantee_digest,
            guarantee_evidence_id=guarantee_evidence_id,
            authorization_evidence_id=authorization_evidence_id,
            authorization_digest=authorization_digest,
            settlement_commitment_id=settlement_commitment_id,
            operation_id=operation_id,
            request_digest=message_digest,
            payload=message.model_dump(mode="json", by_alias=True, exclude_none=True),
        )
        submit_operation = PaymentA2AOperation(
            operation_id=operation_id,
            phase="guarantee-submit",
            canonical_agent_id=PAID_AGENT_ID,
            rpc_endpoint=PAID_RPC_ENDPOINT,
            task_id=row["task_id"],
            context_id=row["task_context_id"],
            order_id=row["order_id"],
            quote_id=row["quote_id"],
            message=message,
        )
        try:
            submitted = executor.execute(submit_operation)
            self._require_same_task(submitted, row, allowed={TaskState.working, TaskState.input_required})
        except PaymentSubmissionRejected as error:
            self._cancel_guarantee(continuation_id, operation_id, type(error).__name__)
            raise DomainError(
                "PAYMENT_FAILED", "Merchant rejected the signed guarantee.", continuation_id
            ) from error
        except Exception as error:
            self._mark_review(continuation_id, operation_id, type(error).__name__)
            raise DomainError(
                "RECONCILIATION_REQUIRED",
                "Guarantee submission result is unknown; no new payment may be created.",
                continuation_id,
            ) from error
        submission_result_digest = canonical_digest(
            submitted.model_dump(mode="json", by_alias=True, exclude_none=True)
        )
        row = self._mark_submitted(continuation_id, operation_id, submission_result_digest)
        settlement_id = _stable_id("settlement", continuation_id)
        settlement_request_digest = canonical_digest(
            {
                "settlementId": settlement_id,
                "guaranteeId": guarantee_id,
                "taskId": row["task_id"],
                "orderId": row["order_id"],
                "amountMinor": row["amount_minor"],
                "currency": row["currency"],
            }
        )
        settlement_receipt = self.profile.settle_receipt(
            attempt_id=settlement_id, success=True
        )
        _, settlement_receipt_digest = self._put_evidence(
            row, "simulation-settlement-receipt", canonical_bytes(settlement_receipt), None
        )
        row = self._settle(
            row,
            settlement_id=settlement_id,
            request_digest=settlement_request_digest,
            receipt=settlement_receipt,
            receipt_digest=settlement_receipt_digest,
        )
        commit_operation_id = f"fulfillment-commit:{continuation_id}:1"
        commit_message = payment_message(
            task_id=row["task_id"],
            context_id=row["task_context_id"],
            message_id=f"message:{commit_operation_id}",
            status="payment-settled",
            payload={
                "schemaVersion": "merchant-fulfillment-commit/1",
                "guaranteeId": guarantee_id,
                "settlementId": settlement_id,
                "settlementReceipt": settlement_receipt,
                "settlementReceiptDigest": settlement_receipt_digest,
            },
            project={"orderId": row["order_id"], "quoteId": row["quote_id"], "simulated": True},
        )
        row = self._begin_fulfillment(row, commit_operation_id, commit_message)
        try:
            completed = executor.execute(
                PaymentA2AOperation(
                    operation_id=commit_operation_id,
                    phase="fulfillment-commit",
                    canonical_agent_id=PAID_AGENT_ID,
                    rpc_endpoint=PAID_RPC_ENDPOINT,
                    task_id=row["task_id"],
                    context_id=row["task_context_id"],
                    order_id=row["order_id"],
                    quote_id=row["quote_id"],
                    message=commit_message,
                )
            )
            self._require_same_task(completed, row, allowed={TaskState.completed})
        except PaymentSubmissionRejected as error:
            self._require_refund(continuation_id, commit_operation_id, type(error).__name__)
            raise DomainError(
                "REFUND_REQUIRED",
                "Settlement succeeded but fulfillment failed; one full refund is required.",
                continuation_id,
            ) from error
        except Exception as error:
            self._mark_review(continuation_id, commit_operation_id, type(error).__name__)
            raise DomainError(
                "RECONCILIATION_REQUIRED",
                "Fulfillment result is unknown; settlement must not be repeated.",
                continuation_id,
            ) from error
        result_digest = canonical_digest(
            completed.model_dump(mode="json", by_alias=True, exclude_none=True)
        )
        row = self._complete(continuation_id, commit_operation_id, result_digest)
        return self._execution_result(row, operation_id)

    def refund(
        self, operation_id: str, continuation_id: str, expected_version: int
    ) -> RefundResult:
        expected_operation = f"refund:{continuation_id}:1"
        if operation_id != expected_operation:
            raise DomainError(
                "IDEMPOTENCY_CONFLICT", "Refund operation ID is not server-derived.", continuation_id
            )
        row = self._get(continuation_id)
        if row["state"] == BridgeState.REFUNDED:
            return self._refund_result(row)
        self._require_state_version(row, BridgeState.REFUND_REQUIRED, expected_version)
        refund_id = _stable_id("refund", continuation_id)
        request = {
            "schemaVersion": "refund-request/1",
            "refundId": refund_id,
            "continuationId": continuation_id,
            "settlementId": row["settlement_id"],
            "taskId": row["task_id"],
            "contextId": row["task_context_id"],
            "orderId": row["order_id"],
            "quoteId": row["quote_id"],
            "amountMinor": row["amount_minor"],
            "currency": row["currency"],
            "reason": "fulfillment-failed",
            "idempotencyKey": operation_id,
        }
        request_digest = canonical_digest(request)
        result = {
            "schemaVersion": "refund-result/1",
            "refundId": refund_id,
            "originalSettlementId": row["settlement_id"],
            "amountMinor": row["amount_minor"],
            "currency": row["currency"],
            "status": "refunded",
            "processedAt": utc_now(),
            "simulated": True,
        }
        result_digest = canonical_digest(result)
        with self.repository.transaction() as conn:
            current = conn.execute(
                "SELECT * FROM payment_continuations_v3 WHERE continuation_id=?",
                (continuation_id,),
            ).fetchone()
            if current is None:
                raise KeyError(continuation_id)
            if current["state"] != BridgeState.REFUND_REQUIRED or current["version"] != expected_version:
                raise DomainError(
                    "STATE_TRANSITION_CONFLICT", "Refund lost CAS.", continuation_id,
                    current_state=current["state"],
                )
            conn.execute(
                "UPDATE rail_accounts_v2 SET balance=balance+?,updated_at=? "
                "WHERE account_id='demo-customer' AND asset=?",
                (current["amount_minor"], utc_now(), current["currency"]),
            )
            conn.execute(
                "UPDATE rail_accounts_v2 SET balance=balance-?,updated_at=? "
                "WHERE account_id='demo-merchant' AND asset=?",
                (current["amount_minor"], utc_now(), current["currency"]),
            )
            conn.execute(
                "INSERT INTO payment_bridge_outbox_v3(outbox_id,continuation_id,event_type,"
                "operation_id,request_digest,payload_json,status,attempts,created_at,completed_at) "
                "VALUES(?,?,?,?,?,?,'done',1,?,?)",
                (
                    f"outbox:{operation_id}",
                    continuation_id,
                    "refund-submit",
                    operation_id,
                    request_digest,
                    canonical_json(request),
                    utc_now(),
                    utc_now(),
                ),
            )
            conn.execute(
                "INSERT INTO payment_bridge_refunds_v3(refund_id,continuation_id,settlement_id,"
                "amount_minor,currency,reason,idempotency_key,request_digest,result_json,result_digest,"
                "state,created_at) VALUES(?,?,?,?,?,'fulfillment-failed',?,?,?,?, 'refunded',?)",
                (
                    refund_id, continuation_id, current["settlement_id"], current["amount_minor"],
                    current["currency"], operation_id, request_digest, canonical_json(result),
                    result_digest, utc_now(),
                ),
            )
            conn.execute(
                "UPDATE payment_continuations_v3 SET state='refunded',version=version+1,updated_at=? "
                "WHERE continuation_id=? AND version=?",
                (utc_now(), continuation_id, expected_version),
            )
        return self._refund_result(self._get(continuation_id))

    def status(self, continuation_id: str, *, owner: OwnerRef | dict[str, Any]) -> BridgeExecutionResult:
        owner = OwnerRef.model_validate(owner)
        row = self._get(continuation_id)
        self._require_owner(row, owner)
        return self._execution_result(row, f"payment-submit:{continuation_id}:1")

    def counts(self, continuation_id: str) -> dict[str, int]:
        with self.repository._connect(self.repository.paths.marketplace) as conn:
            return {
                "continuations": conn.execute(
                    "SELECT COUNT(*) FROM payment_continuations_v3 WHERE continuation_id=?",
                    (continuation_id,),
                ).fetchone()[0],
                "approvals": conn.execute(
                    "SELECT COUNT(*) FROM payment_bridge_approvals_v3 WHERE continuation_id=?",
                    (continuation_id,),
                ).fetchone()[0],
                "guarantees": conn.execute(
                    "SELECT COUNT(*) FROM payment_guarantees_v3 WHERE continuation_id=?",
                    (continuation_id,),
                ).fetchone()[0],
                "settlements": conn.execute(
                    "SELECT COUNT(*) FROM payment_bridge_settlements_v3 WHERE continuation_id=?",
                    (continuation_id,),
                ).fetchone()[0],
                "refunds": conn.execute(
                    "SELECT COUNT(*) FROM payment_bridge_refunds_v3 WHERE continuation_id=?",
                    (continuation_id,),
                ).fetchone()[0],
            }

    def _get(self, continuation_id: str) -> dict[str, Any]:
        with self.repository._connect(self.repository.paths.marketplace) as conn:
            row = conn.execute(
                "SELECT * FROM payment_continuations_v3 WHERE continuation_id=?",
                (continuation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(continuation_id)
        result = dict(row)
        result["requirement"] = json.loads(result.pop("requirement_json"))
        return result

    def _approval(self, continuation_id: str) -> dict[str, Any]:
        with self.repository._connect(self.repository.paths.marketplace) as conn:
            row = conn.execute(
                "SELECT * FROM payment_bridge_approvals_v3 WHERE continuation_id=?",
                (continuation_id,),
            ).fetchone()
        if row is None:
            raise DomainError(
                "PAYMENT_APPROVAL_REQUIRED", "Payment approval evidence is missing.", continuation_id
            )
        return dict(row)

    def _put_evidence(
        self, row: dict[str, Any], kind: str, exact: str | bytes, kid: str | None
    ) -> tuple[str, str]:
        evidence_id = _stable_id(f"evidence:{kind}", row["continuation_id"])
        digest = self.repository.put_evidence(
            workflow_id=row["payment_workflow_id"],
            evidence_id=evidence_id,
            tenant_id=row["tenant_id"],
            kind=kind,
            exact_bytes=exact,
            kid=kid,
            media_type="application/jwt" if isinstance(exact, str) else "application/json",
            profile_id=("AP2-v0.2" if "mandate" in kind or "authorization" in kind else row["profile_id"]),
        )
        return evidence_id, digest

    def _record_guarantee(self, **values: Any) -> None:
        row = values["row"]
        now = utc_now()
        with self.repository.transaction() as conn:
            current = conn.execute(
                "SELECT state,version FROM payment_continuations_v3 WHERE continuation_id=?",
                (row["continuation_id"],),
            ).fetchone()
            if current["state"] != BridgeState.PAYMENT_APPROVED or current["version"] != values["expected_version"]:
                raise DomainError(
                    "STATE_TRANSITION_CONFLICT", "Payment execution lost CAS.", row["continuation_id"],
                    current_state=current["state"],
                )
            conn.execute(
                "INSERT INTO payment_guarantees_v3(guarantee_id,continuation_id,"
                "authorization_envelope_evidence_id,authorization_envelope_digest,guarantee_evidence_id,"
                "guarantee_digest,settlement_commitment_id,state,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,'guaranteed',?,?)",
                (
                    values["guarantee_id"], row["continuation_id"],
                    values["authorization_evidence_id"], values["authorization_digest"],
                    values["guarantee_evidence_id"], values["guarantee_digest"],
                    values["settlement_commitment_id"], now, now,
                ),
            )
            conn.execute(
                "INSERT INTO payment_bridge_outbox_v3(outbox_id,continuation_id,event_type,operation_id,"
                "request_digest,payload_json,status,attempts,created_at) VALUES(?,?,?,?,?,?,'pending',0,?)",
                (
                    f"outbox:{values['operation_id']}", row["continuation_id"], "guarantee-submit",
                    values["operation_id"], values["request_digest"], canonical_json(values["payload"]), now,
                ),
            )
            conn.execute(
                "UPDATE payment_continuations_v3 SET state='guaranteed',version=version+1,"
                "authorization_envelope_digest=?,guarantee_id=?,guarantee_digest=?,updated_at=? "
                "WHERE continuation_id=? AND version=?",
                (
                    values["authorization_digest"], values["guarantee_id"],
                    values["guarantee_digest"], now, row["continuation_id"], values["expected_version"],
                ),
            )

    def _mark_submitted(self, continuation_id: str, operation_id: str, result_digest: str) -> dict[str, Any]:
        with self.repository.transaction() as conn:
            conn.execute(
                "UPDATE payment_bridge_outbox_v3 SET status='done',attempts=attempts+1,completed_at=? "
                "WHERE operation_id=?", (utc_now(), operation_id)
            )
            conn.execute(
                "UPDATE payment_guarantees_v3 SET state='submitted',updated_at=? WHERE continuation_id=?",
                (utc_now(), continuation_id),
            )
            conn.execute(
                "UPDATE payment_continuations_v3 SET state='payment_submitted',version=version+1,"
                "fulfillment_digest=?,updated_at=? WHERE continuation_id=? AND state='guaranteed'",
                (result_digest, utc_now(), continuation_id),
            )
        return self._get(continuation_id)

    def _cancel_guarantee(self, continuation_id: str, operation_id: str, code: str) -> None:
        with self.repository.transaction() as conn:
            conn.execute(
                "UPDATE payment_bridge_outbox_v3 SET status='failed',attempts=attempts+1,last_error_code=?,"
                "completed_at=? WHERE operation_id=?", (code, utc_now(), operation_id)
            )
            conn.execute(
                "UPDATE payment_guarantees_v3 SET state='cancelled',updated_at=? WHERE continuation_id=?",
                (utc_now(), continuation_id),
            )
            conn.execute(
                "UPDATE payment_continuations_v3 SET state='guarantee_cancelled',version=version+1,"
                "last_error_code=?,updated_at=? WHERE continuation_id=? AND state='guaranteed'",
                (code, utc_now(), continuation_id),
            )

    def _mark_review(self, continuation_id: str, operation_id: str, code: str) -> None:
        with self.repository.transaction() as conn:
            conn.execute(
                "UPDATE payment_bridge_outbox_v3 SET status='review-required',attempts=attempts+1,"
                "last_error_code=? WHERE operation_id=?", (code, operation_id)
            )
            conn.execute(
                "UPDATE payment_continuations_v3 SET state='review_required',version=version+1,"
                "last_error_code=?,updated_at=? WHERE continuation_id=? AND state IN "
                "('guaranteed','payment_submitted','fulfillment_committing')",
                (code, utc_now(), continuation_id),
            )

    def _settle(
        self,
        row: dict[str, Any],
        *,
        settlement_id: str,
        request_digest: str,
        receipt: dict[str, Any],
        receipt_digest: str,
    ) -> dict[str, Any]:
        with self.repository.transaction() as conn:
            current = conn.execute(
                "SELECT * FROM payment_continuations_v3 WHERE continuation_id=?",
                (row["continuation_id"],),
            ).fetchone()
            if current["state"] != BridgeState.PAYMENT_SUBMITTED:
                raise DomainError(
                    "STATE_TRANSITION_CONFLICT", "Payment is not settlement-ready.", row["continuation_id"],
                    current_state=current["state"],
                )
            balance = conn.execute(
                "SELECT balance FROM rail_accounts_v2 WHERE account_id='demo-customer' AND asset=?",
                (current["currency"],),
            ).fetchone()
            if balance is None or balance["balance"] < current["amount_minor"]:
                raise DomainError("PAYMENT_FAILED", "Simulation balance is insufficient.", row["continuation_id"])
            conn.execute(
                "UPDATE rail_accounts_v2 SET balance=balance-?,updated_at=? "
                "WHERE account_id='demo-customer' AND asset=?",
                (current["amount_minor"], utc_now(), current["currency"]),
            )
            conn.execute(
                "UPDATE rail_accounts_v2 SET balance=balance+?,updated_at=? "
                "WHERE account_id='demo-merchant' AND asset=?",
                (current["amount_minor"], utc_now(), current["currency"]),
            )
            conn.execute(
                "INSERT INTO payment_bridge_settlements_v3(settlement_id,continuation_id,guarantee_id,"
                "amount_minor,currency,request_digest,receipt_json,receipt_digest,state,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,'settled',?)",
                (
                    settlement_id, row["continuation_id"], current["guarantee_id"],
                    current["amount_minor"], current["currency"], request_digest,
                    canonical_json(receipt), receipt_digest, utc_now(),
                ),
            )
            conn.execute(
                "UPDATE payment_guarantees_v3 SET state='settled',updated_at=? WHERE continuation_id=?",
                (utc_now(), row["continuation_id"]),
            )
            conn.execute(
                "UPDATE payment_continuations_v3 SET state='settled',version=version+1,settlement_id=?,"
                "settlement_receipt_digest=?,updated_at=? WHERE continuation_id=?",
                (settlement_id, receipt_digest, utc_now(), row["continuation_id"]),
            )
        return self._get(row["continuation_id"])

    def _begin_fulfillment(self, row: dict[str, Any], operation_id: str, message: Message) -> dict[str, Any]:
        payload = message.model_dump(mode="json", by_alias=True, exclude_none=True)
        with self.repository.transaction() as conn:
            conn.execute(
                "INSERT INTO payment_bridge_outbox_v3(outbox_id,continuation_id,event_type,operation_id,"
                "request_digest,payload_json,status,attempts,created_at) VALUES(?,?,?,?,?,?,'pending',0,?)",
                (
                    f"outbox:{operation_id}", row["continuation_id"], "fulfillment-commit", operation_id,
                    canonical_digest(payload), canonical_json(payload), utc_now(),
                ),
            )
            conn.execute(
                "UPDATE payment_continuations_v3 SET state='fulfillment_committing',version=version+1,"
                "updated_at=? WHERE continuation_id=? AND state='settled'",
                (utc_now(), row["continuation_id"]),
            )
        return self._get(row["continuation_id"])

    def _complete(self, continuation_id: str, operation_id: str, result_digest: str) -> dict[str, Any]:
        with self.repository.transaction() as conn:
            conn.execute(
                "UPDATE payment_bridge_outbox_v3 SET status='done',attempts=attempts+1,completed_at=? "
                "WHERE operation_id=?", (utc_now(), operation_id)
            )
            changed = conn.execute(
                "UPDATE payment_continuations_v3 SET state='completed',version=version+1,"
                "fulfillment_digest=?,updated_at=? WHERE continuation_id=? "
                "AND state='fulfillment_committing'",
                (result_digest, utc_now(), continuation_id),
            ).rowcount
            if changed != 1:
                raise DomainError(
                    "STATE_TRANSITION_CONFLICT", "Fulfillment completion lost CAS.", continuation_id
                )
        return self._get(continuation_id)

    def _require_refund(self, continuation_id: str, operation_id: str, code: str) -> None:
        with self.repository.transaction() as conn:
            conn.execute(
                "UPDATE payment_bridge_outbox_v3 SET status='failed',attempts=attempts+1,last_error_code=?,"
                "completed_at=? WHERE operation_id=?", (code, utc_now(), operation_id)
            )
            conn.execute(
                "UPDATE payment_continuations_v3 SET state='refund_required',version=version+1,"
                "last_error_code=?,updated_at=? WHERE continuation_id=? AND state='fulfillment_committing'",
                (code, utc_now(), continuation_id),
            )

    @staticmethod
    def _require_same_task(task: Task, row: dict[str, Any], *, allowed: set[TaskState]) -> None:
        if task.id != row["task_id"] or task.context_id != row["task_context_id"]:
            raise PaymentResultUnknown("Merchant returned a different Task/context")
        if task.status.state not in allowed:
            raise PaymentResultUnknown("Merchant returned an invalid Task state")

    @staticmethod
    def _require_state_version(row: dict[str, Any], state: BridgeState, version: int) -> None:
        if row["state"] != state or row["version"] != version:
            raise DomainError(
                "STATE_TRANSITION_CONFLICT", "Payment continuation state changed.",
                row["continuation_id"], current_state=row["state"],
            )

    @staticmethod
    def _require_owner(row: dict[str, Any], owner: OwnerRef) -> None:
        if (
            row["tenant_id"],
            row["subject_id"],
            row["session_id"],
            row["context_id"],
            row["mediation_session_id"],
        ) != PaymentBridge._owner_tuple(owner):
            raise DomainError(
                "TENANT_BINDING_MISMATCH", "Payment continuation owner mismatch.",
                row["continuation_id"],
            )

    @staticmethod
    def _attachment(row: dict[str, Any], *, created: bool) -> BridgeAttachment:
        return BridgeAttachment(
            continuationId=row["continuation_id"], paymentWorkflowId=row["payment_workflow_id"],
            state=BridgeState(row["state"]), version=row["version"], taskId=row["task_id"],
            contextId=row["task_context_id"], orderId=row["order_id"], quoteId=row["quote_id"],
            requirementDigest=row["requirement_digest"], checkoutDigest=sha256_digest(row["checkout_jwt"]),
            created=created,
        )

    @staticmethod
    def _execution_result(row: dict[str, Any], operation_id: str) -> BridgeExecutionResult:
        return BridgeExecutionResult(
            continuationId=row["continuation_id"], operationId=operation_id,
            state=BridgeState(row["state"]), version=row["version"], taskId=row["task_id"],
            contextId=row["task_context_id"], guaranteeId=row.get("guarantee_id"),
            guaranteeDigest=row.get("guarantee_digest"), settlementId=row.get("settlement_id"),
            settlementReceiptDigest=row.get("settlement_receipt_digest"),
            resultDigest=row.get("fulfillment_digest"),
        )

    def _refund_result(self, row: dict[str, Any]) -> RefundResult:
        with self.repository._connect(self.repository.paths.marketplace) as conn:
            refund = conn.execute(
                "SELECT * FROM payment_bridge_refunds_v3 WHERE continuation_id=?",
                (row["continuation_id"],),
            ).fetchone()
        if refund is None:
            raise KeyError(row["continuation_id"])
        return RefundResult(
            continuationId=row["continuation_id"], refundId=refund["refund_id"],
            originalSettlementId=refund["settlement_id"], amountMinor=refund["amount_minor"],
            currency=refund["currency"], state="refunded", resultDigest=refund["result_digest"],
            version=row["version"],
        )
