"""Private adapter between mediation DTOs and the durable payment bridge."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from a2a.types import Task, TaskState, TaskStatus

from secure_mediation_agent.payment_bridge import (
    PaymentA2AOperation,
    PaymentBridge,
    PaymentSubmissionRejected,
)
from secure_mediation_agent.workflow.errors import DomainError

from .a2a_executor import A2AOperation
from .approval_targets import build_payment_approval_target
from .canonical import canonical_digest
from .errors import ReviewRequired, SecurityBlocked
from .models import (
    BridgeApprovalResult,
    BridgeAttachment,
    BridgeA2AExecutionSummary,
    BridgeExecutionResult,
    MediationPlan,
    MediationStep,
    PaymentRequirementSnapshot,
    PlanApproval,
    PrivatePaymentMaterial,
    RefundResult,
    RemoteTaskSnapshot,
    SelectedAgentSnapshot,
)
from .ports import A2AExecutorPort


def _owner_wire(owner: Any) -> dict[str, Any]:
    if not isinstance(owner, dict):
        raise SecurityBlocked("OWNER_INVALID", "The payment owner binding is invalid.")
    return owner


def _translate_domain_error(error: DomainError) -> Exception:
    code = getattr(error, "code", "PAYMENT_BRIDGE_ERROR")
    if code in {
        "RECONCILIATION_REQUIRED",
        "REFUND_REQUIRED",
        "STATE_TRANSITION_CONFLICT",
    }:
        return ReviewRequired(code, "The payment result requires review.")
    return SecurityBlocked(code, "The payment bridge rejected the operation.")


class _SynchronousPaymentExecutor:
    """Runs the async shared executor from the bridge's isolated worker thread."""

    def __init__(self, executor: A2AExecutorPort, *, workflow_id: str) -> None:
        self.executor = executor
        self.workflow_id = workflow_id
        self.last_response: RemoteTaskSnapshot | None = None
        self.executions: list[Any] = []

    @staticmethod
    def _agent(operation: PaymentA2AOperation) -> SelectedAgentSnapshot:
        card_digest = canonical_digest(
            {
                "canonicalAgentId": operation.canonical_agent_id,
                "rpcEndpoint": operation.rpc_endpoint,
            }
        )
        wire = {
            "canonicalAgentId": operation.canonical_agent_id,
            "registryName": "paid_booking_agent",
            "a2aAgentName": "paid-booking-agent",
            "agentCardUrl": "http://127.0.0.1:8005/.well-known/agent-card.json",
            "rpcEndpoint": operation.rpc_endpoint,
            "a2aSkillId": "paid-booking",
            "trustScore": 100,
            "cardDigest": card_digest,
            "paymentExtensionUris": (
                "urn:secure-a2a:extensions:x402-wire-simulation:v1",
            ),
        }
        return SelectedAgentSnapshot(
            **wire, snapshotDigest=canonical_digest(wire)
        )

    def execute(self, operation: PaymentA2AOperation) -> Task:
        message = operation.message.model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
        action = {
            "guarantee-submit": "merchant:payment-guarantee-submit",
            "fulfillment-commit": "merchant:guaranteed-fulfillment-commit",
        }[operation.phase]
        request = {
            "jsonrpc": "2.0",
            "id": operation.operation_id,
            "method": "message/send",
            "params": {
                "action": action,
                "operationId": operation.operation_id,
                "workflowId": self.workflow_id,
                "taskId": operation.task_id,
                "orderId": operation.order_id,
                "capabilityId": f"capability:{operation.operation_id}",
                "message": message,
            },
        }
        mediated = A2AOperation(
            operationId=operation.operation_id,
            kind="payment-submit",
            agent=self._agent(operation),
            request=request,
            requestDigest=canonical_digest(request),
            idempotencyKey=operation.operation_id,
            taskId=operation.task_id,
            contextId=operation.context_id,
        )
        try:
            execution = asyncio.run(self.executor.execute(mediated))
            self.executions.append(execution)
        except SecurityBlocked as error:
            if error.code in {
                "LEGACY_CALLBACK_BEFORE_FAILED",
                "PRE_PAYMENT_SUBMIT_BLOCKED",
                "A2A_REQUEST_REJECTED",
            }:
                raise PaymentSubmissionRejected(error.code) from error
            raise
        response = execution.response.task
        self.last_response = response
        return Task(
            id=response.task_id,
            contextId=response.context_id,
            status=TaskStatus(state=TaskState(response.state)),
        )


class DurablePaymentBridgeAdapter:
    """Combines public Task facts with secrets only at the bridge boundary."""

    def __init__(self, bridge: PaymentBridge, *, executor: A2AExecutorPort) -> None:
        self.bridge = bridge
        self.executor = executor

    def attach(
        self,
        *,
        owner: Any,
        approved_plan: Any,
        step: Any,
        remote_task: Any,
        requirement: Any,
    ) -> BridgeAttachment:
        try:
            plan = MediationPlan.model_validate(approved_plan["plan"])
            approval = PlanApproval.model_validate(approved_plan["approval"])
            selected_step = MediationStep.model_validate(step)
            remote = RemoteTaskSnapshot.model_validate(remote_task)
            public_requirement = PaymentRequirementSnapshot.model_validate(
                requirement["requirement"]
            )
            private = PrivatePaymentMaterial.model_validate(
                requirement["privatePaymentMaterial"]
            )
            raw = self.bridge.attach(
                owner=_owner_wire(owner),
                approved_plan={
                    "planId": plan.plan_id,
                    "planVersion": plan.plan_version,
                    "planDigest": plan.plan_digest,
                    "approvalId": approval.approval_id,
                },
                step={
                    "stepId": selected_step.step_id,
                    "canonicalAgentId": selected_step.selected_agent.canonical_agent_id,
                    "agentCardDigest": selected_step.selected_agent.card_digest,
                    "rpcEndpoint": selected_step.selected_agent.rpc_endpoint,
                    "skillId": selected_step.selected_agent.a2a_skill_id,
                },
                remote_task={
                    "taskId": remote.task_id,
                    "contextId": remote.context_id,
                    "state": remote.state,
                },
                requirement={
                    "schemaVersion": "payment-requirement-snapshot/1",
                    "taskId": remote.task_id,
                    "contextId": remote.context_id,
                    "orderId": public_requirement.order_id,
                    "quoteId": public_requirement.quote_id,
                    "paymentRequired": public_requirement.payment_required,
                    "requirementDigest": public_requirement.requirement_digest,
                    "checkoutJwt": private.checkout_jwt,
                    "checkoutHash": private.checkout_hash,
                    "amountMinor": public_requirement.amount_minor,
                    "currency": public_requirement.currency,
                    "payee": public_requirement.payee,
                    "profileId": public_requirement.profile_id,
                    "extensionUri": public_requirement.extension_uri,
                    "checkoutAudience": public_requirement.checkout_audience,
                    "checkoutNonce": public_requirement.checkout_nonce,
                    "paymentAudience": public_requirement.payment_audience,
                    "paymentNonce": public_requirement.payment_nonce,
                    "expiresAt": public_requirement.expires_at.isoformat().replace(
                        "+00:00", "Z"
                    ),
                },
            )
        except DomainError as error:
            raise _translate_domain_error(error) from error
        return BridgeAttachment(
            continuationId=raw.continuation_id,
            paymentWorkflowId=raw.payment_workflow_id,
            version=raw.version,
        )

    def approve(
        self,
        *,
        owner: Any,
        continuation_id: str,
        expected_version: int,
        approval_text: str,
        expected_approval_target_digest: str,
    ) -> BridgeApprovalResult:
        row = self.bridge._get(continuation_id)
        requirement = row["requirement"]
        target = build_payment_approval_target(
            plan_id=row["plan_id"],
            plan_version=row["plan_version"],
            plan_digest=row["plan_digest"],
            step_id=row["step_id"],
            task_id=row["task_id"],
            context_id=row["task_context_id"],
            order_id=row["order_id"],
            quote_id=row["quote_id"],
            merchant=row["payee"],
            amount_minor=row["amount_minor"],
            currency=row["currency"],
            profile_id=row["profile_id"],
            expires_at=datetime.fromisoformat(
                requirement["expiresAt"].replace("Z", "+00:00")
            ),
            payment_required=requirement["paymentRequired"],
            requirement_digest=row["requirement_digest"],
            checkout_digest=canonical_digest(requirement["checkoutJwt"]),
        )
        if canonical_digest(target) != expected_approval_target_digest:
            raise SecurityBlocked(
                "PAYMENT_APPROVAL_TARGET_MISMATCH",
                "The payment terms changed after they were displayed.",
            )
        try:
            raw = self.bridge.approve(
                continuation_id,
                expected_version,
                approval_text,
                owner=_owner_wire(owner),
            )
        except DomainError as error:
            raise _translate_domain_error(error) from error
        persisted = self.bridge._approval(continuation_id)
        if persisted["display_digest"] != target.bridge_display_digest:
            raise SecurityBlocked(
                "PAYMENT_DISPLAY_DIGEST_MISMATCH",
                "The signed payment display differs from the displayed terms.",
            )
        return BridgeApprovalResult(
            continuationId=raw.continuation_id,
            version=raw.version,
            approvalDigest=raw.approval_digest,
            state="PaymentApproved",
        )

    async def execute_approved_payment(
        self,
        *,
        operation_id: str,
        continuation_id: str,
        expected_version: int,
    ) -> BridgeExecutionResult:
        row = self.bridge._get(continuation_id)
        sync_executor = _SynchronousPaymentExecutor(
            self.executor, workflow_id=str(row["mediation_session_id"])
        )
        try:
            raw = await asyncio.to_thread(
                self.bridge.execute_approved_payment,
                operation_id,
                continuation_id,
                expected_version,
                sync_executor,
            )
        except DomainError as error:
            if getattr(error, "code", None) == "REFUND_REQUIRED":
                failed = self.bridge._get(continuation_id)
                remote = RemoteTaskSnapshot(
                    taskId=failed["task_id"],
                    contextId=failed["task_context_id"],
                    state="failed",
                    taskDigest=canonical_digest(
                        {
                            "continuationId": continuation_id,
                            "state": "refund-required",
                            "lastErrorCode": failed.get("last_error_code"),
                        }
                    ),
                    orderId=failed["order_id"],
                    quoteId=failed["quote_id"],
                )
                return BridgeExecutionResult(
                    continuationId=continuation_id,
                    version=failed["version"],
                    remoteTask=remote,
                    result={
                        "taskState": "failed",
                        "refundEligible": True,
                        "simulation": True,
                        "conformance": "NOT CONFORMANT",
                    },
                    state="refund-required",
                    a2aExecutions=tuple(
                        BridgeA2AExecutionSummary(
                            operationId=execution.operation.operation_id,
                            taskDigest=execution.response.task.task_digest,
                            eventOrder=execution.event_order,
                        )
                        for execution in sync_executor.executions
                    ),
                )
            raise _translate_domain_error(error) from error
        remote = sync_executor.last_response
        if remote is None:
            raise ReviewRequired(
                "PAYMENT_RESULT_UNKNOWN", "The payment Task response is unavailable."
            )
        return BridgeExecutionResult(
            continuationId=raw.continuation_id,
            version=raw.version,
            remoteTask=remote,
            result={
                "taskState": remote.state,
                "taskDigest": remote.task_digest,
                "artifact": remote.artifact,
                "guaranteeDigest": raw.guarantee_digest,
                "settlementReceiptDigest": raw.settlement_receipt_digest,
                "resultDigest": raw.result_digest,
                "refundEligible": False,
                "simulation": True,
                "conformance": "NOT CONFORMANT",
            },
            state=(
                "same-task-completed"
                if str(raw.state) in {"completed", "BridgeState.COMPLETED"}
                else "review-required"
            ),
            a2aExecutions=tuple(
                BridgeA2AExecutionSummary(
                    operationId=execution.operation.operation_id,
                    taskDigest=execution.response.task.task_digest,
                    eventOrder=execution.event_order,
                )
                for execution in sync_executor.executions
            ),
        )

    def refund(
        self,
        *,
        owner: Any,
        operation_id: str,
        continuation_id: str,
        expected_version: int,
    ) -> RefundResult:
        try:
            self.bridge.status(continuation_id, owner=_owner_wire(owner))
            raw = self.bridge.refund(
                operation_id, continuation_id, expected_version
            )
        except DomainError as error:
            raise _translate_domain_error(error) from error
        return RefundResult(
            refundId=raw.refund_id,
            state=raw.state,
            resultDigest=raw.result_digest,
        )
