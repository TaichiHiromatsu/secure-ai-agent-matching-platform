"""Authoritative single-workflow controller for the approved simulation release."""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any, Callable

from secure_mediation_agent.ap2.credential_provider import CredentialProvider
from secure_mediation_agent.ap2.keys import DemoKeySet, public_key
from secure_mediation_agent.ap2.mpp import MerchantPaymentProcessor
from secure_mediation_agent.ap2.receipts import Ap2ReceiptFactory
from secure_mediation_agent.ap2.trusted_surface import MandatePresentations, TrustedSurface
from secure_mediation_agent.ap2.verification import b64url_sha256, closed_reference
from secure_mediation_agent.merchant.service import PaidBookingMerchant
from secure_mediation_agent.payment_profiles.a2a import payment_message
from secure_mediation_agent.payment_profiles.registry import ProfileRegistry

from .approval import ApprovalAction, AuthorizationService, dispatch
from .canonical import canonical_bytes, canonical_digest, canonical_json, sha256_digest
from .errors import DomainError
from .failpoints import crash_after
from .models import (
    AllowedPayment,
    Merchant,
    MessagePart,
    PlanSnapshot,
    PlanStep,
    PublicWorkflowView,
    SelectedAgent,
    WorkflowRequest,
    WorkflowState,
)
from .repository import WorkflowRepository
from .views import build_view


_OUTBOX_FAST_PATH_WAIT_SECONDS = 2.0
_OUTBOX_FAST_PATH_POLL_SECONDS = 0.05


def _id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex}"


def _stable_id(prefix: str, *parts: str) -> str:
    value = "/".join(parts)
    return f"{prefix}:{uuid.uuid5(uuid.NAMESPACE_URL, value).hex}"


def _now() -> datetime:
    return datetime.now(UTC)


def _serialized_mutation(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        # Accepted deployment has one workflow API process. SQLite still owns
        # durable CAS; this lock prevents same-process approval interleaving
        # between evidence commit and aggregate transition.
        with self._mutation_lock:
            return method(self, *args, **kwargs)

    return locked


@dataclass(frozen=True, slots=True)
class Identity:
    tenant_id: str
    customer_id: str


class WorkflowController:
    def __init__(
        self,
        repository: WorkflowRepository,
        keys: DemoKeySet,
        *,
        clock: Callable[[], datetime] = _now,
        rail_fault: str | None = None,
        commit_fault: bool = False,
        merchant: Any | None = None,
    ) -> None:
        self.repository = repository
        self.keys = keys
        self.clock = clock
        self.rail_fault = rail_fault
        self.commit_fault = commit_fault
        self.worker_id = f"workflow-api:{os.getpid()}"
        self._mutation_lock = threading.RLock()
        self.profile = ProfileRegistry.load(
            "x402-wire-simulation/1", simulation_key=keys.simulation_signer
        )
        self.authorization = AuthorizationService(keys.plan_authority)
        self.merchant = merchant or PaidBookingMerchant(repository, keys, self.profile)
        self.trusted_surface = TrustedSurface(keys)
        self.credential_provider = CredentialProvider(keys)
        self.mpp = MerchantPaymentProcessor(self.credential_provider, self.profile)

    @_serialized_mutation
    def create(
        self,
        request: WorkflowRequest,
        *,
        identity: Identity,
        session_id: str,
        context_id: str,
        idempotency_key: str,
    ) -> PublicWorkflowView:
        request_wire = request.model_dump(mode="json", by_alias=True)
        request_digest = canonical_digest(request_wire)
        saved = self.repository.begin_idempotency(
            tenant_id=identity.tenant_id,
            actor_id=identity.customer_id,
            operation="workflow:create",
            key=idempotency_key,
            request_hash=request_digest,
        )
        if saved and saved.get("_idempotencyStatus") == "processing":
            raise DomainError(
                "STATE_TRANSITION_CONFLICT",
                "An identical workflow creation is already processing.",
                idempotency_key,
            )
        if saved and saved.get("workflowId"):
            return self.get(saved["workflowId"], identity=identity)
        active = self.repository.active_workflow(identity.tenant_id, session_id, context_id)
        if active is not None:
            if active["request_digest"] != request_digest:
                self.repository.abandon_idempotency(
                    tenant_id=identity.tenant_id,
                    actor_id=identity.customer_id,
                    operation="workflow:create",
                    key=idempotency_key,
                    request_hash=request_digest,
                )
                raise DomainError(
                    "STATE_TRANSITION_CONFLICT",
                    "This session already has an active workflow.",
                    active["workflow_id"],
                    current_state=active["state"],
                )
            view = self.get(active["workflow_id"], identity=identity)
            self.repository.complete_idempotency(
                tenant_id=identity.tenant_id,
                actor_id=identity.customer_id,
                operation="workflow:create",
                key=idempotency_key,
                request_hash=request_digest,
                response=view.model_dump(mode="json", by_alias=True, exclude_none=True),
                result_id=active["workflow_id"],
            )
            return view
        workflow_id = _id("workflow")
        plan_id = _id("plan")
        created_at = self.clock()
        expires_at = created_at + timedelta(minutes=15)
        card_wire = self.merchant.agent_card().model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
        plan = PlanSnapshot(
            schemaVersion="secure-mediation-plan/1",
            canonicalization="RFC8785",
            planId=plan_id,
            planVersion=1,
            tenantId=identity.tenant_id,
            customerId=identity.customer_id,
            sessionId=session_id,
            contextId=context_id,
            requestDigest=request_digest,
            request={"goal": request.goal, "constraints": {}},
            selectedAgent=SelectedAgent(
                agentId="paid-booking-agent",
                agentCardDigest=canonical_digest(card_wire),
                endpoint="http://127.0.0.1:8005/a2a",
                onboardingVersion="simulation-v1",
                trustKeySetVersion="demo-es256-v1",
            ),
            merchant=Merchant(),
            skillId="paid-booking",
            productId="demo-paid-booking",
            quantity=1,
            steps=(
                PlanStep(
                    stepId="step-1",
                    agentId="paid-booking-agent",
                    skillId="paid-booking",
                    paymentRequired=request.payment_required,
                    inputDigest=canonical_digest({"goal": request.goal}),
                ),
            ),
            maximumCustomerTotal=request.maximum_customer_total,
            currency="USD",
            decimals=2,
            feePolicyVersion="zero-fee-v1",
            allowedPayment=AllowedPayment(
                profile=self.profile.profile_id,
                extensionUri=self.profile.extension_uri,
                schemes=("exact-simulated",),
                networks=("demo:local",),
                assets=("USD",),
                railMode="simulated",
            ),
            fulfillmentConstraints={},
            createdAt=created_at.isoformat().replace("+00:00", "Z"),
            expiresAt=expires_at.isoformat().replace("+00:00", "Z"),
        )
        plan_bytes = canonical_bytes(plan)
        plan_digest = sha256_digest(plan_bytes)
        plan_evidence_id = _id("evidence:plan")
        self.repository.put_evidence(
            workflow_id=workflow_id,
            evidence_id=plan_evidence_id,
            tenant_id=identity.tenant_id,
            kind="plan-snapshot",
            exact_bytes=plan_bytes,
            kid=None,
            media_type="application/jcs+json",
            profile_id=self.profile.profile_id,
        )
        workflow = self.repository.create_workflow(
            workflow_id=workflow_id,
            tenant_id=identity.tenant_id,
            customer_id=identity.customer_id,
            session_id=session_id,
            context_id=context_id,
            request=request_wire,
            request_digest=request_digest,
            plan=plan,
            plan_digest=plan_digest,
            plan_evidence_id=plan_evidence_id,
        )
        view = build_view(workflow, plan=plan)
        response = view.model_dump(mode="json", by_alias=True, exclude_none=True)
        self.repository.complete_idempotency(
            tenant_id=identity.tenant_id,
            actor_id=identity.customer_id,
            operation="workflow:create",
            key=idempotency_key,
            request_hash=request_digest,
            response=response,
            result_id=workflow_id,
        )
        return view

    def get(self, workflow_id: str, *, identity: Identity) -> PublicWorkflowView:
        workflow = self.repository.get_workflow(workflow_id)
        self._require_owner(workflow, identity)
        if (
            workflow["state"] == WorkflowState.PAYMENT_APPROVAL_REQUIRED
            and workflow.get("merchant_task_id")
        ):
            workflow["payment_expires_at"] = self.repository.merchant_task(
                workflow["merchant_task_id"]
            )["requirement"]["expires_at"]
        plan = PlanSnapshot.model_validate_json(self.repository.get_plan_bytes(workflow))
        receipts = self.repository.profile_receipts(workflow["merchant_task_id"]) if workflow.get("merchant_task_id") else []
        return build_view(
            workflow,
            plan=plan,
            artifacts=self.repository.artifact_refs(workflow_id),
            receipts=receipts,
        )

    def active(
        self,
        *,
        identity: Identity,
        session_id: str,
        context_id: str,
    ) -> PublicWorkflowView | None:
        workflow = self.repository.active_workflow(identity.tenant_id, session_id, context_id)
        return None if workflow is None else self.get(workflow["workflow_id"], identity=identity)

    @_serialized_mutation
    def execute_required_refund(
        self,
        workflow_id: str,
        *,
        operator_id: str,
        idempotency_key: str,
        outcome: str = "settled",
    ) -> PublicWorkflowView:
        """Internal operator compensation; never exposed by the public API/nginx."""

        if operator_id != "demo-operator":
            raise DomainError(
                "TENANT_BINDING_MISMATCH", "Authorized demo operator is required.", workflow_id
            )
        workflow = self.repository.get_workflow(workflow_id)
        if workflow["state"] != WorkflowState.REFUND_REQUIRED:
            raise DomainError(
                "STATE_TRANSITION_CONFLICT",
                "Workflow is not awaiting a refund.",
                workflow_id,
                current_state=workflow["state"],
            )
        result = self.repository.refund_simulation(
            workflow_id=workflow_id,
            idempotency_key=idempotency_key,
            outcome=outcome,
        )
        if result["state"] == "settled":
            self.repository.transition(
                workflow_id,
                expected_state=WorkflowState.REFUND_REQUIRED,
                to_state=WorkflowState.REFUNDED,
                actor_id=operator_id,
                actor_role="operator",
                operation="refund-success",
                related_digest=result["request_digest"],
            )
        elif result["state"] == "unknown":
            self.repository.transition(
                workflow_id,
                expected_state=WorkflowState.REFUND_REQUIRED,
                to_state=WorkflowState.RECONCILIATION_REQUIRED,
                actor_id=operator_id,
                actor_role="operator",
                operation="refund-unknown",
                error_code="RECONCILIATION_REQUIRED",
                updates={"last_error_code": "REFUND_OUTCOME_UNKNOWN"},
            )
        return self.get(
            workflow_id,
            identity=Identity(workflow["tenant_id"], workflow["customer_id"]),
        )

    @_serialized_mutation
    def reconcile_unknown(
        self,
        workflow_id: str,
        *,
        operator_id: str,
        idempotency_key: str,
        observed_state: str,
    ) -> PublicWorkflowView:
        """Resolve one saved simulation external ID without creating a new charge."""

        if operator_id != "demo-operator":
            raise DomainError(
                "TENANT_BINDING_MISMATCH", "Authorized demo operator is required.", workflow_id
            )
        workflow = self.repository.get_workflow(workflow_id)
        if workflow["state"] != WorkflowState.RECONCILIATION_REQUIRED:
            raise DomainError(
                "STATE_TRANSITION_CONFLICT",
                "Workflow is not awaiting reconciliation.",
                workflow_id,
                current_state=workflow["state"],
            )
        operation = self.repository.unknown_rail_operation(workflow_id)
        observation = {
            "schemaVersion": "secure-simulation-observation/1",
            "workflowId": workflow_id,
            "kind": operation["kind"],
            "externalId": operation["external_id"],
            "observedState": observed_state,
            "authoritativeSource": "local-simulation-ledger",
            "operatorId": operator_id,
            "observedAt": self.clock().isoformat().replace("+00:00", "Z"),
        }
        evidence_id = _id("evidence:reconciliation")
        evidence_digest = self.repository.put_evidence(
            workflow_id=workflow_id,
            evidence_id=evidence_id,
            tenant_id=workflow["tenant_id"],
            kind="simulation-reconciliation-observation",
            exact_bytes=canonical_bytes(observation),
            kid=None,
            media_type="application/json",
            profile_id=self.profile.profile_id,
        )
        result = self.repository.reconcile_simulation_operation(
            workflow_id=workflow_id,
            operator_id=operator_id,
            idempotency_key=idempotency_key,
            observed_state=observed_state,
            evidence_id=evidence_id,
            evidence_digest=evidence_digest,
        )
        if result["kind"] == "refund":
            target = (
                WorkflowState.REFUNDED
                if result["state"] == "settled"
                else WorkflowState.REFUND_REQUIRED
            )
            self.repository.transition(
                workflow_id,
                expected_state=WorkflowState.RECONCILIATION_REQUIRED,
                to_state=target,
                actor_id=operator_id,
                actor_role="operator",
                operation=f"refund-reconciled-{result['state']}",
                related_digest=evidence_digest,
            )
        else:
            artifacts = {item["kind"]: item for item in self.repository.artifact_refs(workflow_id)}
            payment_reference = artifacts["payment-mandate"]["reference_digest"]
            payment_receipt = Ap2ReceiptFactory.payment(
                key=self.keys.mpp,
                reference=payment_reference,
                issued_at=int(self.clock().timestamp()),
                payment_id=f"payment:{result['operation_id']}",
                simulation_reference=result["external_id"],
                success=result["state"] == "settled",
                error="settlement_failed",
                error_description="Definitive reconciliation reported settlement failure.",
            )
            identity = Identity(workflow["tenant_id"], workflow["customer_id"])
            self._save_artifact(
                workflow,
                identity,
                kind="payment-receipt",
                exact=payment_receipt,
                issuer="demo-mpp",
                kid=self.keys.mpp.get("kid"),
                reference=payment_reference,
            )
            if result["state"] == "settled":
                self.repository.require_refund(
                    refund_id=_id("refund"),
                    workflow_id=workflow_id,
                    attempt_id=result["operation_id"],
                    original_payment_id=f"payment:{result['operation_id']}",
                    reason="late-settlement-reconciliation",
                    idempotency_key=f"late-refund:{result['operation_id']}",
                )
                target = WorkflowState.REFUND_REQUIRED
            else:
                target = WorkflowState.PAYMENT_FAILED
            self.repository.transition(
                workflow_id,
                expected_state=WorkflowState.RECONCILIATION_REQUIRED,
                to_state=target,
                actor_id=operator_id,
                actor_role="operator",
                operation=f"settlement-reconciled-{result['state']}",
                related_digest=evidence_digest,
            )
        return self.get(
            workflow_id,
            identity=Identity(workflow["tenant_id"], workflow["customer_id"]),
        )

    @_serialized_mutation
    def message(
        self,
        workflow_id: str,
        parts: list[MessagePart],
        *,
        identity: Identity,
        message_id: str,
        idempotency_key: str,
        expected_version: int | None = None,
    ) -> PublicWorkflowView:
        workflow = self.repository.get_workflow(workflow_id)
        self._require_owner(workflow, identity)
        request_hash = canonical_digest(
            {
                "workflowId": workflow_id,
                "messageId": message_id,
                "parts": [part.model_dump(mode="json") for part in parts],
            }
        )
        saved = self.repository.begin_idempotency(
            tenant_id=identity.tenant_id,
            actor_id=identity.customer_id,
            operation="workflow:message",
            key=idempotency_key,
            request_hash=request_hash,
        )
        if saved and saved.get("_idempotencyStatus") == "processing":
            raise DomainError(
                "STATE_TRANSITION_CONFLICT",
                "An identical workflow message is already processing.",
                workflow_id,
                current_state=workflow["state"],
            )
        if saved and saved.get("workflowId"):
            return self.get(workflow_id, identity=identity)
        try:
            action = dispatch(parts, workflow["state"])
            if action == ApprovalAction.APPROVE_PLAN:
                self._approve_plan(workflow, identity, message_id, expected_version)
            elif action == ApprovalAction.APPROVE_PAYMENT:
                self._approve_payment(workflow, identity, message_id, expected_version)
            elif action == ApprovalAction.REJECT_CURRENT:
                self._reject(workflow, identity, message_id)
            else:
                raise DomainError(
                    "APPROVAL_NOT_PENDING",
                    "This deterministic release accepts only workflow approval messages.",
                    workflow_id,
                    current_state=workflow["state"],
                )
        except Exception as error:
            self.repository.abandon_idempotency(
                tenant_id=identity.tenant_id,
                actor_id=identity.customer_id,
                operation="workflow:message",
                key=idempotency_key,
                request_hash=request_hash,
            )
            if isinstance(error, DomainError):
                error.correlation_id = workflow_id
            raise
        view = self.get(workflow_id, identity=identity)
        self.repository.complete_idempotency(
            tenant_id=identity.tenant_id,
            actor_id=identity.customer_id,
            operation="workflow:message",
            key=idempotency_key,
            request_hash=request_hash,
            response=view.model_dump(mode="json", by_alias=True, exclude_none=True),
            result_id=workflow_id,
        )
        return view

    def _approve_plan(
        self,
        workflow: dict[str, Any],
        identity: Identity,
        message_id: str,
        expected_version: int | None,
    ) -> None:
        now = self.clock()
        if expected_version is not None and workflow["version"] != expected_version:
            raise DomainError(
                "STATE_TRANSITION_CONFLICT",
                "Workflow state changed concurrently.",
                workflow["workflow_id"],
                current_state=workflow["state"],
            )
        plan = PlanSnapshot.model_validate_json(self.repository.get_plan_bytes(workflow))
        if now > datetime.fromisoformat(plan.expires_at.replace("Z", "+00:00")):
            raise DomainError(
                "PLAN_APPROVAL_EXPIRED",
                "Plan approval window expired; a new plan is required.",
                workflow["workflow_id"],
                current_state=workflow["state"],
            )
        issued = int(now.timestamp())
        approval_id = _id("approval:plan")
        nonce = secrets.token_urlsafe(32)
        expires = issued + 600
        token = self.authorization.issue_plan_authorization(
            {
                "jti": approval_id,
                "aud": "secure-mediation-workflow",
                "intent": "approve-plan",
                "tenantId": identity.tenant_id,
                "customerId": identity.customer_id,
                "sessionId": workflow["session_id"],
                "contextId": workflow["context_id"],
                "workflowId": workflow["workflow_id"],
                "planId": workflow["active_plan_id"],
                "planVersion": 1,
                "planDigest": workflow["plan_digest"],
                "nonce": nonce,
                "iat": issued,
                "exp": expires,
            }
        )
        self.authorization.verify(
            token,
            expected_type="secure-plan-authorization+jwt",
            audience="secure-mediation-workflow",
            now=issued,
        )
        evidence_id = _id("evidence:plan-authorization")
        digest = self.repository.put_evidence(
            workflow_id=workflow["workflow_id"],
            evidence_id=evidence_id,
            tenant_id=identity.tenant_id,
            kind="plan-authorization",
            exact_bytes=token,
            kid=self.keys.plan_authority.get("kid"),
            media_type="application/jwt",
            profile_id="secure-plan-authorization/1",
        )
        self.repository.record_plan_approval(
            workflow_id=workflow["workflow_id"],
            approval_id=approval_id,
            nonce=nonce,
            authorization_evidence_id=evidence_id,
            authorization_digest=digest,
            approved_at=now.isoformat().replace("+00:00", "Z"),
            expires_at=datetime.fromtimestamp(expires, UTC).isoformat().replace("+00:00", "Z"),
        )
        self.repository.transition(
            workflow["workflow_id"],
            expected_state=WorkflowState.PLAN_APPROVAL_REQUIRED,
            to_state=WorkflowState.PLAN_APPROVED,
            expected_version=expected_version,
            actor_id=identity.customer_id,
            actor_role="customer",
            operation="approve-plan",
            approval_intent="approve-plan",
            related_digest=digest,
        )
        if workflow["request"]["paymentRequired"]:
            self._start_paid_task(workflow, identity, approval_id, issued)
        else:
            self.repository.transition(
                workflow["workflow_id"],
                expected_state=WorkflowState.PLAN_APPROVED,
                to_state=WorkflowState.FREE_EXECUTING,
                actor_id="secure-mediator",
                actor_role="shopping-agent",
                operation="free-execution-start",
            )
            current = self.repository.get_workflow(workflow["workflow_id"])
            current = self.repository.transition(
                workflow["workflow_id"],
                expected_state=WorkflowState.FREE_EXECUTING,
                to_state=WorkflowState.FINAL_VALIDATING,
                actor_id="secure-mediator",
                actor_role="shopping-agent",
                operation="free-execution-complete",
            )
            self.repository.transition(
                workflow["workflow_id"],
                expected_state=WorkflowState.FINAL_VALIDATING,
                to_state=WorkflowState.COMPLETED,
                actor_id="secure-mediator",
                actor_role="shopping-agent",
                operation="free-final-validation",
            )

    def _start_paid_task(
        self,
        workflow: dict[str, Any],
        identity: Identity,
        approval_id: str,
        issued_at: int,
    ) -> None:
        task_id = _stable_id("task", workflow["workflow_id"], "merchant-task:start")
        order_id = _stable_id("order", workflow["workflow_id"], "merchant-task:start")
        capability_id, capability_token = self._issue_capability(
            workflow=workflow,
            approval_id=approval_id,
            audience="merchant:demo-merchant",
            operation="merchant-task:start",
            task_id=task_id,
            order_id=order_id,
            issued_at=issued_at,
        )
        start_request = {
            "workflowId": workflow["workflow_id"],
            "planDigest": workflow["plan_digest"],
            "taskId": task_id,
            "orderId": order_id,
            "activation": self.profile.extension_uri,
            "capabilityDigest": sha256_digest(capability_token),
            "capabilityId": capability_id,
            "approvalId": approval_id,
            "tenantId": identity.tenant_id,
            "customerId": identity.customer_id,
            "contextId": workflow["context_id"],
            "issuedAt": issued_at,
            "expiresAt": issued_at + 600,
        }
        operation_id = f"start:{task_id}"
        self.repository.transition(
            workflow["workflow_id"],
            expected_state=WorkflowState.PLAN_APPROVED,
            to_state=WorkflowState.MERCHANT_TASK_STARTING,
            actor_id="secure-mediator",
            actor_role="shopping-agent",
            operation="merchant-task:start",
            updates={"merchant_task_id": task_id, "order_id": order_id},
            outbox=("merchant-task:start", operation_id, start_request),
        )
        self._run_outbox_operation(operation_id)

    def _run_outbox_operation(self, operation_id: str) -> None:
        """Use the durable worker contract even for the API fast path."""

        deadline = time.monotonic() + _OUTBOX_FAST_PATH_WAIT_SECONDS
        row = None
        while row is None:
            row = self.repository.lease_outbox(
                self.worker_id, operation_id=operation_id, lease_seconds=120
            )
            if row is not None:
                break
            existing = self.repository.outbox_row(operation_id)
            if existing and existing["status"] == "done":
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # A just-enqueued row can be momentarily not due if wall time moves
            # backwards.  An active lease can also complete in another worker.
            # Retry with a bounded sleep; lease_outbox remains the sole CAS and
            # therefore never steals an unexpired competing lease.
            time.sleep(min(_OUTBOX_FAST_PATH_POLL_SECONDS, remaining))
        if row is None:
            raise DomainError(
                "OUTBOX_LEASE_UNAVAILABLE",
                "Durable operation is already being recovered.",
                operation_id,
            )
        try:
            self.execute_outbox(row)
        except BaseException as error:
            self.repository.retry_outbox(
                row["outbox_id"],
                self.worker_id,
                error_code=type(error).__name__,
                delay_seconds=0,
            )
            raise
        self.repository.complete_outbox(row["outbox_id"], self.worker_id)

    @_serialized_mutation
    def process_leased_outbox(self, row: dict[str, Any], worker_id: str) -> None:
        """Execute and acknowledge a row leased by the supervised worker."""

        try:
            self.execute_outbox(row)
        except BaseException as error:
            self.repository.retry_outbox(
                row["outbox_id"],
                worker_id,
                error_code=type(error).__name__,
            )
            raise
        self.repository.complete_outbox(row["outbox_id"], worker_id)

    def execute_outbox(self, row: dict[str, Any]) -> None:
        if row["event_type"] == "merchant-task:start":
            self._execute_merchant_start(row["payload"])
            return
        if row["event_type"] == "trusted-surface:issue":
            self._execute_payment_authorization(row["payload"])
            return
        raise DomainError(
            "OUTBOX_EVENT_UNSUPPORTED",
            "Durable operation type is unsupported.",
            row["event_type"],
        )

    @_serialized_mutation
    def recover_workflow(self, workflow: dict[str, Any]) -> None:
        """Resume transient states whose continuation died before enqueue/transition."""

        current = self.repository.get_workflow(workflow["workflow_id"])
        state = WorkflowState(current["state"])
        if state == WorkflowState.PLAN_APPROVED:
            approval = self.repository.plan_approval(current["workflow_id"])
            issued_at = int(
                datetime.fromisoformat(
                    approval["approved_at"].replace("Z", "+00:00")
                ).timestamp()
            )
            if current["request"]["paymentRequired"]:
                self._start_paid_task(
                    current,
                    Identity(current["tenant_id"], current["customer_id"]),
                    approval["approval_id"],
                    issued_at,
                )
            else:
                self.repository.transition(
                    current["workflow_id"],
                    expected_state=WorkflowState.PLAN_APPROVED,
                    to_state=WorkflowState.FREE_EXECUTING,
                    actor_id="secure-mediator",
                    actor_role="shopping-agent",
                    operation="free-execution-start-recovery",
                )
            return
        if state == WorkflowState.FREE_EXECUTING:
            self.repository.transition(
                current["workflow_id"],
                expected_state=WorkflowState.FREE_EXECUTING,
                to_state=WorkflowState.FINAL_VALIDATING,
                actor_id="secure-mediator",
                actor_role="shopping-agent",
                operation="free-execution-complete-recovery",
            )
            return
        if state == WorkflowState.FINAL_VALIDATING:
            self.repository.transition(
                current["workflow_id"],
                expected_state=WorkflowState.FINAL_VALIDATING,
                to_state=WorkflowState.COMPLETED,
                actor_id="secure-mediator",
                actor_role="shopping-agent",
                operation="free-final-validation-recovery",
            )

    def _execute_merchant_start(self, start_request: dict[str, Any]) -> None:
        workflow = self.repository.get_workflow(start_request["workflowId"])
        if workflow["state"] != WorkflowState.MERCHANT_TASK_STARTING:
            if workflow["state"] in {
                WorkflowState.PAYMENT_APPROVAL_REQUIRED,
                WorkflowState.PAYMENT_AUTHORIZING,
                WorkflowState.PAYMENT_APPROVED,
                WorkflowState.PAYMENT_SUBMITTED,
                WorkflowState.PAYMENT_VERIFYING,
                WorkflowState.FULFILLMENT_PREPARING,
                WorkflowState.PAYMENT_SETTLING,
                WorkflowState.FULFILLMENT_COMMITTING,
                WorkflowState.COMPLETED,
                WorkflowState.REPLAN_REQUIRED,
                WorkflowState.PAYMENT_FAILED,
                WorkflowState.RECONCILIATION_REQUIRED,
                WorkflowState.REFUND_REQUIRED,
                WorkflowState.REFUNDED,
                WorkflowState.CANCELLED,
                WorkflowState.EXPIRED,
            }:
                return
            raise DomainError(
                "STATE_TRANSITION_CONFLICT",
                "Merchant start cannot run from the current state.",
                workflow["workflow_id"],
                current_state=workflow["state"],
            )
        identity = Identity(start_request["tenantId"], start_request["customerId"])
        capability = self.repository.capability_for_operation(
            workflow["workflow_id"], "merchant-task:start"
        )
        if capability is None:
            raise DomainError(
                "CAPABILITY_MISSING", "Merchant capability evidence is missing.", workflow["workflow_id"]
            )
        capability_row, capability_token = capability
        request_for_digest = {
            key: value
            for key, value in start_request.items()
            if key
            in {
                "workflowId",
                "planDigest",
                "taskId",
                "orderId",
                "activation",
                "capabilityDigest",
            }
        }
        self.repository.consume_capability(
            capability_row["capability_id"], canonical_digest(request_for_digest)
        )
        result = self.merchant.start_task(
            workflow_id=workflow["workflow_id"],
            plan_digest=workflow["plan_digest"],
            task_id=start_request["taskId"],
            order_id=start_request["orderId"],
            context_id=workflow["context_id"],
            capability_id=capability_row["capability_id"],
            activation={self.profile.extension_uri},
            issued_at=int(start_request["issuedAt"]),
            expires_at=int(start_request["expiresAt"]),
            capability_token=capability_token,
        )
        crash_after("external:merchant-start-returned")
        if result.activation_echo != self.profile.extension_uri:
            self.repository.transition(
                workflow["workflow_id"],
                expected_state=WorkflowState.MERCHANT_TASK_STARTING,
                to_state=WorkflowState.REPLAN_REQUIRED,
                actor_id="secure-mediator",
                actor_role="shopping-agent",
                operation="merchant-activation-drift",
                error_code="X402_ACTIVATION_MISMATCH",
            )
            return
        expected_requirements = self.profile.build_required(amount=1250)
        observed_payment_constraints = {
            key: result.requirements.get(key) for key in expected_requirements
        }
        expected_quote_id = f"quote:{start_request['orderId']}"
        expected_expiry = (
            datetime.fromtimestamp(int(start_request["expiresAt"]), UTC)
            .isoformat()
            .replace("+00:00", "Z")
        )
        if (
            observed_payment_constraints != expected_requirements
            or result.requirements.get("orderId") != start_request["orderId"]
            or result.requirements.get("quoteId") != expected_quote_id
            or result.requirements.get("expiresAt") != expected_expiry
        ):
            self.repository.transition(
                workflow["workflow_id"],
                expected_state=WorkflowState.MERCHANT_TASK_STARTING,
                to_state=WorkflowState.REPLAN_REQUIRED,
                actor_id="secure-mediator",
                actor_role="shopping-agent",
                operation="merchant-constraint-drift",
                error_code="PAYMENT_CONSTRAINT_DRIFT",
            )
            return
        self.merchant.verify_checkout(
            result.checkout_jwt,
            workflow_id=workflow["workflow_id"],
            plan_digest=workflow["plan_digest"],
            task_id=start_request["taskId"],
            capability_token=capability_token,
        )
        task_wire = result.task.model_dump(mode="json", by_alias=True, exclude_none=True)
        task_bytes = canonical_bytes(task_wire)
        task_evidence_id = _stable_id(
            "evidence:merchant-task", workflow["workflow_id"], "merchant-task"
        )
        task_digest = self.repository.put_evidence(
            workflow_id=workflow["workflow_id"],
            evidence_id=task_evidence_id,
            tenant_id=identity.tenant_id,
            kind="merchant-task",
            exact_bytes=task_bytes,
            kid=None,
            media_type="application/a2a+json",
            profile_id=self.profile.profile_id,
        )
        requirements_evidence_id = _stable_id(
            "evidence:payment-requirements", workflow["workflow_id"], "payment-requirements"
        )
        requirements_bytes = canonical_bytes(result.requirements)
        requirements_digest = self.repository.put_evidence(
            workflow_id=workflow["workflow_id"],
            evidence_id=requirements_evidence_id,
            tenant_id=identity.tenant_id,
            kind="payment-requirements",
            exact_bytes=requirements_bytes,
            kid=None,
            media_type="application/json",
            profile_id=self.profile.profile_id,
        )
        self.repository.save_merchant_task(
            workflow_id=workflow["workflow_id"],
            task_id=start_request["taskId"],
            context_id=workflow["context_id"],
            order_id=start_request["orderId"],
            task_json=canonical_json(task_wire),
            task_digest=task_digest,
            task_evidence_id=task_evidence_id,
            requirements_id=_stable_id(
                "requirements", workflow["workflow_id"], "payment-requirements"
            ),
            requirements_json=canonical_json(result.requirements),
            requirements_digest=requirements_digest,
            requirements_evidence_id=requirements_evidence_id,
            checkout_jwt=result.checkout_jwt,
            checkout_hash=result.checkout_hash,
            capability_id=capability_row["capability_id"],
            expires_at=datetime.fromtimestamp(int(start_request["expiresAt"]), UTC).isoformat().replace("+00:00", "Z"),
            agent_card_digest=canonical_digest(
                self.merchant.agent_card().model_dump(mode="json", by_alias=True, exclude_none=True)
            ),
        )
        self._save_artifact(
            self.repository.get_workflow(workflow["workflow_id"]),
            identity,
            kind="checkout-jwt",
            exact=result.checkout_jwt,
            issuer="demo-merchant",
            kid=self.keys.merchant.get("kid"),
        )
        self.repository.transition(
            workflow["workflow_id"],
            expected_state=WorkflowState.MERCHANT_TASK_STARTING,
            to_state=WorkflowState.PAYMENT_APPROVAL_REQUIRED,
            actor_id="demo-merchant",
            actor_role="merchant",
            operation="payment-required",
            related_digest=requirements_digest,
        )

    def _approve_payment(
        self,
        workflow: dict[str, Any],
        identity: Identity,
        message_id: str,
        expected_version: int | None,
    ) -> None:
        merchant_task = self.repository.merchant_task(workflow["merchant_task_id"])
        if expected_version is not None and workflow["version"] != expected_version:
            raise DomainError(
                "STATE_TRANSITION_CONFLICT",
                "Workflow state changed concurrently.",
                workflow["workflow_id"],
                current_state=workflow["state"],
            )
        if self.clock() > datetime.fromisoformat(
            merchant_task["requirement"]["expires_at"].replace("Z", "+00:00")
        ):
            raise DomainError(
                "PAYMENT_APPROVAL_EXPIRED",
                "Payment approval window expired; a new quote is required.",
                workflow["workflow_id"],
                current_state=workflow["state"],
            )
        requirement = merchant_task["requirement"]
        project = merchant_task["task"]["status"]["message"]["metadata"][
            "io.github.taichihiromatsu.secure-mediation.v1"
        ]
        display = {
            "taskId": workflow["merchant_task_id"],
            "orderId": workflow["order_id"],
            "merchant": "demo-merchant",
            "prices": {
                "merchandiseAmount": 1250,
                "customerSurcharge": 0,
                "collectionRailCost": 0,
                "customerTotal": 1250,
                "providerCommission": 0,
                "merchantPayableAmount": 1250,
                "payoutRailCost": 0,
            },
            "profile": self.profile.profile_id,
            "simulated": True,
        }
        display_digest = canonical_digest(display)
        now = self.clock()
        issued_at = int(now.timestamp())
        expires_at = issued_at + 600
        approval_id = _id("approval:payment")
        approval_nonce = secrets.token_urlsafe(32)
        self.repository.record_payment_approval(
            workflow_id=workflow["workflow_id"],
            approval_id=approval_id,
            task_id=workflow["merchant_task_id"],
            checkout_hash=requirement["checkout_hash"],
            nonce=approval_nonce,
            display_digest=display_digest,
            approved_at=now.isoformat().replace("+00:00", "Z"),
            expires_at=datetime.fromtimestamp(expires_at, UTC).isoformat().replace("+00:00", "Z"),
        )
        authorization_request = {
            "workflowId": workflow["workflow_id"],
            "displayDigest": display_digest,
            "tenantId": identity.tenant_id,
            "customerId": identity.customer_id,
            "taskId": workflow["merchant_task_id"],
            "approvalId": approval_id,
            "issuedAt": issued_at,
            "expiresAt": expires_at,
        }
        operation_id = f"authorize:{approval_id}"
        workflow = self.repository.transition(
            workflow["workflow_id"],
            expected_state=WorkflowState.PAYMENT_APPROVAL_REQUIRED,
            to_state=WorkflowState.PAYMENT_AUTHORIZING,
            expected_version=expected_version,
            actor_id=identity.customer_id,
            actor_role="customer",
            operation="approve-payment",
            approval_intent="approve-payment",
            related_digest=display_digest,
            updates={"payment_approval_id": approval_id},
            outbox=("trusted-surface:issue", operation_id, authorization_request),
        )
        self._run_outbox_operation(operation_id)

    def _execute_payment_authorization(
        self, authorization_request: dict[str, Any]
    ) -> None:
        workflow = self.repository.get_workflow(authorization_request["workflowId"])
        if workflow["state"] not in {
            WorkflowState.PAYMENT_AUTHORIZING,
            WorkflowState.PAYMENT_APPROVED,
            WorkflowState.PAYMENT_SUBMITTED,
            WorkflowState.PAYMENT_VERIFYING,
            WorkflowState.FULFILLMENT_PREPARING,
            WorkflowState.PAYMENT_SETTLING,
            WorkflowState.FULFILLMENT_COMMITTING,
        }:
            if workflow["state"] in {
                WorkflowState.COMPLETED,
                WorkflowState.PAYMENT_FAILED,
                WorkflowState.RECONCILIATION_REQUIRED,
                WorkflowState.REFUND_REQUIRED,
                WorkflowState.REFUNDED,
            }:
                return
            raise DomainError(
                "STATE_TRANSITION_CONFLICT",
                "Payment authorization cannot run from the current state.",
                workflow["workflow_id"],
                current_state=workflow["state"],
            )
        identity = Identity(
            authorization_request["tenantId"], authorization_request["customerId"]
        )
        merchant_task = self.repository.merchant_task(workflow["merchant_task_id"])
        project = merchant_task["task"]["status"]["message"]["metadata"][
            "io.github.taichihiromatsu.secure-mediation.v1"
        ]
        # A crash may occur after the idempotent external completion response
        # has replaced the Merchant's current task view. Recover immutable
        # challenges from the originally committed merchant-task evidence.
        if "paymentMandateChallenge" not in project:
            try:
                original_task = self.repository.original_merchant_task(
                    workflow["workflow_id"],
                    tenant_id=identity.tenant_id,
                    actor_id=identity.customer_id,
                )
            except KeyError:
                raise DomainError(
                    "EVIDENCE_MISSING",
                    "Original Merchant task evidence is required for recovery.",
                    workflow["workflow_id"],
                )
            project = original_task["status"]["message"]["metadata"][
                "io.github.taichihiromatsu.secure-mediation.v1"
            ]
        self._authorize_and_execute(
            workflow,
            identity,
            merchant_task,
            project,
            int(authorization_request["issuedAt"]),
            int(authorization_request["expiresAt"]),
        )

    def _authorize_and_execute(
        self,
        workflow: dict[str, Any],
        identity: Identity,
        merchant_task: dict[str, Any],
        project: dict[str, Any],
        issued_at: int,
        expires_at: int,
    ) -> None:
        task_id = workflow["merchant_task_id"]
        requirement = merchant_task["requirement"]
        requirements = requirement["requirements"]
        ts_capability_id, _ = self._issue_capability(
            workflow=workflow,
            approval_id=workflow["payment_approval_id"],
            audience="demo-trusted-surface",
            operation="trusted-surface:issue",
            task_id=task_id,
            order_id=workflow["order_id"],
            issued_at=issued_at,
        )
        self.repository.consume_capability(
            ts_capability_id,
            canonical_digest(
                {
                    "workflowId": workflow["workflow_id"],
                    "taskId": task_id,
                    "checkoutHash": requirement["checkout_hash"],
                    "paymentApprovalId": workflow["payment_approval_id"],
                }
            ),
        )
        saved_checkout = self.repository.artifact_exact(
            workflow["workflow_id"], "checkout-mandate"
        )
        saved_payment = self.repository.artifact_exact(
            workflow["workflow_id"], "payment-mandate"
        )
        generated_mandates = None
        if saved_checkout is None or saved_payment is None:
            generated_mandates = self.trusted_surface.issue_closed_mandates(
                checkout_jwt=requirement["checkout_jwt"],
                merchant_id="demo-merchant",
                merchant_name="Demo Merchant",
                amount=1250,
                currency="USD",
                instrument_id="demo-instrument-1",
                checkout_audience=project["checkoutMandateChallenge"]["aud"],
                checkout_nonce=project["checkoutMandateChallenge"]["nonce"],
                payment_audience=project["paymentMandateChallenge"]["aud"],
                payment_nonce=project["paymentMandateChallenge"]["nonce"],
                issued_at=issued_at,
                expires_at=expires_at,
            )
        mandates = MandatePresentations(
            checkout=(
                saved_checkout[1].decode("utf-8")
                if saved_checkout
                else generated_mandates.checkout
            ),
            payment=(
                saved_payment[1].decode("utf-8")
                if saved_payment
                else generated_mandates.payment
            ),
            checkout_hash=b64url_sha256(requirement["checkout_jwt"]),
        )
        self.credential_provider.verify_payment_mandate(
            mandates.payment,
            nonce=project["paymentMandateChallenge"]["nonce"],
            checkout_hash=mandates.checkout_hash,
            amount=1250,
        )
        checkout_mandate_id, checkout_mandate_digest = self._save_artifact(
            workflow,
            identity,
            kind="checkout-mandate",
            exact=mandates.checkout,
            issuer="demo-user-credential-issuer",
            kid=self.keys.user_root.get("kid"),
            reference=closed_reference(mandates.checkout),
        )
        payment_mandate_id, payment_mandate_digest = self._save_artifact(
            workflow,
            identity,
            kind="payment-mandate",
            exact=mandates.payment,
            issuer="demo-user-credential-issuer",
            kid=self.keys.user_root.get("kid"),
            reference=closed_reference(mandates.payment),
        )
        signer_capability_id, _ = self._issue_capability(
            workflow=workflow,
            approval_id=workflow["payment_approval_id"],
            audience="demo-simulation-signer",
            operation="simulation-signer:sign",
            task_id=task_id,
            order_id=workflow["order_id"],
            issued_at=issued_at,
        )
        saved_proof = self.repository.artifact_exact(
            workflow["workflow_id"], "simulation-payload"
        )
        proof = (
            saved_proof[1].decode("utf-8")
            if saved_proof
            else self.profile.build_proof(
                {
                "jti": _stable_id("simulation-proof", workflow["workflow_id"], "payment"),
                "iss": "demo-simulation-signer",
                "aud": ["merchant:demo-merchant", "demo-mpp"],
                "workflowId": workflow["workflow_id"],
                "taskId": task_id,
                "planDigest": workflow["plan_digest"],
                "checkoutHash": mandates.checkout_hash,
                "paymentMandateDigest": payment_mandate_digest,
                "requirementsDigest": requirement["requirements_digest"],
                "amount": 1250,
                "asset": "USD",
                "network": "demo:local",
                "payTo": "merchant:demo-merchant",
                "nonce": secrets.token_urlsafe(32),
                "iat": issued_at,
                "exp": expires_at,
                }
            )
        )
        proof_id, proof_digest = self._save_artifact(
            workflow,
            identity,
            kind="simulation-payload",
            exact=proof,
            issuer="demo-simulation-signer",
            kid=self.keys.simulation_signer.get("kid"),
        )
        self.repository.consume_capability(signer_capability_id, proof_digest)
        cp_capability_id, _ = self._issue_capability(
            workflow=workflow,
            approval_id=workflow["payment_approval_id"],
            audience="demo-credential-provider",
            operation="credential-provider:issue",
            task_id=task_id,
            order_id=workflow["order_id"],
            issued_at=issued_at,
        )
        saved_credential = self.repository.artifact_exact(
            workflow["workflow_id"], "payment-credential"
        )
        credential = (
            saved_credential[1].decode("utf-8")
            if saved_credential
            else self.credential_provider.issue(
                credential_id=_stable_id("credential", workflow["workflow_id"], "payment"),
                workflow_id=workflow["workflow_id"],
                plan_digest=workflow["plan_digest"],
                task_id=task_id,
                checkout_hash=mandates.checkout_hash,
                payment_mandate=mandates.payment,
                requirements_digest=requirement["requirements_digest"],
                payload_digest=proof_digest,
                nonce=secrets.token_urlsafe(32),
                issued_at=issued_at,
                expires_at=expires_at,
            )
        )
        credential_id, credential_digest = self._save_artifact(
            workflow,
            identity,
            kind="payment-credential",
            exact=credential,
            issuer="demo-credential-provider",
            kid=self.keys.credential_provider.get("kid"),
        )
        self.repository.consume_capability(cp_capability_id, credential_digest)
        verify_capability_id, _ = self._issue_capability(
            workflow=workflow,
            approval_id=workflow["payment_approval_id"],
            audience="demo-mpp",
            operation="mpp:verify",
            task_id=task_id,
            order_id=workflow["order_id"],
            issued_at=issued_at,
        )
        self.mpp.verify_authorization(
            task_id=task_id,
            credential=credential,
            proof=proof,
            requirement=requirements,
        )
        self.repository.consume_capability(
            verify_capability_id,
            canonical_digest(
                {
                    "credentialDigest": credential_digest,
                    "payloadDigest": proof_digest,
                    "requirementsDigest": requirement["requirements_digest"],
                }
            ),
        )
        workflow = self.repository.get_workflow(workflow["workflow_id"])
        if workflow["state"] == WorkflowState.PAYMENT_AUTHORIZING:
            workflow = self.repository.transition(
                workflow["workflow_id"],
                expected_state=WorkflowState.PAYMENT_AUTHORIZING,
                to_state=WorkflowState.PAYMENT_APPROVED,
                actor_id="demo-credential-provider",
                actor_role="credential-provider",
                operation="authorization-evidence-committed",
                related_digest=credential_digest,
            )
        submission = payment_message(
            task_id=task_id,
            context_id=workflow["context_id"],
            message_id=f"message:payment-submitted:{task_id}",
            status="payment-submitted",
            payload=self.profile.build_submission(proof=proof),
            project={
                "profile": self.profile.profile_id,
                "simulated": True,
                "checkoutMandate": {"id": checkout_mandate_id, "digest": checkout_mandate_digest},
                "paymentMandate": {"id": payment_mandate_id, "digest": payment_mandate_digest},
                "credential": {"id": credential_id, "digest": credential_digest},
                "payload": {"id": proof_id, "digest": proof_digest},
            },
        )
        submission_wire = submission.model_dump(mode="json", by_alias=True, exclude_none=True)
        if workflow["state"] == WorkflowState.PAYMENT_APPROVED:
            submit_capability_id, submit_capability_token = self._issue_capability(
                workflow=workflow,
                approval_id=workflow["payment_approval_id"],
                audience="merchant:demo-merchant",
                operation="merchant:payment-submit",
                task_id=task_id,
                order_id=workflow["order_id"],
                issued_at=issued_at,
            )
            self.merchant.submit_payment(
                message=submission,
                checkout_mandate=mandates.checkout,
                checkout_jwt=requirement["checkout_jwt"],
                checkout_nonce=project["checkoutMandateChallenge"]["nonce"],
                capability_id=submit_capability_id,
                capability_token=submit_capability_token,
                workflow_id=workflow["workflow_id"],
                order_id=workflow["order_id"],
            )
            crash_after("external:payment-submit-returned")
            self.repository.consume_capability(
                submit_capability_id, canonical_digest(submission_wire)
            )
            workflow = self.repository.transition(
                workflow["workflow_id"],
                expected_state=WorkflowState.PAYMENT_APPROVED,
                to_state=WorkflowState.PAYMENT_SUBMITTED,
                actor_id="secure-mediator",
                actor_role="shopping-agent",
                operation="payment-submit",
                related_digest=canonical_digest(submission_wire),
            )
        if workflow["state"] == WorkflowState.PAYMENT_SUBMITTED:
            workflow = self.repository.transition(
                workflow["workflow_id"],
                expected_state=WorkflowState.PAYMENT_SUBMITTED,
                to_state=WorkflowState.PAYMENT_VERIFYING,
                actor_id="demo-merchant",
                actor_role="merchant",
                operation="payment-verify",
            )
        if workflow["state"] == WorkflowState.PAYMENT_VERIFYING:
            workflow = self.repository.transition(
                workflow["workflow_id"],
                expected_state=WorkflowState.PAYMENT_VERIFYING,
                to_state=WorkflowState.FULFILLMENT_PREPARING,
                actor_id="demo-merchant",
                actor_role="merchant",
                operation="fulfillment-prepare",
            )
        if workflow["state"] == WorkflowState.FULFILLMENT_PREPARING:
            prepare_capability_id, prepare_capability_token = self._issue_capability(
                workflow=workflow,
                approval_id=workflow["payment_approval_id"],
                audience="merchant:demo-merchant",
                operation="merchant:fulfillment-prepare",
                task_id=task_id,
                order_id=workflow["order_id"],
                issued_at=issued_at,
            )
            prepared = self.merchant.prepare(
                task_id,
                f"prepare:{task_id}",
                workflow_id=workflow["workflow_id"],
                order_id=workflow["order_id"],
                capability_id=prepare_capability_id,
                capability_token=prepare_capability_token,
            )
            self.repository.consume_capability(
                prepare_capability_id, canonical_digest(prepared)
            )
            workflow = self.repository.transition(
                workflow["workflow_id"],
                expected_state=WorkflowState.FULFILLMENT_PREPARING,
                to_state=WorkflowState.PAYMENT_SETTLING,
                actor_id="demo-merchant",
                actor_role="merchant",
                operation="settlement-start",
                related_digest=canonical_digest(prepared),
            )
        attempt_id = _stable_id("attempt", workflow["workflow_id"], "settlement")
        settle_request_digest = canonical_digest(
            {
                "taskId": task_id,
                "credentialDigest": credential_digest,
                "payloadDigest": proof_digest,
            }
        )
        settle_capability_id, _ = self._issue_capability(
            workflow=workflow,
            approval_id=workflow["payment_approval_id"],
            audience="demo-mpp",
            operation="mpp:settle",
            task_id=task_id,
            order_id=workflow["order_id"],
            issued_at=issued_at,
        )
        settle_receipt = self.profile.settle_receipt(
            attempt_id=attempt_id,
            success=self.rail_fault not in {"failed", "unknown"},
            error_reason="SETTLEMENT_FAILED" if self.rail_fault == "failed" else None,
        )
        receipt_evidence_id = _stable_id(
            "evidence:simulation-receipt", workflow["workflow_id"], "settlement"
        )
        receipt_evidence_digest = self.repository.put_evidence(
            workflow_id=workflow["workflow_id"],
            evidence_id=receipt_evidence_id,
            tenant_id=identity.tenant_id,
            kind="simulation-settlement-receipt",
            exact_bytes=canonical_bytes(settle_receipt),
            kid=None,
            media_type="application/json",
            profile_id=self.profile.profile_id,
        )
        attempt = self.repository.settle_simulation(
            attempt_id=attempt_id,
            task_id=task_id,
            idempotency_key=f"settle:{task_id}",
            amount=1250,
            request_digest=settle_request_digest,
            receipt=settle_receipt,
            receipt_evidence_id=receipt_evidence_id,
            receipt_evidence_digest=receipt_evidence_digest,
            unknown=self.rail_fault == "unknown",
            fail=self.rail_fault == "failed",
        )
        crash_after("external:settlement-returned")
        self.repository.consume_capability(settle_capability_id, settle_request_digest)
        if attempt["state"] == "unknown":
            if workflow["state"] == WorkflowState.PAYMENT_SETTLING:
                self.repository.transition(
                    workflow["workflow_id"],
                    expected_state=WorkflowState.PAYMENT_SETTLING,
                    to_state=WorkflowState.RECONCILIATION_REQUIRED,
                    actor_id="demo-mpp",
                    actor_role="merchant-payment-processor",
                    operation="settlement-unknown",
                    error_code="RECONCILIATION_REQUIRED",
                    updates={"last_error_code": "SETTLEMENT_OUTCOME_UNKNOWN"},
                )
            return
        if attempt["state"] == "failed":
            payment_reference = closed_reference(mandates.payment)
            payment_error_receipt = Ap2ReceiptFactory.payment(
                key=self.keys.mpp,
                reference=payment_reference,
                issued_at=issued_at,
                payment_id=f"payment:{attempt_id}",
                simulation_reference=f"sim:{attempt_id}",
                success=False,
                error="settlement_failed",
                error_description="Simulation settlement failed; no payment completed.",
            )
            self._save_artifact(
                workflow,
                identity,
                kind="payment-receipt",
                exact=payment_error_receipt,
                issuer="demo-mpp",
                kid=self.keys.mpp.get("kid"),
                reference=payment_reference,
            )
            Ap2ReceiptFactory.verify_payment(
                payment_error_receipt,
                public_key(self.keys.mpp),
                payment_reference,
            )
            if workflow["state"] == WorkflowState.PAYMENT_SETTLING:
                self.repository.transition(
                    workflow["workflow_id"],
                    expected_state=WorkflowState.PAYMENT_SETTLING,
                    to_state=WorkflowState.PAYMENT_FAILED,
                    actor_id="demo-mpp",
                    actor_role="merchant-payment-processor",
                    operation="settlement-failed",
                    error_code="PAYMENT_FAILED",
                )
            return
        payment_reference = closed_reference(mandates.payment)
        payment_receipt = Ap2ReceiptFactory.payment(
            key=self.keys.mpp,
            reference=payment_reference,
            issued_at=issued_at,
            payment_id=f"payment:{attempt_id}",
            simulation_reference=f"sim:{attempt_id}",
            success=True,
        )
        payment_receipt_id, _ = self._save_artifact(
            workflow,
            identity,
            kind="payment-receipt",
            exact=payment_receipt,
            issuer="demo-mpp",
            kid=self.keys.mpp.get("kid"),
            reference=payment_reference,
        )
        Ap2ReceiptFactory.verify_payment(
            payment_receipt,
            public_key(self.keys.mpp),
            payment_reference,
        )
        if workflow["state"] == WorkflowState.PAYMENT_SETTLING:
            workflow = self.repository.transition(
                workflow["workflow_id"],
                expected_state=WorkflowState.PAYMENT_SETTLING,
                to_state=WorkflowState.FULFILLMENT_COMMITTING,
                actor_id="demo-mpp",
                actor_role="merchant-payment-processor",
                operation="settlement-success",
                related_digest=receipt_evidence_digest,
            )
        else:
            workflow = self.repository.get_workflow(workflow["workflow_id"])
        commit_capability_id, commit_capability_token = self._issue_capability(
            workflow=workflow,
            approval_id=workflow["payment_approval_id"],
            audience="merchant:demo-merchant",
            operation="merchant:fulfillment-commit",
            task_id=task_id,
            order_id=workflow["order_id"],
            issued_at=issued_at,
        )
        if self.commit_fault:
            self.repository.consume_capability(
                commit_capability_id,
                canonical_digest({"taskId": task_id, "outcome": "failed"}),
            )
            checkout_reference = closed_reference(mandates.checkout)
            checkout_error_receipt = Ap2ReceiptFactory.checkout(
                key=self.keys.merchant,
                reference=checkout_reference,
                issued_at=issued_at,
                order_id=workflow["order_id"],
                success=False,
                error="fulfillment_failed",
                error_description="Settlement succeeded but fulfillment commit failed.",
            )
            self._save_artifact(
                workflow,
                identity,
                kind="checkout-receipt",
                exact=checkout_error_receipt,
                issuer="demo-merchant",
                kid=self.keys.merchant.get("kid"),
                reference=checkout_reference,
            )
            Ap2ReceiptFactory.verify_checkout(
                checkout_error_receipt,
                public_key(self.keys.merchant),
                checkout_reference,
            )
            self.repository.require_refund(
                refund_id=_stable_id("refund", workflow["workflow_id"], "settlement"),
                workflow_id=workflow["workflow_id"],
                attempt_id=attempt_id,
                original_payment_id=f"payment:{attempt_id}",
                reason="fulfillment-commit-failed",
                idempotency_key=f"refund-required:{attempt_id}",
            )
            self.repository.transition(
                workflow["workflow_id"],
                expected_state=WorkflowState.FULFILLMENT_COMMITTING,
                to_state=WorkflowState.REFUND_REQUIRED,
                actor_id="demo-merchant",
                actor_role="merchant",
                operation="fulfillment-commit-failed",
                error_code="REFUND_REQUIRED",
            )
            return
        commit = {
            "taskId": task_id,
            "state": "committed",
            "externalSideEffect": False,
            "simulated": True,
        }
        self.repository.save_fulfillment(
            operation_id=f"commit:{task_id}",
            task_id=task_id,
            phase="commit",
            request_digest=canonical_digest(commit),
            state="committed",
            result=commit,
        )
        self.repository.consume_capability(commit_capability_id, canonical_digest(commit))
        checkout_reference = closed_reference(mandates.checkout)
        checkout_receipt = Ap2ReceiptFactory.checkout(
            key=self.keys.merchant,
            reference=checkout_reference,
            issued_at=issued_at,
            order_id=workflow["order_id"],
            success=True,
        )
        checkout_receipt_id, _ = self._save_artifact(
            workflow,
            identity,
            kind="checkout-receipt",
            exact=checkout_receipt,
            issuer="demo-merchant",
            kid=self.keys.merchant.get("kid"),
            reference=checkout_reference,
        )
        Ap2ReceiptFactory.verify_checkout(
            checkout_receipt,
            public_key(self.keys.merchant),
            checkout_reference,
        )
        receipts = [settle_receipt]
        task = self.merchant.complete_task(
            task_id=task_id,
            context_id=workflow["context_id"],
            receipts=receipts,
            checkout_receipt_id=checkout_receipt_id,
            payment_receipt_id=payment_receipt_id,
            workflow_id=workflow["workflow_id"],
            order_id=workflow["order_id"],
            capability_id=commit_capability_id,
            capability_token=commit_capability_token,
        )
        crash_after("external:fulfillment-commit-returned")
        final_task_wire = canonical_bytes(
            task.model_dump(mode="json", by_alias=True, exclude_none=True)
        )
        final_validation_capability_id, _ = self._issue_capability(
            workflow=workflow,
            approval_id=workflow["payment_approval_id"],
            audience="secure-mediator",
            operation="mediator:final-validate",
            task_id=task_id,
            order_id=workflow["order_id"],
            issued_at=issued_at,
        )
        self._save_artifact(
            workflow,
            identity,
            kind="final-merchant-task",
            exact=final_task_wire,
            issuer="demo-merchant",
            kid=self.keys.merchant.get("kid"),
        )
        self.repository.consume_capability(
            final_validation_capability_id, sha256_digest(final_task_wire)
        )
        self.repository.transition(
            workflow["workflow_id"],
            expected_state=WorkflowState.FULFILLMENT_COMMITTING,
            to_state=WorkflowState.COMPLETED,
            actor_id="secure-mediator",
            actor_role="shopping-agent",
            operation="verify-final-evidence",
        )

    def _save_artifact(
        self,
        workflow: dict[str, Any],
        identity: Identity,
        *,
        kind: str,
        exact: str | bytes,
        issuer: str,
        kid: str,
        reference: str | None = None,
    ) -> tuple[str, str]:
        existing = self.repository.artifact_exact(workflow["workflow_id"], kind)
        if existing is not None:
            reference_row, _ = existing
            return reference_row["artifact_id"], reference_row["evidence_digest"]
        artifact_id = _stable_id(f"artifact:{kind}", workflow["workflow_id"], kind)
        evidence_id = _stable_id(f"evidence:{kind}", workflow["workflow_id"], kind)
        role_key = next(
            key
            for key in (
                self.keys.plan_authority,
                self.keys.user_root,
                self.keys.trusted_surface,
                self.keys.merchant,
                self.keys.credential_provider,
                self.keys.simulation_signer,
                self.keys.mpp,
                self.keys.service_auth,
            )
            if key.get("kid") == kid
        )
        trust_evidence_id = f"evidence:trust:{kid}"
        trust_digest = self.repository.put_evidence(
            workflow_id=workflow["workflow_id"],
            evidence_id=trust_evidence_id,
            tenant_id=identity.tenant_id,
            kind="trust-key-snapshot",
            exact_bytes=canonical_bytes(
                {"issuer": issuer, "kid": kid, "jwk": json.loads(public_key(role_key).export())}
            ),
            kid=kid,
            media_type="application/jwk+json",
            profile_id="demo-es256-v1",
        )
        trust_snapshot_id = self.repository.ensure_trust_snapshot(
            snapshot_id=f"trust:{kid}",
            issuer=issuer,
            kid=kid,
            evidence_id=trust_evidence_id,
            evidence_digest=trust_digest,
        )
        digest = self.repository.put_evidence(
            workflow_id=workflow["workflow_id"],
            evidence_id=evidence_id,
            tenant_id=identity.tenant_id,
            kind=kind,
            exact_bytes=exact,
            kid=kid,
            media_type="application/jwt" if isinstance(exact, str) else "application/json",
            profile_id="AP2-v0.2" if "mandate" in kind or "receipt" in kind or kind == "checkout-jwt" else self.profile.profile_id,
        )
        self.repository.insert_artifact_reference(
            artifact_id=artifact_id,
            workflow_id=workflow["workflow_id"],
            task_id=workflow["merchant_task_id"],
            kind=kind,
            evidence_id=evidence_id,
            evidence_digest=digest,
            issuer=issuer,
            kid=kid,
            trust_snapshot_id=trust_snapshot_id,
            reference_digest=reference,
        )
        return artifact_id, digest

    def _issue_capability(
        self,
        *,
        workflow: dict[str, Any],
        approval_id: str,
        audience: str,
        operation: str,
        task_id: str | None,
        order_id: str | None,
        issued_at: int,
    ) -> tuple[str, str]:
        existing = self.repository.capability_for_operation(
            workflow["workflow_id"], operation
        )
        if existing is not None:
            capability, token = existing
            if (
                capability["approval_id"] != approval_id
                or capability["audience"] != audience
                or capability["task_id"] != task_id
                or capability["order_id"] != order_id
            ):
                raise DomainError(
                    "IDEMPOTENCY_CONFLICT",
                    "Capability operation binding changed during recovery.",
                    operation,
                )
            return capability["capability_id"], token
        capability_id = _stable_id(
            "capability", workflow["workflow_id"], approval_id, operation
        )
        nonce = secrets.token_urlsafe(32)
        token = self.authorization.issue_capability(
            {
                "jti": capability_id,
                "aud": audience,
                "operation": operation,
                "approvalId": approval_id,
                "workflowId": workflow["workflow_id"],
                "planId": workflow["active_plan_id"],
                "planDigest": workflow["plan_digest"],
                "orderId": order_id,
                "taskId": task_id,
                "idempotencyScope": f"{operation}/{task_id or workflow['workflow_id']}",
                "nonce": nonce,
                "iat": issued_at,
                "exp": issued_at + 3600,
            }
        )
        self.authorization.verify(
            token,
            expected_type="secure-downstream-capability+jwt",
            audience=audience,
            operation=operation,
            now=issued_at,
        )
        evidence_id = _stable_id(
            "evidence:capability", workflow["workflow_id"], approval_id, operation
        )
        digest = self.repository.put_evidence(
            workflow_id=workflow["workflow_id"],
            evidence_id=evidence_id,
            tenant_id=workflow["tenant_id"],
            kind="downstream-capability",
            exact_bytes=token,
            kid=self.keys.plan_authority.get("kid"),
            media_type="application/jwt",
            profile_id="secure-downstream-capability/1",
        )
        self.repository.insert_capability(
            capability_id=capability_id,
            approval_id=approval_id,
            workflow_id=workflow["workflow_id"],
            plan_digest=workflow["plan_digest"],
            order_id=order_id,
            task_id=task_id,
            audience=audience,
            operation=operation,
            nonce=nonce,
            evidence_id=evidence_id,
            evidence_digest=digest,
            issued_at=issued_at,
            expires_at=issued_at + 3600,
        )
        return capability_id, token

    def _reject(self, workflow: dict[str, Any], identity: Identity, message_id: str) -> None:
        state = WorkflowState(workflow["state"])
        if state == WorkflowState.PAYMENT_APPROVAL_REQUIRED:
            rejection = payment_message(
                task_id=workflow["merchant_task_id"],
                context_id=workflow["context_id"],
                message_id=f"message:payment-rejected:{workflow['merchant_task_id']}",
                status="payment-rejected",
            )
            self.repository.append_merchant_message(
                message_id=rejection.message_id,
                task_id=workflow["merchant_task_id"],
                context_id=workflow["context_id"],
                status="payment-rejected",
                message=rejection.model_dump(mode="json", by_alias=True, exclude_none=True),
            )
        self.repository.transition(
            workflow["workflow_id"],
            expected_state=state,
            to_state=WorkflowState.CANCELLED,
            actor_id=identity.customer_id,
            actor_role="customer",
            operation="reject-current",
        )

    @staticmethod
    def _require_owner(workflow: dict[str, Any], identity: Identity) -> None:
        if workflow["tenant_id"] != identity.tenant_id or workflow["customer_id"] != identity.customer_id:
            raise DomainError(
                "TENANT_BINDING_MISMATCH",
                "Workflow is not available for this authenticated identity.",
                workflow["workflow_id"],
            )
