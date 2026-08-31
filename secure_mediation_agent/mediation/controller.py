"""Deterministic mediation state machine behind the public ADK agent."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock
from typing import Any, Iterable, Literal
from uuid import uuid4

from pydantic import BaseModel

from secure_mediation_agent.demo_catalog import (
    project_confirmation_artifact,
    validate_payment_requirement,
)

from .a2a_executor import A2AExecution, A2AOperation
from .adapters import SIMULATION_EXTENSION, maybe_await
from .approval_targets import (
    build_payment_approval_target,
    build_plan_approval_target,
)
from .canonical import canonical_digest, safe_ref as safe_opaque_ref
from .errors import MediationError, ReviewRequired, SecurityBlocked
from .models import (
    BridgeApprovalResult,
    BridgeAttachment,
    BridgeExecutionResult,
    MediationContinuation,
    MediationPublicView,
    MediationSession,
    MediationState,
    OwnerScope,
    PaymentApprovalTarget,
    PendingAction,
    PlanApprovalTarget,
    PlanApproval,
    RefundResult,
    SubjectScope,
    TextPart,
    TraceEvent,
    utc_now,
)
from .ports import (
    A2AExecutorPort,
    FinalValidationPort,
    MatcherPort,
    MediationStorePort,
    PaymentBridgePort,
    PlannerPort,
    StableGatePort,
)
from .persistence_models import (
    RequestReservation,
    paid_completion_message,
    paid_payment_approval_message,
)


APPROVAL_TEXT = "承認"
REFUND_TEXT = "返金"
SIMULATION_PROFILE = "x402-wire-simulation/1"


def _wire(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return value


def _bridge_owner(owner: OwnerScope) -> dict[str, str]:
    """The bridge receives only the server-owned four-part ownership key."""

    return {
        "tenantId": owner.tenant_id,
        "subjectId": owner.subject,
        "sessionId": owner.adk_session_id,
        "contextId": owner.mediation_session_id,
        "mediationSessionId": owner.mediation_session_id,
    }


class MediationController:
    """Routes new goals, exact approvals, payment continuations, and refunds."""

    def __init__(
        self,
        *,
        store: MediationStorePort,
        matcher: MatcherPort,
        planner: PlannerPort,
        executor: A2AExecutorPort,
        gates: StableGatePort,
        payment_bridge: PaymentBridgePort,
        final_validator: FinalValidationPort,
        durability_profile: Literal["local-durable", "ephemeral-demo"] | None = None,
    ) -> None:
        self.store = store
        self.matcher = matcher
        self.planner = planner
        self.executor = executor
        self.gates = gates
        self.payment_bridge = payment_bridge
        self.final_validator = final_validator
        resolved_profile = durability_profile or getattr(
            store, "durability_profile", "ephemeral-demo"
        )
        if resolved_profile not in {"local-durable", "ephemeral-demo"}:
            raise RuntimeError("unsupported mediation durability profile")
        self.durability_profile: Literal["local-durable", "ephemeral-demo"] = (
            resolved_profile
        )
        # The legacy in-memory demo store predates durable request reservations.
        # Keep its process-local reservation fence here; production SQLite stores
        # implement the atomic port methods below.
        self._ephemeral_request_lock = RLock()
        self._ephemeral_requests: dict[tuple[str, str, str, str], str] = {}

    async def submit(
        self,
        *,
        scope: SubjectScope,
        parts: Iterable[TextPart | dict[str, Any]],
        request_id: str,
        expected_version: int | None = None,
    ) -> MediationPublicView:
        parsed_parts = tuple(
            part if isinstance(part, TextPart) else TextPart.model_validate(part)
            for part in parts
        )
        if not parsed_parts:
            raise MediationError("EMPTY_MESSAGE", "テキストを入力してください。")
        request_digest = self.request_digest(
            scope=scope,
            parts=parsed_parts,
            expected_version=expected_version,
        )
        reservation = self._reserve_request(
            scope,
            request_id,
            request_digest,
            expected_version=expected_version,
        )
        if reservation.status == "completed":
            if reservation.view is None:
                raise MediationError(
                    "MEDIATION_STORE_INTEGRITY",
                    "The completed mediation result is unavailable.",
                )
            return reservation.view

        safe_to_fail = True
        try:
            active = self.store.active_for(scope)
            if expected_version is not None and (
                active is None or active.version != expected_version
            ):
                raise MediationError(
                    "STATE_TRANSITION_CONFLICT",
                    "The mediation session changed; refresh before retrying.",
                )

            text = self._exact_text(parsed_parts)
            if active is None:
                if text == REFUND_TEXT:
                    session = await self._start_refund(scope)
                else:
                    session = await self._create_plan(scope, parsed_parts)
                safe_to_fail = False
            elif active.state == MediationState.WAITING_FOR_PLAN_APPROVAL:
                if text != APPROVAL_TEXT:
                    session = active
                else:
                    safe_to_fail = False
                    session = await self._approve_and_execute_plan(active)
            elif active.state == MediationState.WAITING_FOR_PAYMENT_APPROVAL:
                if text != APPROVAL_TEXT:
                    session = active
                else:
                    safe_to_fail = False
                    session = await self._approve_and_execute_payment(active)
            elif active.state == MediationState.REFUND_PENDING:
                if text != APPROVAL_TEXT:
                    session = active
                else:
                    safe_to_fail = False
                    session = await self._execute_refund(active)
            elif active.state in {
                MediationState.EXECUTING,
                MediationState.PAYMENT_APPROVED,
                MediationState.RESUMING_A2A,
                MediationState.REFUND_SUBMITTING,
            }:
                raise MediationError(
                    "MEDIATION_BUSY",
                    "処理中です。同じ操作を新しく開始しないでください。",
                )
            elif active.state == MediationState.REVIEW_REQUIRED:
                session = active
            else:
                raise SecurityBlocked(
                    "INVALID_MEDIATION_STATE", "現在の状態ではこの操作を実行できません。"
                )

            view = self.public_view(session)
            self._complete_request(
                scope,
                request_id,
                request_digest,
                session=session,
                view=view,
            )
            return view
        except Exception:
            # Approval/payment/refund branches may already have crossed an
            # external-effect or persisted-state boundary. Their reservation is
            # deliberately left processing so a retry fails closed.
            if safe_to_fail:
                self._fail_request(scope, request_id, request_digest)
            raise

    @staticmethod
    def request_digest(
        *,
        scope: SubjectScope,
        parts: Iterable[TextPart | dict[str, Any]],
        expected_version: int | None,
    ) -> str:
        parsed_parts = tuple(
            part if isinstance(part, TextPart) else TextPart.model_validate(part)
            for part in parts
        )
        return canonical_digest(
            {
                "scope": scope.model_dump(mode="json", by_alias=True),
                "parts": [part.model_dump(mode="json") for part in parsed_parts],
                "expectedVersion": expected_version,
            }
        )

    def completed_request_result(
        self,
        *,
        scope: SubjectScope,
        parts: Iterable[TextPart | dict[str, Any]],
        request_id: str,
        expected_version: int | None,
    ) -> RequestReservation | None:
        request_digest = self.request_digest(
            scope=scope,
            parts=parts,
            expected_version=expected_version,
        )
        load = getattr(self.store, "load_request", None)
        if callable(load):
            return load(scope, request_id, request_digest)
        cached = self.store.idempotent_result(scope, request_id, request_digest)
        if cached is None:
            return None
        return RequestReservation(
            status="completed",
            mediation_session_id=cached.owner.mediation_session_id,
            result_version=cached.version,
            view=self.public_view(cached),
        )

    def _reserve_request(
        self,
        scope: SubjectScope,
        request_id: str,
        request_digest: str,
        *,
        expected_version: int | None,
    ) -> RequestReservation:
        reserve = getattr(self.store, "reserve_request", None)
        if callable(reserve):
            return reserve(
                scope,
                request_id,
                request_digest,
                expected_version=expected_version,
            )

        key = (*scope.key, request_id)
        with self._ephemeral_request_lock:
            cached = self.store.idempotent_result(scope, request_id, request_digest)
            if cached is not None:
                view = self.public_view(cached)
                return RequestReservation(
                    status="completed",
                    mediation_session_id=cached.owner.mediation_session_id,
                    result_version=cached.version,
                    view=view,
                )
            processing_digest = self._ephemeral_requests.get(key)
            if processing_digest is not None:
                if processing_digest != request_digest:
                    raise MediationError(
                        "IDEMPOTENCY_CONFLICT",
                        "The request identifier was reused with different content.",
                    )
                raise MediationError(
                    "MEDIATION_REQUEST_IN_PROGRESS",
                    "The mediation request is already processing.",
                )
            self._ephemeral_requests[key] = request_digest
        return RequestReservation(status="reserved")

    def _complete_request(
        self,
        scope: SubjectScope,
        request_id: str,
        request_digest: str,
        *,
        session: MediationSession,
        view: MediationPublicView,
    ) -> None:
        complete = getattr(self.store, "complete_request", None)
        if callable(complete):
            complete(scope, request_id, request_digest, session=session, view=view)
            return
        self.store.remember_result(
            scope, request_id, request_digest, session
        )
        with self._ephemeral_request_lock:
            self._ephemeral_requests.pop((*scope.key, request_id), None)

    def _fail_request(
        self, scope: SubjectScope, request_id: str, request_digest: str
    ) -> None:
        fail = getattr(self.store, "fail_request", None)
        if callable(fail):
            fail(scope, request_id, request_digest)
            return
        with self._ephemeral_request_lock:
            key = (*scope.key, request_id)
            if self._ephemeral_requests.get(key) == request_digest:
                self._ephemeral_requests.pop(key, None)

    @staticmethod
    def _exact_text(parts: tuple[TextPart, ...]) -> str | None:
        if len(parts) != 1:
            return None
        return parts[0].text

    @staticmethod
    def _trace(
        session: MediationSession,
        *,
        stage: str,
        component_id: str,
        layer: str,
        operation_id: str,
        decision: str,
        safe_value: str | None = None,
    ) -> TraceEvent:
        return TraceEvent(
            sequence=len(session.trace) + 1,
            stage=stage,
            componentId=component_id,
            layer=layer,
            operationId=operation_id,
            decision=decision,
            safeRef=safe_opaque_ref(safe_value) if safe_value else None,
        )

    async def _create_plan(
        self, scope: SubjectScope, parts: tuple[TextPart, ...]
    ) -> MediationSession:
        if len(parts) != 1:
            raise MediationError(
                "NEW_GOAL_PARTS_INVALID", "新しい依頼は一つのテキストで送信してください。"
            )
        goal = parts[0].text
        mediation_session_id = f"med-{uuid4()}"
        owner = OwnerScope(
            subject=scope.subject,
            tenantId=scope.tenant_id,
            adkSessionId=scope.adk_session_id,
            mediationSessionId=mediation_session_id,
        )
        candidates = await self.matcher.match(goal)
        plan = await self.planner.create_plan(goal, owner, candidates)
        if plan.owner != owner:
            raise SecurityBlocked(
                "PLAN_OWNER_MISMATCH", "生成された計画の所有者が一致しません。"
            )
        approval_target = build_plan_approval_target(plan)
        approval_target_digest = canonical_digest(approval_target)
        session = MediationSession(
            owner=owner,
            goal=goal,
            state=MediationState.WAITING_FOR_PLAN_APPROVAL,
            version=0,
            plan=plan,
            approvalTargetDigest=approval_target_digest,
            pendingAction=PendingAction(
                kind="approve-plan", targetRef=safe_opaque_ref(approval_target_digest)
            ),
        )
        session.trace.extend(
            [
                self._trace(
                    session,
                    stage="agent-selected",
                    component_id="LegacyMatcherAdapter.match",
                    layer="matcher",
                    operation_id=mediation_session_id,
                    decision="validated",
                    safe_value=plan.active_step.selected_agent.snapshot_digest
                    if hasattr(plan, "active_step")
                    else plan.steps[0].selected_agent.snapshot_digest,
                ),
                TraceEvent(
                    sequence=2,
                    stage="plan-created",
                    componentId="TypedPlannerAdapter.create_plan",
                    layer="planner",
                    operationId=mediation_session_id,
                    decision="awaiting-explicit-approval",
                    safeRef=safe_opaque_ref(plan.plan_digest),
                ),
            ]
        )
        self.store.save_new(session)
        return session

    async def _approve_and_execute_plan(
        self, session: MediationSession
    ) -> MediationSession:
        approval_target = build_plan_approval_target(session.plan)
        approval_target_digest = canonical_digest(approval_target)
        if approval_target_digest != session.approval_target_digest:
            return self._terminal_transition(
                session,
                MediationState.BLOCKED,
                "plan-approval-target-mismatch",
                "MediationController._approve_and_execute_plan",
            )
        if session.plan.expires_at <= utc_now():
            return self._terminal_transition(
                session,
                MediationState.REVIEW_REQUIRED,
                "plan-expired",
                "MediationController._approve_and_execute_plan",
            )
        expected = session.version
        approval = PlanApproval(
            approvalId=f"approval-{uuid4()}",
            planId=session.plan.plan_id,
            planVersion=session.plan.plan_version,
            planDigest=session.plan.plan_digest,
            approvalTargetDigest=approval_target_digest,
            nonce=uuid4().hex,
            issuedAt=utc_now(),
        )
        operation = self._initial_operation(session)
        session.plan_approval = approval
        session.approval_target_digest = None
        session.state = MediationState.EXECUTING
        session.version += 1
        session.pending_action = PendingAction(kind="wait")
        session.trace.append(
            self._trace(
                session,
                stage="plan-approved",
                component_id="MediationController._approve_and_execute_plan",
                layer="controller",
                operation_id=operation.operation_id,
                decision="persisted-before-a2a",
                safe_value=approval.plan_digest,
            )
        )
        session = self.store.compare_and_set(session, expected_version=expected)
        try:
            execution = await self.executor.execute(operation)
            self._append_execution_trace(session, execution)
            return await self._handle_initial_response(session, execution)
        except ReviewRequired:
            return self._terminal_transition(
                session,
                MediationState.REVIEW_REQUIRED,
                "initial-a2a-review",
                "SharedA2AOperationExecutor.execute",
            )
        except SecurityBlocked:
            return self._terminal_transition(
                session,
                MediationState.BLOCKED,
                "initial-a2a-blocked",
                "SharedA2AOperationExecutor.execute",
            )
        except Exception:
            return self._terminal_transition(
                session,
                MediationState.REVIEW_REQUIRED,
                "initial-a2a-result-unknown",
                "SharedA2AOperationExecutor.execute",
            )

    def _initial_operation(self, session: MediationSession) -> A2AOperation:
        step = session.active_step
        operation_id = f"a2a-start-{uuid4()}"
        paid = step.selected_agent.canonical_agent_id == "agent-005"
        task_id = f"task-{uuid4()}" if paid else None
        order_id = f"order-{uuid4()}" if paid else None
        capability_id = f"capability-{uuid4()}" if paid else None
        params: dict[str, Any] = {
            "message": {
                "messageId": f"message-{uuid4()}",
                "role": "user",
                "parts": [{"kind": "text", "text": step.goal}],
                "metadata": {
                    "planId": session.plan.plan_id,
                    "planVersion": session.plan.plan_version,
                    "planDigest": session.plan.plan_digest,
                    "stepId": step.step_id,
                    "skillId": step.selected_agent.a2a_skill_id,
                },
            }
        }
        if paid:
            now = int(datetime.now(timezone.utc).timestamp())
            params.update(
                {
                    "action": "merchant-task:start",
                    "operationId": operation_id,
                    "workflowId": session.owner.mediation_session_id,
                    "planDigest": session.plan.plan_digest,
                    "taskId": task_id,
                    "orderId": order_id,
                    "contextId": session.owner.mediation_session_id,
                    "capabilityId": capability_id,
                    "issuedAt": now,
                    "expiresAt": now + 600,
                }
            )
        request = {
            "jsonrpc": "2.0",
            "id": operation_id,
            "method": "message/send",
            "params": params,
        }
        return A2AOperation(
            operationId=operation_id,
            kind="task-start",
            agent=step.selected_agent,
            request=request,
            requestDigest=canonical_digest(request),
            idempotencyKey=canonical_digest(
                {
                    "owner": session.owner.model_dump(mode="json", by_alias=True),
                    "planDigest": session.plan.plan_digest,
                    "stepId": step.step_id,
                    "kind": "task-start",
                }
            ),
        )

    async def _handle_initial_response(
        self, session: MediationSession, execution: A2AExecution
    ) -> MediationSession:
        response = execution.response.task
        requirement = response.payment_requirement
        if requirement is None:
            if response.state != "completed":
                return self._terminal_transition(
                    session,
                    MediationState.REVIEW_REQUIRED,
                    "unexpected-free-task-state",
                    "MediationController._handle_initial_response",
                )
            result = {
                "taskState": response.state,
                "taskDigest": response.task_digest,
                "artifact": response.artifact,
                "refundEligible": False,
            }
            return await self._finalize(session, result)

        self._validate_requirement(session, response)
        payment_gate = await self.gates.decide(
            "POST_PAYMENT_REQUIREMENT", execution.operation, response
        )
        session.trace.append(
            self._trace(
                session,
                stage="payment-requirement-gated",
                component_id="DeterministicStableGate.decide",
                layer="deterministic-validator",
                operation_id=execution.operation.operation_id,
                decision=payment_gate.decision,
                safe_value=payment_gate.decision_digest,
            )
        )
        if payment_gate.decision == "BLOCK":
            return self._terminal_transition(
                session,
                MediationState.BLOCKED,
                "payment-requirement-blocked",
                "DeterministicStableGate.decide",
            )
        if payment_gate.decision == "REVIEW":
            return self._terminal_transition(
                session,
                MediationState.REVIEW_REQUIRED,
                "payment-requirement-review",
                "DeterministicStableGate.decide",
            )

        private_material = execution.response.private_payment_material
        if private_material is None:
            return self._terminal_transition(
                session,
                MediationState.BLOCKED,
                "private-payment-material-missing",
                "MediationController._handle_initial_response",
            )
        raw_attachment = await maybe_await(
            self.payment_bridge.attach(
                owner=_bridge_owner(session.owner),
                approved_plan={
                    "plan": session.plan,
                    "approval": session.plan_approval,
                },
                step=session.active_step,
                remote_task=response,
                requirement={
                    "requirement": requirement,
                    "privatePaymentMaterial": private_material,
                },
            )
        )
        attachment = BridgeAttachment.model_validate(_wire(raw_attachment))
        continuation = MediationContinuation(
            continuationId=attachment.continuation_id,
            paymentWorkflowId=attachment.payment_workflow_id,
            owner=session.owner,
            planId=session.plan.plan_id,
            planVersion=session.plan.plan_version,
            planDigest=session.plan.plan_digest,
            stepId=session.active_step.step_id,
            remoteTask=response,
            requirement=requirement,
            version=attachment.version,
        )
        expected = session.version
        session.continuation = continuation
        approval_target = self._payment_approval_target(session)
        session.approval_target_digest = canonical_digest(approval_target)
        session.state = MediationState.WAITING_FOR_PAYMENT_APPROVAL
        session.version += 1
        session.pending_action = PendingAction(
            kind="approve-payment",
            targetRef=safe_opaque_ref(session.approval_target_digest),
        )
        session.trace.append(
            self._trace(
                session,
                stage="payment-continuation-attached",
                component_id="PaymentBridge.attach",
                layer="payment-bridge",
                operation_id=execution.operation.operation_id,
                decision="awaiting-exact-approval",
                safe_value=continuation.continuation_id,
            )
        )
        return self.store.compare_and_set(session, expected_version=expected)

    @staticmethod
    def _validate_requirement(session: MediationSession, response: Any) -> None:
        requirement = response.payment_requirement
        if requirement is None:
            raise SecurityBlocked(
                "PAYMENT_REQUIREMENT_MISSING", "支払い要件が見つかりません。"
            )
        try:
            scenario = validate_payment_requirement(requirement.payment_required)
        except ValueError as error:
            raise SecurityBlocked(
                "PAYMENT_SCENARIO_INVALID",
                "支払い要件のデモシナリオが一致しません。",
            ) from error
        step = session.active_step
        expected_extensions = step.selected_agent.payment_extension_uris
        if expected_extensions != (SIMULATION_EXTENSION,):
            raise SecurityBlocked(
                "PAYMENT_EXTENSION_NOT_PINNED", "支払い拡張が計画に固定されていません。"
            )
        if requirement.extension_uri != SIMULATION_EXTENSION:
            raise SecurityBlocked(
                "PAYMENT_EXTENSION_MISMATCH", "支払い拡張が一致しません。"
            )
        if requirement.profile_id != SIMULATION_PROFILE:
            raise SecurityBlocked(
                "PAYMENT_PROFILE_MISMATCH", "支払いプロファイルが一致しません。"
            )
        accepts = requirement.payment_required.get("accepts")
        expected_accept = {
            "scheme": "exact-simulated",
            "network": "demo:local",
            "asset": "USD",
            "payTo": "merchant:demo-merchant",
            "maxAmountRequired": str(requirement.amount_minor),
        }
        if (
            requirement.payment_required.get("x402Version") != 1
            or not isinstance(accepts, list)
            or accepts != [expected_accept]
        ):
            raise SecurityBlocked(
                "PAYMENT_WIRE_PROFILE_MISMATCH",
                "支払いwire条件が選択済みプロファイルと一致しません。",
            )
        if requirement.currency != step.currency:
            raise SecurityBlocked(
                "PAYMENT_CURRENCY_MISMATCH", "支払い通貨が計画と一致しません。"
            )
        if requirement.amount_minor > step.payment_limit_minor:
            raise SecurityBlocked(
                "PAYMENT_LIMIT_EXCEEDED", "支払い金額が承認済み計画の上限を超えています。"
            )
        fee = scenario["arrangementFee"]
        if (
            requirement.amount_minor != fee["amountMinor"]
            or requirement.currency != fee["currency"]
            or requirement.payee != fee["payee"]
        ):
            raise SecurityBlocked(
                "PAYMENT_SCENARIO_TERMS_MISMATCH",
                "支払い条件が固定デモシナリオと一致しません。",
            )
        if requirement.expires_at <= datetime.now(timezone.utc):
            raise SecurityBlocked("PAYMENT_QUOTE_EXPIRED", "支払い見積もりは期限切れです。")

    @staticmethod
    def _payment_approval_target(
        session: MediationSession,
    ) -> PaymentApprovalTarget:
        continuation = session.continuation
        if continuation is None:
            raise SecurityBlocked(
                "CONTINUATION_MISSING", "支払い継続情報が見つかりません。"
            )
        requirement = continuation.requirement
        remote = continuation.remote_task
        return build_payment_approval_target(
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

    async def _approve_and_execute_payment(
        self, session: MediationSession
    ) -> MediationSession:
        continuation = session.continuation
        if continuation is None:
            raise SecurityBlocked(
                "CONTINUATION_MISSING", "支払い継続情報が見つかりません。"
            )
        approval_target = self._payment_approval_target(session)
        approval_target_digest = canonical_digest(approval_target)
        if approval_target_digest != session.approval_target_digest:
            return self._terminal_transition(
                session,
                MediationState.BLOCKED,
                "payment-approval-target-mismatch",
                "MediationController._approve_and_execute_payment",
            )
        raw_approval = await maybe_await(
            self.payment_bridge.approve(
                owner=_bridge_owner(session.owner),
                continuation_id=continuation.continuation_id,
                expected_version=continuation.version,
                approval_text=APPROVAL_TEXT,
                expected_approval_target_digest=approval_target_digest,
            )
        )
        approval = BridgeApprovalResult.model_validate(_wire(raw_approval))
        if approval.continuation_id != continuation.continuation_id:
            raise SecurityBlocked(
                "CONTINUATION_BINDING_MISMATCH", "支払い承認の継続情報が一致しません。"
            )
        expected = session.version
        session.continuation = continuation.model_copy(
            update={"version": approval.version}
        )
        session.approval_target_digest = None
        session.state = MediationState.PAYMENT_APPROVED
        session.version += 1
        session.pending_action = PendingAction(
            kind="execute-approved-payment",
            targetRef=safe_opaque_ref(continuation.continuation_id),
        )
        session.trace.append(
            self._trace(
                session,
                stage="payment-approved",
                component_id="PaymentBridge.approve",
                layer="payment-bridge",
                operation_id=continuation.continuation_id,
                decision="persisted-before-payment-submit",
                safe_value=approval.approval_digest,
            )
        )
        session = self.store.compare_and_set(session, expected_version=expected)
        return await self._execute_approved_payment(session)

    async def _execute_approved_payment(
        self, session: MediationSession
    ) -> MediationSession:
        if session.state != MediationState.PAYMENT_APPROVED:
            raise SecurityBlocked(
                "PAYMENT_STATE_GATE", "承認済み状態でないため支払いを実行できません。"
            )
        continuation = session.continuation
        if continuation is None:
            raise SecurityBlocked(
                "CONTINUATION_MISSING", "支払い継続情報が見つかりません。"
            )
        expected = session.version
        operation_id = f"payment-submit:{continuation.continuation_id}:1"
        session.state = MediationState.RESUMING_A2A
        session.version += 1
        session.pending_action = PendingAction(kind="wait")
        session.trace.append(
            self._trace(
                session,
                stage="payment-submit-started",
                component_id="execute_approved_payment",
                layer="controller",
                operation_id=operation_id,
                decision="state-gate-passed",
                safe_value=continuation.continuation_id,
            )
        )
        session = self.store.compare_and_set(session, expected_version=expected)
        try:
            raw_execution = await maybe_await(
                self.payment_bridge.execute_approved_payment(
                    operation_id=operation_id,
                    continuation_id=continuation.continuation_id,
                    expected_version=continuation.version,
                )
            )
            result = (
                raw_execution
                if isinstance(raw_execution, BridgeExecutionResult)
                else BridgeExecutionResult.model_validate(_wire(raw_execution))
            )
            if result.continuation_id != continuation.continuation_id:
                raise SecurityBlocked(
                    "CONTINUATION_BINDING_MISMATCH",
                    "支払い結果の継続情報が一致しません。",
                )
            if result.state == "refund-required":
                expected = session.version
                session.continuation = continuation.model_copy(
                    update={"version": result.version}
                )
                session.result = result.result
                session.state = MediationState.REFUND_PENDING
                session.version += 1
                session.pending_action = PendingAction(
                    kind="request-refund",
                    targetRef=safe_opaque_ref(continuation.continuation_id),
                )
                session.trace.append(
                    self._trace(
                        session,
                        stage="refund-required",
                        component_id="PaymentBridge.execute_approved_payment",
                        layer="payment-bridge",
                        operation_id=operation_id,
                        decision="awaiting-exact-refund-approval",
                        safe_value=continuation.continuation_id,
                    )
                )
                return self.store.compare_and_set(
                    session, expected_version=expected
                )
            remote = result.remote_task
            if (
                remote.task_id != continuation.remote_task.task_id
                or remote.context_id != continuation.remote_task.context_id
                or remote.order_id != continuation.requirement.order_id
                or remote.quote_id != continuation.requirement.quote_id
            ):
                raise SecurityBlocked(
                    "PAYMENT_RESULT_BINDING_MISMATCH",
                    "支払い結果が承認済みTaskまたは見積もりと一致しません。",
                )
            if result.state != "same-task-completed" or remote.state != "completed":
                raise ReviewRequired(
                    "PAYMENT_RESULT_UNKNOWN", "支払い結果を確定できません。"
                )
            artifact = remote.artifact
            try:
                if artifact != project_confirmation_artifact(remote.task_id):
                    raise ValueError("confirmation artifact is invalid")
            except (TypeError, ValueError) as error:
                raise SecurityBlocked(
                    "PAYMENT_CONFIRMATION_INVALID",
                    "完了したデモ予約確認が固定シナリオと一致しません。",
                ) from error
            for summary in result.a2a_executions:
                layers = {
                    "legacy-callback-before": (
                        "LegacyCallbackHook.before",
                        "callback-hook",
                    ),
                    "legacy-callback-after": (
                        "LegacyCallbackHook.after",
                        "callback-hook",
                    ),
                    "response-persisted": (
                        "OperationObserverPort.persist_response",
                        "controller",
                    ),
                    "transport": ("A2ATransportPort.send", "controller"),
                }
                for item in summary.event_order:
                    component, layer = layers.get(
                        item,
                        (
                            "DeterministicStableGate.decide",
                            "deterministic-validator",
                        ),
                    )
                    session.trace.append(
                        self._trace(
                            session,
                            stage=item,
                            component_id=component,
                            layer=layer,
                            operation_id=summary.operation_id,
                            decision="passed",
                            safe_value=summary.task_digest,
                        )
                    )
            session.trace.append(
                self._trace(
                    session,
                    stage="payment-result-bound",
                    component_id="PaymentBridge.execute_approved_payment",
                    layer="payment-bridge",
                    operation_id=operation_id,
                    decision="same-task-completed",
                    safe_value=remote.task_digest,
                )
            )
            return await self._finalize(session, result.result)
        except ReviewRequired:
            return self._terminal_transition(
                session,
                MediationState.REVIEW_REQUIRED,
                "payment-result-review",
                "PaymentBridge.execute_approved_payment",
            )
        except SecurityBlocked:
            return self._terminal_transition(
                session,
                MediationState.BLOCKED,
                "payment-result-blocked",
                "PaymentBridge.execute_approved_payment",
            )
        except Exception:
            return self._terminal_transition(
                session,
                MediationState.REVIEW_REQUIRED,
                "payment-result-unknown",
                "PaymentBridge.execute_approved_payment",
            )

    async def _finalize(
        self, session: MediationSession, result: dict[str, Any]
    ) -> MediationSession:
        decision = await self.final_validator.validate(session, result)
        session.trace.append(
            self._trace(
                session,
                stage="final-validation",
                component_id="LegacyFinalValidationAdapter.validate",
                layer="final-validator",
                operation_id=session.owner.mediation_session_id,
                decision=decision,
                safe_value=canonical_digest(result),
            )
        )
        if decision == "REVIEW":
            return self._terminal_transition(
                session,
                MediationState.REVIEW_REQUIRED,
                "final-validation-review",
                "LegacyFinalValidationAdapter.validate",
            )
        if decision != "ACCEPT":
            return self._terminal_transition(
                session,
                MediationState.BLOCKED,
                "final-validation-blocked",
                "LegacyFinalValidationAdapter.validate",
            )
        expected = session.version
        session.result = result
        session.approval_target_digest = None
        session.state = MediationState.COMPLETED
        session.version += 1
        session.pending_action = PendingAction(kind="none")
        return self.store.compare_and_set(session, expected_version=expected)

    async def _start_refund(self, scope: SubjectScope) -> MediationSession:
        latest = self.store.latest_for(scope)
        if (
            latest is None
            or latest.state != MediationState.COMPLETED
            or latest.continuation is None
            or not isinstance(latest.result, dict)
            or latest.result.get("refundEligible") is not True
        ):
            raise MediationError(
                "REFUND_NOT_AVAILABLE", "返金可能な完了済み支払いはありません。"
            )
        expected = latest.version
        latest.state = MediationState.REFUND_PENDING
        latest.version += 1
        latest.pending_action = PendingAction(
            kind="request-refund",
            targetRef=safe_opaque_ref(latest.continuation.continuation_id),
        )
        latest.trace.append(
            self._trace(
                latest,
                stage="refund-requested",
                component_id="MediationController._start_refund",
                layer="controller",
                operation_id=latest.continuation.continuation_id,
                decision="awaiting-exact-approval",
                safe_value=latest.continuation.continuation_id,
            )
        )
        return self.store.compare_and_set(latest, expected_version=expected)

    async def _execute_refund(self, session: MediationSession) -> MediationSession:
        continuation = session.continuation
        if continuation is None:
            raise SecurityBlocked("CONTINUATION_MISSING", "返金情報が見つかりません。")
        expected = session.version
        operation_id = f"refund:{continuation.continuation_id}:1"
        session.state = MediationState.REFUND_SUBMITTING
        session.version += 1
        session.pending_action = PendingAction(kind="wait")
        session = self.store.compare_and_set(session, expected_version=expected)
        try:
            raw_refund = await maybe_await(
                self.payment_bridge.refund(
                    owner=_bridge_owner(session.owner),
                    operation_id=operation_id,
                    continuation_id=continuation.continuation_id,
                    expected_version=continuation.version,
                )
            )
            refund = RefundResult.model_validate(_wire(raw_refund))
            if refund.state != "refunded":
                raise ReviewRequired("REFUND_RESULT_UNKNOWN", "返金結果を確定できません。")
            expected = session.version
            session.state = MediationState.REFUNDED
            session.version += 1
            session.result = {
                **(session.result or {}),
                "refundEligible": False,
                "refundState": "refunded",
                "refundResultDigest": refund.result_digest,
            }
            session.pending_action = PendingAction(kind="none")
            session.trace.append(
                self._trace(
                    session,
                    stage="refund-completed",
                    component_id="PaymentBridge.refund",
                    layer="payment-bridge",
                    operation_id=operation_id,
                    decision="refunded",
                    safe_value=refund.refund_id,
                )
            )
            return self.store.compare_and_set(session, expected_version=expected)
        except ReviewRequired:
            return self._terminal_transition(
                session,
                MediationState.REVIEW_REQUIRED,
                "refund-result-review",
                "PaymentBridge.refund",
            )
        except SecurityBlocked:
            return self._terminal_transition(
                session,
                MediationState.BLOCKED,
                "refund-result-blocked",
                "PaymentBridge.refund",
            )
        except Exception:
            return self._terminal_transition(
                session,
                MediationState.REVIEW_REQUIRED,
                "refund-result-unknown",
                "PaymentBridge.refund",
            )

    def _terminal_transition(
        self,
        session: MediationSession,
        state: MediationState,
        stage: str,
        component_id: str,
    ) -> MediationSession:
        expected = session.version
        session.approval_target_digest = None
        session.state = state
        session.version += 1
        session.pending_action = PendingAction(kind="none")
        session.trace.append(
            self._trace(
                session,
                stage=stage,
                component_id=component_id,
                layer="controller",
                operation_id=session.owner.mediation_session_id,
                decision=state.value,
                safe_value=session.plan.plan_digest,
            )
        )
        return self.store.compare_and_set(session, expected_version=expected)

    def _append_execution_trace(
        self, session: MediationSession, execution: A2AExecution
    ) -> None:
        layers = {
            "legacy-callback-before": ("LegacyCallbackHook.before", "callback-hook"),
            "legacy-callback-after": ("LegacyCallbackHook.after", "callback-hook"),
            "response-persisted": ("OperationObserverPort.persist_response", "controller"),
            "transport": ("A2ATransportPort.send", "controller"),
        }
        for item in execution.event_order:
            component, layer = layers.get(
                item, ("DeterministicStableGate.decide", "deterministic-validator")
            )
            session.trace.append(
                self._trace(
                    session,
                    stage=item,
                    component_id=component,
                    layer=layer,
                    operation_id=execution.operation.operation_id,
                    decision="passed",
                    safe_value=execution.response.task.task_digest,
                )
            )

    def public_view(self, session: MediationSession) -> MediationPublicView:
        messages = {
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
        continuation = session.continuation
        message = messages[session.state]
        approval_target: PlanApprovalTarget | PaymentApprovalTarget | None = None
        if session.state == MediationState.WAITING_FOR_PLAN_APPROVAL:
            approval_target = build_plan_approval_target(session.plan)
        if session.state == MediationState.WAITING_FOR_PAYMENT_APPROVAL and continuation:
            requirement = continuation.requirement
            approval_target = MediationController._payment_approval_target(session)
            message = paid_payment_approval_message(requirement)
        completion_message = paid_completion_message(session)
        if completion_message is not None:
            message = completion_message
        return MediationPublicView(
            state=session.state,
            version=session.version,
            message=message,
            agentLabel=session.active_step.selected_agent.registry_name,
            planRef=safe_opaque_ref(session.plan.plan_digest),
            stepRef=safe_opaque_ref(session.active_step.step_id),
            taskRef=(
                safe_opaque_ref(continuation.remote_task.task_id)
                if continuation is not None
                else None
            ),
            approvalTarget=approval_target,
            approvalTargetDigest=session.approval_target_digest,
            pendingAction=session.pending_action,
            trace=tuple(session.trace),
            durabilityProfile=self.durability_profile,
        )
