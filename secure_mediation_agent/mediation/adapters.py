"""Typed adapters around the existing matcher, planner, callback, and final checks."""

from __future__ import annotations

import inspect
import json
import base64
import hashlib
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Sequence
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from secure_mediation_agent.subagents.final_anomaly_detection_agent import (
    calculate_overall_safety_score,
    detect_hallucination_chain,
    detect_prompt_injection,
    verify_request_fulfillment,
)
from secure_mediation_agent.subagents.matching_agent import (
    calculate_matching_score,
    rank_agents_by_trust,
    search_agent_store,
)
from secure_mediation_agent.subagents.orchestration_agent import a2a_security_callback
from secure_mediation_agent.subagents.planning_agent import create_structured_plan

from .a2a_executor import A2AOperation
from .canonical import canonical_digest
from .errors import (
    DefinitiveA2ARejection,
    MediationError,
    ReviewRequired,
    SecurityBlocked,
)
from .models import (
    A2AResponseEnvelope,
    GateDecision,
    MediationPlan,
    MediationSession,
    MediationStep,
    OwnerScope,
    PaymentRequirementSnapshot,
    PrivatePaymentMaterial,
    RemoteTaskSnapshot,
    SelectedAgentSnapshot,
    utc_now,
)


SIMULATION_EXTENSION = "urn:secure-a2a:extensions:x402-wire-simulation:v1"

EXACT_A2A_DESTINATIONS = frozenset(
    {
        "http://127.0.0.1:8002/a2a/hotel_agent/.well-known/agent-card.json",
        "http://127.0.0.1:8002/a2a/hotel_agent",
        "http://127.0.0.1:8005/.well-known/agent-card.json",
        "http://127.0.0.1:8005/a2a",
    }
)


KNOWN_AGENT_MAPPINGS: dict[str, dict[str, Any]] = {
    "agent-002": {
        "registry_name": "hotel_agent",
        "card_name": "hotel_agent",
        "skill_id": "hotel_search",
        "card_url": "http://127.0.0.1:8002/a2a/hotel_agent/.well-known/agent-card.json",
        "rpc_endpoint": "http://127.0.0.1:8002/a2a/hotel_agent",
    },
    "agent-005": {
        "registry_name": "paid_booking_agent",
        "card_name": "paid-booking-agent",
        "skill_id": "paid-booking",
        "card_url": "http://127.0.0.1:8005/.well-known/agent-card.json",
        "rpc_endpoint": "http://127.0.0.1:8005/a2a",
        "extension_uri": SIMULATION_EXTENSION,
    },
}


def _validate_exact_destination(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.username or parsed.password or parsed.fragment:
        raise SecurityBlocked("A2A_DESTINATION_INVALID", "The selected A2A URL is invalid.")
    if url in EXACT_A2A_DESTINATIONS:
        return
    raise SecurityBlocked(
        "A2A_DESTINATION_INVALID", "The selected A2A URL is not allowlisted."
    )


class LegacyMatcherAdapter:
    """Uses the real Registry helper and pins a live Card into a strict snapshot."""

    def __init__(self, *, timeout_seconds: float = 5.0, max_card_bytes: int = 262_144) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_card_bytes = max_card_bytes

    async def _live_card(self, exact_url: str) -> dict[str, Any]:
        _validate_exact_destination(exact_url)
        timeout = httpx.Timeout(self.timeout_seconds)
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=False, trust_env=False
        ) as client:
            response = await client.get(exact_url, headers={"Accept": "application/json"})
        if response.is_redirect:
            raise SecurityBlocked("AGENT_CARD_REDIRECT", "Agent Card redirects are not allowed.")
        response.raise_for_status()
        if len(response.content) > self.max_card_bytes:
            raise SecurityBlocked("AGENT_CARD_TOO_LARGE", "The Agent Card is too large.")
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
        if content_type != "application/json":
            raise SecurityBlocked(
                "AGENT_CARD_CONTENT_TYPE", "The Agent Card content type is invalid."
            )
        value = response.json()
        if not isinstance(value, dict):
            raise SecurityBlocked("AGENT_CARD_INVALID", "The Agent Card is invalid.")
        return value

    @staticmethod
    def _goal_prefers_paid(goal: str) -> bool:
        lowered = goal.lower()
        return any(word in lowered for word in ("paid", "payment", "pay", "有料", "支払", "予約購入"))

    async def match(self, goal: str) -> Sequence[SelectedAgentSnapshot]:
        raw_search = json.loads(await search_agent_store(goal))
        entries = raw_search.get("agents")
        if not isinstance(entries, list) or not entries:
            raise ReviewRequired("NO_AGENT_MATCH", "No trusted Agent matched the request.")

        ranked_wire = json.loads(await rank_agents_by_trust(entries, min_trust_score=30.0))
        ranked = ranked_wire.get("ranked_agents")
        if not isinstance(ranked, list):
            raise ReviewRequired("MATCHER_OUTPUT_INVALID", "The Agent matcher output is invalid.")

        prefer_paid = self._goal_prefers_paid(goal)
        ranked.sort(
            key=lambda item: (
                (item.get("agent_id") == "agent-005") == prefer_paid,
                float(item.get("trust_score", 0)),
            ),
            reverse=True,
        )

        snapshots: list[SelectedAgentSnapshot] = []
        for entry in ranked:
            canonical_id = str(entry.get("agent_id") or "")
            mapping = KNOWN_AGENT_MAPPINGS.get(canonical_id)
            if mapping is None:
                continue
            if entry.get("name") != mapping["registry_name"]:
                continue
            card_url = str(entry.get("agentCardUrl") or "")
            rpc_endpoint = str(entry.get("rpcEndpoint") or "")
            if card_url != mapping["card_url"] or rpc_endpoint != mapping["rpc_endpoint"]:
                continue
            card = await self._live_card(card_url)
            if card.get("name") != mapping["card_name"] or card.get("url") != rpc_endpoint:
                raise SecurityBlocked(
                    "AGENT_CARD_BINDING_MISMATCH",
                    "The selected Agent Card did not match its Registry snapshot.",
                )
            skills = card.get("skills")
            if not isinstance(skills, list) or mapping["skill_id"] not in {
                skill.get("id") for skill in skills if isinstance(skill, dict)
            }:
                raise SecurityBlocked(
                    "AGENT_CARD_SKILL_MISMATCH",
                    "The selected Agent Card did not declare the required skill.",
                )
            extensions = card.get("capabilities", {}).get("extensions", [])
            extension_uris = tuple(
                str(extension["uri"])
                for extension in extensions
                if isinstance(extension, dict)
                and isinstance(extension.get("uri"), str)
                and extension.get("required") is True
            )
            expected_extension = mapping.get("extension_uri")
            if expected_extension and extension_uris != (expected_extension,):
                raise SecurityBlocked(
                    "AGENT_CARD_PROFILE_MISMATCH",
                    "The selected payment extension declaration is invalid.",
                )
            if not expected_extension and extension_uris:
                raise SecurityBlocked(
                    "FREE_AGENT_PAYMENT_EXTENSION",
                    "The free Agent unexpectedly declared a required payment extension.",
                )

            await calculate_matching_score(
                entry,
                {"skills": [mapping["skill_id"]], "input_modes": ["text/plain"]},
            )
            card_digest = canonical_digest(card)
            snapshot_wire = {
                "canonicalAgentId": canonical_id,
                "registryName": mapping["registry_name"],
                "a2aAgentName": mapping["card_name"],
                "agentCardUrl": card_url,
                "rpcEndpoint": rpc_endpoint,
                "a2aSkillId": mapping["skill_id"],
                "trustScore": int(entry.get("trust_score", 0)),
                "cardDigest": card_digest,
                # Keep the strict immutable boundary shape.  canonical_digest
                # serializes this tuple to the same JSON array emitted by the
                # validated DTO, so the digest and snapshot cannot diverge.
                "paymentExtensionUris": extension_uris,
            }
            snapshots.append(
                SelectedAgentSnapshot(
                    **snapshot_wire,
                    snapshotDigest=canonical_digest(snapshot_wire),
                )
            )

        if not snapshots:
            raise ReviewRequired("NO_VALID_AGENT", "No valid live Agent Card matched the request.")
        return snapshots


class TypedPlannerAdapter:
    """Calls the legacy planner helper, then validates a digest-bound typed plan."""

    async def create_plan(
        self,
        goal: str,
        owner: OwnerScope,
        candidates: Sequence[SelectedAgentSnapshot],
    ) -> MediationPlan:
        if not candidates:
            raise MediationError("PLAN_NO_AGENT", "A plan requires a selected Agent.")
        candidate_wire = [
            candidate.model_dump(mode="json", by_alias=True, exclude_none=True)
            for candidate in candidates
        ]
        raw = json.loads(await create_structured_plan(goal, candidate_wire))
        raw_steps = raw.get("steps")
        if not isinstance(raw_steps, list) or len(raw_steps) != 1:
            raise MediationError("PLAN_OUTPUT_INVALID", "The typed planner output is invalid.")
        raw_step = raw_steps[0]
        selected = next(
            (
                item
                for item in candidates
                if item.snapshot_digest == raw_step.get("selectedAgentSnapshotDigest")
            ),
            None,
        )
        if selected is None:
            raise SecurityBlocked(
                "PLAN_AGENT_BINDING_MISMATCH",
                "The plan referenced an Agent outside the validated candidate set.",
            )

        now = utc_now()
        plan_id = f"plan-{uuid4()}"
        step_id = f"step-{uuid4()}"
        goal_digest = canonical_digest({"goal": goal})
        step = MediationStep(
            stepId=step_id,
            ordinal=1,
            selectedAgent=selected,
            inputDigest=goal_digest,
            goal=goal,
            paymentLimitMinor=5000,
            currency="USD" if selected.canonical_agent_id == "agent-005" else "JPY",
        )
        digest_input = {
            "schemaVersion": "mediation-plan-snapshot/1",
            "planId": plan_id,
            "planVersion": 1,
            "goalDigest": goal_digest,
            "owner": owner.model_dump(mode="json", by_alias=True),
            "steps": [step.model_dump(mode="json", by_alias=True)],
            "createdAt": now.isoformat(),
            "expiresAt": (now + timedelta(minutes=10)).isoformat(),
        }
        return MediationPlan(
            planId=plan_id,
            planVersion=1,
            planDigest=canonical_digest(digest_input),
            goalDigest=goal_digest,
            owner=owner,
            steps=(step,),
            createdAt=now,
            expiresAt=now + timedelta(minutes=10),
        )


class _ToolName:
    name = "execute_plan_step"


class LegacyCallbackHook:
    """Runs the same existing callback for both before and after phases."""

    def __init__(
        self,
        callback: Callable[..., Awaitable[dict[str, Any] | None]] = a2a_security_callback,
    ) -> None:
        self.callback = callback

    async def _run(
        self,
        phase: str,
        operation: A2AOperation,
        response: RemoteTaskSnapshot | None,
    ) -> None:
        args = {
            "agent_name": operation.agent.a2a_agent_name,
            "planned_agent": operation.agent.a2a_agent_name,
            "task": operation.kind,
            "plan_context": {
                "operationId": operation.operation_id,
                "phase": phase,
                "requestDigest": operation.request_digest,
            },
        }
        safe_response: dict[str, Any] = {
            "success": True,
            "output": {
                "phase": phase,
                "taskDigest": response.task_digest if response else None,
            },
        }
        result = await self.callback(_ToolName(), args, SimpleNamespace(), safe_response)
        if isinstance(result, dict) and (
            result.get("security_blocked")
            or result.get("security_error")
            or result.get("success") is False
        ):
            raise SecurityBlocked(
                "LEGACY_CALLBACK_BLOCKED",
                "The legacy A2A security callback rejected the operation.",
            )

    async def before(self, operation: A2AOperation) -> None:
        await self._run("before", operation, None)

    async def after(
        self, operation: A2AOperation, response: RemoteTaskSnapshot
    ) -> None:
        await self._run("after", operation, response)


class LocalDeterministicCallbackHook:
    """Fail-closed structural callback for the explicit local release profile.

    This hook deliberately has no model, network, or credential dependency.  It
    receives the same operation and parsed remote Task as the live callback and
    enforces the bindings which are stable for the packaged local simulation.
    Production composition selects it only through an explicit local-only mode.
    """

    @staticmethod
    def _reject(code: str, message: str) -> None:
        raise SecurityBlocked(code, message)

    @classmethod
    def _validate_operation(cls, operation: A2AOperation) -> None:
        _validate_exact_destination(operation.agent.rpc_endpoint)
        if canonical_digest(operation.request) != operation.request_digest:
            cls._reject(
                "LOCAL_CALLBACK_REQUEST_DIGEST_MISMATCH",
                "The local callback rejected a changed A2A request.",
            )
        request = operation.request
        if (
            request.get("jsonrpc") != "2.0"
            or request.get("id") != operation.operation_id
            or request.get("method") != "message/send"
            or not isinstance(request.get("params"), dict)
        ):
            cls._reject(
                "LOCAL_CALLBACK_ENVELOPE_INVALID",
                "The local callback rejected an invalid A2A envelope.",
            )
        params = request["params"]
        message = params.get("message")
        if not isinstance(message, dict):
            cls._reject(
                "LOCAL_CALLBACK_MESSAGE_INVALID",
                "The local callback requires a typed A2A message.",
            )
        if operation.kind == "task-start":
            if operation.task_id is not None or operation.context_id is not None:
                cls._reject(
                    "LOCAL_CALLBACK_START_BINDING_INVALID",
                    "The local callback rejected unexpected start bindings.",
                )
            if operation.agent.payment_extension_uris:
                required = {
                    "action",
                    "workflowId",
                    "taskId",
                    "orderId",
                    "contextId",
                    "capabilityId",
                }
                if params.get("action") != "merchant-task:start" or not required.issubset(
                    params
                ):
                    cls._reject(
                        "LOCAL_CALLBACK_PAID_START_INVALID",
                        "The local callback rejected an incomplete paid Task start.",
                    )
                if not operation.authorization:
                    cls._reject(
                        "LOCAL_CALLBACK_CAPABILITY_MISSING",
                        "The local callback requires a signed Merchant capability.",
                    )
            elif operation.authorization is not None:
                cls._reject(
                    "LOCAL_CALLBACK_UNEXPECTED_CAPABILITY",
                    "The local callback rejected a capability on a free Task.",
                )
        elif operation.kind == "payment-submit":
            if (
                not operation.task_id
                or not operation.context_id
                or params.get("taskId") != operation.task_id
                or not operation.authorization
            ):
                cls._reject(
                    "LOCAL_CALLBACK_PAYMENT_BINDING_INVALID",
                    "The local callback rejected an unbound payment operation.",
                )

    async def before(self, operation: A2AOperation) -> None:
        self._validate_operation(operation)

    async def after(
        self, operation: A2AOperation, response: RemoteTaskSnapshot
    ) -> None:
        self._validate_operation(operation)
        if operation.kind == "payment-submit":
            if (
                response.task_id != operation.task_id
                or response.context_id != operation.context_id
            ):
                self._reject(
                    "LOCAL_CALLBACK_RESPONSE_BINDING_MISMATCH",
                    "The local callback rejected a Task binding mismatch.",
                )
            return
        if operation.agent.payment_extension_uris:
            if (
                response.state != "input-required"
                or response.payment_requirement is None
            ):
                self._reject(
                    "LOCAL_CALLBACK_PAID_RESPONSE_INVALID",
                    "The local callback rejected an invalid payment requirement.",
                )
        elif response.state != "completed" or response.payment_requirement is not None:
            self._reject(
                "LOCAL_CALLBACK_FREE_RESPONSE_INVALID",
                "The local callback rejected an invalid free Task result.",
            )


class DeterministicStableGate:
    """Minimal deterministic Release-1 gate; unknown structure never passes."""

    async def decide(
        self,
        gate_id: str,
        operation: A2AOperation,
        response: RemoteTaskSnapshot | None,
    ) -> GateDecision:
        allowed = {
            "PRE_A2A_START",
            "POST_A2A_RESPONSE",
            "POST_PAYMENT_REQUIREMENT",
            "PRE_PAYMENT_SUBMIT",
            "POST_PAYMENT_RESULT",
        }
        decision = "PASS"
        if gate_id not in allowed:
            decision = "BLOCK"
        elif gate_id.startswith("POST_") and response is None:
            decision = "BLOCK"
        elif gate_id == "POST_PAYMENT_REQUIREMENT" and (
            response is None or response.payment_requirement is None
        ):
            decision = "BLOCK"
        return GateDecision(
            gateId=gate_id,
            decision=decision,
            decisionDigest=canonical_digest(
                {
                    "gateId": gate_id,
                    "operationId": operation.operation_id,
                    "requestDigest": operation.request_digest,
                    "responseDigest": response.task_digest if response else None,
                    "decision": decision,
                }
            ),
        )


class HttpxA2ATransport:
    """Bounded exact-endpoint JSON-RPC transport returning a structured Task."""

    def __init__(self, *, timeout_seconds: float = 15.0, max_response_bytes: int = 1_048_576) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes

    @staticmethod
    def _task_from_result(operation: A2AOperation, result: dict[str, Any]) -> A2AResponseEnvelope:
        task = result.get("task", result)
        if not isinstance(task, dict):
            raise SecurityBlocked("A2A_TASK_INVALID", "The Agent returned an invalid Task.")
        status = task.get("status")
        if not isinstance(status, dict) or not isinstance(status.get("state"), str):
            raise SecurityBlocked("A2A_TASK_INVALID", "The Agent Task status is invalid.")
        state = status["state"]
        message = status.get("message") or {}
        metadata = message.get("metadata") if isinstance(message, dict) else None
        metadata = metadata if isinstance(metadata, dict) else {}
        task_id = task.get("id")
        context_id = task.get("contextId")
        if not isinstance(task_id, str) or not isinstance(context_id, str):
            raise SecurityBlocked("A2A_TASK_INVALID", "The Agent Task correlation is invalid.")

        payment_requirement = None
        status_marker = metadata.get("x402.payment.status")
        required = metadata.get("x402.payment.required")
        project = metadata.get("io.github.taichihiromatsu.secure-mediation.v1")
        if status_marker == "payment-required" or required is not None:
            if (
                state != "input-required"
                or status_marker != "payment-required"
                or not isinstance(required, dict)
                or not isinstance(project, dict)
            ):
                raise SecurityBlocked(
                    "PAYMENT_REQUIRED_INVALID",
                    "The payment requirement metadata was incomplete.",
                )
            accepts = required.get("accepts")
            if not isinstance(accepts, list) or len(accepts) != 1 or not isinstance(accepts[0], dict):
                raise SecurityBlocked(
                    "PAYMENT_REQUIRED_INVALID", "The payment options were invalid."
                )
            option = accepts[0]
            amount_value = option.get("maxAmountRequired") or required.get("maxAmountRequired")
            try:
                amount_minor = int(amount_value)
            except (TypeError, ValueError) as error:
                raise SecurityBlocked(
                    "PAYMENT_REQUIRED_INVALID", "The payment amount was invalid."
                ) from error
            order_id = project.get("orderId") or required.get("orderId")
            quote_id = project.get("quoteId") or required.get("quoteId")
            checkout_digest = project.get("checkoutJwtDigest") or required.get("checkoutDigest")
            checkout_challenge = project.get("checkoutMandateChallenge")
            payment_challenge = project.get("paymentMandateChallenge")
            if not isinstance(checkout_challenge, dict) or not isinstance(payment_challenge, dict):
                raise SecurityBlocked(
                    "PAYMENT_REQUIRED_INVALID", "The payment mandate challenges were invalid."
                )
            profile_id = project.get("profile")
            expires_value = project.get("expiresAt") or required.get("expiresAt")
            if isinstance(expires_value, str):
                try:
                    expires_value = datetime.fromisoformat(
                        expires_value.replace("Z", "+00:00")
                    )
                except ValueError as error:
                    raise SecurityBlocked(
                        "PAYMENT_REQUIRED_INVALID", "The payment expiry was invalid."
                    ) from error
            extension_uri = operation.agent.payment_extension_uris[0] if operation.agent.payment_extension_uris else ""
            raw_requirement = {
                "taskState": "input-required",
                "paymentStatus": "payment-required",
                "extensionUri": extension_uri,
                "profileId": profile_id,
                "orderId": order_id,
                "quoteId": quote_id,
                "amountMinor": amount_minor,
                "currency": option.get("asset") or project.get("currency"),
                "payee": project.get("payee")
                or str(option.get("payTo") or "").removeprefix("merchant:"),
                "expiresAt": expires_value,
                "checkoutDigest": checkout_digest,
                "paymentRequired": required,
                "checkoutAudience": checkout_challenge.get("aud"),
                "checkoutNonce": checkout_challenge.get("nonce"),
                "paymentAudience": payment_challenge.get("aud"),
                "paymentNonce": payment_challenge.get("nonce"),
            }
            raw_requirement["requirementDigest"] = canonical_digest(required)
            payment_requirement = PaymentRequirementSnapshot.model_validate(raw_requirement)

        order_id = None
        quote_id = None
        if payment_requirement:
            order_id = payment_requirement.order_id
            quote_id = payment_requirement.quote_id
        else:
            project = metadata.get("io.github.taichihiromatsu.secure-mediation.v1")
            if isinstance(project, dict):
                order_id = project.get("orderId")
                quote_id = project.get("quoteId")
        task_digest = canonical_digest(task)
        artifacts = task.get("artifacts")
        artifact = None
        if isinstance(artifacts, list) and artifacts and isinstance(artifacts[0], dict):
            artifact = artifacts[0]
        task_snapshot = RemoteTaskSnapshot(
            taskId=task_id,
            contextId=context_id,
            state=state,
            taskDigest=task_digest,
            orderId=order_id,
            quoteId=quote_id,
            paymentRequirement=payment_requirement,
            artifact=artifact,
        )
        private_material = None
        private_wire = result.get("privatePaymentMaterial")
        if private_wire is None and (
            result.get("checkoutJwt") is not None
            or result.get("checkoutHash") is not None
        ):
            private_wire = {
                "checkoutJwt": result.get("checkoutJwt"),
                "checkoutHash": result.get("checkoutHash"),
            }
        if payment_requirement is not None:
            if not isinstance(private_wire, dict):
                raise SecurityBlocked(
                    "PRIVATE_PAYMENT_MATERIAL_MISSING",
                    "The private payment material was not returned in the internal envelope.",
                )
            private_material = PrivatePaymentMaterial.model_validate(private_wire)
            observed_hash = base64.urlsafe_b64encode(
                hashlib.sha256(private_material.checkout_jwt.encode("utf-8")).digest()
            ).rstrip(b"=").decode("ascii")
            if observed_hash != private_material.checkout_hash:
                raise SecurityBlocked(
                    "CHECKOUT_HASH_MISMATCH", "The private Checkout hash was invalid."
                )
            if canonical_digest(private_material.checkout_jwt) != payment_requirement.checkout_digest:
                raise SecurityBlocked(
                    "CHECKOUT_DIGEST_MISMATCH", "The Checkout digest did not match the Task."
                )
        elif private_wire is not None:
            raise SecurityBlocked(
                "UNEXPECTED_PRIVATE_PAYMENT_MATERIAL",
                "A free Task returned unexpected private payment material.",
            )
        return A2AResponseEnvelope(
            task=task_snapshot,
            privatePaymentMaterial=private_material,
            envelopeDigest=canonical_digest(result),
        )

    async def send(self, operation: A2AOperation) -> A2AResponseEnvelope:
        _validate_exact_destination(operation.agent.rpc_endpoint)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Idempotency-Key": operation.idempotency_key,
        }
        if operation.agent.payment_extension_uris:
            headers["X-A2A-Extensions"] = operation.agent.payment_extension_uris[0]
        if operation.authorization:
            headers["Authorization"] = f"Bearer {operation.authorization}"
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
                follow_redirects=False,
                trust_env=False,
            ) as client:
                response = await client.post(
                    operation.agent.rpc_endpoint,
                    json=operation.request,
                    headers=headers,
                )
        except httpx.TimeoutException as error:
            raise TimeoutError from error
        if response.is_redirect:
            raise SecurityBlocked("A2A_REDIRECT", "A2A redirects are not allowed.")
        if 400 <= response.status_code < 500:
            raise DefinitiveA2ARejection(
                "A2A_REQUEST_REJECTED", "The remote Agent rejected the request."
            )
        response.raise_for_status()
        if len(response.content) > self.max_response_bytes:
            raise SecurityBlocked("A2A_RESPONSE_TOO_LARGE", "The A2A response is too large.")
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
        if content_type != "application/json":
            raise SecurityBlocked("A2A_CONTENT_TYPE", "The A2A response type is invalid.")
        wire = response.json()
        if not isinstance(wire, dict) or wire.get("jsonrpc") != "2.0" or "result" not in wire:
            raise SecurityBlocked("A2A_RESPONSE_INVALID", "The A2A response envelope is invalid.")
        result = wire["result"]
        if not isinstance(result, dict):
            raise SecurityBlocked("A2A_RESPONSE_INVALID", "The A2A result is invalid.")
        if (
            operation.kind == "task-start"
            and operation.agent.payment_extension_uris
            and result.get("activationEcho")
            != operation.agent.payment_extension_uris[0]
        ):
            raise SecurityBlocked(
                "PAYMENT_ACTIVATION_ECHO_MISMATCH",
                "The Agent did not echo the selected payment extension.",
            )
        return self._task_from_result(operation, result)


class LegacyFinalValidationAdapter:
    """Runs the real deterministic final helper symbols and returns a stable decision."""

    async def validate(self, session: MediationSession, result: dict[str, Any]) -> str:
        plan = session.plan.model_dump(mode="json", by_alias=True)
        plan["steps"] = [
            {**step, "status": "completed"}
            for step in plan.get("steps", [])
        ]
        fulfillment = json.loads(await verify_request_fulfillment(session.goal, result, plan))
        history = [
            {"input": session.goal, "output": result, "step_id": session.active_step.step_id}
        ]
        injection = json.loads(await detect_prompt_injection(history))
        hallucination = json.loads(
            await detect_hallucination_chain(
                [{"agent_name": session.active_step.selected_agent.a2a_agent_name, "input": session.goal, "output": result}]
            )
        )
        assessment = json.loads(
            await calculate_overall_safety_score(fulfillment, injection, hallucination)
        )
        if assessment.get("critical_issues"):
            return "REJECT"
        level = assessment.get("safety_level")
        if level == "SAFE":
            return "ACCEPT"
        if level == "MODERATE":
            return "REVIEW"
        return "REJECT"


async def maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value
    PrivatePaymentMaterial,
