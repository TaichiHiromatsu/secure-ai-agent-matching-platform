"""Paid booking Merchant for the approved single-product simulation."""

from __future__ import annotations

import base64
import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentSkill,
    Artifact,
    DataPart,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
from ap2.sdk.jwt_helper import create_jwt, verify_jwt

from secure_mediation_agent.ap2.keys import DemoKeySet, public_key
from secure_mediation_agent.ap2.verification import b64url_sha256, verify_terminal_presentation
from secure_mediation_agent.demo_catalog import (
    PRODUCT_ID,
    confirmation_reference,
    project_confirmation,
    project_payment_requirement,
    scenario_digest,
    validate_payment_requirement,
)
from secure_mediation_agent.payment_profiles.a2a import (
    PAYMENT_PAYLOAD,
    PAYMENT_STATUS,
    PROJECT_METADATA,
    final_task_metadata,
    payment_required_task,
)
from secure_mediation_agent.payment_profiles.simulation_v1 import SimulationV1Profile
from secure_mediation_agent.workflow.canonical import (
    canonical_digest,
    canonical_json,
    sha256_digest,
)
from secure_mediation_agent.workflow.errors import DomainError
from secure_mediation_agent.workflow.repository import WorkflowRepository, utc_now

from .fault_injection import FulfillmentFaultTarget, MerchantTestFaults


@dataclass(frozen=True, slots=True)
class MerchantStartResult:
    task: Task
    checkout_jwt: str
    checkout_hash: str
    requirements: dict[str, Any]
    activation_echo: str
    checkout_challenge: str
    payment_challenge: str


class PaidBookingMerchant:
    merchant_id = "demo-merchant"
    merchant_name = "Demo Merchant"

    def __init__(
        self,
        repository: WorkflowRepository,
        keys: DemoKeySet,
        profile: SimulationV1Profile,
        *,
        test_faults: MerchantTestFaults | None = None,
    ) -> None:
        self._repository = repository
        self._keys = keys
        self._profile = profile
        self._test_faults = test_faults

    def configure_test_faults(self, test_faults: MerchantTestFaults | None) -> None:
        self._test_faults = test_faults

    def arm_test_fulfillment_rejection(
        self, target: FulfillmentFaultTarget
    ) -> bool:
        try:
            target.validate()
        except ValueError as error:
            raise DomainError(
                "TEST_FAULT_TARGET_MISMATCH",
                "The Merchant test fault target is invalid.",
                target.task_id,
            ) from error
        try:
            task = self._repository.merchant_task(target.task_id)
        except KeyError as error:
            raise DomainError(
                "TEST_FAULT_TARGET_MISMATCH",
                "The Merchant test fault target does not exist.",
                target.task_id,
            ) from error
        if task["order_id"] != target.order_id:
            raise DomainError(
                "TEST_FAULT_TARGET_MISMATCH",
                "The Merchant test fault target does not match the order.",
                target.task_id,
            )
        if self._test_faults is None:
            raise DomainError(
                "TEST_FAULTS_DISABLED",
                "Merchant test faults are disabled.",
                target.task_id,
            )
        try:
            return self._test_faults.arm(target)
        except (RuntimeError, ValueError) as error:
            raise DomainError(
                "TEST_FAULT_CONFLICT",
                "The Merchant test fault could not be armed.",
                target.task_id,
            ) from error

    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="paid-booking-agent",
            description=(
                "Tokyo business-trip hotel arrangement simulation for 2026-09-12 "
                "through 2026-09-14, two guests. The 12.50 USD arrangement fee "
                "excludes lodging; no real booking or payment."
            ),
            url="http://127.0.0.1:8005/a2a",
            version="2.0.0-simulation",
            protocolVersion="0.3.0",
            capabilities=AgentCapabilities(
                extensions=[
                    AgentExtension(
                        uri=self._profile.extension_uri,
                        required=True,
                        params={
                            "profile": self._profile.profile_id,
                            "simulated": True,
                            "conformance": "NOT_CONFORMANT",
                            "scheme": self._profile.scheme,
                            "network": self._profile.network,
                            "asset": self._profile.asset,
                        },
                    )
                ]
            ),
            skills=[
                AgentSkill(
                    id="paid-booking",
                    name="Tokyo business hotel arrangement simulation",
                    description=(
                        "Issue one simulated confirmation for the fixed Tokyo "
                        "business-trip scenario after the 12.50 USD arrangement-fee "
                        "simulation; lodging is excluded and no real booking occurs."
                    ),
                    tags=["tokyo", "business-trip", "hotel-arrangement", "payment", "simulation"],
                )
            ],
            defaultInputModes=["text/plain", "application/json"],
            defaultOutputModes=["text/plain", "application/json"],
            supportsAuthenticatedExtendedCard=False,
        )

    def start_task(
        self,
        *,
        workflow_id: str,
        plan_digest: str,
        task_id: str,
        order_id: str,
        context_id: str,
        capability_id: str,
        activation: set[str],
        issued_at: int,
        expires_at: int,
        capability_token: str | None = None,
    ) -> MerchantStartResult:
        self._profile.validate_activation(activation)
        try:
            existing = self._repository.merchant_task(task_id)
        except KeyError:
            existing = None
        if existing is not None:
            task = Task.model_validate(existing["task"])
            requirement = existing["requirement"]
            project = task.status.message.metadata[PROJECT_METADATA]
            return MerchantStartResult(
                task=task,
                checkout_jwt=requirement["checkout_jwt"],
                checkout_hash=requirement["checkout_hash"],
                requirements=requirement["requirements"],
                activation_echo=self._profile.extension_uri,
                checkout_challenge=project["checkoutMandateChallenge"]["nonce"],
                payment_challenge=project["paymentMandateChallenge"]["nonce"],
            )
        quote_id = f"quote:{order_id}"
        expires_at_text = (
            datetime.fromtimestamp(expires_at, UTC)
            .isoformat()
            .replace("+00:00", "Z")
        )
        requirements = {
            **project_payment_requirement(self._profile.build_required(amount=1250)),
            "orderId": order_id,
            "quoteId": quote_id,
            "expiresAt": expires_at_text,
        }
        checkout_challenge = secrets.token_urlsafe(32)
        payment_challenge = secrets.token_urlsafe(32)
        checkout_jwt = create_jwt(
            {
                "alg": "ES256",
                "kid": self._keys.merchant.get("kid"),
                "typ": "JWT",
            },
            {
                "iss": self.merchant_id,
                "aud": "secure-mediation-workflow",
                "jti": f"checkout:{order_id}",
                "checkoutNonce": secrets.token_urlsafe(32),
                "workflowId": workflow_id,
                "planDigest": plan_digest,
                "orderId": order_id,
                "quoteId": quote_id,
                "taskId": task_id,
                "merchantId": self.merchant_id,
                "productId": PRODUCT_ID,
                "scenarioDigest": scenario_digest(),
                "quantity": 1,
                "amount": 1250,
                "currency": "USD",
                "decimals": 2,
                "feePolicyVersion": "zero-fee-v1",
                "iat": issued_at,
                "exp": expires_at,
            },
            self._keys.merchant,
        )
        checkout_hash = b64url_sha256(checkout_jwt)
        project = {
            "profile": self._profile.profile_id,
            "simulated": True,
            "conformance": "NOT_CONFORMANT",
            "workflowId": workflow_id,
            "planDigest": plan_digest,
            "orderId": order_id,
            "quoteId": quote_id,
            "expiresAt": expires_at_text,
            "checkoutJwtDigest": canonical_digest(checkout_jwt),
            "scenarioDigest": scenario_digest(),
            "checkoutMandateChallenge": {
                "aud": self.merchant_id,
                "nonce": checkout_challenge,
            },
            "paymentMandateChallenge": {
                "aud": "demo-credential-provider",
                "nonce": payment_challenge,
            },
        }
        task = payment_required_task(
            task_id=task_id,
            context_id=context_id,
            message_id=f"message:payment-required:{task_id}",
            required=requirements,
            project=project,
        )
        task_wire = task.model_dump(mode="json", by_alias=True, exclude_none=True)
        self._repository.save_merchant_origin(
            workflow_id=workflow_id,
            task_id=task_id,
            context_id=context_id,
            order_id=order_id,
            task=task_wire,
            requirements_id=f"requirements:{task_id}",
            requirements=requirements,
            checkout_jwt=checkout_jwt,
            checkout_hash=checkout_hash,
            capability_id=capability_id,
        )
        return MerchantStartResult(
            task=task,
            checkout_jwt=checkout_jwt,
            checkout_hash=checkout_hash,
            requirements=requirements,
            activation_echo=self._profile.extension_uri,
            checkout_challenge=checkout_challenge,
            payment_challenge=payment_challenge,
        )

    def verify_checkout(
        self,
        token: str,
        *,
        workflow_id: str,
        plan_digest: str,
        task_id: str,
        capability_token: str | None = None,
    ) -> dict[str, Any]:
        claims = verify_jwt(token, public_key(self._keys.merchant))
        expected = {
            "iss": self.merchant_id,
            "aud": "secure-mediation-workflow",
            "workflowId": workflow_id,
            "planDigest": plan_digest,
            "taskId": task_id,
            "merchantId": self.merchant_id,
            "productId": PRODUCT_ID,
            "scenarioDigest": scenario_digest(),
            "quantity": 1,
            "amount": 1250,
            "currency": "USD",
            "decimals": 2,
            "feePolicyVersion": "zero-fee-v1",
        }
        for name, value in expected.items():
            if claims.get(name) != value:
                raise ValueError(f"Checkout constraint mismatch: {name}")
        if not claims.get("checkoutNonce"):
            raise ValueError("Checkout entropy missing")
        if claims.get("quoteId") != f"quote:{claims.get('orderId')}":
            raise ValueError("Checkout quote binding mismatch")
        return claims

    def verify_checkout_mandate(
        self,
        presentation: str,
        *,
        checkout_jwt: str,
        nonce: str,
    ) -> dict[str, Any]:
        payload = verify_terminal_presentation(
            presentation,
            root_key=self._keys.user_root,
            audience=self.merchant_id,
            nonce=nonce,
            expected_vct="mandate.checkout.1",
        )
        if payload.get("checkout_jwt") != checkout_jwt:
            raise ValueError("Checkout Mandate exact JWT mismatch")
        if payload.get("checkout_hash") != b64url_sha256(checkout_jwt):
            raise ValueError("Checkout Mandate hash mismatch")
        return payload

    def submit_payment(
        self,
        *,
        message: Message,
        checkout_mandate: str,
        checkout_jwt: str,
        checkout_nonce: str,
        capability_id: str,
        capability_token: str | None = None,
        workflow_id: str,
        order_id: str,
    ) -> None:
        self.verify_checkout_mandate(
            checkout_mandate,
            checkout_jwt=checkout_jwt,
            nonce=checkout_nonce,
        )
        self._repository.append_merchant_message(
            message_id=message.message_id,
            task_id=message.task_id,
            context_id=message.context_id,
            status="payment-submitted",
            message=message.model_dump(mode="json", by_alias=True, exclude_none=True),
        )

    def accept_guarantee(self, *, message: Message) -> Task:
        """Verify a signed guarantee before exposing fulfillment readiness."""

        if not message.task_id or not message.context_id:
            raise DomainError(
                "X402_TASK_CORRELATION_MISMATCH",
                "A payment guarantee must name the existing Task and context.",
                message.message_id,
            )
        metadata = message.metadata or {}
        payload = metadata.get(PAYMENT_PAYLOAD)
        if metadata.get(PAYMENT_STATUS) != "payment-submitted" or not isinstance(
            payload, dict
        ):
            raise DomainError(
                "PAYMENT_GUARANTEE_INVALID",
                "The signed payment guarantee payload is missing.",
                message.task_id,
            )
        if set(payload) != {
            "schemaVersion",
            "profileId",
            "paymentGuarantee",
            "paymentGuaranteeDigest",
            "ap2Evidence",
        }:
            raise DomainError(
                "PAYMENT_GUARANTEE_INVALID",
                "The signed payment guarantee envelope is malformed.",
                message.task_id,
            )
        if (
            payload.get("schemaVersion") != "merchant-payment-guarantee-submission/1"
            or payload.get("profileId") != self._profile.profile_id
        ):
            raise DomainError(
                "PAYMENT_GUARANTEE_INVALID",
                "The signed payment guarantee profile is unsupported.",
                message.task_id,
            )
        guarantee = payload.get("paymentGuarantee")
        if not isinstance(guarantee, str) or payload.get(
            "paymentGuaranteeDigest"
        ) != sha256_digest(guarantee):
            raise DomainError(
                "PAYMENT_GUARANTEE_INVALID",
                "The signed payment guarantee digest does not match.",
                message.task_id,
            )
        ap2_evidence = payload.get("ap2Evidence")
        if not isinstance(ap2_evidence, dict) or set(ap2_evidence) != {
            "checkoutMandateDigest",
            "paymentMandateDigest",
            "authorizationEnvelopeDigest",
        }:
            raise DomainError(
                "PAYMENT_GUARANTEE_INVALID",
                "The AP2 evidence binding is malformed.",
                message.task_id,
            )
        project = metadata.get(PROJECT_METADATA)
        if not isinstance(project, dict):
            raise DomainError(
                "PAYMENT_GUARANTEE_INVALID",
                "The payment correlation metadata is missing.",
                message.task_id,
            )
        try:
            merchant_task = self._repository.merchant_task(message.task_id)
        except KeyError as error:
            raise DomainError(
                "X402_TASK_CORRELATION_MISMATCH",
                "The signed guarantee does not belong to a Merchant Task.",
                message.task_id,
            ) from error
        if (
            merchant_task["context_id"] != message.context_id
            or merchant_task["order_id"] != project.get("orderId")
        ):
            raise DomainError(
                "X402_TASK_CORRELATION_MISMATCH",
                "The signed guarantee correlation does not match the Merchant Task.",
                message.task_id,
            )
        expected = {
            "taskId": message.task_id,
            "contextId": message.context_id,
            "orderId": merchant_task["order_id"],
            "quoteId": project.get("quoteId"),
            "amountMinor": 1250,
            "currency": "USD",
            "payee": self.merchant_id,
            "paymentMandateDigest": ap2_evidence["paymentMandateDigest"],
            "authorizationEnvelopeDigest": ap2_evidence[
                "authorizationEnvelopeDigest"
            ],
        }
        try:
            claims = self._profile.verify_guarantee(
                guarantee,
                public_key(self._keys.simulation_signer),
                expected=expected,
            )
        except Exception as error:
            raise DomainError(
                "PAYMENT_GUARANTEE_INVALID",
                "The Merchant could not verify the signed payment guarantee.",
                message.task_id,
            ) from error
        now = int(datetime.now(UTC).timestamp())
        if (
            claims.get("guaranteeId") != claims.get("jti")
            or not claims.get("settlementCommitmentId")
            or not isinstance(claims.get("nbf"), int)
            or claims["nbf"] > now + 30
            or claims.get("exp", 0) <= now
        ):
            raise DomainError(
                "PAYMENT_GUARANTEE_INVALID",
                "The signed payment guarantee identity or lifetime is invalid.",
                message.task_id,
            )
        previous = Task.model_validate(merchant_task["task"])
        previous_metadata = (
            previous.status.message.metadata
            if previous.status.message is not None
            and previous.status.message.metadata is not None
            else {}
        )
        previous_project = previous_metadata.get(PROJECT_METADATA)
        previous_project = previous_project if isinstance(previous_project, dict) else {}
        response = Message(
            messageId=f"message:payment-guaranteed:{message.task_id}",
            taskId=message.task_id,
            contextId=message.context_id,
            role=Role.agent,
            parts=[Part(root=TextPart(text="Signed payment guarantee accepted."))],
            metadata={
                PAYMENT_STATUS: "payment-guaranteed",
                PROJECT_METADATA: {
                    "profile": self._profile.profile_id,
                    "simulated": True,
                    "state": "GUARANTEED",
                    "workflowId": merchant_task["workflow_id"],
                    "planDigest": previous_project.get("planDigest"),
                    "orderId": merchant_task["order_id"],
                    "quoteId": claims["quoteId"],
                    "guaranteeId": claims["guaranteeId"],
                    "paymentGuaranteeDigest": payload["paymentGuaranteeDigest"],
                },
            },
        )
        accepted = Task(
            id=message.task_id,
            contextId=message.context_id,
            status=TaskStatus(state=TaskState.working, message=response),
            artifacts=previous.artifacts,
            history=[*(previous.history or []), message, response],
        )
        wire = message.model_dump(mode="json", by_alias=True, exclude_none=True)
        accepted_wire = accepted.model_dump(mode="json", by_alias=True, exclude_none=True)
        request_digest = canonical_digest(wire)
        with self._repository.merchant_transaction() as conn:
            existing = conn.execute(
                "SELECT request_digest FROM merchant_guarantees_v3 WHERE task_id=?",
                (message.task_id,),
            ).fetchone()
            if existing is not None and existing["request_digest"] != request_digest:
                raise DomainError(
                    "IDEMPOTENCY_CONFLICT",
                    "A different signed guarantee already owns this Merchant Task.",
                    message.task_id,
                )
            if existing is None:
                if merchant_task["state"] not in {"input-required", "working"}:
                    raise DomainError(
                        "STATE_TRANSITION_CONFLICT",
                        "The Merchant Task is not eligible for a new payment guarantee.",
                        message.task_id,
                        current_state=merchant_task["state"],
                    )
                conn.execute(
                    "INSERT INTO merchant_guarantees_v3(guarantee_id,task_id,context_id,"
                    "order_id,quote_id,guarantee_jwt,guarantee_digest,"
                    "authorization_envelope_digest,payment_mandate_digest,request_digest,"
                    "state,accepted_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,'accepted',?,?)",
                    (
                        claims["guaranteeId"],
                        message.task_id,
                        message.context_id,
                        merchant_task["order_id"],
                        claims["quoteId"],
                        guarantee,
                        payload["paymentGuaranteeDigest"],
                        ap2_evidence["authorizationEnvelopeDigest"],
                        ap2_evidence["paymentMandateDigest"],
                        request_digest,
                        utc_now(),
                        utc_now(),
                    ),
                )
                conn.execute(
                    "INSERT INTO merchant_messages_v2(message_id,task_id,context_id,status,"
                    "message_json,request_digest,created_at) VALUES(?,?,?,?,?,?,?)",
                    (
                        message.message_id,
                        message.task_id,
                        message.context_id,
                        "payment-submitted",
                        canonical_json(wire),
                        request_digest,
                        utc_now(),
                    ),
                )
                conn.execute(
                    "UPDATE merchant_tasks_v2 SET state='working',task_json=?,version=version+1,"
                    "updated_at=? WHERE task_id=? AND context_id=?",
                    (
                        canonical_json(accepted_wire),
                        utc_now(),
                        message.task_id,
                        message.context_id,
                    ),
                )
            else:
                current = conn.execute(
                    "SELECT task_json FROM merchant_tasks_v2 WHERE task_id=?",
                    (message.task_id,),
                ).fetchone()
                return Task.model_validate(json.loads(current["task_json"]))
        return accepted

    def commit_guaranteed_fulfillment(
        self,
        *,
        message: Message,
        operation_id: str | None = None,
        order_id: str | None = None,
    ) -> Task:
        """Commit only after this Merchant accepted the exact signed guarantee."""

        metadata = message.metadata or {}
        payload = metadata.get(PAYMENT_PAYLOAD)
        if metadata.get(PAYMENT_STATUS) != "payment-settled" or not isinstance(
            payload, dict
        ):
            raise DomainError(
                "FULFILLMENT_NOT_AUTHORIZED",
                "A settlement receipt bound to an accepted guarantee is required.",
                message.task_id or message.message_id,
            )
        if set(payload) != {
            "schemaVersion",
            "guaranteeId",
            "settlementId",
            "settlementReceipt",
            "settlementReceiptDigest",
        } or payload.get("schemaVersion") != "merchant-fulfillment-commit/1":
            raise DomainError(
                "FULFILLMENT_NOT_AUTHORIZED",
                "The guaranteed fulfillment request is malformed.",
                message.task_id or message.message_id,
            )
        receipt = payload.get("settlementReceipt")
        if (
            not isinstance(receipt, dict)
            or receipt.get("success") is not True
            or receipt.get("simulated") is not True
            or receipt.get("network") != self._profile.network
            or payload.get("settlementReceiptDigest") != canonical_digest(receipt)
        ):
            raise DomainError(
                "FULFILLMENT_NOT_AUTHORIZED",
                "The simulation settlement receipt is invalid.",
                message.task_id or message.message_id,
            )
        with self._repository.merchant_transaction() as conn:
            guarantee = conn.execute(
                "SELECT * FROM merchant_guarantees_v3 WHERE task_id=?",
                (message.task_id,),
            ).fetchone()
            if (
                guarantee is None
                or guarantee["context_id"] != message.context_id
                or guarantee["guarantee_id"] != payload.get("guaranteeId")
                or guarantee["state"] not in {"accepted", "fulfilled"}
            ):
                raise DomainError(
                    "FULFILLMENT_NOT_AUTHORIZED",
                    "The Merchant has not accepted this signed payment guarantee.",
                    message.task_id or message.message_id,
                )
            project = metadata.get(PROJECT_METADATA) or {}
            resolved_order_id = order_id or project.get("orderId")
            if (
                resolved_order_id != guarantee["order_id"]
                or project.get("orderId") != guarantee["order_id"]
                or project.get("quoteId") != guarantee["quote_id"]
            ):
                raise DomainError(
                    "X402_TASK_CORRELATION_MISMATCH",
                    "The settlement request does not match the guaranteed order.",
                    message.task_id,
                )
            if self._test_faults is not None:
                fault_target = FulfillmentFaultTarget(
                    order_id=str(resolved_order_id),
                    task_id=str(message.task_id),
                    operation_id=operation_id or "",
                )
                try:
                    consumed = self._test_faults.consume_if_exact(fault_target)
                except ValueError as error:
                    raise DomainError(
                        "A2A_REQUEST_INVALID",
                        "The fulfillment operation identifier is invalid.",
                        str(operation_id),
                    ) from error
                if consumed:
                    raise DomainError(
                        "TEST_FULFILLMENT_REJECTED",
                        "The local one-shot Merchant fulfillment fault was consumed.",
                        str(operation_id),
                    )
        self._repository.append_merchant_message(
            message_id=message.message_id,
            task_id=message.task_id,
            context_id=message.context_id,
            status="payment-settled",
            message=message.model_dump(mode="json", by_alias=True, exclude_none=True),
        )
        if guarantee["state"] == "fulfilled":
            if (
                guarantee["settlement_id"] != payload["settlementId"]
                or guarantee["settlement_receipt_digest"]
                != payload["settlementReceiptDigest"]
            ):
                raise DomainError(
                    "IDEMPOTENCY_CONFLICT",
                    "Guaranteed fulfillment replay changed the settlement.",
                    message.task_id,
                )
            return Task.model_validate(self._repository.merchant_task(message.task_id)["task"])
        task = self.complete_task(
            task_id=message.task_id,
            context_id=message.context_id,
            receipts=[receipt],
            checkout_receipt_id=f"guarantee:{guarantee['guarantee_id']}",
            payment_receipt_id=payload["settlementId"],
        )
        with self._repository.merchant_transaction() as conn:
            changed = conn.execute(
                "UPDATE merchant_guarantees_v3 SET state='fulfilled',settlement_id=?,"
                "settlement_receipt_digest=?,updated_at=? "
                "WHERE guarantee_id=? AND state='accepted'",
                (
                    payload["settlementId"],
                    payload["settlementReceiptDigest"],
                    utc_now(),
                    guarantee["guarantee_id"],
                ),
            ).rowcount
            if changed != 1:
                raise DomainError(
                    "STATE_TRANSITION_CONFLICT",
                    "Guaranteed fulfillment lost compare-and-swap.",
                    message.task_id,
                )
        return task

    def prepare(
        self,
        task_id: str,
        operation_id: str,
        *,
        workflow_id: str | None = None,
        order_id: str | None = None,
        capability_id: str | None = None,
        capability_token: str | None = None,
    ) -> dict[str, Any]:
        result = {
            "operationId": operation_id,
            "taskId": task_id,
            "state": "prepared",
            "reversible": True,
            "artifactDraft": "Demo booking draft",
        }
        self._repository.save_fulfillment(
            operation_id=operation_id,
            task_id=task_id,
            phase="prepare",
            request_digest=canonical_digest(result),
            state="prepared",
            result=result,
        )
        return result

    def complete_task(
        self,
        *,
        task_id: str,
        context_id: str,
        receipts: list[dict[str, Any]],
        checkout_receipt_id: str,
        payment_receipt_id: str,
        workflow_id: str | None = None,
        order_id: str | None = None,
        capability_id: str | None = None,
        capability_token: str | None = None,
    ) -> Task:
        existing = self._repository.merchant_task(task_id)
        if existing["state"] == "completed":
            return Task.model_validate(existing["task"])
        stored_requirements = existing["requirement"]["requirements"]
        try:
            validate_payment_requirement(stored_requirements)
        except ValueError as error:
            raise DomainError(
                "PAYMENT_REQUIREMENT_INVALID",
                "Stored Merchant payment requirements do not match the demo catalog.",
                task_id,
            ) from error
        stored_order_id = existing["order_id"]
        stored_quote_id = stored_requirements.get("quoteId")
        if (
            stored_requirements.get("orderId") != stored_order_id
            or not isinstance(stored_quote_id, str)
            or not stored_quote_id
        ):
            raise DomainError(
                "X402_TASK_CORRELATION_MISMATCH",
                "Stored Merchant order and quote correlation is invalid.",
                task_id,
            )
        current_task = Task.model_validate(existing["task"])
        current_metadata = (
            current_task.status.message.metadata
            if current_task.status.message is not None
            and current_task.status.message.metadata is not None
            else {}
        )
        current_project = current_metadata.get(PROJECT_METADATA)
        current_project = current_project if isinstance(current_project, dict) else {}
        confirmation = project_confirmation(task_id)
        artifact = Artifact(
            artifactId=f"artifact:{task_id}",
            name="デモ予約確認（シミュレーション）",
            parts=[Part(root=DataPart(data=confirmation))],
            metadata={
                "schemaVersion": confirmation["schemaVersion"],
                "scenarioDigest": scenario_digest(),
                "confirmationReference": confirmation_reference(task_id),
                "simulated": True,
                "externalCommit": False,
            },
        )
        message = Message(
            messageId=f"message:completed:{task_id}",
            taskId=task_id,
            contextId=context_id,
            role=Role.agent,
            parts=[Part(root=TextPart(text="Demo booking completed."))],
            metadata={
                **final_task_metadata(status="payment-completed", receipts=receipts),
                PROJECT_METADATA: {
                    "profile": self._profile.profile_id,
                    "simulated": True,
                    "conformance": "NOT_CONFORMANT",
                    "workflowId": existing["workflow_id"],
                    "planDigest": current_project.get("planDigest"),
                    "orderId": stored_order_id,
                    "quoteId": stored_quote_id,
                    "scenarioDigest": scenario_digest(),
                    "checkoutReceiptId": checkout_receipt_id,
                    "paymentReceiptId": payment_receipt_id,
                },
            },
        )
        task = Task(
            id=task_id,
            contextId=context_id,
            status=TaskStatus(state=TaskState.completed, message=message),
            artifacts=[artifact],
            history=[message],
        )
        self._repository.complete_merchant_task(
            task_id, task.model_dump(mode="json", by_alias=True, exclude_none=True)
        )
        return task
