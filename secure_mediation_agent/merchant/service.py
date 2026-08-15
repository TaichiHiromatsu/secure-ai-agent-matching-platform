"""Paid booking Merchant for the approved single-product simulation."""

from __future__ import annotations

import base64
import json
import secrets
from dataclasses import dataclass
from typing import Any

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentSkill,
    Artifact,
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
from secure_mediation_agent.payment_profiles.a2a import (
    PROJECT_METADATA,
    final_task_metadata,
    payment_required_task,
)
from secure_mediation_agent.payment_profiles.simulation_v1 import SimulationV1Profile
from secure_mediation_agent.workflow.canonical import canonical_digest, canonical_json
from secure_mediation_agent.workflow.repository import WorkflowRepository


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
    ) -> None:
        self._repository = repository
        self._keys = keys
        self._profile = profile

    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="paid_booking_agent",
            description="AP2 v0.2 Human Present demo Merchant; local simulation only.",
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
                    name="Demo paid booking",
                    description="One fixed local simulation booking product.",
                    tags=["booking", "payment", "simulation"],
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
        requirements = self._profile.build_required(amount=1250)
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
                "taskId": task_id,
                "merchantId": self.merchant_id,
                "productId": "demo-paid-booking",
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
            "checkoutJwtDigest": canonical_digest(checkout_jwt),
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
            "productId": "demo-paid-booking",
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
        artifact = Artifact(
            artifactId=f"artifact:{task_id}",
            name="Demo booking confirmation",
            parts=[Part(root=TextPart(text="Demo booking confirmed."))],
            metadata={"simulated": True, "externalCommit": False},
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
