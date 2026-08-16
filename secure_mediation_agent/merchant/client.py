"""HTTP A2A gateway used by the production workflow runtime."""

from __future__ import annotations

from typing import Any

import httpx
from a2a.types import AgentCard, Message, Task

from secure_mediation_agent.workflow.errors import DomainError

from .service import MerchantStartResult


class HttpPaidBookingMerchant:
    """Strict client for the selected paid Merchant's private A2A endpoint."""

    def __init__(self, base_url: str, *, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _call(
        self,
        action: str,
        operation_id: str,
        params: dict[str, Any],
        *,
        activation: str,
        capability_token: str,
    ) -> dict[str, Any]:
        try:
            response = httpx.post(
                f"{self.base_url}/a2a",
                json={
                    "jsonrpc": "2.0",
                    "id": operation_id,
                    "method": "message/send",
                    "params": {
                        "action": action,
                        "operationId": operation_id,
                        **params,
                    },
                },
                headers={
                    "Authorization": f"Bearer {capability_token}",
                    "X-A2A-Extensions": activation,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            wire = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise DomainError(
                "MERCHANT_A2A_UNAVAILABLE",
                "Selected Merchant A2A operation failed.",
                operation_id,
            ) from error
        if wire.get("jsonrpc") != "2.0" or "result" not in wire:
            raise DomainError(
                "MERCHANT_A2A_INVALID_RESPONSE",
                "Selected Merchant returned an invalid A2A envelope.",
                operation_id,
            )
        return wire["result"]

    def health(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/ready", timeout=2.0)
            return response.status_code == 200 and response.json().get("status") == "ready"
        except (httpx.HTTPError, ValueError):
            return False

    def agent_card(self) -> AgentCard:
        response = httpx.get(
            f"{self.base_url}/.well-known/agent-card.json", timeout=self.timeout
        )
        response.raise_for_status()
        return AgentCard.model_validate(response.json())

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
        capability_token: str,
    ) -> MerchantStartResult:
        extension = next(iter(activation))
        result = self._call(
            "merchant-task:start",
            f"start:{task_id}",
            {
                "workflowId": workflow_id,
                "planDigest": plan_digest,
                "taskId": task_id,
                "orderId": order_id,
                "contextId": context_id,
                "capabilityId": capability_id,
                "issuedAt": issued_at,
                "expiresAt": expires_at,
            },
            activation=extension,
            capability_token=capability_token,
        )
        private = result["privatePaymentMaterial"]
        return MerchantStartResult(
            task=Task.model_validate(result["task"]),
            checkout_jwt=private["checkoutJwt"],
            checkout_hash=private["checkoutHash"],
            requirements=result["requirements"],
            activation_echo=result["activationEcho"],
            checkout_challenge=result["checkoutChallenge"],
            payment_challenge=result["paymentChallenge"],
        )

    def verify_checkout(
        self,
        token: str,
        *,
        workflow_id: str,
        plan_digest: str,
        task_id: str,
        capability_token: str,
    ) -> dict[str, Any]:
        return self._call(
            "merchant-task:start",
            f"verify-checkout:{task_id}",
            {
                "workflowId": workflow_id,
                "planDigest": plan_digest,
                "taskId": task_id,
                "checkoutJwt": token,
            },
            activation="urn:secure-a2a:extensions:x402-wire-simulation:v1",
            capability_token=capability_token,
        )["claims"]

    def submit_payment(
        self,
        *,
        message: Message,
        checkout_mandate: str,
        checkout_jwt: str,
        checkout_nonce: str,
        capability_id: str,
        capability_token: str,
        workflow_id: str,
        order_id: str,
    ) -> None:
        self._call(
            "merchant:payment-submit",
            f"submit:{message.task_id}",
            {
                "workflowId": workflow_id,
                "taskId": message.task_id,
                "orderId": order_id,
                "capabilityId": capability_id,
                "message": message.model_dump(mode="json", by_alias=True, exclude_none=True),
                "checkoutMandate": checkout_mandate,
                "checkoutJwt": checkout_jwt,
                "checkoutNonce": checkout_nonce,
            },
            activation="urn:secure-a2a:extensions:x402-wire-simulation:v1",
            capability_token=capability_token,
        )

    def submit_guarantee(
        self,
        *,
        operation_id: str,
        message: Message,
        workflow_id: str,
        order_id: str,
        capability_id: str,
        capability_token: str,
    ) -> Task:
        result = self._call(
            "merchant:payment-guarantee-submit",
            operation_id,
            {
                "workflowId": workflow_id,
                "taskId": message.task_id,
                "orderId": order_id,
                "capabilityId": capability_id,
                "message": message.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                ),
            },
            activation="urn:secure-a2a:extensions:x402-wire-simulation:v1",
            capability_token=capability_token,
        )
        return Task.model_validate(result["task"])

    def commit_guaranteed_fulfillment(
        self,
        *,
        operation_id: str,
        message: Message,
        workflow_id: str,
        order_id: str,
        capability_id: str,
        capability_token: str,
    ) -> Task:
        result = self._call(
            "merchant:guaranteed-fulfillment-commit",
            operation_id,
            {
                "workflowId": workflow_id,
                "taskId": message.task_id,
                "orderId": order_id,
                "capabilityId": capability_id,
                "message": message.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                ),
            },
            activation="urn:secure-a2a:extensions:x402-wire-simulation:v1",
            capability_token=capability_token,
        )
        return Task.model_validate(result["task"])

    def prepare(
        self,
        task_id: str,
        operation_id: str,
        *,
        workflow_id: str,
        order_id: str,
        capability_id: str,
        capability_token: str,
    ) -> dict[str, Any]:
        return self._call(
            "merchant:fulfillment-prepare",
            operation_id,
            {
                "workflowId": workflow_id,
                "taskId": task_id,
                "orderId": order_id,
                "capabilityId": capability_id,
            },
            activation="urn:secure-a2a:extensions:x402-wire-simulation:v1",
            capability_token=capability_token,
        )

    def complete_task(
        self,
        *,
        task_id: str,
        context_id: str,
        receipts: list[dict[str, Any]],
        checkout_receipt_id: str,
        payment_receipt_id: str,
        workflow_id: str,
        order_id: str,
        capability_id: str,
        capability_token: str,
    ) -> Task:
        result = self._call(
            "merchant:fulfillment-commit",
            f"commit:{task_id}",
            {
                "workflowId": workflow_id,
                "taskId": task_id,
                "orderId": order_id,
                "contextId": context_id,
                "capabilityId": capability_id,
                "receipts": receipts,
                "checkoutReceiptId": checkout_receipt_id,
                "paymentReceiptId": payment_receipt_id,
            },
            activation="urn:secure-a2a:extensions:x402-wire-simulation:v1",
            capability_token=capability_token,
        )
        return Task.model_validate(result["task"])
