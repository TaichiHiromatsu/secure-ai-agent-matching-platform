"""Safe workflow client shared by the CLI and the ADK adapter."""

from __future__ import annotations

import uuid
from typing import Any

import httpx

SESSION_COOKIE_NAME = "__Host-payment-session"


class WorkflowApiError(RuntimeError):
    def __init__(self, error: dict[str, Any]) -> None:
        self.error = error
        super().__init__(f"{error.get('code', 'WORKFLOW_API_ERROR')}: {error.get('message', 'request failed')}")


class WorkflowClient:
    def __init__(
        self,
        base_url: str,
        *,
        identity_assertion: str | None = None,
        session_cookie: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.identity_assertion = identity_assertion
        self.session_cookie = session_cookie
        self.client = client or httpx.Client(timeout=30.0)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        headers = dict(kwargs.pop("headers", {}))
        if self.identity_assertion:
            headers["X-Verified-Identity"] = self.identity_assertion
        if self.session_cookie:
            headers["Cookie"] = f"{SESSION_COOKIE_NAME}={self.session_cookie}"
        response = self.client.request(method, f"{self.base_url}{path}", headers=headers, **kwargs)
        value = response.json()
        if response.status_code >= 400:
            error = value.get("error", {}) if isinstance(value, dict) else {}
            raise WorkflowApiError(error)
        if not isinstance(value, dict):
            raise WorkflowApiError({"code": "INVALID_RESPONSE", "message": "workflow response is not an object"})
        return value

    def create(self, *, goal: str, session_id: str, context_id: str, payment_required: bool = True) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/workflows",
            headers={"Idempotency-Key": f"create-{uuid.uuid4().hex}"},
            json={
                "sessionId": session_id,
                "contextId": context_id,
                "request": {"goal": goal, "paymentRequired": payment_required},
            },
        )

    def message(self, workflow_id: str, *, text: str, expected_version: int | None = None) -> dict[str, Any]:
        return self.message_parts(
            workflow_id,
            parts=[{"kind": "text", "text": text}],
            expected_version=expected_version,
        )

    def message_parts(
        self,
        workflow_id: str,
        *,
        parts: list[dict[str, str]],
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        message_id = f"message:{uuid.uuid4().hex}"
        body: dict[str, Any] = {
            "messageId": message_id,
            "parts": parts,
        }
        if expected_version is not None:
            body["expectedVersion"] = expected_version
        return self._request(
            "POST",
            f"/v1/workflows/{workflow_id}/messages",
            headers={"Idempotency-Key": f"message-{uuid.uuid4().hex}"},
            json=body,
        )

    def get(self, workflow_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/workflows/{workflow_id}")
