"""Loopback client for the single authoritative mediation runtime."""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

import httpx

from .errors import MediationError, SecurityBlocked
from .models import MediationPublicView, TextPart


class HttpMediationAuthority:
    """Use the workflow process as the only mediation controller/store owner."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get(
                "MEDIATION_AUTHORITY_URL", "http://127.0.0.1:8004"
            )
        ).rstrip("/")
        self.transport = transport

    @staticmethod
    def _error(response: httpx.Response) -> MediationError:
        try:
            value = response.json()
        except ValueError:
            value = {}
        envelope = value.get("error", {}) if isinstance(value, dict) else {}
        code = str(envelope.get("code") or "MEDIATION_AUTHORITY_ERROR")
        message = str(
            envelope.get("message")
            or "仲介処理のauthoritative runtimeへ接続できませんでした。"
        )
        if response.status_code == 403:
            return SecurityBlocked(code, message)
        return MediationError(code, message)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        assertion: str,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        request_headers = {
            "X-Verified-Identity": assertion,
            **(headers or {}),
        }
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(300.0),
                follow_redirects=False,
                trust_env=False,
                transport=self.transport,
            ) as client:
                response = await client.request(
                    method,
                    f"{self.base_url}{path}",
                    headers=request_headers,
                    json=json_body,
                )
        except httpx.HTTPError as error:
            raise MediationError(
                "MEDIATION_AUTHORITY_UNAVAILABLE",
                "仲介処理のauthoritative runtimeへ接続できませんでした。",
            ) from error
        if response.status_code >= 400:
            raise self._error(response)
        try:
            return response.json()
        except ValueError as error:
            raise MediationError(
                "MEDIATION_AUTHORITY_RESPONSE_INVALID",
                "仲介処理の応答を確認できませんでした。",
            ) from error

    async def turn(
        self,
        *,
        assertion: str,
        parts: Iterable[TextPart],
        request_id: str,
        expected_version: int | None = None,
    ) -> MediationPublicView:
        body: dict[str, Any] = {
            "schemaVersion": "mediation-turn-request/1",
            "requestId": request_id,
            "message": {
                "parts": [
                    part.model_dump(mode="json", by_alias=True) for part in parts
                ]
            },
            "selectionToken": None,
        }
        if expected_version is not None:
            body["expectedVersion"] = expected_version
        value = await self._request(
            "POST",
            "/v1/turns",
            assertion=assertion,
            headers={
                "Idempotency-Key": request_id,
                "X-Request-ID": request_id,
            },
            json_body=body,
        )
        if not isinstance(value, dict) or not isinstance(value.get("view"), dict):
            raise MediationError(
                "MEDIATION_AUTHORITY_RESPONSE_INVALID",
                "仲介処理の応答を確認できませんでした。",
            )
        return MediationPublicView.model_validate_json(
            json.dumps(value["view"], ensure_ascii=False)
        )

    async def view(self, *, assertion: str) -> MediationPublicView | None:
        value = await self._request(
            "GET", "/v1/view", assertion=assertion
        )
        if value is None:
            return None
        if not isinstance(value, dict):
            raise MediationError(
                "MEDIATION_AUTHORITY_RESPONSE_INVALID",
                "仲介処理の応答を確認できませんでした。",
            )
        return MediationPublicView.model_validate_json(
            json.dumps(value, ensure_ascii=False)
        )
