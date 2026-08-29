"""Client boundary for payment-aware merchant agents.

Only onboarding-approved endpoints may cross this boundary.  Payment evidence is
never passed to an LLM or to the legacy orchestration agent.
"""

from __future__ import annotations

import ipaddress
import socket
import uuid
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urljoin, urlparse

import httpx


class MerchantClientError(RuntimeError):
    pass


class MerchantTimeout(MerchantClientError):
    pass


class MerchantClient(Protocol):
    def create_quote(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def fulfill(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def fulfillment_status(self, order_id: str, guarantee_id: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class EndpointPolicy:
    allowed_hosts: frozenset[str]
    allowed_ports: frozenset[int]
    allow_loopback: bool = False

    def validate(self, base_url: str) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in ({"http", "https"} if self.allow_loopback else {"https"}):
            raise MerchantClientError("SSRF_BLOCKED: unsupported endpoint scheme")
        if not parsed.hostname or parsed.username or parsed.password:
            raise MerchantClientError("SSRF_BLOCKED: invalid endpoint authority")
        host = parsed.hostname.lower().rstrip(".")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if host not in self.allowed_hosts or port not in self.allowed_ports:
            raise MerchantClientError("SSRF_BLOCKED: endpoint is not onboarded")
        try:
            resolved = {item[4][0] for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)}
        except OSError as exc:
            raise MerchantClientError("SSRF_BLOCKED: endpoint DNS resolution failed") from exc
        for raw_ip in resolved:
            address = ipaddress.ip_address(raw_ip)
            forbidden = (
                address.is_private
                or address.is_loopback
                or address.is_link_local
                or address.is_multicast
                or address.is_reserved
                or address.is_unspecified
            )
            if forbidden and not (self.allow_loopback and address.is_loopback):
                raise MerchantClientError("SSRF_BLOCKED: endpoint resolves to a forbidden address")


class HttpMerchantClient:
    def __init__(
        self,
        base_url: str,
        policy: EndpointPolicy,
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        policy.validate(base_url)
        self._base_url = base_url.rstrip("/") + "/"
        self._timeout = timeout_seconds

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=False) as client:
                response = client.request(method, urljoin(self._base_url, path.lstrip("/")), **kwargs)
        except httpx.TimeoutException as exc:
            raise MerchantTimeout("merchant result is unknown") from exc
        except httpx.HTTPError as exc:
            raise MerchantClientError("merchant request failed") from exc
        if 300 <= response.status_code < 400:
            raise MerchantClientError("SSRF_BLOCKED: redirects are not allowed")
        if response.status_code >= 500:
            raise MerchantTimeout("merchant result is unknown")
        if response.status_code >= 400:
            code = "MERCHANT_REQUEST_REJECTED"
            try:
                code = response.json().get("error", {}).get("code", code)
            except ValueError:
                pass
            raise MerchantClientError(code)
        result = response.json()
        if not isinstance(result, dict):
            raise MerchantClientError("merchant returned an invalid response")
        return result

    def create_quote(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._a2a("quote", request=request)

    def fulfill(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = dict(request)
        fault = payload.pop("_testFault", None)
        headers = {"X-Demo-Test-Fault": str(fault)} if fault else None
        return self._a2a("fulfill", request=payload, headers=headers)

    def fulfillment_status(self, order_id: str, guarantee_id: str) -> dict[str, Any]:
        return self._a2a(
            "fulfillment_status", orderId=order_id, guaranteeId=guarantee_id
        )

    def _a2a(self, action: str, *, headers: dict[str, str] | None = None, **data: Any) -> dict[str, Any]:
        request_id = f"merchant-{uuid.uuid4().hex}"
        envelope = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": f"message-{uuid.uuid4().hex}",
                    "role": "user",
                    "parts": [{"kind": "data", "data": {"action": action, **data}}],
                }
            },
        }
        response = self._request("POST", "/a2a", json=envelope, headers=headers)
        if response.get("jsonrpc") != "2.0" or response.get("id") != request_id:
            raise MerchantClientError("merchant returned an invalid A2A envelope")
        result = response.get("result")
        if not isinstance(result, dict):
            raise MerchantClientError("merchant A2A result is invalid")
        return result
