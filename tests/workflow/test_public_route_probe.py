from __future__ import annotations

import asyncio

import httpx

from secure_mediation_agent.workflow.api import _probe_public_routes


def test_live_public_route_probe_checks_method_path_semantics() -> None:
    expected = {
        ("GET", "/mediation-api/v1/view"): 401,
        ("POST", "/mediation-api/v1/turns"): 401,
        ("GET", "/mediation-api/v1/turns"): 404,
        ("POST", "/mediation-api/v1/view"): 404,
        ("POST", "/run"): 401,
        ("GET", "/run"): 404,
        ("GET", "/v1/view"): 404,
        ("GET", "/payment/internal"): 404,
        ("GET", "/paid-agent/internal"): 404,
        ("GET", "/internal/authority"): 404,
    }
    observed: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        observed.append(key)
        return httpx.Response(expected[key])

    ready = asyncio.run(
        _probe_public_routes(
            "http://edge",
            transport=httpx.MockTransport(handler),
        )
    )
    assert ready is True
    assert observed == list(expected)


def test_live_public_route_probe_fails_if_internal_route_is_exposed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/mediation-api/v1/view":
            return httpx.Response(401)
        if request.url.path == "/internal/authority":
            return httpx.Response(200)
        return httpx.Response(404)

    ready = asyncio.run(
        _probe_public_routes(
            "http://edge",
            transport=httpx.MockTransport(handler),
        )
    )
    assert ready is False
