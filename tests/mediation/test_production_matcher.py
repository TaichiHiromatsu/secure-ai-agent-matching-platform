from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import httpx
from fastapi import FastAPI

from secure_mediation_agent.mediation.adapters import LegacyMatcherAdapter
from secure_mediation_agent.mediation.canonical import canonical_digest
from secure_mediation_agent.merchant.api import MerchantRuntime, create_app
from secure_mediation_agent.merchant.service import PaidBookingMerchant
from secure_mediation_agent.payment_profiles.registry import ProfileRegistry
from secure_mediation_agent.workflow.approval import AuthorizationService
from trusted_agent_store.app.services.agent_registry import load_agents


ROOT = Path(__file__).resolve().parents[2]


class _HostRoutingTransport(httpx.AsyncBaseTransport):
    def __init__(self, apps: dict[int, object]) -> None:
        self._transports = {
            port: httpx.ASGITransport(app=app) for port, app in apps.items()
        }

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        transport = self._transports.get(request.url.port or 80)
        if transport is None:
            raise httpx.ConnectError("unexpected local test destination", request=request)
        return await transport.handle_async_request(request)

    async def aclose(self) -> None:
        for transport in self._transports.values():
            await transport.aclose()


def test_production_matcher_accepts_real_registry_and_free_paid_live_cards(
    monkeypatch, workflow_fixture
):
    registry_path = ROOT / "trusted_agent_store/data/agents/registered-agents.json"
    monkeypatch.setattr(
        "trusted_agent_store.app.services.agent_registry.REGISTRY_PATH",
        registry_path,
    )
    registry_app = FastAPI()

    @registry_app.get("/api/agents")
    async def registry_agents(status: str | None = None, limit: int = 100):
        agents = load_agents()
        if status:
            agents = [agent for agent in agents if agent.status == status]
        return {
            "items": [asdict(agent) for agent in agents[:limit]],
            "total": len(agents),
            "limit": limit,
            "offset": 0,
        }

    free_card_path = (
        ROOT / "external-agents/trusted-agents/hotel_agent/agent.json"
    )
    free_app = FastAPI()

    @free_app.get("/a2a/hotel_agent/.well-known/agent-card.json")
    async def free_agent_card():
        return json.loads(free_card_path.read_text(encoding="utf-8"))

    keys = workflow_fixture["keys"]
    profile = ProfileRegistry.load(
        "x402-wire-simulation/1", simulation_key=keys.simulation_signer
    )
    paid_app = create_app(
        MerchantRuntime(
            service=PaidBookingMerchant(
                workflow_fixture["repository"], keys, profile
            ),
            authorization=AuthorizationService(keys.plan_authority),
            paths=workflow_fixture["paths"],
            extension_uri=profile.extension_uri,
        )
    )

    original_async_client = httpx.AsyncClient

    def routed_client(*args, **kwargs):
        kwargs["transport"] = _HostRoutingTransport(
            {8001: registry_app, 8002: free_app, 8005: paid_app}
        )
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", routed_client)

    async def exercise():
        matcher = LegacyMatcherAdapter()
        free = await matcher.match("hotel search")
        paid = await matcher.match("paid payment booking")
        return free, paid

    free, paid = asyncio.run(exercise())

    assert free[0].canonical_agent_id == "agent-002"
    assert free[0].payment_extension_uris == ()
    assert paid[0].canonical_agent_id == "agent-005"
    assert paid[0].payment_extension_uris == (profile.extension_uri,)
    for snapshot in (free[0], paid[0]):
        digest_input = snapshot.model_dump(
            mode="json",
            by_alias=True,
            exclude={"snapshot_digest"},
        )
        assert snapshot.snapshot_digest == canonical_digest(digest_input)
