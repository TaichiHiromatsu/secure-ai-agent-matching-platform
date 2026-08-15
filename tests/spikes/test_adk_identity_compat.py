"""ADK 1.19.0 compatibility checks for the deterministic root adapter seam."""

from __future__ import annotations

import inspect

import pytest
from google.adk.agents import BaseAgent
from google.genai import types


pytestmark = pytest.mark.spike


class _PartsPreservingAgent(BaseAgent):
    async def _run_async_impl(self, ctx):
        if False:  # pragma: no cover - establishes the async-generator contract
            yield ctx


def test_base_agent_subclass_and_raw_parts_are_supported() -> None:
    agent = _PartsPreservingAgent(name="secure_mediator")
    content = types.Content(
        role="user",
        parts=[types.Part(text="承認"), types.Part(text="second")],
    )
    assert agent.name == "secure_mediator"
    assert [part.text for part in content.parts or []] == ["承認", "second"]
    signature = inspect.signature(BaseAgent._run_async_impl)
    assert list(signature.parameters) == ["self", "ctx"]
