"""Secure mediation agent package.

The interactive ADK agent is imported lazily so the deterministic payment
service can run without importing LLM clients, credentials, or orchestration
code into its trusted payment boundary.
"""

from typing import Any

__all__ = ["root_agent"]


def __getattr__(name: str) -> Any:
    if name != "root_agent":
        raise AttributeError(name)
    from .agent import root_agent

    return root_agent
