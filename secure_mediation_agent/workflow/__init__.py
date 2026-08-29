"""Deterministic durable workflow boundary with cycle-free lazy exports."""

from __future__ import annotations

from typing import Any


__all__ = ["WorkflowController", "WorkflowRequest", "WorkflowState"]


def __getattr__(name: str) -> Any:
    if name == "WorkflowController":
        from .controller import WorkflowController

        return WorkflowController
    if name in {"WorkflowRequest", "WorkflowState"}:
        from .models import WorkflowRequest, WorkflowState

        return {"WorkflowRequest": WorkflowRequest, "WorkflowState": WorkflowState}[name]
    raise AttributeError(name)
