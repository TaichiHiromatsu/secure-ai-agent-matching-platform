from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from typing import List

from app.services.agent_registry import load_agents
from app.dependencies import templates


router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_class=HTMLResponse)
async def list_agents(request: Request):
    agents = load_agents()
    # sort by updated_at desc if present
    agents_sorted = sorted(agents, key=lambda a: a.updated_at or a.created_at or "", reverse=True)
    return templates.TemplateResponse(
        "agents.html",
        {
            "request": request,
            "agents": agents_sorted,
        },
    )

