"""Keyless deterministic ADK adapter for the authoritative workflow API."""

from __future__ import annotations

import asyncio
import os
from typing import AsyncGenerator

import httpx
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from .workflow.client import WorkflowApiError, WorkflowClient


WORKFLOW_STATE_KEY = "payment_user_agent_workflow_id"


def _raw_text_parts(context: InvocationContext) -> list[dict[str, str]]:
    content = context.user_content
    if content is None:
        return []
    values: list[dict[str, str]] = []
    for part in content.parts or []:
        if part.text is None:
            raise ValueError("このreleaseではtext partのみ受け付けます。")
        values.append({"kind": "text", "text": part.text})
    return values


def _adapter_identity(subject: str) -> str:
    configured = os.getenv("WORKFLOW_IDENTITY_ASSERTION")
    if configured:
        return configured
    response = httpx.post(
        os.getenv(
            "WORKFLOW_IDENTITY_BROKER_URL",
            "http://127.0.0.1:8003/auth/internal/identity",
        ),
        json={"subject": subject},
        timeout=5.0,
    )
    response.raise_for_status()
    return str(response.json()["assertion"])


class PaymentWorkflowAdapter(BaseAgent):
    """Preserve raw parts; all approval and payment decisions remain deterministic."""

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        actions = EventActions()
        try:
            parts = _raw_text_parts(context)
            assertion = await asyncio.to_thread(
                _adapter_identity, str(context.session.user_id)
            )
            client = WorkflowClient(
                os.getenv("WORKFLOW_API_URL", "http://127.0.0.1:8004"),
                identity_assertion=assertion,
            )
            workflow_id = context.session.state.get(WORKFLOW_STATE_KEY)
            if workflow_id:
                view = await asyncio.to_thread(
                    client.message_parts,
                    str(workflow_id),
                    parts=parts,
                )
            else:
                if len(parts) != 1:
                    raise ValueError("新規依頼は一つのtext partで送信してください。")
                view = await asyncio.to_thread(
                    client.create,
                    goal=parts[0]["text"],
                    session_id=str(context.session.id),
                    context_id=str(context.session.id),
                )
                workflow_id = str(view["workflowId"])
                actions = EventActions(stateDelta={WORKFLOW_STATE_KEY: workflow_id})
            reply = str(view["renderedText"])
        except WorkflowApiError as error:
            reply = f"workflow error: {error.error.get('code', 'WORKFLOW_API_ERROR')}"
        except Exception as error:
            reply = f"workflow adapter error: {type(error).__name__}"
        yield Event(
            invocationId=context.invocation_id,
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=reply)]),
            actions=actions,
        )
