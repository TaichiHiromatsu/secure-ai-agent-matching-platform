"""Deterministic ADK Web chat agent for the mediated payment demo."""

from __future__ import annotations

import asyncio
import os
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from .payment_client import (
    APPROVAL_WORD,
    PaymentMediatorClient,
    format_completion,
    format_payment_request,
)


PENDING_STATE_KEY = "payment_demo_pending"


def _user_text(context: InvocationContext) -> str:
    content = context.user_content
    if content is None:
        return ""
    return "".join(part.text or "" for part in content.parts or [] if part.text).strip()


class PaymentDemoUserAgent(BaseAgent):
    """Two-turn chat: request a quote, then require the exact word ``承認``."""

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        text = _user_text(context)
        pending = context.session.state.get(PENDING_STATE_KEY)
        client = PaymentMediatorClient(
            os.getenv("PAYMENT_MEDIATOR_URL", "http://127.0.0.1:8004")
        )
        actions = EventActions()
        try:
            if text == APPROVAL_WORD:
                if not isinstance(pending, dict) or not pending:
                    reply = "承認対象の支払依頼がありません。先にデモ予約を依頼してください。"
                else:
                    task = await asyncio.to_thread(client.submit_approval, pending)
                    reply = format_completion(task)
                    context.session.state[PENDING_STATE_KEY] = {}
                    actions = EventActions(stateDelta={PENDING_STATE_KEY: {}})
            elif isinstance(pending, dict) and pending:
                if text == "拒否":
                    context.session.state[PENDING_STATE_KEY] = {}
                    actions = EventActions(stateDelta={PENDING_STATE_KEY: {}})
                    reply = "支払依頼を拒否しました。決済は実行していません。"
                else:
                    reply = (
                        f"決済はまだ実行していません。支払う場合は「{APPROVAL_WORD}」、"
                        "取り消す場合は「拒否」と入力してください。"
                    )
            else:
                pending = await asyncio.to_thread(client.request_payment, text)
                context.session.state[PENDING_STATE_KEY] = pending
                actions = EventActions(stateDelta={PENDING_STATE_KEY: pending})
                reply = format_payment_request(pending)
        except Exception as exc:
            reply = f"決済デモを継続できませんでした: {exc}"

        yield Event(
            invocationId=context.invocation_id,
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=reply)]),
            actions=actions,
        )


root_agent = PaymentDemoUserAgent(
    name="payment_demo_user_agent",
    description="仲介エージェントへ接続し、日本語の明示承認後に決済するデモ用利用者エージェント",
)
