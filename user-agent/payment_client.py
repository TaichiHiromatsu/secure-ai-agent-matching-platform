"""CLI adapter for the same authenticated durable workflow used by ADK Web."""

from __future__ import annotations

from typing import Callable

from secure_mediation_agent.workflow.client import WorkflowClient


APPROVAL_WORD = "承認"
DEFAULT_PROMPT = "信頼済みの予約エージェントを使い、デモ予約を1件取得してください。"


def run_interactive(
    client: WorkflowClient,
    *,
    prompt: str,
    plan_approval: str | None = None,
    payment_approval: str | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> dict[str, object] | None:
    """Run the two distinct Human Present approval turns without normalization."""

    view = client.create(goal=prompt, session_id="cli-session", context_id="cli-context")
    output_fn(str(view["renderedText"]))
    first = plan_approval if plan_approval is not None else input_fn("計画承認: ")
    if first != APPROVAL_WORD:
        output_fn("計画承認語が完全一致しないため、実行を開始しませんでした。")
        return None
    view = client.message(
        str(view["workflowId"]), text=first, expected_version=int(view["version"])
    )
    output_fn(str(view["renderedText"]))
    second = payment_approval if payment_approval is not None else input_fn("決済承認: ")
    if second != APPROVAL_WORD:
        output_fn("決済承認語が完全一致しないため、決済を開始しませんでした。")
        return None
    view = client.message(
        str(view["workflowId"]), text=second, expected_version=int(view["version"])
    )
    output_fn(str(view["renderedText"]))
    return view
