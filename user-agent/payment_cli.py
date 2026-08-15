#!/usr/bin/env python3
"""Terminal entry point for the deterministic demo user agent."""

from __future__ import annotations

import argparse
import os

from payment_client import APPROVAL_WORD, DEFAULT_PROMPT, run_interactive
from secure_mediation_agent.workflow.client import WorkflowClient


def main() -> None:
    parser = argparse.ArgumentParser(description="AP2 Human Present / x402 wire simulation workflow client")
    parser.add_argument(
        "--workflow-url",
        default=os.getenv("WORKFLOW_PUBLIC_URL", "http://127.0.0.1:8080/mediation-api"),
        help="Authenticated public workflow route; direct loopback port 8004 is unsupported.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--plan-approval",
        help=f"非対話実行時の計画承認語。完全一致で「{APPROVAL_WORD}」。",
    )
    parser.add_argument(
        "--payment-approval",
        help=f"非対話実行時の決済承認語。完全一致で「{APPROVAL_WORD}」。",
    )
    parser.add_argument("--session-cookie", default=os.getenv("WORKFLOW_SESSION_COOKIE"))
    args = parser.parse_args()
    result = run_interactive(
        WorkflowClient(args.workflow_url, session_cookie=args.session_cookie),
        prompt=args.prompt,
        plan_approval=args.plan_approval,
        payment_approval=args.payment_approval,
    )
    raise SystemExit(0 if result is not None else 2)


if __name__ == "__main__":
    main()
