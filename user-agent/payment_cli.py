#!/usr/bin/env python3
"""Terminal entry point for the deterministic demo user agent."""

from __future__ import annotations

import argparse
import os

from payment_client import APPROVAL_WORD, DEFAULT_PROMPT, PaymentMediatorClient, run_interactive


def main() -> None:
    parser = argparse.ArgumentParser(description="AP2/x402 marketplace demo user agent")
    parser.add_argument("--mediator-url", default=os.getenv("PAYMENT_MEDIATOR_URL", "http://127.0.0.1:8004"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--approval",
        help=f"非対話実行時の承認語。決済する場合は完全一致で「{APPROVAL_WORD}」を指定。",
    )
    args = parser.parse_args()
    result = run_interactive(
        PaymentMediatorClient(args.mediator_url),
        prompt=args.prompt,
        approval=args.approval,
    )
    raise SystemExit(0 if result is not None else 2)


if __name__ == "__main__":
    main()
