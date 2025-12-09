#!/usr/bin/env python3
"""Simple batch evaluation script for Trusted Agent Store.

Usage:
    python scripts/simple_batch_eval.py              # Run both normal and malicious
    python scripts/simple_batch_eval.py --normal     # Run normal agents only
    python scripts/simple_batch_eval.py --malicious  # Run malicious agents only
"""

import argparse
import asyncio
import csv
import sys
from datetime import datetime
from pathlib import Path

import httpx

# Configuration
BASE_URL = "http://localhost:8080/api"
SUBMIT_URL = f"{BASE_URL}/submissions/"

ALL_AGENTS = {
    "normal": ("normal", "http://localhost:8080/a2a/sales_agent/.well-known/agent-card.json"),
    "malicious": ("malicious", "http://localhost:8080/a2a/data_harvester_agent/.well-known/agent-card.json"),
}

RUNS_PER_TYPE = 60  # Number of runs per agent type
MAX_PARALLEL = 1  # Sequential to avoid rate limits
POLL_INTERVAL = 15  # seconds
SG_PROMPTS = 30
ACA_SCENARIOS = 5

LOG_PATH = Path("logs/simple_batch_scores.csv")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


async def submit(client: httpx.AsyncClient, agent_card_url: str) -> dict:
    """Submit an agent for evaluation."""
    payload = {
        "agent_card_url": agent_card_url,
        "endpoint_manifest": {},
        "endpoint_snapshot_hash": "batch",
        "signature_bundle": {},
        "organization_meta": {},
        "request_context": {
            "security_gate": {"max_prompts": SG_PROMPTS},
            "agent_card_accuracy": {"max_scenarios": ACA_SCENARIOS},
            "stages": {
                "precheck": True,
                "security": True,
                "agent_card_accuracy": True,
                "judge": True,
            },
        },
    }
    resp = await client.post(SUBMIT_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


async def poll_until_done(client: httpx.AsyncClient, submission_id: str) -> dict:
    """Poll until submission reaches terminal state."""
    while True:
        resp = await client.get(f"{SUBMIT_URL}{submission_id}", timeout=15)
        resp.raise_for_status()
        data = resp.json()

        state = data.get("state", "")
        auto_decision = data.get("auto_decision")

        # Terminal states
        if state in {"approved", "rejected", "failed"}:
            return data
        if auto_decision in {"auto_approved", "auto_rejected", "requires_human_review"}:
            return data

        print(f"  [{submission_id[:8]}] state={state}, waiting...")
        await asyncio.sleep(POLL_INTERVAL)


def append_log(row: dict) -> None:
    """Append result to CSV log."""
    exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "agent_type", "run_idx", "submission_id",
            "trust_score", "auto_decision", "state"
        ])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


async def main(agent_types: list[str], start_idx: int = 1, end_idx: int = RUNS_PER_TYPE):
    # Select agents based on options
    agents = [ALL_AGENTS[t] for t in agent_types if t in ALL_AGENTS]

    print(f"Base URL: {BASE_URL}")
    print(f"Runs: {start_idx} to {end_idx}")
    print(f"Agents: {[a[0] for a in agents]}")
    print()

    # Build job list
    jobs = []
    for agent_type, card_url in agents:
        for i in range(start_idx, end_idx + 1):
            jobs.append((agent_type, i, card_url))

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3

    async with httpx.AsyncClient(follow_redirects=True) as client:
        for agent_type, idx, card_url in jobs:
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n[STOP] {MAX_CONSECUTIVE_FAILURES} consecutive failures - likely rate limited. Stopping.")
                break

            print(f"[START] {agent_type} #{idx}")
            try:
                result = await submit(client, card_url)
                submission_id = result["id"]
                print(f"  Submitted: {submission_id}")

                final = await poll_until_done(client, submission_id)

                trust_score = final.get("trust_score")
                row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_type": agent_type,
                    "run_idx": idx,
                    "submission_id": submission_id,
                    "trust_score": trust_score,
                    "auto_decision": final.get("auto_decision"),
                    "state": final.get("state"),
                }
                append_log(row)
                print(f"[DONE] {agent_type} #{idx} -> {row['auto_decision']} score={row['trust_score']}")

                # Check if trust_score was obtained
                if trust_score is None:
                    consecutive_failures += 1
                    print(f"  [WARN] No trust_score ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
                else:
                    consecutive_failures = 0  # Reset on success

            except Exception as e:
                print(f"[ERR] {agent_type} #{idx}: {e}", file=sys.stderr)
                consecutive_failures += 1
                print(f"  [WARN] Error occurred ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
                row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_type": agent_type,
                    "run_idx": idx,
                    "submission_id": "error",
                    "trust_score": None,
                    "auto_decision": "error",
                    "state": "failed",
                }
                append_log(row)

    print()
    print(f"Results logged to: {LOG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluation for Trusted Agent Store")
    parser.add_argument("--normal", action="store_true", help="Run normal agents only")
    parser.add_argument("--malicious", action="store_true", help="Run malicious agents only")
    parser.add_argument("--start", type=int, default=1, help="Start index (default: 1)")
    parser.add_argument("--end", type=int, default=RUNS_PER_TYPE, help="End index (default: 60)")
    args = parser.parse_args()

    # Determine which agent types to run
    if args.normal and args.malicious:
        agent_types = ["normal", "malicious"]
    elif args.normal:
        agent_types = ["normal"]
    elif args.malicious:
        agent_types = ["malicious"]
    else:
        # Default: run both
        agent_types = ["normal", "malicious"]

    asyncio.run(main(agent_types, args.start, args.end))
