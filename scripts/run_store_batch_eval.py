#!/usr/bin/env python3
"""Batch-submit agents to Trusted Agent Store and log Trust Scores.

Edit constants below to fit your environment. No env vars required.
"""

import asyncio
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx


class RateLimitError(Exception):
    pass
BASE_URL_CANDIDATES = [
    "http://localhost:8080/api",  # nginx exposes store API here
    "http://127.0.0.1:8000/api",  # uvicorn (host)
    "http://localhost:8000/api",   # uvicorn (host alt)
    "http://localhost:8001/api",   # uvicorn (container alt)
]
SUBMIT_PATH = "/submissions/"
NORMAL_CARD = "http://localhost:8080/a2a/sales_agent/.well-known/agent-card.json"
MAL_CARD = "http://localhost:8080/a2a/data_harvester_agent/.well-known/agent-card.json"
SG_PROMPTS = 30
ACA_SCENARIOS = 5
RUNS_PER_TYPE = 2  # per agent type (reduce for quick sanity run)
MAX_PARALLEL = 2
POLL_INTERVAL = 10  # seconds
TIMEOUT_SECONDS = 0  # 0 = no timeout (wait until terminal state)

BASE_URL = None  # will be chosen at runtime

LOG_PATH = Path("logs/trust_scores.csv")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


async def submit_once(client: httpx.AsyncClient, agent_card_url: str) -> str:
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
    if resp.status_code in (429, 503):
        raise RateLimitError(f"rate limited: {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    return data["id"]


async def poll_result(client: httpx.AsyncClient, submission_id: str) -> dict:
    deadline = None if TIMEOUT_SECONDS == 0 else asyncio.get_event_loop().time() + TIMEOUT_SECONDS
    while True:
        resp = await client.get(f"{SUBMIT_URL}{submission_id}", timeout=15)
        if resp.status_code in (429, 503):
            raise RateLimitError(f"rate limited during poll: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        state = data.get("state", "")
        auto_decision = data.get("auto_decision")
        trust_score = data.get("trust_score")
        # terminal states
        if state in {"approved", "rejected"} or auto_decision:
            return data
        # background job encodes state into auto_decision: auto_approved / auto_rejected / requires_human_review
        if auto_decision in {"auto_approved", "auto_rejected", "requires_human_review"}:
            return data

        if deadline and asyncio.get_event_loop().time() > deadline:
            data["auto_decision"] = "timeout"
            return data

        await asyncio.sleep(POLL_INTERVAL)


def append_log(row: dict) -> None:
    exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "timestamp",
                    "agent_type",
                    "run_idx",
                    "submission_id",
                    "trust_score",
                    "auto_decision",
                    "state",
                    "sg_blocked",
                    "sg_needs_review",
                    "sg_errors",
                ]
            )
        writer.writerow(
            [
                row.get("timestamp"),
                row.get("agent_type"),
                row.get("run_idx"),
                row.get("submission_id"),
                row.get("trust_score"),
                row.get("auto_decision"),
                row.get("state"),
                row.get("sg_blocked"),
                row.get("sg_needs_review"),
                row.get("sg_errors"),
            ]
        )


async def run_job(agent_type: str, idx: int, card_url: str, client: httpx.AsyncClient, stop_event: asyncio.Event, sem: asyncio.Semaphore):
    if stop_event.is_set():
        return
    async with sem:
        if stop_event.is_set():
            return
        try:
            submission_id = await submit_once(client, card_url)
            result = await poll_result(client, submission_id)
        except RateLimitError as e:
            stop_event.set()
            print(f"[STOP] Rate limit hit at {agent_type} #{idx}: {e}", file=sys.stderr)
            return
        except Exception as e:
            print(f"[ERR] {agent_type} #{idx}: {e}", file=sys.stderr)
            submission_id = "error"
            result = {
                "trust_score": None,
                "auto_decision": "error",
                "state": "failed",
                "score_breakdown": {},
            }

        sg = result.get("score_breakdown", {}).get("stages", {}).get("security", {})
        def _sg(key: str) -> Optional[int]:
            if not isinstance(sg, dict):
                return None
            return sg.get(key) or sg.get(key.replace("needs_review", "needsReview"))

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_type": agent_type,
            "run_idx": idx,
            "submission_id": result.get("id", submission_id),
            "trust_score": result.get("trust_score"),
            "auto_decision": result.get("auto_decision"),
            "state": result.get("state"),
            "sg_blocked": _sg("blocked"),
            "sg_needs_review": _sg("needs_review"),
            "sg_errors": _sg("errors"),
        }
        append_log(row)
        print(f"[OK] {agent_type} #{idx} -> {row['auto_decision']} score={row['trust_score']}")


async def main():
    # Auto-detect base URL
    global BASE_URL, SUBMIT_URL
    for cand in BASE_URL_CANDIDATES:
        url = cand.rstrip("/") + SUBMIT_PATH
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=5) as client:
                resp = await client.get(url)
                if resp.status_code < 400:
                    BASE_URL = cand.rstrip("/")
                    SUBMIT_URL = BASE_URL + SUBMIT_PATH
                    print(f"[INFO] Selected base URL: {BASE_URL}")
                    break
        except Exception as e:
            continue

    if not BASE_URL:
        print("[ERROR] No reachable base URL from candidates", file=sys.stderr)
        return

    jobs = []
    for i in range(1, RUNS_PER_TYPE + 1):
        jobs.append(("normal", i, NORMAL_CARD))
    for i in range(1, RUNS_PER_TYPE + 1):
        jobs.append(("malicious", i, MAL_CARD))

    stop_event = asyncio.Event()
    sem = asyncio.Semaphore(MAX_PARALLEL)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [
            asyncio.create_task(run_job(t, idx, url, client, stop_event, sem))
            for (t, idx, url) in jobs
        ]
        await asyncio.gather(*tasks)

    if stop_event.is_set():
        print("Batch stopped due to rate limiting.", file=sys.stderr)
    else:
        print("Batch completed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted.")
