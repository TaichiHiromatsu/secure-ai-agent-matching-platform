#!/usr/bin/env python3
"""Black-box release verifier for the public two-approval simulation path."""

from __future__ import annotations

import argparse
import json
import sqlite3
import uuid

import httpx

from secure_mediation_agent.ap2.evidence_verifier import verify_evidence_graph
from secure_mediation_agent.ap2.keys import DemoKeySet
from secure_mediation_agent.workflow.repository import WorkflowRepository

SESSION_COOKIE_NAME = "__Host-payment-session"


def _headers(key: str) -> dict[str, str]:
    return {"Idempotency-Key": key}


def _expect(response: httpx.Response, status: int) -> dict[str, object]:
    if response.status_code != status:
        raise RuntimeError(f"unexpected HTTP {response.status_code}: {response.text}")
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError("response was not a JSON object")
    return value


def run(args: argparse.Namespace) -> dict[str, object]:
    base = args.public_url.rstrip("/")
    cookies = (
        {SESSION_COOKIE_NAME: args.session_cookie} if args.session_cookie else None
    )
    client = httpx.Client(timeout=30.0, cookies=cookies)
    ready = _expect(client.get(f"{base}/ready"), 200)
    if ready.get("officialX402") != "NOT RUN" or ready.get("onChain") != "NOT RUN":
        raise RuntimeError("simulation classification changed")
    for path in ("/payment/v1/orders", "/paid-agent/ready", "/internal/v1/mpp/settle", "/v1/workflows"):
        response = client.post(args.gateway_url.rstrip("/") + path, json={})
        if response.status_code != 404:
            raise RuntimeError(f"legacy/internal route was exposed: {path}")

    suffix = uuid.uuid4().hex
    created = _expect(
        client.post(
            f"{base}/v1/workflows",
            headers=_headers(f"verify-create-{suffix}"),
            json={
                "sessionId": f"verify-session-{suffix}",
                "contextId": f"verify-context-{suffix}",
                "request": {"goal": "release verification booking", "paymentRequired": True},
            },
        ),
        200,
    )
    workflow_id = str(created["workflowId"])
    rejected_variant = _expect(
        client.post(
            f"{base}/v1/workflows/{workflow_id}/messages",
            headers=_headers(f"verify-nonexact-{suffix}"),
            json={"messageId": f"nonexact-{suffix}", "parts": [{"kind": "text", "text": "承認 "}]},
        ),
        409,
    )
    if rejected_variant.get("error", {}).get("code") != "APPROVAL_EXACT_TOKEN_REQUIRED":
        raise RuntimeError("non-exact approval was not rejected")
    payment = _expect(
        client.post(
            f"{base}/v1/workflows/{workflow_id}/messages",
            headers=_headers(f"verify-plan-{suffix}"),
            json={
                "messageId": f"plan-{suffix}",
                "expectedVersion": created["version"],
                "parts": [{"kind": "text", "text": "承認"}],
            },
        ),
        200,
    )
    if payment.get("state") != "payment_approval_required" or "approval expiry (UTC)" not in str(payment.get("renderedText")):
        raise RuntimeError("payment approval view is incomplete")
    completed = _expect(
        client.post(
            f"{base}/v1/workflows/{workflow_id}/messages",
            headers=_headers(f"verify-payment-{suffix}"),
            json={
                "messageId": f"payment-{suffix}",
                "expectedVersion": payment["version"],
                "parts": [{"kind": "text", "text": "承認"}],
            },
        ),
        200,
    )
    if completed.get("state") != "completed":
        raise RuntimeError("two-approval workflow did not complete")

    reject_suffix = uuid.uuid4().hex
    reject_created = _expect(
        client.post(
            f"{base}/v1/workflows",
            headers=_headers(f"verify-reject-create-{reject_suffix}"),
            json={
                "sessionId": f"verify-reject-session-{reject_suffix}",
                "contextId": f"verify-reject-context-{reject_suffix}",
                "request": {"goal": "release rejection booking", "paymentRequired": True},
            },
        ),
        200,
    )
    reject_payment = _expect(
        client.post(
            f"{base}/v1/workflows/{reject_created['workflowId']}/messages",
            headers=_headers(f"verify-reject-plan-{reject_suffix}"),
            json={"messageId": f"reject-plan-{reject_suffix}", "parts": [{"kind": "text", "text": "承認"}]},
        ),
        200,
    )
    cancelled = _expect(
        client.post(
            f"{base}/v1/workflows/{reject_created['workflowId']}/messages",
            headers=_headers(f"verify-reject-payment-{reject_suffix}"),
            json={"messageId": f"reject-payment-{reject_suffix}", "expectedVersion": reject_payment["version"], "parts": [{"kind": "text", "text": "拒否"}]},
        ),
        200,
    )
    if cancelled.get("state") != "cancelled":
        raise RuntimeError("payment rejection did not cancel")

    repository = WorkflowRepository.open(args.marketplace, args.merchant, args.evidence)
    offline = verify_evidence_graph(repository, DemoKeySet.load(args.key_dir), workflow_id)
    with sqlite3.connect(args.marketplace) as conn:
        unfinished = conn.execute(
            "SELECT COUNT(*) FROM outbox WHERE workflow_id=? AND status<>'done'", (workflow_id,)
        ).fetchone()[0]
    if unfinished:
        raise RuntimeError("completed workflow retained unfinished outbox rows")
    return {
        "status": "PASS",
        "workflowId": workflow_id,
        "rejectedWorkflowId": reject_created["workflowId"],
        "offlineEvidence": offline.get("status", "PASS"),
        "officialX402": "NOT RUN",
        "onChain": "NOT RUN",
    }


def verify_existing(args: argparse.Namespace) -> dict[str, object]:
    client = httpx.Client(
        timeout=30.0,
        cookies=(
            {SESSION_COOKIE_NAME: args.session_cookie}
            if args.session_cookie
            else None
        ),
    )
    value = _expect(
        client.get(f"{args.public_url.rstrip('/')}/v1/workflows/{args.verify_existing}"),
        200,
    )
    if value.get("state") != "completed":
        raise RuntimeError("workflow did not survive process/container restart")
    repository = WorkflowRepository.open(args.marketplace, args.merchant, args.evidence)
    offline = verify_evidence_graph(
        repository, DemoKeySet.load(args.key_dir), args.verify_existing
    )
    return {
        "status": "PASS",
        "workflowId": args.verify_existing,
        "state": "completed",
        "offlineEvidence": offline.get("status", "PASS"),
        "restart": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-url", default="http://127.0.0.1:8080/mediation-api")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8080")
    parser.add_argument("--session-cookie")
    parser.add_argument("--marketplace", default="/app/payment-data/marketplace.db")
    parser.add_argument("--merchant", default="/app/payment-data/paid-agent.db")
    parser.add_argument("--evidence", default="/app/payment-evidence/evidence.db")
    parser.add_argument("--key-dir", default="/run/secrets/ap2-demo")
    parser.add_argument("--verify-existing")
    args = parser.parse_args()
    result = verify_existing(args) if args.verify_existing else run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
