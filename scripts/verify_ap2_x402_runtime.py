#!/usr/bin/env python3
"""Black-box release verifier for the authenticated mediation simulation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import sqlite3
import uuid
from urllib.parse import urlsplit

import httpx

from secure_mediation_agent.ap2.evidence_verifier import verify_evidence_graph
from secure_mediation_agent.ap2.keys import DemoKeySet
from secure_mediation_agent.workflow.repository import WorkflowRepository


SESSION_COOKIE_NAME = "__Host-payment-session"
CSRF_COOKIE_NAME = "__Host-payment-csrf"
PRIVATE_MARKERS = (
    "checkoutJwt",
    "privatePaymentMaterial",
    "_secureVerifiedIdentityAssertion",
    "secure-verified-identity+jwt",
    "X-Verified-Identity",
    "BEGIN PRIVATE KEY",
    "checkoutMandate",
    "paymentMandate",
    "authorizationEnvelope",
)
TEST_FAULT_SECRET_ENV = "MEDIATION_TEST_FAULT_SECRET"
MERCHANT_TEST_FAULT_URL = (
    "http://127.0.0.1:8005/internal/test/faults/fulfillment-rejection"
)


@dataclass(frozen=True, slots=True)
class BrowserBoundary:
    origin: str
    csrf_token: str
    cookie_header: str
    subject: str


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("gateway URL must have an HTTP(S) origin")
    return f"{parsed.scheme}://{parsed.netloc}"


def _expect(response: httpx.Response, status: int) -> dict[str, object]:
    if response.status_code != status:
        request = response.request
        raise RuntimeError(
            f"unexpected HTTP {response.status_code}: {request.method} {request.url.path}"
        )
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError("response was not a JSON object")
    return value


def _expect_status(response: httpx.Response, status: int) -> None:
    if response.status_code != status:
        request = response.request
        raise RuntimeError(
            f"unexpected HTTP {response.status_code}: {request.method} {request.url.path}"
        )


def _bootstrap(
    client: httpx.Client,
    *,
    gateway_url: str,
    session_cookie: str | None,
) -> BrowserBoundary:
    response = client.get(f"{gateway_url.rstrip('/')}/auth/browser-bootstrap")
    value = _expect(response, 200)
    subject = value.get("subject")
    csrf_token = value.get("csrfToken")
    if not isinstance(subject, str) or not subject:
        raise RuntimeError("browser bootstrap omitted the authenticated subject")
    if not isinstance(csrf_token, str) or not csrf_token:
        raise RuntimeError("browser bootstrap omitted the CSRF token")
    cookie_token = next(
        (
            cookie.value
            for cookie in response.cookies.jar
            if cookie.name == CSRF_COOKIE_NAME
        ),
        None,
    )
    if cookie_token != csrf_token:
        raise RuntimeError("browser bootstrap CSRF cookie/header token mismatch")
    for token in (csrf_token, session_cookie):
        if token and any(character in token for character in "\r\n;"):
            raise RuntimeError("authentication cookie contained an invalid character")
    cookies = []
    if session_cookie:
        cookies.append(f"{SESSION_COOKIE_NAME}={session_cookie}")
    cookies.append(f"{CSRF_COOKIE_NAME}={csrf_token}")
    return BrowserBoundary(
        origin=_origin(gateway_url),
        csrf_token=csrf_token,
        cookie_header="; ".join(cookies),
        subject=subject,
    )


def _mutation_headers(boundary: BrowserBoundary, request_id: str) -> dict[str, str]:
    return {
        "Origin": boundary.origin,
        "X-CSRF-Token": boundary.csrf_token,
        "Cookie": boundary.cookie_header,
        "Idempotency-Key": request_id,
        "X-Request-ID": request_id,
    }


def _turn_body(
    request_id: str, text: str, expected_version: int | None = None
) -> dict[str, object]:
    body: dict[str, object] = {
        "schemaVersion": "mediation-turn-request/1",
        "requestId": request_id,
        "message": {"parts": [{"kind": "text", "text": text}]},
    }
    if expected_version is not None:
        body["expectedVersion"] = expected_version
    return body


def _turn(
    client: httpx.Client,
    *,
    base_url: str,
    boundary: BrowserBoundary,
    request_id: str,
    text: str,
    expected_version: int | None = None,
) -> dict[str, object]:
    response = client.post(
        f"{base_url.rstrip('/')}/v1/turns",
        headers=_mutation_headers(boundary, request_id),
        json=_turn_body(request_id, text, expected_version),
    )
    value = _expect(response, 200)
    _assert_public_safe(value, boundary)
    return value


def _assert_public_safe(value: dict[str, object], boundary: BrowserBoundary) -> None:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    cookie_values = tuple(
        item.split("=", 1)[1]
        for item in boundary.cookie_header.split("; ")
        if "=" in item
    )
    for marker in (
        *PRIVATE_MARKERS,
        *cookie_values,
        boundary.csrf_token,
        boundary.subject,
    ):
        if marker and marker in serialized:
            raise RuntimeError("public mediation response exposed private material")
    if value.get("state") and value.get("view"):
        view = value["view"]
        if not isinstance(view, dict):
            raise RuntimeError("mediation response view was not an object")
        if view.get("simulation") is not True or view.get("conformance") != "NOT CONFORMANT":
            raise RuntimeError("simulation classification changed")


def _assert_state(value: dict[str, object], state: str) -> None:
    view = value.get("view")
    if (
        value.get("state") != state
        or not isinstance(view, dict)
        or view.get("state") != state
        or not isinstance(value.get("version"), int)
    ):
        raise RuntimeError(f"mediation did not reach {state}")


def _assert_approval(value: dict[str, object], kind: str) -> None:
    view = value["view"]
    if not isinstance(view, dict):
        raise RuntimeError("mediation response view was not an object")
    target = view.get("approvalTarget")
    if not isinstance(target, dict) or target.get("approvalKind") != kind:
        raise RuntimeError(f"{kind} approval target was not displayed")
    if target.get("approvalToken") != "承認" or not view.get("approvalTargetDigest"):
        raise RuntimeError(f"{kind} approval target was incomplete")
    if kind == "payment":
        display = target.get("bridgeDisplay")
        if (
            target.get("distinctFromPlanApproval") is not True
            or not isinstance(display, dict)
            or display.get("amountMinor") != 1250
            or display.get("currency") != "USD"
        ):
            raise RuntimeError("payment approval target terms changed")


def _view(client: httpx.Client, base_url: str) -> dict[str, object] | None:
    response = client.get(f"{base_url.rstrip('/')}/v1/view")
    _expect_status(response, 200)
    value = response.json()
    if value is not None and not isinstance(value, dict):
        raise RuntimeError("active view was not a JSON object or null")
    return value


def _assert_current_view(
    client: httpx.Client,
    *,
    base_url: str,
    response: dict[str, object],
    boundary: BrowserBoundary,
) -> None:
    current = _view(client, base_url)
    if current != response.get("view"):
        raise RuntimeError("public view did not match the authoritative turn result")
    if current is None:
        raise RuntimeError("authoritative public view disappeared")
    _assert_public_safe(current, boundary)


def _negative_boundary(
    client: httpx.Client,
    *,
    base_url: str,
    boundary: BrowserBoundary,
    suffix: str,
) -> None:
    before = _view(client, base_url)
    no_csrf_id = f"verify-no-csrf-{suffix}"
    no_csrf = client.post(
        f"{base_url.rstrip('/')}/v1/turns",
        headers={
            "Origin": boundary.origin,
            "Cookie": boundary.cookie_header,
            "Idempotency-Key": no_csrf_id,
            "X-Request-ID": no_csrf_id,
        },
        json=_turn_body(no_csrf_id, "boundary probe"),
    )
    _expect_status(no_csrf, 403)

    tampered_id = f"verify-tampered-csrf-{suffix}"
    tampered = client.post(
        f"{base_url.rstrip('/')}/v1/turns",
        headers={
            **_mutation_headers(boundary, tampered_id),
            "X-CSRF-Token": f"{boundary.csrf_token}-tampered",
        },
        json=_turn_body(tampered_id, "boundary probe"),
    )
    _expect_status(tampered, 403)
    if _view(client, base_url) != before:
        raise RuntimeError("rejected CSRF probes changed authoritative mediation state")


def _bridge_counts(marketplace: str, mediation_session_id: str) -> dict[str, int | str]:
    with sqlite3.connect(marketplace, timeout=10) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT continuation_id,state FROM payment_continuations_v3 "
            "WHERE mediation_session_id=?",
            (mediation_session_id,),
        ).fetchone()
        if row is None:
            return {
                "continuations": 0,
                "approvals": 0,
                "guarantees": 0,
                "settlements": 0,
                "refunds": 0,
                "state": "none",
            }
        continuation_id = row["continuation_id"]
        return {
            "continuations": 1,
            "approvals": connection.execute(
                "SELECT COUNT(*) FROM payment_bridge_approvals_v3 WHERE continuation_id=?",
                (continuation_id,),
            ).fetchone()[0],
            "guarantees": connection.execute(
                "SELECT COUNT(*) FROM payment_guarantees_v3 WHERE continuation_id=?",
                (continuation_id,),
            ).fetchone()[0],
            "settlements": connection.execute(
                "SELECT COUNT(*) FROM payment_bridge_settlements_v3 WHERE continuation_id=?",
                (continuation_id,),
            ).fetchone()[0],
            "refunds": connection.execute(
                "SELECT COUNT(*) FROM payment_bridge_refunds_v3 WHERE continuation_id=?",
                (continuation_id,),
            ).fetchone()[0],
            "state": row["state"],
        }


def _expect_counts(
    observed: dict[str, int | str], expected: dict[str, int | str]
) -> None:
    if observed != expected:
        raise RuntimeError("payment side-effect counts did not match the release contract")


def _run_paid(
    client: httpx.Client,
    *,
    base_url: str,
    boundary: BrowserBoundary,
    marketplace: str,
    suffix: str,
) -> None:
    planned = _turn(
        client,
        base_url=base_url,
        boundary=boundary,
        request_id=f"verify-paid-goal-{suffix}",
        text="有料の外部エージェントに、デモ予約商品を1件シミュレーション購入し、デモの予約確認を発行するよう依頼してください。",
    )
    _assert_state(planned, "WaitingForPlanApproval")
    _assert_approval(planned, "plan")
    mediation_id = str(planned["mediationSessionId"])
    _expect_counts(
        _bridge_counts(marketplace, mediation_id),
        {
            "continuations": 0,
            "approvals": 0,
            "guarantees": 0,
            "settlements": 0,
            "refunds": 0,
            "state": "none",
        },
    )

    nonexact = _turn(
        client,
        base_url=base_url,
        boundary=boundary,
        request_id=f"verify-paid-nonexact-{suffix}",
        text="承認 ",
        expected_version=int(planned["version"]),
    )
    _assert_state(nonexact, "WaitingForPlanApproval")
    if (
        nonexact["version"] != planned["version"]
        or nonexact["view"] != planned["view"]
    ):
        raise RuntimeError("non-exact plan approval changed the mediation state")

    payment = _turn(
        client,
        base_url=base_url,
        boundary=boundary,
        request_id=f"verify-paid-plan-{suffix}",
        text="承認",
        expected_version=int(planned["version"]),
    )
    _assert_state(payment, "WaitingForPaymentApproval")
    _assert_approval(payment, "payment")
    before_payment = {
        "continuations": 1,
        "approvals": 0,
        "guarantees": 0,
        "settlements": 0,
        "refunds": 0,
        "state": "waiting_for_payment_approval",
    }
    _expect_counts(_bridge_counts(marketplace, mediation_id), before_payment)

    rejected = _turn(
        client,
        base_url=base_url,
        boundary=boundary,
        request_id=f"verify-paid-reject-{suffix}",
        text="拒否",
        expected_version=int(payment["version"]),
    )
    _assert_state(rejected, "WaitingForPaymentApproval")
    if rejected["version"] != payment["version"] or rejected["view"] != payment["view"]:
        raise RuntimeError("non-approval payment input changed the mediation state")
    _expect_counts(_bridge_counts(marketplace, mediation_id), before_payment)

    completed = _turn(
        client,
        base_url=base_url,
        boundary=boundary,
        request_id=f"verify-paid-payment-{suffix}",
        text="承認",
        expected_version=int(payment["version"]),
    )
    _assert_state(completed, "Completed")
    _expect_counts(
        _bridge_counts(marketplace, mediation_id),
        {
            "continuations": 1,
            "approvals": 1,
            "guarantees": 1,
            "settlements": 1,
            "refunds": 0,
            "state": "completed",
        },
    )
    _assert_current_view(
        client, base_url=base_url, response=completed, boundary=boundary
    )


def _run_free(
    client: httpx.Client,
    *,
    base_url: str,
    boundary: BrowserBoundary,
    marketplace: str,
    suffix: str,
) -> None:
    planned = _turn(
        client,
        base_url=base_url,
        boundary=boundary,
        request_id=f"verify-free-goal-{suffix}",
        text="東京で2026年9月12日から9月14日まで、2名で宿泊できるホテル候補を検索してください。",
    )
    _assert_state(planned, "WaitingForPlanApproval")
    _assert_approval(planned, "plan")
    completed = _turn(
        client,
        base_url=base_url,
        boundary=boundary,
        request_id=f"verify-free-plan-{suffix}",
        text="承認",
        expected_version=int(planned["version"]),
    )
    _assert_state(completed, "Completed")
    mediation_id = str(completed["mediationSessionId"])
    _expect_counts(
        _bridge_counts(marketplace, mediation_id),
        {
            "continuations": 0,
            "approvals": 0,
            "guarantees": 0,
            "settlements": 0,
            "refunds": 0,
            "state": "none",
        },
    )
    _assert_current_view(
        client, base_url=base_url, response=completed, boundary=boundary
    )


def _refund_fault_target(
    marketplace: str, mediation_session_id: str
) -> dict[str, str]:
    with sqlite3.connect(
        f"file:{marketplace}?mode=ro", uri=True, timeout=10
    ) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT continuation_id,task_id,order_id,state "
            "FROM payment_continuations_v3 WHERE mediation_session_id=?",
            (mediation_session_id,),
        ).fetchone()
    if row is None or row["state"] != "waiting_for_payment_approval":
        raise RuntimeError("refund fault target was not an awaiting continuation")
    values = {
        "orderId": row["order_id"],
        "taskId": row["task_id"],
        "operationId": f"fulfillment-commit:{row['continuation_id']}:1",
    }
    if not all(isinstance(value, str) and value for value in values.values()):
        raise RuntimeError("refund fault target was incomplete")
    return values


def _arm_refund_fault(
    *, secret: str | None, target: dict[str, str]
) -> None:
    if not secret or len(secret) < 32:
        raise RuntimeError(
            f"{TEST_FAULT_SECRET_ENV} must contain at least 32 characters"
        )
    with httpx.Client(
        timeout=10.0, follow_redirects=False, trust_env=False
    ) as client:
        response = client.post(
            MERCHANT_TEST_FAULT_URL,
            headers={"X-Mediation-Test-Fault-Secret": secret},
            json=target,
        )
    value = _expect(response, 200)
    if value != {"status": "armed", "target": target}:
        raise RuntimeError("merchant test fault did not bind the exact target")


def _assert_refund_fault_consumed(
    *, secret: str | None, target: dict[str, str]
) -> None:
    if not secret:
        raise RuntimeError(f"{TEST_FAULT_SECRET_ENV} was unavailable")
    with httpx.Client(
        timeout=10.0, follow_redirects=False, trust_env=False
    ) as client:
        response = client.get(
            MERCHANT_TEST_FAULT_URL,
            headers={"X-Mediation-Test-Fault-Secret": secret},
        )
    value = _expect(response, 200)
    audit = value.get("audit")
    if (
        value.get("status") != "consumed"
        or value.get("target") != target
        or not isinstance(audit, list)
        or len(audit) < 2
    ):
        raise RuntimeError("merchant test fault was not consumed")
    for event, name in zip(audit[-2:], ("armed", "consumed"), strict=True):
        if (
            not isinstance(event, dict)
            or event.get("event") != name
            or event.get("target") != target
        ):
            raise RuntimeError("merchant test fault audit was incomplete")


def _run_refund(
    client: httpx.Client,
    *,
    base_url: str,
    boundary: BrowserBoundary,
    marketplace: str,
    test_fault_secret: str | None,
    suffix: str,
) -> None:
    planned = _turn(
        client,
        base_url=base_url,
        boundary=boundary,
        request_id=f"verify-refund-goal-{suffix}",
        text="paid refund-required booking",
    )
    _assert_state(planned, "WaitingForPlanApproval")
    payment = _turn(
        client,
        base_url=base_url,
        boundary=boundary,
        request_id=f"verify-refund-plan-{suffix}",
        text="承認",
        expected_version=int(planned["version"]),
    )
    _assert_state(payment, "WaitingForPaymentApproval")
    mediation_id = str(payment["mediationSessionId"])
    fault_target = _refund_fault_target(marketplace, mediation_id)
    _arm_refund_fault(
        secret=test_fault_secret,
        target=fault_target,
    )
    pending = _turn(
        client,
        base_url=base_url,
        boundary=boundary,
        request_id=f"verify-refund-payment-{suffix}",
        text="承認",
        expected_version=int(payment["version"]),
    )
    _assert_state(pending, "RefundPending")
    _assert_refund_fault_consumed(
        secret=test_fault_secret,
        target=fault_target,
    )
    pending_counts = {
        "continuations": 1,
        "approvals": 1,
        "guarantees": 1,
        "settlements": 1,
        "refunds": 0,
        "state": "refund_required",
    }
    _expect_counts(_bridge_counts(marketplace, mediation_id), pending_counts)

    nonexact = _turn(
        client,
        base_url=base_url,
        boundary=boundary,
        request_id=f"verify-refund-nonexact-{suffix}",
        text="承認 ",
        expected_version=int(pending["version"]),
    )
    _assert_state(nonexact, "RefundPending")
    if nonexact["version"] != pending["version"] or nonexact["view"] != pending["view"]:
        raise RuntimeError("non-exact refund approval changed the mediation state")
    _expect_counts(_bridge_counts(marketplace, mediation_id), pending_counts)

    refund_request_id = f"verify-refund-approve-{suffix}"
    refunded = _turn(
        client,
        base_url=base_url,
        boundary=boundary,
        request_id=refund_request_id,
        text="承認",
        expected_version=int(pending["version"]),
    )
    _assert_state(refunded, "Refunded")
    replay = _turn(
        client,
        base_url=base_url,
        boundary=boundary,
        request_id=refund_request_id,
        text="承認",
        expected_version=int(pending["version"]),
    )
    if replay != refunded:
        raise RuntimeError("refund approval replay changed the result")
    _expect_counts(
        _bridge_counts(marketplace, mediation_id),
        {
            "continuations": 1,
            "approvals": 1,
            "guarantees": 1,
            "settlements": 1,
            "refunds": 1,
            "state": "refunded",
        },
    )
    _assert_current_view(
        client, base_url=base_url, response=refunded, boundary=boundary
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    base = args.public_url.rstrip("/")
    client = httpx.Client(
        timeout=30.0,
        follow_redirects=False,
        trust_env=False,
        cookies=(
            {SESSION_COOKIE_NAME: args.session_cookie}
            if args.session_cookie
            else None
        ),
    )
    boundary = _bootstrap(
        client,
        gateway_url=args.gateway_url,
        session_cookie=args.session_cookie,
    )
    ready = _expect(client.get(f"{base}/ready"), 200)
    if ready.get("officialX402") != "NOT RUN" or ready.get("onChain") != "NOT RUN":
        raise RuntimeError("simulation classification changed")

    suffix = uuid.uuid4().hex
    _negative_boundary(
        client,
        base_url=base,
        boundary=boundary,
        suffix=suffix,
    )
    for index, path in enumerate(
        (
            "/payment/v1/orders",
            "/paid-agent/ready",
            "/internal/v1/mpp/settle",
            "/v1/workflows",
        )
    ):
        request_id = f"verify-private-{index}-{suffix}"
        response = client.post(
            args.gateway_url.rstrip("/") + path,
            headers=_mutation_headers(boundary, request_id),
            json={},
        )
        _expect_status(response, 404)

    _run_paid(
        client,
        base_url=base,
        boundary=boundary,
        marketplace=args.marketplace,
        suffix=suffix,
    )
    _run_free(
        client,
        base_url=base,
        boundary=boundary,
        marketplace=args.marketplace,
        suffix=suffix,
    )
    _run_refund(
        client,
        base_url=base,
        boundary=boundary,
        marketplace=args.marketplace,
        test_fault_secret=os.environ.get(TEST_FAULT_SECRET_ENV),
        suffix=suffix,
    )
    return {
        "status": "PASS",
        "authentication": "server-owned subject",
        "csrfBoundary": "PASS",
        "paid": "PASS",
        "free": "PASS",
        "refund": "PASS",
        "privateMaterialExposed": False,
        "officialX402": "NOT RUN",
        "onChain": "NOT RUN",
    }


def verify_existing(args: argparse.Namespace) -> dict[str, object]:
    client = httpx.Client(
        timeout=30.0,
        follow_redirects=False,
        trust_env=False,
        cookies=(
            {SESSION_COOKIE_NAME: args.session_cookie}
            if args.session_cookie
            else None
        ),
    )
    boundary = _bootstrap(
        client,
        gateway_url=args.gateway_url,
        session_cookie=args.session_cookie,
    )
    value = _view(client, args.public_url)
    if value is None or value.get("state") not in {"Completed", "Refunded"}:
        raise RuntimeError("mediation did not survive process/container restart")
    _assert_public_safe(value, boundary)
    repository = WorkflowRepository.open(args.marketplace, args.merchant, args.evidence)
    offline = verify_evidence_graph(
        repository, DemoKeySet.load(args.key_dir), args.verify_existing
    )
    return {
        "status": "PASS",
        "workflowId": args.verify_existing,
        "state": value["state"],
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
