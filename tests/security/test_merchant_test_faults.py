from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from secure_mediation_agent.merchant.api import (
    TEST_FAULT_HEADER,
    TEST_FAULT_PATH,
    MerchantRuntime,
    create_app,
)
from secure_mediation_agent.merchant.service import PaidBookingMerchant
from secure_mediation_agent.merchant.fault_injection import (
    FulfillmentFaultTarget,
    MerchantTestFaults,
)
from secure_mediation_agent.payment_profiles.a2a import payment_message
from secure_mediation_agent.payment_profiles.registry import ProfileRegistry
from secure_mediation_agent.workflow.approval import AuthorizationService
from secure_mediation_agent.workflow.canonical import canonical_digest, sha256_digest


pytestmark = [pytest.mark.security, pytest.mark.integration]

FAULT_SECRET = "merchant-test-fault-secret-20260817-local-only"
CONTINUATION_ID = "continuation:" + "a" * 32
FAULT_TARGET = FulfillmentFaultTarget(
    order_id="order:fault-boundary",
    task_id="task:fault-boundary",
    operation_id=f"fulfillment-commit:{CONTINUATION_ID}:1",
)


def _runtime(workflow_fixture) -> MerchantRuntime:
    keys = workflow_fixture["keys"]
    profile = ProfileRegistry.load(
        "x402-wire-simulation/1", simulation_key=keys.simulation_signer
    )
    return MerchantRuntime(
        service=PaidBookingMerchant(workflow_fixture["repository"], keys, profile),
        authorization=AuthorizationService(keys.plan_authority),
        paths=workflow_fixture["paths"],
        extension_uri=profile.extension_uri,
    )


def _enable_faults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("DEV_MODE", "true")
    monkeypatch.setenv("MEDIATION_TEST_FAULTS", "true")
    monkeypatch.setenv("MEDIATION_TEST_FAULT_SECRET", FAULT_SECRET)


def _seed_accepted_guarantee(workflow_fixture, service: PaidBookingMerchant):
    profile = service._profile
    now = int(time.time())
    started = service.start_task(
        workflow_id="workflow:fault-boundary",
        plan_digest="sha256:" + "a" * 64,
        task_id=FAULT_TARGET.task_id,
        order_id=FAULT_TARGET.order_id,
        context_id="context:fault-boundary",
        capability_id="capability:fault-start",
        activation={profile.extension_uri},
        issued_at=now,
        expires_at=now + 600,
    )
    project = started.task.status.message.metadata[
        "io.github.taichihiromatsu.secure-mediation.v1"
    ]
    payment_mandate_digest = sha256_digest("payment-mandate:fault")
    authorization_digest = sha256_digest("authorization-envelope:fault")
    guarantee = profile.issue_guarantee(
        {
            "guaranteeId": "guarantee:fault-boundary",
            "iss": "secure-mediator-payment-authority",
            "aud": "a2a-agent:agent-005",
            "operation": "merchant.fulfillment.guarantee",
            "taskId": FAULT_TARGET.task_id,
            "contextId": "context:fault-boundary",
            "orderId": FAULT_TARGET.order_id,
            "quoteId": project["quoteId"],
            "amountMinor": 1250,
            "currency": "USD",
            "payee": "demo-merchant",
            "paymentMandateDigest": payment_mandate_digest,
            "authorizationEnvelopeDigest": authorization_digest,
            "settlementCommitmentId": "settlement-commitment:fault-boundary",
            "jti": "guarantee:fault-boundary",
            "iat": now,
            "nbf": now,
            "exp": now + 600,
        }
    )
    submission = profile.build_guarantee_submission(
        guarantee=guarantee,
        guarantee_digest=sha256_digest(guarantee),
        checkout_mandate_digest=sha256_digest("checkout-mandate:fault"),
        payment_mandate_digest=payment_mandate_digest,
        authorization_envelope_digest=authorization_digest,
    )
    guarantee_message = payment_message(
        task_id=FAULT_TARGET.task_id,
        context_id="context:fault-boundary",
        message_id="message:fault-guarantee",
        status="payment-submitted",
        payload=submission,
        project={"orderId": FAULT_TARGET.order_id, "quoteId": project["quoteId"]},
    )
    service.accept_guarantee(message=guarantee_message)
    receipt = profile.settle_receipt(
        attempt_id="settlement:fault-boundary", success=True
    )
    return payment_message(
        task_id=FAULT_TARGET.task_id,
        context_id="context:fault-boundary",
        message_id="message:fault-commit",
        status="payment-settled",
        payload={
            "schemaVersion": "merchant-fulfillment-commit/1",
            "guaranteeId": "guarantee:fault-boundary",
            "settlementId": "settlement:fault-boundary",
            "settlementReceipt": receipt,
            "settlementReceiptDigest": canonical_digest(receipt),
        },
        project={
            "orderId": FAULT_TARGET.order_id,
            "quoteId": project["quoteId"],
            "simulated": True,
        },
    )


def _commit_request(runtime: MerchantRuntime, message) -> tuple[dict, dict[str, str]]:
    now = int(time.time())
    capability_id = "capability:fault-commit"
    token = runtime.authorization.issue_capability(
        {
            "jti": capability_id,
            "aud": "merchant:demo-merchant",
            "operation": "merchant:guaranteed-fulfillment-commit",
            "approvalId": "approval:fault-boundary",
            "workflowId": "workflow:fault-boundary",
            "planId": "plan:fault-boundary",
            "planDigest": "sha256:" + "a" * 64,
            "orderId": FAULT_TARGET.order_id,
            "taskId": FAULT_TARGET.task_id,
            "idempotencyScope": FAULT_TARGET.operation_id,
            "nonce": "nonce:fault-boundary",
            "iat": now,
            "exp": now + 600,
        }
    )
    body = {
        "jsonrpc": "2.0",
        "id": FAULT_TARGET.operation_id,
        "method": "message/send",
        "params": {
            "action": "merchant:guaranteed-fulfillment-commit",
            "operationId": FAULT_TARGET.operation_id,
            "workflowId": "workflow:fault-boundary",
            "taskId": FAULT_TARGET.task_id,
            "orderId": FAULT_TARGET.order_id,
            "capabilityId": capability_id,
            "message": message.model_dump(
                mode="json", by_alias=True, exclude_none=True
            ),
        },
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "X-A2A-Extensions": runtime.extension_uri,
    }
    return body, headers


def _quick_check(workflow_fixture) -> str:
    repository = workflow_fixture["repository"]
    with repository._connect(repository.paths.merchant) as connection:
        return str(connection.execute("PRAGMA quick_check").fetchone()[0])


def test_fault_route_is_absent_by_default_and_nonlocal_enablement_is_rejected(
    workflow_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "MEDIATION_TEST_FAULTS",
        "MEDIATION_TEST_FAULT_SECRET",
        "APP_ENV",
        "DEV_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    with TestClient(
        create_app(_runtime(workflow_fixture)), client=("127.0.0.1", 51000)
    ) as client:
        assert client.post(TEST_FAULT_PATH, json=FAULT_TARGET.public()).status_code == 404

    monkeypatch.setenv("MEDIATION_TEST_FAULTS", "true")
    monkeypatch.setenv("MEDIATION_TEST_FAULT_SECRET", FAULT_SECRET)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DEV_MODE", "false")
    with pytest.raises(RuntimeError):
        create_app(_runtime(workflow_fixture))


def test_fault_startup_guard_rejects_nonlocal_and_weak_secret(
) -> None:
    script = "deploy/start.sh"
    root = Path(__file__).parents[2]
    base = {
        "PATH": "/usr/bin:/bin",
        "MEDIATION_TEST_FAULTS": "true",
        "MEDIATION_TEST_FAULT_SECRET": FAULT_SECRET,
    }
    nonlocal_result = subprocess.run(
        ["bash", script],
        cwd=str(root),
        env={**base, "APP_ENV": "production", "DEV_MODE": "false"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert nonlocal_result.returncode != 0
    assert "local DEV_MODE only" in nonlocal_result.stdout

    weak_result = subprocess.run(
        ["bash", script],
        cwd=str(root),
        env={
            **base,
            "APP_ENV": "local",
            "DEV_MODE": "true",
            "MEDIATION_TEST_FAULT_SECRET": "too-short",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert weak_result.returncode != 0
    assert "at least 32 characters" in weak_result.stdout
    assert TEST_FAULT_PATH not in (root / "deploy" / "nginx.conf").read_text(
        encoding="utf-8"
    )


def test_fault_control_is_exact_one_shot_under_concurrency() -> None:
    faults = MerchantTestFaults(FAULT_SECRET)
    assert faults.authorized(FAULT_SECRET) is True
    assert faults.authorized("wrong-merchant-test-fault-secret-000000") is False
    assert faults.arm(FAULT_TARGET) is True
    mismatch = FulfillmentFaultTarget(
        order_id="order:other",
        task_id=FAULT_TARGET.task_id,
        operation_id=FAULT_TARGET.operation_id,
    )
    assert faults.consume_if_exact(mismatch) is False
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: faults.consume_if_exact(FAULT_TARGET), range(24)))
    assert results.count(True) == 1
    assert results.count(False) == 23
    status = faults.status()
    assert status["status"] == "consumed"
    assert [event["event"] for event in status["audit"]].count("consumed") == 1
    assert FAULT_SECRET not in json.dumps(status, sort_keys=True)


def test_internal_fault_auth_target_consumption_quick_check_and_restart(
    workflow_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_faults(monkeypatch)
    runtime = _runtime(workflow_fixture)
    commit_message = _seed_accepted_guarantee(
        workflow_fixture, runtime.service
    )
    app = create_app(runtime)
    body, a2a_headers = _commit_request(runtime, commit_message)
    with TestClient(app, client=("127.0.0.1", 52000)) as client:
        wrong_secret = client.post(
            TEST_FAULT_PATH,
            json=FAULT_TARGET.public(),
            headers={TEST_FAULT_HEADER: "wrong-merchant-test-fault-secret-000000"},
        )
        assert wrong_secret.status_code == 403
        assert FAULT_SECRET not in wrong_secret.text

        wrong_order = client.post(
            TEST_FAULT_PATH,
            json={**FAULT_TARGET.public(), "orderId": "order:other"},
            headers={TEST_FAULT_HEADER: FAULT_SECRET},
        )
        assert wrong_order.status_code == 400
        assert wrong_order.json()["error"]["code"] == "TEST_FAULT_TARGET_MISMATCH"

        invalid_operation = client.post(
            TEST_FAULT_PATH,
            json={**FAULT_TARGET.public(), "operationId": "fulfillment-commit:any"},
            headers={TEST_FAULT_HEADER: FAULT_SECRET},
        )
        assert invalid_operation.status_code == 400
        assert (
            invalid_operation.json()["error"]["code"]
            == "TEST_FAULT_TARGET_MISMATCH"
        )

        armed = client.post(
            TEST_FAULT_PATH,
            json=FAULT_TARGET.public(),
            headers={TEST_FAULT_HEADER: FAULT_SECRET},
        )
        assert armed.status_code == 200, armed.text
        assert armed.json() == {"status": "armed", "target": FAULT_TARGET.public()}

        rejected = client.post("/a2a", json=body, headers=a2a_headers)
        assert rejected.status_code == 400
        assert rejected.json()["error"]["code"] == "TEST_FULFILLMENT_REJECTED"

        audit = client.get(
            TEST_FAULT_PATH, headers={TEST_FAULT_HEADER: FAULT_SECRET}
        )
        assert audit.status_code == 200
        assert audit.json()["status"] == "consumed"
        assert [event["event"] for event in audit.json()["audit"]].count("consumed") == 1
        assert FAULT_SECRET not in audit.text

    assert _quick_check(workflow_fixture) == "ok"
    with workflow_fixture["repository"]._connect(
        workflow_fixture["paths"].merchant
    ) as connection:
        guarantee = connection.execute(
            "SELECT state FROM merchant_guarantees_v3 WHERE task_id=?",
            (FAULT_TARGET.task_id,),
        ).fetchone()
        task = connection.execute(
            "SELECT state FROM merchant_tasks_v2 WHERE task_id=?",
            (FAULT_TARGET.task_id,),
        ).fetchone()
    assert guarantee["state"] == "accepted"
    assert task["state"] == "working"

    restarted_runtime = _runtime(workflow_fixture)
    restarted_app = create_app(restarted_runtime)
    retry_body, retry_headers = _commit_request(restarted_runtime, commit_message)
    with TestClient(
        restarted_app, client=("127.0.0.1", 53000)
    ) as restarted_client:
        completed = restarted_client.post(
            "/a2a", json=retry_body, headers=retry_headers
        )
    assert completed.status_code == 200, completed.text
    assert completed.json()["result"]["task"]["status"]["state"] == "completed"
    assert _quick_check(workflow_fixture) == "ok"


def test_fault_control_rejects_non_loopback_even_with_secret(
    workflow_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_faults(monkeypatch)
    app = create_app(_runtime(workflow_fixture))
    with TestClient(app, client=("203.0.113.10", 54000)) as client:
        response = client.post(
            TEST_FAULT_PATH,
            json=FAULT_TARGET.public(),
            headers={TEST_FAULT_HEADER: FAULT_SECRET},
        )
    assert response.status_code == 403
