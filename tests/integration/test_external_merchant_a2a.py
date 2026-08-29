from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from secure_mediation_agent.merchant.api import MerchantRuntime, create_app
from secure_mediation_agent.merchant.service import PaidBookingMerchant
from secure_mediation_agent.payment_profiles.a2a import payment_message
from secure_mediation_agent.payment_profiles.registry import ProfileRegistry
from secure_mediation_agent.workflow.approval import AuthorizationService
from secure_mediation_agent.workflow.canonical import canonical_digest, sha256_digest
from secure_mediation_agent.workflow.controller import Identity, WorkflowController
from secure_mediation_agent.workflow.models import MessagePart, WorkflowRequest


pytestmark = [pytest.mark.integration, pytest.mark.security]


def _request(
    authorization: AuthorizationService,
    *,
    audience: str = "merchant:demo-merchant",
    operation: str = "merchant-task:start",
    issued_at: int | None = None,
    expires_at: int | None = None,
):
    now = int(time.time()) if issued_at is None else issued_at
    expiry = now + 600 if expires_at is None else expires_at
    values = {
        "workflowId": "workflow:external-boundary",
        "planDigest": "sha256:" + "a" * 64,
        "taskId": "task:external-boundary",
        "orderId": "order:external-boundary",
        "contextId": "context-external-boundary",
        "capabilityId": "capability:external-boundary",
        "issuedAt": now,
        "expiresAt": expiry,
    }
    token = authorization.issue_capability(
        {
            "jti": values["capabilityId"],
            "aud": audience,
            "operation": operation,
            "approvalId": "approval:external-boundary",
            "workflowId": values["workflowId"],
            "planId": "plan:external-boundary",
            "planDigest": values["planDigest"],
            "orderId": values["orderId"],
            "taskId": values["taskId"],
            "idempotencyScope": f"{operation}/{values['taskId']}",
            "nonce": "nonce-external-boundary",
            "iat": now,
            "exp": expiry,
        }
    )
    return values, token


def _count_tasks(workflow_fixture) -> int:
    with workflow_fixture["repository"]._connect(workflow_fixture["paths"].merchant) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM merchant_tasks_v2").fetchone()[0])


def _merchant_effect_counts(workflow_fixture) -> tuple[int, int, int]:
    with workflow_fixture["repository"]._connect(workflow_fixture["paths"].merchant) as conn:
        return tuple(
            int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "merchant_tasks_v2",
                "merchant_messages_v2",
                "merchant_operations_v2",
            )
        )


def test_external_a2a_rejects_activation_and_capability_before_task_store(
    workflow_fixture,
) -> None:
    keys = workflow_fixture["keys"]
    profile = ProfileRegistry.load(
        "x402-wire-simulation/1", simulation_key=keys.simulation_signer
    )
    authorization = AuthorizationService(keys.plan_authority)
    runtime = MerchantRuntime(
        service=PaidBookingMerchant(workflow_fixture["repository"], keys, profile),
        authorization=authorization,
        paths=workflow_fixture["paths"],
        extension_uri=profile.extension_uri,
    )
    app = create_app(runtime)
    params, token = _request(authorization)
    body = {
        "jsonrpc": "2.0",
        "id": "start:external-boundary",
        "method": "message/send",
        "params": {
            "action": "merchant-task:start",
            "operationId": "start:external-boundary",
            **params,
        },
    }
    with TestClient(app) as client:
        wrong_activation = client.post(
            "/a2a",
            json=body,
            headers={"Authorization": f"Bearer {token}", "X-A2A-Extensions": "urn:wrong"},
        )
        assert wrong_activation.status_code == 409
        assert _count_tasks(workflow_fixture) == 0

        _, forged = _request(authorization, audience="attacker")
        wrong_capability = client.post(
            "/a2a",
            json=body,
            headers={"Authorization": f"Bearer {forged}", "X-A2A-Extensions": profile.extension_uri},
        )
        assert wrong_capability.status_code == 400
        assert _count_tasks(workflow_fixture) == 0

        accepted = client.post(
            "/a2a",
            json=body,
            headers={"Authorization": f"Bearer {token}", "X-A2A-Extensions": profile.extension_uri},
        )
        repeated = client.post(
            "/a2a",
            json=body,
            headers={"Authorization": f"Bearer {token}", "X-A2A-Extensions": profile.extension_uri},
        )
    assert accepted.status_code == 200, accepted.text
    assert repeated.status_code == 200, repeated.text
    assert accepted.json()["result"] == repeated.json()["result"]
    assert _count_tasks(workflow_fixture) == 1


def test_payment_required_quote_and_expiry_are_stable_and_checkout_bound(
    workflow_fixture,
) -> None:
    keys = workflow_fixture["keys"]
    profile = ProfileRegistry.load(
        "x402-wire-simulation/1", simulation_key=keys.simulation_signer
    )
    authorization = AuthorizationService(keys.plan_authority)
    service = PaidBookingMerchant(workflow_fixture["repository"], keys, profile)
    app = create_app(
        MerchantRuntime(
            service=service,
            authorization=authorization,
            paths=workflow_fixture["paths"],
            extension_uri=profile.extension_uri,
        )
    )
    params, token = _request(authorization)
    body = {
        "jsonrpc": "2.0",
        "id": "start:stable-quote",
        "method": "message/send",
        "params": {
            "action": "merchant-task:start",
            "operationId": "start:stable-quote",
            **params,
        },
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "X-A2A-Extensions": profile.extension_uri,
    }
    with TestClient(app) as client:
        first = client.post("/a2a", json=body, headers=headers)
        replay = client.post("/a2a", json=body, headers=headers)

    assert first.status_code == 200, first.text
    assert replay.status_code == 200, replay.text
    assert replay.json()["result"] == first.json()["result"]
    result = first.json()["result"]
    metadata = result["task"]["status"]["message"]["metadata"]
    required = metadata["x402.payment.required"]
    project = metadata["io.github.taichihiromatsu.secure-mediation.v1"]
    expected_quote = f"quote:{params['orderId']}"
    expected_expiry = (
        datetime.fromtimestamp(params["expiresAt"], UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )
    assert required["orderId"] == project["orderId"] == params["orderId"]
    assert required["quoteId"] == project["quoteId"] == expected_quote
    assert required["expiresAt"] == project["expiresAt"] == expected_expiry
    assert "checkoutJwt" not in result and "checkoutHash" not in result
    private = result["privatePaymentMaterial"]
    checkout = service.verify_checkout(
        private["checkoutJwt"],
        workflow_id=params["workflowId"],
        plan_digest=params["planDigest"],
        task_id=params["taskId"],
    )
    assert checkout["quoteId"] == expected_quote
    assert checkout["exp"] == params["expiresAt"]


def test_guaranteed_http_fulfillment_preserves_order_and_quote_on_completed_task(
    workflow_fixture,
) -> None:
    keys = workflow_fixture["keys"]
    profile = ProfileRegistry.load(
        "x402-wire-simulation/1", simulation_key=keys.simulation_signer
    )
    authorization = AuthorizationService(keys.plan_authority)
    app = create_app(
        MerchantRuntime(
            service=PaidBookingMerchant(
                workflow_fixture["repository"], keys, profile
            ),
            authorization=authorization,
            paths=workflow_fixture["paths"],
            extension_uri=profile.extension_uri,
        )
    )
    start_params, start_token = _request(authorization)
    headers = {
        "Authorization": f"Bearer {start_token}",
        "X-A2A-Extensions": profile.extension_uri,
    }
    with TestClient(app) as client:
        started = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "start:http-guarantee",
                "method": "message/send",
                "params": {
                    "action": "merchant-task:start",
                    "operationId": "start:http-guarantee",
                    **start_params,
                },
            },
            headers=headers,
        )
        assert started.status_code == 200, started.text
        project = started.json()["result"]["task"]["status"]["message"][
            "metadata"
        ]["io.github.taichihiromatsu.secure-mediation.v1"]
        now = int(time.time())
        guarantee = profile.issue_guarantee(
            {
                "guaranteeId": "guarantee:http-1",
                "iss": "secure-mediator-payment-authority",
                "aud": "a2a-agent:agent-005",
                "operation": "merchant.fulfillment.guarantee",
                "taskId": start_params["taskId"],
                "contextId": start_params["contextId"],
                "orderId": start_params["orderId"],
                "quoteId": project["quoteId"],
                "amountMinor": 1250,
                "currency": "USD",
                "payee": "demo-merchant",
                "paymentMandateDigest": sha256_digest("payment-mandate:http"),
                "authorizationEnvelopeDigest": sha256_digest(
                    "authorization-envelope:http"
                ),
                "settlementCommitmentId": "settlement-commitment:http-1",
                "jti": "guarantee:http-1",
                "iat": now,
                "nbf": now,
                "exp": now + 600,
            }
        )
        submission = profile.build_guarantee_submission(
            guarantee=guarantee,
            guarantee_digest=sha256_digest(guarantee),
            checkout_mandate_digest=sha256_digest("checkout-mandate:http"),
            payment_mandate_digest=sha256_digest("payment-mandate:http"),
            authorization_envelope_digest=sha256_digest(
                "authorization-envelope:http"
            ),
        )
        guarantee_message = payment_message(
            task_id=start_params["taskId"],
            context_id=start_params["contextId"],
            message_id="message:http-guarantee",
            status="payment-submitted",
            payload=submission,
            project={
                "orderId": start_params["orderId"],
                "quoteId": project["quoteId"],
            },
        )
        guarantee_params, guarantee_token = _request(
            authorization, operation="merchant:payment-guarantee-submit"
        )
        guaranteed = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "submit:http-guarantee",
                "method": "message/send",
                "params": {
                    "action": "merchant:payment-guarantee-submit",
                    "operationId": "submit:http-guarantee",
                    **guarantee_params,
                    "message": guarantee_message.model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    ),
                },
            },
            headers={
                "Authorization": f"Bearer {guarantee_token}",
                "X-A2A-Extensions": profile.extension_uri,
            },
        )
        assert guaranteed.status_code == 200, guaranteed.text
        receipt = profile.settle_receipt(
            attempt_id="settlement:http-1", success=True
        )
        commit_message = payment_message(
            task_id=start_params["taskId"],
            context_id=start_params["contextId"],
            message_id="message:http-commit",
            status="payment-settled",
            payload={
                "schemaVersion": "merchant-fulfillment-commit/1",
                "guaranteeId": "guarantee:http-1",
                "settlementId": "settlement:http-1",
                "settlementReceipt": receipt,
                "settlementReceiptDigest": canonical_digest(receipt),
            },
            project={
                "orderId": start_params["orderId"],
                "quoteId": project["quoteId"],
                "simulated": True,
            },
        )
        commit_params, commit_token = _request(
            authorization, operation="merchant:guaranteed-fulfillment-commit"
        )
        completed = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "commit:http-guarantee",
                "method": "message/send",
                "params": {
                    "action": "merchant:guaranteed-fulfillment-commit",
                    "operationId": "commit:http-guarantee",
                    **commit_params,
                    "message": commit_message.model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    ),
                },
            },
            headers={
                "Authorization": f"Bearer {commit_token}",
                "X-A2A-Extensions": profile.extension_uri,
            },
        )

    assert completed.status_code == 200, completed.text
    task = completed.json()["result"]["task"]
    assert task["id"] == start_params["taskId"]
    assert task["contextId"] == start_params["contextId"]
    assert task["status"]["state"] == "completed"
    completed_project = task["status"]["message"]["metadata"][
        "io.github.taichihiromatsu.secure-mediation.v1"
    ]
    assert completed_project["orderId"] == start_params["orderId"]
    assert completed_project["quoteId"] == project["quoteId"]
    assert completed_project["workflowId"] == start_params["workflowId"]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("workflowId", "workflow:other"),
        ("taskId", "task:other"),
        ("orderId", "order:other"),
    ],
)
def test_external_capability_cannot_replay_across_workflow_task_or_order(
    workflow_fixture, field: str, replacement: str
) -> None:
    keys = workflow_fixture["keys"]
    profile = ProfileRegistry.load(
        "x402-wire-simulation/1", simulation_key=keys.simulation_signer
    )
    authorization = AuthorizationService(keys.plan_authority)
    app = create_app(
        MerchantRuntime(
            service=PaidBookingMerchant(workflow_fixture["repository"], keys, profile),
            authorization=authorization,
            paths=workflow_fixture["paths"],
            extension_uri=profile.extension_uri,
        )
    )
    params, token = _request(authorization)
    params[field] = replacement
    with TestClient(app) as client:
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": f"cross-scope:{field}",
                "method": "message/send",
                "params": {
                    "action": "merchant-task:start",
                    "operationId": f"cross-scope:{field}",
                    **params,
                },
            },
            headers={
                "Authorization": f"Bearer {token}",
                "X-A2A-Extensions": profile.extension_uri,
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "CAPABILITY_BINDING_MISMATCH"
    assert _count_tasks(workflow_fixture) == 0


@pytest.mark.parametrize(
    "action",
    [
        "merchant-task:start",
        "merchant:payment-submit",
        "merchant:payment-guarantee-submit",
        "merchant:fulfillment-prepare",
        "merchant:fulfillment-commit",
        "merchant:guaranteed-fulfillment-commit",
    ],
)
def test_each_private_merchant_operation_rejects_direct_unsigned_bypass(
    workflow_fixture, action: str
) -> None:
    keys = workflow_fixture["keys"]
    profile = ProfileRegistry.load(
        "x402-wire-simulation/1", simulation_key=keys.simulation_signer
    )
    app = create_app(
        MerchantRuntime(
            service=PaidBookingMerchant(workflow_fixture["repository"], keys, profile),
            authorization=AuthorizationService(keys.plan_authority),
            paths=workflow_fixture["paths"],
            extension_uri=profile.extension_uri,
        )
    )
    before = _merchant_effect_counts(workflow_fixture)
    with TestClient(app) as client:
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": f"unsigned:{action}",
                "method": "message/send",
                "params": {"action": action},
            },
            headers={"X-A2A-Extensions": profile.extension_uri},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "CAPABILITY_MISSING"
    assert _merchant_effect_counts(workflow_fixture) == before


@pytest.mark.parametrize("attack", ["expired", "wrong-role", "tampered"])
def test_integrated_capability_attack_matrix_has_zero_merchant_effects(
    workflow_fixture, attack: str
) -> None:
    keys = workflow_fixture["keys"]
    profile = ProfileRegistry.load(
        "x402-wire-simulation/1", simulation_key=keys.simulation_signer
    )
    verifier = AuthorizationService(keys.plan_authority)
    signer = verifier if attack != "wrong-role" else AuthorizationService(keys.merchant)
    if attack == "expired":
        now = int(time.time())
        params, token = _request(
            signer, issued_at=now - 1200, expires_at=now - 600
        )
    else:
        params, token = _request(signer)
    if attack == "tampered":
        header, payload, signature = token.split(".")
        signature = ("A" if signature[0] != "A" else "B") + signature[1:]
        token = ".".join((header, payload, signature))
    app = create_app(
        MerchantRuntime(
            service=PaidBookingMerchant(workflow_fixture["repository"], keys, profile),
            authorization=verifier,
            paths=workflow_fixture["paths"],
            extension_uri=profile.extension_uri,
        )
    )
    before = _merchant_effect_counts(workflow_fixture)
    with TestClient(app) as client:
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": f"attack:{attack}",
                "method": "message/send",
                "params": {
                    "action": "merchant-task:start",
                    "operationId": f"attack:{attack}",
                    **params,
                },
            },
            headers={
                "Authorization": f"Bearer {token}",
                "X-A2A-Extensions": profile.extension_uri,
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "CAPABILITY_INVALID"
    assert _merchant_effect_counts(workflow_fixture) == before


def test_integrated_revoked_capability_is_rejected_before_replay(
    workflow_fixture,
) -> None:
    repository = workflow_fixture["repository"]
    keys = workflow_fixture["keys"]
    controller = WorkflowController(repository, keys)
    created = controller.create(
        WorkflowRequest(goal="revoked capability fixture"),
        identity=Identity("demo-tenant", "demo-customer"),
        session_id="revoked-session",
        context_id="revoked-context",
        idempotency_key="revoked-create",
    )
    payment = controller.message(
        created.workflow_id,
        [MessagePart(kind="text", text="承認")],
        identity=Identity("demo-tenant", "demo-customer"),
        message_id="revoked-plan",
        idempotency_key="revoked-plan",
    )
    operation_id = f"start:{payment.task_id}"
    outbox = repository.outbox_row(operation_id)
    capability, token = repository.capability_for_operation(
        created.workflow_id, "merchant-task:start"
    )
    assert outbox is not None
    repository.invalidate_capability(capability["capability_id"])
    profile = controller.profile
    app = create_app(
        MerchantRuntime(
            service=PaidBookingMerchant(repository, keys, profile),
            authorization=AuthorizationService(keys.plan_authority),
            paths=workflow_fixture["paths"],
            extension_uri=profile.extension_uri,
        )
    )
    before = _merchant_effect_counts(workflow_fixture)
    params = outbox["payload"]
    with TestClient(app) as client:
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "revoked:merchant-start",
                "method": "message/send",
                "params": {
                    "action": "merchant-task:start",
                    "operationId": operation_id,
                    "capabilityId": capability["capability_id"],
                    **params,
                },
            },
            headers={
                "Authorization": f"Bearer {token}",
                "X-A2A-Extensions": profile.extension_uri,
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "CAPABILITY_REVOKED"
    assert _merchant_effect_counts(workflow_fixture) == before


@pytest.mark.parametrize(
    "legacy_path", ["/payment/v1/orders", "/paid-agent/ready", "/internal/v1/mpp/settle", "/v1/workflows"]
)
def test_external_merchant_has_no_legacy_or_control_plane_routes(
    workflow_fixture, legacy_path: str
) -> None:
    keys = workflow_fixture["keys"]
    profile = ProfileRegistry.load(
        "x402-wire-simulation/1", simulation_key=keys.simulation_signer
    )
    app = create_app(
        MerchantRuntime(
            service=PaidBookingMerchant(workflow_fixture["repository"], keys, profile),
            authorization=AuthorizationService(keys.plan_authority),
            paths=workflow_fixture["paths"],
            extension_uri=profile.extension_uri,
        )
    )
    with TestClient(app) as client:
        response = client.post(legacy_path, json={})
    assert response.status_code == 404
