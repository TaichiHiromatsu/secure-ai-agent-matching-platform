from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from secure_mediation_agent.merchant.api import MerchantRuntime, create_app
from secure_mediation_agent.merchant.service import PaidBookingMerchant
from secure_mediation_agent.payment_profiles.registry import ProfileRegistry
from secure_mediation_agent.workflow.approval import AuthorizationService
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
        "merchant:fulfillment-prepare",
        "merchant:fulfillment-commit",
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
