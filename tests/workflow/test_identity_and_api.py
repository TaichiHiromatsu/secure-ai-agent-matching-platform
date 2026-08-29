from __future__ import annotations

from fastapi.testclient import TestClient

from secure_mediation_agent.identity import issue_identity_assertion


def _headers(assertion: str, key: str) -> dict[str, str]:
    return {"X-Verified-Identity": assertion, "Idempotency-Key": key}


def _create(client: TestClient, assertion: str) -> dict[str, object]:
    response = client.post(
        "/v1/workflows",
        headers=_headers(assertion, "create-api-0001"),
        json={
            "sessionId": "session-api",
            "contextId": "context-api",
            "request": {"goal": "デモ予約", "paymentRequired": True},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_signed_identity_exact_approvals_and_readiness(workflow_fixture) -> None:
    with TestClient(workflow_fixture["app"]) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200
        readiness = ready.json()
        assert readiness["target"] == "explicit-durable-single-host-single-container"
        assert readiness["checks"]["dataDurableVolume"] is True
        assert readiness["checks"]["evidenceDurableVolume"] is True
        assert readiness["durableVolumeMarker"] == "PASS"
        assert readiness["evidenceDurableVolumeMarker"] == "PASS"
        assert "durability" not in readiness
        assert readiness["officialX402"] == "NOT RUN"
        assert readiness["onChain"] == "NOT RUN"

        view = _create(client, workflow_fixture["assertion"])
        assert view["state"] == "plan_approval_required"
        workflow_id = view["workflowId"]
        first = client.post(
            f"/v1/workflows/{workflow_id}/messages",
            headers=_headers(workflow_fixture["assertion"], "message-api-0001"),
            json={
                "messageId": "message:plan",
                "expectedVersion": view["version"],
                "parts": [{"kind": "text", "text": "承認"}],
            },
        )
        assert first.status_code == 200, first.text
        payment = first.json()
        assert payment["state"] == "payment_approval_required"
        second = client.post(
            f"/v1/workflows/{workflow_id}/messages",
            headers=_headers(workflow_fixture["assertion"], "message-api-0002"),
            json={
                "messageId": "message:payment",
                "expectedVersion": payment["version"],
                "parts": [{"kind": "text", "text": "承認"}],
            },
        )
        assert second.status_code == 200, second.text
        completed = second.json()
        assert completed["state"] == "completed"
        assert completed["profile"] == "x402-wire-simulation/1"
        assert "NOT CONFORMANT" in completed["x402Label"]
        assert "no real asset or on-chain transaction" in completed["railLabel"]

    counts = workflow_fixture["repository"].counts(str(workflow_id))
    assert counts == {
        "planApprovals": 1,
        "paymentApprovals": 1,
        "paymentArtifacts": 8,
        "settlements": 1,
        "refunds": 0,
    }


def test_forged_or_missing_identity_is_rejected_before_side_effects(workflow_fixture) -> None:
    bad_mapping = issue_identity_assertion(
        workflow_fixture["keys"].service_auth,
        subject="attacker",
        tenant_id="other-tenant",
        customer_id="other-customer",
    )
    body = {
        "sessionId": "forged-session",
        "contextId": "forged-context",
        "request": {"goal": "デモ予約"},
    }
    with TestClient(workflow_fixture["app"]) as client:
        missing = client.post(
            "/v1/workflows", headers={"Idempotency-Key": "missing-identity"}, json=body
        )
        forged = client.post(
            "/v1/workflows", headers=_headers(bad_mapping, "forged-identity"), json=body
        )
    assert missing.status_code == 403
    assert forged.status_code == 403
    assert missing.json()["error"]["code"] == "TENANT_BINDING_MISMATCH"
    with workflow_fixture["repository"]._connect(workflow_fixture["paths"].marketplace) as conn:
        assert conn.execute("SELECT COUNT(*) FROM workflows").fetchone()[0] == 0


def test_non_exact_approval_has_zero_business_side_effects_and_can_retry(workflow_fixture) -> None:
    with TestClient(workflow_fixture["app"]) as client:
        view = _create(client, workflow_fixture["assertion"])
        body = {
            "messageId": "message:not-exact",
            "parts": [{"kind": "text", "text": "承認 "}],
        }
        for _ in range(2):
            response = client.post(
                f"/v1/workflows/{view['workflowId']}/messages",
                headers=_headers(workflow_fixture["assertion"], "message-not-exact"),
                json=body,
            )
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "APPROVAL_EXACT_TOKEN_REQUIRED"
    counts = workflow_fixture["repository"].counts(str(view["workflowId"]))
    assert counts["planApprovals"] == 0
    assert counts["paymentApprovals"] == 0
    assert counts["settlements"] == 0
