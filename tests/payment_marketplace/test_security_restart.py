from __future__ import annotations

import socket

import pytest

from secure_mediation_agent.payment_marketplace.auth import (
    RequestAuthenticationError,
    build_request_auth,
    verify_request_auth,
)
from secure_mediation_agent.payment_marketplace.config import (
    CUSTOMER_KID,
    CUSTOMER_SUBJECT,
)
from secure_mediation_agent.payment_marketplace.merchant_client import (
    EndpointPolicy,
    MerchantClientError,
)
from secure_mediation_agent.payment_marketplace.store import (
    EvidenceAccessDenied,
    MarketplaceStore,
)


NOW = 1_800_000_000


def _auth(body: dict, **overrides):
    values = {
        "method": "POST",
        "path": "/v1/orders",
        "body": body,
        "subject": CUSTOMER_SUBJECT,
        "role": "customer",
        "tenant_id": CUSTOMER_SUBJECT,
        "kid": CUSTOMER_KID,
        "nonce": "security-nonce",
        "timestamp": NOW,
    }
    values.update(overrides)
    return build_request_auth(**values)


def test_request_auth_binds_body_role_tenant_and_expiry() -> None:
    body = {"productId": "demo-paid-booking", "quantity": 1}
    auth = _auth(body)
    verified = verify_request_auth(
        auth,
        method="POST",
        path="/v1/orders",
        body=body,
        expected_role="customer",
        expected_tenant=CUSTOMER_SUBJECT,
        now=NOW,
    )
    assert verified["subject"] == CUSTOMER_SUBJECT

    cases = (
        {"body": {**body, "quantity": 2}},
        {"expected_role": "operator"},
        {"expected_tenant": "another-tenant"},
        {"path": "/v1/orders/other"},
        {"now": NOW + 301},
    )
    for overrides in cases:
        arguments = {
            "method": "POST",
            "path": "/v1/orders",
            "body": body,
            "expected_role": "customer",
            "expected_tenant": CUSTOMER_SUBJECT,
            "now": NOW,
        }
        arguments.update(overrides)
        with pytest.raises(RequestAuthenticationError):
            verify_request_auth(auth, **arguments)


def test_endpoint_policy_blocks_unapproved_and_private_targets(monkeypatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 8005))],
    )
    production = EndpointPolicy(
        allowed_hosts=frozenset({"merchant.example"}),
        allowed_ports=frozenset({443}),
    )
    with pytest.raises(MerchantClientError, match="scheme"):
        production.validate("http://merchant.example:443")
    with pytest.raises(MerchantClientError, match="forbidden address"):
        production.validate("https://merchant.example")

    demo = EndpointPolicy(
        allowed_hosts=frozenset({"127.0.0.1"}),
        allowed_ports=frozenset({8005}),
        allow_loopback=True,
    )
    demo.validate("http://127.0.0.1:8005")
    with pytest.raises(MerchantClientError, match="authority"):
        demo.validate("http://user:password@127.0.0.1:8005")
    with pytest.raises(MerchantClientError, match="not onboarded"):
        demo.validate("http://127.0.0.1:9000")


def test_evidence_is_tenant_isolated_and_survives_restart(tmp_path) -> None:
    business = tmp_path / "business.db"
    evidence = tmp_path / "evidence.db"
    store = MarketplaceStore(business, evidence)
    raw_proof = b'{"closedMandate":"customer-secret-proof"}'
    stored = store.put_evidence(
        intent_id="intent-proof",
        evidence_id="evidence-proof",
        tenant_type="customer",
        tenant_id=CUSTOMER_SUBJECT,
        kind="ap2-x402-proof",
        exact_bytes=raw_proof,
        kid=CUSTOMER_KID,
    )
    assert stored["state"] == "committed"
    metadata = store.get_evidence_metadata("evidence-proof")
    assert metadata is not None
    assert "exact_bytes" not in metadata

    restarted = MarketplaceStore(business, evidence)
    with pytest.raises(EvidenceAccessDenied):
        restarted.read_evidence(
            "evidence-proof",
            actor_id="demo-merchant",
            actor_role="merchant",
            tenant_type="merchant",
            tenant_id="demo-merchant",
        )
    assert (
        restarted.read_evidence(
            "evidence-proof",
            actor_id=CUSTOMER_SUBJECT,
            actor_role="customer",
            tenant_type="customer",
            tenant_id=CUSTOMER_SUBJECT,
        )
        == raw_proof
    )
    with restarted.evidence_transaction(immediate=False) as connection:
        events = connection.execute(
            "SELECT allowed FROM evidence_access_events WHERE evidence_id=? ORDER BY event_id",
            ("evidence-proof",),
        ).fetchall()
    assert [row["allowed"] for row in events] == [0, 1]
