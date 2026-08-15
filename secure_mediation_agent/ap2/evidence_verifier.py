"""Offline verification of the committed AP2/simulation evidence graph."""

from __future__ import annotations

import json
from typing import Any

from ap2.sdk.jwt_helper import verify_jwt

from secure_mediation_agent.payment_profiles.simulation_v1 import SimulationV1Profile
from secure_mediation_agent.workflow.approval import AuthorizationService
from secure_mediation_agent.workflow.canonical import sha256_digest
from secure_mediation_agent.workflow.repository import WorkflowRepository

from .credential_provider import CredentialProvider
from .keys import DemoKeySet, public_key
from .receipts import Ap2ReceiptFactory
from .verification import verify_role_jwt, verify_terminal_presentation


def verify_evidence_graph(
    repository: WorkflowRepository,
    keys: DemoKeySet,
    workflow_id: str,
) -> dict[str, Any]:
    workflow = repository.get_workflow(workflow_id)
    if workflow["state"] not in {"completed", "payment_failed", "refund_required", "refunded"}:
        raise ValueError("workflow has no definitive evidence graph")
    artifacts = repository.artifact_refs(workflow_id)
    by_kind = {item["kind"]: item for item in artifacts}
    checked: list[str] = []

    authorization = AuthorizationService(keys.plan_authority)
    with repository._connect(repository.paths.marketplace) as conn:
        plan_approval = conn.execute(
            "SELECT * FROM plan_approvals WHERE workflow_id=?", (workflow_id,)
        ).fetchone()
        capabilities = list(
            conn.execute(
                "SELECT * FROM downstream_capabilities WHERE workflow_id=? ORDER BY iat,capability_id",
                (workflow_id,),
            )
        )
    plan_token = repository.read_evidence(
        plan_approval["authorization_evidence_id"],
        actor_id="offline-verifier",
        actor_role="operator",
        tenant_id=workflow["tenant_id"],
    ).decode()
    if sha256_digest(plan_token) != plan_approval["authorization_digest"]:
        raise ValueError("plan authorization digest mismatch")
    plan_claims = verify_jwt(plan_token, public_key(keys.plan_authority))
    authorization.verify(
        plan_token,
        expected_type="secure-plan-authorization+jwt",
        audience="secure-mediation-workflow",
        now=plan_claims["iat"],
    )
    if plan_claims.get("workflowId") != workflow_id or plan_claims.get("planDigest") != workflow["plan_digest"]:
        raise ValueError("plan authorization binding mismatch")
    checked.append("signature:plan-authorization")
    for capability in capabilities:
        token = repository.read_evidence(
            capability["evidence_id"],
            actor_id="offline-verifier",
            actor_role="operator",
            tenant_id=workflow["tenant_id"],
        ).decode()
        if sha256_digest(token) != capability["evidence_digest"]:
            raise ValueError("capability digest mismatch")
        claims = authorization.verify(
            token,
            expected_type="secure-downstream-capability+jwt",
            audience=capability["audience"],
            operation=capability["operation"],
            now=capability["iat"],
        )
        if claims.get("workflowId") != workflow_id or claims.get("planDigest") != workflow["plan_digest"]:
            raise ValueError("capability workflow/plan binding mismatch")
    checked.append(f"signatures:capabilities:{len(capabilities)}")

    def exact(kind: str) -> bytes:
        item = by_kind[kind]
        value = repository.read_evidence(
            item["evidence_id"],
            actor_id="offline-verifier",
            actor_role="operator",
            tenant_id=workflow["tenant_id"],
        )
        if sha256_digest(value) != item["evidence_digest"]:
            raise ValueError(f"{kind} evidence digest mismatch")
        checked.append(f"digest:{kind}")
        return value

    with repository._connect(repository.paths.marketplace) as conn:
        original_task_ref = conn.execute(
            "SELECT task_evidence_id,task_evidence_digest FROM merchant_task_mirrors WHERE task_id=?",
            (workflow["merchant_task_id"],),
        ).fetchone()
    original_task_bytes = repository.read_evidence(
        original_task_ref["task_evidence_id"],
        actor_id="offline-verifier",
        actor_role="operator",
        tenant_id=workflow["tenant_id"],
    )
    if sha256_digest(original_task_bytes) != original_task_ref["task_evidence_digest"]:
        raise ValueError("original Merchant Task digest mismatch")
    original_task = json.loads(original_task_bytes)
    project = original_task["status"]["message"]["metadata"][
        "io.github.taichihiromatsu.secure-mediation.v1"
    ]
    checked.append("digest:original-merchant-task")
    checkout = exact("checkout-jwt").decode()
    checkout_claims = verify_role_jwt(
        checkout,
        public_key=public_key(keys.merchant),
        expected_issuer="demo-merchant",
        expected_kid=keys.merchant.get("kid"),
    )
    for name, value in {
        "workflowId": workflow_id,
        "planDigest": workflow["plan_digest"],
        "taskId": workflow["merchant_task_id"],
        "orderId": workflow["order_id"],
        "amount": 1250,
        "currency": "USD",
    }.items():
        if checkout_claims.get(name) != value:
            raise ValueError(f"Checkout binding mismatch: {name}")
    checked.append("signature:checkout-jwt")

    checkout_mandate = exact("checkout-mandate").decode()
    checkout_leaf = verify_terminal_presentation(
        checkout_mandate,
        root_key=keys.user_root,
        audience="demo-merchant",
        nonce=project["checkoutMandateChallenge"]["nonce"],
        expected_vct="mandate.checkout.1",
    )
    if checkout_leaf.get("checkout_jwt") != checkout:
        raise ValueError("Checkout Mandate does not reference exact Checkout")
    checked.append("signature:checkout-mandate")

    payment_mandate = exact("payment-mandate").decode()
    payment_leaf = verify_terminal_presentation(
        payment_mandate,
        root_key=keys.user_root,
        audience="demo-credential-provider",
        nonce=project["paymentMandateChallenge"]["nonce"],
        expected_vct="mandate.payment.1",
    )
    if payment_leaf.get("payee", {}).get("id") != "demo-merchant":
        raise ValueError("Payment Mandate payee mismatch")
    checked.append("signature:payment-mandate")

    proof = exact("simulation-payload").decode()
    profile = SimulationV1Profile(keys.simulation_signer)
    proof_claims = profile.verify_proof(proof, profile.public_key())
    if proof_claims.get("walletSigned") is not False or proof_claims.get("taskId") != workflow["merchant_task_id"]:
        raise ValueError("simulation proof classification or task mismatch")
    checked.append("signature:simulation-payload")

    credential = exact("payment-credential").decode()
    credential_claims = CredentialProvider(keys).verify(
        credential,
        task_id=workflow["merchant_task_id"],
        payload_digest=sha256_digest(proof),
    )
    if credential_claims.get("paymentMandateDigest") != sha256_digest(payment_mandate):
        raise ValueError("credential Payment Mandate binding mismatch")
    checked.append("signature:payment-credential")

    if "payment-receipt" in by_kind:
        token = exact("payment-receipt").decode()
        Ap2ReceiptFactory.verify_payment(
            token,
            public_key(keys.mpp),
            by_kind["payment-receipt"]["reference_digest"],
        )
        checked.append("signature:payment-receipt")
    if "checkout-receipt" in by_kind:
        token = exact("checkout-receipt").decode()
        Ap2ReceiptFactory.verify_checkout(
            token,
            public_key(keys.merchant),
            by_kind["checkout-receipt"]["reference_digest"],
        )
        checked.append("signature:checkout-receipt")

    with repository._connect(repository.paths.marketplace) as conn:
        profile_receipts = list(
            conn.execute(
                "SELECT * FROM profile_receipts WHERE task_id=? ORDER BY ordinal",
                (workflow["merchant_task_id"],),
            )
        )
    for receipt in profile_receipts:
        value = repository.read_evidence(
            receipt["evidence_id"],
            actor_id="offline-verifier",
            actor_role="operator",
            tenant_id=workflow["tenant_id"],
        )
        if sha256_digest(value) != receipt["evidence_digest"]:
            raise ValueError("selected-profile receipt digest mismatch")
        payload = json.loads(value)
        if payload.get("simulated") is not True or payload.get("network") != "demo:local":
            raise ValueError("selected-profile receipt classification mismatch")
        if payload.get("transaction", "").startswith("0x"):
            raise ValueError("simulation receipt contains an on-chain transaction")
    checked.append(f"simulation-receipts:{len(profile_receipts)}")

    with repository._connect(repository.paths.marketplace) as conn:
        snapshots = {
            row["snapshot_id"]: dict(row)
            for row in conn.execute("SELECT * FROM trust_snapshots")
        }
    for item in artifacts:
        snapshot = snapshots.get(item["trust_snapshot_id"])
        if snapshot is None or snapshot["kid"] != item["kid"]:
            raise ValueError(f"trust snapshot missing for {item['kind']}")
        trust_bytes = repository.read_evidence(
            snapshot["jwks_evidence_id"],
            actor_id="offline-verifier",
            actor_role="operator",
            tenant_id=workflow["tenant_id"],
        )
        if sha256_digest(trust_bytes) != snapshot["jwks_evidence_digest"]:
            raise ValueError("trust snapshot digest mismatch")
        if json.loads(trust_bytes)["kid"] != item["kid"]:
            raise ValueError("trust snapshot kid mismatch")
    checked.append("trust-snapshots")

    return {
        "status": "PASS",
        "workflowId": workflow_id,
        "profile": "x402-wire-simulation/1",
        "ap2": "v0.2 Human Present demo",
        "x402": "v0.1 wire-shape fixture (NOT CONFORMANT)",
        "railMode": "simulated; no real asset or on-chain transaction",
        "checked": checked,
    }
