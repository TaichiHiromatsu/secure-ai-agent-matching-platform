from __future__ import annotations

import json
import time

import pytest

from secure_mediation_agent.ap2.keys import ROLE_KIDS, public_key
from secure_mediation_agent.ap2.receipts import Ap2ReceiptFactory
from secure_mediation_agent.workflow.controller import Identity, WorkflowController
from secure_mediation_agent.workflow.errors import DomainError
from secure_mediation_agent.workflow.models import MessagePart, WorkflowRequest


pytestmark = [pytest.mark.security, pytest.mark.contract_ap2]
IDENTITY = Identity("demo-tenant", "demo-customer")


@pytest.mark.parametrize(
    ("role", "error"),
    [
        ("credential-provider", "invalid_credential"),
        ("payment-network", "constraint_unresolved"),
        ("merchant-payment-processor", "settlement_failed"),
    ],
)
def test_payment_roles_emit_signed_error_receipts(role, error, workflow_fixture) -> None:
    reference = "sha256:" + "a" * 64
    token = Ap2ReceiptFactory.payment(
        key=workflow_fixture["keys"].mpp,
        reference=reference,
        issued_at=int(time.time()),
        payment_id=f"payment:error:{role}",
        simulation_reference=f"sim:error:{role}",
        success=False,
        error=error,
        error_description=f"{role} rejected the payment input.",
    )
    receipt = Ap2ReceiptFactory.verify_payment(
        token, public_key(workflow_fixture["keys"].mpp), reference
    )
    assert receipt.root.status == "Error"
    assert receipt.root.iss == "demo-mpp"
    assert receipt.root.error == error


def test_merchant_emits_signed_checkout_error_receipt(workflow_fixture) -> None:
    reference = "sha256:" + "b" * 64
    token = Ap2ReceiptFactory.checkout(
        key=workflow_fixture["keys"].merchant,
        reference=reference,
        issued_at=int(time.time()),
        order_id="order:error",
        success=False,
        error="invalid_mandate",
    )
    receipt = Ap2ReceiptFactory.verify_checkout(
        token, public_key(workflow_fixture["keys"].merchant), reference
    )
    assert receipt.root.status == "Error"
    assert receipt.root.iss == "demo-merchant"
    assert receipt.root.error == "invalid_mandate"


@pytest.mark.parametrize("fault", [None, "failed", "unknown"])
def test_public_success_failure_and_timeout_outputs_do_not_leak_signing_material(
    workflow_fixture, fault: str | None
) -> None:
    controller = WorkflowController(
        workflow_fixture["repository"], workflow_fixture["keys"], rail_fault=fault
    )
    suffix = fault or "success"
    created = controller.create(
        WorkflowRequest(goal=f"output scan {suffix}"),
        identity=IDENTITY,
        session_id=f"session-output-{suffix}",
        context_id=f"context-output-{suffix}",
        idempotency_key=f"create-output-{suffix}",
    )
    payment = controller.message(
        created.workflow_id,
        [MessagePart(kind="text", text="承認")],
        identity=IDENTITY,
        message_id=f"plan-output-{suffix}",
        idempotency_key=f"plan-output-{suffix}",
    )
    terminal = controller.message(
        created.workflow_id,
        [MessagePart(kind="text", text="承認")],
        identity=IDENTITY,
        message_id=f"payment-output-{suffix}",
        idempotency_key=f"payment-output-{suffix}",
    )
    output = json.dumps(
        {
            "payment": payment.model_dump(mode="json", by_alias=True),
            "terminal": terminal.model_dump(mode="json", by_alias=True),
            "error": DomainError(
                "AP2_CREDENTIAL_INVALID", "Credential rejected.", created.workflow_id
            ).envelope(),
        },
        sort_keys=True,
    )
    forbidden_names = (
        "checkoutJwt",
        "checkoutMandate",
        "paymentMandate",
        "paymentCredential",
        "authorizationPayload",
        "privateKey",
        "rawProof",
    )
    assert all(name not in output for name in forbidden_names)
    for role in ROLE_KIDS:
        key = getattr(workflow_fixture["keys"], role)
        assert key.get("d") not in output
