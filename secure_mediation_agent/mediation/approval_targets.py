"""Canonical approval targets shared by the controller and bridge boundary."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from secure_mediation_agent.demo_catalog import (
    REQUIREMENT_SCHEMA_VERSION,
    validate_payment_requirement,
)

from .canonical import canonical_digest, safe_ref
from .errors import SecurityBlocked
from .models import (
    BridgePaymentDisplay,
    MediationPlan,
    PaymentApprovalTarget,
    PlanApprovalAgentTarget,
    PlanApprovalStepTarget,
    PlanApprovalTarget,
)


APPROVAL_TOKEN = "承認"
PAYMENT_PRODUCT = "デモホテル予約手配サービス（宿泊代を含まないシミュレーション）"
LEGACY_PAYMENT_PRODUCT = "Demo paid booking"
PAYMENT_METHOD = "signed-simulated-payment-guarantee"


def build_plan_approval_target(plan: MediationPlan) -> PlanApprovalTarget:
    steps: list[PlanApprovalStepTarget] = []
    for step in plan.steps:
        selected = step.selected_agent
        if selected.payment_extension_uris:
            conditions = (
                "Execute only the displayed live Agent Card snapshot and pinned RPC endpoint.",
                "Payment may be requested only within the displayed limit and requires a distinct second approval.",
            )
        else:
            conditions = (
                "Execute only the displayed live Agent Card snapshot and pinned RPC endpoint.",
                "This step declares no required payment extension and authorizes no payment.",
            )
        steps.append(
            PlanApprovalStepTarget(
                stepId=step.step_id,
                ordinal=step.ordinal,
                agent=PlanApprovalAgentTarget(
                    canonicalAgentId=selected.canonical_agent_id,
                    registryName=selected.registry_name,
                    a2aAgentName=selected.a2a_agent_name,
                    skillId=selected.a2a_skill_id,
                    rpcEndpoint=selected.rpc_endpoint,
                    cardDigest=selected.card_digest,
                    snapshotDigest=selected.snapshot_digest,
                ),
                goal=step.goal,
                conditions=conditions,
                currency=step.currency,
                paymentLimitMinor=step.payment_limit_minor,
            )
        )
    return PlanApprovalTarget(
        planId=plan.plan_id,
        planVersion=plan.plan_version,
        planDigest=plan.plan_digest,
        steps=tuple(steps),
        expiresAt=plan.expires_at,
        approvalToken=APPROVAL_TOKEN,
    )


def build_bridge_payment_display(
    *,
    task_id: str,
    context_id: str,
    order_id: str,
    quote_id: str,
    merchant: str,
    amount_minor: int,
    currency: str,
    profile_id: str,
) -> BridgePaymentDisplay:
    return BridgePaymentDisplay(
        taskId=task_id,
        contextId=context_id,
        orderId=order_id,
        quoteId=quote_id,
        merchant=merchant,
        amountMinor=amount_minor,
        currency=currency,
        profileId=profile_id,
        simulated=True,
        state="WAITING_FOR_PAYMENT_APPROVAL",
    )


def build_payment_approval_target(
    *,
    plan_id: str,
    plan_version: int,
    plan_digest: str,
    step_id: str,
    task_id: str,
    context_id: str,
    order_id: str,
    quote_id: str,
    merchant: str,
    amount_minor: int,
    currency: str,
    profile_id: str,
    expires_at: datetime,
    payment_required: dict[str, Any],
    requirement_digest: str,
    checkout_digest: str,
) -> PaymentApprovalTarget:
    schema_version = payment_required.get("schemaVersion")
    if schema_version is None:
        # Compatibility is reachable only for a requirement already persisted by
        # the pre-v2 runtime; fresh remote responses are rejected by controller.
        product = LEGACY_PAYMENT_PRODUCT
    elif schema_version == REQUIREMENT_SCHEMA_VERSION:
        try:
            validate_payment_requirement(payment_required)
        except ValueError as error:
            raise SecurityBlocked(
                "PAYMENT_APPROVAL_SCENARIO_INVALID",
                "支払い承認対象のデモシナリオが一致しません。",
            ) from error
        product = PAYMENT_PRODUCT
    else:
        raise SecurityBlocked(
            "PAYMENT_APPROVAL_SCENARIO_VERSION_UNSUPPORTED",
            "支払い承認対象のデモシナリオversionを利用できません。",
        )
    accepts = payment_required.get("accepts")
    if not isinstance(accepts, list) or len(accepts) != 1:
        raise SecurityBlocked(
            "PAYMENT_APPROVAL_TARGET_INVALID",
            "支払い承認対象を安全に表示できません。",
        )
    accept = accepts[0]
    if not isinstance(accept, dict):
        raise SecurityBlocked(
            "PAYMENT_APPROVAL_TARGET_INVALID",
            "支払い承認対象を安全に表示できません。",
        )
    display = build_bridge_payment_display(
        task_id=task_id,
        context_id=context_id,
        order_id=order_id,
        quote_id=quote_id,
        merchant=merchant,
        amount_minor=amount_minor,
        currency=currency,
        profile_id=profile_id,
    )
    return PaymentApprovalTarget(
        planId=plan_id,
        planVersion=plan_version,
        planDigest=plan_digest,
        bridgeDisplay=display,
        bridgeDisplayDigest=canonical_digest(display),
        product=product,
        expiresAt=expires_at,
        paymentMethod=PAYMENT_METHOD,
        scheme=str(accept.get("scheme") or ""),
        network=str(accept.get("network") or ""),
        asset=str(accept.get("asset") or ""),
        stepRef=safe_ref(step_id),
        taskRef=safe_ref(task_id),
        requirementDigest=requirement_digest,
        checkoutDigest=checkout_digest,
        approvalToken=APPROVAL_TOKEN,
    )
