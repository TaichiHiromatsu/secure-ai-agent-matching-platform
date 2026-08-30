"""Deterministic Japanese views shared by API, ADK, and CLI."""

from __future__ import annotations

import os
from typing import Any

from secure_mediation_agent.demo_catalog import demo_scenario

from .models import PlanSnapshot, PublicWorkflowView, WorkflowState


AP2_LABEL = "AP2 v0.2 Human Present demo"
X402_LABEL = "x402 v0.1 wire-shape test fixture (NOT CONFORMANT)"
RAIL_LABEL = "simulated; no real asset or on-chain transaction"
EPHEMERAL_LABEL = "EPHEMERAL DEMO: state and keys may reset on restart"


def _deployment_notice() -> str:
    return (
        EPHEMERAL_LABEL + "\n"
        if os.environ.get("EPHEMERAL_CLOUD_RUN_DEMO") == "true"
        else ""
    )


def _plan_text(plan: PlanSnapshot, digest: str) -> str:
    scenario = demo_scenario()
    return (
        _deployment_notice()
        + "計画の承認\n"
        f"workflow/plan: {plan.plan_id} / {digest}\n"
        "agent/Merchant: paid-booking-agent / Demo Merchant (demo-merchant)\n"
        f"service/product/quantity: {scenario['service']} / {scenario['productId']} / 1\n"
        f"hotel/dates/guests: {scenario['hotel']} / {scenario['dates']['checkIn']}〜{scenario['dates']['checkOut']} / {scenario['guests']}\n"
        f"最大総額: {plan.maximum_customer_total} {plan.currency} (decimals={plan.decimals})\n"
        "fee policy: zero-fee-v1; expiry: " + plan.expires_at + "\n"
        "拒否する場合は「拒否」と入力してください。\n"
        "この「承認」ではまだ決済されません。見積・Checkout取得と実行開始だけを許可します。"
    )


def _payment_text(workflow: dict[str, Any]) -> str:
    scenario = demo_scenario()
    return (
        _deployment_notice()
        + "決済の承認\n"
        f"order/task: {workflow['order_id']} / {workflow['merchant_task_id']}\n"
        "Merchant/payee: Demo Merchant / demo-merchant\n"
        f"line item/quantity: {scenario['service']} / 1\n"
        f"hotel/dates/guests: {scenario['hotel']} / {scenario['dates']['checkIn']}〜{scenario['dates']['checkOut']} / {scenario['guests']}\n"
        "12.50 USDは宿泊代を含まない予約手配サービス料です。実予約ではありません。\n"
        "merchandiseAmount=1250, customerSurcharge=0, collectionRailCost=0, "
        "customerTotal=1250, providerCommission=0, merchantPayableAmount=1250, payoutRailCost=0\n"
        "currency/decimals/instrument: USD / 2 / demo-instrument-1\n"
        "scheme/network/asset/payTo: exact-simulated / demo:local / USD / merchant:demo-merchant\n"
        f"approval expiry (UTC): {workflow['payment_expires_at']}\n"
        f"区分: {RAIL_LABEL}\n{X402_LABEL}\n"
        "課金警告: この「承認」で signed Payment Mandate が生成され、customer charge の verify/settle が開始されます。"
        "期限切れ後は承認できず、新しい quote/Checkout と再承認が必要です。"
    )


def build_view(
    workflow: dict[str, Any],
    *,
    plan: PlanSnapshot,
    artifacts: list[dict[str, Any]] | None = None,
    receipts: list[dict[str, Any]] | None = None,
) -> PublicWorkflowView:
    state = WorkflowState(workflow["state"])
    pending = None
    if state == WorkflowState.PLAN_APPROVAL_REQUIRED:
        pending = "plan"
        text = _plan_text(plan, workflow["plan_digest"])
    elif state == WorkflowState.PAYMENT_APPROVAL_REQUIRED:
        pending = "payment"
        text = _payment_text(workflow)
    elif state == WorkflowState.PAYMENT_AUTHORIZING:
        text = "決済承認済み・認可証跡生成中です。再度の承認は不要です。"
    elif state == WorkflowState.COMPLETED:
        ids = ", ".join(item["artifact_id"] for item in artifacts or [])
        refs = ", ".join(item["receipt_id"] for item in receipts or [])
        text = (
            _deployment_notice()
            + "完了\n"
            f"plan/order/task: {workflow['active_plan_id']} / {workflow['order_id']} / {workflow['merchant_task_id']}\n"
            "業務結果: デモ予約確認（シミュレーション）。実予約ではありません。\n"
            f"AP2 evidence: {ids}\nsettlement receipts: {refs}\n"
            f"{AP2_LABEL}\n{X402_LABEL}\n{RAIL_LABEL}"
        )
    elif state == WorkflowState.RECONCILIATION_REQUIRED:
        text = "結果不明です。追加の決済を作らず、同じ simulation reference を照会します。"
    elif state == WorkflowState.REFUND_REQUIRED:
        text = "決済後の履行失敗です。元の証跡を変更せず返金が必要です。"
    elif state == WorkflowState.REFUNDED:
        text = "返金済みです。元の決済証跡は変更されていません。"
    elif state == WorkflowState.PAYMENT_FAILED:
        text = "決済に失敗しました。completed または実 transaction ではありません。"
    elif state in {WorkflowState.CANCELLED, WorkflowState.EXPIRED}:
        text = "取消済みです。決済・settlement・fulfillment は開始されていません。"
    else:
        text = f"workflow state: {state}"
    evidence = None
    if artifacts or receipts:
        evidence = {"artifacts": artifacts or [], "simulationReceipts": receipts or []}
    return PublicWorkflowView(
        workflowId=workflow["workflow_id"],
        state=state,
        version=workflow["version"],
        pendingApproval=pending,
        planId=workflow.get("active_plan_id"),
        planDigest=workflow.get("plan_digest"),
        orderId=workflow.get("order_id"),
        taskId=workflow.get("merchant_task_id"),
        renderedText=text,
        profile="x402-wire-simulation/1",
        ap2Label=AP2_LABEL,
        x402Label=X402_LABEL,
        railLabel=RAIL_LABEL,
        evidence=evidence,
    )
