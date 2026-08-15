"""Thin ADK UI adapter for the authoritative secure mediation workflow."""

from secure_mediation_agent.agent import PaymentWorkflowAdapter


root_agent = PaymentWorkflowAdapter(
    name="payment_user_agent",
    description=(
        "Durable AP2 v0.2 Human Present payment demo. All decisions execute in "
        "the internal secure mediation workflow; the x402 rail is a local "
        "wire-shape simulation (NOT CONFORMANT; no asset or on-chain transfer)."
    ),
)
