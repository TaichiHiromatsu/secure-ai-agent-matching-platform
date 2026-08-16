"""The only public payment UI agent; all work stays inside secure mediation."""

from secure_mediation_agent.mediation.adk_adapter import SecureMediationAdapter


root_agent = SecureMediationAdapter(
    name="payment_user_agent",
    description=(
        "Public deterministic session router for secure A2A mediation. "
        "The local x402 wire-shape rail is a simulation (NOT CONFORMANT)."
    ),
)

__all__ = ["root_agent"]
