"""Internal secure mediator exposed through a deterministic ADK adapter."""

from .mediation.adk_adapter import SecureMediationAdapter


root_agent = SecureMediationAdapter(
    name="secure_mediator",
    description=(
        "Deterministic secure A2A mediation with explicit plan and payment approval. "
        "The local x402 wire-shape rail is a simulation (NOT CONFORMANT)."
    ),
)

# Transitional import compatibility; this no longer routes to the legacy workflow API.
PaymentWorkflowAdapter = SecureMediationAdapter

__all__ = ["PaymentWorkflowAdapter", "SecureMediationAdapter", "root_agent"]
