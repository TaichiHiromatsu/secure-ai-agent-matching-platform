"""MPP verification boundary for the value-free local rail."""

from __future__ import annotations

from typing import Any

from secure_mediation_agent.payment_profiles.simulation_v1 import SimulationV1Profile
from secure_mediation_agent.workflow.canonical import sha256_digest

from .credential_provider import CredentialProvider


class MerchantPaymentProcessor:
    issuer = "demo-mpp"

    def __init__(
        self,
        credential_provider: CredentialProvider,
        profile: SimulationV1Profile,
    ) -> None:
        self._cp = credential_provider
        self._profile = profile

    def verify_authorization(
        self,
        *,
        task_id: str,
        credential: str,
        proof: str,
        requirement: dict[str, Any],
    ) -> dict[str, Any]:
        proof_payload = self._profile.verify_proof(proof, self._profile.public_key())
        claims = self._cp.verify(
            credential,
            task_id=task_id,
            payload_digest=sha256_digest(proof),
        )
        if claims.get("requirementsDigest") != self._profile.requirements_digest(requirement):
            raise ValueError("credential requirements binding mismatch")
        if proof_payload.get("taskId") != task_id:
            raise ValueError("simulation proof task binding mismatch")
        return claims
