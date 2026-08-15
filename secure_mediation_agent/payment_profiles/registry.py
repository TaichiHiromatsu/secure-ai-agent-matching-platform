"""Exclusive payment profile loader."""

from __future__ import annotations

from jwcrypto.jwk import JWK

from .simulation_v1 import SimulationV1Profile


class ProfileRegistry:
    @staticmethod
    def load(profile_id: str, *, simulation_key: JWK) -> SimulationV1Profile:
        if profile_id != SimulationV1Profile.profile_id:
            raise RuntimeError(
                f"Payment profile {profile_id!r} is disabled; this release is "
                "simulation-only and never falls back between profiles."
            )
        return SimulationV1Profile(simulation_key)
