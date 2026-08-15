"""Fail-closed official x402 profile marker; no adapter is implemented."""

from __future__ import annotations

from .base import ProfileReadiness


CANONICAL_X402_V01_URI = "https://github.com/google-a2a/a2a-x402/v0.1"


class OfficialX402V01Profile:
    profile_id = "a2a-x402/v0.1"
    extension_uri = CANONICAL_X402_V01_URI
    rail_mode = "on-chain"
    conformance_label = "DISABLED / NOT READY / NOT RUN"

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "Official x402 is NOT READY: wallet, facilitator, network, asset, "
            "TLS, amount policy, and ACC-030 evidence are not implemented."
        )

    @staticmethod
    def readiness() -> ProfileReadiness:
        return ProfileReadiness(
            ready=False,
            profile_id="a2a-x402/v0.1",
            rail_mode="on-chain",
            checks={
                "officialProfileEnablement": "NOT READY",
                "walletFacilitatorVerify": "NOT RUN",
                "onChainSettle": "NOT RUN",
            },
        )
