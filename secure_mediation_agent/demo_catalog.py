"""Immutable catalog fixture for the VC paid-booking demonstration."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from secure_mediation_agent.workflow.canonical import canonical_digest


SCENARIO_VERSION = "demo-booking-scenario/1"
SCENARIO_ID = "tokyo-business-hotel-arrangement-20260912-v1"
PRODUCT_ID = "demo-paid-booking"
REQUIREMENT_SCHEMA_VERSION = "demo-payment-requirement/2"

_SCENARIO: dict[str, Any] = {
    "scenarioVersion": SCENARIO_VERSION,
    "scenarioId": SCENARIO_ID,
    "productId": PRODUCT_ID,
    "service": "デモホテル予約手配サービス",
    "hotel": "デモ東京ベイホテル",
    "destination": "東京",
    "dates": {
        "checkIn": "2026-09-12",
        "checkOut": "2026-09-14",
    },
    "guests": 2,
    "arrangementFee": {
        "amountMinor": 1250,
        "currency": "USD",
        "decimals": 2,
        "lodgingExcluded": True,
        "payee": "demo-merchant",
    },
    "terms": {
        "simulationOnly": True,
        "realBooking": False,
        "realInventoryHold": False,
        "realCharge": False,
        "realTransfer": False,
        "legalGuarantee": False,
    },
}


def demo_scenario() -> dict[str, Any]:
    """Return a defensive copy; callers cannot mutate the catalog fixture."""

    return copy.deepcopy(_SCENARIO)


def scenario_digest() -> str:
    return canonical_digest(_SCENARIO)


def project_payment_requirement(required: dict[str, Any]) -> dict[str, Any]:
    """Attach the exact catalog scenario and its canonical digest to x402 terms."""

    return {
        **required,
        "schemaVersion": REQUIREMENT_SCHEMA_VERSION,
        "demoScenario": demo_scenario(),
        "scenarioDigest": scenario_digest(),
    }


def validate_payment_requirement(required: dict[str, Any]) -> dict[str, Any]:
    """Fail closed for missing, unknown, or altered v2 scenario terms."""

    if required.get("schemaVersion") != REQUIREMENT_SCHEMA_VERSION:
        raise ValueError("unsupported demo payment requirement schema")
    scenario = required.get("demoScenario")
    digest = required.get("scenarioDigest")
    if not isinstance(scenario, dict) or not isinstance(digest, str):
        raise ValueError("demo payment requirement scenario binding is missing")
    if scenario != _SCENARIO or digest != scenario_digest():
        raise ValueError("demo payment requirement scenario does not match the catalog")
    if canonical_digest(scenario) != digest:
        raise ValueError("demo payment requirement scenario digest mismatch")
    return copy.deepcopy(scenario)


def confirmation_reference(remote_task_id: str) -> str:
    if not isinstance(remote_task_id, str) or not remote_task_id:
        raise ValueError("remote task id is required for confirmation")
    suffix = hashlib.sha256(remote_task_id.encode("utf-8")).hexdigest()[:12]
    return f"DEMO-TYO-0912-2P-{suffix}"


def project_confirmation(remote_task_id: str) -> dict[str, Any]:
    scenario = demo_scenario()
    return {
        "schemaVersion": "demo-booking-confirmation/1",
        "confirmationReference": confirmation_reference(remote_task_id),
        "status": "SIMULATED",
        "notice": "NOT A REAL BOOKING",
        "scenarioId": scenario["scenarioId"],
        "productId": scenario["productId"],
        "service": scenario["service"],
        "hotel": scenario["hotel"],
        "destination": scenario["destination"],
        "dates": scenario["dates"],
        "guests": scenario["guests"],
        "arrangementFee": scenario["arrangementFee"],
        "realBooking": False,
    }


def validate_confirmation(value: dict[str, Any], *, remote_task_id: str) -> None:
    if value != project_confirmation(remote_task_id):
        raise ValueError("demo booking confirmation does not match the catalog")


def project_confirmation_artifact(remote_task_id: str) -> dict[str, Any]:
    """Return the only public/persisted paid completion artifact shape."""

    confirmation = project_confirmation(remote_task_id)
    return {
        "artifactId": f"artifact:{remote_task_id}",
        "name": "デモ予約確認（シミュレーション）",
        "parts": [{"kind": "data", "data": confirmation}],
        "metadata": {
            "schemaVersion": confirmation["schemaVersion"],
            "scenarioDigest": scenario_digest(),
            "confirmationReference": confirmation["confirmationReference"],
            "simulated": True,
            "externalCommit": False,
        },
    }
