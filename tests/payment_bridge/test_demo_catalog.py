from __future__ import annotations

import copy

import pytest

from secure_mediation_agent.demo_catalog import (
    REQUIREMENT_SCHEMA_VERSION,
    confirmation_reference,
    demo_scenario,
    project_confirmation,
    project_payment_requirement,
    scenario_digest,
    validate_confirmation,
    validate_payment_requirement,
)
from secure_mediation_agent.payment_profiles.registry import ProfileRegistry


def test_catalog_projection_is_immutable_and_digest_bound() -> None:
    scenario = demo_scenario()
    assert scenario["scenarioVersion"] == "demo-booking-scenario/1"
    assert scenario["scenarioId"] == "tokyo-business-hotel-arrangement-20260912-v1"
    assert scenario["hotel"] == "デモ東京ベイホテル"
    assert scenario["arrangementFee"] == {
        "amountMinor": 1250,
        "currency": "USD",
        "decimals": 2,
        "lodgingExcluded": True,
        "payee": "demo-merchant",
    }
    assert scenario["terms"] == {
        "simulationOnly": True,
        "realBooking": False,
        "realInventoryHold": False,
        "realCharge": False,
        "realTransfer": False,
        "legalGuarantee": False,
    }
    scenario["hotel"] = "altered"
    assert demo_scenario()["hotel"] == "デモ東京ベイホテル"

    required = project_payment_requirement({"x402Version": 1, "accepts": []})
    assert required["schemaVersion"] == REQUIREMENT_SCHEMA_VERSION
    assert required["scenarioDigest"] == scenario_digest()
    assert validate_payment_requirement(required) == demo_scenario()


@pytest.mark.parametrize("mutation", ["missing", "unknown", "scenario", "digest"])
def test_requirement_v2_fails_closed(mutation: str) -> None:
    required = project_payment_requirement({"x402Version": 1, "accepts": []})
    changed = copy.deepcopy(required)
    if mutation == "missing":
        changed.pop("schemaVersion")
    elif mutation == "unknown":
        changed["schemaVersion"] = "demo-payment-requirement/999"
    elif mutation == "scenario":
        changed["demoScenario"]["guests"] = 3
    else:
        changed["scenarioDigest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        ProfileRegistry.validate_requirement("x402-wire-simulation/1", changed)


def test_confirmation_is_deterministic_and_strict() -> None:
    task_id = "remote-task-1"
    confirmation = project_confirmation(task_id)
    assert confirmation["confirmationReference"] == confirmation_reference(task_id)
    assert confirmation["confirmationReference"].startswith("DEMO-TYO-0912-2P-")
    assert len(confirmation["confirmationReference"].rsplit("-", 1)[1]) == 12
    assert confirmation["notice"] == "NOT A REAL BOOKING"
    validate_confirmation(confirmation, remote_task_id=task_id)
    changed = copy.deepcopy(confirmation)
    changed["realBooking"] = True
    with pytest.raises(ValueError):
        validate_confirmation(changed, remote_task_id=task_id)
