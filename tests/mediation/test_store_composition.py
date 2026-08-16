from __future__ import annotations

import os

import pytest

import secure_mediation_agent.mediation.persistence as persistence_module
from secure_mediation_agent.mediation.composition import _configured_store
from secure_mediation_agent.mediation.persistence import SqliteMediationStore
from secure_mediation_agent.mediation.store import InMemoryMediationStore


def _private_key_file(tmp_path):
    path = tmp_path / "mediation-store.key"
    path.write_bytes(b"m" * 32)
    path.chmod(0o600)
    return path


def test_sqlite_is_the_default_and_probes_schema_v4(
    workflow_fixture, tmp_path, monkeypatch
) -> None:
    key_file = _private_key_file(tmp_path)
    monkeypatch.delenv("MEDIATION_STORE_MODE", raising=False)
    monkeypatch.setenv("MEDIATION_STORE_KEY_FILE", str(key_file))

    store = _configured_store(workflow_fixture["repository"])

    assert isinstance(store, SqliteMediationStore)
    assert store.kind == "sqlite"
    assert store.durability_profile == "local-durable"
    probe = store.readiness_probe()
    assert probe.schema_version == 4
    assert probe.writable is probe.decryptable is True


def test_memory_store_is_restricted_to_explicit_demo_modes(
    workflow_fixture, monkeypatch
) -> None:
    monkeypatch.setenv("MEDIATION_STORE_MODE", "memory")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DEV_MODE", "false")
    monkeypatch.delenv("EPHEMERAL_CLOUD_RUN_DEMO", raising=False)
    with pytest.raises(RuntimeError, match="restricted"):
        _configured_store(workflow_fixture["repository"])

    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("DEV_MODE", "true")
    assert isinstance(
        _configured_store(workflow_fixture["repository"]),
        InMemoryMediationStore,
    )


def test_cloud_demo_requires_explicit_memory_mode(
    workflow_fixture, monkeypatch
) -> None:
    monkeypatch.setenv("EPHEMERAL_CLOUD_RUN_DEMO", "true")
    monkeypatch.delenv("MEDIATION_STORE_MODE", raising=False)
    with pytest.raises(RuntimeError, match="requires MEDIATION_STORE_MODE=memory"):
        _configured_store(workflow_fixture["repository"])

    monkeypatch.setenv("MEDIATION_STORE_MODE", "sqlite")
    with pytest.raises(RuntimeError, match="requires MEDIATION_STORE_MODE=memory"):
        _configured_store(workflow_fixture["repository"])

    monkeypatch.setenv("MEDIATION_STORE_MODE", "memory")
    assert isinstance(
        _configured_store(workflow_fixture["repository"]),
        InMemoryMediationStore,
    )

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DEV_MODE", "false")
    monkeypatch.setenv("EPHEMERAL_CLOUD_RUN_DEMO", "true")
    assert isinstance(
        _configured_store(workflow_fixture["repository"]),
        InMemoryMediationStore,
    )


def test_unknown_mode_and_sqlite_failure_never_fall_back(
    workflow_fixture, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MEDIATION_STORE_MODE", "unknown")
    with pytest.raises(RuntimeError, match="unsupported MEDIATION_STORE_MODE"):
        _configured_store(workflow_fixture["repository"])

    monkeypatch.setenv("MEDIATION_STORE_MODE", "sqlite")
    monkeypatch.setenv(
        "MEDIATION_STORE_KEY_FILE", str(_private_key_file(tmp_path))
    )

    class BrokenSqliteStore:
        def __init__(self, **kwargs) -> None:
            raise RuntimeError("sqlite unavailable")

    monkeypatch.setattr(
        persistence_module, "SqliteMediationStore", BrokenSqliteStore
    )
    with pytest.raises(RuntimeError, match="sqlite unavailable"):
        _configured_store(workflow_fixture["repository"])


def test_store_mode_does_not_accept_ambiguous_whitespace_only(
    workflow_fixture, monkeypatch
) -> None:
    monkeypatch.setenv("MEDIATION_STORE_MODE", "   ")
    with pytest.raises(RuntimeError, match="unsupported MEDIATION_STORE_MODE"):
        _configured_store(workflow_fixture["repository"])

    assert os.environ["MEDIATION_STORE_MODE"] == "   "
