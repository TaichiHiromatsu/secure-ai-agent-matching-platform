from __future__ import annotations

from pathlib import Path

import pytest

from secure_mediation_agent.ap2.keys import DemoKeySet
from secure_mediation_agent.identity import issue_identity_assertion
from secure_mediation_agent.workflow.api import WorkflowRuntime, create_app
from secure_mediation_agent.workflow.controller import WorkflowController
from secure_mediation_agent.workflow.migrations import DatabasePaths
from secure_mediation_agent.workflow.repository import WorkflowRepository


@pytest.fixture
def workflow_fixture(tmp_path: Path):
    paths = DatabasePaths.resolve(
        tmp_path / "data" / "marketplace.db",
        tmp_path / "data" / "paid-agent.db",
        tmp_path / "evidence" / "evidence.db",
    )
    marker = tmp_path / "data" / ".durable-volume"
    evidence_marker = tmp_path / "evidence" / ".durable-volume"
    marker.write_text("explicit-test-volume\n", encoding="utf-8")
    evidence_marker.write_text("explicit-test-volume\n", encoding="utf-8")
    keys = DemoKeySet.generate_for_test()
    repository = WorkflowRepository(paths)
    repository.heartbeat_worker("pytest-outbox-worker")
    runtime = WorkflowRuntime(
        controller=WorkflowController(repository, keys),
        paths=paths,
        identity_verifier_key=keys.service_auth,
        durable_marker=marker,
        evidence_durable_marker=evidence_marker,
        keys=keys,
        merchant_probe=lambda: True,
        allow_ephemeral_test_dependencies=True,
    )
    return {
        "paths": paths,
        "keys": keys,
        "repository": repository,
        "runtime": runtime,
        "app": create_app(runtime),
        "assertion": issue_identity_assertion(keys.service_auth, subject="test-user"),
    }
