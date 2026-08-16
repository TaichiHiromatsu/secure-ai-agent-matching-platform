from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_runner_module():
    script = Path(__file__).parents[2] / "scripts" / "run_regression_manifest.py"
    spec = importlib.util.spec_from_file_location("run_regression_manifest", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_suite_paths_use_source_layout_when_present(tmp_path: Path) -> None:
    runner = _load_runner_module()
    configured = "trusted_agent_store/evaluation-runner/tests"
    (tmp_path / configured).mkdir(parents=True)

    assert runner._resolve_suite_paths([configured], tmp_path) == [configured]


def test_suite_paths_fall_back_to_release_image_layout(tmp_path: Path) -> None:
    runner = _load_runner_module()
    configured = "trusted_agent_store/evaluation-runner/tests"
    packaged = "evaluation-runner/tests"
    (tmp_path / packaged).mkdir(parents=True)

    assert runner._resolve_suite_paths(
        [configured], tmp_path / "source", packaged_root=tmp_path
    ) == [str(tmp_path / packaged)]


def test_missing_suite_path_fails_closed(tmp_path: Path) -> None:
    runner = _load_runner_module()
    configured = "trusted_agent_store/evaluation-runner/tests"

    with pytest.raises(FileNotFoundError):
        runner._resolve_suite_paths(
            [configured], tmp_path / "source", packaged_root=tmp_path / "image"
        )


@pytest.mark.parametrize(
    "configured",
    ["trusted_agent_store/unknown/tests", "../tests", "/app/tests"],
)
def test_unexpected_or_traversing_suite_path_fails_closed(
    tmp_path: Path, configured: str
) -> None:
    runner = _load_runner_module()

    with pytest.raises(ValueError):
        runner._resolve_suite_paths([configured], tmp_path, packaged_root=tmp_path)


def test_manifest_contains_each_release_suite_exactly_once() -> None:
    runner = _load_runner_module()
    manifest_path = Path(__file__).parents[1] / "regression" / "suite_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    runner._validate_manifest_suites(manifest["suites"])
    names = [suite["name"] for suite in manifest["suites"]]
    assert len(names) == len(set(names)) == 3


def test_duplicate_manifest_suite_fails_closed() -> None:
    runner = _load_runner_module()
    suites = [
        {"name": name, "paths": [path]}
        for name, path in runner._EXPECTED_SUITE_PATHS.items()
    ]

    with pytest.raises(ValueError):
        runner._validate_manifest_suites([*suites, suites[0]])
