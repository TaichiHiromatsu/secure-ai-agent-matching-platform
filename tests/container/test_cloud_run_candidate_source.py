from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts import cloud_run_candidate


pytestmark = pytest.mark.container
BASE_COMMIT = "1" * 40
CURRENT_COMMIT = "2" * 40
SOURCE_DIGEST = "sha256:" + "a" * 64


def _source_info(**overrides: Any) -> dict[str, Any]:
    value = {
        "baseCommit": CURRENT_COMMIT,
        "worktreeDigest": SOURCE_DIGEST,
        "fileCount": 233,
        "algorithm": "path-mode-size-bytes-v1",
    }
    value.update(overrides)
    return value


def _stored_source(**overrides: Any) -> dict[str, Any]:
    value = {
        "baseCommit": BASE_COMMIT,
        "worktreeDigest": SOURCE_DIGEST,
        "fileCount": 233,
        "algorithm": "path-mode-size-bytes-v1",
    }
    value.update(overrides)
    return value


def _accept_ancestor(*command: str) -> str:
    if command == ("git", "cat-file", "-e", f"{BASE_COMMIT}^{{commit}}"):
        return ""
    if command == ("git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"):
        return ""
    raise AssertionError(f"unexpected command: {command}")


def test_source_binding_allows_later_commit_outside_release_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []

    def run(*command: str) -> str:
        commands.append(command)
        return _accept_ancestor(*command)

    monkeypatch.setattr(cloud_run_candidate, "_run", run)
    monkeypatch.setattr(cloud_run_candidate, "_source_info", _source_info)

    cloud_run_candidate._verify_source_binding(_stored_source())

    assert commands[-1] == (
        "git",
        "merge-base",
        "--is-ancestor",
        BASE_COMMIT,
        "HEAD",
    )
    assert BASE_COMMIT != CURRENT_COMMIT


@pytest.mark.parametrize("change", ["bytes", "mode"])
def test_source_binding_rejects_release_source_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    source = tmp_path / "scripts" / "release.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(cloud_run_candidate, "ROOT", tmp_path)
    monkeypatch.setattr(cloud_run_candidate, "REQUIRED_JSON", ())
    monkeypatch.setattr(
        cloud_run_candidate, "_visible_files", lambda: ["scripts/release.py"]
    )
    monkeypatch.setattr(
        cloud_run_candidate,
        "_run",
        lambda *command: BASE_COMMIT
        if command == ("git", "rev-parse", "HEAD")
        else _accept_ancestor(*command),
    )
    stored = cloud_run_candidate._source_info()
    if change == "bytes":
        source.write_text("VALUE = 2\n", encoding="utf-8")
    else:
        source.chmod(0o755)
    current = cloud_run_candidate._source_info()
    assert current["worktreeDigest"] != stored["worktreeDigest"]
    monkeypatch.setattr(cloud_run_candidate, "_source_info", lambda: current)

    with pytest.raises(
        cloud_run_candidate.CandidateError,
        match="source worktreeDigest differs from candidate",
    ):
        cloud_run_candidate._verify_source_binding(stored)


def test_source_binding_rejects_non_ancestor_base_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(*command: str) -> str:
        if command[1] == "cat-file":
            return ""
        raise cloud_run_candidate.CandidateError("not an ancestor")

    monkeypatch.setattr(cloud_run_candidate, "_run", run)

    with pytest.raises(
        cloud_run_candidate.CandidateError,
        match="source base commit is not an ancestor of current HEAD",
    ):
        cloud_run_candidate._verify_source_binding(_stored_source())


@pytest.mark.parametrize("base_commit", ["not-a-commit", "f" * 40])
def test_source_binding_rejects_invalid_or_missing_base_commit(
    monkeypatch: pytest.MonkeyPatch,
    base_commit: str,
) -> None:
    if len(base_commit) == 40:
        monkeypatch.setattr(
            cloud_run_candidate,
            "_run",
            lambda *command: (_ for _ in ()).throw(
                cloud_run_candidate.CandidateError("missing object")
            ),
        )

    with pytest.raises(cloud_run_candidate.CandidateError):
        cloud_run_candidate._verify_source_binding(
            _stored_source(baseCommit=base_commit)
        )


def test_source_binding_accepts_legacy_commit_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cloud_run_candidate, "_run", _accept_ancestor)
    monkeypatch.setattr(cloud_run_candidate, "_source_info", _source_info)
    stored = _stored_source()
    stored["commit"] = stored.pop("baseCommit")

    cloud_run_candidate._verify_source_binding(stored)
