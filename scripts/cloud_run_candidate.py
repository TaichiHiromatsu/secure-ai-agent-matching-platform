#!/usr/bin/env python3
"""Freeze and verify the immutable ephemeral Cloud Run release candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts/cloud-run-candidate.json"
PLATFORM = "linux/amd64"
PROJECT_ID = "gen-lang-client-0585901015"
REGION = "asia-northeast1"
SERVICE_NAME = "payment-user-agent-demo"
REGISTRY_REPOSITORY = (
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/secure-mediation-agent/{SERVICE_NAME}"
)
DEPLOY_ENVIRONMENT = {
    "APP_ENV": "ephemeral-demo",
    "DEV_MODE": "false",
    "EPHEMERAL_CLOUD_RUN_DEMO": "true",
}
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
IMMUTABLE_IMAGE_PATTERN = re.compile(
    rf"^{re.escape(REGISTRY_REPOSITORY)}@(sha256:[0-9a-f]{{64}})$"
)
SOURCE_FILES = {"Dockerfile", "pyproject.toml", "uv.lock"}
SOURCE_PREFIXES = (
    "deploy/",
    "external-agents/",
    "payment_user_agent/",
    "scripts/",
    "secure_mediation_agent/",
    "tests/",
    "trusted_agent_store/",
    "user-agent/",
)
REQUIRED_JSON = (
    "deploy/auth/firebase-config.json",
    "secure_mediation_agent/spec_manifest.json",
    "tests/regression/suite_manifest.json",
    "tests/release/release_manifest.json",
    "docs/ap2_x402_conformance_report.json",
    "trusted_agent_store/evaluation-runner/prompts/aisi/manifest.sample.json",
    "trusted_agent_store/evaluation-runner/prompts/aisi/questions/privacy.data_retention.json",
    "trusted_agent_store/evaluation-runner/prompts/aisi/questions/safety.general.json",
    "trusted_agent_store/evaluation-runner/schemas/fairness_probe.schema.json",
    "trusted_agent_store/evaluation-runner/schemas/policy_score.schema.json",
    "trusted_agent_store/evaluation-runner/schemas/response_sample.schema.json",
)
ARTIFACT_PATHS = {
    "regression": "artifacts/regression-result.json",
    "browser": "artifacts/browser-evidence.json",
    "conformance": "docs/ap2_x402_conformance_report.json",
    "releaseValidation": "artifacts/ap2-x402-release-validation.json",
}
REGRESSION_BASELINES = {
    "payment-release": 166,
    "evaluation-runner": 16,
    "jury-worker": 13,
}
REQUIRED_MARKERS = {
    "spike",
    "unit",
    "contract_ap2",
    "contract_x402_simulation",
    "integration",
    "security",
    "restart",
    "migration",
    "concurrency",
    "container",
    "browser",
}


class CandidateError(RuntimeError):
    """A release-candidate invariant was not satisfied."""


def _run(*command: str) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise CandidateError(f"command failed ({' '.join(command)}): {detail}")
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise CandidateError(f"required artifact is missing: {path.relative_to(ROOT)}")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateError(f"cannot read {path.relative_to(ROOT)}: {error}") from error
    if not isinstance(value, dict):
        raise CandidateError(f"expected a JSON object: {path.relative_to(ROOT)}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _visible_files() -> list[str]:
    output = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return sorted(item.decode("utf-8") for item in output.split(b"\0") if item)


def _source_files() -> list[str]:
    return [
        path
        for path in _visible_files()
        if path in SOURCE_FILES or path.startswith(SOURCE_PREFIXES)
    ]


def _source_info() -> dict[str, Any]:
    visible = set(_visible_files())
    missing = [path for path in REQUIRED_JSON if path not in visible]
    if missing:
        raise CandidateError(
            "required JSON is ignored or absent from the clean-context file set: "
            + ", ".join(missing)
        )
    files = _source_files()
    if not files:
        raise CandidateError("release source set is empty")
    digest = hashlib.sha256()
    for relative in files:
        path = ROOT / relative
        if not path.is_file():
            raise CandidateError(f"release source is not a regular file: {relative}")
        mode = "100755" if path.stat().st_mode & stat.S_IXUSR else "100644"
        payload = path.read_bytes()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(mode.encode("ascii") + b"\0")
        digest.update(str(len(payload)).encode("ascii") + b"\0")
        digest.update(payload)
    return {
        "commit": _run("git", "rev-parse", "HEAD"),
        "worktreeDigest": "sha256:" + digest.hexdigest(),
        "fileCount": len(files),
        "algorithm": "path-mode-size-bytes-v1",
    }


def _artifact_record(relative: str) -> dict[str, str]:
    path = ROOT / relative
    document = _load(path)
    return {
        "path": relative,
        "sha256": _sha256(path),
        "status": str(document.get("status", "MISSING")),
    }


def _assert_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not DIGEST_PATTERN.fullmatch(value):
        raise CandidateError(f"{label} must be an exact sha256 digest")
    return value


def _verify_evidence(image_id: str) -> dict[str, Any]:
    image_id = _assert_digest(image_id, "local image ID")
    regression_path = ROOT / ARTIFACT_PATHS["regression"]
    browser_path = ROOT / ARTIFACT_PATHS["browser"]
    conformance_path = ROOT / ARTIFACT_PATHS["conformance"]
    validation_path = ROOT / ARTIFACT_PATHS["releaseValidation"]
    release_manifest_path = ROOT / "tests/release/release_manifest.json"
    regression = _load(regression_path)
    browser = _load(browser_path)
    conformance = _load(conformance_path)
    validation = _load(validation_path)
    manifest_digest = _sha256(release_manifest_path)

    for label, document in (("regression", regression), ("browser", browser)):
        if document.get("status") != "PASS":
            raise CandidateError(f"{label} evidence is not PASS")
        if document.get("imageDigest") != image_id:
            raise CandidateError(f"{label} evidence is bound to a different image")
        if document.get("releaseManifestDigest") != manifest_digest:
            raise CandidateError(f"{label} evidence has a stale release manifest digest")

    regression_suites = {
        item.get("name"): item for item in regression.get("suites", [])
    }
    for name, minimum in REGRESSION_BASELINES.items():
        result = regression_suites.get(name, {})
        if result.get("status") != "PASS" or int(result.get("collected", -1)) < minimum:
            raise CandidateError(f"regression suite is missing or below baseline: {name}")

    if browser.get("browser") != "chromium-cdp-real-ui":
        raise CandidateError("browser evidence is not from the real Chromium UI suite")
    if browser.get("listApps") != ["payment_user_agent"]:
        raise CandidateError("browser evidence exposed an unexpected ADK app")
    if browser.get("completedAfterRefresh") is not True:
        raise CandidateError("browser evidence did not complete after refresh")

    if validation.get("status") != "PASS":
        raise CandidateError("release validation is not PASS")
    if validation.get("imageDigest") != image_id:
        raise CandidateError("release validation is bound to a different image")
    if validation.get("releaseManifestDigest") != manifest_digest:
        raise CandidateError("release validation has a stale release manifest digest")
    if validation.get("conformanceReportDigest") != _sha256(conformance_path):
        raise CandidateError("release validation has a stale conformance digest")
    if validation.get("failures"):
        raise CandidateError("release validation contains failures")
    marker_suites = validation.get("suites", {})
    if set(marker_suites) != REQUIRED_MARKERS:
        raise CandidateError("release validation marker set is incomplete or unexpected")
    for name, result in marker_suites.items():
        if (
            result.get("status") != "PASS"
            or int(result.get("collected", 0)) < 1
            or int(result.get("failures", -1)) != 0
            or int(result.get("errors", -1)) != 0
            or int(result.get("skippedOrXfailed", -1)) != 0
        ):
            raise CandidateError(f"release marker suite is not clean: {name}")

    tests = conformance.get("tests", {})
    if tests.get("releaseImageDigest") != image_id:
        raise CandidateError("conformance report is bound to a different image")
    if tests.get("releasePlatform") != PLATFORM:
        raise CandidateError("conformance report is not bound to linux/amd64")
    evidence = conformance.get("evidence", {})
    expected_evidence = {
        "releaseManifestDigest": manifest_digest,
        "regressionArtifactDigest": _sha256(regression_path),
        "browserArtifactDigest": _sha256(browser_path),
    }
    for field, expected in expected_evidence.items():
        if evidence.get(field) != expected:
            raise CandidateError(f"conformance report has stale {field}")

    return {
        "regressionSuites": {
            name: {
                "status": regression_suites[name]["status"],
                "collected": regression_suites[name]["collected"],
            }
            for name in sorted(REGRESSION_BASELINES)
        },
        "markerSuites": {
            name: {
                "status": marker_suites[name]["status"],
                "collected": marker_suites[name]["collected"],
            }
            for name in sorted(REQUIRED_MARKERS)
        },
        "browser": {
            "status": browser["status"],
            "engine": browser["browser"],
            "appSelected": browser.get("appSelected"),
            "completedAfterRefresh": browser["completedAfterRefresh"],
        },
    }


def _candidate(image_id: str, status: str, registry_image: str | None) -> dict[str, Any]:
    embedded = _verify_evidence(image_id)
    registry_digest = "NOT_PUSHED"
    registry_value = "NOT_PUSHED"
    if registry_image is not None:
        match = IMMUTABLE_IMAGE_PATTERN.fullmatch(registry_image)
        if not match:
            raise CandidateError("registry image must be the fixed repository at @sha256")
        registry_digest = match.group(1)
        registry_value = registry_image
    return {
        "schemaVersion": "cloud-run-payment-demo-candidate/1",
        "status": status,
        "source": _source_info(),
        "platform": PLATFORM,
        "localImageId": image_id,
        "registry": {
            "image": registry_value,
            "digest": registry_digest,
        },
        "artifacts": {
            name: _artifact_record(path) for name, path in ARTIFACT_PATHS.items()
        },
        "embedded": embedded,
        "deployment": {
            "project": PROJECT_ID,
            "region": REGION,
            "service": SERVICE_NAME,
            "environment": DEPLOY_ENVIRONMENT,
            "durability": "NOT PROVIDED",
            "officialX402": "NOT RUN",
            "onChainSettlement": "NOT RUN",
        },
    }


def _verify_candidate(path: Path, require_pushed: bool, image_id: str | None) -> dict[str, Any]:
    candidate = _load(path)
    expected_status = "PASS" if require_pushed else "LOCAL_VALIDATED_NOT_PUSHED"
    if candidate.get("schemaVersion") != "cloud-run-payment-demo-candidate/1":
        raise CandidateError("unsupported candidate schema")
    if candidate.get("status") != expected_status:
        raise CandidateError(f"candidate status must be {expected_status}")
    if candidate.get("platform") != PLATFORM:
        raise CandidateError("candidate platform must be linux/amd64")
    stored_image_id = _assert_digest(candidate.get("localImageId"), "candidate image ID")
    if image_id is not None and stored_image_id != image_id:
        raise CandidateError("loaded image ID differs from candidate")
    if candidate.get("source") != _source_info():
        raise CandidateError("source commit/worktree digest differs from candidate")
    expected_artifacts = {
        name: _artifact_record(relative) for name, relative in ARTIFACT_PATHS.items()
    }
    if candidate.get("artifacts") != expected_artifacts:
        raise CandidateError("artifact byte digests differ from candidate")
    if candidate.get("embedded") != _verify_evidence(stored_image_id):
        raise CandidateError("embedded regression/marker/browser evidence differs")
    expected_deployment = _candidate(stored_image_id, expected_status, None)["deployment"]
    if candidate.get("deployment") != expected_deployment:
        raise CandidateError("fixed deployment target/environment differs")
    registry = candidate.get("registry", {})
    if require_pushed:
        image = registry.get("image")
        match = IMMUTABLE_IMAGE_PATTERN.fullmatch(str(image))
        if not match or registry.get("digest") != match.group(1):
            raise CandidateError("candidate lacks a matching immutable registry digest")
    elif registry != {"image": "NOT_PUSHED", "digest": "NOT_PUSHED"}:
        raise CandidateError("local candidate unexpectedly claims a registry push")
    return candidate


def _update_conformance(
    image_id: str,
    platform: str,
    regression_path: Path,
    browser_path: Path,
    registry_image: str | None,
) -> None:
    if platform != PLATFORM:
        raise CandidateError("conformance platform must be linux/amd64")
    _assert_digest(image_id, "conformance image ID")
    conformance_path = ROOT / ARTIFACT_PATHS["conformance"]
    conformance = _load(conformance_path)
    regression = _load(regression_path)
    browser = _load(browser_path)
    suites = {item.get("name"): item for item in regression.get("suites", [])}
    conformance["tests"]["repository"] = {
        "status": suites["payment-release"]["status"],
        "passed": suites["payment-release"]["collected"],
        "failed": suites["payment-release"]["failures"],
    }
    conformance["tests"]["evaluationRunner"] = {
        "status": suites["evaluation-runner"]["status"],
        "collected": suites["evaluation-runner"]["collected"],
        "failed": suites["evaluation-runner"]["failures"],
        "unexpectedSkips": len(suites["evaluation-runner"]["unexpectedSkips"]),
    }
    conformance["tests"]["juryWorker"] = {
        "status": suites["jury-worker"]["status"],
        "collected": suites["jury-worker"]["collected"],
        "failed": suites["jury-worker"]["failures"],
        "unexpectedSkips": len(suites["jury-worker"]["unexpectedSkips"]),
        "configuredGoogleSkips": len(suites["jury-worker"]["skipped"]),
    }
    conformance["tests"]["releaseImageDigest"] = image_id
    conformance["tests"]["releasePlatform"] = platform
    conformance["tests"]["realChromiumAdkWeb"] = browser.get("status")
    conformance["status"] = "PASS"
    conformance["evidence"]["releaseManifestDigest"] = _sha256(
        ROOT / "tests/release/release_manifest.json"
    )
    conformance["evidence"]["regressionArtifactDigest"] = _sha256(regression_path)
    conformance["evidence"]["browserArtifactDigest"] = _sha256(browser_path)
    if registry_image is None:
        conformance["tests"]["registryImage"] = "NOT_PUSHED"
        conformance["tests"]["registryDigest"] = "NOT_PUSHED"
        conformance["claims"]["ephemeralCloudRunDemo"] = "LOCAL_AMD64_VALIDATED_NOT_PUSHED"
    else:
        match = IMMUTABLE_IMAGE_PATTERN.fullmatch(registry_image)
        if not match:
            raise CandidateError("conformance registry image is not immutable/fixed")
        conformance["tests"]["registryImage"] = registry_image
        conformance["tests"]["registryDigest"] = match.group(1)
        conformance["claims"]["ephemeralCloudRunDemo"] = "PUSHED_NOT_DEPLOYED"
    _write(conformance_path, conformance)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("source-info")

    update = subparsers.add_parser("update-conformance")
    update.add_argument("--image-id", required=True)
    update.add_argument("--platform", required=True)
    update.add_argument("--regression", required=True)
    update.add_argument("--browser", required=True)
    update.add_argument("--registry-image")

    write_local = subparsers.add_parser("write-local")
    write_local.add_argument("--image-id", required=True)
    write_local.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))

    write_pushed = subparsers.add_parser("write-pushed")
    write_pushed.add_argument("--image-id", required=True)
    write_pushed.add_argument("--registry-image", required=True)
    write_pushed.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))

    verify_local = subparsers.add_parser("verify-local")
    verify_local.add_argument("--image-id", required=True)
    verify_local.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))

    verify_deploy = subparsers.add_parser("verify-deploy")
    verify_deploy.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))

    args = parser.parse_args()
    try:
        if args.command == "source-info":
            print(json.dumps(_source_info(), sort_keys=True))
        elif args.command == "update-conformance":
            _update_conformance(
                args.image_id,
                args.platform,
                Path(args.regression),
                Path(args.browser),
                args.registry_image,
            )
        elif args.command == "write-local":
            _write(
                Path(args.artifact),
                _candidate(args.image_id, "LOCAL_VALIDATED_NOT_PUSHED", None),
            )
        elif args.command == "write-pushed":
            _write(
                Path(args.artifact),
                _candidate(args.image_id, "PASS", args.registry_image),
            )
        elif args.command == "verify-local":
            _verify_candidate(Path(args.artifact), False, args.image_id)
        elif args.command == "verify-deploy":
            candidate = _verify_candidate(Path(args.artifact), True, None)
            print(candidate["registry"]["image"])
    except (CandidateError, KeyError, TypeError, ValueError) as error:
        print(f"release candidate rejected: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
