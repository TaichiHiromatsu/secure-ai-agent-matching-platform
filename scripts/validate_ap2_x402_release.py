#!/usr/bin/env python3
"""Bind marker, regression, browser, manifest, and exact-image release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


REQUIRED_ACCEPTANCE = tuple(
    f"ACC-{number:03d}" for number in (*range(1, 30), *range(31, 36))
)
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _run_suite(marker: str, report: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "tests",
            "-m",
            marker,
            f"--junitxml={report}",
        ],
        check=False,
    )
    root = ET.parse(report).getroot()
    cases = list(root.iter("testcase"))
    failures = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    passed = completed.returncode == 0 and bool(cases) and not (failures or errors or skipped)
    return {
        "status": "PASS" if passed else "FAIL",
        "collected": len(cases),
        "failures": failures,
        "errors": errors,
        "skippedOrXfailed": skipped,
    }


def _load(path: Path, failures: dict[str, object], label: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        failures[label] = f"{type(error).__name__}: {error}"
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-manifest", default="tests/release/release_manifest.json"
    )
    parser.add_argument(
        "--regression-result", default="artifacts/regression-result.json"
    )
    parser.add_argument(
        "--browser-evidence", default="artifacts/browser-evidence.json"
    )
    parser.add_argument("--expected-image-digest", required=True)
    parser.add_argument("--conformance", help="Optional non-image documentation cross-check")
    parser.add_argument(
        "--output", default="artifacts/ap2-x402-release-validation.json"
    )
    parser.add_argument("--skip-suite-execution", action="store_true")
    args = parser.parse_args()

    failures: dict[str, object] = {}
    expected_digest = args.expected_image_digest
    if not DIGEST_PATTERN.fullmatch(expected_digest):
        failures["expectedImageDigest"] = "must be an exact sha256 digest"

    manifest_path = Path(args.release_manifest)
    manifest = _load(manifest_path, failures, "releaseManifest")
    manifest_digest = _digest(manifest_path) if manifest_path.is_file() else "MISSING"
    regression_path = Path(args.regression_result)
    browser_path = Path(args.browser_evidence)
    regression = _load(regression_path, failures, "regressionResult")
    browser = _load(browser_path, failures, "browserEvidence")
    regression_digest = (
        _digest(regression_path) if regression_path.is_file() else "MISSING"
    )
    browser_digest = _digest(browser_path) if browser_path.is_file() else "MISSING"

    acceptance = manifest.get("acceptance", {})
    acceptance_failures = {
        item: acceptance.get(item, "MISSING")
        for item in REQUIRED_ACCEPTANCE
        if not str(acceptance.get(item, "")).startswith("PASS")
    }
    if acceptance.get("ACC-030") != "NOT_RUN_CONDITIONAL":
        acceptance_failures["ACC-030"] = acceptance.get("ACC-030", "MISSING")
    if acceptance_failures:
        failures["acceptance"] = acceptance_failures
    if manifest.get("officialX402") != "NOT RUN" or manifest.get("onChainSettlement") != "NOT RUN":
        failures["simulationBoundary"] = "official x402/on-chain must remain NOT RUN"
    if manifest.get("adkApps") != ["payment_user_agent"]:
        failures["adkApps"] = manifest.get("adkApps")

    for label, artifact in (("regression", regression), ("browser", browser)):
        if artifact.get("status") != "PASS":
            failures[f"{label}Status"] = artifact.get("status", "MISSING")
        if artifact.get("imageDigest") != expected_digest:
            failures[f"{label}ImageDigest"] = artifact.get("imageDigest", "MISSING")
        if artifact.get("releaseManifestDigest") != manifest_digest:
            failures[f"{label}ManifestDigest"] = artifact.get(
                "releaseManifestDigest", "MISSING"
            )

    baselines = manifest.get("regressionBaselines", {})
    actual_suites = {suite.get("name"): suite for suite in regression.get("suites", [])}
    for name, baseline in baselines.items():
        suite = actual_suites.get(name, {})
        if suite.get("status") != "PASS" or int(suite.get("collected", -1)) < int(baseline):
            failures[f"regressionSuite:{name}"] = suite or "MISSING"

    browser_contract = manifest.get("browser", {})
    for field in ("browser", "appSelected", "interactions", "completedAfterRefresh"):
        expected = {
            "browser": browser_contract.get("engine"),
            "appSelected": browser_contract.get("appSelected"),
            "interactions": browser_contract.get("interactions"),
            "completedAfterRefresh": browser_contract.get("completedAfterRefresh"),
        }[field]
        if browser.get(field) != expected:
            failures[f"browser:{field}"] = browser.get(field, "MISSING")
    if browser.get("listApps") != ["payment_user_agent"]:
        failures["browser:listApps"] = browser.get("listApps", "MISSING")

    conformance_digest: str | None = None
    if args.conformance:
        conformance_path = Path(args.conformance)
        conformance = _load(conformance_path, failures, "conformance")
        conformance_digest = (
            _digest(conformance_path) if conformance_path.is_file() else "MISSING"
        )
        if conformance.get("acceptance") != acceptance:
            failures["conformanceAcceptance"] = "documentation differs from frozen manifest"
        conformance_tests = conformance.get("tests", {})
        if conformance_tests.get("releaseImageDigest") != expected_digest:
            failures["conformanceImageDigest"] = conformance_tests.get(
                "releaseImageDigest", "MISSING"
            )
        payment_suite = actual_suites.get("payment-release", {})
        expected_payment_count = payment_suite.get("collected", "MISSING")
        repository_result = conformance_tests.get("repository", {})
        if (
            repository_result.get("status") != "PASS"
            or repository_result.get("failed") != 0
            or repository_result.get("passed") != expected_payment_count
        ):
            failures["conformancePaymentCount"] = {
                "expected": expected_payment_count,
                "observed": repository_result,
            }
        conformance_evidence = conformance.get("evidence", {})
        if conformance_evidence.get("releaseManifestDigest") != manifest_digest:
            failures["conformanceManifestDigest"] = conformance_evidence.get(
                "releaseManifestDigest", "MISSING"
            )
        if conformance_evidence.get("regressionArtifactDigest") != regression_digest:
            failures["conformanceRegressionArtifactDigest"] = (
                conformance_evidence.get("regressionArtifactDigest", "MISSING")
            )
        if conformance_evidence.get("browserArtifactDigest") != browser_digest:
            failures["conformanceBrowserArtifactDigest"] = conformance_evidence.get(
                "browserArtifactDigest", "MISSING"
            )

    suites: dict[str, dict[str, object]] = {}
    if not args.skip_suite_execution:
        with tempfile.TemporaryDirectory(prefix="ap2-x402-release-") as temporary:
            for marker in manifest.get("requiredMarkers", []):
                suites[marker] = _run_suite(
                    marker, Path(temporary) / f"{marker}.xml"
                )
        marker_failures = {
            name: result
            for name, result in suites.items()
            if result["status"] != "PASS"
        }
        if marker_failures:
            failures["markerSuites"] = marker_failures
    else:
        failures["markerSuites"] = "NOT RUN (--skip-suite-execution is non-promotable)"

    result = {
        "schemaVersion": "ap2-x402-release-validation/2",
        "status": "PASS" if not failures else "FAIL",
        "imageDigest": expected_digest,
        "releaseManifestDigest": manifest_digest,
        "conformanceReportDigest": conformance_digest,
        "officialX402": "NOT RUN",
        "onChainSettlement": "NOT RUN",
        "failures": failures,
        "suites": suites,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
