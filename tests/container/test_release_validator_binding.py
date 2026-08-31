from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.container
ROOT = Path(__file__).resolve().parents[2]


def test_validator_binds_exact_image_manifest_regression_and_browser(tmp_path: Path) -> None:
    manifest_path = ROOT / "tests/release/release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_digest = "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    image_digest = "sha256:" + "a" * 64
    regression = {
        "status": "PASS",
        "imageDigest": image_digest,
        "releaseManifestDigest": manifest_digest,
        "suites": [
            {"name": name, "status": "PASS", "collected": baseline}
            for name, baseline in manifest["regressionBaselines"].items()
        ],
    }
    browser = {
        "status": "PASS",
        "imageDigest": image_digest,
        "releaseManifestDigest": manifest_digest,
        "browser": "chromium-cdp-real-ui",
        "listApps": ["payment_user_agent"],
        "appSelected": "payment_user_agent",
        "interactions": [
            "有料の外部エージェントに、デモ予約商品を1件シミュレーション購入し、デモの予約確認を発行するよう依頼してください。",
            "承認",
            "承認",
            "refresh",
            "東京で2026年9月12日から9月14日まで、2名で宿泊できるホテル候補を検索してください。",
            "承認",
            "paid refund-required booking",
            "承認",
            "承認",
            "承認",
            "paid privacy booking",
            "承認",
        ],
        "completedAfterRefresh": True,
    }
    regression_path = tmp_path / "regression.json"
    browser_path = tmp_path / "browser.json"
    conformance_path = tmp_path / "conformance.json"
    output = tmp_path / "validation.json"
    regression_path.write_text(json.dumps(regression), encoding="utf-8")
    browser_path.write_text(json.dumps(browser), encoding="utf-8")
    regression_digest = "sha256:" + hashlib.sha256(regression_path.read_bytes()).hexdigest()
    browser_digest = "sha256:" + hashlib.sha256(browser_path.read_bytes()).hexdigest()
    conformance = {
        "acceptance": manifest["acceptance"],
        "tests": {
            "releaseImageDigest": image_digest,
            "repository": {"status": "PASS", "passed": 166, "failed": 0},
        },
        "evidence": {
            "releaseManifestDigest": manifest_digest,
            "regressionArtifactDigest": regression_digest,
            "browserArtifactDigest": browser_digest,
        },
    }
    conformance_path.write_text(json.dumps(conformance), encoding="utf-8")

    command = [
        sys.executable,
        str(ROOT / "scripts/validate_ap2_x402_release.py"),
        "--release-manifest",
        str(manifest_path),
        "--regression-result",
        str(regression_path),
        "--browser-evidence",
        str(browser_path),
        "--expected-image-digest",
        image_digest,
        "--conformance",
        str(conformance_path),
        "--output",
        str(output),
        "--skip-suite-execution",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 1
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["failures"] == {
        "markerSuites": "NOT RUN (--skip-suite-execution is non-promotable)"
    }

    browser["imageDigest"] = "sha256:" + "b" * 64
    browser_path.write_text(json.dumps(browser), encoding="utf-8")
    subprocess.run(command, check=False, capture_output=True, text=True)
    tampered = json.loads(output.read_text(encoding="utf-8"))
    assert tampered["failures"]["browserImageDigest"] == "sha256:" + "b" * 64
    assert "conformanceBrowserArtifactDigest" in tampered["failures"]

    browser["imageDigest"] = image_digest
    browser_path.write_text(json.dumps(browser), encoding="utf-8")
    conformance["evidence"]["browserArtifactDigest"] = (
        "sha256:" + hashlib.sha256(browser_path.read_bytes()).hexdigest()
    )
    stale_cases = (
        ("conformanceImageDigest", ("tests", "releaseImageDigest"), "sha256:" + "c" * 64),
        ("conformancePaymentCount", ("tests", "repository", "passed"), 165),
        ("conformanceManifestDigest", ("evidence", "releaseManifestDigest"), "sha256:" + "d" * 64),
        ("conformanceRegressionArtifactDigest", ("evidence", "regressionArtifactDigest"), "sha256:" + "f" * 64),
        ("conformanceBrowserArtifactDigest", ("evidence", "browserArtifactDigest"), "sha256:" + "e" * 64),
    )
    for failure_name, path, stale_value in stale_cases:
        stale = copy.deepcopy(conformance)
        cursor = stale
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = stale_value
        conformance_path.write_text(json.dumps(stale), encoding="utf-8")
        subprocess.run(command, check=False, capture_output=True, text=True)
        stale_result = json.loads(output.read_text(encoding="utf-8"))
        assert failure_name in stale_result["failures"]

    conformance_path.write_text(json.dumps(conformance), encoding="utf-8")
    semantic_equivalent = json.dumps(regression, indent=2, sort_keys=True)
    assert json.loads(semantic_equivalent) == regression
    regression_path.write_text(semantic_equivalent, encoding="utf-8")
    assert (
        "sha256:" + hashlib.sha256(regression_path.read_bytes()).hexdigest()
        != conformance["evidence"]["regressionArtifactDigest"]
    )
    mutated = subprocess.run(command, check=False, capture_output=True, text=True)
    assert mutated.returncode != 0
    mutated_result = json.loads(output.read_text(encoding="utf-8"))
    assert "conformanceRegressionArtifactDigest" in mutated_result["failures"]
