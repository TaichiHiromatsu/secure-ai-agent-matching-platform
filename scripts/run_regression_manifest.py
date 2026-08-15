#!/usr/bin/env python3
"""Execute the versioned regression manifest and reject collection shrinkage."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="tests/regression/suite_manifest.json"
    )
    parser.add_argument("--output", default="artifacts/regression-result.json")
    parser.add_argument(
        "--release-manifest", default="tests/release/release_manifest.json"
    )
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    results = []
    overall = True
    with tempfile.TemporaryDirectory(prefix="regression-manifest-") as temporary:
        for suite in manifest["suites"]:
            report = Path(temporary) / f"{suite['name']}.xml"
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-q",
                *suite["paths"],
                f"--junitxml={report}",
            ]
            environment = os.environ.copy()
            environment.setdefault("WANDB_DISABLED", "true")
            completed = subprocess.run(command, check=False, env=environment)
            root = ET.parse(report).getroot()
            cases = list(root.iter("testcase"))
            collected = len(cases)
            failures = sum(case.find("failure") is not None for case in cases)
            errors = sum(case.find("error") is not None for case in cases)
            skipped_nodes = []
            for case in cases:
                if case.find("skipped") is not None:
                    skipped_nodes.append(
                        f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}"
                    )
            unexpected_skips = [
                node
                for node in skipped_nodes
                if not any(
                    fnmatch.fnmatch(node, pattern)
                    for pattern in suite.get("allowedSkips", [])
                )
            ]
            passed = (
                completed.returncode == 0
                and collected >= suite["minimumCollected"]
                and failures == 0
                and errors == 0
                and not unexpected_skips
            )
            overall = overall and passed
            results.append(
                {
                    "name": suite["name"],
                    "status": "PASS" if passed else "FAIL",
                    "collected": collected,
                    "minimumCollected": suite["minimumCollected"],
                    "failures": failures,
                    "errors": errors,
                    "skipped": skipped_nodes,
                    "unexpectedSkips": unexpected_skips,
                }
            )
    output = {
        "schemaVersion": manifest["schemaVersion"],
        "status": "PASS" if overall else "FAIL",
        "imageDigest": os.environ.get("RELEASE_IMAGE_DIGEST", "UNSET"),
        "releaseManifestDigest": "sha256:"
        + hashlib.sha256(Path(args.release_manifest).read_bytes()).hexdigest(),
        "regressionManifestDigest": "sha256:"
        + hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "suites": results,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
