from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.container
ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIGEST = "sha256:" + "a" * 64
OLD_DIGEST = "sha256:" + "b" * 64
REPOSITORY = (
    "asia-northeast1-docker.pkg.dev/gen-lang-client-0585901015/"
    "secure-mediation-agent/payment-user-agent-demo"
)
IMAGE = f"{REPOSITORY}@{IMAGE_DIGEST}"
OLD_IMAGE = f"{REPOSITORY}@{OLD_DIGEST}"
TAG = "payment-candidate-" + "a" * 12
TAG_URL = f"https://{TAG}-fixed.run.app"


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_update_script_is_fixed_tagged_no_traffic_and_reversible() -> None:
    source = _text("deploy/update-payment-demo-cloudrun.sh")
    assert 'PROJECT_ID="gen-lang-client-0585901015"' in source
    assert 'REGION="asia-northeast1"' in source
    assert 'SERVICE_NAME="payment-user-agent-demo"' in source
    assert "gcloud run services update \"$SERVICE_NAME\"" in source
    assert "gcloud run deploy" not in source
    assert "--no-traffic" in source
    assert "--tag \"$candidate_tag\"" in source
    assert "EPHEMERAL_CLOUD_RUN_DEMO=true" in source
    assert "MEDIATION_STORE_MODE=memory" in source
    assert "assert_revision_ephemeral_profile" in source
    assert "cloud-run-tag-e2e/1" in source
    assert '.publicDurabilityProfile == "ephemeral-demo"' in source
    assert '.readiness.mediationStore == {' in source
    assert ".checks.paid" in source
    assert ".checks.free" in source
    assert ".checks.refund" in source
    assert ".checks.browser" in source
    assert ".checks.publicBoundary" in source
    assert "--to-revisions \"${candidate_revision}=100\"" in source
    assert "--to-revisions \"${old_revision}=100\"" in source
    assert "--remove-tags \"$tag\"" in source
    assert "Cloud SQL configuration is forbidden" in source
    assert "services delete" not in source


def _fake_gcloud() -> str:
    return f"""#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >>"$FAKE_GCLOUD_LOG"
phase="initial"
if [ -f "$FAKE_GCLOUD_PHASE" ]; then phase="$(cat "$FAKE_GCLOUD_PHASE")"; fi
case "$1 $2 $3" in
  "config get-value project") printf '%s\n' gen-lang-client-0585901015 ;;
  "run services list") printf '%s\n' payment-user-agent-demo other-service ;;
  "artifacts docker images") printf '%s\n' '{IMAGE_DIGEST}' ;;
  "run revisions describe")
    if printf '%s\n' "$*" | grep -Fq -- '--format=json'; then
      printf '{{"spec":{{"containers":[{{"env":[{{"name":"EPHEMERAL_CLOUD_RUN_DEMO","value":"%s"}},{{"name":"MEDIATION_STORE_MODE","value":"%s"}},{{"name":"APP_ENV","value":"%s"}},{{"name":"DEV_MODE","value":"%s"}}]}}]}}}}\n' \
        "$FAKE_EPHEMERAL_CLOUD_RUN_DEMO" "$FAKE_MEDIATION_STORE_MODE" "$FAKE_APP_ENV" "$FAKE_DEV_MODE"
    elif [ "$4" = "old-revision" ]; then
      printf '%s\n' '{OLD_IMAGE}'
    else
      printf '%s\n' '{IMAGE}'
    fi
    ;;
  "run services update")
    printf '%s' candidate >"$FAKE_GCLOUD_PHASE"
    ;;
  "run services update-traffic")
    if printf '%s\n' "$*" | grep -Fq -- '--to-revisions candidate-revision=100'; then
      printf '%s' promoted >"$FAKE_GCLOUD_PHASE"
    elif printf '%s\n' "$*" | grep -Fq -- '--to-revisions old-revision=100'; then
      printf '%s' rollback >"$FAKE_GCLOUD_PHASE"
    elif printf '%s\n' "$*" | grep -Fq -- '--remove-tags {TAG}'; then
      if [ "$phase" = "promoted" ]; then printf '%s' cleaned >"$FAKE_GCLOUD_PHASE"; fi
    else
      printf 'unexpected traffic update: %s\n' "$*" >&2
      exit 90
    fi
    ;;
  "run services describe")
    case "$phase" in
      initial)
        printf '%s\n' '{{"metadata":{{"name":"payment-user-agent-demo"}},"status":{{"latestReadyRevisionName":"old-revision","traffic":[{{"revisionName":"old-revision","percent":100}}]}}}}'
        ;;
      candidate)
        printf '%s\n' '{{"metadata":{{"name":"payment-user-agent-demo"}},"status":{{"latestReadyRevisionName":"candidate-revision","traffic":[{{"revisionName":"old-revision","percent":100}},{{"revisionName":"candidate-revision","percent":0,"tag":"{TAG}","url":"{TAG_URL}"}}]}}}}'
        ;;
      promoted)
        printf '%s\n' '{{"metadata":{{"name":"payment-user-agent-demo"}},"status":{{"latestReadyRevisionName":"candidate-revision","traffic":[{{"revisionName":"candidate-revision","percent":100}},{{"revisionName":"candidate-revision","percent":0,"tag":"{TAG}","url":"{TAG_URL}"}}]}}}}'
        ;;
      cleaned)
        printf '%s\n' '{{"metadata":{{"name":"payment-user-agent-demo"}},"status":{{"latestReadyRevisionName":"candidate-revision","traffic":[{{"revisionName":"candidate-revision","percent":100}}]}}}}'
        ;;
      rollback)
        printf '%s\n' '{{"metadata":{{"name":"payment-user-agent-demo"}},"status":{{"latestReadyRevisionName":"candidate-revision","traffic":[{{"revisionName":"old-revision","percent":100}}]}}}}'
        ;;
    esac
    ;;
  *) printf 'unexpected fake gcloud call: %s\n' "$*" >&2; exit 91 ;;
esac
"""


def _prepare_fake_workspace(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    script = deploy / "update-payment-demo-cloudrun.sh"
    shutil.copyfile(ROOT / "deploy/update-payment-demo-cloudrun.sh", script)
    script.chmod(0o755)
    (tmp_path / "scripts").mkdir()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gcloud = fake_bin / "gcloud"
    gcloud.write_text(_fake_gcloud(), encoding="utf-8")
    gcloud.chmod(0o755)
    python = fake_bin / "python3"
    python.write_text(f"#!/bin/sh\nprintf '%s\\n' '{IMAGE}'\n", encoding="utf-8")
    python.chmod(0o755)
    curl = fake_bin / "curl"
    curl.write_text("#!/bin/sh\nprintf '%s' OK\n", encoding="utf-8")
    curl.chmod(0o755)

    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_GCLOUD_LOG": str(tmp_path / "gcloud.log"),
        "FAKE_GCLOUD_PHASE": str(tmp_path / "gcloud.phase"),
        "FAKE_EPHEMERAL_CLOUD_RUN_DEMO": "true",
        "FAKE_MEDIATION_STORE_MODE": "memory",
        "FAKE_APP_ENV": "ephemeral-demo",
        "FAKE_DEV_MODE": "false",
    }
    return script, environment


def _run(script: Path, action: str, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), action],
        cwd=script.parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _evidence() -> dict[str, object]:
    return {
        "schemaVersion": "cloud-run-tag-e2e/1",
        "status": "PASS",
        "project": "gen-lang-client-0585901015",
        "region": "asia-northeast1",
        "service": "payment-user-agent-demo",
        "revision": "candidate-revision",
        "image": IMAGE,
        "url": TAG_URL,
        "tag": TAG,
        "publicDurabilityProfile": "ephemeral-demo",
        "readiness": {
            "status": "ready",
            "target": "ephemeral-cloud-run-demo",
            "durability": "NOT PROVIDED",
            "mediationStore": {
                "mode": "memory",
                "durabilityProfile": "ephemeral-demo",
                "schemaVersion": None,
                "writable": True,
                "decryptable": True,
            },
            "checks": {
                "mediationStoreMode": True,
                "mediationStoreProfile": True,
                "mediationStoreSchema": True,
                "mediationStoreProbe": True,
            },
        },
        "checks": {
            "readiness": "PASS",
            "modelProbe": "PASS",
            "paid": "PASS",
            "free": "PASS",
            "refund": "PASS",
            "browser": "PASS",
            "publicBoundary": "PASS",
        },
    }


def test_candidate_requires_exact_evidence_before_promotion_and_rolls_back(
    tmp_path: Path,
) -> None:
    script, environment = _prepare_fake_workspace(tmp_path)

    candidate = _run(script, "candidate", environment)
    assert candidate.returncode == 0, candidate.stderr
    state_path = tmp_path / "artifacts/cloud-run-update-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "CANDIDATE"
    assert state["oldRevision"] == "old-revision"
    assert state["candidateRevision"] == "candidate-revision"
    assert state["candidateImage"] == IMAGE
    assert state["candidateTag"] == TAG
    assert state["candidateUrl"] == TAG_URL

    missing_evidence = _run(script, "verify", environment)
    assert missing_evidence.returncode == 2
    assert "tag-bound E2E evidence is missing" in missing_evidence.stderr
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "CANDIDATE"

    evidence = _evidence()
    mismatched = json.loads(json.dumps(evidence))
    mismatched["readiness"]["mediationStore"]["mode"] = "sqlite"
    (tmp_path / "artifacts/cloud-run-tag-e2e.json").write_text(
        json.dumps(mismatched), encoding="utf-8"
    )
    profile_mismatch = _run(script, "verify", environment)
    assert profile_mismatch.returncode == 2
    assert "E2E evidence is incomplete" in profile_mismatch.stderr
    blocked_promotion = _run(script, "promote", environment)
    assert blocked_promotion.returncode == 2
    assert "promotion requires VERIFIED state" in blocked_promotion.stderr

    (tmp_path / "artifacts/cloud-run-tag-e2e.json").write_text(
        json.dumps(evidence), encoding="utf-8"
    )
    verified = _run(script, "verify", environment)
    assert verified.returncode == 0, verified.stderr
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "VERIFIED"

    promoted = _run(script, "promote", environment)
    assert promoted.returncode == 0, promoted.stderr
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "PROMOTED"

    cleaned = _run(script, "cleanup", environment)
    assert cleaned.returncode == 0, cleaned.stderr
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "CLEANED_AFTER_PROMOTION"

    rolled_back = _run(script, "rollback", environment)
    assert rolled_back.returncode == 0, rolled_back.stderr
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "ROLLED_BACK"

    calls = (tmp_path / "gcloud.log").read_text(encoding="utf-8")
    assert "run deploy" not in calls
    mutating_calls = [line for line in calls.splitlines() if "run services update" in line]
    assert mutating_calls
    assert all("payment-user-agent-demo" in line for line in mutating_calls)
    assert "--no-traffic" in calls
    assert f"--tag {TAG}" in calls
    assert "--to-revisions candidate-revision=100" in calls
    assert "--to-revisions old-revision=100" in calls


def test_revision_profile_mismatch_blocks_promotion_but_not_rollback(
    tmp_path: Path,
) -> None:
    script, environment = _prepare_fake_workspace(tmp_path)
    candidate = _run(script, "candidate", environment)
    assert candidate.returncode == 0, candidate.stderr
    (tmp_path / "artifacts/cloud-run-tag-e2e.json").write_text(
        json.dumps(_evidence()), encoding="utf-8"
    )
    verified = _run(script, "verify", environment)
    assert verified.returncode == 0, verified.stderr

    mismatched_environment = {
        **environment,
        "FAKE_MEDIATION_STORE_MODE": "sqlite",
    }
    blocked = _run(script, "promote", mismatched_environment)
    assert blocked.returncode == 2
    assert "not the exact ephemeral memory-store profile" in blocked.stderr

    rolled_back = _run(script, "rollback", mismatched_environment)
    assert rolled_back.returncode == 0, rolled_back.stderr
    state = json.loads(
        (tmp_path / "artifacts/cloud-run-update-state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["status"] == "ROLLED_BACK"

    calls = (tmp_path / "gcloud.log").read_text(encoding="utf-8")
    assert "--to-revisions candidate-revision=100" not in calls
    assert "--to-revisions old-revision=100" in calls
