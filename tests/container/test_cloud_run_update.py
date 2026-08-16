from __future__ import annotations

import json
import os
from pathlib import Path
import re
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
TAG = "pc-" + "a" * 12
SERVICE_URL = "https://payment-user-agent-demo-kzeuhywicq-an.a.run.app"
TAG_URL = f"https://{TAG}---{SERVICE_URL.removeprefix('https://')}"


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


def _run_candidate_tag_validation(
    service: str, tag: str
) -> subprocess.CompletedProcess[str]:
    source = _text("deploy/update-payment-demo-cloudrun.sh")
    function = re.search(r"assert_candidate_tag\(\) \{.*?\n\}", source, re.DOTALL)
    assert function is not None
    validator = "\n".join(
        (
            "MAX_SERVICE_TAG_LENGTH=46",
            'fail() { printf "%s\\n" "$*" >&2; exit 2; }',
            function.group(0),
            'assert_candidate_tag "$1" "$2"',
        )
    )
    return subprocess.run(
        ["bash", "-c", validator, "candidate-tag-test", service, tag],
        check=False,
        capture_output=True,
        text=True,
    )


def test_candidate_tag_is_deterministic_valid_and_within_fixed_service_limit() -> None:
    assert TAG == f"pc-{IMAGE_DIGEST.removeprefix('sha256:')[:12]}"
    assert re.fullmatch(r"[a-z]([a-z0-9-]{0,61}[a-z0-9])?", TAG)
    assert len(f"payment-user-agent-demo-{TAG}") == 39
    result = _run_candidate_tag_validation("payment-user-agent-demo", TAG)
    assert result.returncode == 0, result.stderr


def test_candidate_tag_accepts_maximum_service_and_fails_fast_past_it() -> None:
    maximum_service = "s" + "1" * 29
    assert len(f"{maximum_service}-{TAG}") == 46
    accepted = _run_candidate_tag_validation(maximum_service, TAG)
    assert accepted.returncode == 0, accepted.stderr

    too_long_service = maximum_service + "1"
    rejected = _run_candidate_tag_validation(too_long_service, TAG)
    assert rejected.returncode == 2
    assert "exceed the 46-character limit" in rejected.stderr

    malformed = _run_candidate_tag_validation(maximum_service, "PC_aaaaaaaaaaaa")
    assert malformed.returncode == 2
    assert "invalid Cloud Run format" in malformed.stderr


def _fake_gcloud(tag_url: str = TAG_URL) -> str:
    return f"""#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >>"$FAKE_GCLOUD_LOG"
phase="initial"
if [ -f "$FAKE_GCLOUD_PHASE" ]; then phase="$(cat "$FAKE_GCLOUD_PHASE")"; fi
case "$1 $2 $3" in
  "config get-value project") printf '%s\n' gen-lang-client-0585901015 ;;
  "run services list") printf '%s\n' payment-user-agent-demo other-service ;;
  "artifacts docker images") printf '%s\n' "$FAKE_REGISTRY_DIGEST" ;;
  "run revisions describe")
    if printf '%s\n' "$*" | grep -Fq -- '--format=json'; then
      printf '{{"metadata":{{"name":"%s","labels":{{"serving.knative.dev/service":"%s"}},"annotations":{{"autoscaling.knative.dev/minScale":"%s","autoscaling.knative.dev/maxScale":"%s"}}}},"spec":{{"containerConcurrency":%s,"timeoutSeconds":%s,"containers":[{{"ports":[{{"containerPort":%s}}],"resources":{{"limits":{{"cpu":"%s","memory":"%s"}}}},"env":[{{"name":"EPHEMERAL_CLOUD_RUN_DEMO","value":"%s"}},{{"name":"MEDIATION_STORE_MODE","value":"%s"}},{{"name":"APP_ENV","value":"%s"}},{{"name":"DEV_MODE","value":"%s"}}%s]}}]}},"status":{{"conditions":[{{"type":"Ready","status":"%s"}}]}}}}\n' \
        "$4" "$FAKE_REVISION_SERVICE" "$FAKE_MIN_SCALE" "$FAKE_MAX_SCALE" \
        "$FAKE_CONCURRENCY" "$FAKE_TIMEOUT" "$FAKE_PORT" "$FAKE_CPU" \
        "$FAKE_MEMORY" "$FAKE_EPHEMERAL_CLOUD_RUN_DEMO" \
        "$FAKE_MEDIATION_STORE_MODE" "$FAKE_APP_ENV" "$FAKE_DEV_MODE" \
        "$FAKE_EXTRA_ENV" "$FAKE_READY_STATUS"
    elif [ "$4" = "old-revision" ]; then
      printf '%s\n' "$FAKE_OLD_IMAGE"
    else
      printf '%s\n' "$FAKE_CANDIDATE_IMAGE"
    fi
    ;;
  "run services update")
    printf '%s' candidate >"$FAKE_GCLOUD_PHASE"
    ;;
  "run services update-traffic")
    if printf '%s\n' "$*" | grep -Fq -- '--to-revisions candidate-revision=100'; then
      printf '%s' promoted >"$FAKE_GCLOUD_PHASE"
    elif printf '%s\n' "$*" | grep -Fq -- '--to-revisions old-revision=100'; then
      if [ "$phase" = "candidate" ]; then
        printf '%s' rollback-tagged >"$FAKE_GCLOUD_PHASE"
      else
        printf '%s' rollback >"$FAKE_GCLOUD_PHASE"
      fi
    elif printf '%s\n' "$*" | grep -Fq -- '--remove-tags {TAG}'; then
      if [ "$phase" = "promoted" ]; then
        printf '%s' cleaned >"$FAKE_GCLOUD_PHASE"
      elif [ "$phase" = "rollback-tagged" ]; then
        printf '%s' rollback >"$FAKE_GCLOUD_PHASE"
      fi
    else
      printf 'unexpected traffic update: %s\n' "$*" >&2
      exit 90
    fi
    ;;
  "run services describe")
    case "$phase" in
      initial)
        printf '%s\n' '{{"metadata":{{"name":"payment-user-agent-demo"}},"status":{{"url":"{SERVICE_URL}","latestReadyRevisionName":"old-revision","traffic":[{{"revisionName":"old-revision","percent":100}}]}}}}'
        ;;
      candidate)
        printf '{{"metadata":{{"name":"payment-user-agent-demo"}},"status":{{"url":"{SERVICE_URL}","latestReadyRevisionName":"candidate-revision","traffic":[{{"revisionName":"old-revision"%s}},{{"revisionName":"%s"%s,"tag":"{TAG}","url":"{tag_url}"}}%s]}}}}\n' \
          "$FAKE_OLD_PERCENT_FRAGMENT" "$FAKE_TAG_REVISION" \
          "$FAKE_CANDIDATE_PERCENT_FRAGMENT" "$FAKE_EXTRA_TRAFFIC"
        ;;
      promoted)
        printf '%s\n' '{{"metadata":{{"name":"payment-user-agent-demo"}},"status":{{"url":"{SERVICE_URL}","latestReadyRevisionName":"candidate-revision","traffic":[{{"revisionName":"candidate-revision","percent":100}},{{"revisionName":"candidate-revision","percent":0,"tag":"{TAG}","url":"{tag_url}"}}]}}}}'
        ;;
      cleaned)
        printf '%s\n' '{{"metadata":{{"name":"payment-user-agent-demo"}},"status":{{"url":"{SERVICE_URL}","latestReadyRevisionName":"candidate-revision","traffic":[{{"revisionName":"candidate-revision","percent":100}}]}}}}'
        ;;
      rollback-tagged)
        printf '%s\n' '{{"metadata":{{"name":"payment-user-agent-demo"}},"status":{{"url":"{SERVICE_URL}","latestReadyRevisionName":"candidate-revision","traffic":[{{"revisionName":"old-revision","percent":100}},{{"revisionName":"candidate-revision","tag":"{TAG}","url":"{TAG_URL}"}},{{"revisionName":"other-revision","tag":"keep-tag","url":"https://keep-tag---payment-user-agent-demo-kzeuhywicq-an.a.run.app"}}]}}}}'
        ;;
      rollback)
        printf '%s\n' '{{"metadata":{{"name":"payment-user-agent-demo"}},"status":{{"url":"{SERVICE_URL}","latestReadyRevisionName":"candidate-revision","traffic":[{{"revisionName":"old-revision","percent":100}},{{"revisionName":"other-revision","tag":"keep-tag","url":"https://keep-tag---payment-user-agent-demo-kzeuhywicq-an.a.run.app"}}]}}}}'
        ;;
    esac
    ;;
  *) printf 'unexpected fake gcloud call: %s\n' "$*" >&2; exit 91 ;;
esac
"""


def _prepare_fake_workspace(
    tmp_path: Path, *, tag_url: str = TAG_URL
) -> tuple[Path, dict[str, str]]:
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    script = deploy / "update-payment-demo-cloudrun.sh"
    shutil.copyfile(ROOT / "deploy/update-payment-demo-cloudrun.sh", script)
    script.chmod(0o755)
    (tmp_path / "scripts").mkdir()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gcloud = fake_bin / "gcloud"
    gcloud.write_text(_fake_gcloud(tag_url), encoding="utf-8")
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
        "FAKE_REVISION_SERVICE": "payment-user-agent-demo",
        "FAKE_READY_STATUS": "True",
        "FAKE_CANDIDATE_IMAGE": IMAGE,
        "FAKE_TAG_REVISION": "candidate-revision",
        "FAKE_CANDIDATE_PERCENT_FRAGMENT": "",
        "FAKE_OLD_IMAGE": OLD_IMAGE,
        "FAKE_REGISTRY_DIGEST": IMAGE_DIGEST,
        "FAKE_OLD_PERCENT_FRAGMENT": ',"percent":100',
        "FAKE_EXTRA_TRAFFIC": "",
        "FAKE_CONCURRENCY": "1",
        "FAKE_TIMEOUT": "3600",
        "FAKE_PORT": "8080",
        "FAKE_CPU": "1",
        "FAKE_MEMORY": "2Gi",
        "FAKE_MIN_SCALE": "1",
        "FAKE_MAX_SCALE": "1",
        "FAKE_EXTRA_ENV": "",
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


def _write_interrupted_preflight(
    tmp_path: Path, *, candidate_tag: str = TAG
) -> Path:
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    state_path = tmp_path / "artifacts/cloud-run-update-state.json"
    state_path.write_text(
        json.dumps(
            {
                "schemaVersion": "cloud-run-payment-demo-update/1",
                "status": "PREFLIGHT",
                "project": "gen-lang-client-0585901015",
                "region": "asia-northeast1",
                "service": "payment-user-agent-demo",
                "oldRevision": "old-revision",
                "oldImage": OLD_IMAGE,
                "oldTraffic": [
                    {"revisionName": "old-revision", "percent": 100}
                ],
                "candidateImage": IMAGE,
                "candidateTag": candidate_tag,
                "candidateRevision": "NOT_CREATED",
                "candidateUrl": "NOT_CREATED",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "gcloud.phase").write_text("candidate", encoding="utf-8")
    return state_path


@pytest.mark.parametrize("candidate_percent_fragment", ["", ',"percent":0'])
def test_adopt_reconciles_exact_existing_candidate_without_cloud_mutation(
    tmp_path: Path, candidate_percent_fragment: str
) -> None:
    script, environment = _prepare_fake_workspace(tmp_path)
    environment["FAKE_CANDIDATE_PERCENT_FRAGMENT"] = candidate_percent_fragment
    state_path = _write_interrupted_preflight(tmp_path)

    adopted = _run(script, "adopt", environment)
    assert adopted.returncode == 0, adopted.stderr
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "CANDIDATE"
    assert state["candidateRevision"] == "candidate-revision"
    assert state["candidateImage"] == IMAGE
    assert state["candidateTag"] == TAG
    assert state["candidateUrl"] == TAG_URL
    assert state["reconciliation"]["mode"] == "READ_ONLY_ADOPTION"
    assert state["reconciliation"]["previousStatus"] == "PREFLIGHT"
    assert state["reconciliation"]["cloudMutation"] is False
    assert state_path.stat().st_mode & 0o777 == 0o600
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
        state["reconciliation"]["adoptedAt"],
    )

    calls = (tmp_path / "gcloud.log").read_text(encoding="utf-8")
    assert "run services update payment-user-agent-demo" not in calls
    assert "run services update-traffic" not in calls
    assert "run services describe" in calls
    assert "run revisions describe" in calls

    missing_evidence = _run(script, "verify", environment)
    assert missing_evidence.returncode == 2
    assert "tag-bound E2E evidence is missing" in missing_evidence.stderr
    (tmp_path / "artifacts/cloud-run-tag-e2e.json").write_text(
        json.dumps(_evidence()), encoding="utf-8"
    )
    verified = _run(script, "verify", environment)
    assert verified.returncode == 0, verified.stderr
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "VERIFIED"


@pytest.mark.parametrize(
    "state_override",
    [
        {"candidateRevision": "candidate-revision"},
        {"candidateUrl": TAG_URL},
        {
            "reconciliation": {
                "mode": "READ_ONLY_ADOPTION",
                "previousStatus": "PREFLIGHT",
                "cloudMutation": False,
            }
        },
    ],
)
def test_adopt_rejects_partial_local_state_without_rewriting_it(
    tmp_path: Path, state_override: dict[str, object]
) -> None:
    script, environment = _prepare_fake_workspace(tmp_path)
    state_path = _write_interrupted_preflight(tmp_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(state_override)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    before = state_path.read_bytes()

    adopted = _run(script, "adopt", environment)
    assert adopted.returncode == 2
    assert "refuses a partially reconciled local state" in adopted.stderr
    assert state_path.read_bytes() == before

    calls = (tmp_path / "gcloud.log").read_text(encoding="utf-8")
    assert "artifacts docker images" not in calls
    assert "run revisions describe" not in calls
    assert "run services update payment-user-agent-demo" not in calls
    assert "run services update-traffic" not in calls


@pytest.mark.parametrize(
    ("environment_override", "expected_error"),
    [
        (
            {"FAKE_CANDIDATE_IMAGE": OLD_IMAGE},
            "revision digest differs from the saved candidate",
        ),
        (
            {"FAKE_CANDIDATE_PERCENT_FRAGMENT": ',"percent":1'},
            "zero-traffic tagged candidate was not found",
        ),
        (
            {"FAKE_CANDIDATE_PERCENT_FRAGMENT": ',"percent":null'},
            "zero-traffic tagged candidate was not found",
        ),
        (
            {"FAKE_TAG_REVISION": "other-revision"},
            "zero-traffic tagged candidate was not found",
        ),
        (
            {
                "FAKE_EXTRA_TRAFFIC": (
                    f',{{"revisionName":"candidate-revision",'
                    f'"tag":"{TAG}","url":"{TAG_URL}"}}'
                )
            },
            "zero-traffic tagged candidate was not found",
        ),
        (
            {
                "FAKE_EXTRA_TRAFFIC": (
                    f',{{"revisionName":"other-revision",'
                    f'"tag":"{TAG}","url":"{TAG_URL}"}}'
                )
            },
            "zero-traffic tagged candidate was not found",
        ),
        (
            {
                "FAKE_EXTRA_TRAFFIC": (
                    ',{"revisionName":"other-revision","percent":1,'
                    '"tag":"other"}'
                )
            },
            "default traffic is not 100% on old-revision",
        ),
        (
            {"FAKE_OLD_PERCENT_FRAGMENT": ',"percent":99'},
            "default traffic is not 100% on old-revision",
        ),
        (
            {"FAKE_OLD_PERCENT_FRAGMENT": ""},
            "default traffic is not 100% on old-revision",
        ),
        (
            {"FAKE_MEDIATION_STORE_MODE": "sqlite"},
            "not the exact ephemeral memory-store profile",
        ),
        (
            {
                "FAKE_EXTRA_ENV": (
                    ',{"name":"UNEXPECTED_SECRET","value":"present"}'
                )
            },
            "not the exact ephemeral memory-store profile",
        ),
        (
            {"FAKE_REVISION_SERVICE": "other-service"},
            "not Ready with the exact fixed service shape",
        ),
        (
            {"FAKE_READY_STATUS": "False"},
            "not Ready with the exact fixed service shape",
        ),
        (
            {"FAKE_CONCURRENCY": "2"},
            "not Ready with the exact fixed service shape",
        ),
        (
            {"FAKE_TIMEOUT": "300"},
            "not Ready with the exact fixed service shape",
        ),
        (
            {"FAKE_PORT": "8081"},
            "not Ready with the exact fixed service shape",
        ),
        (
            {"FAKE_CPU": "2"},
            "not Ready with the exact fixed service shape",
        ),
        (
            {"FAKE_MEMORY": "1Gi"},
            "not Ready with the exact fixed service shape",
        ),
        (
            {"FAKE_MIN_SCALE": "0"},
            "not Ready with the exact fixed service shape",
        ),
        (
            {"FAKE_MAX_SCALE": "2"},
            "not Ready with the exact fixed service shape",
        ),
        (
            {"FAKE_REGISTRY_DIGEST": OLD_DIGEST},
            "registry digest verification failed during candidate adoption",
        ),
        (
            {"FAKE_OLD_IMAGE": IMAGE},
            "saved rollback revision digest changed during candidate adoption",
        ),
    ],
)
def test_adopt_rejects_non_exact_cloud_candidate_and_preserves_preflight(
    tmp_path: Path,
    environment_override: dict[str, str],
    expected_error: str,
) -> None:
    script, environment = _prepare_fake_workspace(tmp_path)
    environment.update(environment_override)
    state_path = _write_interrupted_preflight(tmp_path)
    before = state_path.read_bytes()

    adopted = _run(script, "adopt", environment)
    assert adopted.returncode == 2
    assert expected_error in adopted.stderr
    assert state_path.read_bytes() == before
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "PREFLIGHT"
    assert state["candidateRevision"] == "NOT_CREATED"
    assert state["candidateUrl"] == "NOT_CREATED"

    calls = (tmp_path / "gcloud.log").read_text(encoding="utf-8")
    assert "run services update payment-user-agent-demo" not in calls
    assert "run services update-traffic" not in calls


def test_adopt_rejects_candidate_tag_not_derived_from_saved_digest(
    tmp_path: Path,
) -> None:
    script, environment = _prepare_fake_workspace(tmp_path)
    state_path = _write_interrupted_preflight(tmp_path, candidate_tag="pc-bbbbbbbbbbbb")

    adopted = _run(script, "adopt", environment)
    assert adopted.returncode == 2
    assert "tag is not derived from the candidate digest" in adopted.stderr
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "PREFLIGHT"
    calls = (tmp_path / "gcloud.log").read_text(encoding="utf-8")
    assert "run services update payment-user-agent-demo" not in calls
    assert "run services update-traffic" not in calls


@pytest.mark.parametrize(
    "tag_url",
    [
        f"https://wrong---{SERVICE_URL.removeprefix('https://')}",
        f"https://{TAG}---other-service-kzeuhywicq-an.a.run.app",
        f"{TAG_URL}.example.com",
        f"{TAG_URL}/health",
    ],
)
def test_adopt_rejects_tag_url_not_bound_to_advertised_service_origin(
    tmp_path: Path, tag_url: str
) -> None:
    script, environment = _prepare_fake_workspace(tmp_path, tag_url=tag_url)
    state_path = _write_interrupted_preflight(tmp_path)

    adopted = _run(script, "adopt", environment)
    assert adopted.returncode == 2
    assert "does not match the exact tag" in adopted.stderr
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "PREFLIGHT"
    assert state["candidateRevision"] == "NOT_CREATED"
    assert state["candidateUrl"] == "NOT_CREATED"
    calls = (tmp_path / "gcloud.log").read_text(encoding="utf-8")
    assert "run services update payment-user-agent-demo" not in calls
    assert "run services update-traffic" not in calls


@pytest.mark.parametrize(
    "tag_url",
    [
        f"https://wrong---{SERVICE_URL.removeprefix('https://')}",
        f"https://{TAG}---other-service-kzeuhywicq-an.a.run.app",
        f"{TAG_URL}.example.com",
        f"{TAG_URL}/health",
    ],
)
def test_candidate_rejects_tag_url_not_bound_to_advertised_service_origin(
    tmp_path: Path, tag_url: str
) -> None:
    script, environment = _prepare_fake_workspace(tmp_path, tag_url=tag_url)

    candidate = _run(script, "candidate", environment)
    assert candidate.returncode == 2
    assert "does not match the exact tag" in candidate.stderr

    state = json.loads(
        (tmp_path / "artifacts/cloud-run-update-state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["status"] == "PREFLIGHT"
    assert state["candidateRevision"] == "NOT_CREATED"
    assert state["candidateUrl"] == "NOT_CREATED"

    calls = (tmp_path / "gcloud.log").read_text(encoding="utf-8")
    assert "run services update payment-user-agent-demo" in calls
    assert "--no-traffic" in calls
    assert "run services update-traffic" not in calls


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


def test_interrupted_preflight_rolls_back_only_saved_tag_without_revision_delete(
    tmp_path: Path,
) -> None:
    script, environment = _prepare_fake_workspace(tmp_path)
    state_path = _write_interrupted_preflight(tmp_path)

    rolled_back = _run(script, "rollback", environment)
    assert rolled_back.returncode == 0, rolled_back.stderr

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "ROLLED_BACK"
    assert state["oldRevision"] == "old-revision"
    assert state["candidateRevision"] == "NOT_CREATED"
    assert state["candidateUrl"] == "NOT_CREATED"

    calls = (tmp_path / "gcloud.log").read_text(encoding="utf-8")
    traffic_calls = [
        line for line in calls.splitlines() if "run services update-traffic" in line
    ]
    assert len(traffic_calls) == 2
    assert "--to-revisions old-revision=100" in traffic_calls[0]
    assert f"--remove-tags {TAG}" in traffic_calls[1]
    assert "--remove-tags keep-tag" not in calls
    assert "run revisions delete" not in calls
    assert "run services delete" not in calls
    assert (tmp_path / "gcloud.phase").read_text(encoding="utf-8") == "rollback"
