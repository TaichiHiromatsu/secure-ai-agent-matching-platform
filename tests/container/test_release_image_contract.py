from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from fastapi.testclient import TestClient
import pytest

import deploy.auth.verify as auth
from secure_mediation_agent.mediation.composition import create_production_controller


pytestmark = pytest.mark.container
ROOT = Path(__file__).resolve().parents[2]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_release_image_contains_worker_merchant_browser_and_verifiers() -> None:
    dockerfile = _text("Dockerfile")
    supervisor = _text("deploy/supervisord.conf")
    assert "chromium" in dockerfile
    assert "verify_payment_demo.sh" in dockerfile
    assert "validate_ap2_x402_release.py" in dockerfile
    assert "run_regression_manifest.py" in dockerfile
    assert "COPY tests /app/tests" in dockerfile
    assert "COPY payment_user_agent /app/payment-apps/payment_user_agent" in dockerfile
    assert "cloud_run_candidate.py" in dockerfile
    assert "[program:workflow_outbox_worker]" in supervisor
    assert "[program:paid_booking_merchant]" in supervisor
    assert "--port 8005" in supervisor


def test_release_manifest_keeps_official_x402_disabled_and_exactly_pinned() -> None:
    manifest = json.loads(
        _text("secure_mediation_agent/spec_manifest.json")
    )
    assert manifest["releaseProfile"] == "x402-wire-simulation/1"
    assert manifest["officialX402"] == "DISABLED / NOT RUN"
    assert manifest["ap2"]["commit"] == "e1ea56db72a6385bce3e5c1112b3a56ce60acb43"
    assert manifest["x402"]["commit"] == "125db5526a965d2325459d1a9df2e274a7e42396"


def test_dedicated_cloud_run_demo_is_explicitly_ephemeral_and_single_instance(
    tmp_path: Path,
) -> None:
    script = _text("deploy/deploy-payment-demo-cloudrun.sh")
    assert 'SERVICE_NAME="payment-user-agent-demo"' in script
    assert 'PROJECT_ID="gen-lang-client-0585901015"' in script
    assert 'REGION="asia-northeast1"' in script
    assert "EPHEMERAL_CLOUD_RUN_DEMO=true" in script
    assert "MEDIATION_STORE_MODE=memory" in script
    assert "DEV_MODE=false" in script
    assert "--min-instances 1" in script
    assert "--max-instances 1" in script
    assert "set-secrets" not in script
    assert "gcloud run services list" in script
    assert 'if [ "$#" -ne 0 ]' in script
    assert "docker build" not in script
    assert "docker push" not in script
    assert "cloud_run_candidate.py verify-deploy" in script
    assert '--image "${IMAGE_REFERENCE}"' in script
    assert "@sha256:" in script

    build_script = _text("deploy/build-payment-demo-candidate.sh")
    push_script = _text("deploy/push-payment-demo-candidate.sh")
    assert "--platform \"$PLATFORM\"" in build_script
    assert "--no-cache" in build_script
    assert "--load" in build_script
    assert "run_regression_manifest.py" in build_script
    assert "test_adk_web_browser.py" in build_script
    assert "validate_ap2_x402_release.py" in build_script
    assert "gcloud run deploy" not in build_script
    assert "docker push" in push_script
    assert "verify-local" in push_script
    assert "verify-deploy" in push_script
    assert "gcloud run deploy" not in push_script
    for executable in (
        "deploy/run-local.sh",
        "deploy/start.sh",
        "deploy/start-nginx.sh",
        "deploy/deploy-payment-demo-cloudrun.sh",
        "deploy/build-payment-demo-candidate.sh",
        "deploy/push-payment-demo-candidate.sh",
    ):
        assert (ROOT / executable).stat().st_mode & 0o111

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gcloud = fake_bin / "gcloud"
    gcloud.write_text(
        "#!/bin/sh\nprintf '%s\\n' payment-user-agent-demo\n",
        encoding="utf-8",
    )
    gcloud.chmod(0o755)
    docker_marker = tmp_path / "docker-called"
    docker = fake_bin / "docker"
    docker.write_text(
        f"#!/bin/sh\nprintf called > '{docker_marker}'\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DEV_MODE": "false",
        "EPHEMERAL_CLOUD_RUN_DEMO": "true",
    }
    completed = subprocess.run(
        ["bash", str(ROOT / "deploy/deploy-payment-demo-cloudrun.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 3
    assert "already exists" in completed.stdout
    assert not docker_marker.exists()

    image_digest = "sha256:" + "a" * 64
    image_reference = (
        "asia-northeast1-docker.pkg.dev/gen-lang-client-0585901015/"
        f"secure-mediation-agent/payment-user-agent-demo@{image_digest}"
    )
    python = fake_bin / "python3"
    python.write_text(
        f"#!/bin/sh\nprintf '%s\\n' '{image_reference}'\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    gcloud.write_text(
        """#!/bin/sh
case "$1 $2 $3" in
  "run services list") exit 0 ;;
  "artifacts docker images") printf '%s\n' "$EXPECTED_REGISTRY_DIGEST" ;;
  "run deploy payment-user-agent-demo") exit 0 ;;
  "run services describe") printf '%s\n' payment-user-agent-demo-revision ;;
  "run revisions describe") printf '%s\n' "$FAKE_REVISION_IMAGE" ;;
  *) printf 'unexpected fake gcloud call: %s\n' "$*" >&2; exit 90 ;;
esac
""",
        encoding="utf-8",
    )
    exact_environment = {
        **environment,
        "EXPECTED_REGISTRY_DIGEST": image_digest,
        "FAKE_REVISION_IMAGE": image_reference,
    }
    exact = subprocess.run(
        ["bash", str(ROOT / "deploy/deploy-payment-demo-cloudrun.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=exact_environment,
    )
    assert exact.returncode == 0, exact.stderr
    assert f"Deployed NEW ephemeral demo service at {image_reference}." in exact.stdout

    mismatched_revision_images = (
        image_reference.replace("secure-mediation-agent", "wrong-repository"),
        image_reference.removesuffix(image_digest) + "sha256:" + "b" * 64,
    )
    for mismatched_revision_image in mismatched_revision_images:
        mismatch = subprocess.run(
            ["bash", str(ROOT / "deploy/deploy-payment-demo-cloudrun.sh")],
            check=False,
            capture_output=True,
            text=True,
            env={
                **exact_environment,
                "FAKE_REVISION_IMAGE": mismatched_revision_image,
            },
        )
        assert mismatch.returncode == 4
        assert "ready revision image does not match" in mismatch.stdout


def test_ephemeral_mode_warns_and_provisions_only_local_state(
    workflow_fixture, monkeypatch
) -> None:
    startup = _text("deploy/start.sh")
    view = _text("secure_mediation_agent/workflow/views.py")
    assert "EPHEMERAL DEMO: state and keys may reset on restart" in startup
    assert "provision_ap2_demo_keys.py" in startup
    assert "EPHEMERAL DEMO: state and keys may reset on restart" in view
    assert all(word not in startup for word in ("cloudsql", "firestore", "filestore"))

    monkeypatch.setenv("APP_ENV", "ephemeral-demo")
    monkeypatch.setenv("DEV_MODE", "false")
    monkeypatch.setenv("EPHEMERAL_CLOUD_RUN_DEMO", "true")
    monkeypatch.setenv("MEDIATION_STORE_MODE", "memory")
    workflow_fixture["runtime"].ephemeral_cloud_run_demo = True
    workflow_fixture["runtime"].mediation_controller = create_production_controller(
        repository=workflow_fixture["repository"],
        keys=workflow_fixture["keys"],
    )
    with TestClient(workflow_fixture["app"]) as client:
        readiness = client.get("/ready")
    assert readiness.status_code == 200
    readiness_body = readiness.json()
    assert readiness_body["target"] == "ephemeral-cloud-run-demo"
    assert readiness_body["durability"] == "NOT PROVIDED"
    assert readiness_body["notice"] == "EPHEMERAL DEMO: state and keys may reset on restart"
    assert "dataDurableVolume" not in readiness_body["checks"]
    assert "evidenceDurableVolume" not in readiness_body["checks"]
    assert "durableVolumeMarker" not in readiness_body
    assert "evidenceDurableVolumeMarker" not in readiness_body
    assert readiness_body["checks"]["ephemeralDataPathWritable"] is True
    assert readiness_body["checks"]["ephemeralEvidencePathWritable"] is True
    assert readiness_body["mediationStore"] == {
        "mode": "memory",
        "durabilityProfile": "ephemeral-demo",
        "schemaVersion": None,
        "writable": True,
        "decryptable": True,
    }
    assert readiness_body["checks"]["mediationStoreMode"] is True
    assert readiness_body["checks"]["mediationStoreProfile"] is True
    assert readiness_body["checks"]["mediationStoreSchema"] is True
    assert readiness_body["checks"]["mediationStoreProbe"] is True

    with TestClient(auth.app) as client:
        deployment = client.get("/auth/deployment")
    assert deployment.status_code == 200
    assert deployment.json() == {
        "ephemeral": True,
        "target": "ephemeral-cloud-run-demo",
        "durability": "NOT PROVIDED",
        "notice": "EPHEMERAL DEMO: state and keys may reset on restart",
        "officialX402": "NOT RUN",
        "onChainSettlement": "NOT RUN",
    }
