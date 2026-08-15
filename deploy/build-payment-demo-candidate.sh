#!/bin/bash
# Build and validate a local linux/amd64 candidate. This script never publishes or deploys.

set -euo pipefail

if [ "$#" -ne 0 ]; then
    echo "Refusing candidate build: this release accepts no override flags."
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

readonly IMAGE_TAG="enterprise-a2a-pf:payment-user-agent-cloudrun-amd64"
readonly PLATFORM="linux/amd64"
readonly CANDIDATE_ARTIFACT="artifacts/cloud-run-candidate.json"
readonly REQUIRED_JSON=(
    "deploy/auth/firebase-config.json"
    "secure_mediation_agent/spec_manifest.json"
    "tests/regression/suite_manifest.json"
    "tests/release/release_manifest.json"
    "docs/ap2_x402_conformance_report.json"
    "trusted_agent_store/evaluation-runner/prompts/aisi/manifest.sample.json"
    "trusted_agent_store/evaluation-runner/prompts/aisi/questions/privacy.data_retention.json"
    "trusted_agent_store/evaluation-runner/prompts/aisi/questions/safety.general.json"
    "trusted_agent_store/evaluation-runner/schemas/fairness_probe.schema.json"
    "trusted_agent_store/evaluation-runner/schemas/policy_score.schema.json"
    "trusted_agent_store/evaluation-runner/schemas/response_sample.schema.json"
)

for required_file in "${REQUIRED_JSON[@]}"; do
    if [ ! -f "$required_file" ]; then
        echo "Refusing candidate build: required file is missing: ${required_file}"
        exit 2
    fi
    if git check-ignore -q -- "$required_file"; then
        echo "Refusing candidate build: required file is ignored: ${required_file}"
        exit 2
    fi
    if ! git ls-files --cached --others --exclude-standard -- "$required_file" | grep -Fxq "$required_file"; then
        echo "Refusing candidate build: required file is absent from clean context: ${required_file}"
        exit 2
    fi
done

python3 scripts/cloud_run_candidate.py source-info

build_context="$(mktemp -d)"
cleanup() {
    rm -rf -- "$build_context"
}
trap cleanup EXIT

# Materialize only Git-visible files. Ignored local files cannot make this build pass.
git ls-files --cached --others --exclude-standard -z \
    | tar --null -T - -cf - \
    | tar -xf - -C "$build_context"

docker buildx build \
    --platform "$PLATFORM" \
    --no-cache \
    --provenance=false \
    --load \
    --tag "$IMAGE_TAG" \
    "$build_context"

image_id="$(docker image inspect --format '{{.Id}}' "$IMAGE_TAG")"
image_platform="$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$IMAGE_TAG")"
if [ "$image_platform" != "$PLATFORM" ]; then
    echo "Refusing candidate validation: expected ${PLATFORM}, got ${image_platform}."
    exit 2
fi
if [[ ! "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "Refusing candidate validation: local image ID is not an exact digest."
    exit 2
fi

mkdir -p artifacts
rm -f -- "$CANDIDATE_ARTIFACT"

# Every suite below executes code embedded in the exact image; no source tree is mounted.
docker run --rm --platform "$PLATFORM" \
    --entrypoint /bin/sh \
    --env "RELEASE_IMAGE_DIGEST=${image_id}" \
    --volume "$PWD/artifacts:/evidence" \
    "$IMAGE_TAG" \
    -c 'cd /app && /app/.venv/bin/python scripts/run_regression_manifest.py --output /evidence/regression-result.json'

docker run --rm --platform "$PLATFORM" \
    --entrypoint /bin/sh \
    --env "RELEASE_IMAGE_DIGEST=${image_id}" \
    --env "BROWSER_EVIDENCE_OUTPUT=/evidence/browser-evidence.json" \
    --volume "$PWD/artifacts:/evidence" \
    "$IMAGE_TAG" \
    -c 'cd /app && /app/.venv/bin/python -m pytest -p no:cacheprovider -q tests/browser/test_adk_web_browser.py'

python3 scripts/cloud_run_candidate.py update-conformance \
    --image-id "$image_id" \
    --platform "$PLATFORM" \
    --regression artifacts/regression-result.json \
    --browser artifacts/browser-evidence.json

docker run --rm --platform "$PLATFORM" \
    --entrypoint /bin/sh \
    --volume "$PWD/artifacts:/evidence" \
    --volume "$PWD/docs:/release-docs:ro" \
    "$IMAGE_TAG" \
    -c "/app/.venv/bin/python /app/scripts/validate_ap2_x402_release.py \
        --expected-image-digest '$image_id' \
        --regression-result /evidence/regression-result.json \
        --browser-evidence /evidence/browser-evidence.json \
        --conformance /release-docs/ap2_x402_conformance_report.json \
        --output /evidence/ap2-x402-release-validation.json"

python3 scripts/cloud_run_candidate.py write-local \
    --image-id "$image_id" \
    --artifact "$CANDIDATE_ARTIFACT"
python3 scripts/cloud_run_candidate.py verify-local \
    --image-id "$image_id" \
    --artifact "$CANDIDATE_ARTIFACT"

echo "LOCAL_AMD64_CANDIDATE_PASS image_id=${image_id}"
echo "Candidate artifact: ${CANDIDATE_ARTIFACT}"
echo "No registry push and no Cloud Run deployment were performed."
