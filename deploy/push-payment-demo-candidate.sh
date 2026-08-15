#!/bin/bash
# Publish an already validated exact candidate. This script never deploys Cloud Run.

set -euo pipefail

if [ "$#" -ne 0 ]; then
    echo "Refusing candidate push: this release accepts no override flags."
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

readonly PROJECT_ID="gen-lang-client-0585901015"
readonly REGION="asia-northeast1"
readonly PLATFORM="linux/amd64"
readonly LOCAL_IMAGE="enterprise-a2a-pf:payment-user-agent-cloudrun-amd64"
readonly REGISTRY_REPOSITORY="${REGION}-docker.pkg.dev/${PROJECT_ID}/secure-mediation-agent/payment-user-agent-demo"
readonly CANDIDATE_ARTIFACT="artifacts/cloud-run-candidate.json"

image_id="$(docker image inspect --format '{{.Id}}' "$LOCAL_IMAGE")"
image_platform="$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$LOCAL_IMAGE")"
if [ "$image_platform" != "$PLATFORM" ]; then
    echo "Refusing candidate push: expected ${PLATFORM}, got ${image_platform}."
    exit 2
fi
python3 scripts/cloud_run_candidate.py verify-local \
    --image-id "$image_id" \
    --artifact "$CANDIDATE_ARTIFACT"

source_digest="$(jq -r '.source.worktreeDigest' "$CANDIDATE_ARTIFACT")"
if [[ ! "$source_digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "Refusing candidate push: source digest is invalid."
    exit 2
fi
release_tag="candidate-${source_digest#sha256:}"
tagged_image="${REGISTRY_REPOSITORY}:${release_tag}"

# Read-only target checks precede the authorized image publication.
gcloud artifacts repositories describe secure-mediation-agent \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --format='value(name)' >/dev/null
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

docker tag "$LOCAL_IMAGE" "$tagged_image"
docker push "$tagged_image"

registry_image="$(
    docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$tagged_image" \
        | grep -E "^${REGISTRY_REPOSITORY}@sha256:[0-9a-f]{64}$" \
        | head -n 1
)"
if [[ ! "$registry_image" =~ ^${REGISTRY_REPOSITORY}@sha256:[0-9a-f]{64}$ ]]; then
    echo "Candidate was pushed, but its immutable registry digest could not be resolved."
    exit 2
fi

remote_image="$(docker buildx imagetools inspect "$registry_image" --format '{{json .Image}}')"
remote_platform="$(printf '%s' "$remote_image" | jq -r '.os + "/" + .architecture')"
if [ "$remote_platform" != "$PLATFORM" ]; then
    echo "Candidate was pushed, but registry platform is ${remote_platform}, not ${PLATFORM}."
    exit 2
fi

# Add the registry binding, then rerun the embedded validator so its conformance
# digest covers the final registry reference as well as the local image ID.
python3 scripts/cloud_run_candidate.py update-conformance \
    --image-id "$image_id" \
    --platform "$PLATFORM" \
    --regression artifacts/regression-result.json \
    --browser artifacts/browser-evidence.json \
    --registry-image "$registry_image"

docker run --rm --platform "$PLATFORM" \
    --entrypoint /bin/sh \
    --volume "$PWD/artifacts:/evidence" \
    --volume "$PWD/docs:/release-docs:ro" \
    "$LOCAL_IMAGE" \
    -c "/app/.venv/bin/python /app/scripts/validate_ap2_x402_release.py \
        --expected-image-digest '$image_id' \
        --regression-result /evidence/regression-result.json \
        --browser-evidence /evidence/browser-evidence.json \
        --conformance /release-docs/ap2_x402_conformance_report.json \
        --output /evidence/ap2-x402-release-validation.json"

python3 scripts/cloud_run_candidate.py write-pushed \
    --image-id "$image_id" \
    --registry-image "$registry_image" \
    --artifact "$CANDIDATE_ARTIFACT"
verified_image="$(python3 scripts/cloud_run_candidate.py verify-deploy --artifact "$CANDIDATE_ARTIFACT")"
if [ "$verified_image" != "$registry_image" ]; then
    echo "Candidate was pushed, but the final immutable binding check failed."
    exit 2
fi

echo "REGISTRY_CANDIDATE_PASS image_id=${image_id} registry_image=${registry_image}"
echo "No Cloud Run service was created or updated."
