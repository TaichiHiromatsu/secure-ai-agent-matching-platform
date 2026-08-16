#!/bin/bash
# Deploy a frozen, deliberately ephemeral Cloud Run candidate to a NEW service.
# Build and publication are separate, mandatory release-gate steps.

set -euo pipefail

if [ "$#" -ne 0 ]; then
    echo "Refusing Cloud Run deployment: this release accepts no override flags."
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

readonly PROJECT_ID="gen-lang-client-0585901015"
readonly REGION="asia-northeast1"
readonly SERVICE_NAME="payment-user-agent-demo"
readonly DEPLOY_ENV_VARS="EPHEMERAL_CLOUD_RUN_DEMO=true,MEDIATION_STORE_MODE=memory,APP_ENV=ephemeral-demo,DEV_MODE=false"
readonly CANDIDATE_ARTIFACT="artifacts/cloud-run-candidate.json"

if [ "${PROJECT_ID}/${REGION}/${SERVICE_NAME}" != \
     "gen-lang-client-0585901015/asia-northeast1/payment-user-agent-demo" ]; then
    echo "Refusing Cloud Run deployment: fixed target validation failed."
    exit 2
fi
if [ "${DEV_MODE:-false}" != "false" ]; then
    echo "Refusing Cloud Run deployment: DEV_MODE must be false."
    exit 2
fi
if [ "${EPHEMERAL_CLOUD_RUN_DEMO:-true}" != "true" ]; then
    echo "Refusing Cloud Run deployment: EPHEMERAL_CLOUD_RUN_DEMO must be true."
    exit 2
fi
if [ "$DEPLOY_ENV_VARS" != \
     "EPHEMERAL_CLOUD_RUN_DEMO=true,MEDIATION_STORE_MODE=memory,APP_ENV=ephemeral-demo,DEV_MODE=false" ]; then
    echo "Refusing Cloud Run deployment: deployment environment validation failed."
    exit 2
fi

# Read-only, fail-closed NEW-only preflight. `services list` returns an empty
# result for an absent exact name and a non-zero status for lookup failures.
existing_service="$(
    gcloud run services list \
        --project "$PROJECT_ID" \
        --region "$REGION" \
        --platform managed \
        --filter="metadata.name=${SERVICE_NAME}" \
        --format="value(metadata.name)"
)"
if [ -n "$existing_service" ]; then
    echo "Refusing Cloud Run deployment: service ${SERVICE_NAME} already exists."
    exit 3
fi

IMAGE_REFERENCE="$(
    python3 scripts/cloud_run_candidate.py verify-deploy \
        --artifact "$CANDIDATE_ARTIFACT"
)"
if [[ ! "$IMAGE_REFERENCE" =~ ^asia-northeast1-docker\.pkg\.dev/gen-lang-client-0585901015/secure-mediation-agent/payment-user-agent-demo@sha256:[0-9a-f]{64}$ ]]; then
    echo "Refusing Cloud Run deployment: candidate is not an immutable fixed image."
    exit 2
fi

expected_registry_digest="${IMAGE_REFERENCE##*@}"
observed_registry_digest="$(
    gcloud artifacts docker images describe "$IMAGE_REFERENCE" \
        --project "$PROJECT_ID" \
        --format='value(image_summary.digest)'
)"
if [ "$observed_registry_digest" != "$expected_registry_digest" ]; then
    echo "Refusing Cloud Run deployment: registry digest verification failed."
    exit 2
fi

echo "EPHEMERAL DEMO: state and keys may reset on restart"
echo "Target NEW service: ${SERVICE_NAME} (${PROJECT_ID}/${REGION})"
echo "Official x402 and on-chain settlement: NOT RUN"

gcloud run deploy "${SERVICE_NAME}" \
    --project "${PROJECT_ID}" \
    --image "${IMAGE_REFERENCE}" \
    --platform managed \
    --region "${REGION}" \
    --port 8080 \
    --memory 2Gi \
    --cpu 1 \
    --min-instances 1 \
    --max-instances 1 \
    --concurrency 1 \
    --timeout 3600s \
    --cpu-boost \
    --allow-unauthenticated \
    --set-env-vars "$DEPLOY_ENV_VARS"

revision="$(
    gcloud run services describe "$SERVICE_NAME" \
        --project "$PROJECT_ID" \
        --region "$REGION" \
        --platform managed \
        --format='value(status.latestReadyRevisionName)'
)"
revision_image="$(
    gcloud run revisions describe "$revision" \
        --project "$PROJECT_ID" \
        --region "$REGION" \
        --platform managed \
        --format='value(status.imageDigest)'
)"
if [ "$revision_image" != "$IMAGE_REFERENCE" ]; then
    echo "Deployment completed, but the ready revision image does not match the candidate."
    exit 4
fi

echo "Deployed NEW ephemeral demo service at ${IMAGE_REFERENCE}."
echo "Never claim restart durability."
