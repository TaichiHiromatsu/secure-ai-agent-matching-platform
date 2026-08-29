#!/bin/bash
# Cloud Run deployment script
# Usage: ./deploy/deploy-cloudrun.sh

set -e

if [ "${DEV_MODE:-false}" = "true" ]; then
    echo "Refusing Cloud Run deployment: DEV_MODE=true is forbidden."
    exit 2
fi

echo "Refusing integrated paid deployment: current Cloud Run filesystem is ephemeral."
echo "This release is approved only for an explicit durable single-host/single-container target."
echo "Official x402, wallet/facilitator verification, and on-chain settlement remain NOT RUN."
exit 2

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Load .env
if [ -f .env ]; then
    source .env
fi

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:?GCP_PROJECT_ID is required}"
REGION="${GCP_REGION:-asia-northeast1}"
SERVICE_NAME="${SERVICE_NAME:-secure-mediation-a2a-platform}"
IMAGE_BASE="${REGION}-docker.pkg.dev/${PROJECT_ID}/secure-mediation-agent/${SERVICE_NAME}"
IMAGE_TAG="v$(date +%Y%m%d-%H%M%S)"
IMAGE_NAME="${IMAGE_BASE}:${IMAGE_TAG}"

echo "Deploying to Cloud Run"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Image: ${IMAGE_NAME}"

# Build and push
docker build -t "${IMAGE_NAME}" .
docker push "${IMAGE_NAME}"

# Deploy
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}" \
    --platform managed \
    --region "${REGION}" \
    --port 8080 \
    --memory 2Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 1 \
    --timeout 3600s \
    --concurrency 80 \
    --cpu-boost \
    --allow-unauthenticated

echo "Deployed: $(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')"
