#!/bin/bash
# Local Deployment Script
# Usage: ./deploy/run-local.sh [--no-build]

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Load .env for GCP settings
if [ ! -f .env ]; then
    echo "Error: .env file not found"
    exit 1
fi
source .env

# Image settings (same as Cloud Run)
PROJECT_ID="${GCP_PROJECT_ID:-gen-lang-client-0585901015}"
REGION="${GCP_REGION:-asia-northeast1}"
SERVICE_NAME="${SERVICE_NAME:-secure-mediation-a2a-platform}"
IMAGE_BASE="${REGION}-docker.pkg.dev/${PROJECT_ID}/secure-mediation-agent/${SERVICE_NAME}"
IMAGE_TAG="v$(date +%Y%m%d-%H%M%S)"
IMAGE_NAME="${IMAGE_BASE}:${IMAGE_TAG}"

# Cleanup
docker stop secure-platform 2>/dev/null || true
docker rm secure-platform 2>/dev/null || true

# Build (unless --no-build)
if [ "$1" != "--no-build" ]; then
    docker build -t "${IMAGE_NAME}" .
    echo "Built: ${IMAGE_NAME}"
fi

# Run
docker run -d \
    --name secure-platform \
    -p 8080:8080 \
    --env-file .env \
    "${IMAGE_NAME}"

echo "Started: http://localhost:8080"
echo "Logs: docker logs -f secure-platform"
