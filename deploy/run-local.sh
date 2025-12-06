#!/bin/bash
# Local Deployment Script
# Usage: ./deploy/run-local.sh [OPTIONS]
#
# Options:
#   --no-build     Skip build, use existing image
#   --no-cache     Force full rebuild without Docker cache
#   --store-only   Only rebuild store (trusted_agent_hub/app) - fast rebuild

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Load .env for GCP settings
if [ ! -f .env ]; then
    echo "Error: .env file not found"
    exit 1
fi
source .env

# Parse arguments
NO_BUILD=false
NO_CACHE=false
STORE_ONLY=false

for arg in "$@"; do
    case $arg in
        --no-build)
            NO_BUILD=true
            ;;
        --no-cache)
            NO_CACHE=true
            ;;
        --store-only)
            STORE_ONLY=true
            ;;
    esac
done

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

# Build
if [ "$NO_BUILD" = false ]; then
    if [ "$STORE_ONLY" = true ]; then
        # Fast rebuild: only invalidate store layer by adding build timestamp
        echo "🏪 Store-only rebuild (fast)..."
        # Write timestamp to a file that gets copied, invalidating that layer and all after it
        echo "BUILD_TIME=$(date +%s)" > trusted_agent_hub/app/.build_timestamp
        docker build -t "${IMAGE_NAME}" .
        rm -f trusted_agent_hub/app/.build_timestamp
    elif [ "$NO_CACHE" = true ]; then
        echo "🔄 Full rebuild (no cache)..."
        docker build --no-cache -t "${IMAGE_NAME}" .
    else
        echo "📦 Standard build..."
        docker build -t "${IMAGE_NAME}" .
    fi
    echo "Built: ${IMAGE_NAME}"
fi

# Run
docker run -d \
    --name secure-platform \
    -p 8080:8080 \
    --env-file .env \
    "${IMAGE_NAME}"

echo ""
echo "✅ Started: http://localhost:8080"
echo "📋 Logs: docker logs -f secure-platform"
