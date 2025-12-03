#!/bin/bash
# Cloud Run deployment script for secure-ai-agent-matching-platform
# This script builds and deploys a single container with all services

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:?GCP_PROJECT_ID is required}"
REGION="${GCP_REGION:-asia-northeast1}"
SERVICE_NAME="${SERVICE_NAME:-secure-mediation-a2a-platform}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Cloud Run Deployment ===${NC}"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo ""

# Check if required tools are installed
command -v gcloud >/dev/null 2>&1 || { echo -e "${RED}gcloud CLI is required but not installed.${NC}" >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo -e "${RED}Docker is required but not installed.${NC}" >&2; exit 1; }

# Check if GOOGLE_API_KEY is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo -e "${YELLOW}Warning: GOOGLE_API_KEY is not set. You'll need to set it in Cloud Run.${NC}"
fi

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo -e "${GREEN}Building Docker image...${NC}"
docker build -t "${IMAGE_NAME}:latest" -f deploy/Dockerfile .

echo -e "${GREEN}Pushing to Google Container Registry...${NC}"
docker push "${IMAGE_NAME}:latest"

echo -e "${GREEN}Deploying to Cloud Run...${NC}"
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}:latest" \
    --platform managed \
    --region "${REGION}" \
    --port 8080 \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 1 \
    --timeout 300s \
    --concurrency 80 \
    --cpu-boost \
    --set-env-vars "GOOGLE_API_KEY=${GOOGLE_API_KEY},GOOGLE_GENAI_USE_VERTEXAI=0" \
    --allow-unauthenticated

echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo ""
echo "Service URL:"
gcloud run services describe "${SERVICE_NAME}" --platform managed --region "${REGION}" --format 'value(status.url)'
echo ""
echo -e "${YELLOW}Recommended: Set up a Cloud Scheduler job to ping /health every 5-10 minutes to keep the instance warm and avoid cold starts.${NC}"
