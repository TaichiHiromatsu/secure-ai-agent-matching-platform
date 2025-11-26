#!/bin/bash
# Local Docker deployment script
# Builds and runs the Cloud Run container locally

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Local Docker Deployment ===${NC}"
echo ""

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found. Creating from .env_sample...${NC}"
    if [ -f .env_sample ]; then
        cp .env_sample .env
        echo "Please edit .env with your actual API keys"
        exit 1
    else
        echo "Error: .env_sample not found"
        exit 1
    fi
fi

# Build Docker image
IMAGE_NAME="secure-platform:latest"
echo -e "${GREEN}Building Docker image...${NC}"
docker build -t "${IMAGE_NAME}" -f deploy/Dockerfile.cloudrun .

echo ""
echo -e "${GREEN}Starting container...${NC}"

# Stop existing container if running
docker stop secure-platform 2>/dev/null || true
docker rm secure-platform 2>/dev/null || true

# Run container with environment variables from .env
docker run -d \
    --name secure-platform \
    -p 8080:8080 \
    --env-file .env \
    "${IMAGE_NAME}"

echo ""
echo -e "${GREEN}=== Container Started ===${NC}"
echo ""
echo "Access the application at: http://localhost:8080"
echo ""
echo "Useful commands:"
echo "  docker logs -f secure-platform          # View logs"
echo "  docker exec -it secure-platform bash    # Shell access"
echo "  docker stop secure-platform             # Stop container"
echo "  docker start secure-platform            # Start container"
echo ""
