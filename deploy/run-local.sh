#!/bin/bash
# Local Deployment Script - Fast startup for development
# Starts all components: trusted_agent_hub, external agents, and mediation agent
#
# Usage:
#   ./deploy/run-local.sh              # Start all services
#   ./deploy/run-local.sh --store-only # Rebuild and restart only the store (Docker)

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse arguments
STORE_ONLY=false
for arg in "$@"; do
    case $arg in
        --store-only)
            STORE_ONLY=true
            shift
            ;;
    esac
done

if [ "$STORE_ONLY" = true ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Rebuilding Store Only (Docker)       ${NC}"
    echo -e "${BLUE}========================================${NC}"
else
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Secure AI Agent Platform - Local Dev ${NC}"
    echo -e "${BLUE}========================================${NC}"
fi
echo ""

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# PID directory for tracking background processes
PID_DIR="/tmp/agent-platform-pids"
mkdir -p "$PID_DIR"

# Log directory
LOG_DIR="/tmp/agent-platform-logs"
mkdir -p "$LOG_DIR"

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if [ "$STORE_ONLY" = false ]; then
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python 3 is not installed${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Dependencies OK${NC}"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found. Creating from .env_sample...${NC}"
    if [ -f .env_sample ]; then
        cp .env_sample .env
        echo -e "${RED}Please edit .env with your actual API keys and run again${NC}"
        exit 1
    else
        echo -e "${RED}Error: .env_sample not found${NC}"
        exit 1
    fi
fi

# Load .env
set -a
source .env
set +a

echo -e "${GREEN}✓ Environment loaded${NC}"
echo ""

# ============================================
# 1. Start Trusted Agent Hub (Docker)
# ============================================
echo -e "${BLUE}[1/5] Starting Trusted Agent Hub (Docker)...${NC}"

IMAGE_NAME="secure-platform:latest"

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t "${IMAGE_NAME}" -f deploy/Dockerfile.cloudrun . --quiet

# Stop existing container if running
docker stop secure-platform 2>/dev/null || true
docker rm secure-platform 2>/dev/null || true

# Run container
docker run -d \
    --name secure-platform \
    -p 8080:8080 \
    --env-file .env \
    "${IMAGE_NAME}" > /dev/null

echo -e "${GREEN}✓ Trusted Agent Hub started on port 8080${NC}"

# ============================================
# Load sample data if available
# ============================================
SAMPLE_DB="$PROJECT_ROOT/deploy/sample-data/agent_store.db"
if [ -f "$SAMPLE_DB" ]; then
    echo -e "${YELLOW}Loading sample data...${NC}"
    sleep 2  # Wait for container to fully initialize
    docker cp "$SAMPLE_DB" secure-platform:/app/trusted_agent_hub/data/agent_store.db
    echo -e "${GREEN}✓ Sample data loaded${NC}"
fi
echo ""

# ============================================
# 2. Start A2A API Server (All External Agents)
# ============================================
if [ "$STORE_ONLY" = false ]; then
    echo -e "${BLUE}[2/4] Starting A2A API Server (Airline, Hotel, Car Rental Agents)...${NC}"

    cd "$PROJECT_ROOT"
    nohup "$PROJECT_ROOT/.venv/bin/adk" api_server --a2a --host 0.0.0.0 --port 8002 external-agents/trusted-agents/ > "$LOG_DIR/a2a_api_server.log" 2>&1 &
    A2A_PID=$!
    echo $A2A_PID > "$PID_DIR/a2a_api_server.pid"

    echo -e "${GREEN}✓ A2A API Server started on port 8002 (PID: $A2A_PID)${NC}"
    echo -e "${GREEN}  - Airline Agent: http://localhost:8002/a2a/airline_agent${NC}"
    echo -e "${GREEN}  - Hotel Agent: http://localhost:8002/a2a/hotel_agent${NC}"
    echo -e "${GREEN}  - Car Rental Agent: http://localhost:8002/a2a/car_rental_agent${NC}"
    echo ""

    # ============================================
    # 3. Start Secure Mediation Agent (ADK Web)
    # ============================================
    echo -e "${BLUE}[3/4] Starting Secure Mediation Agent (with ADK Web)...${NC}"

    cd secure-mediation-agent
    nohup "$PROJECT_ROOT/.venv/bin/adk" web . --port 8000 --reload > "$LOG_DIR/secure_mediation_agent.log" 2>&1 &
    MEDIATION_PID=$!
    echo $MEDIATION_PID > "$PID_DIR/secure_mediation_agent.pid"
    cd "$PROJECT_ROOT"

    echo -e "${GREEN}✓ Secure Mediation Agent started on port 8000 (PID: $MEDIATION_PID)${NC}"
    echo ""
fi

# ============================================
# Wait for services to start
# ============================================
echo -e "${YELLOW}Waiting for services to initialize...${NC}"
sleep 3

# ============================================
# Summary
# ============================================
echo ""
if [ "$STORE_ONLY" = true ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Store Rebuilt Successfully!          ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Access URL:${NC}"
    echo -e "  • Trusted Agent Hub:          ${GREEN}http://localhost:8080${NC}"
    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo -e "  docker logs -f secure-platform          # View hub logs"
    echo ""
else
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  All Services Started Successfully! ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Access URLs:${NC}"
    echo -e "  • Trusted Agent Hub:          ${GREEN}http://localhost:8080${NC}"
    echo -e "  • Secure Mediation Agent:     ${GREEN}http://localhost:8000${NC}"
    echo -e "  • ADK Web (Dev UI):           ${GREEN}http://localhost:8000/dev-ui${NC}"
    echo -e "  • A2A API Server:             ${GREEN}http://localhost:8002${NC}"
    echo -e "    - Airline Agent:            ${GREEN}http://localhost:8002/a2a/airline_agent${NC}"
    echo -e "    - Hotel Agent:              ${GREEN}http://localhost:8002/a2a/hotel_agent${NC}"
    echo -e "    - Car Rental Agent:         ${GREEN}http://localhost:8002/a2a/car_rental_agent${NC}"
    echo ""
    echo -e "${BLUE}Agent Card URLs (for submission):${NC}"
    echo -e "  ${YELLOW}http://host.docker.internal:8002/a2a/airline_agent/.well-known/agent.json${NC}"
    echo -e "  ${YELLOW}http://host.docker.internal:8002/a2a/hotel_agent/.well-known/agent.json${NC}"
    echo -e "  ${YELLOW}http://host.docker.internal:8002/a2a/car_rental_agent/.well-known/agent.json${NC}"
    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo -e "  ./deploy/stop-local.sh                  # Stop all services"
    echo -e "  ./deploy/run-local.sh --store-only      # Rebuild store only"
    echo -e "  docker logs -f secure-platform          # View hub logs"
    echo -e "  tail -f $LOG_DIR/a2a_api_server.log     # View A2A API server logs"
    echo -e "  tail -f $LOG_DIR/secure_mediation_agent.log   # View mediation logs"
    echo ""
    echo -e "${YELLOW}Note: It may take 10-20 seconds for all services to be fully ready.${NC}"
fi
echo ""
