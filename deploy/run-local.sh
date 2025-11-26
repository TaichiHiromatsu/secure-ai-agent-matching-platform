#!/bin/bash
# Local Deployment Script - Fast startup for development
# Starts all components: trusted_agent_hub, external agents, and mediation agent

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Secure AI Agent Platform - Local Dev ${NC}"
echo -e "${BLUE}========================================${NC}"
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

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
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
echo ""

# ============================================
# 2. Start Airline Agent (Python)
# ============================================
echo -e "${BLUE}[2/5] Starting Airline Agent...${NC}"

cd external-agents/trusted-agents/airline_agent
nohup python3 agent.py > "$LOG_DIR/airline_agent.log" 2>&1 &
AIRLINE_PID=$!
echo $AIRLINE_PID > "$PID_DIR/airline_agent.pid"
cd "$PROJECT_ROOT"

echo -e "${GREEN}✓ Airline Agent started on port 8002 (PID: $AIRLINE_PID)${NC}"
echo ""

# ============================================
# 3. Start Hotel Agent (Python)
# ============================================
echo -e "${BLUE}[3/5] Starting Hotel Agent...${NC}"

cd external-agents/trusted-agents/hotel_agent
nohup python3 agent.py > "$LOG_DIR/hotel_agent.log" 2>&1 &
HOTEL_PID=$!
echo $HOTEL_PID > "$PID_DIR/hotel_agent.pid"
cd "$PROJECT_ROOT"

echo -e "${GREEN}✓ Hotel Agent started on port 8003 (PID: $HOTEL_PID)${NC}"
echo ""

# ============================================
# 4. Start Car Rental Agent (Python)
# ============================================
echo -e "${BLUE}[4/5] Starting Car Rental Agent...${NC}"

cd external-agents/trusted-agents/car_rental_agent
nohup python3 agent.py > "$LOG_DIR/car_rental_agent.log" 2>&1 &
CAR_PID=$!
echo $CAR_PID > "$PID_DIR/car_rental_agent.pid"
cd "$PROJECT_ROOT"

echo -e "${GREEN}✓ Car Rental Agent started on port 8004 (PID: $CAR_PID)${NC}"
echo ""

# ============================================
# 5. Start Secure Mediation Agent (ADK Web)
# ============================================
echo -e "${BLUE}[5/5] Starting Secure Mediation Agent (with ADK Web)...${NC}"

cd secure-mediation-agent
nohup python3 -m adk web . --port 8000 --reload > "$LOG_DIR/secure_mediation_agent.log" 2>&1 &
MEDIATION_PID=$!
echo $MEDIATION_PID > "$PID_DIR/secure_mediation_agent.pid"
cd "$PROJECT_ROOT"

echo -e "${GREEN}✓ Secure Mediation Agent started on port 8000 (PID: $MEDIATION_PID)${NC}"
echo ""

# ============================================
# Wait for services to start
# ============================================
echo -e "${YELLOW}Waiting for services to initialize...${NC}"
sleep 5

# ============================================
# Summary
# ============================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  All Services Started Successfully! ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Access URLs:${NC}"
echo -e "  • Trusted Agent Hub:          ${GREEN}http://localhost:8080${NC}"
echo -e "  • Secure Mediation Agent:     ${GREEN}http://localhost:8000${NC}"
echo -e "  • ADK Web (Dev UI):           ${GREEN}http://localhost:8000/dev-ui${NC}"
echo -e "  • Airline Agent:              ${GREEN}http://localhost:8002${NC}"
echo -e "  • Hotel Agent:                ${GREEN}http://localhost:8003${NC}"
echo -e "  • Car Rental Agent:           ${GREEN}http://localhost:8004${NC}"
echo ""
echo -e "${BLUE}Agent Card URL (for submission):${NC}"
echo -e "  ${YELLOW}http://host.docker.internal:8002/a2a/airline_agent/.well-known/agent.json${NC}"
echo ""
echo -e "${BLUE}Useful Commands:${NC}"
echo -e "  ./deploy/stop-local.sh                  # Stop all services"
echo -e "  docker logs -f secure-platform          # View hub logs"
echo -e "  tail -f $LOG_DIR/airline_agent.log      # View airline logs"
echo -e "  tail -f $LOG_DIR/secure_mediation_agent.log   # View mediation logs"
echo ""
echo -e "${YELLOW}Note: It may take 10-20 seconds for all services to be fully ready.${NC}"
echo ""
