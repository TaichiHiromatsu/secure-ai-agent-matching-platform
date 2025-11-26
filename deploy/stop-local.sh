#!/bin/bash
# Stop all local services started by run-local.sh

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Stopping All Services ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# PID directory
PID_DIR="/tmp/agent-platform-pids"

# ============================================
# 1. Stop Docker Container
# ============================================
echo -e "${YELLOW}[1/3] Stopping Trusted Agent Hub (Docker)...${NC}"

if docker ps -q -f name=secure-platform &> /dev/null; then
    docker stop secure-platform 2>/dev/null || true
    docker rm secure-platform 2>/dev/null || true
    echo -e "${GREEN}✓ Trusted Agent Hub stopped${NC}"
else
    echo -e "${YELLOW}  Already stopped${NC}"
fi
echo ""

# ============================================
# 2. Stop A2A API Server
# ============================================
echo -e "${YELLOW}[2/3] Stopping A2A API Server...${NC}"

if [ -f "$PID_DIR/a2a_api_server.pid" ]; then
    A2A_PID=$(cat "$PID_DIR/a2a_api_server.pid")
    if ps -p $A2A_PID > /dev/null 2>&1; then
        kill $A2A_PID 2>/dev/null || true
        echo -e "${GREEN}✓ A2A API Server stopped (PID: $A2A_PID)${NC}"
    else
        echo -e "${YELLOW}  Process not found (PID: $A2A_PID)${NC}"
    fi
    rm -f "$PID_DIR/a2a_api_server.pid"
else
    echo -e "${YELLOW}  PID file not found${NC}"
fi
echo ""

# ============================================
# 3. Stop Secure Mediation Agent
# ============================================
echo -e "${YELLOW}[3/3] Stopping Secure Mediation Agent...${NC}"

if [ -f "$PID_DIR/secure_mediation_agent.pid" ]; then
    MEDIATION_PID=$(cat "$PID_DIR/secure_mediation_agent.pid")
    if ps -p $MEDIATION_PID > /dev/null 2>&1; then
        kill $MEDIATION_PID 2>/dev/null || true
        echo -e "${GREEN}✓ Secure Mediation Agent stopped (PID: $MEDIATION_PID)${NC}"
    else
        echo -e "${YELLOW}  Process not found (PID: $MEDIATION_PID)${NC}"
    fi
    rm -f "$PID_DIR/secure_mediation_agent.pid"
else
    echo -e "${YELLOW}  PID file not found${NC}"
fi
echo ""

# ============================================
# Cleanup: Kill any remaining processes on ports
# ============================================
echo -e "${YELLOW}Cleaning up any remaining processes on ports...${NC}"

for PORT in 8000 8002; do
    PID=$(lsof -ti:$PORT 2>/dev/null || true)
    if [ ! -z "$PID" ]; then
        kill $PID 2>/dev/null || true
        echo -e "${GREEN}  Killed process on port $PORT (PID: $PID)${NC}"
    fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  All Services Stopped ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
