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
echo -e "${YELLOW}[1/5] Stopping Trusted Agent Hub (Docker)...${NC}"

if docker ps -q -f name=secure-platform &> /dev/null; then
    docker stop secure-platform 2>/dev/null || true
    docker rm secure-platform 2>/dev/null || true
    echo -e "${GREEN}✓ Trusted Agent Hub stopped${NC}"
else
    echo -e "${YELLOW}  Already stopped${NC}"
fi
echo ""

# ============================================
# 2. Stop Airline Agent
# ============================================
echo -e "${YELLOW}[2/5] Stopping Airline Agent...${NC}"

if [ -f "$PID_DIR/airline_agent.pid" ]; then
    AIRLINE_PID=$(cat "$PID_DIR/airline_agent.pid")
    if ps -p $AIRLINE_PID > /dev/null 2>&1; then
        kill $AIRLINE_PID 2>/dev/null || true
        echo -e "${GREEN}✓ Airline Agent stopped (PID: $AIRLINE_PID)${NC}"
    else
        echo -e "${YELLOW}  Process not found (PID: $AIRLINE_PID)${NC}"
    fi
    rm -f "$PID_DIR/airline_agent.pid"
else
    echo -e "${YELLOW}  PID file not found${NC}"
fi
echo ""

# ============================================
# 3. Stop Hotel Agent
# ============================================
echo -e "${YELLOW}[3/5] Stopping Hotel Agent...${NC}"

if [ -f "$PID_DIR/hotel_agent.pid" ]; then
    HOTEL_PID=$(cat "$PID_DIR/hotel_agent.pid")
    if ps -p $HOTEL_PID > /dev/null 2>&1; then
        kill $HOTEL_PID 2>/dev/null || true
        echo -e "${GREEN}✓ Hotel Agent stopped (PID: $HOTEL_PID)${NC}"
    else
        echo -e "${YELLOW}  Process not found (PID: $HOTEL_PID)${NC}"
    fi
    rm -f "$PID_DIR/hotel_agent.pid"
else
    echo -e "${YELLOW}  PID file not found${NC}"
fi
echo ""

# ============================================
# 4. Stop Car Rental Agent
# ============================================
echo -e "${YELLOW}[4/5] Stopping Car Rental Agent...${NC}"

if [ -f "$PID_DIR/car_rental_agent.pid" ]; then
    CAR_PID=$(cat "$PID_DIR/car_rental_agent.pid")
    if ps -p $CAR_PID > /dev/null 2>&1; then
        kill $CAR_PID 2>/dev/null || true
        echo -e "${GREEN}✓ Car Rental Agent stopped (PID: $CAR_PID)${NC}"
    else
        echo -e "${YELLOW}  Process not found (PID: $CAR_PID)${NC}"
    fi
    rm -f "$PID_DIR/car_rental_agent.pid"
else
    echo -e "${YELLOW}  PID file not found${NC}"
fi
echo ""

# ============================================
# 5. Stop Secure Mediation Agent
# ============================================
echo -e "${YELLOW}[5/5] Stopping Secure Mediation Agent...${NC}"

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

for PORT in 8000 8002 8003 8004; do
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
