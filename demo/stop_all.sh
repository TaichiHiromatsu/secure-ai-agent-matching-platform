#!/bin/bash
# Stop all running agents

# Move to project root
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🛑 Stopping all agents...${NC}"
echo ""

# Kill by PIDs if available
if [ -f .airline.pid ]; then
    kill $(cat .airline.pid) 2>/dev/null && echo -e "${GREEN}✅ Stopped Airline Agent${NC}"
    rm .airline.pid
fi

if [ -f .hotel.pid ]; then
    kill $(cat .hotel.pid) 2>/dev/null && echo -e "${GREEN}✅ Stopped Hotel Agent${NC}"
    rm .hotel.pid
fi

if [ -f .car.pid ]; then
    kill $(cat .car.pid) 2>/dev/null && echo -e "${GREEN}✅ Stopped Car Rental Agent${NC}"
    rm .car.pid
fi

if [ -f .mediation.pid ]; then
    kill $(cat .mediation.pid) 2>/dev/null && echo -e "${GREEN}✅ Stopped Mediation Agent${NC}"
    rm .mediation.pid
fi

# Kill by port as fallback
for port in 8002 8003 8004 8000; do
    if lsof -ti:$port >/dev/null 2>&1; then
        lsof -ti:$port | xargs kill -9 2>/dev/null
        echo -e "${GREEN}✅ Killed process on port $port${NC}"
    fi
done

echo ""
echo -e "${GREEN}✅ All agents stopped${NC}"
