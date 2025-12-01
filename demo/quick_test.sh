#!/bin/bash
# Quick test script - Verify all agents are responding

# Move to project root
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 Testing Agent Connectivity...${NC}"
echo ""

test_agent() {
    local url=$1
    local name=$2

    echo -n "Testing $name... "

    if response=$(curl -s -f "$url/.well-known/agent-card.json" 2>&1); then
        echo -e "${GREEN}✅ OK${NC}"
        echo "  Agent Name: $(echo $response | jq -r '.name // "N/A"')"
        echo "  Description: $(echo $response | jq -r '.description // "N/A"')"
        echo ""
        return 0
    else
        echo -e "${RED}❌ Failed${NC}"
        echo "  Error: Cannot reach $url"
        echo ""
        return 1
    fi
}

# Test external agents
echo -e "${BLUE}External Agents:${NC}"
test_agent "http://localhost:8002" "Airline Agent (Port 8002)"
test_agent "http://localhost:8003" "Hotel Agent (Port 8003)"
test_agent "http://localhost:8004" "Car Rental Agent (Port 8004)"

# Test mediation agent
echo -e "${BLUE}Mediation Agent:${NC}"
if curl -s http://localhost:8000 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Mediation Agent Web UI is running on http://localhost:8000${NC}"
else
    echo -e "${RED}❌ Mediation Agent Web UI is not running${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
