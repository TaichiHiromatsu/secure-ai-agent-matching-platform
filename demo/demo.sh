#!/bin/bash
# Demo script - Start everything with one command

set -e

# Move to project root
cd "$(dirname "$0")/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f "secure-mediation-agent/.env" ]; then
    export $(cat secure-mediation-agent/.env | grep -v '^#' | xargs)
else
    echo -e "${RED}❌ Error: .env file not found in secure-mediation-agent/.env${NC}"
    echo "Please create .env file with GOOGLE_API_KEY"
    exit 1
fi

# Check if GOOGLE_API_KEY is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo -e "${RED}❌ Error: GOOGLE_API_KEY not set in .env file${NC}"
    exit 1
fi

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🚀 Secure AI Agent Platform Demo${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Clean up any previous processes
echo -e "${BLUE}🧹 Cleaning up previous processes...${NC}"

# Kill processes from PID files if they exist
for pidfile in .airline.pid .hotel.pid .car.pid .mediation.pid; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        kill -9 "$pid" 2>/dev/null || true
        rm -f "$pidfile"
    fi
done

# Kill any processes on demo ports
for port in 8002 8003 8004 8000; do
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
done

sleep 2
echo -e "${GREEN}✅ Cleanup complete${NC}"
echo ""

echo ""
echo -e "${BLUE}🏗️  Starting External Agents...${NC}"
echo ""

# Start Airline Agent (port 8002) as A2A server
echo -e "${GREEN}✈️  Starting Airline Agent (A2A) on port 8002...${NC}"
uv run adk api_server --a2a --port 8002 external-agents/trusted-agents/ > /tmp/airline-agent.log 2>&1 &
AIRLINE_PID=$!
sleep 3

# Start Hotel Agent (port 8003) as A2A server
echo -e "${GREEN}🏨 Starting Hotel Agent (A2A) on port 8003...${NC}"
uv run adk api_server --a2a --port 8003 external-agents/trusted-agents/ > /tmp/hotel-agent.log 2>&1 &
HOTEL_PID=$!
sleep 3

# Start Car Rental Agent (port 8004) as A2A server
echo -e "${GREEN}🚗 Starting Car Rental Agent (A2A) on port 8004...${NC}"
uv run adk api_server --a2a --port 8004 external-agents/trusted-agents/ > /tmp/car-agent.log 2>&1 &
CAR_PID=$!
sleep 3

# Verify agents are running
# echo ""
# echo -e "${BLUE}🔍 Verifying agents...${NC}"
# AGENTS_OK=true
#
# check_agent() {
#     local url=$1
#     local name=$2
#     if curl -s -f "$url/.well-known/agent.json" > /dev/null 2>&1; then
#         echo -e "${GREEN}✅ $name is running${NC}"
#         return 0
#     else
#         echo -e "${RED}❌ $name failed to start${NC}"
#         return 1
#     fi
# }
#
# check_agent "http://localhost:8002" "Airline Agent" || AGENTS_OK=false
# check_agent "http://localhost:8003" "Hotel Agent" || AGENTS_OK=false
# check_agent "http://localhost:8004" "Car Rental Agent" || AGENTS_OK=false
#
# if [ "$AGENTS_OK" = false ]; then
#     echo ""
#     echo -e "${RED}❌ Some agents failed to start. Check logs:${NC}"
#     echo "  - /tmp/airline-agent.log"
#     echo "  - /tmp/hotel-agent.log"
#     echo "  - /tmp/car-agent.log"
#     kill $AIRLINE_PID $HOTEL_PID $CAR_PID 2>/dev/null || true
#     exit 1
# fi

echo ""
echo -e "${BLUE}🛡️  Starting Secure Mediation Agent (Web UI)...${NC}"

echo ""
echo -e "${GREEN}🌐 Starting Web UI on http://localhost:8000${NC}"
echo -e "${YELLOW}📝 This will allow reviewers to trace execution in the web interface${NC}"
echo -e "${YELLOW}🔒 A2A Security Judge implementation available in secure-mediation-agent/security/${NC}"
sleep 2

# Start with standard ADK web (Plugin integration requires ADK version upgrade)
# Start from project root so secure-mediation-agent folder appears in the list
uv run adk web --port 8000 --reload > /tmp/mediation-agent.log 2>&1 &
MEDIATION_PID=$!
sleep 5

# Verify mediation agent started
# if curl -s http://localhost:8000 > /dev/null 2>&1; then
#     echo -e "${GREEN}✅ Mediation Agent Web UI is running${NC}"
#
#     # Try to open browser
#     echo -e "${BLUE}🌐 Opening browser...${NC}"
#     if command -v open &> /dev/null; then
#         open http://localhost:8000
#     elif command -v xdg-open &> /dev/null; then
#         xdg-open http://localhost:8000
#     fi
# else
#     echo -e "${RED}❌ Mediation Agent failed to start. Check /tmp/mediation-agent.log${NC}"
#     kill $AIRLINE_PID $HOTEL_PID $CAR_PID 2>/dev/null || true
#     cd ..
#     exit 1
# fi

# Try to open browser
echo -e "${BLUE}🌐 Opening browser...${NC}"
if command -v open &> /dev/null; then
    open http://localhost:8000
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8000
fi

cd ..

# Save PIDs for cleanup
echo "$AIRLINE_PID" > .airline.pid
echo "$HOTEL_PID" > .hotel.pid
echo "$CAR_PID" > .car.pid
echo "$MEDIATION_PID" > .mediation.pid

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ All systems ready!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}📋 Running Services:${NC}"
echo "  ✈️  Airline Agent:        http://localhost:8002"
echo "  🏨 Hotel Agent:          http://localhost:8003"
echo "  🚗 Car Rental Agent:     http://localhost:8004"
echo "  🛡️  Mediation Agent (UI): http://localhost:8000"
echo ""
echo -e "${YELLOW}📝 How to Use:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. ブラウザが開いたら、左上の「Select an agent」をクリック"
echo "2. 「secure-mediation-agent」を選択"
echo "3. 以下の例をコピーして入力してください:"
echo ""
echo "沖縄旅行の予約をお願いします。"
echo ""
echo "人数：2人"
echo "フライト: 羽田→那覇 (12/20-12/23)"
echo "ホテル: 那覇市内 3泊"
echo "レンタカー: コンパクトカー"
echo "予約完了まで完遂してください"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${YELLOW}📚 Logs:${NC}"
echo "  - Airline:   tail -f /tmp/airline-agent.log"
echo "  - Hotel:     tail -f /tmp/hotel-agent.log"
echo "  - Car:       tail -f /tmp/car-agent.log"
echo "  - Mediation: tail -f /tmp/mediation-agent.log"
echo ""
echo -e "${RED}Press Ctrl+C to stop all services${NC}"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Stopping all services...${NC}"
    kill $AIRLINE_PID $HOTEL_PID $CAR_PID $MEDIATION_PID 2>/dev/null || true
    rm -f .airline.pid .hotel.pid .car.pid .mediation.pid
    echo -e "${GREEN}✅ All services stopped${NC}"
    exit 0
}

# Wait for Ctrl+C
trap cleanup INT TERM

wait
