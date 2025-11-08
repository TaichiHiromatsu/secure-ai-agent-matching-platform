#!/bin/bash
# Start all external agents

# Move to project root
cd "$(dirname "$0")/.."

# Load environment variables
if [ -f "secure-mediation-agent/.env" ]; then
    export $(cat secure-mediation-agent/.env | grep -v '^#' | xargs)
fi

echo "🚀 Starting External Agents..."
echo ""

# Start Airline Agent (port 8002) as A2A server
echo "Starting Airline Agent (A2A) on port 8002..."
uv run adk api_server --a2a --port 8002 external-agents/trusted-agents/airline-agent &
AIRLINE_PID=$!
sleep 2

# Start Hotel Agent (port 8003) as A2A server
echo "Starting Hotel Agent (A2A) on port 8003..."
uv run adk api_server --a2a --port 8003 external-agents/trusted-agents/hotel-agent &
HOTEL_PID=$!
sleep 2

# Start Car Rental Agent (port 8004) as A2A server
echo "Starting Car Rental Agent (A2A) on port 8004..."
uv run adk api_server --a2a --port 8004 external-agents/trusted-agents/car-rental-agent &
CAR_PID=$!
sleep 2

echo ""
echo "✅ All agents started!"
echo ""
echo "📋 Agent URLs:"
echo "  - Airline Agent:     http://localhost:8002"
echo "  - Hotel Agent:       http://localhost:8003"
echo "  - Car Rental Agent:  http://localhost:8004"
echo ""
echo "💡 To start the Mediation Agent:"
echo "   ./run_web.sh  (for Web UI on port 8000)"
echo "   ./run_cli.sh  (for CLI)"
echo ""
echo "Press Ctrl+C to stop all agents"

# Save PIDs for cleanup
echo "$AIRLINE_PID" > .airline.pid
echo "$HOTEL_PID" > .hotel.pid
echo "$CAR_PID" > .car.pid

# Wait for Ctrl+C
trap "echo ''; echo 'Stopping all agents...'; kill $AIRLINE_PID $HOTEL_PID $CAR_PID 2>/dev/null; rm -f .airline.pid .hotel.pid .car.pid; echo 'All agents stopped.'; exit 0" INT

wait
