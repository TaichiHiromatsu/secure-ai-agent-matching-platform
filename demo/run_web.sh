#!/bin/bash
# Run ADK Web UI with environment variables

# Move to project root
cd "$(dirname "$0")/.."

# Load environment variables from .env file if it exists
if [ -f "secure-mediation-agent/.env" ]; then
    export $(cat secure-mediation-agent/.env | grep -v '^#' | xargs)
fi

# Run adk web from secure-mediation-agent directory
echo "Starting ADK Web UI..."
echo "Open http://localhost:8000 in your browser"
cd secure-mediation-agent
uv run adk web . --port 8000 --reload
