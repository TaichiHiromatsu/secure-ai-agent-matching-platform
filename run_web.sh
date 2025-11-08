#!/bin/bash
# Run ADK Web UI with environment variables

cd "$(dirname "$0")"

# Load environment variables from .env file if it exists
if [ -f "secure-mediation-agent/.env" ]; then
    export $(cat secure-mediation-agent/.env | grep -v '^#' | xargs)
fi

# Run adk web
echo "Starting ADK Web UI..."
echo "Open http://localhost:8000 in your browser"
adk web . --port 8000 --reload
