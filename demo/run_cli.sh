#!/bin/bash
# Run ADK CLI with environment variables

# Move to project root
cd "$(dirname "$0")/.."

# Load environment variables from .env file if it exists
if [ -f "secure_mediation_agent/.env" ]; then
    export $(cat secure_mediation_agent/.env | grep -v '^#' | xargs)
fi

# Run adk run
echo "Starting ADK CLI..."
echo "Type 'exit' to quit"
echo ""
uv run adk run secure_mediation_agent
