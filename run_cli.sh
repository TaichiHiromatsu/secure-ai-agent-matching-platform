#!/bin/bash
# Run ADK CLI with environment variables

cd "$(dirname "$0")"

# Load environment variables from .env file if it exists
if [ -f "secure-mediation-agent/.env" ]; then
    export $(cat secure-mediation-agent/.env | grep -v '^#' | xargs)
fi

# Run adk run
echo "Starting ADK CLI..."
echo "Type 'exit' to quit"
echo ""
adk run secure-mediation-agent
