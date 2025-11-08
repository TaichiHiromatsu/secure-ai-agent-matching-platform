#!/usr/bin/env python3
"""A2A Server for Airline Agent."""

import uvicorn
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.agents.agent_loader import AgentLoader

def main():
    """Start the airline agent A2A server."""
    # Load the agent
    agent_loader = AgentLoader(".")

    # Create FastAPI app with A2A enabled
    app = get_fast_api_app(
        adk_web_server=None,  # Will be created internally
        a2a=True,
        agents_dir=".",
    )

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8002)

if __name__ == "__main__":
    main()
