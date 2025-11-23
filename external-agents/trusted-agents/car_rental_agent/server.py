#!/usr/bin/env python3
"""A2A Server for Car Rental Agent."""

import uvicorn
from google.adk.cli.fast_api import get_fast_api_app

def main():
    """Start the car rental agent A2A server."""
    import os

    # Change to parent directory (trusted-agents) so agents_dir can find subdirectories
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(agent_dir)
    os.chdir(parent_dir)

    # Create FastAPI app with A2A enabled
    app = get_fast_api_app(
        web=None,  # Disable web UI
        a2a=True,
        agents_dir=".",
    )

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8004)

if __name__ == "__main__":
    main()
