#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple demo of the Secure Mediation Agent."""

import asyncio
import sys
import os

# Add parent directory and secure_mediation_agent to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'secure_mediation_agent'))

from agent import root_agent
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Create session service
session_service = InMemorySessionService()


async def demo_simple_request():
    """Demo: Simple request to the mediation agent."""
    print("=" * 80)
    print("Demo 1: Simple Request")
    print("=" * 80)

    client_request = """
    I need to check if the following numbers are prime: 17, 24, 31, 42, 53
    """

    print(f"\nClient Request:\n{client_request}\n")
    print("-" * 80)

    try:
        # Create session
        session = await session_service.create_session(
            app_name="SecureAgentDemo",
            user_id="demo_user"
        )

        # Create runner
        runner = Runner(agent=root_agent, app_name="SecureAgentDemo", session_service=session_service)

        # Prepare content
        content = types.Content(role='user', parts=[types.Part(text=client_request)])

        # Run the agent
        events = runner.run(user_id="demo_user", session_id=session.id, new_message=content)

        # Process events
        print("\nMediation Agent Response:")
        for event in events:
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if part.text:
                        print(part.text)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


async def demo_complex_request():
    """Demo: More complex multi-step request."""
    print("\n" + "=" * 80)
    print("Demo 2: Complex Multi-step Request")
    print("=" * 80)

    client_request = """
    I need help with the following tasks:
    1. Find the current time in Tokyo
    2. Check if the hour value is a prime number
    3. Give me a summary of the results
    """

    print(f"\nClient Request:\n{client_request}\n")
    print("-" * 80)

    try:
        # Create session
        session = await session_service.create_session(
            app_name="SecureAgentDemo",
            user_id="demo_user"
        )

        runner = Runner(agent=root_agent, app_name="SecureAgentDemo", session_service=session_service)
        content = types.Content(role='user', parts=[types.Part(text=client_request)])
        events = runner.run(user_id="demo_user", session_id=session.id, new_message=content)

        print("\nMediation Agent Response:")
        for event in events:
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if part.text:
                        print(part.text)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


async def demo_with_agent_urls():
    """Demo: Request with specific agent platform URLs."""
    print("\n" + "=" * 80)
    print("Demo 3: Request with Agent Registry")
    print("=" * 80)

    client_request = """
    Search the agent registry for agents that can help with mathematical computations.
    I want to know:
    - Which agents are available
    - Their trust scores
    - What capabilities they have

    Agent registry URLs to search:
    - http://localhost:8002 (example registry)
    """

    print(f"\nClient Request:\n{client_request}\n")
    print("-" * 80)

    try:
        # Create session
        session = await session_service.create_session(
            app_name="SecureAgentDemo",
            user_id="demo_user"
        )

        runner = Runner(agent=root_agent, app_name="SecureAgentDemo", session_service=session_service)
        content = types.Content(role='user', parts=[types.Part(text=client_request)])
        events = runner.run(user_id="demo_user", session_id=session.id, new_message=content)

        print("\nMediation Agent Response:")
        for event in events:
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if part.text:
                        print(part.text)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all demos."""
    print("\n🔒 Secure AI Agent Matching Platform - Demo\n")

    # Demo 1: Simple request
    await demo_simple_request()

    # Wait a bit between demos
    await asyncio.sleep(2)

    # Demo 2: Complex request
    await demo_complex_request()

    # Wait a bit
    await asyncio.sleep(2)

    # Demo 3: Agent registry search
    await demo_with_agent_urls()

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
