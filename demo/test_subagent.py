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

"""Simple test to verify sub-agent delegation works."""

import asyncio
import sys
import os

# Add parent directory and secure-mediation-agent to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'secure-mediation-agent'))

from agent import root_agent
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Create session service
session_service = InMemorySessionService()


async def test_simple_delegation():
    """Test simple delegation to a sub-agent."""
    print("=" * 80)
    print("Testing Sub-Agent Delegation")
    print("=" * 80)

    # Very simple request that should trigger planning_agent
    client_request = "Create a simple plan to say hello to the user."

    print(f"\nClient Request:\n{client_request}\n")
    print("-" * 80)

    try:
        # Create session
        session = await session_service.create_session(
            app_name="SubAgentTest",
            user_id="test_user"
        )

        # Create runner
        runner = Runner(
            agent=root_agent,
            app_name="SubAgentTest",
            session_service=session_service
        )

        # Prepare content
        content = types.Content(role='user', parts=[types.Part(text=client_request)])

        # Run the agent
        events = runner.run(user_id="test_user", session_id=session.id, new_message=content)

        # Process events
        print("\nMediation Agent Response:")
        event_count = 0
        for event in events:
            event_count += 1
            print(f"\n--- Event {event_count} ---")
            print(f"Event type: {type(event).__name__}")

            if hasattr(event, 'agent_name'):
                print(f"Agent name: {event.agent_name}")

            if event.is_final_response() and event.content:
                print("Final response:")
                for part in event.content.parts:
                    if part.text:
                        print(part.text)
            elif event.content:
                print("Intermediate content:")
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        print(f"  Text: {part.text[:100]}...")
                    if hasattr(part, 'function_call'):
                        print(f"  Function call: {part.function_call.name}")

        print(f"\nTotal events: {event_count}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run the test."""
    print("\n🧪 Sub-Agent Delegation Test\n")
    await test_simple_delegation()
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
