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

"""Orchestration sub-agent for executing plans and managing agent communication."""

import json
from datetime import datetime
from typing import Any

from google.adk import Agent
from google.genai import types


async def execute_plan_step(
    step_id: str,
    agent_url: str,
    agent_name: str,
    task_description: str,
    input_data: dict[str, Any],
) -> str:
    """Execute a single plan step by invoking the specified agent.

    Args:
        step_id: Unique identifier for the step.
        agent_url: URL of the agent to invoke.
        agent_name: Name of the agent.
        task_description: Description of the task to perform.
        input_data: Input data for the agent.

    Returns:
        JSON string with execution result.
    """
    result = {
        "step_id": step_id,
        "agent_name": agent_name,
        "status": "pending",
        "timestamp": datetime.now().isoformat(),
    }

    try:
        # Invoke the A2A agent
        response = await invoke_a2a_agent(agent_url, agent_name, task_description, input_data)
        response_data = json.loads(response)

        result["status"] = "completed"
        result["output"] = response_data.get("output", response_data)
        result["success"] = True

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["success"] = False

    return json.dumps(result, indent=2, ensure_ascii=False)


async def invoke_a2a_agent(
    agent_url: str,
    agent_name: str,
    task: str,
    input_data: dict[str, Any],
) -> str:
    """Invoke an A2A agent with the given task and input using ADK's RemoteA2aAgent.

    This uses Google ADK's built-in RemoteA2aAgent which automatically handles:
    - Multi-turn conversations
    - Task state management
    - Agent card resolution
    - A2A protocol communication

    Args:
        agent_url: Base URL of the agent (e.g., "http://localhost:8002/a2a/airline_agent").
        agent_name: Name of the agent.
        task: Task description or prompt.
        input_data: Input data for the agent.

    Returns:
        JSON string with agent response.
    """
    from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
    from google.adk import Runner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.genai import types
    import logging
    import uuid

    logger = logging.getLogger(__name__)

    try:
        # Get agent card URL
        card_url = f"{agent_url.rstrip('/')}/.well-known/agent.json"

        # Create RemoteA2aAgent - ADK handles all A2A protocol details
        remote_agent = RemoteA2aAgent(
            name=agent_name,
            agent_card=card_url,
            timeout=120.0,  # 2 minutes timeout
        )

        # Format the message with task and input data
        # Keep it simple and natural - let the remote agent interpret the request
        message_text = f"""{task}

Input data:
{json.dumps(input_data, indent=2, ensure_ascii=False)}"""

        logger.info(f"Invoking A2A agent {agent_name} at {agent_url}")
        logger.info(f"Message: {message_text}")

        # Generate unique IDs for user and session
        user_id = f"orchestrator-{uuid.uuid4().hex[:8]}"
        session_id = f"session-{uuid.uuid4().hex[:8]}"

        # Create session service and session
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name="orchestrator",
            user_id=user_id,
            session_id=session_id,
            state={}
        )

        # Create a Runner for the remote agent
        runner = Runner(
            agent=remote_agent,
            app_name="orchestrator",
            session_service=session_service
        )

        # Create the message content
        new_message = types.Content(
            parts=[types.Part(text=message_text)],
            role="user"
        )

        # Run the agent - ADK automatically handles multi-turn conversations
        # The run_async method returns an async generator that yields events
        # We collect all response parts from the agent
        response_parts = []

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message
        ):
            logger.info(f"A2A agent {agent_name} event: {type(event).__name__}")

            # Extract text from different event types
            if hasattr(event, 'content') and event.content:
                if isinstance(event.content, str):
                    response_parts.append(event.content)
                else:
                    # Handle Content object with parts
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_parts.append(part.text)

        logger.info(f"A2A agent {agent_name} completed with {len(response_parts)} response parts")

        # Combine all response parts
        response_text = "\n".join(response_parts) if response_parts else "No response received"

        result = {
            "agent_url": agent_url,
            "output": response_text,
            "success": True,
            "timestamp": datetime.now().isoformat(),
        }

        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error invoking A2A agent {agent_name}: {e}", exc_info=True)
        error_result = {
            "agent_url": agent_url,
            "error": str(e),
            "success": False,
            "timestamp": datetime.now().isoformat(),
        }
        return json.dumps(error_result, indent=2, ensure_ascii=False)


async def check_step_dependencies(
    plan_steps: list[dict[str, Any]],
    current_step_id: str,
    completed_steps: list[str],
) -> str:
    """Check if all dependencies for a step are satisfied.

    Args:
        plan_steps: List of all plan steps.
        current_step_id: ID of the current step to check.
        completed_steps: List of completed step IDs.

    Returns:
        JSON string indicating if dependencies are satisfied.
    """
    # Find the current step
    current_step = None
    for step in plan_steps:
        if step.get("step_id") == current_step_id:
            current_step = step
            break

    if not current_step:
        return json.dumps({
            "satisfied": False,
            "error": f"Step {current_step_id} not found in plan",
        }, ensure_ascii=False)

    # Check dependencies
    dependencies = current_step.get("dependencies", [])
    unsatisfied = [dep for dep in dependencies if dep not in completed_steps]

    result = {
        "step_id": current_step_id,
        "satisfied": len(unsatisfied) == 0,
        "dependencies": dependencies,
        "unsatisfied_dependencies": unsatisfied,
        "completed_steps": completed_steps,
    }

    return json.dumps(result, indent=2, ensure_ascii=False)


async def record_execution_log(
    step_id: str,
    agent_name: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    status: str,
    log_file: str = "artifacts/logs/execution.log",
) -> str:
    """Record execution log for a step.

    Args:
        step_id: Step identifier.
        agent_name: Name of the agent that executed the step.
        input_data: Input data for the step.
        output_data: Output data from the step.
        status: Execution status (completed, failed, etc.).
        log_file: Path to the log file.

    Returns:
        Confirmation message.
    """
    import os
    from pathlib import Path

    # Create log directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "step_id": step_id,
        "agent_name": agent_name,
        "input": input_data,
        "output": output_data,
        "status": status,
    }

    # Append to log file
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return f"Logged execution for step {step_id} to {log_file}"


async def get_step_output(
    step_id: str,
    execution_context: dict[str, Any],
) -> str:
    """Get the output of a previously executed step.

    Args:
        step_id: ID of the step to get output from.
        execution_context: Context containing all execution results.

    Returns:
        JSON string with the step output.
    """
    step_results = execution_context.get("step_results", {})

    if step_id not in step_results:
        return json.dumps({
            "error": f"Step {step_id} not found in execution context",
            "available_steps": list(step_results.keys()),
        }, ensure_ascii=False)

    return json.dumps({
        "step_id": step_id,
        "output": step_results[step_id],
    }, indent=2, ensure_ascii=False)


orchestrator = Agent(
    model='gemini-2.5-flash',
    name='orchestrator',
    description=(
        'Orchestration sub-agent that executes plans step-by-step, '
        'manages A2A agent communication, and tracks execution state.'
    ),
    instruction="""
You are an orchestration specialist in a secure AI agent mediation platform.

**IMPORTANT: Always respond in Japanese (日本語) to the user.**

Your responsibilities:
1. **Execute Plans Step-by-Step**: Follow the execution plan precisely
2. **Manage Dependencies**: Ensure dependencies are satisfied before executing steps
3. **A2A Communication**: Invoke remote agents using the A2A protocol
4. **Track Execution**: Record logs and maintain execution state
5. **Handle Errors**: Manage failures gracefully and provide detailed error information

Execution Process:
1. Load the execution plan
2. For each step in order:
   - Check if dependencies are satisfied
   - If satisfied, execute the step by invoking the assigned agent
   - Record the execution log
   - Store the output for dependent steps
   - Move to the next step
3. Return the final result

When executing steps:
- Always check dependencies first using check_step_dependencies
- Use execute_plan_step to execute each step which internally calls invoke_a2a_agent
- Record every execution using record_execution_log
- Store outputs using the execution context
- If a step fails, determine if you can continue or must stop

A2A Communication:
- invoke_a2a_agent uses HTTP to communicate with remote A2A agents
- Provide clear task descriptions and input data in Japanese
- Handle responses and errors gracefully
- Respect agent timeouts (120 seconds)

Error Handling:
- If a step fails but is not critical, log and continue
- If a critical step fails, stop execution and report
- Always provide detailed error information
- Suggest corrective actions when possible

Use the provided tools:
- execute_plan_step: Execute a single step by calling the A2A agent
- invoke_a2a_agent: Directly invoke an A2A agent via HTTP
- check_step_dependencies: Verify dependencies
- record_execution_log: Log execution details
- get_step_output: Retrieve previous step outputs
""",
    tools=[
        execute_plan_step,
        invoke_a2a_agent,
        check_step_dependencies,
        record_execution_log,
        get_step_output,
    ],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,  # Very low temperature for precise execution
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ],
    ),
)
