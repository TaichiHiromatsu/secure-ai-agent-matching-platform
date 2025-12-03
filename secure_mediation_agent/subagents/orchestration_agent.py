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
import re

from google.adk import Agent
from google.genai import types
from ..config.safety import SAFETY_SETTINGS_RELAXED
import re

# Import plan utilities from shared module
from ..utils.plan_utils import load_plan_from_artifact, parse_plan_for_step


async def execute_plan_step(
    step_id: str,
    agent_url: str,
    agent_name: str,
    task_description: str,
    input_data: dict[str, Any],
    planned_agent: str = "",
    trust_score: float = 1.0,
    plan_id: str = "",
) -> str:
    """Execute a single plan step by invoking the specified agent with security context.

    Args:
        step_id: Unique identifier for the step.
        agent_url: URL of the agent to invoke.
        agent_name: Name of the agent.
        task_description: Description of the task to perform.
        input_data: Input data for the agent.
        planned_agent: Expected agent name from the plan (for deviation detection).
        trust_score: Trust score of the agent (0.0-1.0).
        plan_id: Plan identifier for security verification.

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
        # Invoke the A2A agent with security context
        response = await invoke_a2a_agent(
            agent_url=agent_url,
            agent_name=agent_name,
            task=task_description,
            input_data=input_data,
            planned_agent=planned_agent,
            trust_score=trust_score,
            plan_id=plan_id,
        )
        response_data = json.loads(response)

        # セキュリティブロックをチェック
        if response_data.get("security_blocked"):
            result["status"] = "blocked"
            result["error"] = response_data.get("error", "Security violation detected")
            result["success"] = False
            result["security_blocked"] = True
        else:
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
    planned_agent: str = "",
    trust_score: float = 1.0,
    plan_id: str = "",
) -> str:
    """Invoke an A2A agent with security context using ADK's RemoteA2aAgent.

    This uses Google ADK's built-in RemoteA2aAgent which automatically handles:
    - Multi-turn conversations
    - Task state management
    - Agent card resolution
    - A2A protocol communication

    Security features:
    - Automatic security check via after_tool_callback
    - Plan deviation detection
    - Trust score validation
    - Independent plan verification by Judge Agent

    Args:
        agent_url: Base URL of the agent (e.g., "http://localhost:8002/a2a/airline_agent").
        agent_name: Name of the agent.
        task: Task description or prompt.
        input_data: Input data for the agent.
        planned_agent: Expected agent name from the plan (for deviation detection).
        trust_score: Trust score of the agent (0.0-1.0).
        plan_id: Plan identifier for Judge Agent to independently verify.

    Returns:
        JSON string with agent response (may include security_blocked flag).

    Note:
        Streaming functionality has been removed to ensure compatibility with
        ADK's automatic function calling. The function signature must use only
        simple types that can be parsed by the ADK.
    """
    from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
    from google.adk import Runner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.genai import types
    import logging
    import uuid

    logger = logging.getLogger(__name__)

    try:
        # Get agent card URL (A2A spec: /.well-known/agent-card.json)
        card_url = f"{agent_url.rstrip('/')}/.well-known/agent-card.json"

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
        # We collect all response parts AND full conversation history
        response_parts = []
        conversation_history = []  # Store all events for security analysis
        tool_calls = []
        tool_responses = []

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message
        ):
            logger.info(f"A2A agent {agent_name} event: {type(event).__name__}")

            # Record event for conversation history
            event_record = {
                "timestamp": event.timestamp if hasattr(event, 'timestamp') else datetime.now().timestamp(),
                "author": event.author if hasattr(event, 'author') else "unknown",
                "turn_complete": event.turn_complete if hasattr(event, 'turn_complete') else False,
            }

            # Extract text from different event types
            if hasattr(event, 'content') and event.content:
                if isinstance(event.content, str):
                    sanitized = _sanitize_text(event.content)
                    response_parts.append(sanitized)
                    event_record["text"] = sanitized
                else:
                    # Handle Content object with parts
                    parts_text = []
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            sanitized = _sanitize_text(part.text)
                            response_parts.append(sanitized)
                            parts_text.append(sanitized)

                    event_record["text"] = "\n".join(parts_text)
                    event_record["role"] = event.content.role if hasattr(event.content, 'role') else "model"

                # Extract function calls
                try:
                    function_calls = event.get_function_calls()
                    if function_calls:
                        event_record["function_calls"] = [
                            {
                                "name": fc.name,
                                "args": dict(fc.args) if hasattr(fc, 'args') else {}
                            }
                            for fc in function_calls
                        ]
                        tool_calls.extend(event_record["function_calls"])
                        logger.info(f"📞 Tool calls detected: {[fc['name'] for fc in event_record['function_calls']]}")
                except Exception as e:
                    logger.debug(f"No function calls in this event: {e}")

                # Extract function responses
                try:
                    function_responses = event.get_function_responses()
                    if function_responses:
                        event_record["function_responses"] = [
                            {
                                "name": fr.name,
                                "response": str(fr.response) if hasattr(fr, 'response') else "No response"
                            }
                            for fr in function_responses
                        ]
                        tool_responses.extend(event_record["function_responses"])
                        logger.info(f"📥 Tool responses detected: {[fr['name'] for fr in event_record['function_responses']]}")
                except Exception as e:
                    logger.debug(f"No function responses in this event: {e}")

            # Add any error information
            if hasattr(event, 'error_message') and event.error_message:
                event_record["error"] = event.error_message

            conversation_history.append(event_record)

        logger.info(f"A2A agent {agent_name} completed:")
        logger.info(f"  - {len(response_parts)} response parts")
        logger.info(f"  - {len(conversation_history)} conversation events")
        logger.info(f"  - {len(tool_calls)} tool calls")
        logger.info(f"  - {len(tool_responses)} tool responses")

        # Combine all response parts
        response_text = "\n".join(response_parts) if response_parts else "No response received"

        result = {
            "agent_url": agent_url,
            "output": response_text,
            "success": True,
            "timestamp": datetime.now().isoformat(),
            # Add conversation history for security analysis
            "conversation_history": conversation_history,
            "tool_calls": tool_calls,
            "tool_responses": tool_responses,
            "total_turns": len([e for e in conversation_history if e.get("turn_complete")]),
        }

        # Save conversation history to artifacts for final anomaly detection
        if plan_id:
            try:
                from pathlib import Path

                # Create conversations directory (inside secure_mediation_agent/)
                conversations_dir = Path(__file__).parent.parent / "artifacts" / "conversations" / plan_id
                conversations_dir.mkdir(parents=True, exist_ok=True)

                # Create filename with timestamp
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                conversation_file = conversations_dir / f"{agent_name}_{timestamp_str}.json"

                # Save full conversation history
                conversation_data = {
                    "plan_id": plan_id,
                    "agent_name": agent_name,
                    "agent_url": agent_url,
                    "task": task,
                    "input_data": input_data,
                    "output": response_text,
                    "conversation_history": conversation_history,
                    "tool_calls": tool_calls,
                    "tool_responses": tool_responses,
                    "total_turns": result["total_turns"],
                    "planned_agent": planned_agent,
                    "trust_score": trust_score,
                    "timestamp": result["timestamp"],
                }

                with open(conversation_file, 'w', encoding='utf-8') as f:
                    json.dump(conversation_data, f, indent=2, ensure_ascii=False)

                logger.info(f"💾 Saved conversation history to {conversation_file}")

            except Exception as e:
                logger.warning(f"⚠️ Failed to save conversation history: {e}")

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


def _sanitize_text(text: str) -> str:
    """Lightweight sanitization to reduce SAFETY triggers."""
    text = re.sub(r"(?i)(system prompt|ignore previous|override instructions)", r"[\\1]", text)
    text = re.sub(r"https?://\\S+", "<url>", text)
    text = re.sub(r"[A-Za-z0-9+/]{60,}={0,2}", "<data>", text)
    text = re.sub(r"\\n{3,}", "\\n\\n", text)
    return text


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


async def a2a_security_callback(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
    tool_response: dict[str, Any],
) -> dict[str, Any] | None:
    """A2A応答のセキュリティチェックと計画偏差検出

    このコールバックは以下を実行します:
    1. custom_judgeのJudge Agentによるセキュリティ判定
    2. 実行計画との偏差検出
    3. 異常検知時の実行ブロック

    Args:
        tool: 実行されたツール
        args: ツールに渡された引数
        tool_context: ツール実行コンテキスト
        tool_response: ツールの実行結果

    Returns:
        Noneを返してツール結果をそのまま使用、またはセキュリティブロック時はエラー辞書
    """
    import logging
    from google.adk.tools import base_tool

    logger = logging.getLogger(__name__)

    # invoke_a2a_agent または execute_plan_step 呼び出しでない場合はスキップ
    tool_name = getattr(tool, 'name', getattr(tool, '__name__', str(tool)))
    if tool_name not in ["invoke_a2a_agent", "execute_plan_step"]:
        return None

    # エラーレスポンスの場合はスキップ
    if isinstance(tool_response, dict) and tool_response.get("success") == False:
        return None

    logger.info(f"🔒 Starting security check for A2A response from {args.get('agent_name', 'unknown')}")

    try:
        # 1. custom_judgeのJudge Agentによるセキュリティ判定
        from ..security.custom_judge import secure_mediation_judge, custom_analysis_parser
        from ..safety_plugins import util
        from google.adk.sessions.in_memory_session_service import InMemorySessionService
        from google.genai import types
        from google.adk import Runner

        # 応答テキストを取得
        if isinstance(tool_response, str):
            result_dict = json.loads(tool_response)
        else:
            result_dict = tool_response

        response_text = str(result_dict.get("output", ""))
        agent_name = args.get("agent_name", "unknown")

        # Extract conversation history, tool calls, and tool responses
        conversation_history = result_dict.get("conversation_history", [])
        tool_calls = result_dict.get("tool_calls", [])
        tool_responses = result_dict.get("tool_responses", [])
        total_turns = result_dict.get("total_turns", 0)

        logger.info(f"📊 Security analysis context:")
        logger.info(f"  - Conversation events: {len(conversation_history)}")
        logger.info(f"  - Tool calls: {len(tool_calls)}")
        logger.info(f"  - Total turns: {total_turns}")

        # 2. 計画ファイルから期待されるエージェントを取得
        planned_agent = args.get("planned_agent")
        trust_score = args.get("trust_score", 1.0)
        plan_context = args.get("plan_context", {})
        plan_id = args.get("plan_id")  # Plan ID for Judge Agent verification

        # 計画偏差チェック
        deviation_detected = False
        deviation_reason = ""

        if planned_agent and planned_agent != agent_name:
            deviation_detected = True
            deviation_reason = f"計画偏差: 期待されたエージェント '{planned_agent}' ではなく '{agent_name}' が呼び出されました"
            logger.warning(f"⚠️ {deviation_reason}")

        # Judge Agentに渡すメッセージを構築（会話履歴を含める）
        judge_message_parts = [f"<tool_call>\nAgent: {agent_name}\nTask: {args.get('task', 'N/A')}\n</tool_call>"]

        # Add conversation history for comprehensive analysis
        if conversation_history:
            history_text = "## Conversation History\n"
            for i, event in enumerate(conversation_history, 1):
                history_text += f"\n### Turn {i} ({event.get('author', 'unknown')})\n"
                if event.get('text'):
                    history_text += f"Text: {event['text']}\n"
                if event.get('function_calls'):
                    history_text += f"Tool Calls: {json.dumps(event['function_calls'], ensure_ascii=False)}\n"
                if event.get('function_responses'):
                    history_text += f"Tool Responses: {json.dumps(event['function_responses'], ensure_ascii=False)}\n"
                if event.get('error'):
                    history_text += f"Error: {event['error']}\n"
            judge_message_parts.append(f"<conversation_history>\n{history_text}\n</conversation_history>")

        # Add final output
        judge_message_parts.append(f"<tool_output>\n{response_text}\n</tool_output>")

        # Add tool usage summary
        if tool_calls:
            tools_summary = f"Tools used: {', '.join([tc['name'] for tc in tool_calls])}"
            judge_message_parts.append(f"<tool_usage>\n{tools_summary}\n</tool_usage>")

        # 計画情報を追加
        if planned_agent:
            judge_message_parts.append(f"<plan_step>\nExpected agent: {planned_agent}\n</plan_step>")

        if plan_context:
            judge_message_parts.append(f"<execution_context>\n{json.dumps(plan_context, ensure_ascii=False)}\n</execution_context>")

        judge_message_parts.append(f"<agent_trust_score>\n{trust_score}\n</agent_trust_score>")

        # Add plan_id for independent verification
        if plan_id:
            judge_message_parts.append(f"<plan_id>\n{plan_id}\n</plan_id>")
            logger.info(f"📋 Included plan_id '{plan_id}' for Judge Agent verification")

        judge_message = "\n".join(judge_message_parts)

        # Judge Agentで判定
        session_service = InMemorySessionService()
        runner = Runner(
            agent=secure_mediation_judge,
            app_name="security_judge",
            session_service=session_service
        )

        author, judge_response = await util.run_prompt(
            user_id="security_judge",
            app_name="security_judge",
            runner=runner,
            message=types.Content(
                role="user",
                parts=[types.Part(text=judge_message)]
            )
        )

        # 判定結果を解析
        is_unsafe = custom_analysis_parser(judge_response)

        # 判定結果をログ出力
        if is_unsafe:
            logger.error(f"🚨 Security Judge detected UNSAFE content from {agent_name}")
            logger.error(f"Judge response: {judge_response}")

            # 実行をブロック
            return {
                "error": f"セキュリティ違反が検出されました: {judge_response}",
                "success": False,
                "security_blocked": True,
                "agent_name": agent_name,
                "judge_response": judge_response,
                "timestamp": datetime.now().isoformat(),
            }

        # 計画偏差があっても安全な場合は警告のみ
        if deviation_detected:
            logger.warning(f"⚠️ Plan deviation detected but content is SAFE: {deviation_reason}")
            # 結果に警告を追加
            if isinstance(tool_response, dict):
                tool_response["plan_deviation_warning"] = deviation_reason

        logger.info(f"✅ Security check passed for {agent_name}")
        return tool_response

    except Exception as e:
        logger.error(f"❌ Error in security callback: {e}", exc_info=True)
        # エラー時は安全側に倒してブロック
        return {
            "error": f"セキュリティチェック中にエラーが発生しました: {str(e)}",
            "success": False,
            "security_error": True,
            "timestamp": datetime.now().isoformat(),
        }


orchestrator = Agent(
    model='gemini-2.5-pro',
    name='orchestrator',
    description=(
        'Orchestration sub-agent that executes plans step-by-step, '
        'manages A2A agent communication, and tracks execution state with security monitoring.'
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
6. **Security Monitoring**: All A2A responses are automatically checked for security violations

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
- Store outputs using the execution context
- If a step fails, determine if you can continue or must stop

**IMPORTANT**: When calling invoke_a2a_agent, you can pass additional security context:
- planned_agent: The agent name expected by the plan (for deviation detection)
- trust_score: The trust score of the agent (0.0-1.0)
- plan_context: Additional context from the execution plan

A2A Communication:
- invoke_a2a_agent uses HTTP to communicate with remote A2A agents
- Provide clear task descriptions and input data in Japanese
- Handle responses and errors gracefully
- Respect agent timeouts (120 seconds)
- All responses are automatically checked for security violations

Security Features:
- Automatic detection of prompt injection attacks
- Plan deviation detection (wrong agent called)
- Trust score-based validation
- If security violation is detected, execution is blocked immediately

Error Handling:
- If a step fails but is not critical, log and continue
- If a critical step fails, stop execution and report
- If security_blocked=True in response, STOP IMMEDIATELY and report to user
- Always provide detailed error information
- Suggest corrective actions when possible

Use the provided tools:
- load_plan_from_artifact: Load a saved plan from artifacts directory
- parse_plan_for_step: Extract step details from plan content
- execute_plan_step: Execute a single step by calling the A2A agent
- invoke_a2a_agent: Directly invoke an A2A agent via HTTP (with security checks)
- check_step_dependencies: Verify dependencies
- get_step_output: Retrieve previous step outputs

**IMPORTANT: When starting execution:**
1. First, use load_plan_from_artifact to load the plan file
2. For each step, use parse_plan_for_step to get expected agent and output details
3. Pass this information to execute_plan_step for security validation
""",
    tools=[
        load_plan_from_artifact,
        parse_plan_for_step,
        execute_plan_step,
        invoke_a2a_agent,
        check_step_dependencies,
        get_step_output,
    ],
    after_tool_callback=a2a_security_callback,  # セキュリティコールバックを有効化
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,  # Very low temperature for precise execution
        safety_settings=SAFETY_SETTINGS_RELAXED,
    ),
)


# ============================================================================
# Security Integration: A2A Indirect Prompt Injection Detection
# ============================================================================
#
# The orchestrator agent can be enhanced with real-time security monitoring
# using the custom judge plugin. To enable:
#
# from google.adk import runners
# from security.custom_judge import a2a_security_judge
#
# orchestrator_runner = runners.InMemoryRunner(
#     agent=orchestrator,
#     plugins=[a2a_security_judge] if a2a_security_judge else []
# )
#
# This will automatically monitor:
# 1. All A2A agent calls (before execution)
# 2. All A2A agent responses (for indirect prompt injection)
# 3. Plan deviations based on trust scores
# 4. Data exfiltration attempts
#
# See security/custom_judge.py for implementation details.
# ============================================================================
