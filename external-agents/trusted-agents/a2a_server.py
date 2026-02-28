"""Custom A2A Server for external agents.

This server wraps Google ADK agents and serves them with the correct
A2A Protocol v0.3.16 endpoint paths (/.well-known/agent-card.json).

A2A Artifact交換対応:
各エージェントが get_pending_artifacts() を公開している場合、
レスポンスにArtifact（ファイル・バイナリデータ）を含めて返す。
"""

import asyncio
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse

from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor
from a2a.types import AgentCard
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

from google.adk.agents import LiveRequestQueue
from google.adk.agents.run_config import RunConfig
from google.adk.events import Event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ADKAgentExecutor(AgentExecutor):
    """Agent executor that wraps Google ADK agents.

    A2A Artifact交換に対応: エージェントモジュールが get_pending_artifacts() を
    公開している場合、実行後にArtifactを収集してレスポンスに含める。
    """

    def __init__(self, adk_agent, agent_name: str, agent_module=None):
        self.adk_agent = adk_agent
        self.agent_name = agent_name
        self.agent_module = agent_module  # Artifact取得用にモジュール参照を保持
        self._request_queue = None
        self._last_artifacts: list = []  # 直近の実行で収集されたArtifact（HTTPレスポンス用）

    async def execute(self, context, event_queue):
        """Execute the ADK agent with the given context.

        Artifact収集は全イベント処理完了後に行う（修正: ループ内で収集すると
        ADKのtool関数完了前にドレインされてしまう問題を解消）。
        """
        # Try to import InMemoryRunner from various locations
        try:
            from google.adk.agents import InMemoryRunner
        except ImportError:
            try:
                from google.adk.runners import InMemoryRunner
            except ImportError:
                from google.adk import Runner as InMemoryRunner

        from a2a.types import Message, TextPart

        # Get the user message from the context
        user_message = ""
        if hasattr(context, 'message') and context.message:
            for part in context.message.parts:
                if hasattr(part, 'text'):
                    user_message = part.text
                    break

        if not user_message:
            user_message = "Hello"

        logger.info(f"Executing agent {self.agent_name} with message: {user_message[:100]}")

        # Create an InMemoryRunner for the ADK agent
        runner = InMemoryRunner(
            agent=self.adk_agent,
            app_name=self.agent_name,
        )

        # Run the agent — まず全イベントを収集し、最後にArtifactをまとめて付加
        try:
            session_id = f"session_{self.agent_name}"
            user_id = "a2a_user"

            # Phase 1: 全イベントを収集（Artifact収集はまだ行わない）
            collected_content_parts: list[str] = []

            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_message,
            ):
                if hasattr(event, 'content') and event.content:
                    collected_content_parts.append(str(event.content))

            # Phase 2: 全イベント処理完了後にArtifactを収集
            artifacts = self._collect_artifacts()

            # テキストPartを構築
            final_text = "\n".join(collected_content_parts) if collected_content_parts else "(no response)"
            parts = [TextPart(text=final_text)]

            if artifacts:
                # Artifact情報をテキストに付加（DataPartは仕様不安定のため、
                # テキスト付加 + artifacts フィールドの二重送信で確実性を担保）
                art_info = "\n\n[A2A Artifacts]\n"
                for artifact in artifacts:
                    art_info += (
                        f"- {artifact.get('name', 'artifact')}: "
                        f"mime_type={artifact.get('parts', [{}])[0].get('mimeType', 'unknown')}, "
                        f"size={artifact.get('metadata', {}).get('size_bytes', '?')} bytes\n"
                    )
                parts[0] = TextPart(text=final_text + art_info)
                logger.info(f"Artifacts collected after execution: {len(artifacts)} items")

                # artifacts情報をコンテキストに保存（HTTPレスポンスで返すため）
                self._last_artifacts = artifacts

            response_message = Message(
                role="agent",
                parts=parts,
            )
            await event_queue.enqueue_event(response_message)
        except Exception as e:
            logger.error(f"Error executing agent: {e}")
            error_message = Message(
                role="agent",
                parts=[TextPart(text=f"Error: {str(e)}")]
            )
            await event_queue.enqueue_event(error_message)

    def _collect_artifacts(self) -> list:
        """エージェントモジュールからpending artifactsを収集する。"""
        if self.agent_module and hasattr(self.agent_module, "get_pending_artifacts"):
            try:
                return self.agent_module.get_pending_artifacts()
            except Exception as e:
                logger.warning(f"Failed to collect artifacts from {self.agent_name}: {e}")
        return []

    async def cancel(self, context, event_queue):
        """Cancel the execution."""
        pass


def load_agent_card(agent_dir: Path, base_url: str) -> AgentCard:
    """Load agent card from agent.json file."""
    card_path = agent_dir / "agent.json"
    with open(card_path) as f:
        data = json.load(f)

    # Update URL to use the correct base
    agent_name = agent_dir.name
    data["url"] = f"{base_url}/a2a/{agent_name}"

    return AgentCard(**data)


def create_app(host: str, port: int, agents_dir: Path) -> Starlette:
    """Create the Starlette application with all agents."""

    routes = []
    base_url = f"http://{host}:{port}"

    # Discover all agent directories
    for agent_path in agents_dir.iterdir():
        if not agent_path.is_dir():
            continue
        if agent_path.name.startswith((".", "__")):
            continue
        if not (agent_path / "agent.json").exists():
            continue

        agent_name = agent_path.name
        logger.info(f"Loading agent: {agent_name}")

        try:
            # Load the agent card
            agent_card = load_agent_card(agent_path, base_url)

            # Import the ADK agent（名前空間を分離して各エージェント固有のモジュールとしてロード）
            agent_py = agent_path / "agent.py"
            module_name = f"agent_{agent_name}"  # 例: agent_sales_agent, agent_data_harvester_agent
            spec = importlib.util.spec_from_file_location(module_name, str(agent_py))
            agent_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = agent_module  # sysに登録して依存解決を可能に
            # agent.py内の相対import（google.adk等）のためPATHに一時追加
            sys.path.insert(0, str(agent_path))
            try:
                spec.loader.exec_module(agent_module)
            finally:
                sys.path.pop(0)
            adk_agent = agent_module.root_agent

            # Create executor and handler (agent_moduleを渡してArtifact取得に使用)
            executor = ADKAgentExecutor(adk_agent, agent_name, agent_module=agent_module)
            handler = DefaultRequestHandler(
                agent_executor=executor,
                task_store=None,
            )

            # Create A2A application with new spec path
            a2a_app = A2AStarletteApplication(
                agent_card=agent_card,
                http_handler=handler,
            )

            # Get routes with the new agent-card.json path
            agent_routes = a2a_app.routes(
                rpc_url=f"/a2a/{agent_name}",
                agent_card_url=f"/a2a/{agent_name}{AGENT_CARD_WELL_KNOWN_PATH}",
            )

            routes.extend(agent_routes)
            logger.info(f"Successfully loaded agent: {agent_name}")
            logger.info(f"  - RPC URL: /a2a/{agent_name}")
            logger.info(f"  - Agent Card: /a2a/{agent_name}{AGENT_CARD_WELL_KNOWN_PATH}")

        except Exception as e:
            logger.error(f"Failed to load agent {agent_name}: {e}")
            import traceback
            traceback.print_exc()

    # Create root endpoint
    async def root(request):
        return JSONResponse({
            "status": "ok",
            "message": "A2A Server for external agents",
            "agents": [r.path for r in routes if "agent-card.json" in r.path]
        })

    routes.insert(0, Route("/", root))

    app = Starlette(routes=routes)
    return app


async def run_test_dialogue(agent_path: Path, agent_name: str):
    """Run a test dialogue with the specified agent."""
    import sys
    
    print(f"Loading agent from {agent_path}")
    
    # Import the ADK agent（名前空間を分離してロード）
    agent_py = agent_path / "agent.py"
    module_name = f"agent_{agent_name}"
    spec = importlib.util.spec_from_file_location(module_name, str(agent_py))
    agent_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = agent_module
    sys.path.insert(0, str(agent_path))
    try:
        spec.loader.exec_module(agent_module)
        adk_agent = agent_module.root_agent
    except Exception as e:
        logger.error(f"Failed to load agent: {e}")
        return
    finally:
        sys.path.pop(0)

    # Try to import InMemoryRunner from various locations
    try:
        from google.adk.agents import InMemoryRunner
    except ImportError:
        try:
            from google.adk.runners import InMemoryRunner
        except ImportError:
            from google.adk import Runner as InMemoryRunner


    print(f"Initializing runner for {agent_name}...")
    runner = InMemoryRunner(
        agent=adk_agent,
        app_name=agent_name,
    )

    session_id = f"test_session_{agent_name}"
    user_id = "test_user"
    
    # Authenticate/Create session
    print(f"Creating session {session_id} for user {user_id}...")
    try:
        # Check if session service exists and try to create session
        if hasattr(runner, "session_service"):
            session_service = runner.session_service
            # Check for create_session method
            if hasattr(session_service, "create_session"):
                start_session = session_service.create_session
                import inspect
                if inspect.iscoroutinefunction(start_session):
                    await start_session(user_id=user_id, session_id=session_id, app_name=agent_name)
                else:
                    start_session(user_id=user_id, session_id=session_id, app_name=agent_name)
                print("Session created successfully.")
    except Exception as e:
        print(f"Warning: Failed to explicitely create session: {e}")
        # Proceed anyway as some runners might auto-create
    
    dialogue = [
        "沖縄旅行を計画しています。3月10日から2泊3日で那覇のツアーを探して予約したいです。",
        "はい、お願いします。必要な情報はありますか？",
        "名前は山田太郎、メールは taro@example.com です。"
    ]
    
    print(f"\n=== START TEST DIALOGUE: {agent_name} ===\n")
    
    for msg in dialogue:
        print(f"User: {msg}")
        print("-" * 50)
        
        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=msg,
            ):
                if hasattr(event, "content") and event.content:
                    print(f"Agent: {event.content}")
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()
            
        print("=" * 50)
        
    print(f"\n=== END TEST DIALOGUE: {agent_name} ===\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="A2A Server for external agents")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    parser.add_argument("--test-agent", help="Name of agent to test (runs dialogue and exits)")
    parser.add_argument("agents_dir", type=Path, help="Directory containing agents")

    args = parser.parse_args()

    if not args.agents_dir.exists():
        logger.error(f"Agents directory not found: {args.agents_dir}")
        sys.exit(1)

    if args.test_agent:
        agent_path = args.agents_dir / args.test_agent
        if not agent_path.exists():
            logger.error(f"Agent directory not found: {agent_path}")
            sys.exit(1)
            
        asyncio.run(run_test_dialogue(agent_path, args.test_agent))
        return

    app = create_app(args.host, args.port, args.agents_dir)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
