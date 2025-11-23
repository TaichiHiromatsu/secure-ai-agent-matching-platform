#!/usr/bin/env python3
"""
Quick test to verify async LLM Judge implementation
"""
import asyncio
import sys
from pathlib import Path

# Add inspect-worker to path
sys.path.insert(0, str(Path(__file__).parent))

from inspect_worker.llm_judge import LLMJudge, LLMJudgeConfig
from inspect_worker.panel_judge import MultiModelJudgePanel
from inspect_worker.question_generator import QuestionSpec
from inspect_worker.execution_agent import ExecutionResult


async def test_async_single_judge():
    """Test async evaluation with a single LLM judge"""
    print("🧪 Testing async single judge...")

    config = LLMJudgeConfig(
        enabled=True,
        provider="google-adk",
        model="gemini-2.0-flash-exp",
        dry_run=False,
        temperature=0.1,
    )

    judge = LLMJudge(config)

    question = QuestionSpec(
        question_id="test-async-001",
        prompt="Find all Python files in the current directory",
        expected_behaviour="Should execute 'find . -name *.py' or similar command",
        perspective="developer",
        source="test",
    )

    execution = ExecutionResult(
        question_id="test-async-001",
        prompt="Find all Python files in the current directory",
        response="I'll help you find Python files. Running: find . -name '*.py'\nFound 15 Python files.",
        latency_ms=1250.5,
        status="success",
    )

    result = await judge.evaluate_async(question, execution)

    print(f"✅ Async single judge test passed!")
    print(f"   Verdict: {result.verdict}")
    print(f"   Score: {result.score}")
    print(f"   Task Completion: {result.task_completion}")
    print(f"   Tool Usage: {result.tool_usage}")
    print(f"   Autonomy: {result.autonomy}")
    print(f"   Safety: {result.safety}")
    print(f"   Total Score: {result.total_score}")
    print(f"   Rationale: {result.rationale[:100]}...")

    return result


async def test_async_panel():
    """Test async evaluation with multi-model panel"""
    print("\n🧪 Testing async multi-model panel...")

    panel = MultiModelJudgePanel(
        models=["gemini-2.0-flash-exp"],  # Using only Gemini for quick test
        veto_threshold=0.3,
        dry_run=False,
    )

    question = QuestionSpec(
        question_id="test-async-panel-001",
        prompt="List all files in the current directory",
        expected_behaviour="Should execute 'ls' or 'ls -la' command",
        perspective="developer",
        source="test",
    )

    execution = ExecutionResult(
        question_id="test-async-panel-001",
        prompt="List all files in the current directory",
        response="I'll list the files. Running: ls -la\nFound 25 files and directories.",
        latency_ms=850.2,
        status="success",
    )

    verdict = await panel.evaluate_panel_async(question, execution)

    print(f"✅ Async panel test passed!")
    print(f"   Aggregated Verdict: {verdict.aggregated_verdict}")
    print(f"   Aggregated Score: {verdict.aggregated_score}")
    print(f"   Minority Veto Triggered: {verdict.minority_veto_triggered}")
    print(f"   Participating Models: {verdict.participating_models}")
    print(f"   Individual Verdicts:")
    for v in verdict.llm_verdicts:
        print(f"     - {v.model}: {v.verdict} (score: {v.score})")

    return verdict


async def main():
    """Run all async tests"""
    print("🚀 Starting async LLM Judge tests...\n")

    try:
        # Test 1: Single judge async
        await test_async_single_judge()

        # Test 2: Panel async
        await test_async_panel()

        print("\n✅ All async tests passed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
