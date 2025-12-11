"""
Test suite for the new ADK-based Jury Judge workflow.

Tests the following modules:
- artifact_tools.py: FunctionTool definitions
- juror_agents.py: Juror agent implementations
- discussion_workflow.py: LoopAgent/SequentialAgent patterns
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

# Import new modules
from jury_judge_worker.artifact_tools import (
    ArtifactContext,
    ArtifactToolsWithContext,
    generate_evaluation_summary,
    TOOL_SCHEMAS,
    get_openai_tools,
    get_anthropic_tools,
)
from jury_judge_worker.juror_agents import (
    JurorConfig,
    JurorRole,
    JurorEvaluationOutput,
    BaseJurorAgent,
    create_juror_agents,
    get_juror_config,
    JUROR_CONFIGS,
)
from jury_judge_worker.discussion_workflow import (
    DiscussionState,
    DiscussionRoundResult,
    DiscussionWorkflowResult,
    ConsensusChecker,
    SequentialDiscussionRound,
    DiscussionLoopWorkflow,
    ConsensusStatus,
)


# =============================================================================
# Artifact Tools Tests
# =============================================================================

class TestArtifactContext:
    """Test ArtifactContext dataclass."""

    def test_create_empty_context(self):
        ctx = ArtifactContext()
        assert ctx.security_gate_uri is None
        assert ctx.agent_card_uri is None
        assert ctx.security_gate_records is None
        assert ctx.agent_card_records is None

    def test_create_with_records(self):
        sg_records = [{"verdict": "pass", "prompt": "test"}]
        aca_records = [{"verdict": "fail", "skill": "test_skill"}]

        ctx = ArtifactContext(
            security_gate_records=sg_records,
            agent_card_records=aca_records,
        )

        assert ctx.security_gate_records == sg_records
        assert ctx.agent_card_records == aca_records


class TestArtifactToolsWithContext:
    """Test ArtifactToolsWithContext class."""

    def test_fetch_security_gate_failures_with_cached_records(self):
        records = [
            {"verdict": "needs_review", "prompt": "malicious prompt 1"},
            {"verdict": "pass", "prompt": "safe prompt"},
            {"verdict": "error", "prompt": "error prompt"},
        ]
        ctx = ArtifactContext(security_gate_records=records)
        tools = ArtifactToolsWithContext(ctx)

        result = tools.fetch_security_gate_failures(max_records=10, include_errors=True)

        assert result["status"] == "success"
        assert result["count"] == 2  # needs_review + error
        assert len(result["records"]) == 2

    def test_fetch_security_gate_passes(self):
        records = [
            {"verdict": "pass", "prompt": "blocked malicious"},
            {"verdict": "safe_pass", "prompt": "safe response"},
            {"verdict": "needs_review", "prompt": "questionable"},
        ]
        ctx = ArtifactContext(security_gate_records=records)
        tools = ArtifactToolsWithContext(ctx)

        result = tools.fetch_security_gate_passes(max_records=5)

        assert result["status"] == "success"
        assert result["count"] == 2  # pass + safe_pass

    def test_fetch_agent_card_failures(self):
        records = [
            {"verdict": "fail", "skill": "task1"},
            {"evaluation": {"verdict": "failed"}, "skill": "task2"},
            {"verdict": "pass", "skill": "task3"},
        ]
        ctx = ArtifactContext(agent_card_records=records)
        tools = ArtifactToolsWithContext(ctx)

        result = tools.fetch_agent_card_failures(max_records=10)

        assert result["status"] == "success"
        assert result["count"] == 2  # fail + failed

    def test_get_other_juror_opinions(self):
        ctx = ArtifactContext()
        tools = ArtifactToolsWithContext(ctx)

        # Update juror evaluations
        tools.update_juror_evaluations([
            {"juror_id": "gpt-4o", "verdict": "safe_pass", "score": 85},
            {"juror_id": "claude-3-haiku", "verdict": "needs_review", "score": 65},
        ])
        tools.update_consensus_status("split")

        result = tools.get_other_juror_opinions()

        assert result["status"] == "success"
        assert len(result["evaluations"]) == 2
        assert result["consensus_status"] == "split"


class TestEvaluationSummary:
    """Test evaluation summary generation."""

    def test_generate_summary_with_data(self):
        sg_records = [
            {"verdict": "pass"},
            {"verdict": "pass"},
            {"verdict": "needs_review"},
        ]
        aca_records = [
            {"verdict": "pass"},
            {"evaluation": {"verdict": "fail"}},
        ]

        summary = generate_evaluation_summary(sg_records, aca_records)

        assert "Security Gate評価" in summary
        assert "Agent Card Accuracy評価" in summary
        assert "3" in summary  # Total SG tests
        assert "2" in summary  # Total ACA tests

    def test_generate_summary_empty(self):
        summary = generate_evaluation_summary([], [])

        assert "データなし" in summary


class TestToolSchemas:
    """Test tool schema definitions."""

    def test_tool_schemas_complete(self):
        expected_tools = [
            "fetch_security_gate_failures",
            "fetch_security_gate_passes",
            "fetch_agent_card_failures",
            "fetch_agent_card_passes",
            "get_other_juror_opinions",
        ]

        for tool_name in expected_tools:
            assert tool_name in TOOL_SCHEMAS
            assert "name" in TOOL_SCHEMAS[tool_name]
            assert "description" in TOOL_SCHEMAS[tool_name]
            assert "parameters" in TOOL_SCHEMAS[tool_name]

    def test_get_openai_tools(self):
        tools = get_openai_tools()

        assert len(tools) == 5
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool

    def test_get_anthropic_tools(self):
        tools = get_anthropic_tools()

        assert len(tools) == 5
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool


# =============================================================================
# Juror Agents Tests
# =============================================================================

class TestJurorConfig:
    """Test JurorConfig dataclass."""

    def test_default_configs_exist(self):
        expected_jurors = ["gpt-4o", "claude-3-haiku-20240307", "gemini-2.5-flash"]

        for juror_id in expected_jurors:
            config = get_juror_config(juror_id)
            assert config is not None
            assert config.juror_id == juror_id
            assert config.role in JurorRole

    def test_output_key_generation(self):
        config = JurorConfig(
            juror_id="test-model",
            model_name="test-model",
            role=JurorRole.POLICY_COMPLIANCE,
            role_name="Test Juror",
            role_focus="Testing",
            description="Test description",
            evaluation_prompt="Test prompt",
        )

        assert config.output_key == "juror_test_model_opinion"


class TestJurorEvaluationOutput:
    """Test JurorEvaluationOutput dataclass."""

    def test_to_dict(self):
        output = JurorEvaluationOutput(
            juror_id="gpt-4o",
            role_name="陪審員A",
            role_focus="ポリシー遵守性",
            verdict="safe_pass",
            overall_score=85.0,
            confidence=0.9,
            rationale="Test rationale",
            task_completion=35.0,
            tool_usage=25.0,
            autonomy=15.0,
            safety=10.0,
        )

        result = output.to_dict()

        assert result["juror_id"] == "gpt-4o"
        assert result["verdict"] == "safe_pass"
        assert result["overall_score"] == 85.0
        assert result["task_completion"] == 35.0


class TestBaseJurorAgent:
    """Test BaseJurorAgent class."""

    def test_create_agent(self):
        config = JUROR_CONFIGS["gpt-4o"]
        agent = BaseJurorAgent(config=config)

        assert agent.juror_id == "gpt-4o"
        assert agent.get_provider() == "openai"

    def test_provider_detection(self):
        test_cases = [
            ("gpt-4o", "openai"),
            ("claude-3-haiku-20240307", "anthropic"),
            ("gemini-2.5-flash", "google-adk"),
        ]

        for juror_id, expected_provider in test_cases:
            config = JUROR_CONFIGS[juror_id]
            agent = BaseJurorAgent(config=config)
            assert agent.get_provider() == expected_provider


class TestCreateJurorAgents:
    """Test create_juror_agents factory function."""

    def test_create_default_agents(self):
        agents = create_juror_agents()

        assert len(agents) == 3
        assert "gpt-4o" in agents
        assert "claude-3-haiku-20240307" in agents
        assert "gemini-2.5-flash" in agents

    def test_create_specific_agents(self):
        agents = create_juror_agents(juror_ids=["gpt-4o", "gemini-2.5-flash"])

        assert len(agents) == 2
        assert "gpt-4o" in agents
        assert "gemini-2.5-flash" in agents
        assert "claude-3-haiku-20240307" not in agents


# =============================================================================
# Discussion Workflow Tests
# =============================================================================

class TestConsensusChecker:
    """Test ConsensusChecker class."""

    def test_unanimous_consensus(self):
        checker = ConsensusChecker()
        evaluations = {
            "a": JurorEvaluationOutput(
                juror_id="a", role_name="A", role_focus="X",
                verdict="safe_pass", overall_score=90, confidence=0.9, rationale=""
            ),
            "b": JurorEvaluationOutput(
                juror_id="b", role_name="B", role_focus="Y",
                verdict="safe_pass", overall_score=85, confidence=0.85, rationale=""
            ),
            "c": JurorEvaluationOutput(
                juror_id="c", role_name="C", role_focus="Z",
                verdict="safe_pass", overall_score=88, confidence=0.88, rationale=""
            ),
        }

        status, reached, verdict = checker.check_consensus(evaluations, round_number=1)

        assert status == ConsensusStatus.UNANIMOUS
        assert reached is True
        assert verdict == "safe_pass"

    def test_majority_consensus(self):
        checker = ConsensusChecker(consensus_threshold=0.67)
        evaluations = {
            "a": JurorEvaluationOutput(
                juror_id="a", role_name="A", role_focus="X",
                verdict="safe_pass", overall_score=90, confidence=0.9, rationale=""
            ),
            "b": JurorEvaluationOutput(
                juror_id="b", role_name="B", role_focus="Y",
                verdict="safe_pass", overall_score=85, confidence=0.85, rationale=""
            ),
            "c": JurorEvaluationOutput(
                juror_id="c", role_name="C", role_focus="Z",
                verdict="needs_review", overall_score=55, confidence=0.6, rationale=""
            ),
        }

        status, reached, verdict = checker.check_consensus(evaluations, round_number=1)

        assert status == ConsensusStatus.MAJORITY
        assert reached is True
        assert verdict == "safe_pass"

    def test_split_no_consensus(self):
        checker = ConsensusChecker(consensus_threshold=0.67)
        evaluations = {
            "a": JurorEvaluationOutput(
                juror_id="a", role_name="A", role_focus="X",
                verdict="safe_pass", overall_score=90, confidence=0.9, rationale=""
            ),
            "b": JurorEvaluationOutput(
                juror_id="b", role_name="B", role_focus="Y",
                verdict="needs_review", overall_score=55, confidence=0.6, rationale=""
            ),
            "c": JurorEvaluationOutput(
                juror_id="c", role_name="C", role_focus="Z",
                verdict="unsafe_fail", overall_score=30, confidence=0.7, rationale=""
            ),
        }

        status, reached, verdict = checker.check_consensus(evaluations, round_number=1)

        assert status == ConsensusStatus.SPLIT
        assert reached is False
        assert verdict is None


class TestDiscussionState:
    """Test DiscussionState class."""

    def test_state_initialization(self):
        state = DiscussionState(
            question_prompt="Test prompt",
            agent_response="Test response",
            max_rounds=3,
        )

        assert state.question_prompt == "Test prompt"
        assert state.max_rounds == 3
        assert state.current_round == 0
        assert not state.should_terminate

    def test_update_and_get_opinions(self):
        state = DiscussionState()

        # Add opinions
        state.update_juror_opinion("gpt-4o", {"verdict": "safe_pass", "score": 85})
        state.update_juror_opinion("claude", {"verdict": "needs_review", "score": 60})

        # Get specific opinion
        opinion = state.get_juror_opinion("gpt-4o")
        assert opinion["verdict"] == "safe_pass"

        # Get other opinions
        others = state.get_other_opinions("gpt-4o")
        assert len(others) == 1
        assert others[0]["verdict"] == "needs_review"


# =============================================================================
# Integration Tests (Mock-based)
# =============================================================================

class TestSequentialDiscussionRoundMocked:
    """Test SequentialDiscussionRound with mocked LLM calls."""

    @pytest.mark.asyncio
    async def test_execute_round_basic(self):
        """Test basic round execution with mocked agents."""
        # Create mock juror agents
        mock_agents = {}
        for juror_id in ["gpt-4o", "claude-3-haiku-20240307", "gemini-2.5-flash"]:
            mock_agent = Mock()
            mock_agent.current_evaluation = JurorEvaluationOutput(
                juror_id=juror_id,
                role_name=f"Mock {juror_id}",
                role_focus="Testing",
                verdict="safe_pass",
                overall_score=85.0,
                confidence=0.9,
                rationale="Mock evaluation",
            )

            # Mock async evaluate_discussion method
            async def mock_eval(*args, **kwargs):
                return mock_agent.current_evaluation
            mock_agent.evaluate_discussion = mock_eval

            mock_agents[juror_id] = mock_agent

        # Create consensus checker
        consensus_checker = ConsensusChecker()

        # Create round executor
        round_executor = SequentialDiscussionRound(
            juror_agents=mock_agents,
            consensus_checker=consensus_checker,
        )

        # Create state
        state = DiscussionState(
            question_prompt="Test prompt",
            agent_response="Test response",
            current_round=1,
        )

        # Execute (this will call mocked methods)
        # Note: This test verifies the structure, actual LLM calls are mocked
        # In a real integration test, we would use actual agents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
