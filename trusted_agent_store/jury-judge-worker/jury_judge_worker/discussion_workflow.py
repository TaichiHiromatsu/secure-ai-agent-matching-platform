"""
Discussion Workflow for Jury Judge - Google ADK LoopAgent/SequentialAgent Implementation.

Implements the Phase 2 discussion workflow using Google ADK's workflow agents:
- SequentialAgent: Executes jurors in order (A → B → C → ConsensusCheck)
- LoopAgent: Repeats discussion rounds until consensus or max iterations

Design based on:
- https://google.github.io/adk-docs/agents/workflow-agents/loop-agents/
- https://google.github.io/adk-docs/agents/multi-agents/

State Sharing Pattern:
- Each juror writes to session.state via output_key
- Other jurors read from session.state via {key_name} placeholders
- ConsensusChecker sets escalate=True to terminate loop
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum

from .juror_agents import (
    BaseJurorAgent,
    JurorEvaluationOutput,
    create_juror_agents,
    JUROR_CONFIGS,
)
from .artifact_tools import (
    ArtifactContext,
    ArtifactToolsWithContext,
    generate_evaluation_summary,
)

logger = logging.getLogger(__name__)


class ConsensusStatus(str, Enum):
    """Consensus status after a discussion round."""
    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    SPLIT = "split"
    DEADLOCK = "deadlock"


def _truncate_at_sentence(text: str, max_length: int = 500) -> str:
    """
    Truncate text at sentence boundary to avoid cutting mid-sentence.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text ending at a sentence boundary if possible
    """
    if not text or len(text) <= max_length:
        return text or ""

    truncated = text[:max_length]

    # Find last sentence-ending punctuation
    last_period = max(
        truncated.rfind("。"),
        truncated.rfind("．"),
        truncated.rfind("."),
    )

    # If found and not too short, cut at sentence boundary
    if last_period > max_length * 0.5:
        return truncated[:last_period + 1]

    # Otherwise, just add ellipsis
    return truncated + "..."


@dataclass
class DiscussionState:
    """
    Shared state for the discussion workflow.

    This mimics Google ADK's session.state pattern.
    """
    # Artifact context for tool access
    artifact_context: Optional[ArtifactContext] = None

    # Evaluation summary (generated at start)
    evaluation_summary: str = ""

    # Original task data
    question_prompt: str = ""
    agent_response: str = ""

    # Current round number
    current_round: int = 0
    max_rounds: int = 3

    # Juror evaluations (keyed by juror_id)
    juror_opinions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Discussion history (all rounds)
    discussion_history: List[Dict[str, Any]] = field(default_factory=list)

    # Consensus tracking
    consensus_status: ConsensusStatus = ConsensusStatus.SPLIT
    consensus_reached: bool = False
    consensus_verdict: Optional[str] = None

    # Termination control (mimics ADK's escalate)
    should_terminate: bool = False
    termination_reason: Optional[str] = None

    def get_juror_opinion(self, juror_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific juror's current opinion."""
        return self.juror_opinions.get(juror_id)

    def update_juror_opinion(self, juror_id: str, opinion: Dict[str, Any]):
        """Update a juror's opinion in the state."""
        self.juror_opinions[juror_id] = opinion

    def get_other_opinions(self, exclude_juror_id: str) -> List[Dict[str, Any]]:
        """Get all opinions except the specified juror's."""
        return [
            op for jid, op in self.juror_opinions.items()
            if jid != exclude_juror_id
        ]


@dataclass
class DiscussionRoundResult:
    """Result of a single discussion round."""
    round_number: int
    statements: List[Dict[str, Any]]
    consensus_status: ConsensusStatus
    consensus_reached: bool
    consensus_verdict: Optional[str] = None


@dataclass
class DiscussionWorkflowResult:
    """Final result of the discussion workflow."""
    total_rounds: int
    final_evaluations: Dict[str, JurorEvaluationOutput]
    discussion_history: List[DiscussionRoundResult]
    consensus_status: ConsensusStatus
    consensus_reached: bool
    consensus_verdict: Optional[str] = None
    early_termination: bool = False
    termination_reason: Optional[str] = None


class ConsensusChecker:
    """
    Checks for consensus after each discussion round.

    In ADK terms, this would set escalate=True when consensus is reached.
    """

    def __init__(self, consensus_threshold: float = 0.67):
        """
        Initialize consensus checker.

        Args:
            consensus_threshold: Fraction of jurors needed for majority (default: 2/3)
        """
        self.consensus_threshold = consensus_threshold

    def check_consensus(
        self,
        evaluations: Dict[str, JurorEvaluationOutput],
        round_number: int,
    ) -> tuple[ConsensusStatus, bool, Optional[str]]:
        """
        Check if consensus has been reached.

        Args:
            evaluations: Current evaluations from all jurors
            round_number: Current round number

        Returns:
            Tuple of (status, reached, verdict)
        """
        if not evaluations:
            return ConsensusStatus.SPLIT, False, None

        # Count verdicts
        verdict_counts: Dict[str, int] = {}
        for eval_output in evaluations.values():
            v = eval_output.verdict
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

        total = len(evaluations)

        # Check for unanimous
        if len(verdict_counts) == 1:
            verdict = list(verdict_counts.keys())[0]
            return ConsensusStatus.UNANIMOUS, True, verdict

        # Check for majority
        max_count = max(verdict_counts.values())
        majority_verdicts = [v for v, c in verdict_counts.items() if c == max_count]

        if len(majority_verdicts) == 1:
            majority_fraction = max_count / total
            if majority_fraction >= self.consensus_threshold:
                return ConsensusStatus.MAJORITY, True, majority_verdicts[0]

        # No consensus
        return ConsensusStatus.SPLIT, False, None


class SequentialDiscussionRound:
    """
    Executes a single discussion round with sequential juror statements.

    In ADK terms, this is a SequentialAgent containing:
    - JurorA (speaks first, sees Phase 1 evaluations)
    - JurorB (speaks second, sees JurorA's statement)
    - JurorC (speaks third, sees JurorA and JurorB's statements)
    - ConsensusChecker (checks for agreement)
    """

    def __init__(
        self,
        juror_agents: Dict[str, BaseJurorAgent],
        consensus_checker: ConsensusChecker,
        sse_callback: Optional[Callable] = None,
    ):
        self.juror_agents = juror_agents
        self.consensus_checker = consensus_checker
        self.sse_callback = sse_callback
        # Define execution order
        self.execution_order = ["gpt-4o", "claude-3-haiku-20240307", "gemini-2.5-flash"]

    async def execute(
        self,
        state: DiscussionState,
    ) -> DiscussionRoundResult:
        """
        Execute one discussion round.

        Each juror speaks in sequence, seeing all previous statements.

        Args:
            state: Current discussion state

        Returns:
            DiscussionRoundResult for this round
        """
        round_number = state.current_round
        statements = []

        # Notify: round start
        await self._notify_sse({
            "type": "discussion_round_start",
            "round": round_number,
            "speaker_order": self.execution_order,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Execute each juror in sequence
        for juror_id in self.execution_order:
            agent = self.juror_agents.get(juror_id)
            if not agent:
                logger.warning(f"No agent found for juror {juror_id}")
                continue

            # Get other opinions (including statements from earlier in this round)
            other_opinions = state.get_other_opinions(juror_id)

            # Execute juror's turn
            logger.info(f"[Round {round_number}] {juror_id} speaking...")

            try:
                evaluation = await agent.evaluate_discussion(
                    round_number=round_number,
                    other_opinions=other_opinions,
                    question_prompt=state.question_prompt,
                    agent_response=state.agent_response,
                )

                # Update state with this juror's opinion
                opinion_dict = evaluation.to_dict()
                state.update_juror_opinion(juror_id, opinion_dict)

                # Record statement
                statement = {
                    "juror_id": juror_id,
                    "round_number": round_number,
                    "statement_order": len(statements),
                    **opinion_dict,
                }
                statements.append(statement)

                # Notify: juror statement
                # Note: フロントエンドはDiscussion Phaseで"statement"フィールドを期待する
                truncated_rationale = _truncate_at_sentence(evaluation.rationale)
                await self._notify_sse({
                    "type": "juror_statement",
                    "phase": "discussion",
                    "round": round_number,
                    "juror": juror_id,
                    "role_name": evaluation.role_name,
                    "verdict": evaluation.verdict,
                    "score": evaluation.overall_score,
                    "rationale": truncated_rationale,
                    "statement": truncated_rationale,  # フロントエンド互換性のため追加
                    "position_changed": evaluation.position_changed,
                    "timestamp": datetime.utcnow().isoformat(),
                })

            except Exception as e:
                logger.error(f"Error executing juror {juror_id}: {e}")
                # Keep previous evaluation if available
                continue

        # Check consensus
        current_evaluations = {
            jid: agent.current_evaluation
            for jid, agent in self.juror_agents.items()
            if agent.current_evaluation
        }

        status, reached, verdict = self.consensus_checker.check_consensus(
            current_evaluations, round_number
        )

        # Notify: consensus check
        await self._notify_sse({
            "type": "consensus_check",
            "round": round_number,
            "consensus_status": status.value,
            "consensus_reached": reached,
            "consensus_verdict": verdict,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return DiscussionRoundResult(
            round_number=round_number,
            statements=statements,
            consensus_status=status,
            consensus_reached=reached,
            consensus_verdict=verdict,
        )

    async def _notify_sse(self, data: Dict[str, Any]):
        """Send SSE notification if callback is available."""
        if self.sse_callback:
            try:
                if asyncio.iscoroutinefunction(self.sse_callback):
                    await self.sse_callback(data)
                else:
                    self.sse_callback(data)
            except Exception as e:
                logger.warning(f"SSE notification failed: {e}")


class DiscussionLoopWorkflow:
    """
    Main discussion workflow using loop pattern.

    In ADK terms, this is a LoopAgent that:
    - Contains a SequentialAgent (SequentialDiscussionRound)
    - Terminates when consensus is reached (escalate=True)
    - Has a maximum iteration limit (max_iterations)
    """

    def __init__(
        self,
        juror_agents: Dict[str, BaseJurorAgent],
        max_rounds: int = 3,
        consensus_threshold: float = 0.67,
        sse_callback: Optional[Callable] = None,
    ):
        """
        Initialize the discussion workflow.

        Args:
            juror_agents: Dictionary of juror agents
            max_rounds: Maximum discussion rounds (like ADK's max_iterations)
            consensus_threshold: Threshold for majority consensus
            sse_callback: Callback for SSE notifications
        """
        self.juror_agents = juror_agents
        self.max_rounds = max_rounds
        self.sse_callback = sse_callback

        self.consensus_checker = ConsensusChecker(consensus_threshold)
        self.sequential_round = SequentialDiscussionRound(
            juror_agents=juror_agents,
            consensus_checker=self.consensus_checker,
            sse_callback=sse_callback,
        )

    async def execute(
        self,
        initial_evaluations: Dict[str, JurorEvaluationOutput],
        artifact_context: ArtifactContext,
        evaluation_summary: str,
        question_prompt: str,
        agent_response: str,
    ) -> DiscussionWorkflowResult:
        """
        Execute the full discussion workflow.

        This is the main entry point that mimics ADK's LoopAgent behavior:
        1. Initialize state with Phase 1 evaluations
        2. Loop through discussion rounds (SequentialAgent)
        3. Check for termination (escalate) after each round
        4. Return final results

        Args:
            initial_evaluations: Phase 1 evaluation results
            artifact_context: Context for artifact tool access
            evaluation_summary: Summary text for juror context
            question_prompt: Original prompt to the agent
            agent_response: Agent's response

        Returns:
            DiscussionWorkflowResult with all discussion data
        """
        # Initialize state (like ADK's session.state)
        state = DiscussionState(
            artifact_context=artifact_context,
            evaluation_summary=evaluation_summary,
            question_prompt=question_prompt,
            agent_response=agent_response,
            max_rounds=self.max_rounds,
        )

        # Initialize juror opinions from Phase 1
        for juror_id, eval_output in initial_evaluations.items():
            state.update_juror_opinion(juror_id, eval_output.to_dict())

        # Also update agent's internal state
        for juror_id, agent in self.juror_agents.items():
            if juror_id in initial_evaluations:
                agent._current_evaluation = initial_evaluations[juror_id]

        # Notify: discussion phase start
        await self._notify_sse({
            "type": "phase_change",
            "phase": "discussion",
            "phaseNumber": 2,
            "description": "ディスカッション",
            "max_rounds": self.max_rounds,
            "timestamp": datetime.utcnow().isoformat(),
        })

        discussion_history: List[DiscussionRoundResult] = []

        # Main loop (like ADK's LoopAgent)
        for round_num in range(1, self.max_rounds + 1):
            state.current_round = round_num

            logger.info(f"[Discussion] Starting round {round_num}/{self.max_rounds}")

            # Execute sequential round
            round_result = await self.sequential_round.execute(state)
            discussion_history.append(round_result)

            # Update state with consensus result
            state.consensus_status = round_result.consensus_status
            state.consensus_reached = round_result.consensus_reached
            state.consensus_verdict = round_result.consensus_verdict

            # Check termination (like ADK's escalate)
            # 最低2ラウンドは実行（最初の発言者も他の意見を見る機会を与える）
            MIN_ROUNDS = 2
            # 終了条件: 全員がposition_changed=False（議論が収束した状態）
            all_unchanged = all(
                not s.get("position_changed", True)
                for s in round_result.statements
            )
            if all_unchanged and round_num >= MIN_ROUNDS:
                state.should_terminate = True
                state.termination_reason = "no_position_changes"
                logger.info(f"[Discussion] All positions unchanged at round {round_num}, terminating")
                break

        # Collect final evaluations
        final_evaluations = {
            jid: agent.current_evaluation
            for jid, agent in self.juror_agents.items()
            if agent.current_evaluation
        }

        return DiscussionWorkflowResult(
            total_rounds=state.current_round,
            final_evaluations=final_evaluations,
            discussion_history=discussion_history,
            consensus_status=state.consensus_status,
            consensus_reached=state.consensus_reached,
            consensus_verdict=state.consensus_verdict,
            early_termination=state.should_terminate,
            termination_reason=state.termination_reason,
        )

    async def _notify_sse(self, data: Dict[str, Any]):
        """Send SSE notification if callback is available."""
        if self.sse_callback:
            try:
                if asyncio.iscoroutinefunction(self.sse_callback):
                    await self.sse_callback(data)
                else:
                    self.sse_callback(data)
            except Exception as e:
                logger.warning(f"SSE notification failed: {e}")


# =============================================================================
# High-level API
# =============================================================================

async def run_discussion_workflow(
    initial_evaluations: Dict[str, JurorEvaluationOutput],
    artifact_context: ArtifactContext,
    evaluation_summary: str,
    question_prompt: str,
    agent_response: str,
    max_rounds: int = 3,
    consensus_threshold: float = 0.67,
    sse_callback: Optional[Callable] = None,
) -> DiscussionWorkflowResult:
    """
    High-level function to run the discussion workflow.

    Args:
        initial_evaluations: Phase 1 results (keyed by juror_id)
        artifact_context: Context for tool access
        evaluation_summary: Summary for juror context
        question_prompt: Original prompt
        agent_response: Agent's response
        max_rounds: Maximum discussion rounds
        consensus_threshold: Threshold for consensus
        sse_callback: SSE notification callback

    Returns:
        DiscussionWorkflowResult
    """
    # Create juror agents
    juror_ids = list(initial_evaluations.keys())
    artifact_tools = ArtifactToolsWithContext(artifact_context)
    juror_agents = create_juror_agents(
        juror_ids=juror_ids,
        artifact_tools=artifact_tools,
        sse_callback=sse_callback,
    )

    # Create and run workflow
    workflow = DiscussionLoopWorkflow(
        juror_agents=juror_agents,
        max_rounds=max_rounds,
        consensus_threshold=consensus_threshold,
        sse_callback=sse_callback,
    )

    return await workflow.execute(
        initial_evaluations=initial_evaluations,
        artifact_context=artifact_context,
        evaluation_summary=evaluation_summary,
        question_prompt=question_prompt,
        agent_response=agent_response,
    )
