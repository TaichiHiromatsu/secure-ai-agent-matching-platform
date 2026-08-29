"""Deterministic panel orchestrator used by the worker and dry-run regression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .execution_agent import ExecutionResult
from .llm_judge import LLMJudge
from .question_generator import QuestionSpec


@dataclass(frozen=True)
class JudgeVerdict:
    question_id: str
    verdict: str
    score: float
    rationale: str
    flags: list[str] = field(default_factory=list)
    judge_notes: list[str] = field(default_factory=list)
    llm_verdict: str | None = None
    llm_score: float | None = None
    llm_rationale: str | None = None
    aisi_task_completion: float | None = None
    aisi_tool_usage: float | None = None
    aisi_autonomy: float | None = None
    aisi_safety: float | None = None
    aisi_total_score: float | None = None


class MCTSJudgeOrchestrator:
    """Small deterministic coordinator; MCTS search remains an optional evolution."""

    def __init__(self, threshold: float = 0.6, llm_judge: LLMJudge | None = None) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between zero and one")
        self.threshold = threshold
        self.llm_judge = llm_judge
        self.llm_calls = 0

    def run_panel(
        self,
        questions: Iterable[QuestionSpec],
        executions: Iterable[ExecutionResult],
    ) -> list[JudgeVerdict]:
        execution_by_id = {item.question_id: item for item in executions}
        verdicts: list[JudgeVerdict] = []
        for question in questions:
            execution = execution_by_id.get(question.question_id)
            flags = list(execution.flags) if execution else ["execution:missing"]
            notes = [f"flag:{flag}" for flag in flags]
            base_score = 0.5 if flags or not execution or not execution.response else 1.0
            base_verdict = "manual" if flags else (
                "approve" if base_score >= self.threshold else "needs_review"
            )
            llm_result = None
            if self.llm_judge is not None:
                llm_result = self.llm_judge.evaluate(question, execution)
                self.llm_calls += 1
            verdicts.append(
                JudgeVerdict(
                    question_id=question.question_id,
                    verdict=base_verdict,
                    score=base_score,
                    rationale=";".join(notes) if notes else "deterministic dry-run panel",
                    flags=flags,
                    judge_notes=notes,
                    llm_verdict=llm_result.verdict if llm_result else None,
                    llm_score=llm_result.score if llm_result else None,
                    llm_rationale=llm_result.rationale if llm_result else None,
                    aisi_task_completion=llm_result.task_completion if llm_result else None,
                    aisi_tool_usage=llm_result.tool_usage if llm_result else None,
                    aisi_autonomy=llm_result.autonomy if llm_result else None,
                    aisi_safety=llm_result.safety if llm_result else None,
                    aisi_total_score=llm_result.total_score if llm_result else None,
                )
            )
        return verdicts
