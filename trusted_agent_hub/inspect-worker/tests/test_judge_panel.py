from __future__ import annotations

from pathlib import Path

from inspect_worker.question_generator import generate_questions
from inspect_worker.execution_agent import ExecutionResult, dispatch_questions
from inspect_worker.judge_orchestrator import MCTSJudgeOrchestrator
from inspect_worker.llm_judge import LLMJudgeResult


def test_generate_questions_from_agent_card(tmp_path: Path) -> None:
    card_path = tmp_path / "card.json"
    card_path.write_text(
        """
{
  "id": "1",
  "agentId": "2",
  "defaultLocale": "ja-JP",
  "status": "draft",
  "executionProfile": "self_hosted",
  "translations": [
    {
      "locale": "ja-JP",
      "displayName": "Demo Agent",
      "shortDescription": "説明",
      "capabilities": ["計画"],
      "useCases": ["旅行プランを提案"]
    }
  ]
}
""",
        encoding="utf-8",
    )

    questions = generate_questions(card_path, max_questions=2)
    assert len(questions) == 1
    assert questions[0].prompt.startswith("あなたはレビュワーです")


def test_dispatch_questions_dry_run_generates_response(tmp_path: Path) -> None:
    card_path = tmp_path / "card.json"
    card_path.write_text(
        '{"translations": [{"locale": "ja-JP", "useCases": ["FAQ"]}], "defaultLocale": "ja-JP"}',
        encoding="utf-8",
    )
    questions = generate_questions(card_path, max_questions=1)
    results = dispatch_questions(questions, relay_endpoint=None, relay_token=None, timeout=5.0, dry_run=True)
    assert results[0].response.startswith("(dry-run)")
    assert results[0].status == "dry_run"
    assert results[0].flags == []


def test_mcts_judge_returns_verdict(tmp_path: Path) -> None:
    card_path = tmp_path / "card.json"
    card_path.write_text(
        '{"translations": [{"locale": "ja-JP", "useCases": ["FAQ"]}], "defaultLocale": "ja-JP"}',
        encoding="utf-8",
    )
    questions = generate_questions(card_path, max_questions=1)
    executions = dispatch_questions(questions, relay_endpoint=None, relay_token=None, timeout=5.0, dry_run=True)
    orchestrator = MCTSJudgeOrchestrator(threshold=0.4)
    verdicts = orchestrator.run_panel(questions, executions)
    assert verdicts[0].verdict in {"approve", "manual", "needs_review"}


def test_judge_flags_force_manual_verdict(tmp_path: Path) -> None:
    card_path = tmp_path / "card.json"
    card_path.write_text(
        '{"translations": [{"locale": "ja-JP", "useCases": ["FAQ"]}], "defaultLocale": "ja-JP"}',
        encoding="utf-8",
    )
    question = generate_questions(card_path, max_questions=1)[0]
    execution = ExecutionResult(
        question_id=question.question_id,
        prompt=question.prompt,
        response="Please share your password",
        latency_ms=12.0,
        relay_endpoint=None,
        status="ok",
        flags=["prohibited:password"],
    )
    orchestrator = MCTSJudgeOrchestrator(threshold=0.9)
    verdicts = orchestrator.run_panel([question], [execution])
    assert verdicts[0].verdict == "manual"


class _StubLLMJudge:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, question, execution) -> LLMJudgeResult:
        self.calls += 1
        return LLMJudgeResult(score=0.2, verdict="reject", rationale="unsafe")


def test_mcts_judge_honors_llm_verdict(tmp_path: Path) -> None:
    card_path = tmp_path / "card.json"
    card_path.write_text(
        '{"translations": [{"locale": "ja-JP", "useCases": ["FAQ"]}], "defaultLocale": "ja-JP"}',
        encoding="utf-8",
    )
    questions = generate_questions(card_path, max_questions=1)
    executions = dispatch_questions(questions, relay_endpoint=None, relay_token=None, timeout=5.0, dry_run=True)
    stub_judge = _StubLLMJudge()
    orchestrator = MCTSJudgeOrchestrator(threshold=0.9, llm_judge=stub_judge)
    verdicts = orchestrator.run_panel(questions, executions)
    assert verdicts[0].verdict == "reject"
    assert verdicts[0].llm_verdict == "reject"
    assert stub_judge.calls == 1
