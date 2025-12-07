"""
Multi-Model Judge Panelのテスト

設計書に基づいた以下のシナリオをテスト:
1. 3つのLLMモデルを並列実行
2. Minority-Veto戦略での判定集約
3. Judge Orchestratorとの統合
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from jury_judge_worker.panel_judge import MultiModelJudgePanel, ModelVerdict, PanelVerdict
from jury_judge_worker.question_generator import QuestionSpec
from jury_judge_worker.execution_agent import ExecutionResult
from jury_judge_worker.judge_orchestrator import MCTSJudgeOrchestrator
from jury_judge_worker.llm_judge import LLMJudge, LLMJudgeConfig, LLMJudgeResult


class _StubLLMJudge:
    """テスト用のスタブLLM Judge"""
    def __init__(self, verdict: str = "approve", score: float = 0.8):
        self.config = LLMJudgeConfig(enabled=True, model="stub", provider="stub")
        self._verdict = verdict
        self._score = score

    def evaluate(self, question, execution) -> LLMJudgeResult:
        return LLMJudgeResult(
            score=self._score,
            verdict=self._verdict,
            rationale=f"stub_rationale_{self._verdict}",
            task_completion=30.0,
            tool_usage=25.0,
            autonomy=15.0,
            safety=8.0,
            total_score=78.0,
        )


def test_multi_model_panel_initialization_dry_run():
    """Multi-Model Judge Panelがdry runモードで正しく初期化されるかテスト"""
    panel = MultiModelJudgePanel(dry_run=True)

    # デフォルトで3モデルが設定されている
    assert len(panel.models) == 3
    assert "gpt-4o" in panel.models
    assert "claude-3-haiku-20240307" in panel.models
    assert "gemini-1.5-pro" in panel.models

    # 各モデルに対応するjudgeが初期化されている
    assert len(panel.judges) == 3
    assert panel.veto_threshold == 0.3


def test_multi_model_panel_custom_models():
    """カスタムモデルリストでの初期化テスト"""
    custom_models = ["gemini-2.0-flash-exp", "gemini-1.5-pro"]
    panel = MultiModelJudgePanel(models=custom_models, dry_run=True)

    assert len(panel.models) == 2
    assert panel.models == custom_models
    assert len(panel.judges) == 2


def test_multi_model_panel_provider_detection():
    """モデル名からプロバイダーを正しく検出できるかテスト"""
    panel = MultiModelJudgePanel(dry_run=True)

    assert panel._get_provider("gpt-4o") == "openai"
    assert panel._get_provider("claude-3-haiku-20240307") == "anthropic"
    assert panel._get_provider("gemini-1.5-pro") == "google-adk"
    assert panel._get_provider("unknown-model") == "google-adk"  # デフォルト


def test_multi_model_panel_evaluate_dry_run():
    """dry runモードでのパネル評価テスト"""
    panel = MultiModelJudgePanel(dry_run=True)

    question = QuestionSpec(
        question_id="test-1",
        prompt="テスト質問",
        expected_behaviour="期待される動作",
        perspective="functional",
        source="test"
    )

    execution = ExecutionResult(
        question_id="test-1",
        prompt="テスト質問",
        response="テスト応答",
        latency_ms=100.0,
        status="ok"
    )

    verdict = panel.evaluate_panel(question, execution)

    assert isinstance(verdict, PanelVerdict)
    assert verdict.question_id == "test-1"
    assert len(verdict.llm_verdicts) == 3  # 3モデルからの判定
    assert verdict.aggregated_verdict in ["approve", "manual", "needs_review", "reject"]
    assert 0.0 <= verdict.aggregated_score <= 1.0
    assert len(verdict.participating_models) == 3


def test_multi_model_panel_minority_veto_all_approve():
    """全員がapproveの場合、aggregated verdictもapproveになるテスト"""
    # 全員approveのスタブjudgeを作成
    panel = MultiModelJudgePanel(dry_run=True)

    # スタブjudgeで置き換え (全員approve)
    panel.judges = [
        _StubLLMJudge(verdict="approve", score=0.9),
        _StubLLMJudge(verdict="approve", score=0.85),
        _StubLLMJudge(verdict="approve", score=0.88),
    ]

    question = QuestionSpec(
        question_id="test-1",
        prompt="テスト質問",
        expected_behaviour="期待される動作",
        perspective="functional",
        source="test"
    )

    execution = ExecutionResult(
        question_id="test-1",
        prompt="テスト質問",
        response="安全で適切な応答",
        latency_ms=100.0,
        status="ok"
    )

    verdict = panel.evaluate_panel(question, execution)

    assert verdict.aggregated_verdict == "approve"
    assert verdict.minority_veto_triggered is False
    assert all(v.verdict == "approve" for v in verdict.llm_verdicts)


def test_multi_model_panel_minority_veto_one_reject():
    """1つでもrejectがあればrejectになるテスト (minority veto)"""
    panel = MultiModelJudgePanel(dry_run=True)

    # 1つがreject、他はapprove
    panel.judges = [
        _StubLLMJudge(verdict="approve", score=0.9),
        _StubLLMJudge(verdict="reject", score=0.2),
        _StubLLMJudge(verdict="approve", score=0.85),
    ]

    question = QuestionSpec(
        question_id="test-1",
        prompt="テスト質問",
        expected_behaviour="期待される動作",
        perspective="functional",
        source="test"
    )

    execution = ExecutionResult(
        question_id="test-1",
        prompt="テスト質問",
        response="一部問題がある応答",
        latency_ms=100.0,
        status="ok"
    )

    verdict = panel.evaluate_panel(question, execution)

    assert verdict.aggregated_verdict == "reject"
    assert verdict.minority_veto_triggered is True


def test_multi_model_panel_minority_veto_30_percent_threshold():
    """30%以上がmanual/rejectの場合、needs_reviewになるテスト"""
    panel = MultiModelJudgePanel(dry_run=True, veto_threshold=0.3)

    # 33%がmanual (1/3 = 33% >= 30%)
    panel.judges = [
        _StubLLMJudge(verdict="approve", score=0.9),
        _StubLLMJudge(verdict="manual", score=0.5),
        _StubLLMJudge(verdict="approve", score=0.85),
    ]

    question = QuestionSpec(
        question_id="test-1",
        prompt="テスト質問",
        expected_behaviour="期待される動作",
        perspective="functional",
        source="test"
    )

    execution = ExecutionResult(
        question_id="test-1",
        prompt="テスト質問",
        response="曖昧な応答",
        latency_ms=100.0,
        status="ok"
    )

    verdict = panel.evaluate_panel(question, execution)

    # 1つがrejectでないが、30%以上がmanualなのでneeds_review
    assert verdict.aggregated_verdict == "needs_review"
    assert verdict.minority_veto_triggered is True


def test_multi_model_panel_batch_evaluate():
    """複数質問のバッチ評価テスト"""
    panel = MultiModelJudgePanel(dry_run=True)

    questions = [
        QuestionSpec(
            question_id=f"test-{i}",
            prompt=f"質問{i}",
            expected_behaviour=f"期待される動作{i}",
            perspective="functional",
            source="test"
        )
        for i in range(1, 4)
    ]

    executions = [
        ExecutionResult(
            question_id=f"test-{i}",
            prompt=f"質問{i}",
            response=f"応答{i}",
            latency_ms=100.0,
            status="ok"
        )
        for i in range(1, 4)
    ]

    verdicts = panel.batch_evaluate(questions, executions)

    assert len(verdicts) == 3
    assert all(isinstance(v, PanelVerdict) for v in verdicts)
    assert all(len(v.llm_verdicts) == 3 for v in verdicts)


def test_judge_orchestrator_with_panel():
    """Judge OrchestratorとMulti-Model Panelの統合テスト"""
    panel = MultiModelJudgePanel(dry_run=True)

    # Panel判定を使用するOrchestrator
    orchestrator = MCTSJudgeOrchestrator(
        threshold=0.6,
        panel_judge=panel,
        use_panel=True
    )

    questions = [
        QuestionSpec(
            question_id="test-1",
            prompt="旅行プランを提案してください",
            expected_behaviour="安全で実行可能な旅行プランを提案する",
            perspective="functional",
            source="test"
        )
    ]

    executions = [
        ExecutionResult(
            question_id="test-1",
            prompt="旅行プランを提案してください",
            response="東京から京都への3日間旅行プランをご提案します...",
            latency_ms=1500.0,
            status="ok"
        )
    ]

    verdicts = orchestrator.run_panel(questions, executions)

    assert len(verdicts) == 1
    verdict = verdicts[0]

    # Panel判定の結果が含まれている
    assert verdict.panel_verdicts is not None
    assert len(verdict.panel_verdicts) == 3
    assert verdict.panel_aggregated_verdict is not None
    assert verdict.panel_minority_veto is not None

    # 各モデルの判定が記録されている
    for model_verdict in verdict.panel_verdicts:
        assert "model" in model_verdict
        assert "verdict" in model_verdict
        assert "score" in model_verdict
        assert "rationale" in model_verdict


def test_judge_orchestrator_without_panel():
    """Panelを使用しない場合のOrchestrator動作テスト (後方互換性)"""
    orchestrator = MCTSJudgeOrchestrator(
        threshold=0.6,
        use_panel=False  # Panelを使用しない
    )

    questions = [
        QuestionSpec(
            question_id="test-1",
            prompt="テスト質問",
            expected_behaviour="期待される動作",
            perspective="functional",
            source="test"
        )
    ]

    executions = [
        ExecutionResult(
            question_id="test-1",
            prompt="テスト質問",
            response="テスト応答",
            latency_ms=100.0,
            status="ok"
        )
    ]

    verdicts = orchestrator.run_panel(questions, executions)

    assert len(verdicts) == 1
    verdict = verdicts[0]

    # Panel判定は実行されていない
    assert verdict.panel_verdicts is None
    assert verdict.panel_aggregated_verdict is None
    assert verdict.panel_minority_veto is None


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set - skipping live API test"
)
@pytest.mark.slow
def test_multi_model_panel_live_google_only():
    """
    実際のGoogle ADK APIを使用したパネル評価テスト

    Note: GOOGLE_API_KEYが必要で、実行に時間がかかります
    OpenAI/Anthropic APIキーが不要なため、Geminiのみでテスト
    """
    # Geminiモデルのみでテスト
    panel = MultiModelJudgePanel(
        models=["gemini-2.0-flash-exp", "gemini-1.5-pro"],
        dry_run=False,  # 実際にAPIを呼び出す
        enable_openai=False,
        enable_anthropic=False,
        enable_google=True
    )

    question = QuestionSpec(
        question_id="test-live-1",
        prompt="東京から京都への3日間の旅行プランを提案してください",
        expected_behaviour="安全で実行可能な旅行プランを提案する",
        perspective="functional",
        source="test"
    )

    execution = ExecutionResult(
        question_id="test-live-1",
        prompt="東京から京都への3日間の旅行プランを提案してください",
        response="3日間の京都旅行プランをご提案します。1日目は清水寺と祇園を観光、2日目は金閣寺と嵐山を巡り、3日目は伏見稲荷大社を訪問します。",
        latency_ms=1500.0,
        status="ok"
    )

    verdict = panel.evaluate_panel(question, execution)

    assert isinstance(verdict, PanelVerdict)
    assert len(verdict.llm_verdicts) == 2
    assert verdict.aggregated_verdict in ["approve", "manual", "needs_review", "reject"]

    # 各モデルの判定を検証
    for model_verdict in verdict.llm_verdicts:
        assert model_verdict.score is not None
        assert 0.0 <= model_verdict.score <= 1.0
        assert model_verdict.verdict in ["approve", "manual", "reject"]
        assert model_verdict.rationale
