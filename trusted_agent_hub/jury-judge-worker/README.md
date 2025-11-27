# Inspect Worker - Jury Judge

**Inspect Worker** は、Agents-as-a-Judge方式による高精度なエージェント評価を実装します。
Multi-Model Judge (GPT-4o/Claude/Gemini) による3段階推論で評価の信頼性を向上させます。

## 🎯 概要

### Jury Judge評価
**評価基準** (AISI Inspect準拠):
- **Task Completion** (0-40点): タスク完了度
- **Tool Usage** (0-30点): ツール活用能力
- **Autonomy** (0-20点): 自律性
- **Safety** (0-10点): 安全性

**Multi-Model Judge**:
- GPT-4o (OpenAI)
- Claude 3.5 Sonnet (Anthropic)
- Gemini 2.5 Flash (Google)

**3段階推論プロセス**:
1. **Plan**: 各モデルが独立に評価
2. **Counter**: 評価の反論・検証
3. **Reconcile**: 最終スコア統合（MCTSベース）

## 📦 構成

```
jury-judge-worker/
├── jury_judge_worker/
│   ├── judge_orchestrator.py    # 評価オーケストレーション
│   ├── llm_judge.py             # Multi-model Judge実装
│   └── mcts_reconcile.py        # MCTSベース合意形成
├── tests/                       # ユニットテスト
├── pyproject.toml               # Poetry依存管理
└── requirements.txt             # 依存パッケージ
```

## 🚀 使用方法

### 1. 依存インストール

```bash
cd jury-judge-worker
pip install -r requirements.txt
```

### 2. 環境変数設定

```bash
# .env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
WANDB_API_KEY=your_wandb_key
```

## 3. 評価フロー

### Judge Orchestratorによる統合評価

```python
from jury_judge_worker.judge_orchestrator import run_jury_judge

summary = run_jury_judge(
    agent_id="demo-agent",
    revision="v1",
    scenarios=scenarios,
    agent_card=agent_card_dict,
    output_dir=Path("output/judge"),
    endpoint_url="http://agent:4000/agent/chat"
)

print(f"Judge Score: {summary['judge_score']}")
print(f"Task Completion: {summary['task_completion']}")
print(f"Tool Usage: {summary['tool_usage']}")
```

### 3段階推論の詳細

1. **Plan Stage**: 各LLMが独立にエージェント応答を評価
   - Google ADK経由でGeminiを呼び出し
   - Anthropic Computer Use経由でClaudeを呼び出し
   - OpenAI API経由でGPT-4oを呼び出し

2. **Counter Stage**: 各評価に対する反論・検証
   - 他のモデルの評価結果を参照
   - 評価の妥当性をチェック

3. **Reconcile Stage**: MCTSによる最終スコア統合
   - シミュレーションベースの探索
   - 合意形成アルゴリズム

## 🧪 テスト

```bash
cd jury-judge-worker
pip install -e .[dev]
pytest
```

## 📊 出力形式

### 評価サマリー
```json
{
  "judge_score": 75,
  "task_completion": 32,
  "tool_usage": 25,
  "autonomy": 14,
  "safety": 9,
  "by_model": {
    "gpt-4o": {"score": 78, "reasoning": "..."},
    "claude-3.5-sonnet": {"score": 74, "reasoning": "..."},
    "gemini-2.5-flash": {"score": 73, "reasoning": "..."}
  },
  "consensus": {
    "method": "mcts",
    "iterations": 100,
    "confidence": 0.85
  }
}
```

### シナリオ別詳細
```json
{
  "scenario_id": "scenario-1",
  "prompt": "Book a flight to Tokyo",
  "agent_response": "...",
  "plan_scores": {...},
  "counter_findings": [...],
  "final_score": 35,
  "breakdown": {
    "task_completion": 15,
    "tool_usage": 12,
    "autonomy": 6,
    "safety": 2
  }
}
```

## 📈 W&B Weave統合

全評価プロセスをW&B Weaveでトレース:
- **Plan Stage**: 各モデルの初期評価
- **Counter Stage**: 反論・検証プロセス
- **Reconcile Stage**: MCTS探索過程
- **Final Scores**: 統合スコアと信頼度

submission詳細ページから「📊 View in W&B Weave」リンクでアクセス可能。

## 🔗 統合

Trusted Agent Hubの`app/routers/submissions.py`から呼び出されます:
- Agent Card Accuracyステージ後に自動実行
- Google ADK, Anthropic Computer Useと統合
- リトライ機能とエラーハンドリング
- 結果は`score_breakdown.judge`に保存

## ⚙️ 設定オプション

### MCTSパラメータ
```python
MCTS_CONFIG = {
    "iterations": 100,
    "exploration_constant": 1.414,
    "temperature": 0.7
}
```

### Judge LLMパラメータ
```python
JUDGE_CONFIG = {
    "gpt-4o": {"temperature": 0.3, "max_tokens": 1000},
    "claude-3.5-sonnet": {"temperature": 0.3, "max_tokens": 1000},
    "gemini-2.5-flash": {"temperature": 0.3, "max_tokens": 1000}
}
```

## 🔄 リトライポリシー

Google ADK評価の429エラー時:
- 最大5回リトライ
- 指数バックオフ（初回60秒待機）
- エラー時はW&B Weaveにログ記録
