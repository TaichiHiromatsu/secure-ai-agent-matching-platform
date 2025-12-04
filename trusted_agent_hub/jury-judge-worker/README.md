# Inspect Worker - Jury Judge

**Inspect Worker** は、Agents-as-a-Judge方式による高精度なエージェント評価を実装します。
3人の審査エージェント（Juror）による協調評価で信頼性の高い判定を実現します。

## 🎯 概要

### Jury Judge評価
**評価基準** (AISI Inspect準拠):
- **Task Completion** (0-40点): タスク完了度
- **Tool Usage** (0-30点): ツール活用能力
- **Autonomy** (0-20点): 自律性
- **Safety** (0-10点): 安全性

**3人の審査エージェント（Juror）**:
- Juror 1: GPT-4o (OpenAI)
- Juror 2: Claude 3.5 Sonnet (Anthropic)
- Juror 3: Gemini 2.5 Flash (Google)

**協調評価プロセス**:
1. **Phase 1 - Independent Evaluation（独立評価）**: 各Jurorが全シナリオを独立に評価
2. **Phase 2 - Discussion（議論）**: Jurorが評価を共有し、異なる観点を議論（最大3ラウンド）
3. **Phase 3 - Final Judgment（最終判定）**: コンセンサスまたは多数決で最終スコアを決定

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

### 協調評価フェーズの詳細

1. **Phase 1 - Independent Evaluation（独立評価）**
   - 各Jurorが全シナリオを独立に評価（並列実行）
   - Google ADK経由でGemini、Anthropic Computer Use経由でClaude、OpenAI API経由でGPT-4oを呼び出し
   - 各Jurorは Task Completion、Tool Usage、Autonomy、Safety の4軸でスコアリング

2. **Phase 2 - Discussion（議論）**
   - Juror間で評価結果を共有し、意見の相違点を議論
   - 最大3ラウンドの議論を通じて、各Jurorが評価を再検討
   - コンセンサス（全員一致）または停滞（意見が変わらない）を検出

3. **Phase 3 - Final Judgment（最終判定）**
   - コンセンサスが得られた場合: 合意された評価を採用
   - コンセンサスが得られない場合: 多数決または重み付き平均で最終スコアを決定
   - 最終的な Trust Score を算出し、WebSocket経由でリアルタイム更新

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
  "juror_evaluations": {
    "juror_1": {"score": 85, "verdict": "approve", "rationale": "..."},
    "juror_2": {"score": 78, "verdict": "approve", "rationale": "..."},
    "juror_3": {"score": 82, "verdict": "approve", "rationale": "..."}
  },
  "discussion_rounds": [
    {"round": 1, "statements": [...], "consensus_reached": false},
    {"round": 2, "statements": [...], "consensus_reached": true}
  ],
  "final_score": 82,
  "breakdown": {
    "task_completion": 33,
    "tool_usage": 25,
    "autonomy": 16,
    "safety": 8
  }
}
```

## 📈 W&B Weave統合

全評価プロセスをW&B Weaveでトレース:
- **Phase 1 - Independent Evaluation**: 各Jurorの独立評価
- **Phase 2 - Discussion**: ラウンドごとの議論内容と評価の変化
- **Phase 3 - Final Judgment**: 最終判定プロセスと合意形成
- **Final Scores**: 統合スコアと信頼度

submission詳細ページから「📊 View in W&B Weave」リンクでアクセス可能。

## 🔗 統合

Trusted Agent Storeの`app/routers/submissions.py`から呼び出されます:
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
