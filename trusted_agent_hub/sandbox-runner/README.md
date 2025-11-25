# Sandbox Runner

**Sandbox Runner** は、エージェントのセキュリティと機能精度を評価する審査エンジンです。
Security GateとFunctional Accuracyの2つの評価モジュールを提供します。

## 🎯 概要

### Security Gate
AISI Security v0.1ベンチマークによる実攻撃シミュレーション:
- **システムプロンプト流出攻撃** (G6-5)
- **セーフガード回避攻撃** (G6-6): 有害コンテンツ生成要求
- **データソース**: `third_party/aisev/backend/dataset/output/06_aisi_security_v0.1.csv`

### Functional Accuracy
Agent Cardの`skills`に基づく機能テスト:
- シナリオベース評価
- セマンティック類似度による応答品質チェック
- スキルカバレッジ計測

## 📦 構成

```
sandbox-runner/
├── src/sandbox_runner/
│   ├── security_gate.py         # Security Gate評価
│   ├── functional_accuracy.py   # Functional Accuracy評価
│   └── cli.py                   # CLI エントリーポイント
├── tests/                       # ユニットテスト
├── pyproject.toml               # Poetry依存管理
└── Dockerfile                   # コンテナイメージ定義
```

## 🚀 使用方法

### インストール

```bash
cd sandbox-runner
pip install -e .
```

### Security Gate実行

```python
from sandbox_runner.security_gate import run_security_gate

summary = run_security_gate(
    agent_id="demo-agent",
    revision="v1",
    dataset_path=Path("third_party/aisev/backend/dataset/output/06_aisi_security_v0.1.csv"),
    output_dir=Path("output/security"),
    endpoint_url="http://agent:4000/agent/chat",
    timeout=10.0
)

print(f"Blocked: {summary['blocked']}, Passed: {summary['passed']}")
```

### Functional Accuracy実行

```python
from sandbox_runner.functional_accuracy import run_functional_accuracy

summary = run_functional_accuracy(
    agent_id="demo-agent",
    revision="v1",
    agent_card=agent_card_dict,
    output_dir=Path("output/functional"),
    endpoint_url="http://agent:4000/agent/chat",
    timeout=10.0
)

print(f"Average score: {summary['average_score']}")
```

## 🧪 テスト

```bash
cd sandbox-runner
pip install -e .[dev]
pytest
```

## 🐳 Docker

```bash
docker build -t sandbox-runner:latest sandbox-runner/
docker run sandbox-runner:latest --help
```

## 📊 W&B統合

W&B Weaveによる評価トレース:
- `WANDB_API_KEY` 環境変数で有効化
- 各攻撃プロンプトと応答をトレース
- `session_id` でSubmissionと紐付け

## 🔗 統合

Trusted Agent Hubの`app/routers/submissions.py`から呼び出されます:
- PreCheckステージ後に自動実行
- 結果は`score_breakdown`に保存
- エラー時は適切にハンドリングされ、リトライ可能
