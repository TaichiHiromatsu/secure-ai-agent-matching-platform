# W&B Weave トレーシングガイド

## 1. W&B Weave とは

### 1.1 概要

W&B Weave（Weights & Biases Weave）は、LLMアプリケーションの可観測性（Observability）と評価のためのツールキット。AIエージェントやLLMを使ったアプリケーションの開発において、全ての入出力、コスト、レイテンシ、評価メトリクスを自動的にトレース・記録する。

主な用途は3つある。

**トレーシング**: `@weave.op` デコレータを関数に付与するだけで、その関数の入力・出力・実行時間・コストが自動的にW&Bのダッシュボードに記録される。LLMの呼び出しだけでなく、任意の関数（バリデーション、データ変換など）もトレース可能。

**評価（Evaluation）**: `weave.Evaluation` と Scorer を使い、LLMアプリケーションの品質・安全性・その他のメトリクスを体系的に評価する。

**モニタリング**: 本番環境でのAIエージェントの振る舞いを継続的に追跡し、問題の検出・分析を支援する。

### 1.2 基本的な使い方

```python
import weave

# 1. 初期化（プロジェクト名を指定）
weave.init("my-entity/my-project")

# 2. トレースしたい関数に @weave.op() を付与
@weave.op()
def my_llm_call(prompt: str) -> str:
    # LLMを呼び出す処理
    response = client.generate(prompt)
    return response.text

# 3. 関数を呼び出すだけで自動的にトレースされる
result = my_llm_call("こんにちは")
# → W&Bダッシュボードに入力・出力・レイテンシ等が記録される
```

### 1.3 トレースツリー

`@weave.op()` を付与した関数が別の `@weave.op()` 関数を内部で呼び出すと、その呼び出し関係が**トレースツリー**（親子関係）として自動的に記録される。これにより、複雑なマルチステップ処理の各段階を可視化できる。

```
run_judge_panel()                    ← ルート
  └── _run_collaborative_jury_evaluation()  ← 子
      └── evaluate_panel_async()            ← 孫
          └── evaluate_async()              ← ひ孫
              └── _evaluate_with_google_adk_async()  ← 最下層
```

レイテンシやコストはツリーの各レベルで自動集計され、ボトルネックの特定に使える。

### 1.4 Summary（要約メトリクス）

トレース実行中に `weave.get_current_call()` でカレントのトレースコンテキストを取得し、`summary` に任意のメトリクスを記録できる。

```python
current = weave.get_current_call()
if current is not None:
    current.summary = {
        "model": "gemini-2.5-pro",
        "score": 85,
        "verdict": "approve",
    }
```

### 1.5 アーティファクト

W&B Artifacts は、評価レポートやモデルの重みなどのファイルをバージョン管理付きで保存する仕組み。URI形式（`weave://entity/project/artifacts/name/version`）でどこからでも参照できる。

---

## 2. 本システムでのW&B Weave利用箇所

本システムでは、**エージェントストアのMAGI（合議評価）パイプライン**でW&B Weaveが使用されている。セキュア仲介エージェント側では使用されていない。

### 2.1 利用範囲の概観

```
エージェントストア評価パイプライン
│
├── Security Gate        … Weave未使用
├── Agent Card Accuracy  … Weave未使用
│
├── MAGI（Jury Judge）   … ★ Weave でトレース
│   ├── run_judge_panel()                       @weave.op()
│   ├── _run_stage_multi_model_judge_panel()     @weave.op()
│   ├── _run_collaborative_jury_evaluation()     @weave.op()
│   ├── evaluate_panel_async()                   @weave.op()
│   ├── _run_parallel_evaluation_async()         @weave.op()
│   ├── evaluate_async()                         @weave.op()
│   ├── _evaluate_with_google_adk_async()        @weave.op()
│   └── evaluate_stage_chain_async()             @weave.op()
│
└── Artifact Storage     … ★ Weave Artifacts で保存
    ├── security_gate_report.jsonl
    ├── agent_card_accuracy_report.jsonl
    └── jury_judge_report.jsonl
```

### 2.2 環境変数の設定

```bash
# .env または環境変数で設定
WANDB_API_KEY=<W&B APIキー>
WANDB_PROJECT=agent-store-sandbox      # W&Bプロジェクト名
WANDB_ENTITY=<W&Bのユーザー名/チーム名>  # W&Bエンティティ
WANDB_BASE_URL=https://api.wandb.ai    # W&B APIのベースURL
```

これらは `evaluation-runner/src/evaluation_runner/jury_judge.py` で読み込まれ、`weave.init(f"{WANDB_ENTITY}/{WANDB_PROJECT}")` でプロジェクトが初期化される。

---

## 3. トレースされる情報の詳細

### 3.1 MAGI合議プロセスのトレース

MAGIの3フェーズ全てがWeaveでトレースされる。

#### フェーズ1: 独立評価

各ジャッジ（GPT-4o, Claude Haiku, Gemini Flash）の個別評価がトレースされる。

**記録される情報:**
- 各ジャッジに渡されたプロンプト（Security Gate/Agent Card Accuracyの結果サマリを含む）
- 各ジャッジの応答（AISEV 4軸スコア + verdict + reasoning）
- 実行時間（レイテンシ）
- リトライ回数、エラー情報
- SAFETYフィルターによるブロック発生状況

#### フェーズ2: 合議討論

複数ラウンドの討論がトレースされる。

**記録される情報:**
- 各ラウンドの各ジャッジの発言（reasoning）
- スコアの変更有無（positionChanged）
- 合意状況（ConsensusStatus）
- 膠着検出（stagnation）

#### フェーズ3: 最終判定

Final Judgeの統合判定がトレースされる。

**記録される情報:**
- 最終4軸スコア（taskCompletion, toolUsage, autonomy, safety）
- 最終verdict（approve / manual / reject）
- 投票分布（voteDistribution）
- 最終判定の根拠（rationale）
- 信頼度（confidence）

### 3.2 Summary メトリクスの記録

`llm_judge.py` の `_evaluate_with_google_adk_async()` 内で、各評価の完了時にWeave Summaryに以下が記録される:

```python
summary.update({
    "model": self.config.model,        # 使用モデル名
    "provider": self.config.provider,  # プロバイダ（google-adk等）
    "task_completion": result.task_completion,  # 0-20
    "tool_usage": result.tool_usage,            # 0-15
    "autonomy": result.autonomy,                # 0-15
    "safety": result.safety,                    # 0-50
    "total_score": result.total_score,          # 0-100
    "verdict": result.verdict,                  # approve|manual|reject
})
```

### 3.3 アーティファクトの保存

`artifact_storage.py` を通じて、各評価ステージのレポートがW&B Artifactsとして保存される。

| アーティファクト | タイプ | 内容 |
|---|---|---|
| `sg-report-{submission_id}` | security-gate-report | Security Gateの全テスト結果（JSONL） |
| `aca-report-{submission_id}` | agent-card-accuracy-report | Agent Card Accuracyの全テスト結果（JSONL） |
| `judge-report-{submission_id}` | judge-report | MAGI評価の全レポート（JSONL） |

保存処理の流れ:

```python
# submissions.py での利用例
from evaluation_runner.artifact_storage import store_weave_artifact

# Security Gateの結果をArtifactとして保存
sg_artifact_uri = store_weave_artifact(
    output_dir / "security" / "security_gate_report.jsonl",
    f"sg-report-{submission_id}",
    "security-gate-report"
)
# → "weave://entity/project/artifacts/sg-report-abc123/v0"

# このURIをMAGIジャッジに渡し、必要に応じて詳細データを参照できるようにする
compressed_security_results = {
    "summary": { ... },
    "artifacts": {"full_report": sg_artifact_uri}
}
```

ジャッジがトークン上限に達しそうな場合に、サマリだけでなくArtifact URIを参照して詳細データにアクセスできる設計になっている（`judge_artifact_design.md`）。

---

## 4. トレースツリーの構造

本システムでのWeaveトレースの親子関係:

```
run_judge_panel()                                    [ルート @weave.op()]
│
│  weave.init("{WANDB_ENTITY}/{WANDB_PROJECT}")
│
└── _run_stage_multi_model_judge_panel()              [@weave.op()]
    │
    ├── _run_collaborative_jury_evaluation()           [@weave.op()]
    │   │
    │   └── jury_judge.evaluate_collaborative_batch()
    │       │
    │       └── evaluate_panel_async()                 [@weave.op()]
    │           │
    │           └── _run_parallel_evaluation_async()    [@weave.op()]
    │               │
    │               ├── judge_A.evaluate_async()        [@weave.op()] GPT-4o
    │               │   └── _evaluate_with_google_adk_async()  [@weave.op()]
    │               │       └── weave.get_current_call().summary = {...}
    │               │
    │               ├── judge_B.evaluate_async()        [@weave.op()] Claude
    │               │   └── ...
    │               │
    │               └── judge_C.evaluate_async()        [@weave.op()] Gemini
    │                   └── ...
    │
    └── evaluate_stage_chain_async()                   [@weave.op()]
        └── _run_parallel_evaluation_async()            [@weave.op()]
            └── (plan → counter → reconcile ステージ)
```

---

## 5. Graceful Degradation（W&B未インストール時の動作）

W&B Weaveがインストールされていない環境でもシステムが動作するよう、フォールバック機構が実装されている。

```python
# jury_judge.py
try:
    import weave
    HAS_WEAVE = True
except ImportError:
    HAS_WEAVE = False
    # No-opデコレータを定義
    class weave:
        @staticmethod
        def op():
            def decorator(func):
                return func
            return decorator
        @staticmethod
        def init(project_name):
            pass
```

`HAS_WEAVE = False` の場合:
- `@weave.op()` は単なるパススルーデコレータとなり、関数の動作に影響しない
- `weave.init()` は何もしない
- Summary記録の箇所は `if HAS_WEAVE and hasattr(weave, "get_current_call")` でガードされている
- Artifact保存は `import wandb` の失敗時に `None` を返す

つまり、W&Bが未設定でも**評価パイプライン自体は正常に動作する**。トレースが記録されないだけで、評価結果はローカルのJSONL/JSONファイルに保存される。

---

## 6. W&B Weaveの使い方（開発者向け）

### 6.1 セットアップ

```bash
# 1. W&Bアカウントを作成（https://wandb.ai/）
# 2. APIキーを取得（Settings → API Keys）
# 3. 環境変数を設定
export WANDB_API_KEY=<your-api-key>
export WANDB_ENTITY=<your-username-or-team>
export WANDB_PROJECT=agent-store-sandbox

# 4. 依存パッケージの確認
pip install weave wandb
```

### 6.2 トレースの確認方法

1. 評価パイプラインを実行する（エージェントストアでのSubmission処理）
2. W&Bダッシュボード（https://wandb.ai/）にアクセス
3. プロジェクト（例: `agent-store-sandbox`）を選択
4. 「Weave」タブを開く
5. トレース一覧から `run_judge_panel` のトレースを選択
6. トレースツリーで各フェーズの実行状況を確認

### 6.3 確認できる主な情報

**トレースツリー画面:**
- 各関数の実行時間（レイテンシ）
- 各LLMジャッジへの入力プロンプトと出力
- エラーやリトライの発生状況
- 親子関係での処理の流れ

**Summary画面:**
- 各ジャッジのAISEV 4軸スコア
- 使用モデルとプロバイダ
- 最終verdict

**Artifacts画面:**
- Security Gate / Agent Card Accuracy / Judge の各レポートファイル
- バージョン管理されたレポート履歴

### 6.4 UIからのアクセス

エージェントストアのUI上では、MAGI評価完了後に「📊 View in W&B Weave」リンクが表示され、該当トレースに直接ジャンプできる。

### 6.5 デバッグ時の活用

MAGI評価で予期しないスコアが出た場合:

1. W&Bでトレースを開く
2. `evaluate_async()` レベルのトレースを確認し、各ジャッジに渡された入力を確認
3. ジャッジの生の応答（`raw_response`）を確認
4. JSONパースエラーやSAFETYブロックの発生有無を確認
5. Claudeフォールバックが発動したかどうかを確認
6. Artifact内の詳細レポートと照合

---

## 7. アーキテクチャ上の設計判断

### 7.1 なぜMAGIだけにWeaveを使っているのか

Security GateとAgent Card Accuracyは比較的単純な入出力（プロンプト→レスポンス→分類）であり、結果はJSONLファイルで十分追跡できる。一方、MAGIは3つの異なるLLMモデルによる合議プロセスであり、フェーズ1（独立評価）→ フェーズ2（討論）→ フェーズ3（最終判定）という複雑な多段階処理が発生する。この複雑さをデバッグ・分析するためにWeaveのトレースツリーが有効。

### 7.2 Artifact設計の意図

MAGI評価では、ジャッジLLMのコンテキストウィンドウ（トークン上限）が問題になりうる。Security Gateの全テスト結果やAgent Card Accuracyの全シナリオ結果を全てプロンプトに含めるとトークンオーバーフローする。そこで、要約（summary）だけをプロンプトに含め、詳細データはArtifact URIとして渡す設計になっている。ジャッジが必要に応じてArtifactを参照できるようにすることで、トークン効率と情報量を両立している。

### 7.3 仲介エージェント側でWeaveを使わない理由

セキュア仲介エージェントはGoogle ADK上で動作しており、ADK自体がセッション管理やイベント記録の機構を持つ。また、会話アーティファクト（`artifacts/conversations/`）として独自の記録システムが実装済み。そのため、Weaveによる追加のトレーシングは現時点では導入されていない。ただし、Google ADKはWeaveとの統合をサポートしており、将来的に導入する余地はある。

---

## 8. 関連ファイル一覧

| ファイル | 役割 |
|---|---|
| `evaluation-runner/src/evaluation_runner/jury_judge.py` | Weave初期化、ルートの@weave.op()、MAGI全体のオーケストレーション |
| `jury-judge-worker/jury_judge_worker/llm_judge.py` | 個別ジャッジの@weave.op()、Summary記録 |
| `jury-judge-worker/jury_judge_worker/multi_model_judge.py` | マルチモデル評価の@weave.op() |
| `jury-judge-worker/jury_judge_worker/jury_judge_collaborative.py` | 合議プロセスのトレース |
| `evaluation-runner/src/evaluation_runner/artifact_storage.py` | Artifact保存・取得・フォーマット |
| `app/routers/submissions.py` | パイプラインからのArtifact保存呼び出し |
| `.env` / `.env_sample` | WANDB_* 環境変数の設定 |

---

## 9. 参考リンク

- [W&B Weave 公式ドキュメント](https://docs.wandb.ai/weave)
- [W&B Weave GitHub](https://github.com/wandb/weave)
- [W&B Weave Ops ガイド](https://weave-docs.wandb.ai/guides/tracking/ops/)
- [W&B Weave Scorers ガイド](https://github.com/wandb/weave/blob/master/docs/docs/guides/evaluation/scorers.md)
- [Google ADK × W&B Weave 統合](https://google.github.io/adk-docs/observability/weave/)
- [W&B Weave コース](https://wandb.ai/site/courses/weave/)

---

*本文書は2026-02-28時点でのコードベース分析に基づく*
