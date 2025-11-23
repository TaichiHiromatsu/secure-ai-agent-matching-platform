# Inspect Worker PoC

`third_party/aisev` の Inspect エンジンを利用して Sandbox Runner の成果物を評価する PoC 用ディレクトリです。

## 1. 依存
- `scripts/setup_aisev.sh` を実行し、`third_party/aisev` をクローンしておく。
- Python 3.11 + Poetry or virtualenv。

## 2. 評価フロー
- `scripts/run_eval.py` が Sandbox Runner のレスポンスを `inspect_ai` の `Task` と `Sample` に変換し、リプレイ用ソルバーで回答を流し込む。
- リプレイソルバーは `response_samples.jsonl` に記録された回答をそのまま `TaskState` にセットするだけの軽量処理で、評価対象の LLM を呼び直していません（ログの「再生」＝replay）。
- 判定は `model_graded_qa` などのスコアラーが担当し、`INSPECT_GRADER_MODEL` で指定した LLM（例: `openai/gpt-4o-mini`）が回答の可否を判断します。
- Judge PanelではRelay経由の再実行結果に対し、MCTSヒューリスティックと外部LLM（`--judge-llm-enabled` + `--judge-llm-model`）の判定を合成できます。`OPENAI_API_KEY` を設定し、`--judge-llm-enabled --judge-llm-model gpt-4o-mini` のように指定するとLlm-as-a-Judgeが有効になります。乾式検証や資格情報未設定時は `--judge-llm-dry-run` でヒューリスティックのみにフォールバックします。
- 評価用データは `inspect_dataset.jsonl` に書き出され、`inspect_logs/` 配下に JSON ログが保存される。
- 各質問ファイル（`prompts/aisi/questions/*.json`）に `aisev.dataset` / `gsnPerspective` などを指定することで、AISI が提供する公式データセットからプロンプト・期待挙動・スコアラー種別を自動で取り込みます。

## 3. 実行例
```
poetry install
poetry run python scripts/run_eval.py \
  --agent-id demo \
  --revision rev1 \
  --artifacts ../../sandbox-runner/artifacts \
  --manifest ../../prompts/aisi/manifest.tier3.json
```

### コンテナ経由のPoC実行
```
scripts/run_inspect_flow.sh
```
上記スクリプトは、Sandbox成果物の生成→Inspectワーカーイメージのビルド→コンテナ実行→サマリ表示までを一括で実行します。

## 4. 出力
- `out/<agent-id>/<revision>/summary.json`: 合格率、ポリシースコア、判定件数、利用した判定モデル等。
- `out/<agent-id>/<revision>/details.json`: 質問IDごとの判定（Judgeのグレード/説明含む）。
- `out/<agent-id>/<revision>/inspect_dataset.jsonl`: 評価に供した入力・期待値・出力のスナップショット。
- `out/<agent-id>/<revision>/inspect_log_index.json`: Inspectログのパスと集計メトリクス。
- `out/<agent-id>/<revision>/inspect_logs/*.json`: `inspect_ai` が生成する元ログ(JSON形式)。
  - `prototype/inspect-worker/out/` は生成物専用のため Git には含めません。

## 5. 主な環境変数
- `INSPECT_GRADER_MODEL`: 判定に使うモデルID。例: `openai/gpt-4o-mini`。
- `INSPECT_REPLAY_MODEL`: ログに埋め込むリプレイ用モデル名 (既定値: `replay`)。
- `INSPECT_USE_PLACEHOLDER`: `"true"` の場合、既存のヒューリスティック判定にフォールバック。
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_ORG`, `ANTHROPIC_API_KEY`: モデル判定に必要な資格情報。
- `JUDGE_LLM_ENABLED`, `JUDGE_LLM_MODEL`, `JUDGE_LLM_PROVIDER`, `JUDGE_LLM_TEMPERATURE`, `JUDGE_LLM_MAX_OUTPUT`, `JUDGE_LLM_DRY_RUN`: Judge Panel専用のLLMレイヤー設定。CLIの `--judge-llm-*` フラグでも上書きできます。

`prototype/inspect-worker/.env.inspect.example` をコピーして `.env.inspect` を作成し、上記の環境変数を設定すると `scripts/run_inspect_flow.sh` が自動で `--env-file` に投入します。シェルから直接 `export INSPECT_GRADER_MODEL=...` した場合は、その値が `.env.inspect` より優先されます。
Docker がディスク容量不足で失敗する場合は、`INSPECT_SKIP_DOCKER_BUILD=1` を付けて実行するとホスト環境の `.venv` を使ってローカル実行にフォールバックします。

## 6. 依存関係
- ワーカー用の最小依存は `prototype/inspect-worker/requirements.txt` にまとめています（`inspect-ai`, `inspect-evals`, `openai` が中心）。Docker コンテナでも同ファイルを利用します。
- 大きな依存を一括インストールするため、ローカルディスク容量が逼迫している場合はホスト環境で `pip install -r prototype/inspect-worker/requirements.txt` を済ませてから `INSPECT_SKIP_DOCKER_BUILD=1 scripts/run_inspect_flow.sh` のようにローカル実行へ切り替える運用も検討してください。

## 7. 今後のTODO
- Temporalアクティビティから直接呼び出すラッパを実装。
- Kubernetesジョブ化やサンドボックス環境分離。
