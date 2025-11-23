# Sandbox Runner (Skeleton)

`docs/design/sandbox-runner-spec.md` および `docs/design/sandbox-runner-implementation-plan.md` に基づく実装用ディレクトリ。
現在はPoC段階のため、最小限のCLIとDockerfile・pytestスモークテストを配置しています。

## CLI (仮)
```
python -m sandbox_runner.cli --agent-id demo-agent --revision rev1 --template google-adk
```
`--dry-run` オプションで外部依存を呼ばずに成果物を生成します。`WANDB_DISABLED=true` がデフォルトで、`WANDB_DISABLED=false` と `WANDB_API_KEY` を設定すると実際に WandB Run を作成します (`wandb` パッケージ要インストール)。`--wandb-base-url`/`--wandb-entity`/`--wandb-project` でRun URLの生成を制御できます。`--prompt-manifest` を指定すると `response_samples.jsonl` に含まれる `questionId` が AISI プロンプト一覧と整合しているか検証されます。`--generate-fairness` を付けると `fairness_probe.json` を出力し、スキーマ検証が行われます。

## テスト
```
cd sandbox-runner
pip install -e .[dev]
pytest
```

## Docker
```
docker build -t sandbox-runner:dev sandbox-runner/
```
