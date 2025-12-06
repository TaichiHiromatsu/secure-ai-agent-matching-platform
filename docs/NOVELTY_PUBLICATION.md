# 公開度と透明性

本システムの成果公開と透明性確保の取り組みをまとめる。

## 1. ソースコード・評価基盤の公開
- GitHub 公開リポジトリ（https://github.com/TaichiHiromatsu/secure-ai-agent-matching-platform）で評価コードとスコアリング実装を一般公開し、外部からの再現性検証を受け付け。
- Japan AISI 評価環境に準拠したベンチマーク設定・プロンプトサンプルを同梱。
- W&B Weave による評価ログを公開ビューで共有（匿名化済みメトリクス: エージェント名・ユーザー入力はマスク/ハッシュ化し、スコアや通過率のみ開示）。
- W&B Core Dashboard（project: `agent-store-sandbox`, entity: `naoya_yasuda-freelancer`）で主要メトリクスを閲覧可能。※ 公開する場合は W&B 側で Public/Shared (read-only) 設定を有効化。
- 環境変数設定（`.env` 既定値）: `WANDB_PROJECT=agent-store-sandbox`, `WANDB_ENTITY=naoya_yasuda-freelancer`, `WANDB_DISABLED=false`（未設定時は有効）。
- **GENIAC-PRIZE 審査期間の特例**: 公平性のため現在は W&B Weave/Core を全体公開設定とし、登録側も全申請のログを閲覧可能にしている（審査終了後は Private に戻す予定）。

## 2. 技術ブログでの知見共有
- [InsightEdge Tech Blog: Secure A2A Platform の技術解説](https://techblog.insightedge.jp/entry/geniac-prize-secure-a2a-platform)
  - アーキテクチャと MAGI/Jury Judge の設計意図
  - 異常検知とスコアフィードバックの運用手順
  - 実運用で得た学びと失敗事例

## 3. ライセンスと第三者コンテンツ
- 本リポジトリ全体のライセンス: **Apache License 2.0**（`LICENSE` に記載）。
- 含まれる Japan AISI 評価環境（aisev）は元リポジトリ同様 Apache License 2.0 準拠。再配布時は著作権表示とライセンス条項を保持。
- 依存OSSのライセンス一覧は `pyproject.toml` から生成可能（例: `pip-licenses` 等）。公開配布時は第三者ライブラリのライセンス表記を同梱する。
