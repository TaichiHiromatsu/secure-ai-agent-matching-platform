# 公開度と透明性

本システムの成果公開と透明性確保の取り組みをまとめる。

## 1. ソースコード・評価基盤の公開
- GitHub 公開リポジトリ（https://github.com/TaichiHiromatsu/secure-ai-agent-matching-platform）で評価コードとスコアリング実装を一般公開し、外部からの再現性検証を受け付け。
- Japan AISI 評価環境に準拠したベンチマーク設定・プロンプトサンプルを同梱。
- W&B Weave による評価ログを公開ビューで共有（匿名化済みメトリクス）。
- W&B Core Dashboard（project: `trusted-agent-hub`, read-only 公開）で主要メトリクスを閲覧可能。

## 2. 技術ブログでの知見共有
- [InsightEdge Tech Blog: Secure A2A Platform の技術解説](https://techblog.insightedge.jp/entry/geniac-prize-secure-a2a-platform)
  - アーキテクチャと MAGI/Jury Judge の設計意図
  - 異常検知とスコアフィードバックの運用手順
  - 実運用で得た学びと失敗事例

## 3. ライセンスと第三者コンテンツ
- 本リポジトリ全体のライセンス: **Apache License 2.0**（`LICENSE` に記載）。
- 含まれる Japan AISI 評価環境（aisev）は元リポジトリ同様 Apache License 2.0 準拠。再配布時は著作権表示とライセンス条項を保持。
- 依存OSSのライセンス一覧は `pyproject.toml` から生成可能（例: `pip-licenses` 等）。公開配布時は第三者ライブラリのライセンス表記を同梱する。
