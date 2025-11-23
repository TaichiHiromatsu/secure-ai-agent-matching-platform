# エージェントストア関連リサーチメモ

## 収集済み

| ファイル | 年 | タイトル | 主題 / エージェントストアへの示唆 |
| --- | --- | --- | --- |
| `docs/papers/responsible-ai-agents-2502.18359.pdf` | 2025 | Responsible AI Agents: Policy Choices for Open Agent Ecosystems | オープンなエージェント市場で必要となる規制・ライセンシング枠組みを整理。審査ポリシー、ライフサイクル監視の要件定義に活用。 |
| `docs/papers/automated-risky-game-2506.00073.pdf` | 2025 | The Automated but Risky Game: Governing General-Purpose AI Agents | マルチステークホルダー協調とリスク分層の設計論。AISI判定ルールやエスカレーションパスを洗練する材料。 |
| `docs/papers/fama-fair-matching-2509.03890.pdf` | 2025 | FaMA: Fair Matching for Agent Marketplaces | エージェント需要と供給のマッチングを公平性・効率性で最適化する手法。カタログ推薦ロジックの将来拡張に示唆。 |
| `docs/papers/governance-as-a-service-2508.18765.pdf` | 2025 | Governance-as-a-Service for Multi-Agent Ecosystems | サービスとしてのガバナンスを提案。審査パイプラインのモジュール化と外部委託(例: AISI)の制御ポイント検討に役立つ。 |
| `docs/papers/decentralized-gradient-marketplaces-2509.05833.pdf` | 2025 | Benchmarking Robust Aggregation in Decentralized Gradient Marketplaces | 分散マーケットプレイスのロバスト性指標と評価方法。エージェント提供者の信頼度スコアリングやメトリクス設計の参考。 |
| `docs/papers/marketplace-for-ai-models-2003.01593.pdf` | 2020 | Marketplace for AI Models | 初期のAIモデル市場の課題と設計指針。コンプライアンス/収益化/監査要素の歴史的整理に利用。 |

## docs/README.md 反映箇所
- `docs/README.md` セクション0: リサーチドリブン設計原則 (全論文の示唆をマッピング)
- セクション1: ガバナンス層/公平推薦/TrustSignal (`responsible-ai-agents`, `automated-risky-game`, `fama-fair-matching`, `governance-as-a-service`, `decentralized-gradient-marketplaces`)
- セクション2.1/2.2: データモデルにおける `FairnessMetricSnapshot`, `TrustSignal`, `AuditLog.hash_chain_id` など (`fama-fair-matching`, `decentralized-gradient-marketplaces`, `marketplace-for-ai-models`)
- セクション3.1/3.2: リスクTier、多段審査、Sandbox Validation (`automated-risky-game`, `responsible-ai-agents`, `governance-as-a-service`)
- セクション3.3/3.4: ガバナンスハブ、透明性レポート (`responsible-ai-agents`, `governance-as-a-service`, `marketplace-for-ai-models`)
- セクション3.1/3.3: Sandbox Validationでの Weights & Biases MCP 活用方針 (Pre-flight ゲーティングと監査証跡の分離) ※運用方針メモ
- セクション4: 次のアクション(公平性・収益化・外部監査計画) (`responsible-ai-agents`, `fama-fair-matching`, `marketplace-for-ai-models`)

## 未収集・追加候補
- LangGraph / LangSmith 運用に関する公式ドキュメント・ホワイトペーパー
- Google ADK を利用したエージェントデプロイ事例 (Google Cloud / Research)
- AI Agent Marketplace の経済モデルや課金スキームに関する最新研究 (2025年以降)

## 次に行うこと
- 各PDFの要点を抽出し、エージェント登録・審査・公開ワークフローへの適用可能性を整理するサマリーテンプレートを作成。
- 主要論文については引用箇所を特定し、アーキテクチャドキュメント(`docs/README.md`)へ反映する。

最終更新: 2025-10-29 (UTC-07:00 推定)
