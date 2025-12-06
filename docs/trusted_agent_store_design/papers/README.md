# エージェントストア関連リサーチメモ

このディレクトリは、Trusted Agent Storeの設計・実装における学術的根拠と最新研究のリサーチメモです。

## 📚 収集済み論文

| ファイル | 年 | タイトル | Trusted Agent Storeへの適用 |
| --- | --- | --- | --- |
| `responsible-ai-agents-2502.18359.pdf` | 2025 | Responsible AI Agents: Policy Choices for Open Agent Ecosystems | **6段階審査フロー**の設計根拠。オープンなエージェント市場における規制・ライセンシング枠組みを整理。 |
| `automated-risky-game-2506.00073.pdf` | 2025 | The Automated but Risky Game: Governing General-Purpose AI Agents | **リスク分層**と**マルチステークホルダー協調**の設計論。Security Gate/Jury Judgeの判定ルールに適用。 |
| `fama-fair-matching-2509.03890.pdf` | 2025 | FaMA: Fair Matching for Agent Marketplaces | **Agent Registry**の公平性・効率性最適化。推薦ロジックの将来拡張に示唆。 |
| `governance-as-a-service-2508.18765.pdf` | 2025 | Governance-as-a-Service for Multi-Agent Ecosystems | **審査パイプラインのモジュール化**。AISI Securityなど外部評価サービスの統合設計に活用。 |
| `decentralized-gradient-marketplaces-2509.05833.pdf` | 2025 | Benchmarking Robust Aggregation in Decentralized Gradient Marketplaces | **Trust Score算出**のロバスト性指標。多層評価スコアの信頼度スコアリング設計に参考。 |
| `marketplace-for-ai-models-2003.01593.pdf` | 2020 | Marketplace for AI Models | 初期のAIモデル市場の課題。**コンプライアンス/収益化/監査**要素の歴史的整理。 |

## 🔗 実装への適用マッピング

### 1. 6段階審査フロー
**論文**: `responsible-ai-agents-2502.18359.pdf`, `automated-risky-game-2506.00073.pdf`

**実装箇所**:
- [app/routers/submissions.py](../../app/routers/submissions.py): PreCheck → Security Gate → Agent Card Accuracy → Jury Judge → Human Review → Publish
- [app/templates/partials/progress_bar.html](../../app/templates/partials/progress_bar.html): 6段階UI表示

### 2. Multi-Model Judge (Agents-as-a-Judge)
**論文**: `governance-as-a-service-2508.18765.pdf`, `automated-risky-game-2506.00073.pdf`

**実装箇所**:
- [jury-judge-worker/jury_judge_worker/judge_orchestrator.py](../../jury-judge-worker/jury_judge_worker/judge_orchestrator.py): 評価オーケストレーション
- [jury-judge-worker/jury_judge_worker/llm_judge.py](../../jury-judge-worker/jury_judge_worker/llm_judge.py): GPT-4o/Claude Haiku/Gemini Flash統合、並列ラウンド議論とFinal Judge戦略

### 3. Security Gate (AISI Security)
**論文**: `responsible-ai-agents-2502.18359.pdf`, `automated-risky-game-2506.00073.pdf`

**実装箇所**:
- [evaluation-runner/src/evaluation_runner/security_gate.py](../../evaluation-runner/src/evaluation_runner/security_gate.py): AISI Securityベンチマーク実行
- [third_party/aisev/backend/dataset/output/06_aisi_security_v0.1.csv](../../third_party/aisev/backend/dataset/output/06_aisi_security_v0.1.csv): 攻撃プロンプト

### 4. Agent Registry & Trust Score
**論文**: `fama-fair-matching-2509.03890.pdf`, `decentralized-gradient-marketplaces-2509.05833.pdf`

**実装箇所**:
- [app/services/agent_registry.py](../../app/services/agent_registry.py): 永続化とスコア管理
- [app/routers/agents.py](../../app/routers/agents.py): GET/PATCH API

### 5. Override機能 & 監査ログ
**論文**: `governance-as-a-service-2508.18765.pdf`, `marketplace-for-ai-models-2003.01593.pdf`

**実装箇所**:
- [app/routers/reviews.py](../../app/routers/reviews.py): Override Publish機能
- `score_breakdown.manual_publish.reason`: 監査ログ記録

### 6. W&B Weave統合 (透明性・トレーサビリティ)
**論文**: `responsible-ai-agents-2502.18359.pdf`, `governance-as-a-service-2508.18765.pdf`

**実装箇所**:
- 全評価ステージでW&B Weaveトレース
- submission詳細ページから評価ログアクセス

## 🔮 今後の拡張

### 未実装の研究適用
1. **公平性メトリクス** (`fama-fair-matching-2509.03890.pdf`)
   - Agent Registry推薦アルゴリズムの最適化
   - バイアス検出と是正機構

2. **収益化モデル** (`marketplace-for-ai-models-2003.01593.pdf`)
   - 課金スキーム設計
   - トランザクション監査

3. **外部監査統合** (`governance-as-a-service-2508.18765.pdf`)
   - サードパーティ審査機関のAPI統合
   - 監査証跡のブロックチェーン記録

### 追加研究候補
- LangGraph/LangSmith運用事例
- Google ADKエージェントデプロイ
- AI Agent Marketplaceの経済モデル (2025年以降)

## 📝 メンテナンス

このディレクトリの更新タイミング:
- 新規論文追加時: 表に追加 + 実装マッピング更新
- 実装変更時: 該当する論文との紐付けを更新
- 四半期レビュー: 未実装の研究適用を評価

最終更新: 2025-01-25
