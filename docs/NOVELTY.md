# システムの新規性に関する分析

本ドキュメントでは、Secure AI Agent Matching Platformの新規性について、学術論文や業界レポートを参照しながら分析する。

---

## 1. ストア（Trusted Agent Store）側の新規性

### 1.1 既存エージェントストアとの比較

| プラットフォーム | セキュリティ評価 | A2A対応 | 動的信頼スコア |
|-----------------|----------------|---------|---------------|
| [Microsoft Agent Store](https://devblogs.microsoft.com/microsoft365dev/introducing-the-agent-store-build-publish-and-discover-agents-in-microsoft-365-copilot/) | Microsoft審査（詳細非公開） | 予定（Public Preview段階） | なし |
| [OpenAI GPT Store](https://openai.com/index/introducing-the-gpt-store/) | OpenAI審査（詳細非公開） | なし | なし |
| [AWS AI Agents Marketplace](https://aws.amazon.com/about-aws/whats-new/2025/07/ai-agents-tools-aws-marketplace/) | AWS審査（詳細非公開） | なし | なし |
| **本システム** | **6段階評価（公開ベンチマーク）** | **完全対応** | **実行時フィードバック** |

### 1.2 6段階評価パイプライン

**新規性**: 既存のエージェントストアは審査プロセスが**ブラックボックス**であるのに対し、本システムは[Japan AISI（AIセーフティ・インスティテュート）の評価環境](https://github.com/Japan-AISI/aisev)に基づく**公開ベンチマーク**で評価し、W&B Weaveで全評価ログを可視化。公開と透明性の詳細は `docs/NOVELTY_PUBLICATION.md` を参照。

| 段階 | 評価内容 |
|------|----------|
| PreCheck | A2Aプロトコル準拠確認 |
| Security Gate | AISIベンチマークによるセキュリティテスト |
| Agent Card Accuracy | 宣言された機能の実際の動作確認 |
| Jury Judge | マルチモデルLLM評価 |
| Human Review | 人間による最終審査 |
| Publish | 信頼スコア計算・登録 |

### 1.3 A2Aプロトコル完全対応

**新規性**: [Google A2A Protocol](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)（2025年4月発表）に準拠したエージェントの登録・評価・配信を**完全実装**。

- [Microsoft Agent Store](https://www.microsoft.com/en-us/microsoft-cloud/blog/2025/05/07/empowering-multi-agent-apps-with-the-open-agent2agent-a2a-protocol/): A2A対応は2025年5月に発表されたが、まだPublic Preview段階
- 本システム: A2Aプロトコル準拠エージェントの登録・評価・配信を完全実装済み

### 1.4 AgentWare対策

**新規性**: [Threat Intelligence](https://www.threatintelligence.com/blog/ai-agentware)によると、悪意あるエージェント（AgentWare）がアプリストアにアップロードされるリスクが指摘されている。本システムは6段階評価パイプライン + 実行時監視 + 信頼スコアフィードバックで対策。

### 1.5 Jury Judge（Agents-as-a-Judge）

**新規性**: **複数のLLMモデル（GPT-4o, Claude Haiku, Gemini Flash）** による合議制評価と、**Final Judgeによる最終合議**。

- [LLMs-as-Judges: A Comprehensive Survey](https://arxiv.org/abs/2412.05579) (2024) では、Multi-LLM評価システムが「より包括的な評価を可能にする」と述べられているが、**合意形成アルゴリズムは研究途上**
- 本システムは**3人の陪審員による並列ラウンド議論**と**Final Judge（Gemini 2.5 Pro）による最終合議**を実装

**3フェーズ評価プロセス**:
1. **Phase 1 - 独立評価**: 3人の陪審員（GPT-4o, Claude Haiku, Gemini Flash）が独立して並列評価
2. **Phase 2 - 並列ラウンド議論**: 3人が同時に発言を生成し議論（最大3ラウンド）
3. **Phase 3 - Final Judge**: 独立した最終ジャッジエージェント（Gemini 2.5 Pro）が全員の議論を総合して最終判定を下す

---

## 2. 仲介エージェント（Secure Mediation Agent）側の新規性

### 2.1 A2A通信におけるリアルタイムセキュリティ監視

**新規性**: 外部エージェントからのレスポンスを**リアルタイムで監視**し、間接的プロンプトインジェクションを検出する仕組みは、現在の研究でも最先端の課題として認識されている。

- 既存研究の多くは**単一エージェントシステム**を対象としており、マルチエージェント環境でのセキュリティは未成熟
- [Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems](https://arxiv.org/abs/2410.07283) (2024) では、マルチエージェント間で悪意あるプロンプトが**ウイルスのように伝播**する脅威が指摘されている
- 本システムは**異常検知サブエージェント**と**セキュリティジャッジ**の2層構造でこれに対処
- CourtGuard (Wu & Maslowski, 2025年10月, [arXiv:2510.19844](https://arxiv.org/abs/2510.19844)) は「検察官・弁護人・裁判官」3役をLLMで構成する合議型防御を提案し、単一判定器の偏りを低減する一方で一部ケースで検出力が落ちることを報告。本システムはリアルタイム監視＋ストア側スコア反映の閉ループでその弱点を補完する。

### 2.2 プラン逸脱検出（Plan Deviation Detection）

**新規性**: 事前に作成した実行計画との**乖離を事後検証**するアプローチ。

- [arXiv:2506.08837](https://arxiv.org/abs/2506.08837) ではPlan-Then-Execute (P-t-E) パターンが間接的プロンプトインジェクションに対して頑健であることが示されている
- 本システムは**計画をMarkdownアーティファクトとして保存**し、実行後に計画との乖離を検出して監査可能性を確保

### 2.3 信頼スコアに基づく適応的セキュリティ感度

**新規性**: エージェントの信頼スコアに応じて**セキュリティ監視の厳しさを動的に調整**する仕組み。

- 低信頼エージェント → 高感度監視
- 高信頼エージェント → 標準監視
- [Security Boulevard](https://securityboulevard.com/2025/11/agent-trust-management-how-to-secure-ai-agents-without-losing-revenue/) (2025) では「エージェント信頼管理は信頼を動的に扱う」ことが重要と述べられている

---

## 3. 仲介エージェント + ストアの組み合わせによる新規性

### 3.1 自己改善型セキュリティエコシステム

**最大の新規性**: 仲介エージェントでの実行時セキュリティ問題が、ストアの信頼スコアに**フィードバックされる閉ループ**。

```
実行時に問題検出 → 信頼スコア減点 → 次回マッチングで順位低下 → 悪意あるエージェントの自動排除
```

- [World Economic Forum "Trust is the new currency in the AI agent economy"](https://www.weforum.org/stories/2025/07/ai-agent-economy-trust/) (2025) では、エージェント間の信頼が「パフォーマンス履歴、評判データ、予測可能な動作」の交換によって形成されると述べている
- 本システムはこの概念を**技術的に実装**

### 3.2 多層防御の完全実装

**新規性**: 登録時評価（事前）+ マッチング時フィルタリング + 実行時監視 + 事後検証 + フィードバックという**5層の防御**を統合。

| 層 | タイミング | 機能 |
|----|-----------|------|
| 1 | 登録時 | 6段階評価パイプライン |
| 2 | マッチング時 | 信頼スコア閾値フィルタ |
| 3 | 実行時 | 異常検知サブエージェント |
| 4 | 実行後 | 最終異常検知サブエージェント |
| 5 | フィードバック | 信頼スコア動的更新 |

- [Microsoft の間接的プロンプトインジェクション防御](https://www.microsoft.com/en-us/msrc/blog/2025/07/how-microsoft-defends-against-indirect-prompt-injection-attacks) (2025) でも多層防御が推奨されているが、**登録評価とランタイム監視を統合したシステムは稀少**

---

## 4. まとめ：新規性の位置づけ

| 観点 | 本システムの新規性 | 既存プラットフォームとの差異 |
|------|-------------------|---------------------------|
| **ストア** | 公開ベンチマークによる6段階評価 + A2A完全対応 | 既存ストアは審査がブラックボックス、A2A未対応 |
| **仲介エージェント** | A2A通信のリアルタイム監視 + プラン逸脱検出 | 既存研究は単一エージェント中心 |
| **統合** | 自己改善型閉ループセキュリティ | 登録時評価と実行時監視の統合は稀少 |

本システムは、2025年に[Linux Foundationに移管](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents)されたA2Aプロトコルのセキュリティ課題に対して、**実用的かつ包括的な解決策を提供する点で新規性が高い**。

---

## 5. スライド用文章（1スライド）

### Trusted Agent Store の新規性

**既存エージェントストアの課題**
- [Microsoft Agent Store](https://devblogs.microsoft.com/microsoft365dev/introducing-the-agent-store-build-publish-and-discover-agents-in-microsoft-365-copilot/)、[OpenAI GPT Store](https://openai.com/index/introducing-the-gpt-store/)、[AWS Marketplace](https://aws.amazon.com/about-aws/whats-new/2025/07/ai-agents-tools-aws-marketplace/)は審査プロセスが**ブラックボックス**
- [Threat Intelligence](https://www.threatintelligence.com/blog/ai-agentware)によると、悪意あるエージェント（AgentWare）がストアにアップロードされるリスクが指摘されている
- Microsoft Agent StoreのA2A対応はまだPublic Preview段階

**本システムの差別化**
- [Japan AISI評価環境](https://github.com/Japan-AISI/aisev)に基づく**公開ベンチマーク**で6段階評価、W&B Weaveで全評価ログを可視化
- [Google A2A Protocol](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)（2025年4月発表）に**完全対応**した唯一のエージェントストア
- [LLMs-as-Judges](https://arxiv.org/abs/2412.05579)の研究を発展させ、**3人の陪審員による並列ラウンド議論**と**Final Judge（Gemini 2.5 Pro）による最終合議**を実装

### Secure Mediation Agent の新規性

**マルチエージェントセキュリティの課題**
- [arXiv:2410.07283](https://arxiv.org/abs/2410.07283) (2024): マルチエージェント間で悪意あるプロンプトが**ウイルスのように伝播**する脅威（Prompt Infection）
- [Microsoft Security Blog](https://www.microsoft.com/en-us/msrc/blog/2025/07/how-microsoft-defends-against-indirect-prompt-injection-attacks) (2025): 多層防御を推奨

**本システムの差別化**
- **異常検知サブエージェント**と**セキュリティジャッジ**の2層構造でリアルタイム検出
- 計画をMarkdownアーティファクトとして保存し、**プラン逸脱を事後検証**
- [Security Boulevard](https://securityboulevard.com/2025/11/agent-trust-management-how-to-secure-ai-agents-without-losing-revenue/): 信頼スコアに基づく**適応的セキュリティ感度**

### 統合による最大の新規性

**自己改善型セキュリティエコシステム**
- [WEF](https://www.weforum.org/stories/2025/07/ai-agent-economy-trust/) (2025): 「信頼はAIエージェント経済の新たな通貨」
- 本システム: 実行時の問題検出 → 信頼スコア減点 → 悪意あるエージェントの自動排除という**閉ループ**を技術的に実装
- **登録時評価とランタイム監視を統合**したシステムは稀少

---

## 参考文献

### エージェントストア関連
- [Microsoft Agent Store](https://devblogs.microsoft.com/microsoft365dev/introducing-the-agent-store-build-publish-and-discover-agents-in-microsoft-365-copilot/)
- [OpenAI GPT Store](https://openai.com/index/introducing-the-gpt-store/)
- [AWS AI Agents Marketplace](https://aws.amazon.com/about-aws/whats-new/2025/07/ai-agents-tools-aws-marketplace/)
- [Threat Intelligence: AgentWare](https://www.threatintelligence.com/blog/ai-agentware)

### A2Aプロトコル関連
- [Google A2A Protocol Announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- [Microsoft A2A Support](https://www.microsoft.com/en-us/microsoft-cloud/blog/2025/05/07/empowering-multi-agent-apps-with-the-open-agent2agent-a2a-protocol/)
- [Linux Foundation A2A Project](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents)

### マルチエージェントセキュリティ関連
- [Prompt Infection in Multi-Agent Systems (arXiv:2410.07283)](https://arxiv.org/abs/2410.07283)
- [Design Patterns for Securing LLM Agents (arXiv:2506.08837)](https://arxiv.org/abs/2506.08837)
- [OWASP LLM Top 10 2025](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
- [CourtGuard: Adversarial Court for LLM Safety (arXiv:2510.19844)](https://arxiv.org/abs/2510.19844)

### LLM-as-a-Judge関連
- [LLMs-as-Judges: A Comprehensive Survey (arXiv:2412.05579)](https://arxiv.org/abs/2412.05579)

### AI安全性評価関連
- [Japan AISI評価環境（AIセーフティ・インスティテュート）](https://github.com/Japan-AISI/aisev)

### 信頼スコア・エージェント経済関連
- [WEF: Trust in AI Agent Economy](https://www.weforum.org/stories/2025/07/ai-agent-economy-trust/)
- [Security Boulevard: Agent Trust Management](https://securityboulevard.com/2025/11/agent-trust-management-how-to-secure-ai-agents-without-losing-revenue/)

### 防御手法関連
- [Microsoft Indirect Prompt Injection Defense](https://www.microsoft.com/en-us/msrc/blog/2025/07/how-microsoft-defends-against-indirect-prompt-injection-attacks)

---

*最終更新: 2025年12月*
