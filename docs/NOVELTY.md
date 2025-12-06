# システムの新規性に関する分析

本ドキュメントでは、Secure AI Agent Matching Platformの新規性について、学術論文や業界レポートを参照しながら分析する。

---

## 1. 仲介エージェント（Secure Mediation Agent）側の新規性

### 1.1 A2A通信におけるリアルタイムセキュリティ監視

**新規性**: 外部エージェントからのレスポンスを**リアルタイムで監視**し、間接的プロンプトインジェクションを検出する仕組みは、現在の研究でも最先端の課題として認識されている。

- 既存研究の多くは**単一エージェントシステム**を対象としており、マルチエージェント環境でのセキュリティは未成熟
- [Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems](https://arxiv.org/abs/2410.07283) (2024) では、マルチエージェント間で悪意あるプロンプトが**ウイルスのように伝播**する脅威が指摘されている
- 本システムは**異常検知サブエージェント**と**セキュリティジャッジ**の2層構造でこれに対処

### 1.2 プラン逸脱検出（Plan Deviation Detection）

**新規性**: 事前に作成した実行計画との**乖離を検出**するアプローチは、マルチエージェントオーケストレーション特有の防御手法である。

- [Amazon Bedrock のガイド](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/)では、plan-verify-execute (PVE) パターンが間接的プロンプトインジェクションに対して頑健であることが示されている
- 本システムはこれをさらに発展させ、**計画をMarkdownアーティファクトとして保存**し、監査可能性を確保

### 1.3 信頼スコアに基づく適応的セキュリティ感度

**新規性**: エージェントの信頼スコアに応じて**セキュリティ監視の厳しさを動的に調整**する仕組み。

- 低信頼エージェント → 高感度監視
- 高信頼エージェント → 標準監視
- [Dynamic Trust Score研究](https://thesciencebrigade.com/JAIR/article/view/611) (2025) ではZero Trust Architectureでの動的信頼評価が議論されているが、**A2Aプロトコルへの適用は新規**

---

## 2. ストア（Trusted Agent Store）側の新規性

### 2.1 6段階評価パイプライン

**新規性**: エージェント登録前に**6段階の厳格な評価**を行うプラットフォームは、現在のAIエージェントマーケットプレイスにはほとんど存在しない。

| 段階 | 評価内容 |
|------|----------|
| PreCheck | A2Aプロトコル準拠確認 |
| Security Gate | AISIベンチマークによるセキュリティテスト |
| Agent Card Accuracy | 宣言された機能の実際の動作確認 |
| Jury Judge | マルチモデルLLM評価 |
| Human Review | 人間による最終審査 |
| Publish | 信頼スコア計算・登録 |

- [UK AI Safety Institute (AISI)](https://www.aisi.gov.uk/blog/inspect-evals) のInspect Evalsフレームワークを活用
- [CB Insights AI Agent Market Map (2025)](https://www.cbinsights.com/research/ai-agent-market-map/) によると、エージェントスタートアップは「信頼構築の5つの方法」を模索中であり、体系的な評価パイプラインは差別化要素

### 2.2 Jury Judge（Agents-as-a-Judge）

**新規性**: **複数のLLMモデル（GPT-4o, Claude Haiku, Gemini Flash）** による合議制評価と、**Minority-Veto戦略**による合意形成。

- [LLMs-as-Judges: A Comprehensive Survey](https://arxiv.org/abs/2412.05579) (2024) では、Multi-LLM評価システムが「より包括的な評価を可能にする」と述べられているが、**合意形成アルゴリズムは研究途上**
- 本システムは**3人の陪審員による並列ラウンド議論**と**Minority-Veto戦略**（30%以上が問題検出→needs_review、1人でもreject→reject）を実装

### 2.3 セキュリティベンチマーク統合

**新規性**: AISIのセキュリティベンチマーク（System Prompt Leakage, Jailbreak, Toxic Content, Robustness, Fairness）を**優先度ベースでサンプリング**し、実用的な評価時間内で網羅的テストを実現。

- [OWASP LLM Top 10 2025](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) ではプロンプトインジェクションが第1位のリスクとして認定
- Priority 1テストは全件実施、Priority 2-4は60%/30%/10%でサンプリングという実用的アプローチ

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

### 3.3 A2Aプロトコル標準との統合

**新規性**: [Google A2A Protocol](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) への完全準拠と、その上にセキュリティレイヤーを構築。

- [CSA Threat Modeling for A2A](https://cloudsecurityalliance.org/blog/2025/04/30/threat-modeling-google-s-a2a-protocol-with-the-maestro-framework) (2025) では「A2Aプロトコルは大きな可能性を秘めているが、重大なセキュリティ課題も導入する」と警告
- [arXiv: Building A Secure Agentic AI Application Leveraging A2A Protocol](https://arxiv.org/abs/2504.16902) (2025) が包括的セキュリティ分析を行っているが、本システムは**実装レベルで解決策を提供**

---

## 4. まとめ：新規性の位置づけ

| 観点 | 本システムの新規性 | 関連研究との差異 |
|------|-------------------|------------------|
| **仲介エージェント** | A2A通信のリアルタイム監視 + プラン逸脱検出 | 既存研究は単一エージェント中心 |
| **ストア** | 6段階評価 + マルチモデルLLM合議制 | 既存マーケットプレイスは信頼構築が課題 |
| **統合** | 自己改善型閉ループセキュリティ | 登録時評価と実行時監視の統合は稀少 |

本システムは、2025年に[Linux Foundationに移管](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents)されたA2Aプロトコルのセキュリティ課題に対して、**実用的かつ包括的な解決策を提供する点で新規性が高い**。

---

## 参考文献

### A2Aプロトコル関連
- [Google A2A Protocol Announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- [A2A Protocol Security Paper (arXiv:2504.16902)](https://arxiv.org/abs/2504.16902)
- [CSA Threat Modeling for A2A](https://cloudsecurityalliance.org/blog/2025/04/30/threat-modeling-google-s-a2a-protocol-with-the-maestro-framework)
- [Linux Foundation A2A Project](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents)

### マルチエージェントセキュリティ関連
- [Prompt Infection in Multi-Agent Systems (arXiv:2410.07283)](https://arxiv.org/abs/2410.07283)
- [Multi-Agent LLM Defense Pipeline (arXiv:2509.14285)](https://arxiv.org/abs/2509.14285)
- [Prompt Injection Detection via Multi-Agent NLP (arXiv:2503.11517)](https://arxiv.org/abs/2503.11517)
- [OWASP LLM Top 10 2025](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)

### LLM-as-a-Judge関連
- [A Survey on LLM-as-a-Judge (arXiv:2411.15594)](https://arxiv.org/abs/2411.15594)
- [LLMs-as-Judges: A Comprehensive Survey (arXiv:2412.05579)](https://arxiv.org/abs/2412.05579)

### AI安全性評価関連
- [UK AISI Inspect Evals](https://www.aisi.gov.uk/blog/inspect-evals)
- [UK AISI Early Lessons from Evaluating Frontier AI](https://www.aisi.gov.uk/work/early-lessons-from-evaluating-frontier-ai-systems)

### 信頼スコア・エージェント経済関連
- [WEF: Trust in AI Agent Economy](https://www.weforum.org/stories/2025/07/ai-agent-economy-trust/)
- [Dynamic Trust Score with LLMs](https://thesciencebrigade.com/JAIR/article/view/611)
- [Cleanlab: Benchmarking Real-time Trust Scoring](https://cleanlab.ai/blog/agent-tlm-hallucination-benchmarking/)

### 防御手法関連
- [Microsoft Indirect Prompt Injection Defense](https://www.microsoft.com/en-us/msrc/blog/2025/07/how-microsoft-defends-against-indirect-prompt-injection-attacks)
- [Amazon Bedrock Agent Security Guide](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/)

### 市場動向
- [CB Insights AI Agent Market Map 2025](https://www.cbinsights.com/research/ai-agent-market-map/)

---

*最終更新: 2025年12月*
