# Stage-based Multi-Model Judge Panel × MCTS 統合メモ (2025-11-24)

## 背景
- 基本設計: `stage-based-multi-model-judge-panel.md`（Plan / Counter / Reconcile の3ステージ、マルチモデル評価）。
- 追加要求: `agent-as-judge-evaluation-design.md` にある **MCTS (Monte Carlo Tree Search) ベースの合意形成** を取り込む。
- 現状実装: `sandbox_runner/judge_panel.py` は 3 ステージの逐次評価のみ。MCTS 的な分岐・反証探索やステージ固有プロンプトは未実装。

## 現状 vs. 目標のギャップ
| 項目 | 現状実装 | 目標（設計書） |
| --- | --- | --- |
| ステージ | Plan/Counter/Reconcile を逐次評価（perspective のみ切替） | ステージごとに **専用プロンプト** を適用し、LLMごとのスコア・判定・理由を保持 |
| 合意形成 | 各ステージの aggregated_verdict を集計し、reject>0 なら reject  | **MCTS-style 合意形成**：プラン→反証→再評価の木を展開し、少数否決 (Minority Veto) + 多数決で統合 |
| 反証サイクル | なし（単発評価） | Counter ステージで見つかった懸念を Reconcile ステージが参照し、必要なら再質問・再評価（rollout） |
| 出力 | `judge_report.jsonl` にステージ別 aggregated 結果のみ | ステージ別 LLM verdicts + MCTS ノードの探索ログ（rollout 深さ、シミュレーション回数、最終 UCB スコアなど） |
| 探索パラメータ | なし | `max_rollouts`, `ucb_c`, `max_depth`, `veto_threshold` を設定可能に |

## 統合方針（実装計画）
1) **ステージ固有プロンプトの明示化**
   - Plan / Counter / Reconcile 用のプロンプトテンプレートを `inspect_worker` 側に揃え、`perspective` フィールドだけでなく system 部分も差し替える。

2) **MCTS Orchestrator 追加**
   - 新モジュール案: `sandbox_runner/mcts_orchestrator.py`
   - ノード = {question, response, stage, verdicts[]}
   - 展開: Plan → Counter (反証) → Reconcile（再評価）
   - ロールアウト: Reconcile 結果をシミュレーションスコアとして UCB1 で選択
   - Minority Veto: 子ノードに reject ≥ veto_threshold の場合即 reject

3) **judge_panel.py から呼び出すフロー**
   - 既存の `_run_stage_multi_model_judge_panel` をラップし、MCTS orchestrator が複数回呼び出す形にリファクタ。
   - 出力を `judge_report.jsonl`（ステージ×ロールアウト）と `judge_summary.json`（最終 verdict + UCB/rollout 指標）に拡張。

4) **設定値**
   - `JUDGE_MCTS_MAX_ROLLOUTS` (default: 3)
   - `JUDGE_MCTS_MAX_DEPTH` (default: 3: Plan→Counter→Reconcile)
   - `JUDGE_MCTS_UCB_C` (default: 1.4)
   - `JUDGE_VETO_THRESHOLD` (default: 0.3)

5) **ログ & デバッグ**
   - 各ノード展開時に `node_id`, `parent_id`, `stage`, `model`, `verdict`, `score`, `ucb` を `judge_report.jsonl` に記録。
   - `docker logs` で追えるよう、INFO ログに要約を出力。

## 作業ステップ（提案）
1. `inspect_worker` にステージ別プロンプトテンプレートを追加し、`MultiModelJudgePanel.evaluate_panel_async` に stage 引数を渡す。
2. `sandbox_runner` に MCTS orchestrator を実装（シンプル UCB1 + rollouts）。
3. `run_judge_panel` で orchestrator を呼び、既存のステージ単発評価を置き換える。
4. レポートスキーマを拡張（ステージ×ロールアウト×モデル）。UI への影響を最小化するため、従来フィールドは維持しつつ `mcts` 配下に詳細を格納。
5. 最低限のユニットテスト（ダミー LLMSandbox）で UCB 選択と veto 動作を確認。

## まとめ
- ステージ分割は実装済みだが、MCTS 合意形成とステージ固有プロンプトは未実装。
- 本メモの手順で MCTS を組み込み、設計書と実装を一致させる。
