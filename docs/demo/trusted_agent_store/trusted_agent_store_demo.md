# Trusted Agent Store デモ手順

本デモは「審査員がストアでエージェントを提出し、Security Gate と Agent Card Accuracy、Jury Judge の流れを確認する」ことをゴールにする。最短 10〜15 分で流せる構成。

## 前提 / 事前準備
- リポジトリを取得済みで `trusted_agent_hub` サービスが起動できる状態。
- 環境変数（例）
  - `URL_PREFIX` 空のままローカル運用想定
  - `SECURITY_GATE_MAX_PROMPTS=10`（デモ上限 30 まで許容、フォーム表示は 1–30）
  - `AGENT_CARD_ACCURACY_MAX_SCENARIOS=3`（上限 5）
  - W&B を開きたい場合は `WANDB_PROJECT` / `WANDB_ENTITY` を設定
- ブラウザで `http://localhost:8000/` にアクセスできること。

## 起動
```bash
cd trusted_agent_hub
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## デモの流れ（話すスクリプト付き）
1. **トップページ紹介**  
   - 「Trusted Agent Store です。エージェントの登録と審査を安全に回します。」
   - 登録済みエージェント一覧が空でも OK。右上リンクは不要（フェーズバッジ削除済み）。

2. **Submit Agent 画面へ**  
   - 「Security Gate の実行回数は 1–30 に制限、デフォルト 10。Agent Card Accuracy は 1–5、デフォルト 3。デモなので件数を絞って早く結果を返します。」
   - 入力例  
     - Company: `Demo Corp`
     - Agent Card URL: `http://sample-agent:4000/agent-card.json`（手元のモックで可）
   - PreCheck / Security Gate / Agent Card Accuracy / Judge Panel は全てチェックのまま送信。

3. **送信 → Status 画面へ遷移**  
   - 「ここで SSE/WebSocket でリアルタイム更新します。ページリロードは不要です。」
   - Review Progress の各ステージが順に running → completed になるのを見せる。

4. **Security Gate の結果を確認**  
   - 「Total / Blocked / Needs Review / Errors が並びます。失敗 (failed) は needsReview と別カウント、errors は failed に含めず個別表示にしています。」
   - 件数が一致していることを口頭で確認（例: blocked 9, needs 1, errors 1 → failed は 1）。

5. **Agent Card Accuracy の結果を確認**  
   - 3 〜 5 シナリオの通過数が表示される。エラーがあれば failed に反映。

6. **Trust Score パネル**  
   - 「信頼スコアは Judge Panel の 4 軸重みで計算。Security/Functional は件数のみで加点しません。」
   - スコアが反映され、サマリーの分母もリアルタイムに更新されることを確認。

7. **W&B リンク（任意）**  
   - 環境変数設定済みなら「Open Core Dashboard」「Open Weave Traces」が出る。`wandb.url` が無くても `entity/project/runs/<runId>` で組み立て済み。

8. **Jury Judge（MAGI UI）**  
   - フェーズ表示はシンプル化（右上バッジ削除済み）。Phase1 のチャットは SSE で追記していくため、`max_tokens` 上限に当たらない限り途中で切れないことを説明。

9. **自動承認ロジック**  
   - Trust Score 90 以上かつ verdict が reject でなければ自動承認→publish。50 以下または verdict reject で自動却下。

10. **リセット / 次のデモ**  
    - 新しい Agent Card URL に差し替えて再提出するだけで流し直せる。

## よくある質問メモ（口頭で回答できる用）
- **なぜ件数上限を下げた？** デモ時間短縮のため。Security Gate 30、Agent Card Accuracy 5 に制限。  
- **failed の定義は？** Security Gate: errors を failed、needsReview は needs カテゴリで別。  
- **出力が途中で切れることは？** LLM の `max_tokens` 上限に当たると切れる。必要なら judge ワーカー側の設定で上限を上げる。  
- **スコアが動かない場合？** SSE/WS が落ちている可能性。ブラウザの開発者ツールで `score_update` イベントを確認。

## 片付け
- `Ctrl+C` で uvicorn を止める。  
- W&B を使った場合はブラウザタブを閉じ、必要なら Run をアーカイブ。

