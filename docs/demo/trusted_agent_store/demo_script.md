# Trusted Agent Store デモ台本（Geniac Prize 審査向け）

所要 8〜10 分。ブラウザのみで実行可能。

## 0. アクセス・ログイン（1分）
- URL: https://secure-mediation-a2a-platform-343404053218.asia-northeast1.run.app/store
- 共有済みの審査用 ID / パスワードでログイン。
- ゴールを宣言：提出 → 自動審査 → 信頼スコア反映 → 公開一覧 を一気見せ。

## 1. 事業者登録（1分）
- 左上「事業者登録」で会社名・メール・URL を入力し送信。
- 「環境構築なしでブラウザだけ」を強調。
- まだ未実装だが、登録した事業者のみ申請できるような機能を提供予定

## 2. エージェント提出（2分）
- 「Submit Agent」を開く。
- 入力例:
  - Company: `Demo Corp`
  - Agent Card URL: `https://secure-mediation-a2a-platform-343404053218.asia-northeast1.run.app/a2a/sales_agent/.well-known/agent-card.json`
  - Security Gate prompts: `10`
  - Agent Card Accuracy scenarios: `3`
- 4チェック（PreCheck / Security Gate / Agent Card Accuracy / Judge Panel）はオンのまま Submit。
- ポイント: A2A カード URL だけで審査が走る簡便さ。

## 3. 審査ライブ閲覧（3分）
- Status 画面で SSE 進行を見せる。
- Security Gate: Total/Blocked/Needs Review/Errors がリアルタイム更新。空応答等は即 needs_review へ。
- Agent Card Accuracy: シナリオ通過数。
- MAGI / Jury Judge: GPT‑4o / Claude / Gemini の合議と Final Judge、4軸スコア + verdict。
- Trust Score: 即時計算し自動承認判定（90 以上 auto_approved, 50 以下 auto_rejected）。

## 4. 公開一覧確認（1分）
- 「Registered Agents」で `sales_agent` が active で掲載され、Use Cases と Trust Score が反映されていることを確認。
- 審査結果がストア公開情報に即反映される点を強調。

## 5. ログの透明性（任意 1分）
- Core / Weave リンクを示し、審査期間中は公開ビューで詳細トレースを確認できると一言。終了後は Private に戻す運用。

## 6. 繰り返し可能性（30秒）
- 同じ URL で何度でも提出し直せる。別の Agent Card でも即比較デモが可能と締める。

## キーメッセージ（話すときの要点）
- 「カード URL を貼るだけでクラウド上で多層審査が走る」
- 「結果が Trust Score としてストアに即反映され、利用者は安全なエージェントだけを選べる」
- 「透明性：審査ログ（Core/Weave）を公開している」
