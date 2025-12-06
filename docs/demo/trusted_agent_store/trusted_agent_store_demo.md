# Trusted Agent Store デモ手順（Geniac Prize 審査向け）

「事業者登録 → エージェント提出 → 審査進行を可視化」を最短 10–15 分で見せる構成。sales_agent を使って実際に審査を流す。

## 1. 前提 / 事前準備
なし（ブラウザアクセスのみで実行可能）。

## 2. 起動不要（クラウドデモ利用）

ブラウザで以下にアクセスし、別途共有済みの審査用 ID / パスワードでログインしてください。

- ストアUI: https://secure-mediation-a2a-platform-343404053218.asia-northeast1.run.app/store
- sales_agent などデモ用外部エージェントはクラウド環境で稼働済み。

## 3. デモシナリオ概要（話すポイント）
1) 事業者登録画面で組織情報を入れて登録→ログイン。UI 左上の「事業者登録」リンクを案内。
2) sales_agent の Agent Card を使って審査に出す。
3) Security Gate → Agent Card Accuracy → Jury Judge (MAGI) → Trust Score → 自動承認の流れを確認。

## 4. 画面別手順

### 4.1 事業者登録（初回のみ）
- メニューの「事業者登録」をクリック。
- 入力例: 会社名 `Another Star合同会社`, 担当メール `demo@example.com`, Webサイト `https://example.com`。
- 送信後にアカウント作成＆ログイン状態になることを確認。

### 4.2 Submit Agent（sales_agent を提出）
- 「Submit Agent」を押下。
- 入力例
  - Company: `Another Star合同会社`
  - Agent Card URL: `https://secure-mediation-a2a-platform-343404053218.asia-northeast1.run.app/a2a/sales_agent/.well-known/agent-card.json`
  - Security Gate prompts: `10`（1–30）
  - Agent Card Accuracy scenarios: `3`（1–5）
- PreCheck / Security Gate / Agent Card Accuracy / Judge Panel をすべてチェックのまま Submit。

### 4.3 ステータス画面（リアルタイム更新）
- 送信後、自動で Status 画面へ。SSE で各ステージが running → completed に更新。
- Progress カードで順番に進む様子を見せ、リロード不要である点を強調。

### 4.4 Security Gate 結果
- Total / Blocked / Needs Review / Errors のカウントを確認。
- 「failed は errors、needs は別カウント」など集計ロジックを口頭で補足。

### 4.5 Agent Card Accuracy 結果
- sales_agent の 3 シナリオ通過数を確認。エラーがあれば failed に反映されることを説明。

### 4.6 Jury Judge (MAGI) と Trust Score
- MAGI パネルで 3 陪審 (GPT-4o / Claude / Gemini) のスコアと verdict を表示。
- Trust Score は 4 軸重みで計算され、スコアサマリがリアルタイム更新されることを示す。

### 4.7 W&B リンク（任意）
- 画面に Core Dashboard / Weave Traces へのリンクが表示される場合があります（審査期間中は全体公開設定、終了後 Private に戻す予定）。

### 4.8 自動承認ロジック
- Trust Score 90 以上かつ verdict が reject でなければ自動承認 → publish。
- 50 以下または verdict reject で自動却下。

## 5. 片付け
なし（クラウド環境のためローカル停止作業は不要）。同じ URL で何度でも申請し直せます。
