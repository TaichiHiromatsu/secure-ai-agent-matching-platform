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
  - Agent Card URL: 下記から選択
  - Security Gate prompts: `10`
  - Agent Card Accuracy scenarios: `3`
- 4チェック（PreCheck / Security Gate / Agent Card Accuracy / Judge Panel）はオンのまま Submit。
- ポイント: A2A カード URL だけで審査が走る簡便さ。

### サンプルエージェント Agent Card URL

| エージェント | URL | 期待結果 |
|-------------|-----|---------|
| 営業支援 | `http://localhost:8002/a2a/sales_agent/.well-known/agent-card.json` | 高スコアで承認 |
| 多言語翻訳 | `http://localhost:8002/a2a/translation_agent/.well-known/agent-card.json` | 高スコアで承認 |
| 悪意あり（テスト用） | `http://localhost:8002/a2a/data_harvester_agent/.well-known/agent-card.json` | 低スコアで却下 |

※ **デモでは悪意のあるエージェント（data_harvester_agent）を使うと、Security Gateでブロックされる様子を見せられます。**

※ クラウド環境の場合は `localhost:8002` を `https://secure-mediation-a2a-platform-343404053218.asia-northeast1.run.app` に置き換えてください。

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

---

# デモ読み上げ台本

## 導入（30秒）

「それでは、Trusted Agent Store のデモをご覧いただきます。

このストアは、AIエージェントを安全に公開・利用するためのマーケットプレイスです。
エージェント開発者がAgent CardのURLを提出するだけで、自動的に多層審査が実行され、信頼スコアが算出されます。

今日は、実際に悪意のあるエージェントを提出して、セキュリティゲートでブロックされる様子をお見せします。」

---

## 画面を開く（30秒）

「まず、ストアの画面を開きます。

こちらがTrusted Agent Storeのトップページです。
左側にメニューがあり、Submit Agent からエージェントを提出できます。

では、Submit Agent をクリックします。」

---

## エージェント提出（1分）

「こちらがエージェント提出画面です。

まず、会社名を入力します。Demo Corp としましょう。

次に、Agent Card URLを入力します。
今日は、悪意のあるエージェントをテストしてみます。
このエージェントは、表向きはデータ分析エージェントですが、実際にはユーザーのクレジットカード番号やパスワードを聞き出そうとする悪意のある設計になっています。

URLをコピーして貼り付けます。

審査オプションはすべてオンのままにしておきます。
- PreCheck: エージェントカードの基本検証
- Security Gate: 有害プロンプト耐性テスト
- Agent Card Accuracy: 記載内容と実動作の整合性確認
- Judge Panel: 3つのLLM陪審員による信頼性評価

では、Submit をクリックします。」

---

## 審査進行を見る（2分）

「審査が始まりました。リアルタイムで進行状況が更新されます。

まず PreCheck が実行されています。これはエージェントカードの必須フィールドや接続確認を行います。
...完了しました。

次に Security Gate です。これが今日の注目ポイントです。
悪意のあるプロンプトをエージェントに投げかけて、どう反応するかテストします。

ご覧ください。Blocked の数が増えていきます。
このエージェントは、個人情報を聞き出そうとする設計なので、セキュリティテストでブロックされています。

Total 10件中、Blocked が...（数を読み上げ）件です。
これは、このエージェントが有害な応答を返したことを示しています。

次に Agent Card Accuracy です。
エージェントカードに書かれている機能と、実際の動作が一致するか確認します。

最後に Judge Panel、MAGIシステムです。
GPT-4o、Claude、Geminiの3つのLLMが陪審員として、このエージェントを評価します。

各陪審員のスコアと判定が表示されています。
...ご覧の通り、verdict が reject になっています。
3つのLLMすべてが、このエージェントは信頼できないと判断しました。」

---

## 結果確認（1分）

「Trust Score を見てください。

50点以下になっており、自動的に却下されました。
ステータスが auto_rejected になっています。

このように、悪意のあるエージェントは自動審査で検出され、ストアへの公開がブロックされます。

一方、正規のエージェント、例えば営業支援エージェントを提出すると、高いスコアで承認され、ストアに公開されます。」

---

## まとめ（30秒）

「まとめです。

Trusted Agent Store は、Agent Card の URL を貼るだけで多層審査が自動実行されます。
Security Gate で有害プロンプト耐性をテストし、MAGI システムで複数のLLMが合議して信頼性を評価します。

結果は Trust Score として即座にストアに反映され、利用者は安全なエージェントだけを選べるようになります。

すべての審査ログは透明性のため公開しており、後から確認することも可能です。

以上で、Trusted Agent Store のデモを終わります。ご質問があればお願いします。」
