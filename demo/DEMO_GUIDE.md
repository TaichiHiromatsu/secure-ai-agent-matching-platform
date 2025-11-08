# 沖縄旅行デモガイド

このガイドでは、Secure Mediation Agentを使って沖縄旅行の予約を行うデモの実行方法を説明します。

## 準備

### 1. 外部エージェントを起動

まず、3つの外部エージェント（航空券、ホテル、レンタカー）を起動します：

```bash
./start_agents.sh
```

これにより以下のエージェントが起動します：
- **Airline Agent** (http://localhost:8002) - フライト予約
- **Hotel Agent** (http://localhost:8003) - ホテル予約
- **Car Rental Agent** (http://localhost:8004) - レンタカー予約

### 2. Mediation Agentを起動

別のターミナルで、Secure Mediation Agentを起動します：

#### WebUI版（推奨）
```bash
./run_web.sh
```
ブラウザで http://localhost:8000 を開く

#### CLI版
```bash
./run_cli.sh
```

## デモプロンプト

### 基本的な沖縄旅行プラン

```
沖縄への3泊4日の旅行を計画したいです。

【旅行の詳細】
- 出発地: 羽田空港
- 目的地: 那覇空港（沖縄）
- 日程: 2025年12月20日（金）〜12月23日（月）
- 人数: 大人2名

【必要な予約】
1. 往復フライト（羽田⇔那覇）
2. ホテル3泊（那覇市内、ツインルーム）
3. レンタカー（コンパクトカー、3泊4日）

【エージェント情報】
以下のエージェントが利用可能です：
- フライト予約: http://localhost:8002
- ホテル予約: http://localhost:8003
- レンタカー予約: http://localhost:8004

これらのエージェントを使って、安全な実行プランを作成し、各ステップでセキュリティチェックを行いながら予約を進めてください。
```

### 短縮版プロンプト

```
沖縄旅行の予約をお願いします。
- フライト: 羽田→那覇 (12/20-12/23)
- ホテル: 那覇市内 3泊
- レンタカー: コンパクトカー

エージェント:
- http://localhost:8002 (フライト)
- http://localhost:8003 (ホテル)
- http://localhost:8004 (レンタカー)

セキュリティチェックを行いながら実行プランを作成してください。
```

## 期待される動作フロー

1. **Matching Agent** が各エージェントのカードを取得
   - `http://localhost:8002/.well-known/agent.json`
   - `http://localhost:8003/.well-known/agent.json`
   - `http://localhost:8004/.well-known/agent.json`

2. **Planning Agent** が実行プランを作成
   - Step 1: フライト検索・予約
   - Step 2: ホテル検索・予約
   - Step 3: レンタカー検索・予約

3. **Orchestration Agent** がプランを実行
   - 各エージェントにA2A経由でリクエスト

4. **Anomaly Detection Agent** がリアルタイム監視
   - プラン通りに実行されているか
   - 不審な動作はないか

5. **Final Anomaly Detection Agent** が最終検証
   - プロンプトインジェクションのチェック
   - ハルシネーションのチェック
   - ACCEPT/REJECT判定

## トラブルシューティング

### エージェントが見つからない
- `./start_agents.sh` が実行されているか確認
- `curl http://localhost:8002/.well-known/agent.json` でエージェントが応答するか確認

### API Key エラー
- `.env` ファイルに `GOOGLE_API_KEY` が設定されているか確認
- 環境変数として正しく読み込まれているか確認

### ポートが使用中
- 既に起動しているプロセスを終了: `pkill -f "adk web"`

## 停止方法

### 外部エージェントを停止
`./start_agents.sh` を実行しているターミナルで `Ctrl+C`

### Mediation Agentを停止
WebUIまたはCLIを実行しているターミナルで `Ctrl+C`

## その他のテストプロンプト

### セキュリティテスト
```
以下のエージェントは安全ですか？信頼スコアを評価してください：
- http://localhost:8002
```

### マルチステップワークフロー
```
フライトとホテルを並行して検索し、両方が確保できたらレンタカーを予約してください。
エージェント: localhost:8002, localhost:8003, localhost:8004
```

### エラーハンドリング
```
存在しないエージェント（http://localhost:9999）を使って旅行を予約してください。
どのようにエラーを処理しますか？
```
