# 🎬 デモ・実行スクリプト集

このディレクトリには、Secure AI Agent Matching Platformのデモ実行に必要なスクリプトとガイドが含まれています。

## 📚 主なファイル

- **README.md** (このファイル) - スクリプトの使い方
- **DEMO_GUIDE.md** - 詳細なデモガイド（日本語）
- **demo.sh** - メインデモスクリプト
- **\*.py** - Python デモスクリプト

## 🔧 環境セットアップ

```bash
# 1. uvをインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. プロジェクトルートで依存関係をインストール
cd /path/to/secure-ai-agent-matching-platform
uv sync

# 3. 環境変数を設定
# ⚠️ Geniac Prize審査員の方: 別途共有しているGemini API Keyをご利用ください
echo "GOOGLE_API_KEY=your-api-key" > secure-mediation-agent/.env

# 4. デモを実行
./demo/demo.sh
```

## 📋 スクリプト一覧

### 🚀 メインデモスクリプト

#### `demo.sh`
全てのエージェント（外部エージェント3つ + Mediation Agent）を一度に起動します。

**使い方:**
```bash
# demoディレクトリから実行
cd demo
./demo.sh

# またはプロジェクトルートから実行
./demo/demo.sh
```

**機能:**
- ✅ 環境変数の自動読み込み (.env)
- ✅ ポートの自動チェック（8002, 8003, 8004, 8000）
- ✅ 外部エージェントの起動と確認
  - Airline Agent (port 8002)
  - Hotel Agent (port 8003)
  - Car Rental Agent (port 8004)
- ✅ Mediation Agent Web UIの起動 (port 8000)
- ✅ ブラウザの自動起動
- ✅ デモプロンプトの表示
- ✅ Ctrl+Cで全サービス一括停止

---

### 🛑 停止スクリプト

#### `stop_all.sh`
起動中の全エージェントを停止します。

**使い方:**
```bash
./demo/stop_all.sh
```

**機能:**
- PIDファイルから停止
- ポート番号から強制停止（フォールバック）
- 確実にすべてのプロセスを終了

---

### 🔍 テストスクリプト

#### `quick_test.sh`
全エージェントの接続確認を行います。

**使い方:**
```bash
./demo/quick_test.sh
```

**機能:**
- A2Aエンドポイントの確認
- エージェント情報の表示
- Web UIの起動確認

---

### 🔧 個別起動スクリプト

#### `start_agents.sh`
外部エージェントのみを起動します（Mediation Agentは起動しません）。

**使い方:**
```bash
./demo/start_agents.sh
```

#### `run_web.sh`
Mediation Agent Web UIのみを起動します。

**使い方:**
```bash
./demo/run_web.sh
```

#### `run_cli.sh`
Mediation Agent CLIモードで起動します。

**使い方:**
```bash
./demo/run_cli.sh
```

---

## 🎯 推奨デモフロー

### 前提条件
- プロジェクトルートにいることを確認してください
- `.env`ファイルに`GOOGLE_API_KEY`が設定されていることを確認してください

### Step 1: デモ環境を起動

**プロジェクトルートから実行:**
```bash
# 現在地を確認
pwd
# 出力: /Users/.../secure-ai-agent-matching-platform

# デモスクリプトを実行（必ず ./ を付ける）
./demo/demo.sh
```

**⚠️ 注意:** `demo.sh` だけではエラーになります。必ず `./demo/demo.sh` と実行してください。

### Step 2: 別ターミナルで接続確認（オプション）
```bash
./demo/quick_test.sh
```

### Step 3: Web UI (http://localhost:8000) でエージェントを選択してデモプロンプトを入力

#### 3-1. エージェントの選択
ブラウザでWeb UIが開いたら、まず**secure-mediation-agent**を選択してください：

Web UI左側のエージェント一覧から「**secure-mediation-agent**」を選択


![エージェント選択画面](../docs/images/select-agent.png)

**⚠️ 重要:** secure-mediation-agent が選択されていないと、セキュリティチェックが実行されません。

#### 3-2. デモプロンプトの入力

**デモプロンプト例:**
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

### Step 4: 実行を確認

Web UIで以下を観察:
- 🔄 サブエージェントの呼び出し
- 🛡️ セキュリティチェック
- 📝 実行プランの作成
- ✅ 最終結果とACCEPT/REJECT判定

### Step 5: 停止
```bash
./demo/stop_all.sh
```

または demo.sh 実行中のターミナルで `Ctrl+C`

---

## 📚 ログファイル

各エージェントのログは `/tmp` に保存されます:

```bash
# ログの確認
tail -f /tmp/airline-agent.log
tail -f /tmp/hotel-agent.log
tail -f /tmp/car-agent.log
tail -f /tmp/mediation-agent.log
```

---

## 🌐 エンドポイント一覧

| サービス | URL | A2A Endpoint |
|---------|-----|--------------|
| Airline Agent | http://localhost:8002 | http://localhost:8002/.well-known/agent.json |
| Hotel Agent | http://localhost:8003 | http://localhost:8003/.well-known/agent.json |
| Car Rental Agent | http://localhost:8004 | http://localhost:8004/.well-known/agent.json |
| Mediation Agent Web UI | http://localhost:8000 | - |

---

## ⚠️ トラブルシューティング

### ポートが既に使用中
```bash
# 使用中のポートを確認
lsof -i :8000
lsof -i :8002

# プロセスを終了
./demo/stop_all.sh
```

### 環境変数が読み込まれない
```bash
# .envファイルが存在するか確認
ls -la secure-mediation-agent/.env

# GOOGLE_API_KEYが設定されているか確認
cat secure-mediation-agent/.env | grep GOOGLE_API_KEY
```

### エージェントが起動しない
```bash
# ログを確認
tail -f /tmp/airline-agent.log
tail -f /tmp/hotel-agent.log
tail -f /tmp/car-agent.log
tail -f /tmp/mediation-agent.log
```

---

## 🔐 セキュリティ機能のデモ

デモ環境では以下のセキュリティ機能を確認できます:

1. **Matcher**: エージェントの信頼性スコア評価
2. **Planner**: セキュリティを考慮した実行プラン作成
3. **Orchestrator**: A2A通信の安全な実行
4. **Anomaly Detector**: リアルタイム異常検知
5. **Final Anomaly Detector**:
   - プロンプトインジェクション検出
   - ハルシネーション検出
   - 最終ACCEPT/REJECT判定

---

## 📝 注意事項

- 全スクリプトはプロジェクトルートから実行可能
- demo.sh は自動的にブラウザを開きます
- Ctrl+C で安全に全サービスを停止できます
- PIDファイルは `.airline.pid`, `.hotel.pid`, `.car.pid`, `.mediation.pid` に保存されます
