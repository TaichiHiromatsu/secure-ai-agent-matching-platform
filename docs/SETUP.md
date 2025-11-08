# セットアップ手順書（Mac OS）

> **審査員向け再現手順書** - Geniac Prize 2025

このドキュメントは、Mac OS環境でセキュアAIエージェントマッチングプラットフォームをゼロからセットアップし、デモを実行するための詳細な手順を提供します。

## 📋 目次

1. [前提条件の確認](#前提条件の確認)
2. [環境セットアップ](#環境セットアップ)
3. [依存関係のインストール](#依存関係のインストール)
4. [エージェントの起動](#エージェントの起動)
5. [動作確認](#動作確認)
6. [トラブルシューティング](#トラブルシューティング)

---

## 1. 前提条件の確認

### 1.1 オペレーティングシステム

```bash
# macOSのバージョン確認
sw_vers

# 期待される出力例:
# ProductName:    macOS
# ProductVersion: 14.0
# BuildVersion:   23A344
```

**必要なバージョン**: macOS 12.0 (Monterey) 以降

### 1.2 Homebrewのインストール確認

```bash
# Homebrewがインストールされているか確認
brew --version

# インストールされていない場合は以下を実行
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 1.3 Pythonのインストール確認

```bash
# Pythonバージョン確認
python3 --version

# 期待される出力: Python 3.12.0 以上
```

Python 3.12がインストールされていない場合:

```bash
# Homebrewでインストール
brew install python@3.12

# パスを追加（~/.zshrc または ~/.bash_profile に追記）
echo 'export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# 確認
python3.12 --version
```

---

## 2. 環境セットアップ

### 2.1 リポジトリのクローン

```bash
# 任意のディレクトリに移動
cd ~/Development

# リポジトリをクローン
git clone <repository-url>
cd secure-ai-agent-matching-platform

# ディレクトリ構造を確認
ls -la
```

**期待される出力**:
```
secure-mediation-agent/
external-agents/
demo/
docs/
SPECIFICATION.md
README.md
```

### 2.2 Google Gemini API キーの取得

1. [Google AI Studio](https://makersuite.google.com/app/apikey) にアクセス
2. 「Create API Key」をクリック
3. APIキーをコピー

### 2.3 環境変数の設定

```bash
# .envファイルを作成
cat > .env << EOF
GOOGLE_API_KEY=your-actual-api-key-here
GOOGLE_GENAI_USE_VERTEXAI=0
EOF

# 環境変数を読み込み
export $(cat .env | xargs)

# 確認
echo $GOOGLE_API_KEY
```

**重要**: `your-actual-api-key-here` を実際のAPIキーに置き換えてください。

---

## 3. 依存関係のインストール

### 3.1 仮想環境の作成（推奨）

```bash
# 仮想環境を作成
python3.12 -m venv venv

# 仮想環境を有効化
source venv/bin/activate

# 確認（venvが表示されるはず）
which python
```

### 3.2 Google ADKのインストール

```bash
# pipをアップグレード
pip install --upgrade pip

# Google ADKをインストール
pip install google-adk

# 確認
adk --version
```

**期待される出力**: `adk version x.x.x`

### 3.3 その他の依存関係

```bash
# httpxをインストール（A2A通信用）
pip install httpx

# インストール済みパッケージを確認
pip list
```

**必要なパッケージ**:
- `google-adk`
- `httpx`
- `google-generativeai`

---

## 4. エージェントの起動

### 4.1 ターミナルの準備

**合計5つのターミナルウィンドウ/タブを開いてください:**

1. ターミナル1: セキュア仲介エージェント
2. ターミナル2: 航空会社エージェント
3. ターミナル3: ホテルエージェント
4. ターミナル4: レンタカーエージェント
5. ターミナル5: デモ実行用

**全てのターミナルで以下を実行**:
```bash
cd /path/to/secure-ai-agent-matching-platform
source venv/bin/activate
export $(cat .env | xargs)
```

### 4.2 エージェントの起動手順

#### ターミナル1: セキュア仲介エージェント（ポート 8001）

```bash
adk serve secure-mediation-agent/agent.py --port 8001
```

**期待される出力**:
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

#### ターミナル2: 航空会社エージェント（ポート 8002）

```bash
adk serve external-agents/trusted-agents/airline-agent/agent.py --port 8002
```

#### ターミナル3: ホテルエージェント（ポート 8003）

```bash
adk serve external-agents/trusted-agents/hotel-agent/agent.py --port 8003
```

#### ターミナル4: レンタカーエージェント（ポート 8004）

```bash
adk serve external-agents/trusted-agents/car-rental-agent/agent.py --port 8004
```

**注意**: 各エージェントの起動には20-30秒かかる場合があります。"Application startup complete" が表示されるまで待ってください。

---

## 5. 動作確認

### 5.1 エージェントカードの確認

新しいターミナル（ターミナル5）で以下を実行:

```bash
# 仲介エージェント
curl http://localhost:8001/.well-known/agent.json | jq

# 航空会社エージェント
curl http://localhost:8002/.well-known/agent.json | jq

# ホテルエージェント
curl http://localhost:8003/.well-known/agent.json | jq

# レンタカーエージェント
curl http://localhost:8004/.well-known/agent.json | jq
```

**期待される出力**: 各エージェントのA2Aエージェントカード（JSON形式）

`jq`がインストールされていない場合:
```bash
brew install jq
```

### 5.2 シンプルデモの実行

```bash
python demo/simple_demo.py
```

**期待される出力**: エージェントからの応答が表示される

### 5.3 沖縄旅行デモの実行

```bash
python demo/okinawa_trip_demo.py
```

**期待される結果**:
1. ✅ ユーザーの要望が表示される
2. ✅ 仲介エージェントが処理を開始
3. ✅ マッチングされたエージェント一覧
4. ✅ 実行プランが生成・保存される
5. ✅ 各エージェントとの通信ログ
6. ✅ 最終的な予約確認情報

### 5.4 生成されたアーティファクトの確認

```bash
# 実行プランを確認
ls -la artifacts/plans/
cat artifacts/plans/*.md

# 実行ログを確認
ls -la artifacts/logs/
cat artifacts/logs/execution.log
```

---

## 6. トラブルシューティング

### 問題1: `ModuleNotFoundError: No module named 'google.adk'`

**原因**: Google ADKがインストールされていない

**解決策**:
```bash
pip install google-adk
```

### 問題2: `Error: API key not found`

**原因**: 環境変数が設定されていない

**解決策**:
```bash
# 環境変数を再設定
export GOOGLE_API_KEY="your-api-key"

# または.envファイルを再読み込み
export $(cat .env | xargs)
```

### 問題3: `Address already in use`

**原因**: ポートが既に使用されている

**解決策**:
```bash
# 使用中のプロセスを確認
lsof -i :8001

# プロセスを終了
kill -9 <PID>
```

### 問題4: エージェントが起動しない

**原因**: Pythonバージョンが古い

**解決策**:
```bash
# Pythonバージョンを確認
python3 --version

# Python 3.12以上が必要
brew install python@3.12
```

### 問題5: `ImportError` が発生する

**原因**: パッケージの依存関係が壊れている

**解決策**:
```bash
# 仮想環境を削除して再作成
deactivate
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate

# パッケージを再インストール
pip install google-adk httpx
```

### 問題6: デモが動作しない

**確認事項チェックリスト**:

1. ✅ 全4つのエージェントが起動していますか？
   ```bash
   curl http://localhost:8001/.well-known/agent.json
   curl http://localhost:8002/.well-known/agent.json
   curl http://localhost:8003/.well-known/agent.json
   curl http://localhost:8004/.well-known/agent.json
   ```

2. ✅ 環境変数は設定されていますか？
   ```bash
   echo $GOOGLE_API_KEY
   ```

3. ✅ 仮想環境は有効化されていますか？
   ```bash
   which python  # venv/bin/python が表示されるべき
   ```

---

## 7. クリーンアップ

デモ終了後、リソースをクリーンアップ:

```bash
# 1. 全てのエージェントを停止（各ターミナルで Ctrl+C）

# 2. 仮想環境を無効化
deactivate

# 3. （オプション）生成されたアーティファクトを削除
rm -rf artifacts/plans/*
rm -rf artifacts/logs/*
```

---

## 📞 サポート

問題が解決しない場合:

1. [トラブルシューティング](#トラブルシューティング) を再確認
2. GitHubのIssueで報告
3. ログファイル (`artifacts/logs/`) を添付

---

## ✅ チェックリスト

セットアップ完了の確認:

- [ ] Python 3.12+ がインストールされている
- [ ] Google ADKがインストールされている
- [ ] 環境変数が設定されている
- [ ] 4つのエージェントが全て起動している
- [ ] エージェントカードが取得できる
- [ ] デモが正常に実行できる
- [ ] アーティファクトが生成されている

全てチェックが入れば、準備完了です！🎉

---

**次のステップ**: [デモ実行手順](DEMO.md) を参照してください。
