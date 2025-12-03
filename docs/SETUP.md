# セットアップ手順書（Mac OS）

> **Geniac Prize審査員向け再現手順書** - Geniac Prize 2025

このドキュメントは、**動作確認環境パターン2（macOS Sequoia 15.3.2 / Apple M3 Max）** でセキュアAIエージェントマッチングプラットフォームをゼロからセットアップし、デモを実行するための詳細な手順を提供します。

## 🖥️ 動作確認環境（パターン2）

このシステムは以下の環境で検証済みです：
- **OS**: macOS Sequoia 15.3.2 (x64)
- **プロセッサ**: Apple M3 Max (x64)
- **メモリ**: 36GB
- **ストレージ**: Macintosh HD 1TB
- **システムの種類**: 64ビット オペレーティング システム、x64ベース プロセッサ
- **利用ブラウザ**: Microsoft Edge for Business (公式ビルド) (arm64)

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

# 期待される出力（動作確認環境パターン2）:
# ProductName:    macOS
# ProductVersion: 15.3.2
# BuildVersion:   24D2075
```

**動作確認済みバージョン**: macOS Sequoia 15.3.2 (x64)
**最小要件**: macOS 12.0 (Monterey) 以降

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

# 期待される出力（動作確認環境パターン2）:
# Python 3.12.2 以上
```

**動作確認済みバージョン**: Python 3.12.2
**最小要件**: Python 3.12.0 以上

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
secure_mediation_agent/
external-agents/
demo/
docs/
SPECIFICATION.md
README.md
```

### 2.2 Google Gemini API キーの取得

**⚠️ Geniac Prize審査員の方**: 別途共有しているGemini API Keyを使用してください。以下の手順はスキップできます。

**一般の方向け**:
1. [Google AI Studio](https://makersuite.google.com/app/apikey) にアクセス
2. 「Create API Key」をクリック
3. APIキーをコピー

### 2.3 環境変数の設定

```bash
# .envファイルを作成（secure_mediation_agent配下に）
cat > secure_mediation_agent/.env << EOF
GOOGLE_API_KEY=your-actual-api-key-here
EOF

# 確認
cat secure_mediation_agent/.env
```

**⚠️ Geniac Prize審査員の方**: `your-actual-api-key-here` を別途共有しているGemini API Keyに置き換えてください。

---

## 3. 依存関係のインストール

### 3.1 uvのインストール

**Geniac Prize審査員の方: uvを使用した環境構築を推奨します。**

```bash
# uvをインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# シェルを再起動するか、パスを読み込む
source ~/.zshrc  # または ~/.bash_profile

# 確認
uv --version
```

### 3.2 依存関係のインストール

```bash
# プロジェクトルートで依存関係をインストール
cd /path/to/secure-ai-agent-matching-platform
uv sync

# 確認
uv pip list
```

**インストールされるパッケージ**:
- `google-adk`: Google Agent Development Kit（メインフレームワーク）
- `a2a-server`: A2A Protocol実装（エージェント間通信に必須）
- `httpx`: HTTPクライアント（外部エージェント呼び出しに使用）

その他の依存関係は`pyproject.toml`と`uv.lock`により自動管理されます。

---

## 4. エージェントの起動

### 4.1 デモスクリプトを使用した起動（推奨）

**最も簡単な方法**: `demo.sh`スクリプトを使用

```bash
# プロジェクトルートから実行
./demo/demo.sh
```

このスクリプトは以下を自動的に実行します：
- ✅ 環境変数の読み込み
- ✅ ポートの使用状況チェック
- ✅ 外部エージェント3つの起動（航空、ホテル、レンタカー）
- ✅ Mediation Agent Web UIの起動
- ✅ ブラウザの自動起動

**停止方法**: 起動したターミナルで `Ctrl+C` を押す、または別ターミナルで `./demo/stop_all.sh` を実行

### 4.2 手動での起動方法（詳細確認用）

**合計4つのターミナルウィンドウ/タブを開く場合:**

**全てのターミナルで以下を実行**:
```bash
cd /path/to/secure-ai-agent-matching-platform
```

#### ターミナル1: 航空会社エージェント（ポート 8002）

```bash
uv run adk api_server --a2a --port 8002 external-agents/trusted-agents/airline-agent
```

#### ターミナル2: ホテルエージェント（ポート 8003）

```bash
uv run adk api_server --a2a --port 8003 external-agents/trusted-agents/hotel-agent
```

#### ターミナル3: レンタカーエージェント（ポート 8004）

```bash
uv run adk api_server --a2a --port 8004 external-agents/trusted-agents/car-rental-agent
```

#### ターミナル4: Mediation Agent Web UI（ポート 8000）

```bash
uv run adk web secure_mediation_agent --port 8000
```

**注意**: 各エージェントの起動には20-30秒かかる場合があります。

---

## 5. 動作確認

### 5.1 エージェントカードの確認

新しいターミナル（ターミナル5）で以下を実行:

```bash
# 仲介エージェント
curl http://localhost:8000/.well-known/agent-card.json | jq

# 航空会社エージェント
curl http://localhost:8002/a2a/airline_agent/.well-known/agent-card.json | jq

# ホテルエージェント
curl http://localhost:8002/a2a/hotel_agent/.well-known/agent-card.json | jq

# レンタカーエージェント
curl http://localhost:8002/a2a/car_rental_agent/.well-known/agent-card.json | jq
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

# 会話履歴を確認
ls -la artifacts/conversations/
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
   curl http://localhost:8000/.well-known/agent-card.json
   curl http://localhost:8002/a2a/airline_agent/.well-known/agent-card.json
   curl http://localhost:8002/a2a/hotel_agent/.well-known/agent-card.json
   curl http://localhost:8002/a2a/car_rental_agent/.well-known/agent-card.json
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
rm -rf artifacts/conversations/*
```

---

## 📞 サポート

問題が解決しない場合:

1. [トラブルシューティング](#トラブルシューティング) を再確認
2. GitHubのIssueで報告
3. 標準出力ログを添付

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
