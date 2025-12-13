# ローカル環境での実行（開発者向け）

このドキュメントでは、ローカル環境でプラットフォームを実行する方法を説明します。

## 環境要件

- **OS**: macOS 12.0 以降
- **Python**: 3.12 以上
- **Homebrew**: インストール済み
- **Google Cloud アカウント**: Gemini API キー取得済み

## セットアップ手順

```bash
# 1. リポジトリをクローン
git clone <repository-url>
cd secure-ai-agent-matching-platform

# 2. uvをインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 依存関係をインストール
uv sync

# 4. 環境変数を設定
echo "GOOGLE_API_KEY=your-gemini-api-key" > .env

# 5. ローカル実行スクリプトで全サービスを起動
./deploy/run-local.sh

# ✅ ブラウザで http://localhost:8080 を開きます
```

## デモプロンプト例

```
沖縄旅行の予約をお願いします。
- 人数：2人
- フライト: 羽田→那覇 (12/20-12/23)
- ホテル: 那覇市内 3泊
- レンタカー: コンパクトカー

セキュリティチェックを行いながら実行プランを作成してください。
```

## トラブルシューティング

### ポートが使用中の場合

```bash
# 使用中のポートを確認
lsof -i :8080

# プロセスを終了
kill -9 <PID>
```

### 依存関係のエラー

```bash
# キャッシュをクリアして再インストール
uv cache clean
uv sync --refresh
```

## 関連ドキュメント

- [デモ操作手順](demo/DEMO.md)
- [アーキテクチャ詳細](secure_mediation_agent_design/ARCHITECTURE.md)
