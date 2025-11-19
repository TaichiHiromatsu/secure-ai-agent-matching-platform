# Trusted Agent Store - Standalone Edition

完全に独立して動作するTrusted Agent Storeの単一コンテナ版です。このディレクトリをどこにでもコピーして起動できます。

## 特徴

- **単一コンテナ**: すべての機能が1つのPythonコンテナに統合
- **完全独立**: 外部依存なし、このディレクトリだけで起動可能
- **FastAPI + SQLite**: 軽量で高速なアーキテクチャ
- **Cloud Run Ready**: Google Cloud Runへ簡単にデプロイ可能

## ディレクトリ構造

```
trusted_agent_store/
├── app/                        # FastAPIアプリケーション
│   ├── main.py                 # エントリーポイント
│   ├── database.py             # SQLiteデータベース
│   ├── routers/                # APIルーティング
│   ├── services/               # ビジネスロジック
│   └── templates/              # Jinja2テンプレート
├── static/                     # 静的ファイル (CSS等)
├── lib/                        # 内部依存ライブラリ
│   ├── sandbox-runner/         # エージェント実行環境
│   └── inspect-worker/         # セキュリティ検査ツール
├── Dockerfile                  # コンテナ定義
├── requirements.txt            # Python依存関係
└── README.md                   # このファイル
```

## クイックスタート

### 方法1: Dockerで起動（推奨）

```bash
# このディレクトリに移動
cd trusted_agent_store

# Dockerイメージをビルド
docker build -t trusted-agent-store .

# コンテナを起動
docker run -p 8080:8080 -v $(pwd)/data:/app/data trusted-agent-store

# ブラウザで開く
open http://localhost:8080
```

### 方法2: Pythonで直接実行

```bash
# このディレクトリに移動
cd agent_store

# 依存関係をインストール
pip install -r requirements.txt
pip install ./lib/sandbox-runner

# サーバー起動
uvicorn app.main:app --reload --port 8080

# ブラウザで開く
open http://localhost:8080
```

## Cloud Runへのデプロイ

```bash
# Google Cloudプロジェクトを設定
export PROJECT_ID=your-project-id
export REGION=asia-northeast1

# このディレクトリからデプロイ
gcloud run deploy agent-store \
  --source . \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated

# デプロイされたURLを確認
gcloud run services describe agent-store \
  --region $REGION \
  --format 'value(status.url)'
```

## 環境変数

| 変数名 | デフォルト | 説明 |
|--------|-----------|------|
| `DATABASE_URL` | `sqlite:////app/data/agent_store.db` | データベース接続URL |
| `PORT` | `8080` | サーバーポート |

## API エンドポイント

### 提出API
- `POST /api/submissions` - エージェント登録
- `GET /api/submissions/{id}` - 提出詳細取得
- `GET /api/submissions/{id}/progress` - 審査進捗確認
- `GET /api/submissions` - 提出一覧

### レビューAPI
- `POST /api/review/{id}/approve` - 承認
- `POST /api/review/{id}/reject` - 差戻し
- `POST /api/review/{id}/decision` - 判定送信

### UI
- `GET /` - ホーム
- `GET /submit` - エージェント登録画面
- `GET /review/{id}` - 審査状況詳細
- `GET /submissions` - 提出一覧

## データの永続化

### ローカル実行時

データベースファイルは `./agent_store.db` に保存されます。

### Docker実行時

ボリュームマウントでデータを永続化:

```bash
docker run -p 8080:8080 -v $(pwd)/data:/app/data agent-store
```

### Cloud Run実行時

Cloud SQLへの移行を推奨します:

```bash
# DATABASE_URL環境変数を設定
gcloud run deploy agent-store \
  --source . \
  --region $REGION \
  --set-env-vars DATABASE_URL=postgresql://user:pass@host/db
```

## トラブルシューティング

### ポートが使用中

```bash
# 別のポートで起動
docker run -p 8081:8080 -v $(pwd)/data:/app/data agent-store
```

### データベースをリセット

```bash
# Docker実行時
rm -rf data/agent_store.db

# ローカル実行時
rm -rf agent_store.db
```

### ログ確認

```bash
# Docker実行時
docker logs <container_id>

# ローカル実行時
uvicorn app.main:app --reload --log-level debug
```

## 開発

### 依存関係の追加

```bash
# requirements.txtに追加してから
pip install -r requirements.txt
```

### テストの実行

```bash
pytest tests/
```

## アーキテクチャ

従来の9コンテナ構成から1コンテナに削減:

| コンポーネント | 従来 | 現在 |
|--------------|------|------|
| 認証 | auth-service | Cloud Run IAM |
| データベース | PostgreSQL x2 | SQLite |
| ワークフロー | Temporal | BackgroundTasks |
| API | Node.js | FastAPI |
| UI | Next.js x2 | Jinja2 |

## 制限事項

- SQLiteのため、大規模データや高並列アクセスには向きません
- Temporalの高度なワークフロー機能は簡素化されています
- 本番環境では Cloud SQL + より強力なインスタンスを推奨

## ライセンス

元のAgent Storeプロジェクトと同じライセンスに従います。

## 備考

このディレクトリ(`trusted_agent_store`)は完全に独立しており、外部にコピーするだけで起動できます。
