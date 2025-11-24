# CODEX Review – Security Gate Stage (submissions.py)

## 必ず修正すべきブロッカー
- セキュリティゲート失敗が成功扱いになる: 例外時も`stages.security`と`state`を`completed`/後続工程へ進め、誤承認のリスク (`trusted_agent_hub/app/routers/submissions.py`:307-341, 371-378)。
- 0.0.0.0正規化で不正URL生成: ポート未指定でも`service_name:{parsed_endpoint.port}`を組み立て、`None`入りのURLを使用する恐れ (`.../submissions.py`:277-306)。
- データセット存在チェックなし: ハードコードCSV欠落時に例外→上記「失敗を成功扱い」に吸収される。実行前に存在検証しステージ失敗として扱うべき (`.../submissions.py`:259-261)。

## 高優先度
- URL欠如時の記録不足: `state="failed"`にするだけで`stages.security`やW&Bに理由が残らず運用から見えない。失敗ステージとメッセージを保存すること (`.../submissions.py`:262-275)。
- サマリ辞書前提でクラッシュしうる: `security_summary`が`None`/非dictだと`.get`で例外、パイプライン停止 (`.../submissions.py`:333-337)。防御的デフォルトを入れる。
- `needsReview`/`notExecuted`を成功扱い: 未実行・要確認があっても`status=completed`でスコア計算・次工程へ進む。非ゼロならスコア付与や自動判定を保留/失敗扱いにすべき (`.../submissions.py`:339-378)。

## 中優先度（可観測性）
- 失敗時にW&Bへ記録されない: 成功パスのみログ/アーティファクト保存。失敗時もエラー内容と途中成果物を送るようにして追跡可能性を確保 (`.../submissions.py`:323-325)。

## メモ
- 実装編集は行っていません。上記を優先順に対応してください。
