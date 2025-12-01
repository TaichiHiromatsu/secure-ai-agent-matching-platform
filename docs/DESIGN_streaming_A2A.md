# Streaming A2A & Safety Relaxation Design

目的: A2A で 3 エージェント協議と Jury Judge ディベートを低遅延ストリーミングで可視化しつつ、Gemini 安全フィルターによるブロックを最小化する。

## 1. モデル選択とフォールバック階層
- 優先: `gemini-2.5-flash`（低遅延・標準ストリーム対応）
- フォールバック: `gemini-2.5-pro` → `gemini-1.5-pro-002`
- Live API オプション: 双方向/音声リアルタイムが必要な場合のみ `gemini-live-2.5-flash` 系へルーティング
- モデル選択とブロック理由をメトリクス送信（model, block_category, latency_ms, retry_flag）

## 2. 安全設定プリセット
全 HarmCategory を `BLOCK_NONE`（運用上の最小許容閾値）で統一するプリセットを共通利用。
```python
safety_settings_relaxed = [
    SafetySetting(category=HARM_CATEGORY_HARASSMENT, threshold=BLOCK_NONE),
    SafetySetting(category=HARM_CATEGORY_HATE_SPEECH, threshold=BLOCK_NONE),
    SafetySetting(category=HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=BLOCK_NONE),
    SafetySetting(category=HARM_CATEGORY_DANGEROUS_CONTENT, threshold=BLOCK_NONE),
    SafetySetting(category=HARM_CATEGORY_CIVIC_INTEGRITY, threshold=BLOCK_NONE),
    SafetySetting(category=HARM_CATEGORY_JAILBREAK, threshold=BLOCK_NONE),
]
```
- A2A 協議用サブエージェント（orchestrator / matcher / planning_agent ほか）と RemoteA2aAgent 呼び出しの `GenerateContentConfig` に適用。
- safety block 発生時は 1 回だけ閾値を維持したままモデルをフォールバックし再実行。再ブロックで fail-safe へ。

## 3. API 変更（サーバ/エージェント側）
- `invoke_a2a_agent(..., stream=True, on_chunk: Callable[[str], Awaitable[None]])` を追加。
- `runner.run_async` から受け取る event を即時 `on_chunk` に流しつつ、従来通り `conversation_history` には蓄積（セキュリティ検査・永続化用）。
- 非ストリーム互換: `stream=False` 時は現行のバッファ結合を維持。
- ストリーム経路では tool_calls / function_calls も chunk として逐次送出（type と payload を明示）。

## 4. Jury Judge ディベート画面設計（リアルタイムチャット）
### 表示要素
- タイムライン型チャット: 左から時系列、メッセージ毎に「モデル名 / 役割（陪審員A,B,C, 最終判事）」バッジ + タイムスタンプ。
- ストリーム中は「typing dots」インジケーターを表示し、チャンク到着毎に追記（部分レンダー）。
- 安全フィルターイベント: ブロック発生時は赤ラベルでカテゴリと閾値を即時表示、再試行時は黄ラベル。
- モデル切替トースト: フォールバック発動時に「gemini-2.5-flash → gemini-2.5-pro (safety block)」を右上トーストで提示。
- ツール/関数コール: 折りたたみの小バッジで関数名を表示し、展開すると引数 JSON を閲覧可能。
- スクロール固定/解除トグル: 長い議論でも最新メッセージを追えるよう下部に「Auto-scroll」スイッチ。

### ストリーミング実装メモ
- API: SSE または WebSocket を採用。チャンクイベントのスキーマ例:
```json
{
  "type": "message_chunk",
  "role": "juror_b",
  "model": "gemini-2.5-flash",
  "content": "部分テキスト...",
  "timestamp": 1700000000.123,
  "sequence": 42
}
```
- `type: "tool_call"` / `"tool_response"` / `"safety_block"` / `"model_switch"` などを同一ストリームで送る。
- フロントは sequence 昇順で追記し、同 role の連続 chunk は一つの気泡にマージ（段落区切りは `\n\n`）。
- タイピングインジケーターは 1 秒チャンク無通信で点滅、5 秒無通信で「再試行中」バナーを表示。

## 5. 低遅延のための運用ルール
- 3エージェント協議ステップでは常に `stream=True` を要求し、UI もストリームに依存した進行を前提化。
- プロンプトは「指示フェーズ」と「評価フェーズ」に分割し、各フェーズのトークン数を抑制。
- 同一セッションで複数質問を連続実行し、セッション確立オーバーヘッドを削減。

## 6. ロギングと監査
- 受信チャンクをそのまま `conversation_history` に記録（既存フロー維持）。
- メトリクス: latency_per_chunk, total_latency, chunks_count, safety_block_count, fallback_count を計測。
- ブロック理由は category/probability を保持し、ダッシュボードでフィルタリング可能に。

## 7. 変更対象ファイル（実装時の目安）
- `secure-mediation-agent/subagents/orchestration_agent.py` : `stream`/`on_chunk` 追加、safety プリセット適用。
- `secure-mediation-agent/agent.py` および各 subagent: `GenerateContentConfig` を共通プリセット参照に変更。
- `external-agents/trusted-agents/*`（デモ用）: 同プリセットを適用し挙動を合わせる。
- UI クライアント（Jury Judge ビュー）: SSE/WS クライアントとチャットタイムライン実装、トースト/インジケーター追加。

## 8. 接続方式の決定（SSE 採用）
- Cloud Run での片方向リアルタイム配信に最適なため **SSE を正式採用**。UI は EventSource を利用し、双方向が必要になった場合のみオプションで WebSocket を提供する。
- サーバは `text/event-stream` で `Cache-Control: no-cache`, `X-Accel-Buffering: no` を付与し、25s 毎に `event: ping` を送信。
- イベントスキーマは JSON Line（`data: {...}`）で統一し、フロントが sequence 昇順でレンダーする。
- WebSocket ルーターは削除済み（後方互換を持たない構成）。

## 9. 残タスク
- イベントスキーマ型定義を共通ライブラリに切り出し済み（`app/schemas/jury_stream.py`）だが、送り手/受け手の未対応部分を適用する。
- Live API を併用する場合の権限確認。  
- 計測メトリクスを既存監視に統合（latency, block_count, fallback_count）。  
