# ラウンド→ターン変更の詳細設計ドキュメント

## 概要

Jury Judge Phase 2の議論をラウンドベースから継続的なターンベースに変更する。

**変更日**: 2025-11-30
**ステータス**: 設計完了・実装待ち

---

## 1. 現状分析

### 現在の動作（ラウンドベース）
```
Round 1:
  - Juror A が発言
  - Juror B が発言
  - Juror C が発言
  - 合意チェック

Round 2:
  - Juror A が発言
  - Juror B が発言
  - Juror C が発言
  - 合意チェック

Round 3:
  - Juror A が発言
  - Juror B が発言
  - Juror C が発言
  - 合意チェック → Phase 3へ
```

### 問題点
- ✗ ラウンドごとに区切られており、連続的な議論になっていない
- ✗ UIに「Round 1」「Round 2」と表示される
- ✗ WebSocketイベントが `round_started` / `round_completed` として発火
- ✗ 環境変数が `JURY_MAX_DISCUSSION_ROUNDS` という名前
- ✗ データモデルが `DiscussionRound` というクラス名

---

## 2. 目標とする動作（ターンベース）

### 新しい動作
```
Turn 1: Juror A が発言
Turn 2: Juror B が発言
Turn 3: Juror C が発言
→ 合意チェック（3ターンごと）

Turn 4: Juror A が発言
Turn 5: Juror B が発言
Turn 6: Juror C が発言
→ 合意チェック（3ターンごと）

Turn 7: Juror A が発言
Turn 8: Juror B が発言
Turn 9: Juror C が発言
→ 合意チェック → 最大ターンまたは合意達成で終了
```

### 改善点
- ✓ 連続的な会話フローを実現
- ✓ ラウンドの概念を完全削除
- ✓ 合意チェックは3ターンごと（全員が1回ずつ発言した後）
- ✓ 最大9ターン（各陪審員が最大3回発言）
- ✓ 合意に達したら即座に終了

---

## 3. 環境変数の変更

### 変更内容

**削除（エラーとする）:**
```bash
JURY_MAX_DISCUSSION_ROUNDS=3  # ← この変数は使用不可
```

**新規（必須）:**
```bash
JURY_MAX_DISCUSSION_TURNS=9
```

### エラーハンドリング

**ファイル**: `trusted_agent_hub/evaluation-runner/src/evaluation_runner/jury_judge.py`

**実装**:
```python
# 環境変数から設定を読み込む
use_collaborative = os.environ.get("JURY_USE_COLLABORATIVE", "true").lower() == "true"

# 古い環境変数のチェック（エラーとする）
if os.environ.get("JURY_MAX_DISCUSSION_ROUNDS") is not None:
    raise ValueError(
        "JURY_MAX_DISCUSSION_ROUNDS is deprecated. "
        "Please use JURY_MAX_DISCUSSION_TURNS instead. "
        "See docs/round-to-turn-migration.md for migration guide."
    )

# 新しい環境変数を使用
max_discussion_turns = int(os.environ.get("JURY_MAX_DISCUSSION_TURNS", "9"))
consensus_threshold = float(os.environ.get("JURY_CONSENSUS_THRESHOLD", "2.0"))
final_judgment_method = os.environ.get("JURY_FINAL_JUDGMENT_METHOD", "final_judge")
final_judge_model = os.environ.get("JURY_FINAL_JUDGE_MODEL", "gemini-2.5-pro")

# Collaborative Jury Judgeを初期化
jury_judge = CollaborativeJuryJudge(
    max_discussion_turns=max_discussion_turns,
    consensus_threshold=consensus_threshold,
    stagnation_threshold=2,
    final_judgment_method=final_judgment_method,
    final_judge_model=final_judge_model,
)
```

---

## 4. データモデルの変更

### 4.1 新しいデータ構造

**ファイル**: `trusted_agent_hub/jury-judge-worker/jury_judge_worker/jury_judge_collaborative.py`

**削除:**
```python
@dataclass
class DiscussionRound:
    round_number: int
    statements: List[JurorStatement]
    consensus_check: ConsensusResult
    speaker_order: List[str]
```

**新規追加:**
```python
@dataclass
class DiscussionResult:
    """Phase 2議論の結果"""
    total_turns: int
    discussion_messages: List[Dict[str, Any]]  # turn, juror_id, statement, timestamp
    final_consensus: ConsensusResult
    early_termination: bool
    speaker_order: List[str]  # 発言順序（固定）
```

### 4.2 CollaborativeJuryResult の変更

**変更前:**
```python
@dataclass
class CollaborativeJuryResult:
    phase1_evaluations: List[JurorEvaluation]
    phase1_consensus: ConsensusResult
    discussion_rounds: List[DiscussionRound]  # ← 削除
    total_rounds: int  # ← 削除
    phase3_judgment: FinalJudgment
    final_verdict: str
    final_score: int
    early_termination: bool
```

**変更後:**
```python
@dataclass
class CollaborativeJuryResult:
    phase1_evaluations: List[JurorEvaluation]
    phase1_consensus: ConsensusResult
    discussion_result: DiscussionResult  # ← 新規
    total_turns: int  # ← 新規
    phase3_judgment: FinalJudgment
    final_verdict: str
    final_score: int
    early_termination: bool
```

---

## 5. CollaborativeJuryJudge クラスの変更

### 5.1 コンストラクタの変更

**ファイル**: `trusted_agent_hub/jury-judge-worker/jury_judge_worker/jury_judge_collaborative.py`

**変更前:**
```python
def __init__(
    self,
    max_discussion_rounds: int = 3,
    consensus_threshold: float = 2.0,
    stagnation_threshold: int = 2,
    final_judgment_method: str = "final_judge",
    final_judge_model: str = "gemini-2.5-pro",
):
    self.max_discussion_rounds = max_discussion_rounds
    self.consensus_threshold = consensus_threshold
    self.stagnation_threshold = stagnation_threshold
    self.final_judgment_method = final_judgment_method
    self.final_judge_model = final_judge_model
```

**変更後:**
```python
def __init__(
    self,
    max_discussion_turns: int = 9,
    consensus_threshold: float = 2.0,
    stagnation_threshold: int = 2,
    final_judgment_method: str = "final_judge",
    final_judge_model: str = "gemini-2.5-pro",
):
    self.max_discussion_turns = max_discussion_turns
    self.num_jurors = 3  # 陪審員数（固定）
    self.consensus_threshold = consensus_threshold
    self.stagnation_threshold = stagnation_threshold
    self.final_judgment_method = final_judgment_method
    self.final_judge_model = final_judge_model
```

### 5.2 Phase 2 Discussion ループの完全リファクタリング

**変更前（推定構造）:**
```python
async def _phase2_discussion(self, ...):
    discussion_rounds = []

    for round_num in range(1, self.max_discussion_rounds + 1):
        # WebSocket: round_started
        await self._emit_ws("round_started", {"round": round_num, ...})

        statements = []
        for juror_id in speaker_order:
            statement = await self._generate_statement(juror_id, ...)
            statements.append(statement)
            # WebSocket: juror_statement
            await self._emit_ws("juror_statement", {"round": round_num, ...})

        # WebSocket: round_completed
        await self._emit_ws("round_completed", {"round": round_num, ...})

        # 合意チェック
        consensus = self._check_consensus(...)

        discussion_rounds.append(DiscussionRound(
            round_number=round_num,
            statements=statements,
            consensus_check=consensus,
            speaker_order=speaker_order
        ))

        if consensus.consensus_reached:
            break

    return discussion_rounds
```

**変更後:**
```python
async def _phase2_discussion(
    self,
    submission_id: str,
    phase1_evaluations: List[JurorEvaluation],
    speaker_order: List[str],
    agent_card_data: Dict[str, Any],
    conversation_log: List[Dict[str, Any]],
) -> DiscussionResult:
    """
    Phase 2: 継続的なターンベース議論

    各陪審員が順番に発言し、3ターンごとに合意をチェック。
    合意に達するか最大ターン数に達するまで継続。
    """
    current_turn = 0
    discussion_messages = []
    consensus = None

    logger.info(f"[Phase 2] Starting turn-based discussion (max {self.max_discussion_turns} turns)")

    # ターン制ループ（最大max_discussion_turns回）
    while current_turn < self.max_discussion_turns:
        current_turn += 1
        juror_index = (current_turn - 1) % self.num_jurors
        juror_id = speaker_order[juror_index]

        logger.info(f"[Phase 2] Turn {current_turn}: {juror_id} is speaking")

        # 陪審員の発言を生成
        statement = await self._generate_discussion_statement(
            juror_id=juror_id,
            turn_number=current_turn,
            discussion_history=discussion_messages,
            phase1_evaluations=phase1_evaluations,
            agent_card_data=agent_card_data,
            conversation_log=conversation_log,
        )

        # メッセージを記録
        discussion_messages.append({
            "turn": current_turn,
            "juror_id": juror_id,
            "statement": statement.content,
            "position": statement.position,
            "confidence": statement.confidence,
            "revised_score": statement.revised_score,
            "timestamp": datetime.now().isoformat()
        })

        # WebSocket: juror_statement（ターン情報のみ）
        await self._emit_ws("juror_statement", {
            "submissionId": submission_id,
            "turn": current_turn,
            "jurorId": juror_id,
            "statement": statement.content,
            "position": statement.position,
            "confidence": statement.confidence,
            "revisedScore": statement.revised_score,
        })

        # 3ターンごとに合意チェック（全員が1回ずつ発言した後）
        if current_turn % self.num_jurors == 0:
            consensus = await self._check_consensus_after_turns(
                turn_number=current_turn,
                discussion_messages=discussion_messages,
                phase1_evaluations=phase1_evaluations,
            )

            logger.info(
                f"[Phase 2] Consensus check after turn {current_turn}: "
                f"{consensus.consensus_status.value} (agreement: {consensus.agreement_level:.2f})"
            )

            # WebSocket: consensus_check
            await self._emit_ws("consensus_check", {
                "submissionId": submission_id,
                "turn": current_turn,
                "consensusStatus": consensus.consensus_status.value,
                "agreementLevel": consensus.agreement_level,
                "majorityPosition": consensus.majority_position,
            })

            # 合意に達した場合は終了
            if consensus.consensus_reached:
                logger.info(f"[Phase 2] Consensus reached at turn {current_turn}. Ending discussion.")
                break

    # 最終的な合意状態（ループが最大ターンで終了した場合は最後のチェック結果）
    if consensus is None:
        # 最大ターン数に達したが3の倍数でない場合、最後に合意チェック
        consensus = await self._check_consensus_after_turns(
            turn_number=current_turn,
            discussion_messages=discussion_messages,
            phase1_evaluations=phase1_evaluations,
        )

    return DiscussionResult(
        total_turns=current_turn,
        discussion_messages=discussion_messages,
        final_consensus=consensus,
        early_termination=consensus.consensus_reached,
        speaker_order=speaker_order,
    )
```

### 5.3 新しい合意チェックメソッド

```python
async def _check_consensus_after_turns(
    self,
    turn_number: int,
    discussion_messages: List[Dict[str, Any]],
    phase1_evaluations: List[JurorEvaluation],
) -> ConsensusResult:
    """
    指定されたターン数の後に合意をチェック

    Args:
        turn_number: 現在のターン番号
        discussion_messages: これまでの議論メッセージ
        phase1_evaluations: Phase 1の評価結果

    Returns:
        ConsensusResult: 合意状況
    """
    # 最新3ターン（全陪審員の最新発言）を取得
    recent_turns = discussion_messages[-self.num_jurors:] if len(discussion_messages) >= self.num_jurors else discussion_messages

    # 各陪審員の最新の立場を抽出
    latest_positions = {}
    latest_scores = {}

    for msg in recent_turns:
        juror_id = msg["juror_id"]
        latest_positions[juror_id] = msg["position"]
        latest_scores[juror_id] = msg.get("revised_score") or self._get_phase1_score(juror_id, phase1_evaluations)

    # 合意レベルを計算
    positions = list(latest_positions.values())
    unique_positions = set(positions)

    if len(unique_positions) == 1:
        # 全員一致
        consensus_status = ConsensusStatus.UNANIMOUS
        agreement_level = 1.0
        consensus_reached = True
        majority_position = positions[0]
    elif len([p for p in positions if p == max(set(positions), key=positions.count)]) >= 2:
        # 多数派形成（3人中2人以上が同じ立場）
        majority_position = max(set(positions), key=positions.count)
        consensus_status = ConsensusStatus.MAJORITY
        agreement_level = 2.0 / 3.0
        # consensus_thresholdと比較
        consensus_reached = (agreement_level >= self.consensus_threshold)
    else:
        # 意見が分かれている
        consensus_status = ConsensusStatus.SPLIT
        agreement_level = 1.0 / 3.0
        consensus_reached = False
        majority_position = None

    return ConsensusResult(
        consensus_status=consensus_status,
        agreement_level=agreement_level,
        consensus_reached=consensus_reached,
        majority_position=majority_position,
        turn_number=turn_number,
    )

def _get_phase1_score(self, juror_id: str, phase1_evaluations: List[JurorEvaluation]) -> int:
    """Phase 1評価から陪審員のスコアを取得"""
    for ev in phase1_evaluations:
        if ev.juror_id == juror_id:
            return ev.overall_score
    return 0
```

---

## 6. WebSocketイベントの変更

### 6.1 削除するイベント

**完全削除:**
- `round_started` - ラウンド開始イベント
- `round_completed` - ラウンド完了イベント

### 6.2 変更するイベント

**`juror_statement` イベント:**

**変更前:**
```javascript
{
    "submissionId": "abc123",
    "round": 1,
    "jurorId": "juror-gpt",
    "statement": "...",
    "position": "safe_pass",
    "confidence": 0.85,
    "revisedScore": 75
}
```

**変更後:**
```javascript
{
    "submissionId": "abc123",
    "turn": 1,  // ← roundからturnに変更
    "jurorId": "juror-gpt",
    "statement": "...",
    "position": "safe_pass",
    "confidence": 0.85,
    "revisedScore": 75
}
```

### 6.3 新規追加イベント

**`consensus_check` イベント:**
```javascript
{
    "submissionId": "abc123",
    "turn": 3,  // 合意チェックが実行されたターン番号
    "consensusStatus": "unanimous",  // unanimous, majority, split
    "agreementLevel": 1.0,  // 0.33, 0.67, 1.0
    "majorityPosition": "safe_pass"  // 多数派の立場
}
```

---

## 7. UIテンプレートの変更

### 7.1 HTMLの変更

**ファイル**: `trusted_agent_hub/app/templates/partials/submission_content.html`

**Line 703: ラウンド情報の削除**

**変更前:**
```html
<div id="current-round-info" class="text-xs text-gray-600 mb-2"></div>
```

**変更後:**
```html
<div id="discussion-status" class="text-xs text-gray-600 mb-2 flex items-center gap-4">
    <span id="turn-counter" class="font-semibold">
        Turn <span id="current-turn" class="text-blue-600">0</span> / <span id="max-turns">9</span>
    </span>
    <span id="consensus-indicator" class="text-sm"></span>
</div>
```

### 7.2 JavaScriptイベントハンドラの変更

**Line 1271-1284: `round_started` ハンドラの削除**

**削除:**
```javascript
juryWS.on('round_started', (data) => {
    console.log('[DEBUG] 🎯 round_started event handler called with data:', data);

    const container = document.getElementById('discussion-container');
    const roundInfo = document.getElementById('current-round-info');

    if (container && roundInfo) {
        container.classList.remove('hidden');
        const speakerOrderText = data.speakerOrder && Array.isArray(data.speakerOrder)
            ? ` - 発言順: ${data.speakerOrder.join(', ')}`
            : '';
        roundInfo.textContent = `Round ${data.round}${speakerOrderText}`;
    }
});
```

**Line 1287-: `juror_statement` ハンドラの修正**

**変更前:**
```javascript
juryWS.on('juror_statement', (data) => {
    console.log('[DEBUG] 🗣️ juror_statement event handler called with data:', data);

    // ラウンド情報を使用してメッセージを表示
    const roundInfo = `Round ${data.round}`;
    // ...
});
```

**変更後:**
```javascript
juryWS.on('juror_statement', (data) => {
    console.log('[DEBUG] 🗣️ juror_statement event handler called with data:', data);

    // ターンカウンターを更新
    const turnCounter = document.getElementById('current-turn');
    if (turnCounter) {
        turnCounter.textContent = data.turn;
    }

    // ディスカッションコンテナを表示
    const discussionContainer = document.getElementById('discussion-container');
    if (discussionContainer) {
        discussionContainer.classList.remove('hidden');
    }

    // メッセージを追加
    const messagesContainer = document.getElementById('discussion-messages');
    if (messagesContainer) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'bg-white p-3 rounded border border-gray-200';
        messageDiv.innerHTML = `
            <div class="flex items-start justify-between mb-2">
                <div class="flex items-center gap-2">
                    <span class="text-xs font-semibold text-gray-500">Turn ${data.turn}</span>
                    <span class="text-sm font-semibold text-gray-800">${getJurorName(data.jurorId)}</span>
                    <span class="text-lg">${getJurorEmoji(data.jurorId)}</span>
                </div>
            </div>
            <div class="text-sm text-gray-700 mb-2 whitespace-pre-wrap">${escapeHtml(data.statement)}</div>
            <div class="flex items-center gap-3 text-xs text-gray-500">
                <span>Position: <strong class="${getPositionClass(data.position)}">${data.position || 'N/A'}</strong></span>
                <span>Confidence: <strong>${data.confidence ? (data.confidence * 100).toFixed(0) + '%' : 'N/A'}</strong></span>
                ${data.revisedScore ? `<span>Score: <strong>${data.revisedScore}</strong></span>` : ''}
            </div>
        `;
        messagesContainer.appendChild(messageDiv);

        // スクロールを最新メッセージに
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
});

// ヘルパー関数
function getJurorName(jurorId) {
    const names = {
        'juror-gpt': 'GPT-4o',
        'juror-claude': 'Claude 3.5',
        'juror-gemini': 'Gemini 2.5'
    };
    return names[jurorId] || jurorId;
}

function getJurorEmoji(jurorId) {
    const emojis = {
        'juror-gpt': '🤖',
        'juror-claude': '🧠',
        'juror-gemini': '💎'
    };
    return emojis[jurorId] || '👤';
}

function getPositionClass(position) {
    const classes = {
        'safe_pass': 'text-green-600',
        'needs_review': 'text-yellow-600',
        'unsafe_fail': 'text-red-600'
    };
    return classes[position] || 'text-gray-600';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
```

**新規追加: `consensus_check` ハンドラ**

```javascript
juryWS.on('consensus_check', (data) => {
    console.log('[DEBUG] 🤝 consensus_check event handler called with data:', data);

    // 合意状況インジケーターを更新
    const indicator = document.getElementById('consensus-indicator');
    if (indicator) {
        if (data.consensusStatus === 'unanimous') {
            indicator.innerHTML = '<span class="text-green-600 font-semibold">✓ 全員一致</span>';
        } else if (data.consensusStatus === 'majority') {
            indicator.innerHTML = '<span class="text-yellow-600 font-semibold">⚠ 多数派形成 (2/3)</span>';
        } else if (data.consensusStatus === 'split') {
            indicator.innerHTML = '<span class="text-gray-600">意見分裂中...</span>';
        }
    }

    // 合意コンテナを表示・更新
    const consensusContainer = document.getElementById('consensus-container');
    const consensusStatus = document.getElementById('consensus-status');
    const consensusDetails = document.getElementById('consensus-details');

    if (consensusContainer && consensusStatus && consensusDetails) {
        consensusContainer.classList.remove('hidden');

        const statusText = {
            'unanimous': '全員一致',
            'majority': '多数派形成',
            'split': '意見分裂'
        }[data.consensusStatus] || data.consensusStatus;

        consensusStatus.textContent = statusText;
        consensusDetails.textContent = `Turn ${data.turn}で合意チェック実行 - 合意度: ${(data.agreementLevel * 100).toFixed(0)}%`;
    }
});
```

---

## 8. 実装手順

### Phase 1: バックエンド修正（優先度: 高）

- [ ] **1.1** `.env` の `JURY_MAX_DISCUSSION_ROUNDS` を `JURY_MAX_DISCUSSION_TURNS=9` に変更
- [ ] **1.2** `jury_judge.py` (Line 337-347) の環境変数読み取り修正
  - 古い変数のチェック追加（エラー）
  - 新しい変数の読み取り
  - `CollaborativeJuryJudge` 初期化パラメータ変更
- [ ] **1.3** `jury_judge_collaborative.py` のデータモデル変更
  - `DiscussionRound` クラス削除
  - `DiscussionResult` クラス追加
  - `CollaborativeJuryResult` の `discussion_rounds` → `discussion_result` 変更
- [ ] **1.4** `jury_judge_collaborative.py` のコンストラクタ変更
  - `max_discussion_rounds` → `max_discussion_turns`
  - `self.num_jurors = 3` 追加
- [ ] **1.5** `_phase2_discussion` メソッドをターンベースに完全書き換え
- [ ] **1.6** `_check_consensus_after_turns` メソッド追加
- [ ] **1.7** `_get_phase1_score` ヘルパーメソッド追加

### Phase 2: WebSocketイベント修正（優先度: 高）

- [ ] **2.1** `round_started` イベントの発火箇所を削除
- [ ] **2.2** `round_completed` イベントの発火箇所を削除
- [ ] **2.3** `juror_statement` イベントから `round` フィールド削除、`turn` フィールド追加
- [ ] **2.4** `consensus_check` イベントの発火追加

### Phase 3: フロントエンド修正（優先度: 中）

- [ ] **3.1** `submission_content.html` (Line 703) のHTML変更
  - `current-round-info` → `discussion-status` に変更
  - ターンカウンター表示追加
- [ ] **3.2** JavaScript `round_started` ハンドラ削除 (Line 1271-1284)
- [ ] **3.3** JavaScript `juror_statement` ハンドラ修正
  - `data.round` → `data.turn` に変更
  - ターンカウンター更新処理追加
  - メッセージ表示の改善
- [ ] **3.4** JavaScript `consensus_check` ハンドラ追加
- [ ] **3.5** ヘルパー関数追加（`getJurorName`, `getJurorEmoji`, `getPositionClass`, `escapeHtml`）

### Phase 4: テストと検証（優先度: 高）

- [ ] **4.1** Dockerを再起動して動作確認
- [ ] **4.2** 3人の陪審員が順番に発言することを確認
- [ ] **4.3** Turn 1, 2, 3と順番にカウントされることを確認
- [ ] **4.4** 3ターンごとに合意チェックが実行されることを確認
- [ ] **4.5** 最大9ターンで停止することを確認
- [ ] **4.6** 合意達成時に早期終了することを確認
- [ ] **4.7** UIでターン番号が正しく表示されることを確認
- [ ] **4.8** 古い環境変数使用時にエラーが発生することを確認

---

## 9. リスクと対策

### リスク

1. **既存データの互換性喪失**
   - 既存の評価データが `discussion_rounds` 形式で保存されている
   - 新しいコードで読み込めなくなる可能性

2. **WebSocketクライアントの破壊**
   - 既存のクライアントが `round_started` を待っている可能性
   - イベントが来ないため UI が固まる可能性

3. **デバッグの難しさ**
   - ターンベースは状態管理が複雑
   - バグがあると特定が難しい

### 対策

1. **既存データ対策**
   - ✓ **後方互換性は提供しない**（要件）
   - ✓ 古いデータ読み込み時はエラーとする
   - ✓ データベースを初期化してクリーンスタート

2. **WebSocket対策**
   - ✓ フロントエンドとバックエンドを同時にデプロイ
   - ✓ ブラウザキャッシュをクリア
   - ✓ 段階的ロールアウト（まず開発環境）

3. **デバッグ対策**
   - ✓ 各ターンでログ出力を強化
   - ✓ WebSocketイベントをコンソールログで確認
   - ✓ 小規模テスト（max_turns=3）で動作確認

---

## 10. 期待される効果

### ユーザー体験の向上

1. **より自然な会話フロー**
   - ✓ ラウンドの区切りがなくなり、連続的な議論が実現
   - ✓ ユーザーは「Round 1」「Round 2」という人工的な区切りを意識しない

2. **リアルタイム性の向上**
   - ✓ Turn 1, 2, 3... と進行状況が明確
   - ✓ 合意チェックのタイミングが理解しやすい（3ターンごと）

3. **早期終了による効率化**
   - ✓ 合意に達したらすぐに終了（最大9ターンを待たない）
   - ✓ 評価時間の短縮

### システムの改善

1. **コードの明確化**
   - ✓ ラウンドとターンの概念が統一される
   - ✓ データモデルがシンプルになる

2. **拡張性の向上**
   - ✓ 陪審員数を変更しやすい（`self.num_jurors`）
   - ✓ 合意チェックのタイミングを調整しやすい

---

## 11. ロールバック計画

万が一問題が発生した場合のロールバック手順：

1. `.env` を元に戻す
   ```bash
   JURY_MAX_DISCUSSION_ROUNDS=3
   ```

2. Git で変更をrevert
   ```bash
   git revert <commit-hash>
   ```

3. Dockerを再ビルド
   ```bash
   ./deploy/stop-local.sh
   ./deploy/run-local.sh
   ```

4. データベースを復元（必要に応じて）

---

## 12. 参考資料

- **環境変数ドキュメント**: `.env` ファイルのコメント参照
- **WebSocketイベント仕様**: `app/routers/submissions.py` の `emit_jury_event` 関数
- **データモデル定義**: `jury_judge_collaborative.py` の dataclass 定義

---

## 変更履歴

| 日付 | バージョン | 変更内容 | 作成者 |
|------|-----------|---------|--------|
| 2025-11-30 | 1.0 | 初版作成 | Claude Code |

---

**このドキュメントは実装の指針として使用してください。**
**実装前に必ずレビューを行い、テスト環境で十分に検証してください。**
