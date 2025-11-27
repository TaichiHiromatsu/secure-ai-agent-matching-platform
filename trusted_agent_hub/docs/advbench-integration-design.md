# AdvBench + AISI Security 統合設計

**作成日**: 2025-01-25
**最終更新**: 2025-01-25
**ステータス**: ✅ 実装完了

## 📊 現状分析

### AISI Security v0.1の特性

**データ数**: 8行（非常に小規模）

**内容**: システムプロンプト流出、セーフガード回避攻撃

**カテゴリ**:
- **G6-5**: システムプロンプト流出攻撃
- **G6-6**: セーフガード回避攻撃（爆弾製造、生物兵器、人種差別など有害コンテンツ生成）

**言語**: 日本語

**ファイルパス**: `third_party/aisev/backend/dataset/output/06_aisi_security_v0.1.csv`

### 問題点

1. **データ数が極端に少ない**（8行のみ）
2. `agent_card_accuracy.py`にAdvBench読み込みコードが誤配置されている
3. 本来のAdvBench 520プロンプトが存在しない（コードのみ実装済み）

### 利用可能なAISIデータセット

```
third_party/aisev/backend/dataset/output/
├── 01_aisi_toxic_v0.1.csv (43K)
├── 02_aisi_misinformation_v0.1.csv (3.9K)
├── 03_aisi_fairness_v0.1.csv (34K)
├── 06_aisi_security_v0.1.csv (3.0K) ← 現在使用中
├── 07_aisi_explainability_v0.1.csv (1.5K)
└── 08_aisi_robustness_v0.1.csv (5.2K)
```

## 🎯 設計目標

1. **多様性確保**: 様々な攻撃タイプをカバー
2. **実行効率**: 最大50プロンプトに制限してコスト・時間を最適化
3. **トレーサビリティ**: どのデータセットから何が評価されたか追跡可能
4. **拡張性**: 将来的に他のAISIデータセット追加が容易

## 📐 アーキテクチャ設計

### 1. データセット構成

```
Security Gate評価プロンプト総数: 最大50件
├─ AISI Security v0.1 (system/safeguard attacks): 8件 (全件使用)
├─ AISI Toxic v0.1 (toxic content): 15件 (サンプリング)
├─ AISI Robustness v0.1 (adversarial): 15件 (サンプリング)
└─ AISI Fairness v0.1 (bias/discrimination): 12件 (サンプリング)
```

**優先度ルール**:
- **Priority 1 (必須)**: AISI Security v0.1 - 全8件実行
- **Priority 2 (高)**: AISI Toxic v0.1 - 有害コンテンツ生成攻撃
- **Priority 3 (中)**: AISI Robustness v0.1 - 敵対的ロバストネス
- **Priority 4 (低)**: AISI Fairness v0.1 - バイアス・差別テスト

### 2. コード構造の修正

**Before (誤った設計)**:
```
agent_card_accuracy.py
└─ load_advbench_scenarios()  ← 間違った配置
```

**After (正しい設計)**:
```
security_gate.py
├─ load_security_prompts()  (既存: 単一CSV用)
├─ load_multi_dataset_prompts()  (新規: 複数データセット統合)
└─ AttackPrompt に dataset_source フィールド追加
```

### 3. 新しいデータ構造

```python
@dataclass
class AttackPrompt:
    prompt_id: str
    text: str
    requirement: str
    perspective: str
    gsn_perspective: str
    dataset_source: str  # 新規: "aisi_security_v0.1", "aisi_toxic_v0.1" など
    priority: int  # 新規: 1=必須, 2=高, 3=中, 4=低

@dataclass
class SecurityGateConfig:
    datasets: List[DatasetConfig]
    max_total_prompts: int = 50
    sampling_strategy: str = "priority_balanced"  # priority_balanced, random, category_balanced

@dataclass
class DatasetConfig:
    name: str
    csv_path: Path
    priority: int
    max_samples: Optional[int]  # None = 全件使用
```

### 4. サンプリング戦略

```python
def sample_prompts(
    all_prompts: List[AttackPrompt],
    config: SecurityGateConfig
) -> List[AttackPrompt]:
    """
    優先度ベースでプロンプトをサンプリング

    1. Priority 1 (必須) を全件選択
    2. 残り枠を Priority 2, 3, 4 から比率に基づいて選択
    3. 各データセット内ではランダムサンプリング（再現性のためseed固定可能）
    """
    selected = []
    remaining_quota = config.max_total_prompts

    # Priority 1: 全件必須
    p1_prompts = [p for p in all_prompts if p.priority == 1]
    selected.extend(p1_prompts)
    remaining_quota -= len(p1_prompts)

    # Priority 2-4: 比率配分 (例: 60%, 30%, 10%)
    p2_prompts = [p for p in all_prompts if p.priority == 2]
    p3_prompts = [p for p in all_prompts if p.priority == 3]
    p4_prompts = [p for p in all_prompts if p.priority == 4]

    p2_quota = int(remaining_quota * 0.60)
    p3_quota = int(remaining_quota * 0.30)
    p4_quota = remaining_quota - p2_quota - p3_quota

    selected.extend(random.sample(p2_prompts, min(p2_quota, len(p2_prompts))))
    selected.extend(random.sample(p3_prompts, min(p3_quota, len(p3_prompts))))
    selected.extend(random.sample(p4_prompts, min(p4_quota, len(p4_prompts))))

    return selected
```

### 5. 結果トレーサビリティ

```python
@dataclass
class AttackResult:
    prompt_id: str
    prompt_text: str
    requirement: str
    response_text: Optional[str]
    verdict: str
    reason: str
    dataset_source: str  # 新規: トレーサビリティ用
    priority: int  # 新規
    metadata: Dict[str, Any]
```

**サマリーにデータセット別集計を追加**:
```json
{
    "total_attacks": 50,
    "blocked": 45,
    "passed": 3,
    "errors": 2,
    "by_dataset": {
        "aisi_security_v0.1": {"total": 8, "blocked": 7, "passed": 1},
        "aisi_toxic_v0.1": {"total": 15, "blocked": 14, "passed": 1},
        "aisi_robustness_v0.1": {"total": 15, "blocked": 13, "passed": 1, "errors": 1},
        "aisi_fairness_v0.1": {"total": 12, "blocked": 11, "passed": 0, "errors": 1}
    },
    "by_priority": {
        "1": {"total": 8, "blocked": 7, "passed": 1},
        "2": {"total": 15, "blocked": 14, "passed": 1},
        "3": {"total": 15, "blocked": 13, "passed": 2},
        "4": {"total": 12, "blocked": 11, "passed": 1}
    }
}
```

### 6. submissions.pyでの呼び出し

**Before (現在)**:
```python
security_summary = run_security_gate(
    dataset_path=dataset_path,  # 単一ファイルパス
    ...
)
```

**After (提案)**:
```python
security_config = SecurityGateConfig(
    datasets=[
        DatasetConfig(
            name="aisi_security_v0.1",
            csv_path=Path("third_party/aisev/backend/dataset/output/06_aisi_security_v0.1.csv"),
            priority=1,
            max_samples=None  # 全件
        ),
        DatasetConfig(
            name="aisi_toxic_v0.1",
            csv_path=Path("third_party/aisev/backend/dataset/output/01_aisi_toxic_v0.1.csv"),
            priority=2,
            max_samples=15
        ),
        DatasetConfig(
            name="aisi_robustness_v0.1",
            csv_path=Path("third_party/aisev/backend/dataset/output/08_aisi_robustness_v0.1.csv"),
            priority=3,
            max_samples=15
        ),
        DatasetConfig(
            name="aisi_fairness_v0.1",
            csv_path=Path("third_party/aisev/backend/dataset/output/03_aisi_fairness_v0.1.csv"),
            priority=4,
            max_samples=12
        )
    ],
    max_total_prompts=50,
    sampling_strategy="priority_balanced"
)

security_summary = run_security_gate(
    config=security_config,  # 新しい設定オブジェクト
    ...
)
```

## 📈 パフォーマンス影響分析

### 現在
- **データ数**: 8プロンプト
- **実行時間**: 約1-2分 (タイムアウト10秒/プロンプト)
- **コスト**: 極小

### 提案後
- **データ数**: 50プロンプト (6.25倍)
- **実行時間**: 約5-10分
- **コスト**: 中程度（API呼び出し50回 + Judge評価）

### 最適化オプション

1. **並列実行**: 5並列で実行時間を1/5に短縮
2. **キャッシング**: 同じagent_id + revisionの結果をキャッシュ
3. **段階的実行**: Priority 1のみ必須、失敗が多ければPriority 2-4をスキップ

## 🔄 実装フェーズ

### Phase 1: データ構造拡張 ✅ 完了
**ファイル**: `evaluation-runner/src/evaluation_runner/security_gate.py`

**実装内容**:
- `AttackPrompt`に`dataset_source: str`と`priority: int`フィールドを追加
- `AttackResult`に同様のフィールドを追加
- `SecurityGateConfig`と`DatasetConfig`のdataclassを作成
- 後方互換性維持（既存の`dataset_path`パラメータも引き続き動作）

### Phase 2: マルチデータセット読み込み ✅ 完了
**ファイル**: `evaluation-runner/src/evaluation_runner/security_gate.py`

**実装内容**:
- `load_security_prompts()`: 単一CSVファイルからの読み込み（既存関数を拡張）
- `load_multi_dataset_prompts()`: 複数データセットからの統合読み込み
- `_sample_by_priority()`: 優先度ベースサンプリングロジック
  - `priority_balanced`: Priority 1全件 + Priority 2-4を60%/30%/10%配分
  - `random`: ランダムサンプリング
  - デフォルト: Priority順ソート
- `run_security_gate()`: `config`パラメータでマルチデータセットモードをサポート

### Phase 3: 結果集計強化 ✅ 完了
**ファイル**: `evaluation-runner/src/evaluation_runner/security_gate.py`, `app/routers/submissions.py`

**実装内容**:
- データセット別集計: `byDataset` フィールドをサマリーに追加
- 優先度別集計: `byPriority` フィールドをサマリーに追加
- 結果JSONLファイルに`datasetSource`と`priority`を記録
- トレーサビリティ完全対応

### Phase 4: submissions.py統合 ✅ 完了
**ファイル**: `app/routers/submissions.py`

**実装内容**:
- `SecurityGateConfig`を作成し、4つのAISIデータセットを統合:
  - AISI Security v0.1 (priority=1, 全8件)
  - AISI Toxic v0.1 (priority=2, max 15件)
  - AISI Robustness v0.1 (priority=2, max 15件)
  - AISI Fairness v0.1 (priority=3, max 12件)
- `run_security_gate(config=security_gate_config)`でマルチデータセット評価を実行
- 最大50プロンプトでバランス良く評価

### Phase 5: 統合テストとドキュメント更新 🔄 進行中
**対象ファイル**:
- README.md更新（マルチデータセット対応を明記）
- テスト実行とデバッグ
- W&B Weaveでのトレース確認

## ✅ 期待される効果

- ✅ 8→50プロンプトで多様性大幅向上
- ✅ 優先度ベースで重要な攻撃を確実に評価
- ✅ データソース追跡でデバッグ・分析が容易
- ✅ 将来的なデータセット追加が簡単
- ✅ コスト・時間の予測可能性確保

## 📝 実装メモ

### AdvBenchについて
- 本来のAdvBench（Zou et al., 2023）は520の有害プロンプト
- 現在のコードベースには存在せず、AIISIデータセットのみ
- 実装は「AISI複数データセット統合」として進める
- 将来的にAdvBench 520プロンプトを追加する場合もこの設計で対応可能

### データセット優先度の根拠
- **Priority 1 (Security)**: 最も重大なセキュリティ脆弱性
- **Priority 2 (Toxic)**: 有害コンテンツ生成リスク
- **Priority 3 (Robustness)**: 敵対的入力への耐性
- **Priority 4 (Fairness)**: バイアス・差別検出

## 🔗 関連ファイル

- [security_gate.py](../evaluation-runner/src/evaluation_runner/security_gate.py)
- [agent_card_accuracy.py](../evaluation-runner/src/evaluation_runner/agent_card_accuracy.py)
- [submissions.py](../app/routers/submissions.py)
- [AISI datasets](../third_party/aisev/backend/dataset/output/)

## 📚 参考文献

- Zou et al. (2023). "Universal and Transferable Adversarial Attacks on Aligned Language Models"
- AISI Security Benchmark v0.1 Documentation
- [docs/papers/responsible-ai-agents-2502.18359.pdf](papers/responsible-ai-agents-2502.18359.pdf)
- [docs/papers/automated-risky-game-2506.00073.pdf](papers/automated-risky-game-2506.00073.pdf)
