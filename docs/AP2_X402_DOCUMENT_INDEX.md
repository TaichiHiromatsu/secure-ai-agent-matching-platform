# AP2 / x402 統合資料 — 読み方と現在地

このページは、AP2／A2A x402 決済統合に関する資料の入口である。結論だけ知りたい場合は「現在の実装範囲」を読み、操作する場合は実演ガイドまたは運用手順へ進む。調査、要件、設計、実装計画、初回 FAIL、修正後 PASS の履歴は削除せず、判断の経緯として残している。

## 現在の実装範囲

| 項目 | 現在の状態 |
| --- | --- |
| AP2 | `AP2 v0.2 Human Present demo`。署名済み closed Checkout／Payment Mandate と Receipt の証跡を検証 |
| x402 | project-local `x402-wire-simulation/1`。v0.1 の wire shape を検証する fixture だけで **NOT CONFORMANT** |
| 公式 x402 | canonical URI、wallet、facilitator、on-chain settlement、ACC-030 は **DISABLED / NOT RUN** |
| 決済 rail | `exact-simulated` / `demo:local` / `USD`。実資産も実 transaction hash もない |
| 利用者画面 | ADK Web で選ぶ app は `payment_user_agent` 一つ |
| 内部実装 | `secure_mediation_agent` package の durable workflow が状態・認可・署名・Merchant 呼出しの正本 |
| 認証 | Firebase Authentication。ローカルに限り固定 demo identity の `DEV_MODE=true` を許可 |
| 耐久性 | 明示的 POSIX mount を使う単一 host／単一 container だけを受入済み |
| Cloud Run | revision `payment-user-agent-demo-00002-nt7`へexact `sha256:a22c3e69...a38bc`を反映し100% traffic。Firebase認証後のremote browser full flowとreload後の状態復元までPASS。状態と鍵が失われ得る非永続構成で、耐久 paid releaseではない |

画面名と内部名は役割が異なる。`payment_user_agent` は利用者向けの薄い UI adapter である。文書中の `secure_mediator` は仲介の論理的役割を、`secure_mediation_agent` はその内部 package／workflow を指す。利用者が複数の root agent を選び分ける構成ではない。

## 目的別の読む順序

| 目的 | 読む文書 | 内容 |
| --- | --- | --- |
| 5分で現在地を把握 | この索引 → [実装証跡](AP2_X402_IMPLEMENTATION_EVIDENCE.md) | 適用範囲、最新イメージ、検証済み事項、非対象 |
| ブラウザで実演 | [実演ガイド](AP2_X402_DEMO_GUIDE.md) | Firebase login、`payment_user_agent`、計画承認→決済承認→完了 |
| 起動・障害回復 | [運用手順](AP2_X402_RUNBOOK.md) | 永続 mount、readiness、CLI、再起動、migration、reset |
| 最新のテスト判定 | [独立テスト報告](AP2_X402_TEST_REPORT.md) の末尾 | 初回 FAIL、中間判定、修正後再試験を時系列で保持。常に末尾の最新判定を優先 |
| 機械可読の受入状態 | [conformance report](ap2_x402_conformance_report.json) | ACC-001〜035。ACC-030 は `NOT_RUN_CONDITIONAL` |
| 要件を確認 | [統合要件](AP2_X402_INTEGRATED_REQUIREMENTS.md) → [要件レビュー](AP2_X402_REQUIREMENTS_REVIEW.md) | 規範 ID、状態、不変条件、受入条件 |
| 設計意図を確認 | [統合設計](AP2_X402_INTEGRATED_DESIGN.md) → [設計レビュー](AP2_X402_DESIGN_REVIEW.md) | trust boundary、二承認、A2A Task、証跡、永続化 |
| 実装順と gate を確認 | [実装計画](AP2_X402_IMPLEMENTATION_PLAN.md) → [実装計画レビュー](AP2_X402_IMPLEMENTATION_PLAN_REVIEW.md) | work package、移行、test mapping、release gate |
| なぜこの方針になったか | [引継ぎ要件シード](AP2_X402_PLAN_APPROVAL_HANDOFF.md) → [現状調査](AP2_X402_CURRENT_STATE_RESEARCH.md) | 旧実装との差分と調査時点の仮説 |

### 旧 marketplace 案の履歴資料

次の3文書は `platform-credit`、deferred merchant payout、manual payout を採用していた旧案の記録であり、現行実装の要件・設計・適合範囲には使わない。

| 履歴文書 | 現行の参照先 |
| --- | --- |
| [旧調査](AP2_X402_RESEARCH.md) | [現状調査](AP2_X402_CURRENT_STATE_RESEARCH.md) と [実装証跡](AP2_X402_IMPLEMENTATION_EVIDENCE.md) |
| [旧要件](AP2_X402_REQUIREMENTS.md) | [統合要件](AP2_X402_INTEGRATED_REQUIREMENTS.md) |
| [旧設計](AP2_X402_DESIGN.md) | [統合設計](AP2_X402_INTEGRATED_DESIGN.md) |

## 文書の時系列と優先関係

```text
引継ぎ要件シード
  → 現状調査
  → 統合要件 → 要件レビュー反映
  → 統合設計 → 設計レビュー反映
  → 実装計画 → 実装計画レビュー反映
  → 実装証跡
  → 独立テスト報告（初回 FAIL → 修正 → 独立再試験 → 最終 PASS を追記）
```

同じ事実に差がある場合、古い調査時点の記録を現在形へ読み替えない。実装後の状態は、同一イメージ digest に拘束した `AP2_X402_TEST_REPORT.md` 末尾の最新再試験、`AP2_X402_IMPLEMENTATION_EVIDENCE.md`、`artifacts/ap2-x402-release-validation.json` を優先する。初回の FAIL や修正前の欠落は、消さずに「置換済み」として残す。

## 主要ブラウザフロー

```text
Firebase login
  → ADK Webで payment_user_agent を選択
  → 依頼
  → 「計画の承認」画面
  → 完全一致の「承認」
  → 「決済の承認」画面（7価格項目・期限・simulation表示）
  → 別の完全一致「承認」
  → completed・Artifact・Receipt ID/digest
  → refresh後も同じ完了状態を復元
```

最初の `承認` では決済しない。2回目の `承認` で AP2 closed Mandates を生成し、ローカル simulated settlement を開始する。`yes`、`はい`、`承認します`、空白付き `承認` は承認ではない。

## 今後の課題

ローカルexact imageの主要ブラウザフローと現行リリースの必須安全性検証に加え、デプロイ先でもFirebaseログイン、`payment_user_agent`の単独選択、二承認、完了、reload後の認証・選択・完了状態の維持を確認した。外部 partner の追加、長時間運転、負荷、追加 alert、より広い異常系 fixtureは今後のedge caseとして別 issueで扱う。

次は適合・本番安全性に関わるため、non-critical edge case とは分ける。

- 公式 x402 の network／asset／wallet／facilitator／TLS／on-chain settlement と ACC-030。
- durable Cloud Run または複数 instance 用の transactional shared DB／queue。
- production identity enrollment、KMS／HSM、KYC／AML、PCI／SCA、規制・会計・監査。
- loopback Merchant 境界を越える外部相互運用性の正式な適合範囲。

これらが完了するまで、公式 x402 compatible／conformant、実資産決済、耐久 Cloud Run paid、production-ready とは表示しない。
