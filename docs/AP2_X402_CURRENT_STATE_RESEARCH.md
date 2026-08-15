# AP2 / A2A x402 現状調査

> **文書状態:** 2026-08-15の実装前スナップショットであり、現在の実装状態を示す文書ではない。最新状態は `AP2_X402_DOCUMENT_INDEX.md` と実装・テスト証跡を参照する。

調査日: 2026-08-15 (Asia/Tokyo)
対象: Section 12 Step 1（現状調査のみ。実装・設計確定は対象外）

## 1. 結論

現在のリポジトリには、耐リプレイ、金額・受取人・Checkout・Task の厳密な結合、SQLite による状態永続化、決済情報を LLM に渡さない境界など、再利用価値の高い決済基盤がある。ただし、利用者から見た仲介主体はまだ一つではない。

実際には次の二つの経路が分離している。

1. `payment_demo_user_agent` → payment API (`:8004`) → `paid_booking_agent` (`:8005`)
2. `secure_mediator` → matcher / planner → boolean plan approval → orchestrator → legacy external agents (`:8002`)

`secure_mediator` から payment API への接続はなく、orchestrator の一般 A2A 呼び出しは自然言語 `Message`、payment API と paid agent は独自 `data.action` 契約である。このため、現時点では「match → plan → plan approval → paid-agent invocation → PaymentRequired → payment approval → settlement/fulfillment」を単一の仲介主体で完遂できない。

優先順位は次のように整理する。

| 優先度 | 現状ギャップ | 判定 |
| --- | --- | --- |
| P0 | payment の直接経路が `secure_mediator` の plan / approval / orchestration に接続されていない | 最優先の統合ブロッカー |
| P1 | AP2 v0.2 の Mandate Content は概ね似ているが、公式の署名済み Mandate、役割別検証、Checkout/Payment Receipt ではない | AP2 準拠ブロッカー |
| P2 | A2A x402 v0.1 の URI、activation、metadata、payload、receipt、on-chain settlement と異なる | x402 v0.1 準拠ブロッカー |

AP2 と A2A x402 は別の仕様として扱う必要がある。AP2 は agent-performed payment の認可・証跡モデルであり、Commerce Protocol の API/transport を規定しない。A2A x402 v0.1 は A2A Message metadata と Task state 上の on-chain payment extension である。現行の一つの project-local URN を両方の公式準拠 URI とみなすことはできない。

## 2. 調査スナップショットと固定参照

### 2.1 ローカルリポジトリ

| 項目 | 値 |
| --- | --- |
| repository HEAD | `9730a597a3359f7ecac0f2bf10513a80f9b3c56e` |
| branch | `codex/ap2-x402-integration` |
| PR | `#25` (draft)、head はローカル HEAD と一致 |
| 調査開始時の worktree | clean |

本書のローカル行番号は上記 commit に対するものとする。

### 2.2 公式 AP2

| 項目 | 固定値 |
| --- | --- |
| repository | [google-agentic-commerce/AP2](https://github.com/google-agentic-commerce/AP2) |
| inspected `main` | [`e1ea56db72a6385bce3e5c1112b3a56ce60acb43`](https://github.com/google-agentic-commerce/AP2/tree/e1ea56db72a6385bce3e5c1112b3a56ce60acb43) |
| specification | [Agentic Payment Protocol v0.2](https://github.com/google-agentic-commerce/AP2/blob/e1ea56db72a6385bce3e5c1112b3a56ce60acb43/docs/ap2/specification.md) |
| release tag | `v0.2.0` → `b4587ac1d055888a73b4b21750973cffba961793` |
| spec SHA-256 | `32c3be5011f481d2760e56e7b9935b0842c3da0d5f7d7b8a68402a599f1e6aa3` |

`v0.2.0` と inspected `main` の AP2 spec および closed Checkout/Payment Mandate JSON Schema は byte-identical だった。比較対象は `main` の固定 SHA とする。

主要参照:

- [Human Present / Human Not Present flows](https://github.com/google-agentic-commerce/AP2/blob/e1ea56db72a6385bce3e5c1112b3a56ce60acb43/docs/ap2/flows.md)
- [Agent Authorization](https://github.com/google-agentic-commerce/AP2/blob/e1ea56db72a6385bce3e5c1112b3a56ce60acb43/docs/ap2/agent_authorization.md)
- [Checkout Mandate](https://github.com/google-agentic-commerce/AP2/blob/e1ea56db72a6385bce3e5c1112b3a56ce60acb43/docs/ap2/checkout_mandate.md)
- [Payment Mandate](https://github.com/google-agentic-commerce/AP2/blob/e1ea56db72a6385bce3e5c1112b3a56ce60acb43/docs/ap2/payment_mandate.md)
- [canonical JSON Schemas](https://github.com/google-agentic-commerce/AP2/tree/e1ea56db72a6385bce3e5c1112b3a56ce60acb43/code/sdk/schemas/ap2)
- [Python SDK runtime](https://github.com/google-agentic-commerce/AP2/tree/e1ea56db72a6385bce3e5c1112b3a56ce60acb43/code/sdk/python/ap2)

### 2.3 公式 A2A x402 Payments Extension

| 項目 | 固定値 |
| --- | --- |
| repository | [google-agentic-commerce/a2a-x402](https://github.com/google-agentic-commerce/a2a-x402) |
| inspected `main` | [`125db5526a965d2325459d1a9df2e274a7e42396`](https://github.com/google-agentic-commerce/a2a-x402/tree/125db5526a965d2325459d1a9df2e274a7e42396) |
| normative target | [`spec/v0.1/spec.md`](https://github.com/google-agentic-commerce/a2a-x402/blob/125db5526a965d2325459d1a9df2e274a7e42396/spec/v0.1/spec.md) |
| tag | `v0.1.0` → `a5590226fc4190fe7a9d2c13ab70128714db3125` |
| spec content SHA-256 | `5cdc35ed8c4d7a93bb120f1782fd06e2cc3ef19036684f772e27d0d644c66940` |

`v0.1.0` の `v0.1/spec.md` と inspected `main` の `spec/v0.1/spec.md` は同一内容で、変更はパス移動だけだった。

## 3. 実際の接続構造

```text
Payment demo lane
  ADK Web user
    -> payment_demo_user_agent
    -> POST :8004/a2a  (custom data.action=start_order|submit_payment)
    -> MarketplaceService + SQLite + simulated rail
    -> POST :8005/a2a  (custom data.action=quote|fulfill|fulfillment_status)
    -> paid_booking_agent

Secure mediation lane
  ADK Web user
    -> secure_mediator
       -> matcher -> Trusted Agent Store :8001
       -> planner -> markdown artifact + session boolean
       -> approve_plan -> state['plan_approved']=True
       -> orchestrator -> RemoteA2aAgent text Message -> legacy agents :8002
```

両 lane の間に呼び出し edge はない。Docker/supervisord でも `secure_mediator` (`:8000`) と payment API (`:8004`)、paid agent (`:8005`) は別プロセスであり、`user-agent` は別の ADK root agent として配置される。

payment API は upstream では利用者に PaymentRequired を出す merchant-like role、downstream では paid agent から quote/fulfillment を買う client/marketplace role、内部では rail/ledger/guarantee issuer を兼ねている。paid agent は Store と Agent Card 上では Merchant だが、利用者の `PaymentMandate.payee` は mediation platform である。従って、誰が AP2 Merchant / MPP / CP で、誰が x402 Merchant / Client かは現状の名称だけでは一意に決まらない。

### 3.1 コンポーネント別の接続点

| コンポーネント | 実装上の connection point | 現状の意味 |
| --- | --- | --- |
| `secure_mediator` | `secure_mediation_agent/agent.py:29-56` | plan approval は session boolean。payment tool/client はない |
| secure workflow instruction | `secure_mediation_agent/agent.py:98-140` | plan 承認後に一般 orchestrator を起動。`OK` / `はい` も承認として許す |
| planner | `secure_mediation_agent/subagents/planning_agent.py:28-105` | free-form markdown を保存し `plan_approved=False` にする |
| structured plan helper | `secure_mediation_agent/subagents/planning_agent.py:108-131` | `plan_id`, request, steps, status のみ。金額・merchant・snapshot digest の認可境界はない |
| matcher | `secure_mediation_agent/subagents/matching_agent.py:93-118` | Store の `capabilities.extensions` は保持するが、公式 extension URI を eligibility 条件にしない |
| orchestrator | `secure_mediation_agent/subagents/orchestration_agent.py:96-204` | Agent Card を解決し、task/input を一つの自然言語 Message として送る |
| orchestration gate | `secure_mediation_agent/subagents/orchestration_agent.py:650-667` | boolean が true かだけを検査。どの plan/snapshot を承認したかは検査しない |
| payment user agent | `user-agent/agent.py:32-81` | 別 root agent。最初の発話ですぐ quote/payment request、正確な `承認` で submit |
| payment A2A client | `user-agent/payment_client.py:67-195` | `:8004/a2a` へ project URN と custom `data.action` を送る |
| payment API | `secure_mediation_agent/payment_marketplace/api.py:427-505` | `start_order` / `submit_payment` / `payout_status` を独自 dispatch |
| payment service | `secure_mediation_agent/payment_marketplace/service.py:158-365` | order/task を作り、paid agent へ quote を要求し、`input-required` を永続化 |
| paid agent discovery | `trusted_agent_store/data/agents/registered-agents.json:99-156` | `:8005` と project-local extension、platform-credit 能力を登録 |
| mediator → paid agent | `secure_mediation_agent/payment_marketplace/merchant_client.py:105-139` | extension header/taskId なしの custom `quote` / `fulfill` action |
| paid agent adapter | `external-agents/paid-booking-agent/app.py:263-305` | custom data action を plain result dict で返し、A2A Task は返さない |

### 3.2 当時の決済経路の時系列

1. `payment_demo_user_agent` は一般発話を固定商品 `demo-paid-booking` に写像する。
2. payment API は Task/Context/Order を作成し、即座に paid agent へ quote を要求する。
3. paid agent は merchant-signed compact JWT と独自 quote requirement を返す。
4. payment API は quote を検証し、charge を `required` で作り、Task を `input-required` にする。
5. user agent は価格を表示し、正確な `承認` を待つ。
6. deterministic Trusted Surface fixture が closed Checkout/Payment Mandate Content と project authorization を構成する。
7. payment API は checkout/order/task/quote/challenge/nonce/audience/payee/amount/instrument/time を deterministic code で照合する。
8. simulated rail を settle し、platform-credit guarantee を発行して paid agent を fulfill する。
9. custom receipts をまとめ、元の Task を `completed` にする。

この経路には planner、plan artifact、plan approval、matcher が存在しない。payment API の `/v1/orders` と `/a2a` は、それらの承認を証明しなくても呼び出せる。

## 4. P0: 単一の仲介主体へ統合する際の差分

これは外部仕様の準拠以前に解決が必要な repository-level gap である。

| 必要な境界 | 現状 | gap / 影響 |
| --- | --- | --- |
| 一つの利用者向け root actor | `secure_mediator` と `payment_demo_user_agent` が別 root | 同じ依頼が plan と payment の別 session/state に分断される |
| matcher が paid capability を選ぶ | metadata は保存するだけ | project URN/公式 x402 URI の要件を満たす agent かを決定しない |
| plan approval が実行対象を固定 | boolean だけ | 承認した plan、agent、price ceiling、merchant、request snapshot を再検証できない |
| plan 承認後に paid agent を呼ぶ | payment lane は最初の発話で quote | plan gate を迂回できる |
| PaymentRequired を同じ Task/plan に結ぶ | payment Task は payment lane 内だけ | secure workflow の planId と payment taskId/orderId の結合がない |
| payment approval を plan approval と分離 | payment lane では分離されているが secure lane には payment approval がない | 統合後も二つの明示承認を混同しない境界が必要 |
| deterministic payment execution | payment service 内には存在 | secure orchestrator の LLM text path から安全に到達する専用 boundary がない |
| paid agent の A2A 呼び出し | custom data action | 一般 `RemoteA2aAgent` text invocation と wire contract が不一致 |

重要な現状評価:

- `secure_mediator` の plan approval は AP2 Mandate ではない。
- payment lane の payment approval は plan approval ではない。
- project-local authorization は `orderId`, `taskId`, `quoteId`, `challengeId`, nonce, amount 等を強く結合するが、`planId` または承認済み plan digest を結合しない。
- AP2 は planner を規定しないため、plan binding は AP2 の field を勝手に拡張するのではなく、platform authorization boundary として AP2 Mandate の外側で明確に扱う必要がある。
- 一つの entity が AP2 の複数 role を担うこと自体は公式仕様で許される。ただし統合した `secure_mediator` / payment boundary は、担う role ごとの deterministic verification responsibility を満たす必要がある。

## 5. 公式 AP2 v0.2 の基準

### 5.1 仕様範囲

AP2 v0.2 は agent-performed payment の security/authorization protocol であり、catalog、checkout update、role 間 API など Commerce Protocol の具体的 transport は scope 外である。従って AP2 そのものに A2A extension URI や `X-A2A-Extensions` activation を求めてはいけない。

AP2 の五つの logical role は Shopping Agent、Credential Provider、Merchant、Merchant Payment Processor、Trusted Surface である。Shopping Agent は agentic と想定され、Trusted Surface は non-agentic でなければならない。role の validation/processing は、role 自体が agentic でも deterministic code で実行しなければならない。

### 5.2 Human Present closed-Mandate フロー

現行デモが目指しているのは Human Present (`direct`) である。この flow では:

1. Merchant が signed Checkout を Shopping Agent に返す。
2. Shopping Agent が closed Checkout/Payment Mandate Content を組み立てる。
3. Trusted Surface が両方を表示し、認証と同意を得て署名済み Mandate を作る。
4. Credential Provider が Payment Mandate を検証し、transaction-scoped payment credential/token を作る。
5. Merchant は Checkout Mandate と token を受け取り、Checkout Mandate を自らの最新 checkout と照合する。
6. MPP は token 内の Payment Mandate と checkout binding を検証する。
7. MPP-signed Payment Receipt と Merchant-signed Checkout Receipt が返る。

### 5.3 closed Mandate の正規内容

`CheckoutMandate` の required content:

| field | official v0.2 |
| --- | --- |
| `vct` | exact `mandate.checkout.1` |
| `checkout_jwt` | merchant-signed serialized JWT |
| `checkout_hash` | `checkout_jwt` 文字列の base64url hash。`_sd_alg`、なければ SHA-256 |

`PaymentMandate` の required content:

| field | official v0.2 |
| --- | --- |
| `vct` | exact `mandate.payment.1` |
| `transaction_id` | associated `checkout_jwt` hash |
| `payee` | Merchant (`id`, `name`, optional `website`) |
| `payment_amount` | integer minor units と ISO-4217 currency |
| `payment_instrument` | `id`, `type`, optional `description` |

`iat`, `exp`, `pisp`, `execution_date`, `risk_data` は closed Payment Mandate の optional fields である。

Mandate Content の field が一致するだけでは AP2 Mandate にはならない。公式の authorization model では、Human Present の closed Mandates は User Credential または trusted Agent Provider に根ざした署名済み VDC として提示・検証される。現行 spec/SDK の concrete mechanism は SD-JWT / delegated SD-JWT chain で、receipt は verifier-signed JWT である。

## 6. AP2適合性の差分表

判定:

- **Aligned**: 仕様上の要求を現実装が満たす。
- **Partial**: 一部の形・順序・security property はあるが、公式 wire/crypto/role boundary ではない。
- **Missing**: 対象 flow に必要だが存在しない。
- **Out of target**: Human Present の今回対象には不要。準拠を主張する場合は scope 表示が必要。

| AP2 v0.2 観点 | 公式基準 | ローカル実装 | 判定 | gap の具体点 |
| --- | --- | --- | --- | --- |
| protocol identity | AP2 v0.2 と正確な schema version を扱う | docs/comments は v0.2、Mandate `vct` は `.1` | Partial | official SDK/schema dependency はなく、独自 profile として再実装 |
| Human Present order | signed checkout → trusted display/consent → signed closed Mandates | quote の checkout JWT 後に deterministic fixture が両 Mandate content を作る | Partial | plan lane と分離。認証は chat の正確な語だけで、正式な user credential enrollment はない |
| Trusted Surface isolation | non-agentic、key を Agent/LLM に触らせない | `PaymentDemoUserAgent` と `TrustedSurface` は deterministic。payment evidence は legacy LLM path に渡さない | Partial | test HMAC key は同一 application package 内。production trust/key boundary ではない |
| role / payee mapping | Merchant は Checkout を提供・検証し、`PaymentMandate.payee` は支払を受ける Merchant | paid agent を Merchant と登録する一方、payee は mediation platform。payment API が merchant-like/MPP-like/marketplace role を兼務 | Missing decision/binding | mediator が merchant of record なのか、paid agent が AP2 Merchant なのかを verification/receipt issuer と共に固定していない |
| Checkout Mandate content | `mandate.checkout.1`, `checkout_jwt`, `checkout_hash` | `secure_mediation_agent/payment_marketplace/models.py:288-301` は field-level で一致 | Aligned (content only) | 署名済み Mandate token ではなく plain JSON model |
| Payment Mandate content | versioned vct、checkout hash、payee、minor-unit amount、instrument | `secure_mediation_agent/payment_marketplace/models.py:320-335`; `transaction_id == checkout_hash` を検査 | Aligned (content only) | official Mandate token ではない。payee は mediation platform 固定 |
| Merchant checkout JWT entropy/signature | merchant-signed JWT。v0.2 は deterministic signature を避けるよう MUST | paid agent は compact `HS256` JWT (`external-agents/paid-booking-agent/service.py:202-209`) | Missing | HMAC-SHA256 は deterministic。spec の rainbow-table mitigation と不一致 |
| signed closed Mandates | direct flow では User / trusted Agent Provider に根ざす署名を verifier が検証 | Mandate dict の digest を外側の `ProjectAuthorization` HS256 が署名 | Missing | SD-JWT/VDC/mandate chain ではなく、Mandate 自体の issuer/trust/presentation を検証しない |
| Checkout ↔ Payment binding | `transaction_id == hash(checkout_jwt)` | `secure_mediation_agent/payment_marketplace/trusted_surface.py:105-123,165-190` と service checks で厳密に検査 | Aligned (binding property) | hash source JWT の署名方式は上記の非準拠 |
| Merchant verification | Merchant が Checkout Mandate を受け、signature、latest checkout hash、constraints を検証 | mediator が approval を検証し、paid agent は mediator guarantee のみ検証 | Missing | paid agent は Checkout Mandate を受信せず、Merchant role の公式 verification を実施しない |
| CP / Network verification | Payment Mandate 検証後に scoped payment credential/token を返す | payment API が独自 approval を検証後、simulated rail を直接 settle | Missing | CP role/trust resolution/scoped credential がない |
| MPP verification | Checkout に scope された credential を検証して payment を処理 | payment API/rail が combined role のように動く | Partial | official credential/Payment Mandate token を受けて検証する boundary ではない |
| Checkout Receipt | Merchant-signed JWT、closed Checkout Mandate hash を `reference` に持つ | custom `merchant-order` receipt envelope | Missing | official CheckoutReceipt schema/signature/reference semantics ではない |
| Payment Receipt | MPP-signed JWT、closed Payment Mandate hash reference、payment/PSP/network IDs | custom `ap2-payment` HMAC envelope | Missing | official PaymentReceipt schema と verifier-signed JWT ではない |
| rejection receipt | accept/reject のどちらでも signed Mandate Receipt | errors は MarketplaceError / JSON-RPC error が中心 | Missing | `invalid_credential`, `invalid_mandate`, `unresolved_constraint` 等の receipt semantics がない |
| dispute evidence | Mandates と両 Receipt の signature/reference chain を再検証可能 | SQLite evidence、digests、nonce、quote/rail/guarantee records は durable | Partial | durable base は強いが、公式 four-object evidence chain ではない |
| HNP open Mandates | open Checkout/Payment Mandates、constraints、`cnf`、selective disclosure | なし | Out of target | Human Present 限定なら blocker ではない。autonomous AP2 conformance は主張不可 |
| AP2 transport | Commerce Protocol transport は AP2 scope 外 | project URN を AP2/x402 combined extension と呼ぶ | Partial | project URI は使えるが、AP2 の公式 A2A activation URI ではない |

### 6.1 AP2 でそのまま再利用できる強み

- `checkout_jwt` の exact-byte hash と `PaymentMandate.transaction_id` の一致。
- amount を integer minor units として扱い、displayed total と payment amount を一致させる検査。
- order/task/quote/challenge/audience/payee/instrument/time の fail-closed 検査。
- nonce の durable consume、idempotency、状態遷移、evidence persistence。
- Trusted Surface 用 deterministic code と、payment proof を legacy LLM/orchestrator に渡さない方針。
- simulated rail であることを Agent Card/API/UI に明示する姿勢。

これらは AP2 の signed Mandate / role verification / signed Receipt を置き換えるものではないが、その下の policy enforcement と evidence store として再利用できる。

## 7. 公式 A2A x402 v0.1 の基準

### 7.1 extension の識別子と activation

canonical URI は厳密に:

```text
https://github.com/google-a2a/a2a-x402/v0.1
```

Agent Card declaration と activation の両方でこの URI を使う。client は `X-A2A-Extensions` で要求し、server は response header に同じ URI を echo しなければならない。

### 7.2 metadata と Task の相関

PaymentRequired:

```json
{
  "x402.payment.status": "payment-required",
  "x402.payment.required": {
    "x402Version": 1,
    "accepts": [
      {
        "scheme": "exact",
        "network": "base",
        "asset": "<token contract>",
        "payTo": "<wallet address>",
        "maxAmountRequired": "<minor unit string>"
      }
    ]
  }
}
```

Task state は `input-required`。metadata は `Task.status.message.metadata` にある二つの dotted sibling keys であり、`"x402.payment": { ... }` という nested object ではない。

Payment submission は元 Task の `taskId` を持つ新しい A2A Message で、Message metadata に以下を置く。

```json
{
  "x402.payment.status": "payment-submitted",
  "x402.payment.payload": {
    "x402Version": 1,
    "network": "<network>",
    "scheme": "<scheme>",
    "payload": {}
  }
}
```

Merchant は `taskId` で自らが以前提示した original `PaymentRequirements` を取得し、submission と照合する。

### 7.3 settlement と receipt

Merchant は facilitator 等を通じて signature を verify して on-chain settle する。各 settlement attempt の `x402SettleResponse` は `x402.payment.receipts` に append-only で蓄積し、最終 `TaskStatus.message` は Task lifetime の全 receipt を含まなければならない。

receipt の required fields は `success` と `network`。成功時は `transaction`、任意で `payer`、失敗時は `errorReason` を持つ。payment failure は `x402.payment.status=payment-failed` と reason/error metadata を TaskStatus message に返す。

## 8. A2A x402 v0.1適合性の差分表

| x402 v0.1 観点 | 公式基準 | ローカル実装 | 判定 | gap の具体点 |
| --- | --- | --- | --- | --- |
| canonical URI | exact `https://github.com/google-a2a/a2a-x402/v0.1` | `urn:secure-a2a:extensions:ap2-x402-marketplace:v1` | Missing | declaration/activation とも公式 URI ではない |
| Agent Card | canonical URI を `capabilities.extensions` に宣言 | payment API と paid agent は project URN を宣言 | Missing | `required:true` だけは形が近い |
| activation request | canonical URI を `X-A2A-Extensions` で送る | user client は project URN を送る | Missing | official activation にならない |
| activation echo | server response header に URI を echo | payment API は header を検査するが echo しない。paid agent は header 自体を検査しない | Missing | end-to-end activation confirmation がない |
| PaymentRequired location | `Task.status.message.metadata` の dotted sibling keys | `metadata["x402.payment"]={status, requirement,...}` | Missing | `required` ではなく nested `requirement` |
| wire version | `x402Version: 1` | `x402Version: 2` | Missing | v0.1 extension data structure と互換でない |
| requirements fields | `scheme`, `network`, `asset`, `payTo`, `maxAmountRequired` | `amount`, `decimals`, top-level resource object、`exact-simulated`, `demo:local`, `USD` | Missing | field 名と rail identity の両方が異なる |
| A2A Task state | payment request は `input-required` | payment API が `input-required` を返す | Aligned | metadata は非準拠 |
| submission transport | correlated Message metadata の dotted keys | custom data part `action=submit_payment`; `paymentPayload.accepted` | Missing | payment payload は `network`/`scheme` top-level も欠く |
| task correlation | new Message の `taskId` で original requirements を検索・照合 | user client は `taskId` を送る。service 内 DB binding は強い | Partial | A2A adapter は incoming Message `taskId` と order を明示照合せず、mediator→paid agent Message には taskId がない |
| merchant state | Task に original requirements を保持 | payment API は SQLite で durable に保持 | Partial/strong | official metadata contract ではないが durability は reference sample より強い |
| verify/settle | wallet-signed payload を facilitator で verify、on-chain settle | customer HMAC authorization + in-process balance rail | Missing by simulation | simulated rail では意図的に満たせない |
| work-before-settle | Merchant SHOULD work completion before settlement | rail settle → guarantee → merchant fulfill | Diverges | official recommendation と逆順。marketplace guarantee model 固有 |
| receipt schema | append-only `x402SettleResponse[]` | custom signed `ReceiptEnvelope[]` | Missing | `success` / `network` / on-chain `transaction` schema ではない |
| receipt history | 全 settlement attempt を append、最終 TaskStatus message に全件 | final Task metadata に複数 custom receipt はある | Partial | raw rail attempt receipt は最終配列に含まれず、retry/failed attempt の official append-only history でもない |
| payment failure | `payment-failed`, error, failed settle receipt を Task message に返す | service error は主に JSON-RPC error | Missing | official payment state/history を持つ Task response ではない |
| service result | final/updated Task の Artifact と payment receipt | paid agent result を mediator の custom completion/receipt にまとめる | Partial | x402 Merchant Task の Artifact contract ではない |
| private-key isolation | LLM は private key を扱わない | payment proof は deterministic lane。test HMAC secrets は LLM path に送らない | Aligned in demo boundary | official wallet/on-chain signing service はない |
| replay/input validation | signatures, nonce, payment structures を厳格検証 | durable nonce、strict Pydantic、exact binding | Aligned in property | official PaymentPayload signature の検証ではない |
| TLS | A2A communication は HTTPS/TLS | local URLs は HTTP、loopback/internal deployment | Missing for conformance | local demo 例外を仕様準拠とは主張できない |

### 8.1 公式参照実装の扱い

[Python reference package](https://github.com/google-agentic-commerce/a2a-x402/tree/125db5526a965d2325459d1a9df2e274a7e42396/python/x402_a2a) は wire `x402Version=1`、dotted metadata、Task correlation、wallet signing、facilitator verify/settle の有用な実装例である。current lock は `x402==0.2.0`, `a2a-sdk==0.3.1` だが、Python package version と A2A extension wire version は別物である。

ただし、reference executor 自体は activation helper を必ず enforcement しておらず、requirements store も in-memory である。したがって conformance oracle は reference code ではなく normative `spec/v0.1/spec.md` とする。ローカルの durable task/order store は保持すべき優位点である。

## 9. 次工程へ渡す依存順序

これは設計決定ではなく、現状ギャップから導かれる dependency order である。

1. **Single mediation actor**: `secure_mediator` の workflow から payment capability に到達できる一つの execution boundary を確立する。
2. **Plan binding**: boolean ではなく、承認済み plan/snapshot、選択 agent、上限金額等と、その後の payment Task を結合できる現行ギャップを解消する。
3. **Two approvals**: plan approval と Human Present payment consent を別イベントとして保存・検証する。
4. **AP2 core first**: official closed Mandate schema、signed Mandate presentation、role-specific deterministic verification、signed Checkout/Payment Receipts を採用する。
5. **x402 transport next**: official URI/activation、dotted metadata、v1 requirements/payload、task correlation、receipt history を A2A boundary に適用する。
6. **simulation label**: on-chain facilitator/transaction のない local rail は明示的な non-conformant simulation として残し、wire-shape conformance と settlement conformance を分けて表示する。

この順序により、AP2 認可を独立した payment demo に追加してしまうのではなく、まず本来の仲介主体の execution path に決済を組み込み、その後に認可と transport を公式境界へ合わせられる。

## 10. 現時点で主張できること / できないこと

主張できる:

- AP2 v0.2 **shaped closed Mandate content** を使った Human Present simulation。
- A2A Task の `input-required` / `completed` を利用する project-local payment extension。
- durable nonce/idempotency/state/evidence と厳密な transaction binding を備えた simulated marketplace rail。
- 決済証跡を legacy LLM orchestration に渡さない分離境界。

まだ主張できない:

- AP2 v0.2 conformance。
- A2A x402 Payments Extension v0.1 conformance。
- on-chain x402 settlement または legal payment guarantee。
- `secure_mediator` が plan から paid fulfillment までを仲介する end-to-end flow。
- Human Not Present / autonomous AP2 flow。

## 11. 調査上の制約

- 本 step ではコード変更、API 実装、schema migration、deployment 変更を行っていない。
- host に `uv`、現在の Python environment に `pytest` がなく、既存 container にも repository の payment tests が配置されていなかったため、baseline test suite は再実行していない。
- 静的には payment marketplace 関連に `57` 個の `test_*` function を確認したが、parametrization を含む実行 case 数や pass 数は本調査では主張しない。
- 公式 repository は mutable な `main` ではなく、上記 SHA の permalink と content hash に固定して評価した。
