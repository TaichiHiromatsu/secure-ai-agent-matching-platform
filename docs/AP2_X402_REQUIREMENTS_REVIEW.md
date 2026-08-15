# AP2 / A2A x402 統合要件レビュー

- レビュー日: 2026-08-15 (Asia/Tokyo)
- 対象: `docs/AP2_X402_INTEGRATED_REQUIREMENTS.md` 文書版 1.0-draft
- レビュー反映版: 1.1-reviewed
- 工程: Section 12 Step 3（要件レビューのみ。設計・実装は対象外）
- 結論: **要件は設計工程へ進めてよい（設計着手を承認）**。ただし simulation は project-local の非準拠 profile とし、ACC-030 未達の build で official x402 profile を有効化したり、適合を表示したりしてはならない。

## 1. レビュー基準と確認方法

次の pinned primary source を独立に取得し、文書記載の commit と content hash を確認した。

| 対象 | 固定 commit | 確認結果 |
| --- | --- | --- |
| [AP2 v0.2 specification](https://github.com/google-agentic-commerce/AP2/blob/e1ea56db72a6385bce3e5c1112b3a56ce60acb43/docs/ap2/specification.md) と関連 flow/authorization/schema | `e1ea56db72a6385bce3e5c1112b3a56ce60acb43` | spec SHA-256 `32c3be5011f481d2760e56e7b9935b0842c3da0d5f7d7b8a68402a599f1e6aa3` 一致 |
| [A2A x402 Payments Extension v0.1](https://github.com/google-agentic-commerce/a2a-x402/blob/125db5526a965d2325459d1a9df2e274a7e42396/spec/v0.1/spec.md) | `125db5526a965d2325459d1a9df2e274a7e42396` | spec SHA-256 `5cdc35ed8c4d7a93bb120f1782fd06e2cc3ef19036684f772e27d0d644c66940` 一致 |

AP2 は `specification.md`、`flows.md`、`agent_authorization.md`、Checkout/Payment Mandate 文書、canonical Mandate/Receipt schemas、pinned Python SDK の generated models/receipt helper を照合した。x402 は declaration、activation、PaymentRequired、submission、state、receipt history、error、security の全節を照合した。

## 2. 重要度順の指摘と解決

### ブロッカー

#### RQ-B01 — simulation が official x402 extension を実装したように見える

- 影響 ID: `SCOPE-006`〜`SCOPE-008`, `BASE-003`, `ROLE-004`, `ROLE-009`, `X402-001`〜`X402-004`, `X402-019`〜`X402-021`, `GATE-004`, Appendix A, `ACC-006`〜`ACC-008`, `ACC-014`, `ACC-028`〜`ACC-030`
- 問題: draft は canonical URI を宣言・activation しながら `exact-simulated`、`demo:local`、`USD`、synthetic transaction を使用し、「wire-compatible simulation」と表示できた。pinned x402 v0.1 は on-chain cryptocurrency payment、wallet-signed payload、facilitator verify/settle、blockchain network、token contract、on-chain transaction をプロトコルの意味としている。data shape だけの一致で canonical extension support を広告するのは不正確である。
- 修正: official on-chain profile と project-local simulation profile/URI を分離した。simulation は canonical URI を宣言・activationせず、表示を `wire-shape test fixture (NOT CONFORMANT)` に限定した。official profile enablement は TLS、wallet、facilitator、実 network/asset/transaction と `ACC-030` pass を必須にした。
- 解決状態: **Resolved**。

#### RQ-B02 — payment submission 以降の direct bypass gate が不足

- 影響 ID: `GATE-001`〜`GATE-005`, `COMP-005`, `TEST-006`, `ACC-005`, `ACC-025`
- 問題: order/task creation は plan authorization を要求したが、保持可能な `submit_payment` や verify/settle/fulfill entrypoint に、payment approval、Mandates、credential、original Task の必須 gate が明記されていなかった。plan authorization または order ID だけで後段へ直行できる余地があった。
- 修正: `GATE-010` を追加し、全 transport の submit/verify/settle/fulfill に current state、payment approval、signed Mandates、scoped credential、original task/context、selected profile、operation capability、idempotency を要求した。`COMP-005`, `TEST-006`, `ACC-005` も後段 endpoint を含むよう更新した。
- 解決状態: **Resolved**。

### 重要

#### RQ-H01 — single-use plan approval nonce の service 間再利用が曖昧

- 影響 ID: `WF-002`, `PLAN-009`〜`PLAN-011`, `GATE-002`〜`GATE-004`
- 問題: approval nonce を最初の consumption で消費する一方、payment service と Merchant の双方に同じ signed plan authorization を要求していた。同一 token を転送すれば二番目の verifier が replay と判定し、再利用可能にすれば bypass 範囲が広がる。
- 修正: primary approval nonce は `plan_approval_required → plan_approved` で一回だけ consume し、その後は元 approval/plan digest に拘束した audience/operation/workflow/task 別 downstream capability を発行するモデルに変更した。`PLAN-014` を追加し、capability ごとの nonce/idempotency を必須化した。
- 解決状態: **Resolved**。

#### RQ-H02 — x402 Signing Service/wallet と AP2 credential の結合が未定義

- 影響 ID: role mapping, `ROLE-011`, `AP2-009`〜`AP2-011`, `AP2-023`, `AP2-024`, `X402-009`〜`X402-011`, `X402-024`, `ACC-010`, `ACC-012`〜`ACC-014`, `ACC-017`, `ACC-030`
- 問題: draft は CP token と x402 `PaymentPayload` の両方を要求したが、誰が wallet payload を作り、両 object が同じ Mandate/requirements/task/payment approval を表すことをどう保証するかがなかった。Merchant が AP2 で検証した credential と facilitator へ渡す payload を差し替えられる余地があった。
- 修正: deterministic Signing Service/wallet boundary を CP/payment-support 側に明記した。official profile では signed Payment Mandate、CP credential、accepted requirements、wallet payload、task/context、plan capability を exact digest で一対一に相関し、Merchant→MPP/facilitator でも再検証する。final AP2 Payment Receipt と selected-profile settlement receipt も同じ attempt/network/transaction へ拘束する。simulation payload/reference は synthetic として分離した。
- 解決状態: **Resolved**。

#### RQ-H03 — AP2 Receipt の canonical fields と rejection issuer flow が不完全

- 影響 ID: `AP2-018`〜`AP2-020`, `ERR-005`, `TEST-002`, `ACC-016`
- 問題: Payment Receipt の Error にも必須な `payment_id`、Success の `psp_confirmation_id`/`network_confirmation_id`、Checkout Receipt Success の `order_id`、共通 `status/iss/iat/reference` が列挙されていなかった。また MPP rejection に偏り、Merchant の Checkout rejection、CP/Network の Payment Mandate rejection が曖昧だった。
- 修正: pinned canonical schema の Success/Error discriminator と必須 field を正確に要求した。Merchant は Checkout Receipt、CP/Network/MPP は各 verifier boundary の Payment Receipt を返す flow を明記し、transport/service-auth failure と Mandate Action Authorization rejection を区別した。
- 解決状態: **Resolved**。

#### RQ-H04 — mandatory refund/reconciliation scope に受入証拠がない

- 影響 ID: `SCOPE-001`, `WF-010`, state model, `DATA-003`, `RES-005`, `RES-006`
- 問題: refund/reconciliation を必須 release scope としたが、refund record の意味、AP2/x402 receipt との関係、authoritative reconciliation、acceptance test が不足していた。
- 修正: `WF-012`, `WF-013`, `TEST-017`, `ACC-034`, `ACC-035` を追加した。refund は AP2/x402 object を上書きしない project-local compensating record、reconciliation は既存 external ID の照会のみとし、追加 charge を禁止した。AP2/x402 が refund protocol を定義すると主張することも禁止した。
- 解決状態: **Resolved**。

### 中程度

#### RQ-M01 — x402 の `payment-rejected` branch が欠落

- 影響 ID: state model, `X402-023`, `TEST-003`, `TEST-016`, `ACC-033`
- 問題: pinned x402 v0.1 は Client rejection を `x402.payment.status: payment-rejected` とするが、draft は UI cancellation を local `cancelled` にするだけで original Task への correlated Message を要求していなかった。
- 修正: original `taskId` を持つ rejection Message、payload/settlement/success Receipt/fulfillment 0件、同一 Task の非成功終了を要求・受入条件化した。
- 解決状態: **Resolved**。

#### RQ-M02 — AP2 に存在しない CP credential wire schema を official fixture のように扱う

- 影響 ID: `AP2-009`, `AP2-011`, `TEST-002`, resolved-decision table
- 問題: AP2 は CP が scoped payment credential/token を返す責務を定義するが、その wire schema や role 間 API は Commerce Protocol/payment ecosystem の範囲であり AP2 v0.2 canonical schema ではない。draft の contract-test 表現は official CP credential schema があるように読めた。
- 修正: credential の field/binding は project-local profile と明記し、AP2 schema conformance test ではなく独立した binding/security test とした。Mandate/Receipt official objectsの中へ project field を挿入しない要件は維持した。
- 解決状態: **Resolved**。

#### RQ-M03 — x402 failure code と stable domain error の mapping が未指定

- 影響 ID: `ERR-001`, `ERR-004`, `X402-016`
- 問題: `payment-failed` は要求されていたが、pinned common error codes と local stable code の deterministic mapping が acceptance から抜けていた。
- 修正: `X402-025` を追加し、pinned common codes または versioned safe extension code を `x402.payment.error` に使用し、domain error と決定論的に対応付けるようにした。`TEST-003`, `ACC-028` へ coverage を追加した。
- 解決状態: **Resolved**。

## 3. 問題なしと確認した重要論点

- `APPROVAL-001`〜`APPROVAL-007` は plan approval と payment approval を state、ID、intent、nonce、signature、event で分離しており、AP2 Mandate と plan approval を混同していない。
- `ROLE-004`/`ROLE-005` は paid external Merchant を AP2 payee とし、旧 platform-payee marketplace flow を隔離している。AP2 `PaymentMandate.payee`、Merchant-signed Checkout/Receipt の主体関係と整合する。
- `AP2-003`/`AP2-004` は `mandate.checkout.1`、`mandate.payment.1` と AP2 protocol v0.2 を混同せず、pinned canonical schema の exact `vct` と Checkout hash binding に一致する。
- `X402-005`〜`X402-010`, `X402-013`〜`X402-018` は dotted sibling metadata、`x402Version: 1`、original `taskId`、append-only all-attempt receipts、payment failure semantics と一致する。
- `GATE-001`, `GATE-003`〜`GATE-005`, `COMP-004`, `COMP-005` は payment-only root、public legacy route、Merchant direct invocation の bypass を閉じる方向で整合している。今回 `GATE-010` で後段も閉じた。

## 4. 残存リスクと設計上の制約

1. pinned x402 spec は「four distinct architectural roles」と記す一方、role heading は Client Agent と Merchant Agent の二つだけで、Signing Service と facilitator は本文中の logical boundary として現れる。要件は両者を明示的 trust boundary とした。後続設計で同居させても検証・key・audit boundary を消してはならない。
2. AP2 Checkout JWT は rainbow-table mitigation のため deterministic signature を禁止する。algorithm、fresh entropy の入れ方、key storage は未決定だが、`AP2-002`, `TRUST-003`, contract fixtures を満たす必要がある。
3. AP2 credential/token と AP2 role API は project-local profile である。外部 Merchant/CP/MPP との相互運用には、AP2 適合とは別にその profile 契約が必要である。
4. x402 の work-before-settle は SHOULD であり、不可逆予約等では prepare/commit または文書化した deviation が必要である。`WF-011`/`X402-017` の設計証拠が必要である。
5. official x402 profile は network、asset、wallet、facilitator が未選定である。これは simulation-only design の blocker ではないが、canonical URI の runtime enablement と x402 compatibility/conformance claim の blocker である。
6. review は要件と primary source の静的整合性を対象とし、コード、schema migration、container、E2E の実現可能性は未検証である。後続設計は requirement-to-component と requirement-to-test の traceability matrix を作る必要がある。

## 5. 設計可否判定

**APPROVED FOR DESIGN**。

1.1-reviewed では、未解決の BLOCKER/HIGH requirement contradiction はない。設計 baseline は simulation-only integrated flow とし、simulation Agent Card/runtime は project-local URI と `NOT CONFORMANT` label を使用する。official x402 on-chain profile の設計を並行してよいが、SCOPE-008 と ACC-030 を満たすまで enable/適合表示してはならない。

設計レビュー時の必須 exit checks は次のとおり。

- primary plan approval consume と downstream scoped capability の sequence/nonce table がある。
- AP2 Checkout/Payment Mandate、CP credential、x402 payload、二つの AP2 Receipt、selected-profile receipt の exact-byte/digest graph がある。
- every start/submit/verify/settle/fulfill route が `GATE-003` または `GATE-010` に割り当てられている。
- simulation と official profile の Agent Card、URI、rail、test report、UI label が同時に混在しない。
- `ACC-001`〜`ACC-035` のうち selected release に必要な acceptance と test owner が trace されている。
