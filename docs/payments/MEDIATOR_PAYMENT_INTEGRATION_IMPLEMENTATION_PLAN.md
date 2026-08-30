# 仲介エージェント決済統合：final6実装・完了計画

> [!WARNING]
> この文書は作成時点の実装計画snapshotであり、現在仕様の正本ではない。現行責務は[アーキテクチャ](ARCHITECTURE.md#actorと責務の正本)と[Payment Bridge設計](mediator-payment-integration-design/04_PAYMENT_BRIDGE_AP2_X402.md)を参照する。本文は履歴証跡として変更しない。

- status: **local final6 implementation complete / external release gates not run**
- updated: 2026-08-17（Asia/Tokyo）
- exact image: `sha256:3e4e089643564e00bd6563d08a575fcf2aa2eff94ae60fd1ca518900022a89f0`
- normative source: [REQUIREMENTS](MEDIATOR_PAYMENT_INTEGRATION_REQUIREMENTS.md)
- design source: [設計一式](mediator-payment-integration-design/README.md)
- evidence source: [TEST REPORT](MEDIATOR_PAYMENT_INTEGRATION_TEST_REPORT.md)
- deployment: **NOT RUN**

## 1. 現在の完成範囲

final6は、local single-host／single-containerの概念決済デモとして次を実装・検証済みである。

| Scenario | current result |
| --- | --- |
| `PAID-HAPPY-01` | 通常依頼、計画完全一致承認、closed payment target完全一致承認、AP2 Mandate検証、simulation guarantee、same Task resume、settlement、fulfillment、`Completed` |
| `FREE-HAPPY-01` | 同じmediator入口からfree Agentを実行し、payment／guarantee／settlement／refundを作らず`Completed` |
| `REFUND-01` | settled paid operationのfulfillment failureを`RefundPending`へ移し、別の完全一致承認後に相関付き全額refundを一回だけ実行して`Refunded` |
| `PRIVACY-01` | DOM、console、network bodyにcookie、CSRF、identity assertion、Mandate、capability、proof、JWT、private keyを露出しない |

本実装が提供するのは `x402-wire-simulation/1` とデモ独自の `signed-payment-guarantee/1` である。official x402、wallet、facilitator、on-chain settlementへの適合は主張しない。

## 2. final6 runtime topology

```text
Browser / same-origin session
  -> nginx exact route and method allowlist
  -> Firebase-session boundary / signed internal identity
  -> payment_user_agent
  -> HttpMediationAuthority
  -> loopback workflow API :8004
  -> MediationController (single public mutation authority)
       -> typed matcher / planner
       -> exact plan approval
       -> shared A2A executor + legacy callback + stable gates
       -> free Agent :8002 OR Merchant / paid Agent :8005
       -> exact payment or refund approval
       -> PaymentBridge / AP2 verifier / simulation rail
       -> final validator
  -> SQLite v4 mediation/payment state + Merchant DB + evidence DB
  -> outbox worker
```

公開mutationは `POST /mediation-api/v1/turns`、readは `GET /mediation-api/v1/view` に限定する。subject、tenant、ADK sessionはproxyが検証したidentityから決まり、body／path／headerのworkflow selectorをauthorityにしない。`requestId`、canonical digest、expected version、owner scopeをCAS／replay境界にする。

## 3. SQLite v4 closure

local durable profileは `MEDIATION_STORE_MODE=sqlite` を使い、schema v4で次を追加する。

- `mediation_sessions_v4`: encrypted authoritative session、owner HMAC、version、state、key sentinel binding。
- `mediation_requests_v4`: owner＋request ID＋canonical digest reservation、encrypted exact result、replay／conflict判定。
- AES-GCM AAD: owner、request、session、version、schemaを結合。
- `schema_migrations`: marketplace、Merchant、evidenceの三DBをv4として確認。
- readiness: store mode/profile/schema、writable、decryptable、worker、outbox、profileをfail closedで検査。

v3 payment／Merchant／evidence rowを破壊的に書き換えず、v4 mediation tableをadditiveに導入する。pre-sentinel v4や復号不能keyを自動採用しない。

## 4. 完了工程

| Phase | 完了内容 | final6 evidence |
| --- | --- | --- |
| 0. Scope／design | 139 normative ID、Release-1必須126＋future 13、設計owner一意化 | `11_TRACEABILITY_RELEASE.md` YAML |
| 1. Single authority | public rootとworkflow APIを一つの`MediationController`へ統合 | integration／container／browser suites |
| 2. Identity／public boundary | session cookie、Origin、CSRF、signed identity、selector拒否、internal route deny | security 84、container 16、wrong-owner `null` |
| 3. Paid／free | typed matcher／planner、二承認、same Task、無料時payment 0 | canonical payment 285、browser paid/free |
| 4. Payment／AP2 simulation | Checkout／Payment Mandate、capability、guarantee、settlement相関 | AP2 contract 17、x402 simulation contract 2 |
| 5. Refund | exact owner-bound approval、one-shot local fault、相関refund | browser refund、Merchant fault tests |
| 6. Persistence | SQLite v4、request reservation、exact replay、restart recovery | paused v2／Completed v5 restart、三DB `quick_check=ok` |
| 7. Exact candidate | canonical regression、browser artifact、11-marker validatorを同一digestへbinding | three final6 JSON artifacts |
| 8. Cleanup | final6 runtime containerと三つの専用volumeだけを削除 | cleanup observation |

## 5. final6 verification binding

| Item | Value |
| --- | --- |
| image | `sha256:3e4e089643564e00bd6563d08a575fcf2aa2eff94ae60fd1ca518900022a89f0` |
| regression manifest | `sha256:7d72de56a96a3f7438b539e0131167e3e7c9acd2c8e0fa204916dbdd7cfd7339` |
| release manifest | `sha256:852aeaba0e024469eb35adfa45a1dd6fabd054484d68aa1b58739ddaf8457f37` |
| regression artifact | `artifacts/regression-result-final6.json` / `f64da6ec882b3a6a14f27a8df5448ad971c01c208c3f8bcf6070335edfa84ded` |
| browser artifact | `artifacts/browser-evidence-final6.json` / `1059985e2fac45b8c7c70ed316e2359d1c6da64acc004ebf0207560a3796fa50` |
| release validation | `artifacts/ap2-x402-release-validation-final6.json` / `4f4aa723d9a5bc02eec4c09d6f097c749f2d6f6652c66f0dc2b0a72573cf96ce` |

canonical runnerはpayment 285、evaluation 17、jury 13（5 PASS／8 allowed skip）をPASSした。11-marker validatorは全markerをfailure／error／skip 0でPASSした。環境変数なしraw fullは304 PASS／3 FAIL／8 skipであり、3 FAILはすべて既存evaluation-runnerのW&B API key未設定である。

## 6. Cloud Runとexternal gate

Cloud Run向けprofileはlocal durable profileと別契約にする。

- `EPHEMERAL_CLOUD_RUN_DEMO=true` と `MEDIATION_STORE_MODE=memory` を同時に必須とする。
- `durability=NOT PROVIDED`、state reset warning、simulation／`NOT CONFORMANT`を表示する。
- Cloud SQL、外部永続DB、新service、他project／region／serviceを追加・変更しない。
- local v4 restart PASSをCloud Run instance replacementの回復証拠へ転用しない。
- fixed service更新scriptのlocal guard testがPASSしても、実revision／tag／trafficを検証済みとは扱わない。

本作業ではbuild、push、revision作成、tag、traffic、IAM、Origin、Cloud SQLを含むCloud Run操作を一切行っていない。

## 7. 未完了gate

次はfinal6 local simulationの完成範囲外であり、完了するまでCloud Run promotionまたはproduction-readyを主張しない。

1. 126件を個別PASSへ結ぶcandidate-specific 139 ledger。現在のdesign mappingは139 exactly-onceだが、candidate verification statusはrequired 126=`PARTIAL`、future 13=`DESIGNED`である。
2. candidate-bound conformance report。release validatorの`conformanceReportDigest`は`null`である。
3. 実Firebase credential／ID token exchange。
4. Vertex ADC、service account、IAM、quota、model availability、latency。
5. official x402 wallet／facilitator／on-chain settlement。
6. 実Cloud Run tag URLでのbrowser／boundary／readinessとtraffic 0→promotion／rollback。
7. W&B-enabled evaluation-runnerの認証／serialization修正。canonical release契約は`WANDB_DISABLED=true`でPASSするがraw fullは未greenである。
8. future-work 13件の高度restart、first-response-loss、複雑競合、DNS rebinding、完全edge matrix。

## 8. 完了判定

final6は **local simulation demo candidate verified** である。paid／free／refund／privacy、single authority、SQLite v4 local restart、exact replay、wrong-owner isolation、canonical regression、browser、release validatorは完了した。

一方、**Cloud Run promotion、production durability、official x402／on-chain conformance、全139要件のrelease closureは未完了**である。外部gateまたはcandidate ledgerが欠けた状態を`PASS`へ繰り上げず、結果不明時に新しいTask、payment、refundで補償しない。
