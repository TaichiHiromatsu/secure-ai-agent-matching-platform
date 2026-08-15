# AP2 / A2A x402 統合設計レビュー

- レビュー日: 2026-08-15 (Asia/Tokyo)
- 対象: `docs/AP2_X402_INTEGRATED_DESIGN.md` 1.0-design
- 反映版: 1.1-design-reviewed
- 工程: Section 12 Step 5（設計レビューのみ。コード実装なし）
- 総合判定: **承認 — simulation-only／明示的な耐久 single-host target の実装計画へ進めてよい**
- 非承認範囲: **現行 Cloud Run deployment と official x402 profile は非承認／無効**。それぞれ durable backend、network／asset／wallet／facilitator／amount policy、ACC-030 が揃うまで有効化／適合表示してはならない。

## 1. 独立確認

要件 1.1-reviewed と要件レビューを全行確認し、現行 code/config の root agent、planner/orchestrator、payment API、Merchant service/DB、Agent Card、nginx/auth、supervisor、Docker/Cloud Run、lock fileを照合した。特に次を実コードで確認した。

- `secure_mediation_agent/agent.py` は LLM tool が ADK session boolean `plan_approved` を更新する。
- `user-agent/agent.py` は textを `strip()` し、payment-only rootとして `:8004` を直接呼ぶ。
- `deploy/auth/verify.py` は200/401だけを返し verified subjectをupstreamへ伝えず、`deploy/nginx.conf` も主体headerを注入しない。一方 pinned ADK Web 1.19.0 API は client requestの `user_id` をsession/run authorityとして受ける。
- mediation DBは `/app/payment-data/marketplace.db`、Merchant DBは `/app/payment-data/paid-agent.db`、evidence DBは `/app/payment-evidence/evidence.db` であり、設計draftの `business.db` は現行sourceではない。
- Merchant `:8005` は独立SQLiteを所有する。mediation DBのTask rowだけでは Merchant Task authority/restartを代用できない。
- pinned `a2a-sdk==0.3.19` の `DefaultRequestHandler` は初回Messageに未知の `taskId` があると `TaskNotFound` にする。draftのclient事前割当Task IDはstock handlerのままでは動かない。
- current Cloud Run deployは `max-instances=1` だが persistent volume/databaseを設定せず、`/app` SQLiteはcontainer recreationでdurableではない。

primary source は pinned exact contentを再取得し、AP2 spec SHA-256 `32c3be5011f481d2760e56e7b9935b0842c3da0d5f7d7b8a68402a599f1e6aa3`、x402 v0.1 spec SHA-256 `5cdc35ed8c4d7a93bb120f1782fd06e2cc3ef19036684f772e27d0d644c66940` の一致を確認した。AP2 SDK root package、`MandateClient`、generated Mandate/Receipt models、`ReceiptClient`、AP2 Human Present flow、x402全flow、A2A Python v0.3.19 types/default handler/request context、ADK Web v1.19.0 user/session APIを照合した。

## 2. 重要度順の指摘と解決

### ブロッカー

#### DR-B01 — authenticated user が approval authority に届かない

- 参照: draft Design §5.2, §10.1, §11.1; requirements `ROLE-003`, `TRUST-001`, `WF-001`, `PLAN-008`, `APPROVAL-003`〜`APPROVAL-006`, `SEC-003`, `SEC-004`, `ACC-019`, `ACC-023`〜`ACC-025`
- 問題: draftは「auth layerから tenant/customerを得る」とだけ書いたが、現行auth/nginxはverified subjectをADKへ渡さず、ADK Webはbody/pathの `user_id` を受ける。client-controlled identityでplan/payment approvalを別主体へbindingできる余地があり、二承認の信頼根拠が成立しない。
- 修正: Design §2〜§5、§10〜§13へ `VerifiedIdentityMiddleware` / `IdentityBroker`、custom ADK ASGI wrapper、short-lived service assertion、fixed one-tenant demo mapping、CLIのauthenticated nginx route、raw tokenをagent stateへ置かない規則を追加した。auth/nginx/web appのfile mappingとidentity forgery testsも追加した。
- 解決: **Resolved in design**。実装開始gateは、forged ADK `user_id`/header/CLI subjectを拒否する executable test。

#### DR-B02 — `payment_approved` が要件上のevidence完成前に開く

- 参照: draft Design §5.1, §6.3, §7.2; requirements §4 state invariant, `WF-004`, `WF-005`, `WF-007`, `AP2-009`〜`AP2-011`, `GATE-010`, `RES-004`, `ACC-010`, `ACC-020`
- 問題: draftはexact payment approvalで `payment_approved` へ移り、その後outboxでTrusted Surface/CP/signingを実行した。しかし要件の `payment_approved` はsigned Mandatesとcredentialが既に存在する状態であり、crash windowでsubmit gateが不完全evidenceをauthorityとして扱う。
- 修正: Design §5.1にdurable internal state `payment_authorizing` を追加。approval record/nonce consume後はこのstateに留まり、Mandates/credential/payload exact evidenceのcommit後だけCASで `payment_approved` へ進む。Trusted Surface issuance gateとrestart/UI/test matrixも追加した。
- 解決: **Resolved in design**。

### 重要

#### DR-H01 — Merchant Task authority と mediation mirror が混在

- 参照: draft Design §3, §9.1, §9.3, §13.1; requirements `WF-001`〜`WF-005`, `ROLE-004`, `TRUST-002`, `DATA-001`, `RES-004`, `RES-005`, `ACC-012`, `ACC-020`
- 問題: draft ERDは中央 `merchant_tasks` をTask authorityのように扱う一方、現行Merchantは別process/DBである。shared central rowではMerchant独自のTask history、capability consume、duplicate Message、restart sourceが曖昧になる。
- 修正: `/app/payment-data/paid-agent.db` をauthoritative Merchant store、`marketplace.db.merchant_task_mirrors` をauthenticated A2A observationに固定。Merchant v2 tables/constraints、persistent A2A `TaskStore`、mediationからMerchant DBへのdirect read禁止、`tasks/get` recoveryをDesign §3, §8.5, §9, §13へ追加した。
- 解決: **Resolved in design**。

#### DR-H02 — client-preallocated Task ID が A2A SDK 0.3.19 と非互換

- 参照: draft Design §6.3, §8.3; requirements `WF-003`, `PLAN-014`, `X402-007`〜`X402-010`, `TEST-003`, `ACC-006`, `ACC-012`
- 問題: pinned SDK stock handlerは初回 `Message.taskId` が未登録なら拒否する。draftの「controllerがIDを先に割当てMerchantが採用」にはadapter手順がなく、そのまま実装できない。
- 修正: Design §8.5で初回標準 `taskId` は省略し、signed start capability内の予約IDをactivation/auth/capability検証後にcustom `AuthorizedRequestContextBuilder` が `RequestContext.task_id` へ設定する方式を固定。以後だけ標準task correlationを使う。SDK model/kind/Artifact/Card modesとupstream contract spikeをimplementation order先頭へ追加した。
- 解決: **Resolved in design; executable spike required before implementation**。

#### DR-H03 — AP2 Payment Receipt issuer/timing と pinned helperのdefaultが不正確

- 参照: draft Design §7.5; requirements `AP2-018`〜`AP2-021`, `ERR-005`, `TEST-002`, `ACC-016`; AP2 spec Verification / Human Present flow
- 問題: draftはCP/Network/MPPがaccept/rejectごとにPayment Receiptを発行するよう読めた。AP2ではCP/Network successはcredentialを返し、error時にError Receipt、final payment accept/rejectはMPP Receiptである。また pinned `ReceiptClient.create_payment_receipt()` はSuccess-onlyでissuerをPISP/emptyから選び、role-specific Error Receiptを作らない。
- 修正: Design §7.5でCP/Networkはreject時だけError Receipt、MPPはfinal Success/Errorと固定。generated discriminated model + `create_jwt` の `Ap2ReceiptFactory`、configured issuer/kid、事前予約 `payment_id`、Receipt verifierを定義した。settle success/commit failureはPayment success + Checkout Error + local refundとした。
- 解決: **Resolved in design; pinned SDK receipt spike required**。

#### DR-H04 — migration source、三DB cutover、post-write rollbackが不安全

- 参照: draft Design §9.1, §12.1, §13.1; requirements `MIG-001`〜`MIG-006`, `DATA-007`, `TEST-012`, `ACC-026`, `ACC-032`
- 問題: draftは現行にない `business.db` を継続するとし、Merchant DBをmigration対象に含めなかった。毎startup timestamp backupはidempotentでなく、v2 write後にpre-v2 backup + v1 binaryへ戻る手順は新dataを失う。
- 修正: Design §9/§12で現行三pathを固定しwriter quiescence、content/migration-IDで一回だけの三backup、guarded v1 schema table alteration、三DB integrity/cross-reference checkを追加。pre-cutoverだけrestore可、post-writeはv2 DB保全 + compatible rollback/forward fixとした。
- 解決: **Resolved in design**。

#### DR-H05 — current Cloud Run はdurable workflowを満たさない

- 参照: draft Design §2, §16; requirements `WF-001`, `RES-004`〜`RES-006`, `DATA-007`, `TEST-014`, `ACC-020`, `ACC-032`
- 問題: `max-instances=1` はlocal filesystem persistenceを提供しない。現行deploy scriptにvolume/shared DBがなく、restart/container recreation acceptanceを満たせない。
- 修正: Design §1, §13, §16でaccepted targetをexplicit durable POSIX volumeのsingle-host simulationに限定し、ephemeral filesystemでpaid readinessを拒否。current Cloud Runを明示blockし、durable backend導入後のACC-020/032再実行を要求した。
- 解決: **Resolved by scope boundary**。Cloud Run integrated paid deploymentは残存 blocker。

#### DR-H06 — official x402 token units と AP2 ISO minor unitsのmapping欠落

- 参照: draft Design §8.1; requirements `AP2-016`, `AP2-023`, `X402-006`, `X402-009`, `X402-024`, `SCOPE-008`, `ACC-030`
- 問題: draftはAP2 `1250 USD` とtoken `maxAmountRequired` の換算policyを定義せず、同じ整数を誤用できた。
- 修正: Design §8.1にFXなし `iso-token-exact/1` を追加し、currency/token decimalsからinteger-only scaling、plan/UI/credential/payload binding、非整除拒否を定義。official enablement条件へamount policyを追加した。
- 解決: **Resolved in design; official profile remains disabled**。

### 中程度

#### DR-M01 — activation echo failure後の「Task 0件」は分散境界では保証不能

- 参照: draft Design §8.2; requirements `X402-003`, `X402-004`, `GATE-004`, `ACC-007`
- 問題: clientがresponse echo欠落を観測する時点ではMerchantがTaskをdurable作成済みの場合がある。proxy header strip後に物理Task 0件へ戻すことはappend-only/durabilityと両立しない。
- 修正: request activation mismatchはTask前に0件、application success responseはTaskとechoを同一builderで生成、transport-level echo不明は同一operation/Taskを照会して `reconciliation_required`、新Task/payment approval 0件とした。
- 解決: **Resolved for controlled acceptance boundary**。literal ACC-007をuntrusted intermediaryまで広げる場合は要件clarificationが必要。

#### DR-M02 — content-addressed evidence refs のread authorizationが再利用曖昧

- 参照: draft Design §8.3, §10.2; requirements `TRUST-004`, `PLAN-014`, `AP2-023`, `GATE-010`, `SEC-003`, `DATA-002`
- 問題: submit capability一個で三evidenceをfetchするとsingle-use nonceとretryが衝突し、digestだけでread可能にするとproof disclosureになる。
- 修正: evidence ID/digest/workflow/task/Merchant audienceごとのshort-lived `evidence:read` grantを追加。同じreference/read idempotencyだけexact saved bytesを返す。
- 解決: **Resolved in design**。

#### DR-M03 — legacy flag が統合image内のbypassを再開し得る

- 参照: draft Design §12.2; requirements `GATE-001`, `GATE-005`, `COMP-004`, `COMP-005`, `ACC-005`, `ACC-025`
- 問題: `ENABLE_LEGACY_PAYMENT_DEMO=1` がsame image/processで旧root/APIを再有効化する余地があった。
- 修正: integrated imageではflag復活を禁止。legacy testsは別image/process、別v1 DB copy、loopback-only、統合rail/keyなしとした。
- 解決: **Resolved in design**。

#### DR-M04 — active workflow unique constraintがsession再利用を永久に阻害

- 参照: draft Design §9.1; requirements `WF-001`, `RES-004`, `UI-006`, `UI-007`
- 問題: unconditional `UNIQUE(tenant,session,context)` はcompleted後も同じcontextで新workflowを作れない。
- 修正: terminalを除くpartial uniqueへ変更し、active lookupを一意にしつつ後続workflowを許可した。
- 解決: **Resolved in design**。

#### DR-M05 — keyless planner/free compatibilityのprocess edgeが未定義

- 参照: draft Design §2〜§4, §5.1; requirements `ROLE-002`, `ROLE-003`, `COMP-001`, `UI-007`, `ACC-024`, `ACC-027`
- 問題: `StructuredPlanner` のownerが`:8004`かLLM process`:8000`か曖昧で、CLI parityとkey isolation、既存anomaly/final validation保持を同時に満たせなかった。
- 修正: non-public service-auth `PlannerExecutionGateway` / free execution gatewayをkeyless `:8000` に置き、`:8004` ownerがoutbox/idempotencyで呼ぶ方式を固定した。
- 解決: **Resolved in design**。

## 3. 問題なしと確認した事項

- plan approvalとpayment approvalは別intent/nonce/evidence/stateで、AP2 Mandateをplan authorizationに流用しない。
- paid MerchantがAP2 Merchant/payeeで、official profileだけ同主体のx402 Merchant/payTo walletになる。旧platform-payee/guarantee/payoutは新flowから隔離される。
- AP2 closed Checkout/Payment Mandateのexact `vct`、Checkout JWT hash、Payment `transaction_id`、receipt closed-leaf referenceはpinned schema/SDKと整合する。
- project-local CP credentialとx402 payloadの一方向digest bindingは循環hashを避け、AP2/x402 official objectを拡張しない。
- simulation/official profileのURI、Card、activation、rail、label、reportは分離され、official未準備時fallbackしない。
- x402 dotted metadata、v1 payload、original Task correlation、payment-rejected、all-attempt receipt history、work-before-settle prepare/commitはpinned v0.1と整合する。
- outbox/CAS/idempotency/nonce、unknown settlement reconciliation、append-only refundはrestart/concurrency test seamを持つ。

## 4. 残存リスクと実装開始条件

1. **P0 first gate:** verified identity wrapperを実装し、ADK body/path user、forged header、CLI、DEV fixed identityのnegative testsがpassするまでapproval signingを実装しない。
2. **P0 compatibility gate:** pinned AP2 Git dependencyとA2A 0.3.19 custom handler/TaskStoreのexecutable spikeを先に行う。失敗時はwire/profile semanticsを変えずdesignへ戻す。
3. **Deployment gate:** local/single-hostでも三DB directoryを明示volume mountしない構成はACC-020/032対象外。current Cloud Runはpaid disabled。
4. **Official gate:** network/token/wallet/facilitator/TLS/amount policyとACC-030がない限りcanonical URIはCard/header/reportへ出さない。
5. `a2a-sdk` custom context builderとdefault handler内部契約は0.3.19にversion-coupledである。upgradeはdedicated adapter contract testを必須とする。
6. response echoをuntrusted proxyがstripするケースはTask orphanを完全に0件へできない。設計はsame-ID reconciliationで追加business effectを0件にする。
7. SQLiteはsingle-host restrictionである。multi-instance/shared filesystemへ拡張する場合はDB/queue architecture reviewをやり直す。

## 5. 判定

**APPROVE**。

反映後のDesign 1.1は、explicit durable single-hostのsimulation-only implementation planningへ進めてよい。未解決のBLOCKER/HIGH design contradictionはなく、上記P0 gatesはimplementation order §15の最初に置かれた。

ただし次は明示的に **REJECT / NOT READY** である。

- 現行ephemeral Cloud Run deploymentでintegrated paid workflowをenableすること。
- official x402 profile/canonical URI/conformance claimをenableすること。
- integrated image内でlegacy payment-only root/APIをflag復活させること。
