# AP2 / A2A x402 統合実装計画レビュー

- レビュー日: 2026-08-15 (Asia/Tokyo)
- 対象: `docs/AP2_X402_IMPLEMENTATION_PLAN.md` 1.0
- 反映版: `docs/AP2_X402_IMPLEMENTATION_PLAN.md` 1.1-plan-reviewed
- 工程: Section 12 Step 7（実装計画レビューのみ。コード/DB/route/dependencyの実装変更なし）
- 総合判定: **承認 — 明示的な耐久 single-host／single-container の simulation-only 実装に限る**
- 非承認範囲: **official x402 profile／canonical URI／conformance claim と現行 Cloud Run paid deployment は非承認／NOT READY**

## 1. レビュー範囲と確認方法

次の4文書を全行確認し、要件→reviewed design→implementation work package→test/ACC gateの順で照合した。

- `docs/AP2_X402_INTEGRATED_REQUIREMENTS.md` 1.1-reviewed
- `docs/AP2_X402_INTEGRATED_DESIGN.md` 1.1-design-reviewed
- `docs/AP2_X402_DESIGN_REVIEW.md`
- `docs/AP2_X402_IMPLEMENTATION_PLAN.md` 1.0

現行repositoryについては、root/approval boolean、payment-only client、`:8004` legacy API、paid Merchant、Trusted Agent Store、三SQLite path/schema、nginx/auth/supervisor/Docker/local/Cloud Run script、root/subproject test inventory、`pyproject.toml`/`uv.lock`をread-onlyで照合した。特に次を計画の前提として再確認した。

- 現行rootは`secure_mediation_agent/agent.py`のLLM tool/ADK session booleanをapproval authorityとし、`user-agent/agent.py` はtextを`strip()`してpayment-only rootから`:8004`を直接呼ぶ。
- 現行のDB pathは`marketplace.db`、`paid-agent.db`、`evidence.db`の三つで、Merchantは別DB authorityである。
- root payment suiteは57個のtop-level `test_*`を持つが、Trusted Agent Store配下にも独自subproject testがある。最終regressionを新free-flow test一本で代用できない。
- current Cloud Run scriptにSQLiteのdurable backendはない。対象はexplicit durable POSIX mountを持つsingle-host containerに限られる。

## 2. 重要度別の集計

| 重要度 | 未解決 | plan 1.1で解決 | 結論 |
| --- | ---: | ---: | --- |
| P0 | 0 | 0 | implementation planningを即時rejectする矛盾はない |
| P1 | 0 | 6 | 下記全findingを計画本文へ反映済み |

## 3. P1指摘と解決

### IPR-P1-01 — G2 vertical slice が未実装のmatcher/plan assemblyに依存していた

- 参照: Plan 1.0 §3 G2、§5 WP-03 vertical slice A、WP-04。Design §15 implementation order 4〜5。
- 問題: WP-03はWP-04の`EligibilityMatcher`/`PlanAssembler`/signed Plan Authorizationより先なのに、実の`plan_approval_required` viewとrestart後approval targetをexitにしていた。temporary unsigned/dummy planを作るか、依存を逆転させない限り実行不可能である。
- 修正: Plan 1.1 §3 G2と§5 WP-03のexitをdurable `request_received → planning`、dispatcher分類、side effect 0へ縮小した。実plan、`plan_approval_required`、signed approval、restart後approval targetはWP-04 vertical slice Bへ移した。
- 解決: **Resolved**。

### IPR-P1-02 — 後段capabilityの発行時期とPayment Receipt/commit順序が曖昧だった

- 参照: Plan 1.0 §5 WP-04、WP-07、WP-08、§7 route matrix。Requirements `WF-007`、`WF-010`、`PLAN-014`、`GATE-010`、`AP2-018`〜`AP2-024`、`ACC-010`、`ACC-034`。Design §6.3、§7.5。
- 問題: WP-04の文言はplan approval時にTS/CP/sign/submit/settle/commit/refundまでpre-issueすると読め、least privilegeと二番目のapproval gateが不明確だった。またWP-07はMPP Payment ReceiptがMerchant commit後に発行されると読め、settle success後commit failureのimmutable payment evidence/refund前提を壊し得た。
- 修正: Plan 1.1 §7.4にcapabilityごとのearliest issuance/consume/invalidation表を追加し、just-in-time issuanceとpre-issue禁止を固定した。WP-07はsettlement result + selected-profile receipt + MPP Payment Receiptを同attemptの証跡としてcommitした後だけMerchant commitを許可し、commit failure時はCheckout Error Receipt + `refund_required`とした。
- 解決: **Resolved**。

### IPR-P1-03 — 並行streamのfile owner/freezeとestimateが現実の依存に合っていなかった

- 参照: Plan 1.0 §4 file manifest、§5 WP-01〜09、§10 parallelization/estimate。
- 問題: `web_app.py`/`nginx.conf`、repository/schema、controller/API、Merchant service、deploy scriptsなどのowner/handoffがなく、「parallel可」が同一file/contractへの競合を許容していた。またgroup estimateの総和は69〜103 person-daysであり、WP-03→04→05→06→07→08→09はほぼ直列なのに、「一人61〜91、3 streamで35〜50 days」は算術/依存と不整合だった。
- 修正: Plan 1.1 §10.1にdependency/contracts、identity edge、persistence、workflow、Merchant/Store、AP2/profile、deploy/release、testsのexclusive owner/freeze/handoff/collision ruleを追加した。G0 compatibilityとG1c fixture freezeを分け、並行化可能な基盤後半を明示した。見積は総工数69〜103 person-days、3 streamの最短elapsed 50〜75 engineering daysへ修正し、G0/G1後のre-estimateを必須にした。
- 解決: **Resolved**。

### IPR-P1-04 — G0 baselineとG8 final regressionが対称でなく、collection/skipでfalse greenにできた

- 参照: Plan 1.0 §2 tests baseline、§5 WP-00/WP-09、§8 ACC-027、§9 verification commands、§12.1。Requirements `COMP-001`、`TEST-012`、`TEST-015`、`ACC-027`、`ACC-032`。
- 問題: G0は「repository test」を実行する一方、G8はmarker選択と新free-flow testが中心で、Trusted Agent Store/subproject testや旧suite移植前後のcollection減少を見逃せた。pytestの0 collection、skip、xfailをrequired PASSと別管理するgateもなかった。
- 修正: Plan 1.1 §4.1/§4.2、WP-00、WP-09、§9、§12に`tests/regression/suite_manifest.json`とrunnerを追加し、G0/G8で同一suite/command/environmentを実行する。baseline/final collected count、allowlist外skip/xfail、subproject差分をrelease blockerにした。
- 解決: **Resolved**。

### IPR-P1-05 — ACC-023/024の実browser gateとACC結果の機械的completion gateがなかった

- 参照: Plan 1.0 §4.2 test files、§5 WP-09、§8 ACC-001〜035 mapping、§9 commands、§12.1。Requirements `TEST-010`、`TEST-011`、`ACC-023`、`ACC-024`、§17.1。
- 問題: ACC-023は「container suiteでrecord」としか指定されず、browser test file/driver/command/public-ingress constraint/artifact/non-skip gateがなかった。ACC mappingもMarkdown行の存在だけでPASSにでき、欠落/重複/別RC artifact/ACC-030の誤PASSを機械的に拒否できなかった。
- 修正: Plan 1.1 §4.2、WP-09、§8、§9、§12に実Chromium/Playwrightのpublic-ingress E2E、root/message/two approvals/refresh/reconnect操作、trace/screenshot/video、CLI parity、skip/0-collection failureを追加した。`acc-results.json`はACC-001〜035をexact一行ずつ持ち、initial releaseは001〜029/031〜035=`PASS`、030だけ=`NOT_RUN`、全artifactが同じRC image/lock digestであることをvalidatorが強制する。
- 解決: **Resolved**。

### IPR-P1-06 — 三DB cutover／restore途中のcrashとpost-write image compatibilityが閉じていなかった

- 参照: Plan 1.0 §6.2〜6.4、WP-02、WP-09、§12.2。Requirements `MIG-001`〜`MIG-006`、`RES-004`〜`RES-006`、`ACC-020`、`ACC-026`、`ACC-032`。Design §12.1。
- 問題: 各DBのtransactionと個別atomic renameは三file全体をatomicにしない。apply/restore中のhost killでv1/v2またはsource/restored fileが混在し得たが、durable phase journal/resumeがなかった。「v2-compatible previous image」もschema range/checksumの事前証明がなく、paid disabledからenabledへの変更は同一RCとは限らなかった。
- 修正: Plan 1.1 WP-02、§6、WP-09、§12にexplicit durable migration directory、append-only／fsync phase journal、DBごとのapply／verify／restore phase、unfinished／mixed startupのfail-closed、同じmigration IDでのresume、restore rename途中のkill testを追加した。cutoverはwrite不可の`PRE_CUTOVER`から同じimage／manifestのimmutable enabled configへrestartし、最初のv2 writeはDB rowを正本としてpost-writeを判定する。post-write rollback imageはmanifest済みschema range／checksum testを必須にした。
- 解決: **Resolved**。

## 4. 完了条件と追跡可能性のレビュー

修正後Plan 1.1は次を満たす。

- ACC-001〜ACC-035の全35 IDが§8にexact一行ずつあり、simulation release mandatory setはRequirements §17.1と一致する。ACC-030のみconditional `NOT_RUN`である。
- G0→G1a/G1b/G1c→G2〜G8の依存は、identity前のapproval signing、schema freeze前のrepository consumer、fixture freeze前のAP2/A2A domain実装を許さない。
- 二つのexact approval、`payment_authorizing`、original Merchant Task、AP2 exact evidence、selected simulation profile payload/history、Payment Receipt→Merchant commit→Checkout Receipt/refundの順序がwork package/gate/testで相関する。
- unit/AP2 contract/x402 simulation contract/integration/security/restart/migration/concurrency/container/browser/full regressionは別gateで、simulation PASSをofficial PASSに集計しない。
- clean volumeとmigrated volume、process restartとcontainer recreation、route isolation、ephemeral mount refusal、実browser/CLI parityがG8のhard gateである。
- pre-cutover restoreとpost-write rollbackを分け、unknown settlement/refundをDB/image rollbackで推測せずsame external ID reconciliationへ残す。

## 5. 適用範囲のレビュー

Plan 1.1はinitial runtimeを次に限定し、reviewed requirements/designのapproved scopeを超えていない。

- 利用者向け root は `payment_user_agent` 一つで、内部 `secure_mediation_agent` workflow が認可の正本を持つ。`payment_demo_user_agent` と legacy payment route は integrated image／public ingress から除外する。
- AP2は`AP2 v0.2 Human Present demo`のclosed Mandate/official schema/SDK contractを実装するが、real identity/KMS/適合認証を主張しない。
- payment profileはproject-local `x402-wire-simulation/1`、`exact-simulated`、`demo:local`、synthetic proofだけ。canonical x402 URI、official `exact`、wallet/facilitator/on-chain transactionは実装/実行/宣言しない。
- deploymentはexplicit durable POSIX volumeの、single-host/single-containerに限る。multi-instance/shared filesystem/current Cloud Run paidは不可。
- Merchant/payeeはpaid external Merchantで、platform-payee/guarantee/deferred payout/manual payoutはnew flowへ入れない。

## 6. 残存リスクと実装時のブロッカー

1. G0のpinned AP2 Git dependency、generated Receipt variants、A2A 0.3.19 custom handler、ADK 1.19.0 identity wrapperは未実行である。いずれかが失敗したら計画にdomain workaroundを追加せずreviewed designへ戻す。
2. Playwright/ChromiumのselectorはADK Web 1.19.0 UIにversion-coupledである。browser test image/browser versionをdigest固定し、ADK upgradeで再spikeする。
3. regression manifestのlive external API/key依存testは意図的skipになり得る。allowlistはID/理由/owner/expiryを持ち、新しいskipを自動承認しない。
4. sanitized migration fixtureのみで実data特有のsize/path/permission/corruptionは完全に再現できない。cutover前にproduction-like copy上で同image/manifestのrehearsalが必要である。
5. SQLite WAL/outbox/leaseはsingle-host制約に依存する。multi-instance、network filesystem、Cloud RunはDB/queue architecture reviewとACC-020/032再実行なしに対象へ追加できない。
6. official x402のnetwork/asset/wallet/facilitator/TLS/amount mappingとACC-030は未定/未実装である。canonical URI/compatible/conformant/on-chain claimは引き続きblockする。

## 7. 判定

**APPROVE**。

Plan 1.1-plan-reviewedは、AP2 v0.2 Human Present demoとproject-local x402 v0.1 wire-shape simulation fixtureを、画面上の`payment_user_agent`と内部`secure_mediation_agent`による一つのdurable workflowへ統合する実装に着手できる。未解決のP0/P1計画指摘はない。ただしG0/G1/G2以降の完了条件は未実行であり、この判定は実装完了またはrelease readyを意味しない。

次は引き続き **REJECT / NOT READY** である。

- official x402 profile/canonical URI/wallet/facilitator/on-chain settlement/conformance claim。
- current ephemeral Cloud Runのintegrated paid workflow。
- multi-host/multi-instance SQLite deployment。
- integrated image内のlegacy payment-only root/API再有効化。
