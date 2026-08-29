# 仲介エージェント決済統合：Release-1実装計画レビュー

- review date: 2026-08-16
- reviewed plan: `MEDIATOR_PAYMENT_INTEGRATION_IMPLEMENTATION_PLAN.md`
- reviewed inputs: 最新 `MEDIATOR_PAYMENT_INTEGRATION_REQUIREMENTS.md`、`mediator-payment-integration-design/README.md` と `01`〜`12`、`MEDIATOR_PAYMENT_INTEGRATION_DESIGN_REVIEW_SECURITY.md` §10
- review scope: 正常paid、正常free、基本refundの実行可能性。高度recovery、複雑競合、DNS rebinding、部分／複数refund、網羅的malicious matrixは評価対象の実装scopeへ戻さない

## 1. 判定

**総合判定: NO-GO as written / 4件の局所修正後GO。**

計画の中心構造は実行可能である。とくにSecurity再レビューの残存HIGH 3件を先頭のPhase Cへ置き、解消前のpayment／settlement／refund DTOとwire validatorのfreeze・実装を禁止した点は正しい。正常実装の範囲を広げる必要も、architectureの作り直しもない。

ただし、同一Taskへのpayment A2Aで従来callbackを確実に通す記述と、0% Cloud Run candidateを直接検証する経路が閉じていない。この2件はそれぞれpaid正常系とtraffic切替前検証を成立させないためHIGHとする。下記HIGHを計画本文へ反映するまではPhase 4のpayment submit実装とPhase 7のCloud Run更新は開始しない。Phase Cの契約修正は直ちに開始してよい。

| Severity | 件数 | 判定への影響 |
| --- | ---: | --- |
| BLOCKER | 0 | なし |
| HIGH | 2 | 修正前は計画全体NO-GO |
| MEDIUM | 2 | 該当phase開始前に修正 |
| LOW | 0 | なし |

## 2. 指摘

### HIGH-01 — same-Task payment A2Aのlegacy callback前後実行が実装seamまで閉じていない

計画Phase 3は初回A2Aについて `orchestration_agent.py` のlegacy callbackをbefore／afterでfail closedにする。一方、Phase 4の同一Task payment Messageは `merchant/client.py`、bridge、workerの変更として記載され、完了条件はfive stable gatesだけを数えている。従って、実装者が `PRE_PAYMENT_SUBMIT`／`POST_PAYMENT_RESULT` を実装しても、stable gateとは別層であるlegacy callbackをpayment A2Aの前後で省略できる余地がある。

これは `SEC-016` の「各A2A operationの前後」、03 §9の「before payment operation／after」、05 §7の「毎回必須」と一致しない。初回A2Aだけcallbackを通す実装では `PAID-HAPPY-01` をPASSにできない。

**具体的修正案:** Phase 3〜4に、初回Task startとsame-Task payment submitが共有する一つのtyped A2A operation executorを明記する。payment側の固定順を `legacy callback before -> PRE_PAYMENT_SUBMIT -> transport -> response永続化 -> legacy callback after -> POST_PAYMENT_RESULT` とする。before失敗ではtransport 0、after失敗ではsettlement／fulfillment／final 0とする。integration oracleはpaidで「初回A2A前後＋payment A2A前後」、freeで「初回A2A前後のみ」をoperation IDと実symbol eventでexact countし、refundをA2A operationとして実装する場合も同じexecutorを通す。

### HIGH-02 — `--no-traffic` revisionへ到達するcandidate URLの作成手順がない

Phase 7は `--no-traffic` で0% revisionを作成した後、「candidate URL」でreadiness、model probe、三browser scenario、deny matrixを実行する。しかし、計画にはそのrevision専用URLを生成・取得し、exact revisionへ結び付けるtraffic tagの作成がない。通常のservice URLは既存trafficへrouteされるため、このままでは旧revisionを検証してcandidateを100%へ切り替える偽PASSが起こり得る。

Cloud Runの公式手順も、trafficを受けないrevisionの事前testにはtagged revisionを用い、traffic tagが専用URLを公開するとしている: [Rollbacks, gradual rollouts, and traffic migration](https://cloud.google.com/run/docs/rollouts-rollbacks-traffic-migration)。

**具体的修正案:** 固定serviceへのupdateを、immutable digest、0% traffic、candidate固有traffic tagの三点セットにする。tag URLをAPI／CLI出力から取得し、tagがexact revision名とimage digestへ結合すること、およびdefault service trafficが旧revisionのままであることを検証してからbrowser／black-boxを実行する。candidate Originは既存のexact Origin／CSRF policyを弱めず明示許可し、shiftまたはrollback後はtagを除去する。container testに「tagなしではvalidate不可」「tag URLのrevision不一致で停止」「validation中のdefault traffic変更0」を追加する。

### MEDIUM-01 — contract closureとtyped legacy seamの変更対象が不足している

Phase Cの `RER-SEC-002` はMerchant wireを `merchant-payment-guarantee-submission/1` へ統一するとしているが、変更対象一覧から `05_SECURITY_TRUST_BOUNDARIES.md` が漏れている。同文書§9には現在も旧名 `merchant-payment-submission/v1` が残り、Security再レビュー§10.3も05を対象に含めている。このままではHIGH 0の再レビュー条件を満たせない可能性がある。

また、計画の実装pathは正しい `anomaly_detection_agent.py`／`final_anomaly_detection_agent.py` を使う一方、01のproduction composition seamには旧path `anomaly_detector.py`／`final_anomaly_detector.py` が残る。これはSecurity再レビュー§10.4の既知の非blocking不整合である。

**具体的修正案:** Phase Cの変更対象へ05を加え、旧wire名とallowlist説明を06の最小guarantee submissionへ一致させる。Phase 0または3の変更対象へ01を加えて二つの実pathを訂正し、production composition tableに列挙されたmatcher／planner／orchestrator callback／anomaly／final symbolのimport可能性と、typed input／outputをspike testで固定する。これは文書・契約整合だけであり、新機能を追加しない。

### MEDIUM-02 — futureの網羅matrixとRelease-1必須の基本fail-closed負例の境界がtest gate上で曖昧

計画は `TEST-009`、`TEST-018`、`AC-008`を正しくfutureへ分離している。しかし最新REQUIREMENTSは、網羅matrixをfutureにしても `SEC-006` の基本接続先固定、`SEC-008` の外部入力fail closed、`SEC-011` の結果不明時の成功禁止をRelease-1必須に残している。最小test gateにはprofile／capability不一致はあるが、どの代表負例でこの三要件を閉じるかが一意でない。三正常oracleだけを追加して既存testを「実行する」記載では、新しいtyped seamを実際に検査するtestが欠落し得る。

**具体的修正案:** 網羅的matrixへ広げず、Release-1専用の最小negative setを明記する。

1. 自由文だけの支払指示はpayment-requiredにしない。
2. unknown／破損profile、必須metadata欠落はpayment／guarantee 0で停止する。
3. Task／context／order／quoteまたはsigned capability scope不一致はMerchant副作用0で拒否する。
4. Card／RPC endpoint差替え、redirectはA2A送信0で拒否する。
5. legacy callback例外／timeoutは次の外部副作用0、transport結果不明は新Task／新paymentを作らず `ReviewRequired` とする。

これらを `TEST-001`／`004`／`005`／`006`／`008` のRelease-1 evidenceへ結び、全malicious matrix、DNS rebinding全case、複雑retry／reconciliationは引き続き `TEST-009`／`017`／`018` と対応ACのfuture statusに留める。

## 3. 指定観点の確認結果

| 観点 | 判定 | コメント |
| --- | --- | --- |
| contract closure HIGH 3が先頭 | PASS | Phase Cが依存なしの先頭で、PASS前のpayment／settlement／refund DTO・wire実装を禁止している。05の対象漏れだけMEDIUM-01で補う |
| 旧`secure_mediator` typed seam | PASS WITH FIX | 実装pathと実symbol eventは妥当。01の二つの旧file pathをMEDIUM-01で同期する |
| Trusted Surface／AP2 actor | PASS | non-agentic Trusted Surfaceだけが同意／user signatureを発行し、Shopping Agent／orchestratorは認可artifactを進行するだけ |
| orchestrator approved-payment tool | PASS | `execute_approved_payment`はIDとexpected versionだけを受け、金額、payee、Task、approval、endpointをserver-side snapshotから解決する |
| demo guarantee／same Task | PASS | guaranteeをsettlementと誤称せず、同じtask/contextの後続Messageを一回だけ送る。Phase Cでsettlement点を閉じる |
| callback hook／final | PARTIAL | finalはsettlement／fulfillment後へ固定されている。payment A2Aのcallback前後実行をHIGH-01で明示する必要がある |
| identity／public boundary | PASS | Firebase由来owner 4-tuple、session-only turn route、body selector禁止、nginx allowlist／loopback／内部route denyが一貫している |
| 必要最小test | PARTIAL | 三正常oracleと基本refundは適切。Release-1の代表fail-closed負例だけMEDIUM-02で補う |
| container／browser | PASS | local containerと同一candidateのpaid／free／refund browser、secret非露出を要求している |
| 固定Cloud Run | PARTIAL | project／region／service固定、immutable digest、0% updateは妥当。0% revision専用URLをHIGH-02で閉じる |
| rollback | PASS | 旧revision名／digestをpreflight固定し、schemaはforward-only、結果不明は成功扱いしない |
| edge future分離 | PASS | 139規範IDを機械集計でき、計画のfuture 13件は最新REQUIREMENTSの集合と完全一致する。基本安全境界はRelease-1に残る |

## 4. GO条件

次を満たした時点で本レビューの判定をGOへ更新できる。

1. 計画本文へHIGH-01、HIGH-02の実装seam、順序、exact oracleを反映する。
2. Phase C対象へ05、typed seam整合対象へ01を追加する。
3. MEDIUM-02の代表負例をRelease-1 test manifestへ追加し、future 13件を拡大しない。
4. Phase Cで `RER-SEC-001`〜`003` を実際に解消し、Security再レビューがHIGH 0、139 = Release-1 126 + future 13、重複／欠落0を確認する。

修正後のcritical pathは既存どおり `C -> 0 -> 1 -> 2 -> 3 -> FREE-HAPPY-01 / paid wait -> 4 -> PAID-HAPPY-01 -> 5 -> REFUND-01 -> 6 -> 7` でよい。新たな業務scenarioや高度edge caseを追加する必要はない。
