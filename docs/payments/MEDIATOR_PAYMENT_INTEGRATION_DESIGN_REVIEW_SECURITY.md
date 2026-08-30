# 仲介エージェント決済統合：Security／AP2／x402／A2A／公開境界 独立設計レビュー

> [!WARNING]
> この文書は作成時点の引継ぎ／レビューsnapshotであり、現在仕様の正本ではない。現行責務は[アーキテクチャ](ARCHITECTURE.md#actorと責務の正本)と[Payment Bridge設計](mediator-payment-integration-design/04_PAYMENT_BRIDGE_AP2_X402.md)を参照する。本文は履歴証跡として変更しない。

## 1. レビュー概要

- レビュー日: 2026-08-16
- レビュー方式: read-only の設計・現行実装・設定・テスト突合
- 判定対象: `MEDIATOR_PAYMENT_INTEGRATION_HANDOFF.md`、`MEDIATOR_PAYMENT_INTEGRATION_REQUIREMENTS.md`、`MEDIATOR_PAYMENT_INTEGRATION_DESIGN_STRUCTURE.md`、`mediator-payment-integration-design/01`〜`13`、関連する現行コード／設定／tests
- 重点範囲: Security、AP2 Human Present、A2A x402 extension、A2A Task継続、公開HTTP／WebSocket／identity境界、Cloud Run release境界
- 変更制約: 本レビューでは設計、コード、設定、testsを変更していない。本ファイルのみを新規作成した。

### 1.1 今回の実装scope override

本レビューの確定時点で、今回の成果物は**概念決済デモの正常系を先に成立させる**範囲へ明示的に絞られた。従って、正常系に必須な次の境界は重大指摘として維持する。

- 完全一致承認のauthoritative routingを公開routeから迂回できないこと
- 構築可能なcorrelation schemaでsubject／tenant／session／plan／step／Agent／Task／order／quoteを結合すること
- payment profile／extension header／wire metadata／signed capabilityを副作用前に検証すること
- 正常系で同じremote Taskと同じstepを継続し、基本idempotencyを守ること
- secret／private subject bindingをuntrusted Merchant、UI、LLM、logへ過剰開示しないこと
- 公開経路をexact allowlistへ限定し、identityを検証済みsubjectへ結合すること

一方、DNS再束縛の高度対策、初回Task応答喪失からの完全自動回復、複雑なretry／並行競合、悪意入力matrixの網羅は今回の実装blocking条件にしない。これらは既知課題／将来対応へ降格する。今回必須の例外フローは返金だけとし、それ以外の高度なreconciliation／自動復旧は適合を主張しない `NOT RUN` または既知制限として扱う。このscope overrideと既存REQUIREMENTSの広い記述が食い違う箇所は、release説明とconformance reportで今回非対象と明示し、PASSへ数えない。

## 2. 結論

**総合判定: FAIL（設計承認不可）／現行実装: NOT CONFORMANT**

設計には、二段階完全一致承認、支払要求の三条件認識、signed capability、profileのsilent fallback禁止、五つのanomaly gate、`BLOCK > REVIEW > PASS`、同一Task／同一step継続、outbox／reconciliation、simulationの常時 `NOT CONFORMANT` 表示、ephemeral Cloud Run表示など、採用できる強い骨格がある。

しかし、次の2件はその骨格を実装可能なcontractへ落とせない、または公開境界から迂回できるため、設計確定前に解消が必須である。

1. 公開されたworkflow ID指定message routeが、session単位の完全一致承認routingを迂回できる。
2. 支払提出前のimmutable correlation envelopeへ将来のReceipt／resultを含め、さらに相互参照させるevidence graphが時間的・hash的に循環している。

また、subject binding、identity assertion発行、restricted evidenceのMerchant送信、payment profile／wire contract、現行公開proxy、target正常系の実装、固定Cloud Run service更新にHIGHが残る。したがって、設計承認、実装完了、release candidate、Cloud Run更新のいずれも現時点では合格を主張できない。

### 重大度集計

| Severity | 件数 | 扱い |
| --- | ---: | --- |
| BLOCKER | 2 | 設計確定前に必須修正 |
| HIGH | 7 | security／release合格前に必須修正 |
| MEDIUM | 4 | 今回非blockingの既知課題を含む。適合を主張しない |
| LOW | 0 | なし |
| 重大指摘（BLOCKER＋HIGH） | **9** | 未解消ならFAIL |

## 3. 判定基準

| 判定 | 意味 |
| --- | --- |
| PASS | 要件、設計、実装、試験oracleが一貫し、禁止副作用0件まで検証可能 |
| PARTIAL | 設計意図は妥当だが、wire schema、数値policy、owner、試験oracle等が未確定 |
| NOT RUN | 実行証跡がない、または本read-onlyレビューでは実行していない |
| NOT CONFORMANT | 要件／設計との不一致、迂回可能性、または既存証跡の流用禁止に該当 |

Severityは次で付与した。

- `BLOCKER`: 設計のままでは安全な一意実装を構成できない、または承認／権限境界を迂回できる。
- `HIGH`: 主体分離、支払副作用、秘密・restricted data、外向き通信、障害時exactly-once、公開境界、release操作に直接影響する。
- `MEDIUM`: 直ちに副作用を許すとは限らないが、異なる実装解釈や不正確な適合主張を生む。

## 4. 指摘一覧

### SEC-DR-001 — 公開workflow message routeがauthoritative routingを迂回する

- Severity: **BLOCKER**
- 合否: **FAIL**
- 対象:
  - `mediator-payment-integration-design/03_MEDIATION_FLOW.md` — 「入口とauthoritative routing」「完全一致承認とrouting表」
  - `mediator-payment-integration-design/06_API_A2A_CONTRACTS.md` — `mediation.turn.submit/1`
  - `mediator-payment-integration-design/09_DEPLOYMENT_PUBLIC_BOUNDARY.md` — `TBL-ROUTE-01` の `/mediation-api/v1/workflows/{workflow_id}/messages`
  - `mediator-payment-integration-design/12_DECISIONS_OPEN_QUESTIONS.md` — `OQ-010`
  - 現行 `secure_mediation_agent/agent.py:63-80`、`tests/workflow/test_identity_and_api.py:42-71`
- 根拠:
  - 03は、同一 `subject / tenant / ADK session / mediation session` のpending候補を列挙し、payment approval pendingを優先し、候補が複数なら `APPROVAL_TARGET_AMBIGUOUS`、一件だけならその対象へrouteすると規定する。また、raw workflow IDを承認selectorにしない。
  - 06の論理公開operation `mediation.turn.submit/1` はworkflow IDを入力に持たず、このroutingをcontrollerの責任にしている。
  - 09は同時に、認証browserが任意のopaque `workflow_id` をpathへ置けるmessage mutation routeを公開している。owner checkだけでは「どのpending承認へ入力を届けるか」の優先順位・一意性を保証しない。既知またはUIから得た同一ownerの別workflow IDを指定すれば、authoritative routingを迂回できる。
  - 現行adapterもsession state内の単一workflow IDへ直接messageを送り、候補集合を再評価しない。
- 修正案:
  1. 公開mutationをsession-levelの一つの `mediation.turn.submit/1` に統一し、workflow ID指定message routeは削除またはinternal-onlyにする。
  2. 各turnでserver側identityから完全owner tupleを組み立て、同一transaction／同一snapshot上で候補件数と優先順位を判定する。
  3. selection tokenは候補集合digest、owner tuple、approval kind、target ID、expiry、nonceへ署名し、一回だけconsumeする。raw workflow IDは認可根拠にしない。
  4. payment pending 2件、plan pending 2件、payment＋plan混在、期限切れtoken、別session／subjectのID指定をblack-box testに追加し、誤対象への書込み・Merchant call・支払副作用が0件であることをassertする。並行POSTの網羅的race matrixは将来課題としてよい。

### SEC-DR-002 — correlation envelopeが未来の結果を含む循環参照になっている

- Severity: **BLOCKER**
- 合否: **FAIL**
- 対象:
  - `mediator-payment-integration-design/04_PAYMENT_BRIDGE_AP2_X402.md` — `FIG-PAY-01`、7章 `仲介correlationのevidence binding`、`TBL-PAY-02`
  - `mediator-payment-integration-design/06_API_A2A_CONTRACTS.md` — 10章 `Payment-submitted contract`、12章 `Payment result Task`
  - `mediator-payment-integration-design/12_DECISIONS_OPEN_QUESTIONS.md` — `OQ-008`
  - `mediator-payment-integration-design/10_TEST_STRATEGY.md` — `TEST-002`
- 根拠:
  - 04は一つの署名済みimmutable envelopeへ、Credential、proof、AP2 Receipt、x402 resultを含む全descriptorを入れ、AP2 objectと「相互参照」すると規定する。evidence graphでもReceipt／x402 resultからenvelopeへ辺がある。
  - 06はその同じenvelopeとdigestを、Receipt／final resultがまだ存在しない `payment-submitted` messageへ必須で載せる。一方、最終Taskは同じ `correlationEnvelopeDigest` と後から生成されるreceipt digestを返す。
  - したがって、提出前envelopeは将来のReceipt／result digestを計算できず、Receiptがenvelope digestを持ち、envelope rootがそのReceipt digestを持つならhash cycleになる。placeholder、後書換え、同じschema名での再署名はいずれも「immutable」「同じdigest」「offline完全検証」と両立しない。
  - 06は「serialized envelope fieldのowner」とされるが、実際には文字列placeholderしかなく、完全なJSON schema、canonicalization、署名対象、phase/version、unknown field policyがない。
- 修正案:
  1. 提出前の `mediation-authorization-envelope/v1` と、結果後の `mediation-completion-manifest/v1` を分離する。
  2. authorization envelopeはsubject binding、plan／step／Agent／Task／order／quote、二承認、Checkout、Mandates、Credential、profile／capabilityまでを確定し、そのdigestを支払payloadとReceiptへ一方向参照させる。
  3. completion manifestはauthorization digest、全Receipt／result digest、attempt順序、final gate decisionを含めて後から署名する。過去artifactにcompletion digestを逆参照させない。
  4. 代替としてappend-only Merkle logを採る場合も、各entryが直前rootだけを参照する一方向chainにする。
  5. 両schemaの必須field、canonical JSON、署名key role、`typ`／`aud`、expiry、size、redaction、version migrationを06に完全定義し、作成可能性、各field mutation、順序入替え、欠落、別phase混入をunit testする。

### SEC-DR-003 — plan wire／digestのownerからFirebase subjectが欠落する

- Severity: **HIGH**
- 合否: **FAIL**
- 対象:
  - `MEDIATOR_PAYMENT_INTEGRATION_REQUIREMENTS.md` — `SEC-001`、`SEC-002`、`DATA-001`、`TEST-002`
  - `mediator-payment-integration-design/02_DOMAIN_DATA_STATE.md` — ID命名・owner scope、plan snapshot
  - `mediator-payment-integration-design/05_SECURITY_TRUST_BOUNDARIES.md` — 5章 `Identity binding`
  - `mediator-payment-integration-design/06_API_A2A_CONTRACTS.md` — `planner.plan.result/1` の `ownerRef`
  - 現行 `secure_mediation_agent/workflow/api.py:175-191`、`secure_mediation_agent/workflow/controller.py:115-154,1867-1874`
- 根拠:
  - 要件と02／05はFirebase `subject` をtenant、ADK session、mediation session、plan、continuation、payment workflow、evidence、承認、再開へ一貫して結び付ける。
  - 06のplan snapshot例の `ownerRef` は `tenantId / adkSessionId / mediationSessionId` だけで、subjectまたはsubject-binding digestがない。plan digestがこのwireから作られるなら、第一承認がsubjectへ暗号学的に結合されない。
  - 現行APIは署名済みassertionを検証して `VerifiedIdentity.subject` を得た直後、domain `Identity` へ変換するときにsubjectを捨てる。repository／owner checkもtenant＋customer中心である。demoでは異なるFirebase subjectが同じ固定tenant／customerへ写像されるため、subject分離要件を満たさない。
- 修正案:
  1. domain owner keyを `subject + tenant + adk_session_id + mediation_session_id` とし、全aggregate、unique index、query、CAS、idempotency scopeへ同じ順序で入れる。
  2. plan snapshotへcontroller由来のsubject、または用途・salt・versionを固定した不可逆 `subjectBindingDigest` を含め、plan digest／第一承認署名の対象にする。plannerの自己申告subjectは信用しない。
  3. 二つの有効なFirebase subjectが同じdemo tenant／customerを共有するcaseで、参照、承認、active lookup、idempotency、Task再開のcross-subject拒否を試験する。

### SEC-DR-004 — internal identity assertionを任意subjectからmintでき、header contractも二重化している

- Severity: **HIGH**
- 合否: **FAIL**
- 対象:
  - `MEDIATOR_PAYMENT_INTEGRATION_REQUIREMENTS.md` — `SEC-003`、`HTTP-002`、`HTTP-005`
  - `mediator-payment-integration-design/05_SECURITY_TRUST_BOUNDARIES.md` — identity header strip／signed assertion
  - `mediator-payment-integration-design/06_API_A2A_CONTRACTS.md` — `mediation.turn.submit/1`、Payment Bridge transport
  - `mediator-payment-integration-design/09_DEPLOYMENT_PUBLIC_BOUNDARY.md` — 4章 identity header、5章ADK user path
  - 現行 `secure_mediation_agent/agent.py:33-61`、`deploy/auth/verify.py:178-190`、`secure_mediation_agent/workflow/api.py:175-191`、`deploy/nginx.conf:129-164`
- 根拠:
  - target設計は外部のidentity関連headerを全除去し、proxyがFirebase sessionから作る短命な `X-Internal-Identity` だけを内部で使う。
  - 現行は `X-Verified-Identity` を使い、adapterはADKの `session.user_id` をbodyへ入れてloopback brokerへ送り、brokerはその文字列を検証済みFirebase sessionへ照合せず署名する。loopback到達可能な同一container process、またはclient入力由来のADK user IDを制御できる経路が、任意subjectのassertionを取得できる。
  - targetの `{user}` pathを認証主体から導出する規定にも、path segmentとassertion subjectの照合wire／失敗codeがない。header aliasの列挙も現行名を明示しておらず、移行時に古いheaderを取りこぼす余地がある。
- 修正案:
  1. assertionはproxy/auth serviceが検証済みFirebase cookieからだけ発行し、adapterへそのassertionをtrusted channelで渡す。任意subjectを受けるmint APIを廃止する。
  2. canonical internal headerを一つにし、edgeで `X-Internal-Identity`、`X-Verified-Identity`、`X-Authenticated-*`、`X-Subject`、`X-Tenant-*` 等の外部入力を明示的に空へしてから内部値を設定する。
  3. `{user}` はserver側でsubjectから導出し、client指定値とconstant-time一致しなければupstreamへ流さない。
  4. forged header、duplicate header、underscore alias、mixed case、別user path、loopback mint、同一tenant内別subjectのnegative black-box testを追加する。

### SEC-DR-005 — restricted correlation evidenceをuntrusted Merchantへ過剰開示する

- Severity: **HIGH**
- 合否: **FAIL**
- 対象:
  - `MEDIATOR_PAYMENT_INTEGRATION_REQUIREMENTS.md` — `SEC-008`、`SEC-010`、`DATA-006`
  - `mediator-payment-integration-design/04_PAYMENT_BRIDGE_AP2_X402.md` — `TBL-PAY-02` のprivate subject binding
  - `mediator-payment-integration-design/05_SECURITY_TRUST_BOUNDARIES.md` — trust boundary、data classification／output projection
  - `mediator-payment-integration-design/06_API_A2A_CONTRACTS.md` — `Payment-submitted contract`、特にraw restricted evidenceのMerchant送信
- 根拠:
  - 05はMerchantをexternal／untrustedと分類し、full subject、session IDs、endpoint、signed envelopeをrestrictedとしてowner-checked internal API／auditへ限定する。
  - 04の一つのenvelopeはsubject、tenant、ADK／mediation sessionをprivate bindingとして含む。
  - 06はそのraw envelope、Checkout JWT、Mandates、CredentialをMerchantへのA2A partに送る。TLSは経路暗号化であって、untrusted recipientへの最小開示や保存・log・再利用を防がない。05の分類規則と06のwireが矛盾する。
- 修正案:
  1. local restricted identity envelopeとMerchant-scoped authorization packageを分ける。
  2. Merchant wireにはopaque correlation IDまたは目的別HMAC binding、Merchantが検証に必要なTask／order／quote／terms／capability／Mandateだけを載せ、Firebase subject、tenant内部ID、ADK／mediation session、内部endpointを送らない。
  3. 完全offline bundleはaccess-controlled evidence storeからowner／auditorへだけ提供する。
  4. Merchant request capture、structured log、A2A history、UI DOM、browser network responseへ禁止fieldがないことをfield-level testする。

### SEC-DR-006 — 外向き通信の高度なSSRF／retry policyが未確定

- Severity: **MEDIUM（既知課題／今回非blocking）**
- 合否: **PARTIAL**
- 対象:
  - `MEDIATOR_PAYMENT_INTEGRATION_REQUIREMENTS.md` — `NFR-004`、`SEC-006`、`TEST-005`、`TEST-009`
  - `mediator-payment-integration-design/05_SECURITY_TRUST_BOUNDARIES.md` — 9章 external communication policy
  - `mediator-payment-integration-design/06_API_A2A_CONTRACTS.md` — 共通wire上限、Agent Card／A2A request
  - `mediator-payment-integration-design/08_PERSISTENCE_RECOVERY.md` — retry／outbox
  - `mediator-payment-integration-design/10_TEST_STRATEGY.md` — `TEST-005`、`TEST-009`
- 根拠:
  - scheme／host／network、redirect 0、response size、Content-Typeを列挙している点は妥当である。
  - ただしoperation別のconnect/read/total timeout秒、retry最大回数、DNS再解決、全A/AAAA検査、接続先peer IP再照合、圧縮後／展開後上限等は未確定である。
  - 今回の正常系デモでは、exact destination allowlist、HTTPSまたは明示loopback、redirect 0、有限timeout、有限response size、Content-Type検査までをblocking minimumとする。DNS rebinding、proxy環境変数、gzip bomb、複雑retry等の強化は適合を主張しない将来課題とする。
- 修正案:
  1. 今回はCard fetch、A2A start／submit／Task getのdestination、redirect 0、timeout、response size、Content-Typeを最小表で固定する。
  2. demo loopback例外はexact host＋port＋environmentで限定し、その他のprivate／link-local／metadata destinationを拒否する。
  3. DNS rebinding、peer IP再照合、decompression、複雑retryは既知課題へ記録し、将来production化前に脅威試験を追加する。

### SEC-DR-007 — 初回Task応答喪失時の完全自動回復は未定義

- Severity: **MEDIUM（既知課題／今回非blocking）**
- 合否: **PARTIAL**
- 対象:
  - `MEDIATOR_PAYMENT_INTEGRATION_REQUIREMENTS.md` — `FR-009`、`FR-013`、`SEC-011`、`TEST-009`
  - `mediator-payment-integration-design/06_API_A2A_CONTRACTS.md` — `FIG-A2A-01`、初回 `message/send`、同じTaskの照会
  - `mediator-payment-integration-design/08_PERSISTENCE_RECOVERY.md` — outbox、checkpoint、result unknown／reconciliation
- 根拠:
  - 初回 `message/send` はTask IDを持たず、MerchantがTask IDを発行する。MerchantがTaskを作成した後、responseがorchestratorへ届く前にtimeout／crashすると、local側には照会に必要なTask IDがない。
  - 08はremote TaskStore照会または同じidempotency keyの再送を想定するが、Agent Card／profile negotiationはMerchantが `Idempotency-Key` の同一digest再送とoperation-ID lookupを保証するversioned capabilityを宣言するcontractを持たない。標準の `tasks/get` は既知Task IDなしでは使えない。
  - 未交渉の再送は二つのTaskを作り得て、再送しなければ同じTaskを回復できない。今回の正常系では、初回responseで得たTask IDを保存し、その後のsubmit／getが同じTaskを使うことと、既知Taskに対する基本idempotencyをblocking minimumとする。初回response喪失の完全自動回復は将来課題とする。
- 修正案:
  1. 今回は受信したTask ID／context IDを支払要求表示前に保存し、同じsubmit idempotency keyとdigestによる同一Task継続を正常系試験で証明する。
  2. 初回responseが不明な場合は自動再送で支払へ進まず、安全停止して既知制限を表示する。完全回復を今回のPASS条件にはしない。
  3. 将来はCard／Registryにversioned task-start idempotencyとstable client operation ID lookupを定義し、fault injectionを追加する。

### SEC-DR-008 — target anomaly gates／動的仲介統合の実装証跡がなく、現行経路を流用できない

- Severity: **HIGH**
- 合否: **NOT CONFORMANT**
- 対象:
  - `mediator-payment-integration-design/03_MEDIATION_FLOW.md` — gate schedule、動的Agent選定
  - `mediator-payment-integration-design/05_SECURITY_TRUST_BOUNDARIES.md` — `BLOCK > REVIEW > PASS`
  - `mediator-payment-integration-design/10_TEST_STRATEGY.md` — `TEST-006`、`TEST-013`、`TEST-015`
  - `mediator-payment-integration-design/11_TRACEABILITY_RELEASE.md` — 127件のcoverage manifest
  - 現行 `payment_user_agent/agent.py:3-12`、`secure_mediation_agent/agent.py:49-80`、`secure_mediation_agent/workflow/models.py:20-43`
- 根拠:
  - 現行公開経路は `payment_user_agent -> PaymentWorkflowAdapter -> 固定workflow` で、matcher／planner／orchestratorが作る従来仲介planを経由しない。
  - target stable gate ID `PRE_A2A_START`、`POST_A2A_RESPONSE`、`POST_PAYMENT_REQUIREMENT`、`PRE_PAYMENT_SUBMIT`、`POST_PAYMENT_RESULT`、`selectionToken`、`APPROVAL_TARGET_AMBIGUOUS` は現行コード／testsに存在しない。現行state enumにもtargetの `Blocked`／`ReviewRequired` と同じ意味を持つ共通gate stateがない。
  - 11の127 requirement recordはすべて `implementation_refs: []` で、これはtarget-design段階としては許されても、実装／release合格の証拠ではない。要件は既存の直接workflow証跡の流用を明示的に禁止する。
- 修正案:
  1. target routing、全gate、BLOCK／REVIEW、continuation、dynamic selected snapshotを実装した後、各requirementにexact code／test／evidence refsを記録する。
  2. gateごとにorder、input/output digest、call count=1、decision、actor、時刻と、gate前禁止副作用=0を機械assertする。
  3. 既存direct workflowのPASS artifactへ新統合のPASSを上書きせず、対象commit／image／revisionで新規証跡を採る。

### SEC-DR-009 — 現行proxyはtarget公開allowlist／WebSocket denyを満たさない

- Severity: **HIGH**
- 合否: **NOT CONFORMANT**
- 対象:
  - `MEDIATOR_PAYMENT_INTEGRATION_REQUIREMENTS.md` — `HTTP-001`〜`HTTP-006`、`TEST-012`
  - `mediator-payment-integration-design/09_DEPLOYMENT_PUBLIC_BOUNDARY.md` — `TBL-ROUTE-01`、internal route deny matrix
  - 現行 `deploy/nginx.conf:65-88,129-175`
  - 現行 `tests/security/test_release_boundaries.py:31-38`
- 根拠:
  - targetはraw `/ws`、`/api`、`/store`、internal control、direct A2Aをedgeで404にし、未知path／methodもupstreamへ渡さない。
  - 現行nginxは `/ws/` を認証なしで公開し、`/api/` を認証付きで公開し、root prefixをUIへ広くproxyする。targetのexact method＋normalized path allowlistではない。
  - 現行testは設定文字列の存在確認が中心で、encoded separator、dot segment、duplicate slash、unknown method、WebSocket upgrade、cross-subject response、internal routeのbody／header非露出をblack-boxで証明しない。
- 修正案:
  1. nginxまたはedge routerをdefault 404のexact allowlistへ置換し、正規化前後のambiguous pathをproxy前に拒否する。
  2. raw WebSocketを閉じる。必要なpushは認証済みsession-scoped SSEまたは用途限定channelとして別contract化する。
  3. deployed URLに対するmethod×path×auth×CSRF matrixを自動生成し、status、content type、body sentinel、redirect、upstream非到達をblack-box testする。

### SEC-DR-010 — 固定Cloud Run serviceを安全に更新する実行経路がない

- Severity: **HIGH**
- 合否: **NOT CONFORMANT**
- 対象:
  - `MEDIATOR_PAYMENT_INTEGRATION_REQUIREMENTS.md` — `OPS-002`〜`OPS-009`
  - `mediator-payment-integration-design/09_DEPLOYMENT_PUBLIC_BOUNDARY.md` — fixed target、0% traffic update、post-deploy verification
  - 現行 `deploy/deploy-payment-demo-cloudrun.sh:1-3,14-17,39-52,74-89`
- 根拠:
  - target設計は既存の固定serviceへimmutable image digestで新revisionを0% trafficに作り、検証後にtrafficを切り替え、失敗時は旧revisionを維持する。
  - 現行scriptは明示的に `NEW service` 専用で、同名serviceが存在すれば終了する。その後の `gcloud run deploy` もtargetの0% traffic／candidate revision検証／明示traffic切替手順を表現しない。
  - 既存serviceがある通常の更新局面では、設計どおりの唯一の安全な操作手順を実行できない。
- 修正案:
  1. fixed project／region／serviceを再検証したうえで、immutable digestから `--no-traffic` のcandidate revisionを作る専用update pathを用意する。
  2. revision名、image digest、env、ephemeral label、ready、route deny、negative payment、副作用0、conformance statusを確認してから明示traffic切替する。
  3. preflight failure、同名revision、partial update、readiness failure、traffic切替失敗、rollbackを試験し、既存trafficが意図せず変わらないことをassertする。

### SEC-DR-011 — Agent Cardのprofile declarationと完全一致対象が未定義

- Severity: **HIGH**
- 合否: **FAIL**
- 対象:
  - `MEDIATOR_PAYMENT_INTEGRATION_REQUIREMENTS.md` — `SEC-007`、`SEC-013`、`SEC-014`、`TEST-008`
  - `mediator-payment-integration-design/04_PAYMENT_BRIDGE_AP2_X402.md` — 8章 profile選択
  - `mediator-payment-integration-design/06_API_A2A_CONTRACTS.md` — Live Agent Card、payment-required検証、profile activation／metadata
  - pinned [A2A x402 Payments Extension v0.1](https://github.com/google-agentic-commerce/a2a-x402/blob/125db5526a965d2325459d1a9df2e274a7e42396/spec/v0.1/spec.md)
- 根拠:
  - payment-requiredを `input-required`＋exact status＋strict required objectの三条件だけで認識し、free textやunknown keyを除外する設計はPASS相当である。
  - しかし06のLive Card例が宣言するのはextension `uri / description / required` だけであるのに、後段はprofile ID、URI、scheme、network、asset、payTo、requirements digestを「Card、header、Task、capability、payload、receipt」で完全一致させる。
  - pinned x402 Card例もextension URI中心であり、上記commerce parameterをCardのどのschema位置から取得するか、official fieldかproject-local extension paramsかが定義されていない。このままでは正当なCardを常にrejectする実装と、Card比較を省略する実装がどちらも書ける。
- 修正案:
  1. Card-level capabilityはcanonical URI／required／A2A capabilityへ限定し、scheme／network／asset／payToはsigned payment-required、runtime readiness、capability、payload、receipt間で照合すると整理する。
  2. Cardへproject-local parameterを載せるなら、namespace、strict schema、canonicalization、署名／digest範囲、unknown field policyを06へ追加する。
  3. missing URI、multiple URI、official＋simulation混在、Card swap、runtime echo欠落、scheme／network／asset／payTo各不一致を個別試験する。

### SEC-DR-012 — `orderId` のownerと生成時点が矛盾する

- Severity: **MEDIUM**
- 合否: **PARTIAL**
- 対象:
  - `mediator-payment-integration-design/02_DOMAIN_DATA_STATE.md` — identifier owner表
  - `mediator-payment-integration-design/06_API_A2A_CONTRACTS.md` — 初回 `message/send` metadata、payment-required contract
- 根拠:
  - 02は `order_id / quote_id` をMerchant ownerとする。
  - 06はMerchantが初回Task／payment requirementを返す前のstart request metadataへ `orderId` を入れる一方、payment-required responseでもMerchantのorder／quoteを必須とする。client生成IDとMerchant正本IDが同名になり、上書き、echo、完全一致の意味が不明である。
- 修正案:
  1. start requestでは `clientRequestCorrelationId` または `merchantOrderIntentId` を使い、Merchantが返すcanonical `orderId / quoteId` と分離する。
  2. clientがorder IDを予約するprotocolならowner表を共同／client予約へ修正し、Merchantのexact echo、重複、別client collisionを定義する。
  3. Merchantが別order IDを返すcase、同一client correlationの再送、quote更新時のorder継続を試験する。

### SEC-DR-013 — 要件の「AP2 Intent」とpinned AP2／target設計の分類が一致しない

- Severity: **MEDIUM**
- 合否: **PARTIAL**
- 対象:
  - `MEDIATOR_PAYMENT_INTEGRATION_HANDOFF.md` — `FR-008`、8.6 AP2とA2A x402の適合判定
  - `MEDIATOR_PAYMENT_INTEGRATION_REQUIREMENTS.md` — `FR-008`、`SEC-012`
  - `mediator-payment-integration-design/04_PAYMENT_BRIDGE_AP2_X402.md` — `TBL-PAY-01`
  - `mediator-payment-integration-design/12_DECISIONS_OPEN_QUESTIONS.md` — `OQ-008`、`OQ-009`
  - pinned [AP2 v0.2 specification](https://github.com/google-agentic-commerce/AP2/blob/e1ea56db72a6385bce3e5c1112b3a56ce60acb43/docs/ap2/specification.md)
- 根拠:
  - HANDOFF／REQUIREMENTSは第一承認を「AP2 Intent Mandate」の入力と呼び、AP2 evidence一覧にもIntentを含める。
  - target設計は計画承認を `project-local plan approval / Intent evidence` と分類し、AP2 Mandateと呼ばない。これはpinned AP2 v0.2のHuman Present direct flowをCheckout／Payment Mandate中心に扱う点では安全側だが、normative requirementの文言とは一致しない。
  - このままではproject-local Intent evidenceを作って要件PASSとするか、存在しない／別versionのAP2 objectを要求するかがreviewerによって変わり、適合主張が一意にならない。
- 修正案:
  1. pinned AP2 v0.2で標準化されているobject名とproject-local evidenceを要件で分離する。
  2. 第一承認は「project-local plan approval／Intent evidenceであり、closed Checkout／Payment Mandateの生成条件とcorrelation input」と明記し、AP2標準objectとは主張しない。
  3. conformance reportでもAP2標準項目とproject-local補強を別行にし、project-local objectをAP2 PASSの根拠にしない。

## 5. 観点別評価

| 観点 | 判定 | 要点 |
| --- | --- | --- |
| 二段階完全一致承認 | PARTIAL | 各承認の文字列完全一致、期限、digest、nonce、CASは明確。公開workflow指定routeが一意routingを迂回する。 |
| subject／tenant／session ownership | FAIL | targetの完全tuple方針は妥当だがplan DTOからsubject欠落。現行domainはsubjectを捨てる。 |
| plan／step／agent／card／task／order／quote／workflow correlation | FAIL | 大半のfield表はあるが、envelope循環、order owner矛盾、serialized schema欠落がある。 |
| signed capability／extension | PARTIAL | JWT claim／scope／aud／nonce／expiry／idempotencyは詳細。Card parameterの所在が不明。 |
| Agent Card profile negotiation | FAIL | exact URI、live Card pin、runtime echo、silent fallback禁止は妥当だが、Cardとcommerce parameterの比較schemaが不足し、正常系wire判定が一意でない。 |
| payment-required偽陽性防止／fail-closed | PASS（設計） | `input-required`、exact status、strict required objectの全成立を要求し、free text／unknown key／片側だけを拒否する。 |
| AP2 Human Present evidence／secret非露出 | FAIL | role分離、署名検証、secret除外は妥当。envelope循環とuntrusted Merchantへのrestricted data送信が残る。 |
| anomaly gate前後／BLOCK／REVIEW | PARTIAL | 五gateの順序と `BLOCK > REVIEW > PASS` は明確。現行実装／testsにはtarget stable IDの証跡がない。 |
| SSRF／redirect／timeout／size | PARTIAL（非blocking） | 正常系はdestination allowlist、redirect 0、有限timeout／sizeを必須とする。DNS rebinding、peer IP、複雑retry等は既知課題。 |
| same Task／idempotency／outbox／reconciliation | PARTIAL | 正常系の既知Task継続と基本idempotencyは必須。初回response喪失の完全回復と高度なreconciliationは今回非blocking。例外フローは返金のみ必須。 |
| 公開HTTP／WebSocket／identity header | FAIL | target deny matrixは良いがmutation routingとidentity contractに穴。現行nginxもtarget非適合。 |
| ephemeral Cloud Run／固定service update | FAIL | target表示／0% revision方針は良い。現行実行scriptはNEW-onlyで更新不能。 |
| simulationの `NOT CONFORMANT` 表示 | PASS（設計） | UI、evidence、receipt、profile URIをofficialから分離し、silent fallbackを禁止する。実装証跡はNOT RUN。 |
| 障害時REVIEW | NOT RUN（今回非blocking） | 設計方針はあるが、高度な自動復旧／reconciliationは今回適合を主張しない。必須の例外フローは返金に限定する。 |
| テスト可能性 | PARTIAL | 127要件matrixはある。今回は正常系、承認routing、profile／wire、subject分離、秘密非露出、公開deny、基本idempotency、返金を優先し、網羅的fault／悪意入力matrixは将来課題とする。 |

## 6. 一次資料pinの再確認

2026-08-16に固定commitのraw specificationを再取得し、設計記載のSHA-256と比較した。

| 資料 | 固定commit | 再計算SHA-256 | 判定 |
| --- | --- | --- | --- |
| AP2 v0.2 specification | `e1ea56db72a6385bce3e5c1112b3a56ce60acb43` | `32c3be5011f481d2760e56e7b9935b0842c3da0d5f7d7b8a68402a599f1e6aa3` | pin一致 |
| A2A x402 Payments Extension v0.1 | `125db5526a965d2325459d1a9df2e274a7e42396` | `5cdc35ed8c4d7a93bb120f1782fd06e2cc3ef19036684f772e27d0d644c66940` | pin一致 |

pin一致は仕様参照の再現性だけを示し、target実装済み、公式x402 settlement済み、AP2 production conformanceを示さない。repository移転やlatestとの差分を暗黙追従せず、release前に同じcommit／path／hashで再検証する方針は妥当である。

## 7. 合格へ必要な最小閉鎖条件

次をすべて満たすまで総合判定をPASSへ変更しない。

1. `SEC-DR-001` の公開mutation routeをsession-level authoritative routingへ一本化する。
2. `SEC-DR-002` のevidenceを提出前authorizationと結果後completionの一方向chainへ変更し、完全wire schemaを定義する。
3. subjectをplan digestを含む全owner keyへ通し、identity assertionをFirebase検証済み主体からだけ発行する。
4. Merchant wireを最小開示へ分割し、restricted subject／session envelopeを送らない。
5. 正常系の外向き通信にexact destination、redirect 0、有限timeout／size、Content-Type検査を適用する。高度なDNS／retry対策は既知課題として残してよい。
6. 正常系でTask ID保存後のsubmit／getを同一Task・同一step・同一idempotency keyへ限定する。初回response喪失の完全回復は今回非blockingとする。
7. exact public allowlist、WebSocket deny、canonical identity headerをdeployed edgeでblack-box検証する。
8. 正常系の五gateと禁止副作用0件、same Task／same step、profile mismatch、capability tamper、offline evidence mutation、および必須例外フローの返金をtarget経路で実行する。高度なreconciliationはPASSに数えない。
9. 今回scopeに対応するrequirementのimplementation／test／evidence refsを対象commit／image／Cloud Run revisionへ結び付け、非対象項目は `NOT RUN`／既知制限とする。
10. fixed existing serviceへの0% traffic revision作成、検証、明示traffic切替、rollbackを実行可能にする。

## 8. レビュー実施上の注記

- 本レビューは静的なread-only突合であり、test suite、browser test、Cloud Run操作、支払simulationを実行していない。実行結果はすべて `NOT RUN` と扱う。
- 現行のdirect payment workflowには再利用可能なCAS、outbox、署名、same-Task submission、simulation表示、Firebase入口認証の部品がある。ただし、それらの既存PASS artifactは新しい仲介統合経路のPASS evidenceではない。
- 未commitの既存ファイルは変更していない。

## 9. 最終判定

- 設計security sign-off: **FAIL**
- AP2／x402適合表現: **PARTIAL。simulationは引き続き `NOT CONFORMANT` と表示必須**
- 現行実装のtarget要件適合: **NOT CONFORMANT**
- release readiness: **FAIL**
- 重大指摘件数: **9件（BLOCKER 2、HIGH 7）**
- 本レビューによる変更ファイル: `docs/payments/MEDIATOR_PAYMENT_INTEGRATION_DESIGN_REVIEW_SECURITY.md` のみ

## 10. 最新設計の最終再レビュー

> 2026-08-16追記。本節は設計リードによる反映後の再レビューであり、**設計readinessについては9章の初回判定を更新する**。現行コードの `NOT CONFORMANT` 判定は、まだtarget実装前であるため変わらない。

### 10.1 Scopeと更新判定

今回blockingは正常paid、正常free、基本refund、identity／公開経路／Merchant最小開示の安全境界に限定した。高度restart、first-response-loss完全回復、複雑retry／concurrency、DNS rebinding、網羅的malicious matrixは12のfuture-workとしてよく、今回のGO判定へ含めない。

**更新判定: 正常実装計画へは CONDITIONAL GO。**

architectureと主要security invariantは実装計画を開始できる水準へ改善された。ただし、下記3件のHIGHを最初のcontract-closure作業として解消するまで、payment result／settlement／refundおよびwire validatorの仕様をfreezeしてはならない。従って、これは無条件のdesign sign-offやrelease PASSではない。

### 10.2 指定観点の再確認

| 観点 | 再判定 | 根拠 |
| --- | --- | --- |
| typed legacy seam | PARTIAL | 01 §Production composition seamで実symbol、typed adapter、実call eventを固定した。anomaly二fileの名称だけ現行pathと不一致。 |
| session-only routing | PASS | 03 §3と09 `TBL-ROUTE-01` は公開mutationを `POST /mediation-api/v1/turns` だけにし、workflow ID直指定mutationを除外した。 |
| Trusted Surface | PASS | 03 §9、04 §3／§6、06 §10でHuman Presentの同意／user signatureをnon-agentic Trusted Surfaceへ固定し、LLM承認を禁止した。 |
| Shopping Agent／orchestrator | PASS | 認可済みartifactを入力にpayment toolを進行できるが、承認・署名・検証を決めない役割へ修正された。 |
| deterministic payment tool | PASS | Mandate／terms／binding検証とguarantee発行前条件が決定論的codeへ分離された。 |
| demo guarantee | PARTIAL | `signed-payment-guarantee/1` をAP2標準artifactやsettlementと誤称しない点はPASS。後段settlementとの遷移が未閉鎖。 |
| same Task | PASS | 初回Task保存後、同じtask／contextへのsubmit、同じTask result／get、別Task拒否が03／06へ固定された。 |
| identity | PASS（設計） | owner tupleへsubjectを戻し、plan digest、continuation、CAS、内部assertionへ結合した。任意subject mint APIも明示的な廃止対象。 |
| callback／anomaly分離 | PASS | 既存A2A callback hook、deterministic gate、semantic reviewer、final validatorを別layer／別eventとし、互いの代用を禁止した。 |
| paid／free | PASS | paidはTask継続と五gate、freeはpayment artifact／settlement 0件と前後callback／final validationを明示した。 |
| 基本refund | PARTIAL | owner、元payment／receipt、明示承認、CAS、idempotency、1回上限は明確。refund前提となるsettlement完了点が未定義。 |
| public boundary | PASS（設計） | exact route allowlist、raw WebSocket／internal route deny、loopback backend、session-level mutationを定義した。現行nginxへの実装は未実施。 |

### 10.3 残存する正常系HIGH

#### RER-SEC-001 — `GUARANTEED` からsettlement／refundへ至る正規遷移がない

- Severity: **HIGH**
- 合否: **要contract closure**
- 対象:
  - `04_PAYMENT_BRIDGE_AP2_X402.md` §10、特に `FIG-PAY-02` と「Refund正常系」
  - `06_API_A2A_CONTRACTS.md` §10、§12 `Payment result Task`／`Refund contract`
  - `08_PERSISTENCE_RECOVERY.md` §4、§7
  - `10_TEST_STRATEGY.md` `PAID-HAPPY-01`、`TEST-016`
- 根拠:
  - A2A submit時はMerchantがsettlementをせずlocal ledgerは `GUARANTEED` だけ、と明記された。
  - しかし同じ正常sequenceには後段settlement operation、actor、wire、state guard、receipt生成点がなく、最終Task例は `payment-completed`／`sim:attempt` receiptを返す。
  - 基本refundは「実simulation settlement成功済み」を前提にするため、どこで `GUARANTEED -> SETTLED` となるか不明なままではpaid happy pathとrefund fixtureを一意に実装できない。
- 修正条件:
  1. settlement owner、trigger、operation ID／idempotency key、入力、guard、result／receipt、`GUARANTEED -> SETTLED|FAILED|REVIEW` を一つのsequenceへ追加する。
  2. `payment-completed`、AP2 Receipt、completion manifest、final ACCEPTをguarantee受理時とsettlement完了時のどちらで許すか固定する。
  3. fulfillment成功、fulfillment失敗前settlement、未settlement guarantee cancel、settled後基本refundの4caseでcounterとledger stateを定義する。

#### RER-SEC-002 — 06のwire正本に旧profile／restricted submission記述が残る

- Severity: **HIGH**
- 合否: **要contract closure**
- 対象:
  - `04_PAYMENT_BRIDGE_AP2_X402.md` §8
  - `05_SECURITY_TRUST_BOUNDARIES.md` §9
  - `06_API_A2A_CONTRACTS.md:487,646,874,897,929`
- 根拠:
  - 更新済み本文は、Cardではextension URI／required／capabilityだけを検証し、scheme／network／asset／payToはsigned requirement、runtime、capability、payload、guarantee／receipt間で比較すると正しく分離した。
  - 06の検証規則とprofile節には依然、scheme／network／asset／payTo等をCardへ一致させる旧文言がある。
  - payment submission本文は最小guarantee packageだけを送りraw Mandate／envelope等を禁止したが、末尾tableはなお `restricted AP2 part` を必須とする。Decision参照と図にも廃止した `mediation-correlation-envelope/v1` が残る。
  - 06はserialized contractの正本であるため、実装者が旧行に従うと正当profileの誤拒否またはrestricted evidence漏えいが起こり得る。
- 修正条件:
  1. 06:487／646を04 §8と06 §7の新しい比較責務へ統一する。
  2. 06:929を `merchant-payment-guarantee-submission/1` のallowlistへ変更する。
  3. 04:397、06:874／897の旧envelope名をauthorization envelope＋completion manifestへ更新する。

#### RER-SEC-003 — requirement scopeとcoverage ledgerが同期していない

- Severity: **HIGH**
- 合否: **要contract closure**
- 対象:
  - `MEDIATOR_PAYMENT_INTEGRATION_REQUIREMENTS.md` — 132件の規範見出し
  - `11_TRACEABILITY_RELEASE.md` — `requirement_records: 127`、`record_count: 127`
  - `12_DECISIONS_OPEN_QUESTIONS.md` — `ADR-012`／future-work register
- 根拠:
  - REQUIREMENTSは `FR-016/017`、`SEC-016/017`、`TEST-016/017/018`、`AC-014/015` 等を含み、規範見出しは132件になった。
  - 11のcoverage manifestは127件のままで、新しい基本refund要件／試験／受入を含まない。
  - ADR-012の「blocking 126件＋future 13件」も全132件と算術上一致しない。これでは基本refundを正常scopeに含めた実装計画とclosure validatorが一致しない。
- 修正条件:
  1. 132件を唯一のsourceから再生成し、各recordへ `release-1-blocking|future-work` を付ける。
  2. `FR-016`、`SEC-017`、`TEST-016`、`AC-014` をRelease-1 blockingとして04／05／06／08／10へforward traceする。
  3. ADR-012の件数をmanifestの機械集計値へ一致させる。

### 10.4 非blockingの整合修正

- 01の `SemanticAnomalyReviewer`／`FinalValidationAdapter` の既存pathは、現行の `secure_mediation_agent/subagents/anomaly_detection_agent.py`／`final_anomaly_detection_agent.py` と完全一致させる。列挙されたfunction自体は存在する。
- 09のallowlist後に残る `{workflow_id}` 説明は、公開tableに該当routeがないため削除またはinternal ID一般説明へ変更する。
- 高度recovery／DNS／並行raceのfuture-work判定は維持し、今回の正常系PASS数へ混ぜない。

### 10.5 最終進行判断

- 正常実装**計画**への着手: **CONDITIONAL GO**
- payment／settlement／refund／wire schemaのfreeze: **RER-SEC-001〜003解消までNO-GO**
- 現行コードのtarget適合: **NOT CONFORMANT（実装前のため変更なし）**
- release／Cloud Run更新: **NOT RUN。target実装と対象candidate証跡が必要**
- 最新再レビューの残存重大指摘: **3件（HIGH 3、BLOCKER 0）**
