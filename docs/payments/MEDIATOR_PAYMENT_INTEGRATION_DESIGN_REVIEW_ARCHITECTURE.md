# 仲介エージェント決済統合：独立アーキテクチャ設計レビュー

- review date: 2026-08-16
- reviewer role: Independent architecture / implementation-feasibility reviewer
- reviewed lifecycle: `target-design`
- verdict: **FAIL — BLOCKER／HIGH解消まで実装着手不可**
- review target: [HANDOFF](MEDIATOR_PAYMENT_INTEGRATION_HANDOFF.md)、[CURRENT_STATE](MEDIATOR_PAYMENT_INTEGRATION_CURRENT_STATE.md)、[REQUIREMENTS](MEDIATOR_PAYMENT_INTEGRATION_REQUIREMENTS.md)、既存review、[DESIGN_STRUCTURE](MEDIATOR_PAYMENT_INTEGRATION_DESIGN_STRUCTURE.md)、[design README](mediator-payment-integration-design/README.md) と `01`〜`12`、および現行code／deployment設定のread-only突合

## 1. Scope override

本レビューはconceptual payment demoとして、まず次の正常系を実装可能にすることを優先する。

- 公開 `payment_user_agent` から既存 `secure_mediator` の実matcher、planner、承認済みplan、orchestrator、legacy callback、anomaly gate、final validationを通る。
- 同じ入口からpaid／freeの双方を実A2A HTTPで完走する。
- paidはplan approvalとpayment approvalを分け、同じremote A2A Taskへ支払を提出する。
- 全correlation、基本的なCAS／idempotency、二重Task／二重支払防止を成立させる。
- 例外フローは返金だけを今回の実装対象にする。

高度なrestart／recovery、応答喪失、複雑な並行競合、複数pending選択、網羅的failure matrixの細部は将来課題とし、今回の実装開始をblockしない。このoverrideは127件の設計traceabilityを削除するものではなく、実装順序と重大度だけを変更する。

## 2. 結論

設計は、public boundary、二段階承認、paid／free分岐、同一Task継続、deterministic security gate、final validation、target／implemented／verifiedの分離まで、目標構造を概ね正しく捉えている。一方、normal vertical sliceをcodeへ落とす直前のcontractが4か所で閉じていない。

最重要は、AP2 object digestとcorrelation envelope digestの相互参照が循環しており、canonical bytesを一意に生成できない点である。また、設計上の `MatcherAdapter` 等が現行の実subagentをどう実行するか、free Agent processをどこに配置するか、normal paid flowと返金をどの物理record／制約で守るかが実装者の推測に残っている。

| Severity | Count | Release／implementation判断 |
| --- | ---: | --- |
| BLOCKER | 1 | 正常系artifactを生成できないため、修正必須 |
| HIGH | 4 | normal vertical sliceの実経路または安全性が未確定。実装着手前に修正必須 |
| MEDIUM | 4 | 実装と並行修正可能。ただしdesign closure／release前に必須 |
| LOW | 0 | なし |
| Total | 9 | **FAIL** |

## 3. 127要件と文書構造の検証結果

### 3.1 形式検証はPASS

read-only parserで次を確認した。

- `REQUIREMENTS` の規範H3は127件、重複0。
- `11_TRACEABILITY_RELEASE.md` のfront matterは127 record、重複0、missing／unknown ID 0。
- 全recordにprimary design file／anchorが一つあり、owner件数は `01:1, 02:19, 03:7, 04:7, 05:9, 06:4, 07:10, 08:5, 09:14, 10:27, 11:24` で合計127。
- `implementation_refs` は全127件が空だが、design closureでは空を許すという[11 §2](mediator-payment-integration-design/11_TRACEABILITY_RELEASE.md#2-coverage-manifest-schemaと生成方向)の規則には適合する。

### 3.2 観点別判定

| 観点 | 判定 | 根拠 |
| --- | --- | --- |
| 127 ID集合、primary owner | PASS | H3、19.3、front matterの集合と件数が一致 |
| 責務分割 | PARTIAL | semantic／wire／persistence ownerは明記。ただし物理contractとproduction adapter seamが不足 |
| public rootから実secure mediator | FAIL | `Adapter`名はあるが、現行実subagent symbolへ到達するproduction compositionが未定義 |
| matcher→planner→承認済plan→orchestrator | FAIL | target順序は明確だが、legacy string APIからtyped contractへの変換責務と実呼出し証明が不足 |
| continuation／state／基本CAS | FAIL | 論理規則はあるが、normal flowを守る最低限のDDL／unique constraint／CAS predicateがない |
| paid／free | FAIL | flow semanticsは明確だが、free A2A runtimeのdeploy contractがない |
| same remote Task | PASS（設計概念） | [06 §8](mediator-payment-integration-design/06_API_A2A_CONTRACTS.md#8-a2a-taskmessage-lifecycle-contract)でtask/contextを後続messageへ固定 |
| final validation | PASS（設計概念） | [03 §11](mediator-payment-integration-design/03_MEDIATION_FLOW.md#11-final-validation)で全step後の入力とACCEPT／REVIEW／REJECTを定義 |
| UI／public boundary | PASS（normal flow） | 認証、CSRF、allowlist、safe view、single public appの境界は実装可能 |
| deploy | FAIL | free runtime欠落と`:8001` process名称不整合あり |
| test／release | PARTIAL | main integration scenarioはあるが、normal sliceのwire／count oracleと生成物検証が薄い |
| cross-link／ADR | FAIL | accepted OQのaffected IDsとmanifest `decision_refs`が20件不一致 |
| 現行基盤再利用 | FAIL | 変更対象fileは示すが、既存実subagentを残すproduction seamが一意でない |

## 4. 指摘

### ARC-BLOCKER-001 — AP2 artifactとcorrelation envelopeのdigest参照が循環する

- severity: **BLOCKER**
- pass／fail: **FAIL**
- affected normal flow: paid evidence生成、Merchant検証、offline correlation、same Taskへのpayment submit
- design anchors: [04 §7](mediator-payment-integration-design/04_PAYMENT_BRIDGE_AP2_X402.md#7-仲介correlationのevidence-binding)、[06 payment submission](mediator-payment-integration-design/06_API_A2A_CONTRACTS.md#10-支払提出contract)、[OQ-008](mediator-payment-integration-design/12_DECISIONS_OPEN_QUESTIONS.md#oq-008)

根拠:

- `04` はAP2 canonical objectへ独自fieldを追加しない一方、署名済みenvelopeとAP2 objectのimmutable bytes／digestを「相互参照」させるとしている。
- OQ-008はenvelopeに全AP2 object digestを含め、AP2 objectもenvelope digestをreferenceとして持つとしている。
- `06` のpayment submissionにはAP2 objectとenvelopeが並列で置かれるが、`mediation-correlation-envelope/v1` のexact schema、canonicalization、生成／署名順序はない。

AP2 object bytesがenvelope digestを含み、envelope bytesがそのAP2 object digestを含むなら、どちらのdigestも先に確定できない。AP2 objectへreference fieldを加えない場合は、OQ-008の「AP2 objectがreferenceを持つ」記述とも両立しない。

具体的修正:

1. 結合を一方向にする。AP2 canonical objectは変更せず先にexact bytes／digestを確定し、project-local signed envelopeがそれらのdigestをcommitする。
2. envelope digestを参照できるのはproject-local A2A metadata、result、evidence indexに限定し、AP2 canonical bytesから除外する。
3. `06` にexact envelope JSON schema、field型、required、canonicalization、署名対象、生成順序、verification順序を置く。
4. `04`、`06`、OQ-008を同時に修正し、外部DBなしのoffline verifier fixtureで全field一致／一field改ざんを確認する。

再review PASS条件: 同じ入力から有限手順で一意なbytes／digest／signatureを生成でき、AP2 schemaを変更せずoffline verificationが成功すること。

### ARC-HIGH-001 — public rootが既存secure mediatorの実subagentを通るproduction seamが未確定

- severity: **HIGH**
- pass／fail: **FAIL**
- affected requirements: `FR-001`, `FR-003`, `FR-004`, `FR-010`, `FR-011`, `TEST-006`
- design anchors: [01 component topology](mediator-payment-integration-design/01_OVERVIEW_ARCHITECTURE.md#4-target-component-topology)、[01 FR-001](mediator-payment-integration-design/01_OVERVIEW_ARCHITECTURE.md#fr-001)、[10 TEST-006](mediator-payment-integration-design/10_TEST_STRATEGY.md#test-006)
- code anchors: `payment_user_agent/agent.py:3-12`, `secure_mediation_agent/agent.py:49-91`, `secure_mediation_agent/__init__.py:10-18`, `secure_mediation_agent/subagents/matching_agent.py:33-49`, `planning_agent.py:108-131`, `orchestration_agent.py:136-204`

根拠:

- target topologyは `MediationController -> MatcherAdapter -> TypedPlannerAdapter -> ... -> OrchestratorAdapter` を要求し、import／label／mock traceだけでは不可としている。
- しかしproduction factory、実装class／function、入力出力型、transaction境界、component revisionを固定していないため、新しい同名adapterがlegacy実装を呼ばなくても設計表面上は成立する。
- 現行public rootは `PaymentWorkflowAdapter` だけである。`secure_mediation_agent.__init__` は存在しない `agent.root_agent` をlazy importしようとする。
- 現行matcherはURLへAgent Card pathを文字列追加し、planner helperは空stepsのJSON stringを返し、orchestratorは毎回random user／sessionとin-memory serviceを作ってtext中心に収集する。targetのpinned endpoint、typed approved plan、same Task metadata、durable continuationとは互換でない。

具体的修正:

1. `payment_user_agent.root_agent` からproduction `MediationController` factoryまでの一意なcompositionを図とsymbol tableで定義する。
2. Matcher／Planner／Orchestrator／legacy callbacks／anomaly detector／final detectorごとに「再利用する現行symbol」「typed adapter」「置換するlegacy behavior」を列挙する。
3. adapter input／outputを`06`のstrict DTOへ固定し、planner outputはvalidation後のplanだけ、orchestrator inputは承認済みplan snapshotだけにする。
4. correlation traceへcomponent ID、implementation revision、input／output digest、call ordinalを保存する。
5. TEST-006はproduction compositionを起動し、上記実componentが各一回実行されたことをspy／event／network captureで証明する。

再review PASS条件: public request一件から、実matcher、実planner、plan approval、実orchestrator、legacy callback、stable gates、final validatorまでのsymbol-level traceを一意に説明できること。

### ARC-HIGH-002 — normal paid／free flowを守る最低限のphysical persistence contractがない

- severity: **HIGH**
- pass／fail: **FAIL**
- affected requirements: `FR-004`, `FR-006`, `FR-007`, `FR-009`, `DATA-001`〜`DATA-006`, `STATE-002`〜`STATE-005`
- design anchor: [08 §4](mediator-payment-integration-design/08_PERSISTENCE_RECOVERY.md#4-logical-modelからphysical-storeへのmapping)、[08 §5](mediator-payment-integration-design/08_PERSISTENCE_RECOVERY.md#5-transaction-boundaryとcas)、[08 §7](mediator-payment-integration-design/08_PERSISTENCE_RECOVERY.md#7-idempotency-scopeとside-effect-count)
- code anchor: `secure_mediation_agent/workflow/migrations.py`

根拠:

`08` は三DBのrecord群、continuation最低field、transaction、CAS、outbox、stable keyを説明するが、現行migrationへ落とすtable／column／FK／unique／check／index、owner predicate、CAS SQLのrow-count条件を所有していない。normal flowだけでも、承認前Task start禁止、同一stepのTask一件、同一continuationのpayment Message／settlement一件、同じsubject／tenant／sessionだけのresumeをDB制約とtransactionで保証する必要がある。

具体的修正:

1. 今回範囲の最小schemaとしてsession、validated plan／step、plan approval、continuation／remote Task snapshot、payment requirement、payment approval、gate decision、idempotency、side-effect／outbox、refund record、audit／evidence referenceを物理mappingする。
2. 各columnの型、nullability、FK、check、index、既存／新規を表にし、`migrations.py`のtarget revisionへ結ぶ。
3. owner 4-tuple＋expected state＋expected versionのCAS predicate、成功row count=`1`、不一致時副作用=`0`を定義する。
4. Task start、payment submit、settlement、fulfillment、refundのstable keyにunique constraintを置く。

今回blockしないもの: 多process failover、長期lease回収、全checkpoint restart、応答喪失の完全reconciliation、複雑なwinner matrix。

再review PASS条件: paid／free正常系と一回の返金について、各state transitionと外部callを一つの明示transaction／unique key／outbox rowへ対応付けられること。

### ARC-HIGH-003 — free A2A Agentがdeployment topologyから欠落している

- severity: **HIGH**
- pass／fail: **FAIL**
- affected requirements: `FR-012`, `TEST-007`, `AC-002`, `OPS-004`
- design anchors: [06 §7](mediator-payment-integration-design/06_API_A2A_CONTRACTS.md#7-agent-registrycard-contract)、[09 §4](mediator-payment-integration-design/09_DEPLOYMENT_PUBLIC_BOUNDARY.md#4-process-topologyとlisten-boundary)、[09 §9](mediator-payment-integration-design/09_DEPLOYMENT_PUBLIC_BOUNDARY.md#9-readinessとhealth)、[10 TEST-007](mediator-payment-integration-design/10_TEST_STRATEGY.md#test-007)
- code anchor: `deploy/supervisord.conf:19-51`

根拠:

- TEST-007は同じmediator入口からpaid Agentとfree Agentを実A2A HTTPで呼ぶ。
- `09` のtarget process tableにはMerchant／paid Agent `:8005` しかなく、free Agent、listen port、startup、readiness、Registry／Card fixtureがない。
- 現行deploymentにはfree候補を含む `external_agents` が`:8002`にある。設計はこれをretain、adapt、replace、removeのいずれにするか決めていない。
- `09` は`:8001`をmarketplace storeと呼ぶが、現行processは`trusted_agent_store`であり、`marketplace.db` ownerとの区別も曖昧になる。

具体的修正:

1. 現行`:8002 external_agents`をnormal demo用free runtimeとしてretain／adaptするか、新しいfree A2A fixtureへ置換するかを決定する。
2. topology、listen table、startup、readiness、SSRF allowlistへfree endpointを追加する。
3. canonical Agent ID、Card URL、RPC endpoint、skill ID、Card digestのtyped Registry fixtureを`06`へ追加する。
4. `:8001`を`Trusted Agent Store`と正しく命名し、workflow persistenceの`marketplace.db`と分離して示す。

再review PASS条件: 一つのcandidate image内でpaid／free双方のreal A2A endpointが起動し、Card照合、readiness、TEST-007の呼出し先が一意になること。

### ARC-HIGH-004 — 唯一の例外フローとする返金contractが未設計

- severity: **HIGH**
- pass／fail: **FAIL**
- affected scope: settled後に業務を完了できないconceptual demoの返金
- design anchors: [08 §6](mediator-payment-integration-design/08_PERSISTENCE_RECOVERY.md#6-outboxworkerlease)、[08 §7](mediator-payment-integration-design/08_PERSISTENCE_RECOVERY.md#7-idempotency-scopeとside-effect-count)、[08 recovery table](mediator-payment-integration-design/08_PERSISTENCE_RECOVERY.md#tbl-rec-01)、[10 test counters](mediator-payment-integration-design/10_TEST_STRATEGY.md#4-fixtureとcounter)

根拠:

返金は`08`で`refund_required`、stable key、再実行前照会の語だけが現れる。返金を開始する条件、承認主体、状態、入力／出力DTO、元settlementとの結合、Merchant／rail operation、gate、Receipt／evidence、UI表示、final result、test scenarioが`03`／`04`／`06`／`07`／`10`にない。このままでは実装者が返金を自動実行するか、人手承認するか、単なるlocal ledger更新にするかを選べてしまう。

具体的修正:

1. concept demoの唯一の例外を「settlement成功後、fulfillment commitが明示失敗した場合」に限定する。
2. `RefundRequired -> Refunding -> Refunded | RefundFailed/ReviewRequired` の最小stateと、開始権限を定義する。自動返金にするなら上限とdeterministic policy、人手承認にするなら承認artifactを明記する。
3. typed `RefundRequest`／`RefundResult`をoriginal settlement ID／digest、order、Task、amount／currency、reason、idempotency keyへ固定する。
4. simulation railの一回のrefund operation、Receipt、evidence、final validation、safe UI viewを定義する。
5. TEST-009またはAC-007配下に一つの返金scenarioを追加し、settlement=`1`、fulfillment=`0`、refund=`1`、二回目=`0`をoracleにする。

再review PASS条件: 一つの明示triggerから一回のrefund Receiptまでをstate、wire、persistence、UI、testでend-to-end追跡できること。

### ARC-MEDIUM-001 — normal vertical sliceのtest oracleが一文へ圧縮されている

- severity: **MEDIUM**
- pass／fail: **FAIL（実装と並行修正可）**
- design anchors: [10 TEST-006](mediator-payment-integration-design/10_TEST_STRATEGY.md#test-006)、[TEST-007](mediator-payment-integration-design/10_TEST_STRATEGY.md#test-007)、[TEST-008](mediator-payment-integration-design/10_TEST_STRATEGY.md#test-008)、[AC-001](mediator-payment-integration-design/10_TEST_STRATEGY.md#ac-001)、[AC-002](mediator-payment-integration-design/10_TEST_STRATEGY.md#ac-002)

根拠:

main scenario自体は定義済みだが、fixture、実component identity、操作、expected state、wire oracle、DB oracle、forbidden side effect、exact countをstable case単位で結んでいない。特に「actual secure mediator経由」と「同じTask」は、trace labelだけで誤PASSしやすい。

具体的修正:

- `PAID-HAPPY-01`、`FREE-HAPPY-01`、`REFUND-01` の三caseだけを先に固定する。
- 各caseにcomponent call ordinal、gate順／回数、Task／context、approval ID／digest、A2A HTTP capture、DB state、Task start／payment／settlement／fulfillment／refundのexact countを置く。
- 高度なrestart、全field tamper、複雑な並行承認matrixはfuture suiteへ明示的に分離する。

再review PASS条件: 三caseのexpected traceと副作用countを、実装者が追加判断なしでtest codeへ転記できること。

### ARC-MEDIUM-002 — accepted OQのaffected IDsとmanifest `decision_refs`が20件不一致

- severity: **MEDIUM**
- pass／fail: **FAIL（design closure前に必須）**
- design anchors: [11 front matter](mediator-payment-integration-design/11_TRACEABILITY_RELEASE.md)、[11 §3](mediator-payment-integration-design/11_TRACEABILITY_RELEASE.md#3-127件coverage-ruleとvalidator責務)、[12 accepted decisions](mediator-payment-integration-design/12_DECISIONS_OPEN_QUESTIONS.md#3-accepted-decision-records)

read-onlyでOQ側`affected requirement IDs`を展開し、manifestの逆参照と比較した結果、次が欠けている。

| Decision | Missing `decision_refs` |
| --- | --- |
| OQ-001 | FR-009, DATA-001〜DATA-005, STATE-002〜STATE-005 |
| OQ-002 | FR-008 |
| OQ-003 | FR-007, OPS-005 |
| OQ-004 | SEC-004, DATA-004 |
| OQ-005 | DATA-008, STATE-007 |
| OQ-006 | NFR-004, SEC-010 |
| OQ-009 | CLAIM-003 |

合計20 recordであり、referenceの実在検査だけでは検出できない。decision変更時の影響範囲がcoverage manifestへ伝播しない。

具体的修正: OQのrange表記を展開したinverse mapをgeneratorで作り、`affected IDs == manifest decision_refs` の双方向集合一致をvalidatorへ追加し、20 recordを再生成する。

再review PASS条件: 全accepted OQでmissing／extraが0になり、意図的な例外だけ理由付きallowlistになること。

### ARC-MEDIUM-003 — generated coverage viewの完全性を再現できない

- severity: **MEDIUM**
- pass／fail: **FAIL（design closure前に必須）**
- design anchors: [DESIGN_STRUCTURE §15](MEDIATOR_PAYMENT_INTEGRATION_DESIGN_STRUCTURE.md#15-coverageの機械可読正本)、[design README §5](mediator-payment-integration-design/README.md#5-traceabilityと生成規則)、[11 §6](mediator-payment-integration-design/11_TRACEABILITY_RELEASE.md#6-coverage自動検査)

根拠:

- structureは全owner tableにgenerated marker、generator version、manifest content digest、byte-equivalent regenerationを要求する。
- 実fileでは`01`〜`03`だけが簡易`GENERATED` markerを持ち、`04`〜`11`にはmarkerがない。manifest content digestはどのviewにもない。
- generator／validatorの固定path、実行command、baseline reportが示されていない。

具体的修正: generatorとvalidatorのrepository path／commandを固定し、全生成viewへmarker、version、manifest digestを付けて再生成する。CIはbyte差分、anchor、OQ inverse mapをfail closedにする。

再review PASS条件: clean checkoutで一command再生成した結果がbyte-equivalentで、手編集を検知できること。

### ARC-MEDIUM-004 — candidate ledgerはfield列挙のみで機械検証schemaになっていない

- severity: **MEDIUM**
- pass／fail: **FAIL（release前に必須）**
- design anchor: [11 §10](mediator-payment-integration-design/11_TRACEABILITY_RELEASE.md#10-candidate-ledger-schema)

根拠:

ledgerの必要情報は一文で列挙されるが、固定artifact path、JSON schema ID／version、field型、127件のuniqueness、evidence URI／hashの意味、judge identity、signature、status transition、validation commandがない。この状態では`127 PASS`を異なる形式や別candidate evidenceから組み立てられる。

具体的修正: ledger pathとJSON schemaを固定し、candidate digest、requirement ID unique、status enum、implementation／test／evidence refs、judge／time／reason、evidence hash検証をrequiredにする。`PASS`は同candidateの到達可能evidenceがある場合だけ許す。

再review PASS条件: 空ledger、重複ID、別candidate evidence、証跡なしPASS、hash不一致がvalidatorで非0になること。

## 5. 非blockingの将来課題

今回の正常系＋返金vertical sliceをblockしないが、127要件の最終closureには次が残る。

- 全checkpoint restart、process death、lease回収、response unknownの完全reconciliation。
- 複数同種pendingのselection token詳細と複雑な承認競合matrix。
- `tasks/get`、任意の拒否messageを含む全A2A operationのcallback before／after matrix。
- TEST-001〜015／AC-001〜013の全field tamper、全失敗注入、browser／Cloud Run／rollback evidence。

これらはcandidate releaseで127件を`PASS`にする前には必要だが、今回のconcept demo実装開始条件にはしない。

## 6. 再review gate

実装開始可へ変更する最低条件は次の5点である。

1. ARC-BLOCKER-001の一方向evidence graphとexact envelope schemaを確定する。
2. ARC-HIGH-001のpublic root→実secure mediator production compositionをsymbol単位で確定する。
3. ARC-HIGH-002のnormal flow最低schema／CAS／unique keyを確定する。
4. ARC-HIGH-003のfree A2A runtimeをdeployment contractへ追加する。
5. ARC-HIGH-004の返金trigger、state、wire、persistence、testを確定する。

上記修正後、`PAID-HAPPY-01`、`FREE-HAPPY-01`、`REFUND-01`の設計walkthroughが矛盾なく完走できれば、MEDIUMを追跡課題として実装開始可にできる。release判定はMEDIUMを含む全指摘と127件candidate ledgerが閉じるまでFAILのままである。

## 7. 最終判定

**FAIL — 現状は設計意図が良い一方、normal paid／free＋返金を一意に実装するcontractが未閉鎖である。BLOCKER 1件、HIGH 4件を修正してから実装へ進むこと。**
