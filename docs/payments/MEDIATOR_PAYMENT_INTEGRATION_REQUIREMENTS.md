# 従来の仲介エージェントへの決済統合：要件定義書

- 対象リポジトリ: `enterprise-a2a-pf`
- 対象機能: 従来の `secure_mediator` に対する AP2 Human Present 決済サブフローの統合
- 作成日: 2026-08-16（Asia/Tokyo）
- 要件の正本: [MEDIATOR_PAYMENT_INTEGRATION_HANDOFF.md](MEDIATOR_PAYMENT_INTEGRATION_HANDOFF.md)
- 現状分析: [MEDIATOR_PAYMENT_INTEGRATION_CURRENT_STATE.md](MEDIATOR_PAYMENT_INTEGRATION_CURRENT_STATE.md)
- 実装・完了計画: [MEDIATOR_PAYMENT_INTEGRATION_IMPLEMENTATION_PLAN.md](MEDIATOR_PAYMENT_INTEGRATION_IMPLEMENTATION_PLAN.md)
- final6検証証跡: [MEDIATOR_PAYMENT_INTEGRATION_TEST_REPORT.md](MEDIATOR_PAYMENT_INTEGRATION_TEST_REPORT.md)
- canonical design coverage: [11_TRACEABILITY_RELEASE.md](mediator-payment-integration-design/11_TRACEABILITY_RELEASE.md)
- 文書種別: 要件定義（設計、実装、試験実施、デプロイを含まない）

## 1. 文書の位置づけと規範

本書は、従来の仲介エージェントの通常実行へ決済を統合するための、検証可能な機能要件、セキュリティ要件、運用要件、受入条件、リリース条件を定義する。

`MEDIATOR_PAYMENT_INTEGRATION_HANDOFF.md` の要件を本書から削除してはならない。2026-08-16のユーザーによる最新scope overrideは、同文書より新しい指示として、要件の存否ではなくRelease-1でrelease-blockingにする範囲を変更する。本書と引継ぎ文書にその他の解釈差がある場合は、引継ぎ文書を優先する。`MEDIATOR_PAYMENT_INTEGRATION_CURRENT_STATE.md` は現状の事実を示す非規範の基準線であり、目標要件や適合証跡ではない。

本書の「必須」「してはならない」「のみ」は規範要件を表す。各要件の充足は、実装の存在ではなく、対応する試験と証跡によって判定する。

### 1.1 scope変更履歴と適用区分

| 日付 | 変更 | 規範上の扱い |
| --- | --- | --- |
| 2026-08-16 | 初版 | HANDOFFの全要件を一つのrelease gateとして定義 |
| 2026-08-16 | 最新ユーザーscope override | 概念決済デモは正常系と返金までを優先し、全要件を `Release-1必須` と `設計のみ・将来課題` に分類。要件本文は削除しない |

- `Release-1必須`: 対象candidateで実装・試験・証跡が揃い、要件statusが `PASS` でなければリリースできない。
- `設計のみ・将来課題`: 要件を維持し、設計判断、既知課題、実装triggerを記録する。Release-1では `DESIGNED`、`PARTIAL`、`NOT RUN` を許容し、当該statusだけではリリースを阻止しない。
- 正常系の認可・相関・profile検証、二重支払防止、秘密非露出、公開境界、各anomaly gate、final validation、fail-closedは将来課題へ移してはならない。
- 適用区分の正本は19.3とし、Release-1必須集合と将来課題集合の和集合は全規範IDと完全一致し、積集合は空でなければならない。

## 2. 目的と成功状態

利用者は、唯一の公開アプリ `payment_user_agent` から自然文で依頼する。内部の `secure_mediator` は従来どおり Agent を検索し、計画を作り、第一の利用者承認後に A2A タスクを実行する。実行中に検証済みの支払要求が返った場合だけ対象stepを停止し、第二の利用者承認、AP2認可・証跡、交渉済み支払profileによる提出を経て、同じremote A2A Taskと同じ仲介stepを再開する。全stepの終了後は、有料・無料を問わず最終異常検知を経てから結果を返す。

無料のAgentまたは支払要求のない応答では、決済workflow、Payment Mandate、settlementを一切作成しない。支払完了後に業務完了不能またはデモ上の返金要求が確定した場合は、元の支払へ相関した返金正常系を提供する。

成功の優先順位は、次の順とする。

1. 従来の `secure_mediator` の実経由
2. AP2 Human Presentの認可モデルと証跡
3. A2A x402 extensionへの、利用可能なprofile範囲での適合
4. 誤操作なく短時間で実演できる公開UI

## 3. 範囲

### 3.1 対象範囲

- `payment_user_agent` を唯一の公開アプリとするUI入口
- `secure_mediator` の `matcher`、`planner`、`orchestrator`、security callback／Judge、決定論的policy、final validationの実行経路
- 仲介計画承認と決済承認の二段階のHuman Present認可
- A2A応答に基づく無料／有料の実行時分岐
- 仲介step、remote A2A Task、AP2証跡、支払workflowの相関と継続
- AP2認可・証跡と、公式profileまたは明示的なローカルx402 simulation transport
- 認証、主体分離、公開HTTP境界、異常検知、冪等性、再試行、reconciliation
- 単一Cloud Runデモサービスのephemeral運用とリリース検証
- unit、integration、regression、security、restart、実ブラウザ、black-box試験の受入基準

### 3.2 変更可能な外部対象

- Cloud Runで更新してよい対象は、project `gen-lang-client-0585901015`、region `asia-northeast1` の既存サービス `payment-user-agent-demo` のみとする。
- Cloud SQLを含む新規または既存の外部永続DBを、本デモのために追加・変更してはならない。

## 4. 現状の基準線（非規範）

2026-08-16時点の調査では、公開経路は `payment_user_agent -> PaymentWorkflowAdapter -> 固定決済workflow -> 固定Merchant` である。固定workflow内の厳密な二段階承認、AP2風証跡、同一Taskへの提出、CAS、outbox、Firebase入口認証には再利用可能な資産がある。

一方、公開経路から従来の `matcher -> planner -> orchestrator -> anomaly_detector -> final_anomaly_detector` は切断され、Agent、商品、計画、支払要否が固定されている。現行Cloud Run revision、既存browser evidence、既存release validationは、この目標要件への適合証拠として扱わない。

現状の詳細と根拠は `MEDIATOR_PAYMENT_INTEGRATION_CURRENT_STATE.md` を参照する。後続工程は、現状の「実装済み」という記述を、本書の「必須」と混同してはならない。

## 5. 用語

| 用語 | 定義 |
| --- | --- |
| `payment_user_agent` | ADK Webで利用者が選択する唯一の公開アプリ。入口と表示を担うが、仲介判断や支払認可の正本ではない |
| `secure_mediator` | Agent検索、計画、実行、異常検知、最終検証を統括する従来の内部仲介ルート |
| 仲介計画 | `matcher` の選定結果を使い `planner` が生成する、ID、version、digest、stepを持つ計画 |
| 計画承認 | 対象の仲介plan ID、version、digestに対する第一の利用者承認 |
| 決済承認 | Merchantが返した閉じたCheckout条件に対する第二の利用者承認 |
| 決済workflow | 承認、AP2証跡、Merchant通信、支払、冪等性、再試行を担う決定論的な内部サブフロー |
| Merchant A2A | 業務Taskを処理し、必要な場合に構造化された `payment-required` を返す相手Agent |
| remote Task | 選定Agent上で開始されるA2A Task。同じ `taskId` と `contextId` で後続messageを受ける |
| continuation | 停止した仲介step、remote Task、支払workflow、主体を結び、後続ターンで同じ処理を再開する記録 |
| closed Checkout | 商品、金額、通貨、payee、期限、quote、支払方式が確定し、変更検知可能な支払条件 |
| AP2 evidence | Intent、Checkout、Payment Mandate、Credential、Receiptおよびそれらの署名・相関証跡 |
| Human approval | 画面に提示した計画または支払条件への利用者の完全一致`承認`。AP2 Mandateそのものではない |
| pre-payment authorization envelope | 承認、Mandate digest、terms、Task相関を束ねる仲介内部artifact。外部Agentへ送らない |
| real rail hold | 実決済レールによる資金引当。現在のSQLite simulationには実装しない |
| 公式profile | リポジトリが固定したversionについて、Agent Cardとruntimeが必要条件を満たし、wallet等を検証できる支払profile |
| `x402-wire-simulation/1` | このデモMerchant限定のローカルsimulation。公式x402準拠または実資産決済ではない |
| anomaly gate | 実行の特定境界で、構造化入力に対し `PASS`、`BLOCK`、`REVIEW` を強制する検査 |
| final validation | 全step終了後、最終結果を利用者へ返す前に必ず行う最終異常検知 |

## 6. Actorと信頼境界

| Actor / コンポーネント | 責務 | 信頼境界と制約 |
| --- | --- | --- |
| 利用者 | 自然文依頼、計画承認、決済承認、拒否 | 認証済みsubjectで識別する。入力は信頼しない |
| ブラウザ / ADK Web | UI表示、入力送信、段階と安全な相関情報の表示 | 公開境界。token、秘密鍵、完全な署名原文を表示しない |
| Firebase認証proxy | token検証、session、CSRF、origin検証、内部identity生成 | 外部identity headerを信用せず、検証後の値だけを上流へ渡す |
| `payment_user_agent` | 唯一の公開root、承認待ち状態に応じた入口制御 | 仲介や決済の権限判定をLLMへ委ねない |
| `secure_mediator` | 検索、計画、承認gate、A2A実行、最終検証の統括 | 内部コンポーネント。公開appとして列挙しない |
| `matcher` / Trusted Agent Store | 候補、canonical Agent ID、trust、capability、Card情報の提供 | Storeのデータもlive Agent Cardと突合し、暗黙のaliasを信用しない |
| `planner` | 選定結果に基づく仲介計画の生成 | LLM出力を直接権限にせず、正規化・digest・承認対象化する |
| `orchestrator` | 承認済みstepのA2A実行、停止、継続 | 保存済みAgent snapshotとremote Task相関から逸脱してはならない |
| `anomaly_detector` / `final_anomaly_detector` | 境界ごとの異常判定、最終判定 | 失敗、timeout、解析不能を自動許可に変換しない |
| Merchant A2A | Task、支払要求、支払結果、Artifactの返却 | 外部入力として扱う。自由文、URL、metadataを無条件に信用しない |
| 決済workflow / AP2 verifier | Human Present認可、証跡、支払提出、冪等性 | LLMから分離された決定論的な権限境界とする |
| wallet / facilitator / simulation rail | 支払transportまたはローカルsimulation | 対応profileとruntime検証なしに利用しない。simulationを公式と表示しない |
| SQLite / outbox / worker | instance内の状態、非同期処理、reconciliation | ephemeralであり、instance置換を越える永続性を主張しない |
| Cloud Run | デモの実行境界 | 変更対象を固定サービスに限定し、内部processをloopbackへ閉じる |

## 7. 機能要件

### FR-001 従来の仲介ルート

公開UIからの通常依頼は、実行時に `secure_mediator`、`matcher`、`planner`、計画承認gate、`orchestrator`、A2A前後のsecurity callback／決定論的policy、final validationを通らなければならない。semantic anomaly sub-agentは不確定な高risk結果を`REVIEW`へescalateする補助であり、各境界での明示呼出しを必須enforcementと誤記しない。決済workflowが仲介全体を置き換えてはならず、各通過は同一相関系列のtraceで証明できなければならない。

### FR-002 単一の公開アプリ

外部へ公開するADKアプリは `payment_user_agent` 一つだけとし、認証後は選択済みでなければならない。`secure_mediator` と決済workflowは内部コンポーネントとし、同じ公開階層の兄弟アプリとして列挙してはならない。

### FR-003 動的なAgent選定と計画

通常経路のAgentと計画は、固定値ではなく `matcher` と `planner` の実結果から決定しなければならない。matcher出力のcanonical Agent ID、Agent Card digest、Agent Card URL、RPC endpoint、skill、trust score、capabilityを同一plan stepへ保存し、orchestratorの実HTTP送信先とcapability制約へ引き継がなければならない。trace labelの一致だけを充足証拠としてはならない。

### FR-004 計画承認gate

orchestrator開始前に、対象plan ID、version、digestに結び付く第一承認を決定論的に検証しなければならない。計画承認は単一text partの完全一致 `承認` のみ受理し、`はい`、`OK`、`承認します` 等を自動承認してはならない。計画変更時は旧承認を失効させ、再承認を要求する。承認前にMerchant Task、Checkout、支払または外部副作用を開始してはならない。

### FR-005 A2A応答による支払要否判定

支払要否はクライアント入力や固定フラグではなく、選定Agentから返ったA2A Taskの許容stateと、正規化・検証済みpayment extensionの組合せから実行時に判断しなければならない。自由文だけを根拠に分岐してはならない。Agent Cardの支払能力と応答profileが未表明、未知、不一致、破損、または条件不正の場合はfail closedとする。

### FR-006 仲介stepの停止と継続

検証済み支払要求を受けたstepは、失敗または完了ではなく明示的な決済承認待ちとして保存しなければならない。待機中に一回のADK実行を長時間ブロックせず、後続ターンで同一subject、session、plan、step、remote Taskを検証して再開しなければならない。別依頼、別step、別Taskの承認を流用してはならない。

### FR-007 二段階承認の分離

計画承認と決済承認は、別の対象digest、nonce、承認ID、時刻、状態として保持しなければならない。決済承認には、Merchantから得た商品、正の整数最小単位の金額、通貨、payee、期限、支払方式を提示し、単一text partの完全一致 `承認` のみを受理する。

承認入力のrouting候補は、同一の認証済みsubject、tenant、ADK session、mediation sessionへ束縛された未期限切れpending recordだけとし、次の順序で排他的に決定しなければならない。

| 優先順位 | 候補件数 | 必須routing |
| ---: | --- | --- |
| 1 | `waiting_for_payment_approval` が1件 | 計画承認待ちの有無または件数にかかわらず、その1件だけを決済承認対象とする |
| 1 | `waiting_for_payment_approval` が2件以上 | どの対象も承認せず、同種の対象を明示選択させる |
| 2 | payment pendingが0件、`waiting_for_plan_approval` が1件 | その1件だけを計画承認対象とする |
| 2 | payment pendingが0件、plan pendingが2件以上 | どの対象も承認せず、同種の対象を明示選択させる |
| 3 | payment pendingとplan pendingがともに0件 | 承認として扱わず、新しい通常依頼として `secure_mediator` へ渡す |

別subject／tenant／ADK session／mediation sessionのrecord、期限切れrecord、完了済みrecordは候補件数に含めず、承認を適用してはならない。決済承認を計画へ、計画承認を決済へ、または一つの承認を複数recordへ適用してはならない。決済承認前にCredential、Payment Mandate、支払提出、settlementを開始してはならない。拒否、期限切れ、条件変更時は支払わず、stepの中断理由を返す。

### FR-008 AP2証跡と仲介計画の結合

第一の計画承認をAP2 Intent Mandateの入力として取り込み、Intent、Checkout、Payment Mandate、Credential、Receiptを、次の最低必須fieldへ直接または改ざん検知可能なimmutable referenceで結合しなければならない。

- 認証済み利用者subject、tenant、ADK session、mediation session
- 仲介plan ID、version、digest、step ID
- 選定Agentのcanonical ID、Agent Card digest、skill ID、RPC endpoint
- remote context ID、task ID、order ID、quote ID
- 商品、正の整数最小単位の金額、通貨、payee、期限、支払方式
- 計画承認ID、計画承認nonce、計画承認issued-at
- 決済承認ID、決済承認nonce、決済承認issued-at
- Intent、Checkout、Payment Mandate、Credential、ReceiptそれぞれのID、issuer、subject／audience、nonce、issued-at、expiry、署名または署名対象digest

各承認recordのnonce／issued-atと各AP2 object自身のnonce／issued-at／expiryを区別し、同名fieldの暗黙な流用を禁止する。offline verifierは外部DBの暗黙知なしに署名連鎖と上記fieldの一致または不一致を判定できなければならない。同じ計画内容の第一承認を決済workflow内でもう一度要求してはならない。closed Checkout条件の変更時は決済承認を失効させる。

### FR-009 同じremote A2A Taskへの支払提出

支払提出は、orchestratorが最初に開始した保存済みremote Taskへの後続 `message/send` とし、同じ `contextId`、`taskId`、`orderId`、`quoteId`、`legacy_step_id`、payment workflowを検証しなければならない。支払提出messageは、plan、step、canonical Agent、許可operation、remote task／context、expiryへ限定した検証済みsigned capabilityと、交渉済みprofileが必須とするextension headerおよびmetadataを含まなければならない。Merchantは副作用前に、capabilityの署名、issuer、audience、scope、operation、expiry、task／context bindingと、profile header／metadataの存在、version、内容を検証し、欠落、改ざん、不一致、期限切れを副作用なしで拒否しなければならない。Task開始requestは当該stepにつき一回だけとし、支払後に無関係な新規Taskを作ってはならない。成功、相関一致、capability検証成功、profile検証成功の場合だけ同じ仲介stepを完了へ進める。

### FR-010 強制的なsecurity / anomaly gate

次のstable gate IDを、記載順に独立した判定として実行しなければならない。同じ初回A2A応答が支払要求を含む場合も、`POST_A2A_RESPONSE` と `POST_PAYMENT_REQUIREMENT` をこの順で各1回実行し、一方で他方を代用してはならない。

| Gate ID | 発火条件と最低入力 | `PASS` 後に初めて許される副作用 |
| --- | --- | --- |
| `PRE_A2A_START` | 各stepの初回remote Task開始前。承認済みplan／step、Agent snapshot、request、capabilityのdigest | 当該RPC endpointへのTask開始request |
| `POST_A2A_RESPONSE` | 各stepの初回A2A応答受領後。完全な構造化Task、履歴、Artifact／metadata digest | free結果のstep取込み、またはpayment requirement専用gateへの移行 |
| `POST_PAYMENT_REQUIREMENT` | `POST_A2A_RESPONSE=PASS` かつ検証済みpayment-requiredのとき1回。Task、Agent Card、plan上限、Checkout／profile digest | continuation／決済承認待ちの作成と安全な条件表示 |
| `PRE_PAYMENT_SUBMIT` | 第二承認後かつ支払提出前に1回。二承認、AP2 evidence、相関、signed capability、profile wire digest | 支払提出messageまたはsettlement副作用 |
| `POST_PAYMENT_RESULT` | 支払結果受領後に1回。送信wire digest、Merchant結果、Task／context、Receipt／Artifact digest | 同一stepへの結果取込みと完了遷移 |

各gateは、gate ID、detector／policy version、入力schema version、入力digest、呼出回数、判定、理由、時刻をtraceまたは監査証跡へ残さなければならない。`BLOCK` は後続副作用を禁止し、`REVIEW` は自動継続せず人手確認可能な停止状態にする。検査の例外、timeout、解析不能、証跡不足を自動許可してはならない。

### FR-011 最終異常検知

有料・無料を問わず、全step終了後に、元依頼、承認済み計画、全A2A履歴、決済要約、仲介結果を入力としてfinal validationを実行しなければならない。実装はcallback hook／Judgeと決定論的validatorを組み合わせ、特定名のsub-agent明示呼出しを要件にしない。判定が`ACCEPT`になる前に利用者へ最終成功を返さず、`REJECT`と`REVIEW`を成功へ変換しない。

### FR-012 無料経路

支払要求がない応答は、従来の仲介経路、必要なanomaly gate、final validationを通って完了しなければならない。無料stepに対し、決済承認、Payment Mandate、payment workflow record、settlement recordを作成してはならない。複数stepの一部だけが有料の場合は、該当stepだけを停止する。

### FR-013 基本冪等性と二重支払防止

同じ承認、message、outbox item、支払提出の通常の再送または重複操作で、二重Task、二重settlement、二重返金を起こしてはならない。支払と返金はそれぞれ元plan／step／Task／order／operationへ結び付く安定した冪等性キーを持ち、成功済みoperationの再送は同じ結果を返さなければならない。状態更新とoutbox追加は同一transactionとして整合しなければならない。結果不明時は推測で成功にせず `REVIEW` とする。

### FR-014 実経路の可観測性

UIまたは監査表示で、依頼受付、Agent検索、計画作成、計画承認待ち、A2A実行、支払要求受領、決済承認待ち、AP2認可・支払提出、同一Task再開・完了、最終異常検知を識別できなければならない。各段階はagent名、plan／step／taskの安全な短縮ID、時刻または順序番号を持つ実traceとし、見せかけのsleepを追加してはならない。

### FR-015 デモ運用境界

デモは単一Cloud Runサービス内のloopback構成とephemeral状態を利用できる。Cloud SQLを追加・変更してはならず、Cloud Runの変更対象は固定サービス `payment-user-agent-demo` のみとする。公開面はFirebase認証後のUIと必要なsame-origin APIに限定し、内部Store、Merchant、workflow、worker、identity brokerを外部公開してはならない。

### FR-016 返金正常系

支払が成功済みである一方、同じ業務stepのfulfillmentを完了できないことが確定した場合、または認証済み利用者がデモ上の返金対象を明示した場合、元のsubject、session、plan、step、remote task／context、order、quote、settlement、Receiptへ結合した返金を開始できなければならない。返金額は元支払額を超えてはならず、返金前の認可・相関検証、返金用の一意な冪等性キー、返金結果とReceipt、同一stepへの結果反映を必須とする。同じ返金要求の再送で返金を重複させず、未settled支払または別主体・別orderへの返金を拒否する。Release-1では、この相関付き全額返金正常系までを必須とし、部分返金、複数回返金、外部rail障害からの完全自動回復は将来課題にできる。

### FR-017 高度な競合・再試行・reconciliation

同一stepへの高競合な並行承認は一つだけを成功させ、初回A2A応答喪失、複数段retry、outbox lease競合、同一instance内の子process再起動、Merchant成功後の仲介側更新失敗から、同じ冪等性キーとremote Taskを用いてreconciliationしなければならない。本要件は削除せず、Release-1では設計のみ・将来課題として扱える。

## 8. 非機能・セキュリティ要件

### NFR-001 応答性と実演性

通常依頼、承認待ち、支払待ち、結果照合の各段階は、実処理の進捗を利用者へ返せなければならない。速さまたは遅さを演出するsleepを追加せず、承認待ちを単一requestの長時間保持で実現してはならない。

### NFR-002 決定性と再現性

承認、相関、profile選択、状態遷移、冪等性、security判定の強制部分は、同じ正規化入力とversionに対し機械的に再現可能でなければならない。LLMの自然文出力だけで結果が変わる設計を認可境界に用いてはならない。

### NFR-003 監査可能性

各仲介段階、承認、外部呼出し、anomaly gate、支払、再試行、final validationを、順序付きの相関証跡として機械取得できなければならない。証跡は機密情報を含まず、対象candidateと結び付けられなければならない。

### NFR-004 境界付き外部通信

Agent Card取得、A2A request、model call、支払transportを含む外部通信は、timeout、再試行上限、response size、redirect、接続先を明示的に制限し、失敗時に安全な停止または同じ冪等性単位での回復を可能にしなければならない。

### SEC-001 認証済み主体の終端間binding

Firebaseで検証したsubjectを、tenant、ADK session、mediation session、仲介plan、continuation、payment workflow、AP2 evidence、承認、照会、再開へ一貫して結び付けなければならない。query parameterの `userId` だけを認証根拠にしてはならない。

### SEC-002 主体とsessionの分離

異なる有効subject、tenant、sessionから、別主体のplan、workflow、continuation、承認、支払、artifactを閲覧、変更、承認、再開できてはならない。

### SEC-003 内部identity

外部から受信した `X-Verified-Identity` その他の内部identity headerを破棄し、認証proxyが検証後に生成した署名付きidentity assertionだけを内部APIが受理しなければならない。内部assertionの無効、期限切れ、対象不一致は拒否する。

### SEC-004 支払条件の正規化

金額は正の整数最小単位で扱い、浮動小数点で認可比較してはならない。通貨、network、asset、payee、scheme、期限、quote、profileをallowlist、計画上限、契約条件と照合しなければならない。

### SEC-005 Checkout変更

商品、金額、通貨、payee、期限、支払方式、network、asset、quote、profileの認可対象項目が変化した場合、以前の決済承認を再利用してはならない。新条件への再承認または安全な中断を要求する。

### SEC-006 Agent接続の固定とSSRF防御

A2A送信先はmatcherで検証済みのRPC endpointだけとし、Agent Card URLとは別フィールドで保持しなければならない。許可scheme、host／network policy、redirect制限、timeout、response size、content typeを検証し、文字列連結で一方のURLから他方を推測してはならない。

### SEC-007 Agent identityとcapabilityの固定

Registry recordとlive Agent Cardを、canonical ID、許可alias、endpoint、Card digest、skill、profile、capabilityで照合しなければならない。支払提出用signed capabilityは、信頼されたissuerが発行し、Merchantをaudienceとし、plan、step、canonical Agent、許可operation、remote task／context、期限へ限定しなければならない。未登録alias、skill不一致、endpointまたはCard差し替え、scope外operationを拒否する。

### SEC-008 外部A2A内容の不信

Agentから返るprompt、自由文、URL、支払条件、extension、Artifactを外部入力として扱い、plan逸脱、prompt injection、相関不一致、上限超過、未知profile、破損metadataを検出してfail closedとしなければならない。

### SEC-009 LLMからの権限制御分離

LLMは、計画承認済み、支払承認済み、支払可能、検査合格、完了の最終権限を持ってはならない。これらは永続状態、構造化データ、明示的policy、完全一致承認を用いた決定論的gateで判定する。

### SEC-010 秘密情報と最小開示

秘密鍵、署名seed、Firebase service credential、Bearer token、Credentialの秘密情報、完全なPayment Mandate、署名対象原文をrepository、LLM prompt、UI、通常ログ、trace、network responseへ露出してはならない。必要な表示はredact済み要約と短縮IDに限定する。

### SEC-011 障害時のfail closed

検証不能、detector異常、署名不正、期限切れ、相関不一致、結果不明時は完了または再支払を推測してはならない。timeout後は既存の冪等性キーとremote Taskを照合し、解決できなければ `REVIEW` とする。

### SEC-012 AP2 Human Present

AP2のIntent、closed Checkout、Payment Mandate、Credential、Receiptについて、発行者、署名、対象、nonce、有効期限、相関を検証し、後からoffline verificationできる証跡を保持しなければならない。

### SEC-013 x402 profile選択とsilent fallback禁止

支払transportは、Agent Cardとruntime双方が対応しwallet／facilitator等を検証できる公式profile、次いでこのデモMerchant専用の `x402-wire-simulation/1` の順で選択する。未表明、未知、不一致、検証失敗の場合は支払を行わず `BLOCKED` または `REVIEW` とし、AP2-only形式または直接railへのsilent fallbackをしてはならない。x402形式の生成または提出に失敗した場合もrailを直接呼んで迂回せず、同じ冪等性キーで安全に再試行できない限り停止する。

### SEC-014 simulation表示

`x402-wire-simulation/1` を利用する場合、UI、証跡、conformance reportは `simulation` かつ `NOT CONFORMANT` と明示しなければならない。「公式準拠」「settled on-chain」「実資産決済」と表示してはならない。

### SEC-015 Merchantの支払認可検証

本要件は二つのA2A operationを区別する。従来一文だったSEC-015を、実装済み二段階contractへ追跡可能に分割する。

- `SEC-015-A` guarantee submission受理前: Merchantはsigned simulation guarantee、signed capability、remote Task／context／order／quote相関、安全なAP2 digest要約、交渉済みprofile metadataを検証する。成功時は同一Taskを`working`で返すだけで、fulfillmentをcommitしない。
- `SEC-015-B` fulfillment commit前: Merchantは、同一Taskで受理済みの保証、仲介railが発行したsettlement receiptとdigest、settlement／guarantee／order相関を検証する。成功後だけ業務を履行して同一Taskを`completed`にする。

どちらも欠落・不一致・改ざん・期限切れを状態変更と業務副作用なしで拒否する。Merchantはraw Mandateを受け取らず、決済処理やsettlementを実行しない。

pre-payment authorization envelopeは仲介内部の証跡に限る。

simulation railは実hold／authorizeなしの同期settlementだけを記録する。

### SEC-016 従来security callbackの維持

従来のA2A呼出し前後に実行されていたsecurity callbackを、FR-010のstable anomaly gateとは別の防御層として維持し、各A2A operationの前後で実行しなければならない。置換または統合する場合は、正本変更の明示承認と、入力範囲、拒否規則、fail-closed挙動、副作用阻止が同等以上であることの比較証跡が必要であり、anomaly gateの存在だけをcallback廃止の根拠としてはならない。

### SEC-017 返金認可

返金は、元支払と同じ認証済みsubject／tenant／session、plan／step、Task／context、order、settlementへ結び付く認可済みoperationとして扱わなければならない。元支払の存在・成功、未返金残額、返金額、通貨、payee／payer、返金冪等性キーを副作用前に検証し、別主体、未settled支払、上限超過、相関不一致、重複返金をfail closedで拒否する。秘密情報非露出と監査相関は支払と同じ境界を維持する。

## 9. データと相関要件

### DATA-001 主体相関

continuationは、少なくとも `subject`、`tenant_id`、`adk_session_id`、`mediation_session_id` を保持し、全read/write/approve/resume操作で照合しなければならない。

### DATA-002 仲介計画相関

continuationとAP2 evidenceは、少なくとも `plan_id`、`plan_version`、`plan_digest`、`legacy_step_id`、`plan_approval_id`、`plan_approval_nonce`、`plan_approval_issued_at` を保持し、各AP2 objectから改ざん検知可能に参照できなければならない。

### DATA-003 選定Agent snapshot

選定時のcanonical `agent_id`、実Agent名、`agent_card_digest`、`agent_card_url`、`rpc_endpoint`、canonicalおよびA2Aの `skill_id`、`trust_score`、capability、選定時刻を変更不能なsnapshotとして相関できなければならない。

### DATA-004 remote Task相関

少なくとも `context_id`、`task_id`、`order_id`、`quote_id`、Task state、payment requirement digest、直近応答／Artifact digestを保存し、支払提出と再開の前後で完全一致を検証しなければならない。

### DATA-005 決済相関

少なくとも `payment_workflow_id`、`payment_approval_id`、`payment_approval_nonce`、`payment_approval_issued_at`、Checkout ID／digest、Intent ID、Payment Mandate ID、Credential ID、Receipt ID、各AP2 objectのnonce／issued-at／expiry、支払状態、冪等性キーをplan、step、remote Taskへ結合しなければならない。計画承認、決済承認、各AP2 objectのnonceと時刻は意味ごとに独立したfieldとして識別できなければならない。

### DATA-006 継続制御

continuationは、`continuation_id`、状態、version、作成／更新／期限時刻、retry count、last errorを保持し、compare-and-setで競合更新を検出できなければならない。

### DATA-007 識別子の正規化

セキュリティ主体のcanonical Agent IDはTrusted Registryの不変IDとし、registry名、service slug、A2A Agent Card名、registry skill、A2A skillの許可対応を型付きmappingで明示しなければならない。暗黙の文字列変換または未登録aliasの受理を禁止する。

### DATA-008 監査相関

計画承認、従来security callback、FR-010の各stable gate ID、支払要求、決済承認、AP2検証、支払提出、remote結果、final validationは、同一correlation chain上で順序、入力digest、出力digest、呼出回数、判定、actor、時刻を追跡できなければならない。

### DATA-009 返金相関

返金recordは、`refund_id`、元の `settlement_id`／Receipt ID、subject、session、plan ID／version／digest、step ID、remote task／context、order ID、通貨、元支払額、返金額、理由、状態、冪等性キー、発行時刻、返金Receipt／result digestを保持し、元支払と改ざん検知可能に相関できなければならない。

## 10. 状態遷移制約

規範上の仲介状態は、少なくとも `Discovering`、`Planning`、`WaitingForPlanApproval`、`Executing`、`WaitingForPaymentApproval`、`PaymentSubmitting`、`ResumingA2A`、`StepCompleted`、`FinalValidation`、`Completed`、`RefundPending`、`RefundSubmitting`、`Refunded`、`Cancelled`、`Blocked`、`Rejected`、`ReviewRequired` を区別する。

### STATE-001 計画承認前

`WaitingForPlanApproval` からは、完全一致の有効な計画承認によってのみ `Executing` へ進み、拒否時は `Cancelled` へ進む。

### STATE-002 A2A実行分岐

`Executing` は、支払要求なしの有効な完了応答で `StepCompleted`、検証済み `payment-required` で `WaitingForPaymentApproval`、security違反で `Blocked` または `ReviewRequired` へ進む。

### STATE-003 決済承認待ち

`WaitingForPaymentApproval` は、完全一致の有効な決済承認によってのみ `PaymentSubmitting` へ進み、拒否または期限切れでは支払わず `Cancelled`、`Blocked`、または再計画可能な停止状態へ進む。

### STATE-004 支払提出

`PaymentSubmitting` は、AP2、profile、相関、冪等性、anomaly gateの成功後のみ `ResumingA2A` へ進む。結果不明は `ReviewRequired` とする。

### STATE-005 同一Task再開

`ResumingA2A` は、保存済みの同一remote Taskへの後続messageと相関一致を確認した場合のみ `StepCompleted` へ進む。相関不一致または結果不明は `Blocked` または `ReviewRequired` とする。

### STATE-006 複数step

`StepCompleted` は、未実行stepがあれば次の `Executing` へ進み、全step終了時だけ `FinalValidation` へ進む。

### STATE-007 最終判定

`FinalValidation` は、`ACCEPT` でのみ `Completed`、`REJECT` で `Rejected`、`REVIEW` で `ReviewRequired` へ進む。

### STATE-008 再計画

計画または認可対象条件が変わる再計画ではplan versionとdigestを更新し、旧計画承認と旧決済承認を新しいplanまたはCheckoutへ流用してはならない。

### STATE-009 非同期待機

承認待ち、worker待ち、結果照合待ちでは状態を保存して利用者へ応答し、単一のUI／ADK requestを長時間保持することを前提としてはならない。

### STATE-010 禁止遷移

次の遷移を禁止する。

- `WaitingForPlanApproval` から `PaymentSubmitting`
- `Executing` から決済承認なしの `PaymentSubmitting`
- `WaitingForPaymentApproval` から無関係な新規A2A Task開始
- `StepCompleted` から `FinalValidation` を経ない `Completed`
- 無料stepにおけるPayment Mandateまたはsettlement生成
- `BLOCK`、`REVIEW`、detector失敗後の自動継続

### STATE-011 返金状態

支払成功済みの対象だけを `RefundPending` へ進め、認可・相関検証成功後に `RefundSubmitting`、返金成功時に `Refunded` へ進めなければならない。未settled支払からの返金、返金済み金額を超える返金、別Task／orderへの遷移を禁止する。結果不明時は成功と推測せず `ReviewRequired` とする。

## 11. 公開UI・HTTP境界要件

### UI-001 認証後の入口

未認証利用者はFirebase認証へ誘導され、認証後は `payment_user_agent` が選択済みのADK Webへ到達しなければならない。利用者に内部Agentの再選択を要求してはならない。

### UI-002 計画承認表示

第一承認の前に、plannerが生成した実際の計画、対象planの識別情報、各stepの選定Agent、条件と上限、`承認` の完全一致要件を表示しなければならない。

### UI-003 決済承認表示

第二承認の前に、商品、正確な金額と通貨、payee、期限、支払方式、対象step／Taskの安全な短縮ID、第一承認とは別であること、`承認` の完全一致要件を表示しなければならない。

### UI-004 実trace

FR-014の全段階、実agent名、短縮相関ID、順序または時刻、gate結果を利用者が識別できなければならない。固定文だけで実行済みに見せたり、人為的な遅延を入れたりしてはならない。

### UI-005 安全なエラー

拒否、期限切れ、profile未対応、不一致、状態消失、結果不明、`BLOCKED`、`REVIEW` を成功と区別し、利用者が次の安全な操作を判断できる文言を返さなければならない。

### UI-006 simulation表記

ローカルsimulationの承認画面、処理中表示、完了または停止表示は、`simulation` と `NOT CONFORMANT` を常に認識可能な形で示さなければならない。

### UI-007 機密情報非表示

UIとブラウザnetwork responseに、token、秘密鍵、内部credential、完全なmandateまたは署名原文を含めてはならない。

### UI-008 デモ依頼

正式なデモ手順とpromptは `docs/payments/DEMO.md` を正本とし、promptは固定決済workflowを直接操作する命令ではなく、条件と上限を含む通常の仲介依頼として解釈できなければならない。画面は計画承認と決済承認が別の行為であることを明示する。

### UI-009 返金表示

返金対象、元支払の安全な短縮ID、返金額・通貨、理由、状態、返金結果を表示し、支払成功と返金済みを区別しなければならない。token、credential、完全な署名原文を表示せず、重複または不正な返金要求は安全なエラーとして示す。

### HTTP-001 公開app一覧

`/list-apps` または同等の一覧は `payment_user_agent` だけを返さなければならない。

### HTTP-002 認証必須面

UI、許可された公開WebSocket経路、必要なsame-origin APIは、未認証時に認証へ安全に誘導するか401／403で拒否しなければならない。

### HTTP-003 Store非公開

外部からの `/store`、`/store/`、`/store/sse/`、`/store/health`、`/api`、`/api/`、`/ws`、`/ws/` は、認証状態にかかわらず404としなければならない。matcherはcontainer内loopback endpointのみを使う。

### HTTP-004 A2Aと内部APIの非公開

外部からの `/a2a`、`/a2a/`、`/v1`、`/v1/`、`/internal`、`/internal/`、workflow API、Merchant API、identity broker、旧payment routeは、exact pathとprefixの両方で404としなければならない。Merchant A2Aはcontainer内loopback endpointだけで利用する。

### HTTP-005 identity header偽造防止

外部identity headerを用いた認証迂回を拒否し、proxyが検証後に作成したassertionだけを上流へ渡さなければならない。

### HTTP-006 許可routeの限定

外部公開は認証入口、必要なhealth、認証後UI、必要なsame-origin mediation routeの明示allowlistに限定し、上流の偶発的404や認証redirectを非公開化の代用としてはならない。

### HTTP-007 返金経路の公開境界

返金操作は、Firebase認証済みUIから必要なsame-origin mediation routeを通る場合だけ受理しなければならない。Merchant、rail、workflow、workerの返金用内部endpointを外部公開せず、外部identity headerまたは内部URLの直接呼出しによる返金を拒否する。

## 12. 運用・ephemeral境界要件

### OPS-001 固定Cloud Run対象

Cloud Runの更新対象は、project `gen-lang-client-0585901015`、region `asia-northeast1`、service `payment-user-agent-demo` の完全一致に限定しなければならない。他のCloud Runサービス、revision、traffic、設定を変更してはならない。

### OPS-002 Cloud SQL禁止

本デモのためにCloud SQLまたは他の外部永続DBを新設、接続、変更してはならない。

### OPS-003 ephemeral仕様

Cloud Run instanceの置換、scale down、revision更新時に、instance内SQLite、outbox、生成key、workflow状態が消失してよい。これを耐久性のある運用と表示または主張してはならない。

### OPS-004 同一instance内回復

同一Cloud Run instance内でworkflow API、worker、Merchant等の子processだけが再起動した場合、残存するSQLite、outbox、冪等性キー、remote Taskからreconciliationを試みなければならない。

### OPS-005 状態消失時の扱い

instance置換等で状態が失われた場合、過去の処理を成功と推測せず、「デモ状態が失われたため再実行が必要」と明示し、古い承認やworkflowの再利用を拒否しなければならない。

### OPS-006 loopback境界

Trusted Agent Store、Merchant、workflow API、worker、identity brokerは、必要に応じて同一containerのprocessとして実行できるが、相互通信をloopbackに限定し、外部listenまたは公開proxy routeに依存してはならない。

### OPS-007 更新専用手順

既存serviceを新revisionへ更新する手順は、新規作成専用手順と分離または明示modeで区別し、対象project／region／serviceを固定しなければならない。任意service名を外部入力として受ける曖昧な更新を禁止する。

### OPS-008 デプロイfail-closed guard

更新前後に、immutable image digest、candidate artifact、registry digest、対象service完全一致、更新revision image、readiness、traffic、ephemeral環境、rollbackに必要な旧revisionを検証し、不一致時は更新または完了判定を停止しなければならない。

### OPS-009 認証とmodel実行環境

Firebase認証設定と認証後redirectを維持し、`secure_mediator` が使用するmodelの認証、project、location、IAM、quota、timeoutをrelease前に検証しなければならない。credentialをimage、repository、command output、PR、文書へ含めてはならない。

## 13. テスト要件

### TEST-001 unit: 支払要求

A2A Task state、extension、profile、金額、通貨、payee、期限、quoteの正規化と、自由文による偽陽性防止をunit testで検証しなければならない。

### TEST-002 unit: 相関と識別子

主体、plan、step、Agent snapshot、Card digest、remote Task、payment requirement、workflow、AP2 evidenceの相関と、canonical ID／許可aliasの正規化をunit testで検証しなければならない。FR-008の全最低必須fieldについて、一致caseと各fieldの不一致caseを検証し、計画承認nonce／issued-at、決済承認nonce／issued-at、各AP2 objectのnonce／issued-at／expiryを取り違えないこと、およびoffline verifierが外部DBの暗黙知なしに署名連鎖とfield一致を判定できることをassertしなければならない。

### TEST-003 unit: 承認と状態

二つの承認の分離、単一text partの完全一致 `承認`、計画／Checkout変更時の失効、状態遷移、禁止遷移、通常の重複送信に対する基本冪等性をunit testで検証しなければならない。FR-007のrouting決定表について、payment 1件＋plan 1件はpaymentだけ、payment 1件＋plan複数はpaymentだけ、payment複数は拒否、payment 0件＋plan 1件はplanだけ、payment 0件＋plan複数は拒否、pending 0件は通常依頼、別subject／tenant／ADK session／mediation sessionの混在と期限切れは候補除外、という全caseをassertしなければならない。高競合な並行承認はTEST-017へ分離する。

### TEST-004 unit: 支払policy

正常系の計画上限、正の整数最小単位、通貨、network、asset、payee、scheme、期限、profile選択、silent fallback禁止と、代表的な不正条件のfail-closedをunit testで検証しなければならない。価格変更、期限切れ、quote変更の全分岐matrixはTEST-018へ分離する。

### TEST-005 unit: security

subject／session分離、signed capabilityのscope、SSRF、redirect、Card／endpoint差し替え、detector異常のfail closed、secret redactionをunit testで検証しなければならない。

### TEST-006 integration: 実仲介chain

実際の `secure_mediator` 構成で、`matcher`、`planner`、計画承認gate、`orchestrator`、loopback A2A HTTP、従来security callback、FR-010の各stable gate、final validationまでを実行しなければならない。有料の単一stepでは `PRE_A2A_START`、`POST_A2A_RESPONSE`、`POST_PAYMENT_REQUIREMENT`、`PRE_PAYMENT_SUBMIT`、`POST_PAYMENT_RESULT` がこの順で各1回、無料の単一stepでは最初の2つだけが各1回で残り3つが0回であることを機械的にassertする。各gateの入力schema／digest、`PASS`後の許可副作用、`BLOCK`／`REVIEW`／timeout／parse failure後の禁止副作用と、A2A operation前後の従来security callback呼出しをassertしなければならない。test doubleまたはtrace labelだけで仲介統合済みとしてはならない。

### TEST-007 integration: 有料と無料

無料経路では決済recordなしの完了を、有料経路では初回 `payment-required`、決済待ち、第二承認、同じTaskへの支払提出、同じstepの再開を検証しなければならない。

### TEST-008 integration: HTTP相関

matcherのAgent ID、Card digest、RPC endpoint、skill、trustがplan stepと一致し、そのRPC endpointへの初回Task requestが一回だけで、支払後は同一 `contextId`、`taskId`、`orderId`、`quoteId` の後続messageであることを機械的にassertしなければならない。実HTTP wireに、scope限定signed capability、signed simulation guarantee、Task相関、安全なAP2 digest要約とprofile metadataが存在することをassertする。欠落・改ざん・不一致ごとにMerchantのTask状態変更とfulfillmentが0件であること、Merchant側のrail呼出し／settlement実装が存在しないことをassertする。仲介側simulation railの副作用counterは別に検証する。

### TEST-009 integration: 異常と障害

拒否、期限切れ、価格変更、profile未対応／破損、悪意ある応答、timeout、結果不明、replay、並行承認、worker障害を検証し、二重副作用と検査迂回がないことをassertしなければならない。

### TEST-010 regression

既存の決済、AP2、A2A、security、outbox、reconciliation、restart試験と、従来仲介の主要試験を維持しなければならない。旧構造を前提とする試験は新要件へ更新し、単に削除してはならない。

### TEST-011 実ブラウザ

ローカルcontainerとCloud Runの両方で、Firebase認証、`payment_user_agent` 自動選択、自然文依頼、全仲介段階、第一の `承認`、決済条件、第二の `承認`、同一Task再開、完了、final validation、返金正常系と `Refunded` 表示、再読込後の安全な表示、機密非露出を実ブラウザで確認しなければならない。

### TEST-012 公開境界black-box

未認証と認証済みの両方で、公開allowlist、内部routeのexact/prefix 404、WebSocket認証、app一覧、identity header偽造拒否、same-origin必要routeのみの成功をblack-boxで検証しなければならない。

### TEST-013 restart

restart testは次のcheckpoint別caseを分け、各caseで初期record、再起動対象process、期待record数、期待state、期待外部call数を明示しなければならない。

- 計画承認待ち: 同一instanceの子process再起動前後でMerchant Taskと支払副作用が0件、同じ計画承認待ちrecordが復元され、有効承認後のTask開始が1件だけであること
- 決済承認待ち: 既存task／context、payment requirement、continuationが復元され、決済承認前の支払提出とsettlementが0件であること
- 支払後またはoutbox lease中: 同じtask／contextと冪等性キーで照合し、Task開始、支払提出、settlementの成功件数を増やさずにreconciliationすること
- instance置換またはrevision更新: ephemeral状態消失を許容し、古いcontinuation／workflowを成功扱いまたは再利用せず、所定の再実行案内を返すこと

### TEST-014 release artifact

配布する正確なimage内で全suiteを実行し、source digest、image digest、revision、test result、browser evidence、conformance reportを相互に結合しなければならない。

### TEST-015 要件coverage

対象candidateの要件適合ledgerについて、19.3 forward traceability matrixのID列と本書の規範要件見出しIDの集合が完全一致し、全規範IDが一回ずつ列挙されていることを自動検査しなければならない。19.3の `Release-1必須` 集合と `設計のみ・将来課題` 集合の和集合が全規範ID、積集合が空であることも検査する。各ledger行は19.3の適用区分、設計・実装責務、試験ID／判定規則、AC、必要証跡、candidate digest、許可status、判定時刻、判定者へ結合されなければならない。Release-1必須IDの欠落、未知ID、重複ID、証跡なしまたは `PASS` 以外、Release-1期限のOQ未解決が一つでもあればrelease closureを失敗させる。将来課題IDは `DESIGNED`、`PARTIAL`、`NOT RUN` であっても、既知課題、設計参照、実装triggerが記録されていればRelease-1 closureを阻止しない。

### TEST-016 返金integration

成功済み支払に対する相関付き全額返金正常系を実行し、元subject／session、plan／step、Task／context、order、settlement、Receipt、金額、通貨、返金冪等性キーの一致、返金Receipt、UI状態をassertしなければならない。同じ返金要求の再送は返金成功件数を増やさず、未settled支払、別subject／session／order、元支払額超過を副作用0件で拒否することを検証する。

### TEST-017 高度競合・recovery

高競合な並行承認、初回A2A応答喪失、複数段retry、outbox lease競合、子process再起動、Merchant成功後の仲介更新失敗を対象に、FR-017の一意成功とreconciliationを検証しなければならない。本試験は設計のみ・将来課題に分類できる。

### TEST-018 価格・期限・悪意入力の拡張matrix

価格、通貨、payee、期限、quoteの変更組合せ、悪意あるA2A入力の網羅matrix、DNS再束縛等の高度SSRF、長期運用時の境界caseを検証しなければならない。本試験は基本的な正常系検証とfail-closedを置き換えず、設計のみ・将来課題に分類できる。

## 14. テスト可能な受入条件

### AC-001 有料タスクの正常系

自然文依頼から `secure_mediator -> matcher -> planner -> 計画承認gate -> orchestrator` が同一traceで実行され、承認前のMerchant callが0件であること。第一の完全一致 `承認` 後にTask開始requestを一回送り、第二の完全一致 `承認` 前の支払副作用を0件とする。承認後、決定論的workflowがAP2 evidenceとpre-payment envelopeを内部生成・検証し、仲介payment authorityがsigned simulation guaranteeを発行する。同一Taskへの後続messageはraw Mandateでなく保証、scope限定capability、Task相関、安全なAP2 digest要約を運ぶ。Merchantが検証して`working`を返した後、仲介railが実holdなしのsimulationを処理し、settlement receipt付きcommitを送る。Merchantはreceiptを検証して業務を履行し、決済／settlementを行わず同一Taskを完了する。security callbackとstable gateを定めた順に通し、final validation後だけ結果を表示する。

### AC-002 無料タスク

実matcher、planner、計画承認、orchestrator、A2A前後の従来security callback、`PRE_A2A_START` と `POST_A2A_RESPONSE`、final validationを通り、matcher snapshot、plan step、実HTTP送信先が一致すること。`POST_PAYMENT_REQUIREMENT`、`PRE_PAYMENT_SUBMIT`、`POST_PAYMENT_RESULT` の呼出しが0件であること。支払要求がなければ決済UIを出さず、Payment Mandate、payment workflow、settlement recordの件数が0であること。

### AC-003 計画拒否

第一承認を拒否または不一致入力とした場合、Merchant A2A call、Checkout、Payment Mandate、payment workflow、settlement、その他外部副作用の件数が0であること。

### AC-004 決済拒否

計画承認後に第二承認を拒否または不一致入力とした場合、Payment Mandate、支払提出、settlementの件数が0であり、対象stepが明示的な中断または取消理由を持つこと。

### AC-005 価格変更・期限切れ

quote金額、通貨、payee、期限その他のclosed Checkout項目を変更または期限切れにすると、以前の決済承認が拒否され、新条件の再承認待ちまたは安全な停止となり、旧条件で支払われないこと。

### AC-006 基本replay・routing

同じ `承認` の通常の連打または再送でもsettlementと支払提出が一回だけであり、異なるsubjectまたはsessionから同じcontinuationを承認できないこと。FR-007のrouting決定表の全caseで対象が一意になり、payment pending 1件とplan pending 1件または複数の併存ではpayment 1件だけに適用され、同種pending複数では拒否され、pending 0件では通常依頼になること。別subject／tenant／ADK session／mediation sessionのrecordと期限切れrecordが候補から除外され、計画承認と決済承認の相互誤適用が0件であること。高競合な並行requestはAC-015へ分離する。

### AC-007 Merchant障害

timeout、接続断、応答喪失時に新規Taskまたは二重支払を作らず、同じ冪等性キーとremote Taskを照合し、回復不能なら `REVIEW` を返すこと。

### AC-008 悪意あるA2A応答

自由文だけの支払指示、外部URL誘導、plan外Agent、上限超過、未知profile、壊れたextension、Card／endpoint／skill／相関の不一致を投入すると、支払副作用なしで `BLOCKED` または `REVIEW` となり、anomaly gateを迂回しないこと。

### AC-009 最終異常検知

有料・無料の双方でfinal validationを未実行、失敗、timeout、`REJECT`、`REVIEW`にすると、最終成功を返さないこと。semantic anomaly sub-agentの明示呼出し有無ではなく、実callback／Judge／決定論的validatorの判定と順序を検証する。

### AC-010 UI階層と認証

未認証のデモURLはFirebase認証へ移り、認証後は `payment_user_agent` 選択済みであること。公開app一覧に `secure_mediator`、決済workflow、内部Agentが存在せず、内部Agentを利用者が選び直さず自然文依頼を開始できること。

### AC-011 再起動とephemeral境界

計画承認待ちの子process再起動ではTaskが0件のまま承認待ちを復元し、承認後に1件だけ開始すること。決済承認待ちでは同じtask／contextと条件を復元し、承認前の支払提出が0件であること。支払後またはoutbox lease中では同じtask／contextと冪等性キーでreconciliationし、Task開始とsettlementを増やさないこと。instance置換またはrevision更新後は状態消失を許容するが、古いworkflowを成功扱いせず再実行案内を返すこと。Cloud SQLその他の外部永続DBを利用していないこと。

### AC-012 x402 profile分岐

公式profileはAgent Cardとruntimeの必要条件を満たす場合だけ選ばれること。デモMerchantでは `x402-wire-simulation/1` が選ばれ、UIと証跡に `simulation` と `NOT CONFORMANT` が表示されること。未表明または対応profileなしは `PAYMENT_PROFILE_UNAVAILABLE` としてAP2 Payment Mandateと支払副作用なしで停止し、表明済みprofileの破損または不一致は `PAYMENT_PROFILE_INVALID` として `BLOCKED` になること。いずれもAP2-onlyまたは直接railへfallbackしないこと。

### AC-013 公開HTTP境界

app一覧は `payment_user_agent` のみを返すこと。未認証UI、WebSocket、same-origin APIは認証誘導または401／403になること。HTTP-003とHTTP-004の全exact/prefix routeは、認証状態にかかわらず外部から404になること。偽造identity headerは破棄され、内部StoreとMerchantはloopbackでのみ利用されること。

### AC-014 返金正常系

有料正常系の支払成功後にfulfillment不能または認証済み利用者のデモ返金要求を発生させると、元支払と同じsubject／session、plan／step、Task／context、order、settlement、Receiptへ相関した全額返金が一回だけ成功し、返金Receiptと `Refunded` が表示されること。同じ要求の再送で返金件数が増えず、未settled支払、別主体・別order、元支払額超過では返金副作用が0件であること。

### AC-015 高度な並行承認

同一stepへ高競合な並行承認を送っても一件だけが成功し、Task開始、settlement、返金の成功件数がそれぞれ認可済み回数を超えないこと。本条件は設計のみ・将来課題に分類できる。

## 15. デリバリー工程と変更管理要件

### PRC-001 既存変更の保護

後続工程の開始時にbranch、PR、worktree差分を確認し、ユーザーまたは別作業の未commit変更を上書きしてはならない。

### PRC-002 現行と置換前挙動の基準化

実装前に、置換前の仲介責務と現行決済資産を読み取り、復元対象と再利用対象を試験可能な基準線として確定しなければならない。旧仲介コードを無条件に巻き戻して、曖昧承認またはLLM主導の権限判断を再導入してはならない。

### PRC-003 縦切りの順序

後続実装は、公開rootから実matcher／planner、厳密な計画承認、remote A2A開始、無料完了および有料の支払待ちまでを先に成立・検証し、その後に既存決済workflowを内部サブフローとして接続しなければならない。

### PRC-004 中心経路の完成順

同一Taskへの支払提出と同一step再開を成立させてから、Release-1必須の拒否、基本replay、返金を完成させなければならない。期限切れの全分岐、高度recovery、悪意入力の網羅matrixは、設計と既知課題を記録して将来工程へ送ることができる。

### PRC-005 検証の順序

全自動試験と独立レビューの重大指摘を解消した後に、ローカルcontainerの実ブラウザ試験を行い、その成功後にのみCloud Run candidateを更新対象へ反映しなければならない。

### PRC-006 リリース更新の順序

Cloud Run更新は、`linux/amd64` のimmutable candidate、固定対象update手順、更新後revision／traffic検証、Cloud Run上の実ブラウザ試験の順で行わなければならない。

### PRC-007 文書とPR

日本語文書、デモプロンプト、conformance reportを実装事実と証跡に一致させ、PRはdraftではない通常PRとしてテスト証跡と既知課題を記載しなければならない。完了報告用にPR URL／番号、base／head、head commit SHA、draft状態、candidate／Cloud Run revisionとの対応を取得できなければならない。

## 16. リリース条件と必要証跡

### REL-001 仲介統合

19.3で `Release-1必須` に分類した規範IDをすべて満たし、その集合に関係する期限到来済みOQを解決するまでは、「従来の `secure_mediator` へのRelease-1決済統合完了」と判断してはならない。`設計のみ・将来課題` のIDは削除せず、設計参照、既知課題、実装trigger、許可statusをledgerへ記録する。18章の範囲外事項と将来課題は、Release-1の正常系安全境界または表示・主張制約を適用除外する理由にしてはならない。

### REL-002 自動試験

19.3で `Release-1必須` とされた行が要求するunit、integration、regression、security、paid／free正常系、refund、browser、公開境界black-box、release artifact、要件coverageの全suiteと全必須caseの集合が、配布candidate imageと結び付いた証跡で成功しなければならない。将来課題のTESTは未実行でもreleaseを阻止しないが、`DESIGNED`、`PARTIAL`、`NOT RUN` のstatusと既知課題を記録する。Release-1必須行の試験または判定規則を、他suiteの成功で代替してはならない。

### REL-003 独立レビュー

独立コードレビューと独立試験を実施し、重大または高優先度の未解決指摘を残したままリリースしてはならない。

### REL-004 実ブラウザ

ローカルcontainerとCloud Runの双方でTEST-011を満たし、Cloud RunではFirebase認証を含むend-to-end経路を確認しなければならない。

### REL-005 deploy対象と永続性

Cloud SQLを追加しておらず、`payment-user-agent-demo` 以外のCloud Runサービスを変更しておらず、固定対象update手順で新revisionを反映したことを証明しなければならない。

### REL-006 外部仕様の一次資料再確認

設計確定前およびリリース前に、AP2とA2A x402の公式一次資料でversion、必須field、Task state、extension、署名・相関要件を再確認しなければならない。versionを暗黙に変更してはならず、更新する場合は固定versionと互換性差分を文書化し、該当要件と試験を再評価する。

### REL-007 適合文書

`docs/payments/AP2.md`、`docs/payments/A2A_X402.md`、`docs/ap2_x402_conformance_report.json` を実装後に更新し、各適合項目を根拠付きで `PASS`、`PARTIAL`、`NOT RUN`、`NOT CONFORMANT` のいずれかに評価しなければならない。

### REL-008 必須trace証跡

実matcher、planner、二つの承認gate、orchestrator、Merchant A2A、従来security callback、`PRE_A2A_START`、`POST_A2A_RESPONSE`、`POST_PAYMENT_REQUIREMENT`、`PRE_PAYMENT_SUBMIT`、`POST_PAYMENT_RESULT`、決済workflow、有料／無料分岐、同一Task再開、final validation、返金正常系を機械的に追跡できる証跡を保存しなければならない。各stable gateについて入力schema／digest、呼出回数、判定、理由、判定前後の副作用件数を記録しなければならない。

### REL-009 必須副作用・相関証跡

計画承認前と決済承認前の副作用0件、Task開始1件、支払提出1件、remote IDsとlegacy stepの連続性を証明しなければならない。支払wire上のsigned capabilityと交渉済みprofileの必須extension header／metadataの存在、scope一致、Merchant検証結果、欠落・改ざん時の副作用0件を証明しなければならない。offline verifierが外部DBの暗黙知なしにFR-008の全最低必須field、各承認のnonce／issued-at、各AP2 objectのnonce／issued-at／expiry、署名連鎖の一致と不一致を判定できる証跡を保存しなければならない。

### REL-010 Cloud Run証跡

対象service、URL、revision、immutable image digest、source digest、traffic、ephemeral環境、Firebase認証後のapp選択、black-box結果、他service非変更を記録しなければならない。

### REL-011 文書と主張

日本語文書、デモ手順、PR説明、既知課題は、対象revisionの実装事実と証跡に一致しなければならない。PRはdraftではない通常PRでなければならず、完了報告へPR URL／番号、base／head、head commit SHA、`draft=false`、対象candidate image digest、Cloud Run revisionとの対応、既知課題を記録しなければならない。既存の直接workflow版の証跡を新しい仲介統合の証跡として流用してはならない。

### REL-012 リリース判定

全規範IDを適合ledgerへ一回ずつ登録し、`Release-1必須` の全IDが証跡付き `PASS`、関連する期限到来済みOQが解決済みである場合だけRelease-1を完了としてよい。Release-1必須IDに `PARTIAL`、`NOT RUN`、`NOT CONFORMANT`、証跡なし、重複、欠落、未知IDが一つでもある場合は完了としてはならない。将来課題IDは、設計参照、既知課題、実装triggerを伴う `DESIGNED`、`PARTIAL`、`NOT RUN` を許容する。`BLOCKED` と `REVIEW` は実行状態または運用判定として併記できるが、要件適合statusの代用にはしない。

### REL-013 全規範IDの適合ledger

対象candidateごとに、本書の全規範IDを一回ずつ列挙した適合ledgerを作成し、各行へ `requirement_id`、19.3の適用区分、試験ID／判定規則、AC、必要証跡、candidate source／image digest、判定者、判定時刻、statusを記録しなければならない。Release-1必須IDのstatusは `PASS`、`PARTIAL`、`NOT RUN`、`NOT CONFORMANT` のいずれか、将来課題IDはこれらに `DESIGNED` を加えた集合から一つとする。simulation自体の外部仕様評価が `NOT CONFORMANT` であっても、SEC-014等の「正しくNOT CONFORMANTと表示する要件」は、その表示と制約を証明できれば要件statusを `PASS` とする。規範IDに適用外statusを設けず、範囲外事項は18章とCLAIM-003の制約として評価する。

## 17. 主張管理

### CLAIM-001 現時点で許される主張

新要件の証跡が揃う前に主張してよいのは、決定論的な二段階承認とAP2 Human Present demoの基盤、A2A Task上のpayment wire shapeを使うローカルsimulation、固定決済workflow単体のFirebase認証付きCloud Run実演までとする。

### CLAIM-002 現時点で禁止する主張

新要件の証跡が揃う前に、従来 `secure_mediator` への統合完了、実matcher／planner／orchestrator／各anomaly detectorの通過、公式A2A x402準拠、on-chain settlement、実資産決済、現行revisionの本書適合を主張してはならない。

### CLAIM-003 リリース後の限定

本書のリリース条件を満たした後も、simulation利用時は公式x402準拠または実資産決済を主張してはならず、範囲外事項を既知制約として併記しなければならない。

## 18. 範囲外

次は今回のリリース範囲外とする。ただし、範囲外であることを文書とPRに明記する。

- 公式A2A x402 profileとの完全な相互運用
- wallet、facilitator、実network、実asset、on-chain settlement
- production向けKMS／HSM、KYC／AML、PCI／SCA
- 複数Cloud Run instance間の永続状態共有
- 長期監査DB
- Human Not Presentまたは自律購入
- 全Merchant、全通貨、全決済方式への一般化

これらが範囲外であることを理由に、従来仲介ルートの実経由、二段階の完全一致 `承認`、同一remote Taskと同一stepの再開、AP2相関、各anomaly gate、final validation、公開境界を省略してはならない。

## 19. 要件トレーサビリティ

### 19.1 引継ぎ文書の機能要件・受入条件

| 引継ぎ正本 | 本書の対応要件 | 補助要件 |
| --- | --- | --- |
| FR-001 | FR-001 | TEST-006、AC-001、REL-001 |
| FR-002 | FR-002 | UI-001、HTTP-001、AC-010 |
| FR-003 | FR-003 | DATA-003、TEST-008、AC-001／002 |
| FR-004 | FR-004 | STATE-001、TEST-003、AC-003 |
| FR-005 | FR-005 | TEST-001／004、AC-012 |
| FR-006 | FR-006 | DATA-001〜006、STATE-002〜005、TEST-007 |
| FR-007 | FR-007 | UI-002／003、TEST-003、AC-004／005 |
| FR-008 | FR-008 | SEC-012、DATA-002／005、REL-009 |
| FR-009 | FR-009 | DATA-004、STATE-005、TEST-008、AC-001 |
| FR-010 | FR-010 | SEC-008／009／011、TEST-009、AC-008 |
| FR-011 | FR-011 | STATE-007、AC-009、REL-008 |
| FR-012 | FR-012 | TEST-007、AC-002 |
| FR-013 | FR-013／017 | STATE-009、OPS-004／005、AC-006／007／011／015 |
| FR-014 | FR-014 | UI-004、REL-008 |
| FR-015 | FR-015 | HTTP-001〜006、OPS-001〜008、AC-010／011／013 |
| AC-001 | AC-001 | FR-001、FR-003〜011、SEC-013／014、REL-008／009 |
| AC-002 | AC-002 | FR-001〜005、FR-010〜012 |
| AC-003 | AC-003 | FR-004、STATE-001 |
| AC-004 | AC-004 | FR-007、STATE-003 |
| AC-005 | AC-005 | SEC-005、STATE-008 |
| AC-006 | AC-006／015 | FR-013／017、SEC-002、DATA-006 |
| AC-007 | AC-007 | FR-013、SEC-011 |
| AC-008 | AC-008 | FR-005／010、SEC-006〜009 |
| AC-009 | AC-009 | FR-011、STATE-007 |
| AC-010 | AC-010 | FR-002、UI-001、HTTP-001 |
| AC-011 | AC-011 | FR-013／015、OPS-002〜005、TEST-013 |
| AC-012 | AC-012 | SEC-013／014、UI-006、TEST-004 |
| AC-013 | AC-013 | HTTP-001〜006、TEST-012 |

### 19.2 引継ぎ文書の章への逆引き

| 引継ぎ正本の章 | 本書の対応要件・章 |
| --- | --- |
| 8. 非機能・セキュリティ要件 | NFR-001〜004、SEC-001〜017、DATA-001〜009、HTTP-002〜007 |
| 12. テスト要件 | TEST-001〜018、AC-001〜015 |
| 15. 実装順序 | PRC-001〜007 |
| 16. リリース条件 | REL-001〜013 |
| 17. デプロイ前後の扱い | OPS-001〜009、PRC-006、REL-005／010 |
| 18. 既知の範囲外事項 | 本書18章、CLAIM-003 |
| 19. 現在主張してよいこと・いけないこと | CLAIM-001〜003、REL-011〜013 |
| 20. 作業開始時の確認 | PRC-001／002 |
| 21. 完了報告に必ず含める証跡 | REL-002〜013、TEST-015／016 |
| 22. 判断に迷った場合の原則 | 本書1.1・2章の優先順位、FR-005／016、SEC-009／013／017、FR-014、OPS-002、REL-001／012／013 |

### 19.3 全規範IDのforward traceability matrix

次表のID列は本書の139個の規範要件見出しと一対一で対応する。適用区分は次の集合で一意に定める。

- `設計のみ・将来課題`（13件）: FR-017、STATE-008、OPS-004、OPS-005、TEST-009、TEST-013、TEST-017、TEST-018、AC-005、AC-007、AC-008、AC-011、AC-015
- `Release-1必須`（126件）: 上記13件を除く、次表の全規範ID

高度restart／recovery、初回応答喪失、複雑retry／concurrency、DNS再束縛等の高度SSRF、価格変更／期限切れの全分岐、悪意入力matrix網羅、長期運用edge caseは将来課題側の設計・試験で追跡する。SEC-006の基本接続先固定、SEC-008のfail-closed、SEC-011の結果不明時の成功禁止はRelease-1必須のままとする。対象candidateのledgerは各行を適用区分、設計・実装責務、試験または判定規則、AC、必要証跡へ結ぶ。release closureはREL-012／013に従い、Release-1必須の全行が証跡付き `PASS` である場合だけ成立する。

| 規範ID | 設計・実装責務 | 試験ID／判定規則 | AC／判定規則 | 必要証跡 | status／release |
| --- | --- | --- | --- | --- | --- |
| FR-001 | 実仲介実行graph | TEST-006 | AC-001／002 | 相関trace | 4値・REL-012／013 |
| FR-002 | 単一公開root | TEST-011／012 | AC-010／013 | app一覧・browser記録 | 4値・REL-012／013 |
| FR-003 | Agent snapshotから実送信先へのbinding | TEST-002／008 | AC-001／002 | plan・request照合 | 4値・REL-012／013 |
| FR-004 | 計画承認gate | TEST-003／007 | AC-003 | 承認record・副作用件数 | 4値・REL-012／013 |
| FR-005 | 構造化payment-required分岐 | TEST-001／004／007 | AC-001／002／012 | Task・extension検証記録 | 4値・REL-012／013 |
| FR-006 | step停止・continuation再開 | TEST-003／007／013 | AC-001／006／011 | continuation・state履歴 | 4値・REL-012／013 |
| FR-007 | 二承認と一意routing | TEST-003 | AC-004／006 | routing全case・承認record | 4値・REL-012／013 |
| FR-008 | AP2全field binding | TEST-002 | AC-001 | offline verification結果 | 4値・REL-012／013 |
| FR-009 | 同一Task・認可付き支払wire | TEST-008 | AC-001 | HTTP wire・Merchant検証 | 4値・REL-012／013 |
| FR-010 | stable anomaly gate | TEST-006／009 | AC-001／002／008 | gate順序・回数・副作用 | 4値・REL-012／013 |
| FR-011 | final validation強制 | TEST-006／007 | AC-009 | final判定trace | 4値・REL-012／013 |
| FR-012 | 無料経路の決済非生成 | TEST-007 | AC-002 | record件数・trace | 4値・REL-012／013 |
| FR-013 | 基本冪等性・二重支払／返金防止 | TEST-003／008／016 | AC-006／014 | 冪等性key・副作用件数 | 4値・REL-012／013 |
| FR-014 | 実trace表示 | TEST-006／011 | AC-001／002 | UI・監査trace | 4値・REL-012／013 |
| FR-015 | デモ運用境界 | TEST-012／013／014 | AC-010／011／013 | route・deploy証跡 | 4値・REL-012／013 |
| FR-016 | 相関付き全額返金正常系 | TEST-016 | AC-014 | refund record・Receipt・UI | 4値・REL-012／013 |
| FR-017 | 高度競合・retry・recovery | TEST-013／017 | AC-007／011／015 | recovery設計・既知課題 | DESIGNED／PARTIAL／NOT RUN・non-blocking |
| NFR-001 | 非block型進捗と人工遅延禁止 | TEST-007／011 | AC-001／002 | timing・UI記録 | 4値・REL-012／013 |
| NFR-002 | 決定的認可・遷移 | TEST-003／004／006 | AC-003〜009 | 再現試験結果 | 4値・REL-012／013 |
| NFR-003 | 相関監査 | TEST-002／006／014 | AC-001／002／009 | candidate結合trace | 4値・REL-012／013 |
| NFR-004 | 外部通信制限 | TEST-005／009 | AC-007／008 | timeout・size・redirect結果 | 4値・REL-012／013 |
| SEC-001 | subject終端binding | TEST-002／005 | AC-006／010 | identity相関証跡 | 4値・REL-012／013 |
| SEC-002 | 主体・session分離 | TEST-003／005 | AC-006 | negative access結果 | 4値・REL-012／013 |
| SEC-003 | 内部identity検証 | TEST-005／012 | AC-013 | header偽造試験 | 4値・REL-012／013 |
| SEC-004 | 支払条件正規化 | TEST-001／004 | AC-005／008／012 | policy判定記録 | 4値・REL-012／013 |
| SEC-005 | Checkout変更失効 | TEST-003／004 | AC-005 | 旧承認拒否記録 | 4値・REL-012／013 |
| SEC-006 | endpoint固定・SSRF防御 | TEST-005／008 | AC-008 | 接続先negative試験 | 4値・REL-012／013 |
| SEC-007 | Agent・capability scope固定 | TEST-002／005／008 | AC-001／008 | capability検証記録 | 4値・REL-012／013 |
| SEC-008 | A2A入力不信 | TEST-001／009 | AC-008 | 悪意応答試験 | 4値・REL-012／013 |
| SEC-009 | LLMと権限の分離 | TEST-003／006 | AC-003／004／008 | gate強制証跡 | 4値・REL-012／013 |
| SEC-010 | secret最小開示 | TEST-005／011 | AC-010／013 | redaction・network記録 | 4値・REL-012／013 |
| SEC-011 | 障害時fail closed | TEST-006／009 | AC-007／008／009 | failure state・副作用件数 | 4値・REL-012／013 |
| SEC-012 | AP2 Human Present検証 | TEST-002 | AC-001 | offline署名検証 | 4値・REL-012／013 |
| SEC-013 | profile選択・fallback禁止 | TEST-004／009 | AC-012 | profile分岐記録 | 4値・REL-012／013 |
| SEC-014 | simulation表示 | TEST-004／011 | AC-001／012 | UI・conformance証跡 | 4値・REL-012／013 |
| SEC-015 | Merchant認可検証 | TEST-008／009 | AC-001／008 | wire改ざん・副作用0件 | 4値・REL-012／013 |
| SEC-016 | 従来security callback維持 | TEST-006／010 | AC-001／002 | callback前後trace | 4値・REL-012／013 |
| SEC-017 | 返金認可 | TEST-016 | AC-014 | refund認可・negative結果 | 4値・REL-012／013 |
| DATA-001 | 主体相関schema | TEST-002／003／005 | AC-006 | record・query照合 | 4値・REL-012／013 |
| DATA-002 | plan・第一承認field | TEST-002 | AC-001 | evidence field照合 | 4値・REL-012／013 |
| DATA-003 | Agent immutable snapshot | TEST-002／008 | AC-001／002 | snapshot・wire照合 | 4値・REL-012／013 |
| DATA-004 | remote Task相関 | TEST-002／008 | AC-001 | Task ID・digest履歴 | 4値・REL-012／013 |
| DATA-005 | 決済・第二承認field | TEST-002 | AC-001／005 | evidence field照合 | 4値・REL-012／013 |
| DATA-006 | continuation CAS | TEST-003／009／013 | AC-006／007／011 | version・競合記録 | 4値・REL-012／013 |
| DATA-007 | identifier mapping | TEST-002／005 | AC-001／008 | alias mapping試験 | 4値・REL-012／013 |
| DATA-008 | 監査相関chain | TEST-002／006 | AC-001／009 | 順序付き監査event | 4値・REL-012／013 |
| DATA-009 | 返金相関record | TEST-016 | AC-014 | 元支払・返金field照合 | 4値・REL-012／013 |
| STATE-001 | 計画承認待ち遷移 | TEST-003／007 | AC-003 | state transition履歴 | 4値・REL-012／013 |
| STATE-002 | A2A応答分岐 | TEST-001／006／007 | AC-001／002／008 | 分岐・gate記録 | 4値・REL-012／013 |
| STATE-003 | 決済承認待ち遷移 | TEST-003／007 | AC-004／005／006 | state・副作用件数 | 4値・REL-012／013 |
| STATE-004 | 支払提出遷移 | TEST-004／006／009 | AC-001／007／008 | state・gate履歴 | 4値・REL-012／013 |
| STATE-005 | 同一Task再開遷移 | TEST-008／009 | AC-001／007 | Task相関・state履歴 | 4値・REL-012／013 |
| STATE-006 | 複数step遷移 | TEST-007 | AC-001／002 | step履歴 | 4値・REL-012／013 |
| STATE-007 | final判定遷移 | TEST-006／007 | AC-009 | ACCEPT・REJECT・REVIEW結果 | 4値・REL-012／013 |
| STATE-008 | 再計画・承認失効 | TEST-003／004／018 | AC-005 | version・digest変更設計 | DESIGNED／PARTIAL／NOT RUN・non-blocking |
| STATE-009 | 非同期待機 | TEST-007／013 | AC-006／007／011 | request終了・復元記録 | 4値・REL-012／013 |
| STATE-010 | 禁止遷移 | TEST-003／009 | AC-003／004／008／009 | negative transition結果 | 4値・REL-012／013 |
| STATE-011 | 返金状態遷移 | TEST-016 | AC-014 | Refund state履歴 | 4値・REL-012／013 |
| UI-001 | 認証後入口 | TEST-011／012 | AC-010 | browser・redirect記録 | 4値・REL-012／013 |
| UI-002 | 計画承認表示 | TEST-003／011 | AC-001／003 | screenshot・表示payload | 4値・REL-012／013 |
| UI-003 | 決済承認表示 | TEST-003／011 | AC-001／004／005 | screenshot・表示payload | 4値・REL-012／013 |
| UI-004 | 実trace表示 | TEST-006／011 | AC-001／002／009 | screenshot・trace照合 | 4値・REL-012／013 |
| UI-005 | 安全なerror表示 | TEST-009／011 | AC-004／005／007／008／011／012 | 各error画面 | 4値・REL-012／013 |
| UI-006 | simulation表記 | TEST-004／011 | AC-001／012 | screenshot・evidence | 4値・REL-012／013 |
| UI-007 | 機密非表示 | TEST-005／011 | AC-010／013 | DOM・network検査 | 4値・REL-012／013 |
| UI-008 | 通常仲介demo依頼 | TEST-011 | AC-001／002 | DEMO・browser記録 | 4値・REL-012／013 |
| UI-009 | 返金状態表示 | TEST-011／016 | AC-014 | refund UI・機密検査 | 4値・REL-012／013 |
| HTTP-001 | app一覧限定 | TEST-012 | AC-010／013 | list-apps応答 | 4値・REL-012／013 |
| HTTP-002 | 公開面認証 | TEST-012 | AC-010／013 | 未認証status記録 | 4値・REL-012／013 |
| HTTP-003 | Store route 404 | TEST-012 | AC-013 | exact・prefix matrix | 4値・REL-012／013 |
| HTTP-004 | 内部API route 404 | TEST-012 | AC-013 | exact・prefix matrix | 4値・REL-012／013 |
| HTTP-005 | identity header防御 | TEST-005／012 | AC-013 | 偽造header結果 | 4値・REL-012／013 |
| HTTP-006 | 公開allowlist | TEST-012 | AC-013 | allowlist black-box結果 | 4値・REL-012／013 |
| HTTP-007 | 返金same-origin境界 | TEST-012／016 | AC-013／014 | route・identity negative結果 | 4値・REL-012／013 |
| OPS-001 | 固定Cloud Run対象 | TEST-014 | AC-011 | project・region・service差分 | 4値・REL-012／013 |
| OPS-002 | Cloud SQL禁止 | TEST-010／014 | AC-011 | resource・config inventory | 4値・REL-012／013 |
| OPS-003 | ephemeral仕様 | TEST-013 | AC-011 | instance置換結果 | 4値・REL-012／013 |
| OPS-004 | 同一instance回復 | TEST-013／017 | AC-011 | checkpoint別restart設計 | DESIGNED／PARTIAL／NOT RUN・non-blocking |
| OPS-005 | 状態消失案内 | TEST-013 | AC-011 | 古いworkflow拒否設計 | DESIGNED／PARTIAL／NOT RUN・non-blocking |
| OPS-006 | loopback境界 | TEST-008／012 | AC-013 | listen・route検査 | 4値・REL-012／013 |
| OPS-007 | 更新専用手順 | TEST-014 | AC-011 | update guard実行記録 | 4値・REL-012／013 |
| OPS-008 | deploy fail-closed guard | TEST-014 | AC-011 | digest・traffic・rollback証跡 | 4値・REL-012／013 |
| OPS-009 | 認証・model環境 | TEST-006／011／014 | AC-001／010 | readiness・IAM・quota記録 | 4値・REL-012／013 |
| TEST-001 | 支払要求unit suite | 当該TEST全case | AC-001／002／008／012 | unit report | 4値・REL-012／013 |
| TEST-002 | 相関unit suite | 当該TEST全case | AC-001／005／006 | unit・offline verifier report | 4値・REL-012／013 |
| TEST-003 | 承認・状態unit suite | 当該TEST全case | AC-003〜006 | routing・state report | 4値・REL-012／013 |
| TEST-004 | 支払policy unit suite | 当該TEST全case | AC-005／008／012 | policy report | 4値・REL-012／013 |
| TEST-005 | security unit suite | 当該TEST全case | AC-006／008／010／013 | security report | 4値・REL-012／013 |
| TEST-006 | 実仲介integration suite | 当該TEST全case | AC-001／002／009 | integration trace | 4値・REL-012／013 |
| TEST-007 | paid・free integration suite | 当該TEST全case | AC-001〜005 | integration report | 4値・REL-012／013 |
| TEST-008 | HTTP相関integration suite | 当該TEST全case | AC-001／002／008 | captured wire・assert結果 | 4値・REL-012／013 |
| TEST-009 | 異常・障害integration suite | 当該TEST全case | AC-005／007／008／011 | failure injection設計 | DESIGNED／PARTIAL／NOT RUN・non-blocking |
| TEST-010 | regression suite | 当該TEST全case | Release-1必須ACの回帰判定 | regression report | 4値・REL-012／013 |
| TEST-011 | browser suite | 当該TEST全case | AC-001／002／010〜013 | local・Cloud Run evidence | 4値・REL-012／013 |
| TEST-012 | public boundary suite | 当該TEST全case | AC-010／013 | black-box matrix | 4値・REL-012／013 |
| TEST-013 | restart suite | 当該TEST全checkpoint | AC-007／011 | checkpoint試験設計 | DESIGNED／PARTIAL／NOT RUN・non-blocking |
| TEST-014 | release artifact suite | 当該TEST全case | REL-005／007／010 | digest結合report | 4値・REL-012／013 |
| TEST-015 | coverage suite | 見出し・matrix・ledger集合一致 | REL-012／013 | coverage machine report | 4値・REL-012／013 |
| TEST-016 | refund integration suite | 当該TEST全case | AC-014 | refund trace・negative件数 | 4値・REL-012／013 |
| TEST-017 | 高度競合・recovery suite | 当該TEST全case | AC-007／011／015 | recovery試験設計 | DESIGNED／PARTIAL／NOT RUN・non-blocking |
| TEST-018 | 拡張edge matrix suite | 当該TEST全case | AC-005／008 | edge matrix設計 | DESIGNED／PARTIAL／NOT RUN・non-blocking |
| AC-001 | 有料E2E scenario | TEST-002／006〜009／011 | 当該AC全条件 | paid trace・wire・browser証跡 | 4値・REL-012／013 |
| AC-002 | 無料E2E scenario | TEST-006／007／011 | 当該AC全条件 | free trace・record件数 | 4値・REL-012／013 |
| AC-003 | 計画拒否scenario | TEST-003／007 | 当該AC全条件 | 副作用0件 | 4値・REL-012／013 |
| AC-004 | 決済拒否scenario | TEST-003／007 | 当該AC全条件 | 副作用0件・中断理由 | 4値・REL-012／013 |
| AC-005 | 条件変更scenario | TEST-002〜004／009／018 | 当該AC全条件 | 旧承認拒否・再承認設計 | DESIGNED／PARTIAL／NOT RUN・non-blocking |
| AC-006 | basic replay・routing scenario | TEST-003／005 | 当該AC全条件 | routing matrix・件数 | 4値・REL-012／013 |
| AC-007 | Merchant障害scenario | TEST-009／013／017 | 当該AC全条件 | retry・REVIEW設計 | DESIGNED／PARTIAL／NOT RUN・non-blocking |
| AC-008 | 悪意応答matrix | TEST-001／005／009／018 | 当該AC全条件 | BLOCKED matrix設計 | DESIGNED／PARTIAL／NOT RUN・non-blocking |
| AC-009 | final anomaly scenario | TEST-006 | 当該AC全条件 | 最終成功阻止証跡 | 4値・REL-012／013 |
| AC-010 | UI階層・認証scenario | TEST-011／012 | 当該AC全条件 | browser・app一覧 | 4値・REL-012／013 |
| AC-011 | restart・ephemeral scenario | TEST-013／014／017 | 当該AC全条件 | checkpoint・resource設計 | DESIGNED／PARTIAL／NOT RUN・non-blocking |
| AC-012 | x402 profile scenario | TEST-004／011 | 当該AC全条件 | profile・表示証跡 | 4値・REL-012／013 |
| AC-013 | HTTP boundary scenario | TEST-005／012 | 当該AC全条件 | black-box・header結果 | 4値・REL-012／013 |
| AC-014 | 返金正常系scenario | TEST-016 | 当該AC全条件 | refund trace・Receipt・件数 | 4値・REL-012／013 |
| AC-015 | 高度並行承認scenario | TEST-017 | 当該AC全条件 | concurrency試験設計 | DESIGNED／PARTIAL／NOT RUN・non-blocking |
| PRC-001 | worktree保護手順 | 変更inventory判定 | REL-011 | 開始時status・差分 | 4値・REL-012／013 |
| PRC-002 | 旧仲介・現決済baseline | TEST-010 | AC-001／002 | baseline比較・回帰report | 4値・REL-012／013 |
| PRC-003 | 縦切り順序gate | TEST-006／007 | AC-001／002 | milestone試験履歴 | 4値・REL-012／013 |
| PRC-004 | 中心・Release-1例外完成順 | TEST-007／016 | AC-001／004／006／014 | milestone試験履歴 | 4値・REL-012／013 |
| PRC-005 | 自動試験・review・browser順 | TEST-010／011／014／015 | REL-002〜004 | dated gate evidence | 4値・REL-012／013 |
| PRC-006 | candidate・deploy順 | TEST-011／014 | AC-010／011／013 | candidate・revision履歴 | 4値・REL-012／013 |
| PRC-007 | 文書・通常PR | TEST-014／015 | REL-011 | PR metadata・文書差分 | 4値・REL-012／013 |
| REL-001 | Release-1必須ID・関連OQ closure | TEST-015 | Release-1必須AC | ledger・OQ decision log | 4値・REL-012／013 |
| REL-002 | Release-1必須suite完了 | TEST-015でmatrix参照 | Release-1必須AC | candidate test manifest | 4値・REL-012／013 |
| REL-003 | 独立review・試験 | 独立結果判定 | 全AC | review・test report | 4値・REL-012／013 |
| REL-004 | local・Cloud Run browser | TEST-011 | AC-001／002／010 | browser evidence | 4値・REL-012／013 |
| REL-005 | deploy対象・永続性 | TEST-013／014 | AC-011 | resource・deploy差分 | 4値・REL-012／013 |
| REL-006 | 一次仕様再確認 | version review判定 | AC-012 | 一次資料・互換差分 | 4値・REL-012／013 |
| REL-007 | 適合文書更新 | TEST-014／015 | AC-012 | 3適合文書・status | 4値・REL-012／013 |
| REL-008 | 必須trace | TEST-006／007 | AC-001／002／009 | gate・callback・final trace | 4値・REL-012／013 |
| REL-009 | 副作用・相関 | TEST-002／008／016 | AC-001／003／004／006／014 | wire・offline・支払／返金件数 | 4値・REL-012／013 |
| REL-010 | Cloud Run証跡 | TEST-011／012／014 | AC-010／011／013 | service・revision・digest | 4値・REL-012／013 |
| REL-011 | 文書・PR証跡 | TEST-014／015 | CLAIM-001〜003判定 | PR URL・SHA・draft・既知課題 | 4値・REL-012／013 |
| REL-012 | release判定 | TEST-015 | Release-1必須IDがPASS | closure・deferred report | 4値・REL-012／013 |
| REL-013 | 適合ledger | TEST-015 | 全規範IDが一回・2区分が排他的 | 139行ledger | 4値・REL-012／013 |
| CLAIM-001 | 現時点の許可主張 | 文書・PR claim監査 | REL-011 | claim inventory | 4値・REL-012／013 |
| CLAIM-002 | 禁止主張 | 文書・PR claim監査 | REL-011 | negative phrase監査 | 4値・REL-012／013 |
| CLAIM-003 | release後の限定主張 | TEST-014／015 | AC-012／REL-011 | simulation・範囲外表記 | 4値・REL-012／013 |

## 20. 未決事項

未決事項は後続の設計またはリリース準備で解消する。OQ-001〜009はRelease-1関連期限までに解決し、OQ-010は設計のみ・将来課題として既知課題とtriggerを記録できる。いずれの選択もRelease-1の正常系安全境界を弱める理由にはならない。

| ID | 未決事項 | 確定期限 | 不変制約 |
| --- | --- | --- | --- |
| OQ-001 | continuationのcanonical schemaの所有境界と、仲介状態・決済状態間のtransaction境界 | 設計確定前 | DATA-001〜008、FR-009、CAS・冪等性を満たす |
| OQ-002 | `agent-005`、registry名、service slug、registry skill、A2A skillのcanonical値、許可alias、migration規則 | 設計確定前 | DATA-007とSEC-007により暗黙変換を禁止する |
| OQ-003 | 既存demo recordへFirebase subjectを導入するmigration／default規則 | 設計確定前 | SEC-001／002を満たさないrecordへ越権アクセスを許さない |
| OQ-004 | 固定するA2A payment extensionのversion、Task state、必須part／field、canonicalization | 設計確定前 | 自由文判定とsilent fallbackを禁止する |
| OQ-005 | anomaly／final anomalyの入力schema、detector version、timeout、parse failure、critical issue、scoreから強制判定へのmapping | 設計確定前 | 例外を自動許可せず、`BLOCK`／`REVIEW`後に継続しない |
| OQ-006 | GeminiのDeveloper API／Vertex AI選択、Cloud Run service account、project／location、IAM、quota、timeout | Cloud Run受入前 | credential非露出、readinessでのfail closedを満たす |
| OQ-007 | 認証入口、health、UI、same-origin mediation APIの最終的な公開allowlist | 公開境界設計前 | HTTP-003／004の内部面は必ず404とする |
| OQ-008 | AP2標準schemaを壊さず仲介correlationを結合するevidence envelope／extensionの位置 | 設計確定前 | SEC-012、FR-008、offline verificationを満たす |
| OQ-009 | 公式一次資料の再確認後に固定するAP2／A2A x402のversionと互換性差分 | 設計確定前およびリリース前 | REL-006に従い暗黙更新しない |
| OQ-010 | 決済拒否、期限切れ、条件変更後のUX上の再計画／取消選択と、複数保留対象の明示選択方法 | 将来課題のUI設計前 | FR-007のroutingは変更せず、曖昧な `承認` をどの対象にも適用しない |

## 21. 完了判定

本書の要件定義工程の完了は、要件IDが一意であり、引継ぎ正本のFR-001〜015、AC-001〜013、8章、12章、15〜22章を逆引きでき、現状と目標を分離し、未決事項が不変制約付きで明示されていることで判定する。

製品実装またはリリースの完了は本書作成だけでは成立しない。後続工程で19.3の全139規範IDを対象candidateの適合ledgerへ一回ずつ登録し、`Release-1必須` 126件が証跡付き `PASS`、`設計のみ・将来課題` 13件が許可statusと既知課題・設計参照・triggerを持ち、Release-1関連期限のOQが解決済みであることをTEST-015で確認した時点でのみRelease-1完了と判定できる。
