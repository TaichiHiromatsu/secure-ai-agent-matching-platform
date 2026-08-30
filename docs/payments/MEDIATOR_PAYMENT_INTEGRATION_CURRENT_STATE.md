# 従来の仲介エージェントへの決済統合：現状分析

> [!WARNING]
> この文書は作成時点の実装調査snapshotであり、現在仕様の正本ではない。現行責務は[アーキテクチャ](ARCHITECTURE.md#actorと責務の正本)と[Payment Bridge設計](mediator-payment-integration-design/04_PAYMENT_BRIDGE_AP2_X402.md)を参照する。本文は履歴証跡として変更しない。

最終確認日: 2026-08-16（Asia/Tokyo）

## 0. この文書の位置づけ

この文書は、[MEDIATOR_PAYMENT_INTEGRATION_HANDOFF.md](MEDIATOR_PAYMENT_INTEGRATION_HANDOFF.md) を実装する前に、現在のリポジトリ、ローカル証跡、Cloud Run、Pull Request を読み取り専用で調査した結果である。要件や設計を新たに確定する文書ではなく、後続の要件定義・設計・実装が依拠できる現状の基準線を示す。

結論は次のとおりである。

- 現在の公開アプリは `payment_user_agent` 一つであり、計画承認、決済承認、AP2風の署名証跡、同一A2A Taskへの支払提出、冪等性、ブラウザ試験、Cloud Run実演までを備えた「固定された有料予約の直接workflow」としては動作している。
- しかし、従来の `matcher -> planner -> orchestrator -> anomaly_detector -> final_anomaly_detector` という仲介経路は公開アプリから切断されている。エージェント選定、計画、支払要否、無料／有料分岐は、実際の仲介処理やA2A応答から導出されていない。
- したがって、現行の成功証跡は新要件の仲介統合を証明しない。特に FR-001、FR-003、FR-005、FR-006、FR-010、FR-011、FR-012、FR-014、FR-015 は未達である。
- 決済workflow、署名、Task再利用、CAS、outbox、Firebase入口認証などは再利用価値が高い。一方、旧仲介ルートをそのまま復元すると、曖昧な承認、LLM依存の制御、構造化A2A Taskの欠落、主体分離不足を再導入するため、単純な巻き戻しはできない。

## 1. 確認できた事実とコード根拠

### 1.1 現在の公開呼出し経路

現在の実行経路は次のとおりである。

```text
Firebase認証
  -> nginx
  -> ADK Webの payment_user_agent
  -> PaymentWorkflowAdapter
  -> workflow API / controller
  -> 固定された paid-booking Merchant
```

根拠:

- `payment_user_agent/agent.py:3-6` は `secure_mediation_agent.agent.PaymentWorkflowAdapter` を直接 `root_agent` として公開している。
- `secure_mediation_agent/agent.py:49-89` は最初の発話を `workflow` の `create`、以後を同じworkflowへのメッセージ送信へ渡すadapterである。`matcher`、`planner`、`orchestrator`、`anomaly_detector`、`final_anomaly_detector` はimportも実行もしていない。
- `secure_mediation_agent/agent.py` 自体には `secure_mediator` の `root_agent` が存在しない。一方、`secure_mediation_agent/__init__.py:13-18` は `root_agent` の遅延importを試みるため、packageから従来rootを直接取得する経路には不整合がある。
- `Dockerfile:45-51` は `payment_user_agent` だけを `/app/payment-apps` 配下へ置き、`secure_mediation_agent` を内部コードとして配置する。
- `deploy/supervisord.conf:31-40` はADK Webの発見rootを `/app/payment-apps` に限定する。この構成により公開アプリ一覧が一つである点は満たしている。

### 1.2 従来の仲介部品は残っているが、現在は使われていない

次の実装はリポジトリに残っている。

- `secure_mediation_agent/subagents/matching_agent.py`: Trusted Agent Storeから候補を検索し、信頼度やcapabilityを返す。
- `secure_mediation_agent/subagents/planning_agent.py`: 選択されたAgentを用いて計画を生成し、計画artifactを保存する。
- `secure_mediation_agent/subagents/orchestration_agent.py`: Remote A2A Agentを呼び、tool後のjudgeを実行する。
- `secure_mediation_agent/subagents/anomaly_detection_agent.py`: Agent、status、outputの逸脱を評価する関数とAgentを持つ。
- `secure_mediation_agent/subagents/final_anomaly_detection_agent.py`: 会話、計画、artifactを使った最終評価を行う関数とAgentを持つ。

ただし、これらは現行rootから到達しない。Git履歴上の `dbd88af` より前の `secure_mediation_agent/agent.py` には、これらをsub-agentに持つ `secure_mediator` rootが存在したが、そのrootは次の理由でそのまま復元できない。

- 計画承認はstateのbooleanだけであり、計画ID、version、digestを承認へ固定していない。
- 「承認します」「OK」「はい」も承認として扱うLLM instructionであり、現在の厳密な `承認` 判定より弱い。
- sub-agent間の遷移を主にLLM instructionへ委ねており、停止、再開、失敗時の制御を永続状態機械で保証していない。

### 1.3 Agent選定と計画の現状

- `secure_mediation_agent/subagents/matching_agent.py:30` はTrusted Agent Storeを既定で `http://127.0.0.1:8001/api/agents` から取得する。
- 同ファイル `105-118` の変換結果はname、汎用的なurl、skills、capabilities、trust、agent_idを保持するが、`agent_card_url` とRPC endpointを別々の型として保持せず、Agent Card digestも固定しない。
- `secure_mediation_agent/subagents/planning_agent.py:28-105` はMarkdown計画を保存して `plan_approved=False` を設定するが、正規化された計画ID、version、digestを持つ承認対象を作らない。
- 現行workflowの `secure_mediation_agent/workflow/controller.py:104-247` はmatcherとplannerを呼ばず、`paid-booking-agent`、`127.0.0.1:8005/a2a`、固定skill、固定商品、固定金額を含む計画をcontroller自身で生成する。
- `secure_mediation_agent/workflow/models.py:62-82` のrequest既定値は `demo-paid-booking`、quantity 1、USD 1250であり、`paymentRequired` はクライアント入力のbooleanである。
- `secure_mediation_agent/workflow/models.py:109-167` のAgent、step、product、currency、profileはLiteralまたは固定値である。

従って、現在表示される「計画」は仲介plannerの出力ではなく、決済デモ専用controllerが作る固定plan snapshotである。

### 1.4 承認と状態遷移の現状

- `secure_mediation_agent/workflow/approval.py:25-62` は、pending状態に応じて、単一text partが厳密に `承認` と一致する場合だけ計画承認または決済承認を受理する。この判定は再利用できる。
- `secure_mediation_agent/workflow/controller.py:523-633` は計画承認を署名付きauthorizationとして保存する。計画ID、version、digest、session、contextを承認へ結び付ける。
- 同箇所 `605-632` はrequestに事前設定された `paymentRequired` だけで無料／有料を分岐する。Remote A2A Taskの応答は判定材料にならない。
- 無料分岐は `FREE_EXECUTING -> FINAL_VALIDATING -> COMPLETED` と状態を進めるが、従来orchestratorの実処理も最終異常検知も呼ばない。
- 有料分岐では `secure_mediation_agent/workflow/controller.py:634-678` がTask、context、orderを作り、outboxを発行する。その後、Merchantとの開始、決済承認、支払提出、settlement、fulfillmentを処理する。
- repositoryは `secure_mediation_agent/workflow/repository.py` のtransition処理で期待stateとversionを照合し、状態更新とoutbox追加を同一transactionで行う。outboxにはleaseと再試行処理がある。

現行workflowの状態機械とCASは堅いが、これは「仲介stepを待機させ、同じstepを継続する」状態機械ではない。`mediation_session`、legacy plan/step、continuation、待機中Remote Taskを保存する項目は存在しない。

### 1.5 A2A Taskと決済証跡の現状

- `secure_mediation_agent/payment_profiles/a2a.py:37-62` は `input-required` のTaskと構造化された支払metadataを生成する。
- 同ファイル `65-86` は同じtask IDとcontext IDへ支払提出messageを追加する。
- `secure_mediation_agent/merchant/client.py:80-170` と `secure_mediation_agent/merchant/service.py:100-284` は、開始時に受けたTaskへ支払を提出する。
- Merchantは支払提出後も同一task/contextを維持し、`secure_mediation_agent/merchant/service.py:313-362` で完了させる。
- `secure_mediation_agent/workflow/controller.py:1210-1347` はworkflow ID、task ID、plan digest、checkout hash、payment requirements等をAP2 credential、proof、支払messageへ結び付ける。
- schemaにはimmutableなplan snapshot、計画承認、決済承認、capability、nonce、Task mirror、artifact、receipt、settlement idempotency、workflow event、outboxがある（`secure_mediation_agent/workflow/migrations.py:43-350`）。

従って「決済workflow内で生成したTaskを同じ決済workflow内で再利用する」ことは実装済みである。一方、「従来orchestratorが開始したRemote A2A Taskを停止し、その同じTaskと仲介stepを決済後に再開する」ことは未実装である。

さらに、現在のAP2周辺証跡にはworkflow ID、Task ID、plan digest等はあるが、次が明示的には結合されていない。

- 仲介計画ID、仲介計画version、仲介step ID
- Agent Registryの正規ID
- Agent Card URL、RPC endpoint、選定時のAgent Card digest、trust score
- quote ID
- Firebase subject

### 1.6 orchestratorと異常検知の現状

- `secure_mediation_agent/subagents/orchestration_agent.py:96-339` はRemote Agent Cardを取得してA2Aを呼べるが、毎回新しいuser/sessionを作り、応答を主にsanitized text/historyへ変換する。`input-required` Taskのpayment metadata、task ID、context IDをworkflowへ引き渡す型がない。
- 同ファイル `147` は汎用的なAgent URLへ `/.well-known/agent-card.json` を追加する。Registryにcard URLとRPC endpointを別々に保持する設計と整合しない。
- 同ファイル `32-93` はblock以外の応答を完了として扱い、支払待ち状態を返さない。
- 同ファイル `421-647` のafter-tool callbackは `custom_judge` を呼ぶが、`anomaly_detection_agent.py` の `anomaly_detector` sub-agentを実行しない。計画逸脱もjudgeがsafeと返せばwarningに留まる経路がある。
- `secure_mediation_agent/subagents/final_anomaly_detection_agent.py:303-374` はweighted scoreを計算する。critical issueを記録してもrecommendationがscoreだけで決まるため、低confidenceのcritical検知を常にfail-closedにする保証はない。
- 現行workflowの終端は `secure_mediation_agent/workflow/controller.py:1652-1679` が `mediator:final-validate` capabilityを発行し、final taskを保存した後に状態遷移するだけである。`final_anomaly_detector` は呼ばれない。

従って、現行コードに「異常検知」という名前や状態はあるが、FR-010とFR-011が求める実際の仲介経路上の強制検査ではない。

### 1.7 profile、Agent Registry、Agent Cardの現状

- `secure_mediation_agent/payment_profiles/registry.py:10-17` はsimulation profileだけを有効にし、それ以外をfail-closedにする。
- `secure_mediation_agent/payment_profiles/simulation_v1.py:20-125` は `x402-wire-simulation/1`、`exact-simulated`、`demo:local`、USDのローカルsimulationを定義する。
- `secure_mediation_agent/payment_profiles/x402_v01.py:8-34` のofficial x402 profileは無効であり、明示的にfail-closedにする。
- `trusted_agent_store/data/agents/registered-agents.json:100-156` の登録AgentはID `agent-005`、name `paid_booking_agent`、skill `paid_booking`、endpoint base URL、Agent Card URL、trust 90を持つ。
- そのRegistry extensionは旧 `ap2-x402-marketplace:v1` とplatform-credit/demo mediation ledgerを示す一方、実際のMerchant Agent Card（`secure_mediation_agent/merchant/service.py:64-98`）は `x402-wire-simulation:v1`、skill `paid-booking`、`exact-simulated`、`demo:local` を示す。

登録情報とlive Agent CardにはID、skill表記、profile extensionの不一致がある。後続実装が正規化とcard検証をfail-closedで行う場合、現在のままでは対象Agentを選べても実行前に拒否するのが正しい。

### 1.8 認証、主体、公開境界の現状

- `deploy/auth/verify.py:66-79` はFirebase tokenのproject、issuer、subjectを検証する。
- 同ファイル `142-164` はSecure、HttpOnly、SameSite=Strictのsession cookie、CSRF、origin検証を持つ。
- nginxは `/mediation-api/` とADK Webへ、外部から渡されたidentity headerを置換して内部署名identityを渡す。
- `secure_mediation_agent/identity.py:30-55` はFirebase subjectを署名payloadに含めるが、tenant/customerは全subjectに対して同じdemo値を既定とする。
- `secure_mediation_agent/identity.py:83-91` と `secure_mediation_agent/workflow/api.py` のdomain identity変換は最終的にtenant/customerだけをworkflow ownershipへ使い、subjectを永続化・照合しない。

そのため、二つの正当なFirebase subjectが同じdemo tenant/customerへ写像される。workflow IDを知った別subjectをownerとして分離できない。現在のsecurity testはtenant/customer偽装を拒否するが、異なる有効subject間の分離を証明しない。

nginxの公開境界にも新要件との差がある。

- `deploy/nginx.conf:63-67` が404にするのは `/payment/`、`/paid-agent/`、`/internal/`、`/v1/` のprefixである。
- `/mediation-api/`、`/`、`/api/`、`/static/`、`/store/`、`/store/sse/`、`/store/health`、`/a2a/` は認証後にproxyされる。
- `/ws/` は認証checkを通さずproxyされる。
- exact path `/v1` と `/internal` は404規則に一致せず、root側へfall throughする。
- `/store` のredirectは `$scheme` を使うため、Cloud RunのTLS終端後に外部へ `http://.../store/` を返すことがある。

公開アプリが一つであることと、内部APIが外部から到達不能であることは別である。現行構成は前者を満たすが、後者のFR-015/AC-013は満たさない。

### 1.9 UIの現状

- `secure_mediation_agent/workflow/views.py:25-54` のAgent、商品、価格表示は固定値である。
- 同ファイル `57-115` の公開viewは現在stateと最終ID等を返すが、matcher、planner、計画承認、orchestrator、支払要否判定、決済承認、step再開、anomaly、final anomalyの時系列traceを返さない。
- 現行ブラウザでは固定予約、二段階承認、決済結果は確認できるが、従来仲介経路を通ったことは確認できない。

### 1.10 永続性と実行環境の現状

- `deploy/start.sh:9-17` のCloud Run demo modeは、起動時にローカルkeyを生成し、writable filesystemへSQLiteと証跡を置く。revision再起動でstateとkeyが失われ得ることを明示している。
- durable modeは明示的なvolume markerと `/run/secrets/ap2-demo` のkey mountがなければfail-closedにする。
- 現在のCloud Runは `EPHEMERAL_CLOUD_RUN_DEMO=true` であり、単一revisionの実演環境であってdurable deploymentではない。
- 仲介sub-agentは `gemini-2.5-pro` または `gemini-2.5-flash` を指定する（`secure_mediation_agent/subagents/*.py`）。現在のCloud Run service設定には、これらのmodelを呼ぶproject/location/API key/Secret Manager bindingが明示されていない。再接続前に認証経路を確定する必要がある。
- `secure_mediation_agent/payment_marketplace/` には別系統の旧marketplace実装が残るが、現在のsupervisorはこれをserviceとして起動せず、現行デモは `workflow` と `merchant` を使う。名前の類似だけでこの旧系統を統合対象へ戻すべきではない。

## 2. 引継ぎ要件との差分

| 要件 | 現状 | 判定 | 主な差分 |
|---|---|---|---|
| FR-001 従来の仲介ルート | 公開rootは `PaymentWorkflowAdapter` | 未達 | matcher/planner/orchestrator/anomaly/final anomalyをrootへ再接続する必要がある |
| FR-002 公開アプリ一つ | ADK発見rootは `payment_user_agent` のみ | 構造上達成 | 内部HTTP routeの非公開化はFR-015として別途必要 |
| FR-003 動的なAgent選定と計画 | controllerがAgent、step、商品、価格を固定生成 | 未達 | Registry選定結果とplanner出力をtyped planへ渡す必要がある |
| FR-004 計画承認 | 厳密な `承認` と署名済みplan digestがある | 部分達成 | 仲介plannerの計画を承認対象にし、承認済み版だけをorchestratorへ渡す必要がある |
| FR-005 A2A応答で支払要否判定 | requestのbooleanで事前分岐 | 未達 | Remote Taskのstateと構造化payment metadataを検証して分岐する必要がある |
| FR-006 step停止・継続 | 決済workflow内のTask再利用のみ | 未達 | mediation session、legacy plan/step、Remote Task、resume token相当を永続化する必要がある |
| FR-007 二つの承認を分離 | plan approvalとpayment approvalは別record | 大部分再利用可 | 仲介計画承認との結合、UI表示、拒否時の仲介step終了規則が必要 |
| FR-008 AP2証跡と仲介計画 | workflow/task/plan digestは結合 | 部分達成 | 仲介plan ID/version/step、registry/card、quote、subjectのbindingが不足 |
| FR-009 同じA2A Taskへ支払提出 | 決済workflow内では達成 | 部分達成 | そのTaskがorchestrator開始Taskと同一であることを保証できない |
| FR-010 セキュリティ検証 | 旧部品は存在するが現行routeで未実行 | 未達 | 実tool応答に対する強制gateとfail-closedな遷移が必要 |
| FR-011 最終異常検知 | 状態名とcapabilityだけで実Agent未実行 | 未達 | 保存済み全証跡を入力に実detectorを呼び、失敗・異常時は完了禁止が必要 |
| FR-012 無料処理 | boolean falseで形式的に完了 | 未達 | Remote Agentの無料応答を実orchestratorで完了し、異常検知を通す必要がある |
| FR-013 冪等性・再試行・競合 | CAS、nonce、outbox、settlement idempotencyあり | 部分達成 | 仲介continuationとRemote Task submitにも同じ保証を拡張する必要がある |
| FR-014 仲介経路のUI表示 | 固定workflow stateのみ | 未達 | 全stage、対象Agent、step、支払分岐、検査結果を時系列表示する必要がある |
| FR-015 デモ運用境界 | simulation表示とephemeral警告あり | 部分達成 | store/api/ws/a2a/internal routeの外部404、update deploy、永続性表示が不足 |

非機能・セキュリティ要件については、次が主要差分である。

- 認証と主体の結合: Firebase subjectがworkflow ownerと証跡へ終端まで結合されていない。
- 支払条件の検証: 現行固定条件の検証はあるが、Remote Agentが返した動的な金額、通貨、期限、network、asset、payTo、profileを仲介planと突合する経路がない。
- A2A接続の防御: Registry URL、Agent Card URL、RPC endpointの型分離、card digest pinning、SSRF防御、redirect/timeout/size/content-type制限の一貫した設計がない。
- 秘密情報: AP2 demo keyはlocalで扱えるが、Gemini/Vertex認証とCloud Run Secret Managerの使用方針が未確定である。
- 障害時原則: 決済workflowは多数のfail-closed検証を持つが、旧orchestratorのLLM制御、callback例外、final detector判定は全経路でfail-closedと証明できない。
- 適合判定: 現行はローカルsimulationであり、official x402はNOT RUNである。新しい仲介統合ができても自動的にofficial x402 compliantになるわけではない。

## 3. 再利用できる資産

### 3.1 ほぼそのまま再利用できるもの

- `payment_user_agent` 一つだけを公開するDocker/ADK配置。
- Firebase token検証、session cookie、CSRF、origin検証、外部identity headerの除去と内部署名identity。
- `workflow/approval.py` の厳密な `承認` 判定。
- immutable plan approvalとpayment approvalを分けた保存方式。
- canonical JSON/digest、署名、nonce、capability、receipt、evidence verifier。
- repositoryのversion CAS、transactional outbox、lease、retry、idempotency key、append-only artifact/event。
- Merchant client/serviceと `payment_profiles/a2a.py` の同一task/context継続。
- simulation profileを明示し、未知profileと無効なofficial x402をfail-closedにするregistry。
- image digest、source digest、回帰試験、browser evidence、release validationを結び付けるcandidate build/push検証。

### 3.2 adapter化して再利用すべきもの

- matcher: Store検索とtrust/capability情報は使い、戻り値をcanonical Agent ID、skill ID、Agent Card URL、RPC endpoint、card digestへ正規化する。
- planner: LLMによる計画生成は使えるが、出力を検証済みtyped planへ変換し、ID/version/digestを付ける。
- anomaly detector: 比較ロジックをtool呼出し後の決定的gateへ組み込み、warningだけで継続しない。
- final anomaly detector: 計画、全Task、支払証跡、artifact、receiptを入力にし、critical issueと実行例外を必ずrejectへ写像するwrapperが必要である。
- PaymentWorkflowAdapter/controller: UI向けview、承認、署名、決済state machineは使い、固定requestから作るのではなく仲介continuationを入力とするbridgeへ分離する。
- nginx/Firebase構成: 認証入口としては使い、内部routeは明示的404へ変更する。

### 3.3 参照に留め、そのまま戻してはいけないもの

- Git履歴上の旧 `secure_mediator` root: 望ましい全体像の参照にはなるが、曖昧な承認とLLM主導遷移を持つ。
- `payment_marketplace/`: 現行AP2/A2A workflowとは別系統であり、無条件にruntimeへ復帰させると二重の決済正本が生じる。
- 現行固定 `WorkflowRequest.paymentRequired`: テストfixtureには使えてもruntimeの支払要否判定には使えない。
- 現行固定Agent/skill/productのLiteral: migration互換やfixture以外では使わない。

## 4. 技術的リスク

### 4.1 最優先リスク

1. **正当な別subject間の越権**
   Firebase認証自体が正しくても、workflow ownerが共通demo tenant/customerだけで判定される。subjectをschema、identity、query条件、証跡へ追加しない限り、推測・漏洩したworkflow IDへのアクセス分離を証明できない。

2. **支払Taskの取り違え**
   旧orchestratorは構造化Taskを落としてtextへ寄せる。Task ID、context ID、step ID、quoteを型付きで保存せずにbridgeをつなぐと、別Taskへ支払を提出する危険がある。

3. **Registryとlive cardの不一致**
   `agent-005` / `paid_booking_agent` / `paid_booking` と、Merchant側の `paid-booking-agent` / `paid-booking`、profile extensionが一致しない。aliasを場当たり的に増やすと署名・監査上の主体が曖昧になる。

4. **検査を呼んだように見えて迂回すること**
   現行はfinal validationという状態名だけ、旧orchestratorは別judge callbackだけである。detector名、入力digest、結果、判定規則を永続化し、異常・timeout・parse errorをblockへ統一しなければならない。

5. **承認境界の混同**
   旧rootの自然言語承認を復元すると厳密な二段階承認を弱める。計画承認と決済承認は別のpending state、別nonce、別digestのまま維持する必要がある。

### 4.2 実装・運用リスク

- `WorkflowRequest.paymentRequired` を残すと、Remote A2A応答ではなくUI入力が決済分岐を決め続ける。
- 計画変更、quote変更、期限切れ時に、旧plan approvalを再利用しないversion規則が必要である。
- payment approvalのrequest処理中に長い後続処理を同期実行すると、ADK/Cloud Run timeout、二重送信、ユーザーが次のturnを送る時点との競合が起きる。outbox workerへ委譲する境界を明示する必要がある。
- 現在のSQLite + ephemeral Cloud Run + concurrency 1はデモには使えるが、再起動耐性AC-011を満たさない。単にmin instancesを1にしても永続性は得られない。
- `/ws/`、`/api/`、`/store/`、`/a2a/` が外から到達できる構成は、認証済みユーザーに内部面を公開する。upstreamが偶然404を返すことは境界保証にならない。
- `/store` のHTTP redirectは経路情報の漏れとdowngradeを招く。外部公開しない方針ならredirect自体を削除し404へ統一する。
- matcher/planner/orchestrator/final detectorはGemini modelへ依存する。Cloud Runのidentity、Vertex AI利用可否、project/location、quota、timeout、再試行を確定せず接続すると環境ごとに挙動が変わる。
- LLM出力をそのまま計画やsecurity判定へ使うと非決定性がrelease証跡を壊す。schema validationと決定的なpolicy gateをLLMの外側へ置く必要がある。
- AP2 mandateの標準schemaへ独自の仲介fieldを直接追加すると相互運用性を損なう可能性がある。仲介correlationは、既存の署名credential/evidence envelopeで結合するのか、extensionとして定義するのかを先に決める必要がある。
- 旧marketplaceと現workflowのDB、用語、profileが共存するため、誤ったmoduleを再利用すると決済の正本が二つになる。

## 5. テスト、ローカル、Cloud Run、PRの現状

### 5.1 既存テストと証跡

リポジトリには25 test file、parametrize展開前で121 test functionがある。既存のrelease artifactは現行の直接workflowに対して成功している。

- `artifacts/regression-result.json`: payment-release 173、evaluation 17、jury 13がPASS。Google依存8件はmanifestで許可されたskip。
- `artifacts/browser-evidence.json`: 実Chromiumで固定予約、厳密な二回の `承認`、完了、公開アプリが `payment_user_agent` 一つであることを確認。
- `artifacts/cloud-run-candidate.json`: `linux/amd64` candidateと正確なimage digestを記録。
- `artifacts/ap2-x402-release-validation.json`: 現行requirementsとsimulation profileに対するrelease validationがPASS。
- `docs/ap2_x402_conformance_report.json`: ローカルsimulationは検証済み、official x402はNOT RUN。

ただし、これらは新しい仲介統合の受入証跡ではない。

- `tests/security/test_release_boundaries.py:18-29` は、むしろ `secure_mediation_agent/agent.py` に `root_agent` がないことと、公開rootが `PaymentWorkflowAdapter` であることを要求する。FR-001実装時に意図的な更新が必要である。
- `tests/browser/test_adk_web_browser.py:125-305` はpayment appとworkflow test serverだけを起動し、matcher、planner、orchestrator、anomaly、final anomaly、Firebase/nginxを通さない。
- 実際の仲介chain、legacy plan/step、continuation、A2A応答によるpaid/free分岐を検証するtestはない。
- 異なる有効Firebase subject間のownership isolation testはない。
- AC-001からAC-013を新構成で通すtestとblack-box route testはまだない。

今回の調査では、system Pythonにpytest環境がなく、既存candidate artifactが現行コード相当のsource digestを検証済みだったため、全suiteの再実行やcontainerの再起動は行っていない。既存のPASSは過去のartifactに対する記録であり、未実装の新要件に対するPASSとは扱わない。

### 5.2 ローカル状態

- branch: `codex/ap2-x402-integration`
- HEAD: `59e821c`
- `origin/codex/ap2-x402-integration` に対してahead 0 / behind 0。
- `origin/main` との差分は115 file、約21,057 insertion / 466 deletionであり、決済実装、試験、文書、deployが一つの大きなPRに入っている。
- 調査開始時から `docs/payments/README.md` にuser変更があり、`docs/payments/MEDIATOR_PAYMENT_INTEGRATION_HANDOFF.md` がuntrackedだった。これらには変更を加えていない。
- `scripts/cloud_run_candidate.py source-info` と既存candidateのverify結果から、実行コードに関係するsource digestは現candidateと整合していた。ただしworktree全体は上記文書変更によりcleanではない。

### 5.3 Cloud Run

2026-08-16にread-onlyで確認した状態:

- project: `gen-lang-client-0585901015`
- region: `asia-northeast1`
- service: `payment-user-agent-demo`
- ready/latest revision: `payment-user-agent-demo-00002-nt7`
- traffic: latest revisionへ100%
- image digest: `sha256:a22c3e...`（完全値はcandidate/deployment artifactを正本とする）
- min/max instances: 1/1、concurrency: 1、timeout: 3600秒
- mode: `EPHEMERAL_CLOUD_RUN_DEMO=true`

未認証black-box確認では、`/` はloginへ302、`/health` は200、`/mediation-api/ready` はloginへ302だった。内部routeは一律404ではなく、`/store/`、`/store/health`、`/api/`、`/a2a/` はloginへ302、`/v1` と `/internal` も302、末尾slash付き `/v1/` と `/internal/` は404だった。`/ws/` は現時点で404だったが、nginx上は未認証proxy routeである。`/store` は外部HTTP URLへ301した。

従って、serviceは現行直接workflowのdemoとして稼働中だが、AC-013の公開境界を満たさず、新仲介統合も未deployである。

### 5.4 deploy script

- `deploy/build-payment-demo-candidate.sh` はclean build context、固定platform、image内test、source/image digest bindingを行う。
- push scriptは固定Artifact Registryとcandidate provenanceを検証する。
- `deploy/deploy-payment-demo-cloudrun.sh:39-52` はNEW service専用で、serviceが既に存在するとexit 3で停止する。
- 既存serviceをimmutable digestで更新し、revision readiness、traffic、black-box boundary、rollback情報を検証するupdate scriptは存在しない。

従って、現serviceへ新実装を反映する前に「既存service更新用script」を作る必要がある。create-only scriptの条件を緩めて流用するより、更新・検証・rollbackを明示した別scriptにする方が安全である。

### 5.5 Pull Request

2026-08-16にread-onlyで確認した状態:

- PR: `#25 仲介エージェントへAP2決済デモを統合`
- URL: <https://github.com/TaichiHiromatsu/secure-ai-agent-matching-platform/pull/25>
- state: OPEN、draftではない、GitHub上はMERGEABLE
- base/head: `main` <- `codex/ap2-x402-integration`
- review decision: なし
- status checks: なし

PR名は仲介統合済みに読めるが、コードは固定決済workflowへの直接接続である。GitHubのMERGEABLEはGit競合がないことを示すだけで、引継ぎ要件の充足やCI成功を示さない。

## 6. 作業開始前に解消すべきblocker

次は、実装者の局所判断で進めると互換性・セキュリティに影響するため、要件または設計として先に確定すべき項目である。

### B-001 仲介計画と継続recordの正本

一つのcanonical schemaで次を定義する必要がある。

- mediation session ID
- plan ID、version、digest
- step IDとstep status
- canonical Agent IDとskill ID
- Agent Card URL、RPC endpoint、card digest、選定時trust/capability
- Remote A2A task ID、context ID
- payment requirement/quote IDとそのdigest
- workflow ID、payment approval ID
- Firebase subject、tenant、customer、ADK session
- resume状態、retry count、last error、expiry

このrecordを仲介側と決済workflow側のどちらが正本にするか、またtransaction境界をどう分けるかを決める必要がある。

### B-002 識別子とprofileの正規化

`agent-005`、`paid_booking_agent`、`paid-booking-agent`、`paid_booking`、`paid-booking` のcanonical値と許可aliasを確定する。Registry extensionとlive Agent Cardのprofile差もmigrationする。実装は不一致を黙って吸収せず、明示されたalias/version以外を拒否する。

### B-003 主体分離

Firebase subjectをworkflow ownerへ昇格し、既存demo recordのmigration/default規則を決める。少なくともworkflow取得、承認、支払、artifact取得の全queryでsubjectを照合し、異なる正当subjectのnegative testを追加する必要がある。

### B-004 支払要否を表すA2A contract

Remote AgentのどのTask state、part、extension、profile fieldを支払要求の正本とするかを確定する。金額、通貨、network、asset、payTo、expiry、quote ID、profile、task/contextの必須性とcanonicalizationも必要である。単なるtext解析は禁止する。

### B-005 異常検知の決定規則

anomaly/final anomalyのinput schema、timeout、parse failure、model failure、critical issue、score thresholdを、明確なblock/allowへ写像する。LLMの自然文recommendationだけで実行継続を決めない。検査対象digestと検査結果を保存する。

### B-006 model実行環境

GeminiをDeveloper APIとVertex AIのどちらで呼ぶか、Cloud Run service account、project/location、IAM、Secret Manager、quota、timeoutを決める。認証情報をimageやrepositoryへ含めず、readinessで不足を検出する。

### B-007 Cloud Runの永続性と更新方式

AC-011をCloud Runで受け入れるなら、SQLite ephemeral filesystemでは満たせない。Cloud SQL等のdurable store、keyの永続管理、migration、revision間互換を決める。単発demoのままなら、再起動耐性を受入条件から分離し、UIと文書で明記する。併せて既存service update/rollback scriptが必要である。

### B-008 公開route一覧

外部許可routeをallowlistで確定する。引継ぎ要件どおりなら、`/store`、`/store/`、`/store/sse/`、`/store/health`、`/api`、`/api/`、`/ws`、`/ws/`、`/a2a`、`/a2a/`、`/internal`、`/internal/`、`/v1`、`/v1/`、旧payment routeをexact/prefixとも404にする。内部serviceはloopbackからだけ利用する。

## 7. 後続の要件定義・設計への入力

### 7.1 推奨する責務境界

```text
payment_user_agent（唯一の公開root）
  -> secure_mediator controller
       -> matcher adapter
       -> typed planner adapter
       -> plan approval gate
       -> orchestrator adapter
            -> Remote A2A Task
            -> anomaly gate
            -> payment-required の場合だけ payment bridge
                 -> payment approval gate
                 -> AP2/x402 simulation workflow
                 -> 同じRemote A2A Taskへ提出
            -> step resume
       -> final anomaly gate
       -> completed / blocked
```

公開rootは一つのまま維持し、LLM sub-agentのinstructionではなく、永続controllerがstage遷移と承認を支配する構成が望ましい。Payment workflowは独立した公開rootではなく、支払要求が確認されたstepだけが呼ぶ内部bridgeとする。

### 7.2 最小のdata contract

後続設計では、少なくとも次のtyped objectを定義する必要がある。

1. `SelectedAgentSnapshot`: canonical agent/skill、card URL、RPC endpoint、card digest、trust、capability、選定時刻。
2. `MediationPlanSnapshot`: plan ID/version/digest、steps、各stepのAgent snapshot、承認状態。
3. `A2aTaskSnapshot`: task/context、state、structured parts、artifact digest、受信時刻。
4. `PaymentRequirementSnapshot`: quote、amount/currency、network/asset/payTo、profile、expiry、source task、digest。
5. `MediationContinuation`: session、plan/step、Remote Task、workflow、subject、resume state、retry/idempotency。
6. `SecurityDecision`: detector/version、input digest、decision、reasons、critical flag、timestamp。

これらをJSON textだけで受け渡さず、validation済みmodelとDB制約で結ぶ。

### 7.3 状態遷移設計への入力

- 計画承認前にRemote Agentを実行しない。
- Remote Taskが無料完了ならpayment workflowを作らない。
- payment-requiredなら仲介stepを `WAITING_FOR_PAYMENT_APPROVAL` へ置き、Remote Taskを保存する。
- 決済承認拒否、価格変更、期限切れ、profile不一致はstepを勝手に完了させず、明示的なblocked/replan状態へ移す。
- 支払提出は保存済みtask/contextへ一度だけ行い、同じidempotency keyでretryする。
- step再開後もanomaly gateを通し、全step完了後にfinal anomaly gateを通す。
- detector timeout、例外、parse failure、証跡不足では `COMPLETED` へ進めない。
- replan時はversion/digestを更新し、旧計画承認と旧決済承認を新planへ流用しない。

### 7.4 テスト設計への入力

既存suiteを残しながら、引継ぎ文書のAC-001からAC-013を次の層で追加する。

- unit: identifier normalization、payment requirement parser、binding digest、subject ownership、detector policy、state transition。
- integration: matcher -> planner -> approval -> orchestratorの無料系、有料系、同一Task再開、Merchant障害、価格変更、replay、並行承認。
- security: malicious Agent Card/A2A response、SSRF、card digest差替え、別subjectアクセス、検査timeout/invalid JSONのfail-closed。
- browser: 実Firebase/nginx相当入口から、全stage traceと二つの承認を確認する。
- black-box: 許可routeだけが200/302となり、内部routeのexact/prefixが未認証・認証済みとも404になることを確認する。
- restart: 待機中step、支払後、outbox lease中の各時点で再起動し、同じTaskを一度だけ継続する。
- release: exact image内で全suiteを実行し、source/image/revision digestを新しいconformance reportへ結合する。

### 7.5 文書・主張への入力

- 現在主張できるのは「ローカルsimulation profileによる固定決済demoが、既存証跡の対象imageで成功した」ことまでである。
- 仲介統合、動的paid/free分岐、異常検知、再起動耐性、内部route非公開は、実装と新証跡が揃うまで主張しない。
- official x402は引き続きNOT RUNであり、simulation profileと混同しない。
- PR説明、ARCHITECTURE、REQUIREMENTS、SECURITY、VERIFICATION、OPERATIONS、DEMO、conformance reportは、新しい実装と証跡に合わせて同時更新する。
- PR #25をreadyのまま維持するかdraftへ戻すかはowner判断だが、少なくとも現状とPR titleの差、CI checkがないことを説明へ明記する。

## 8. 実装着手の判定

B-001からB-005は、schemaとsecurity境界を左右するため実装前に確定が必要である。B-006からB-008は、local実装を始めること自体は妨げないが、Cloud Run受入・release前には必須である。

最初の安全な実装単位は、既存決済ロジックを変更せず、次の縦切りをlocalで成立させることである。

1. 公開rootを永続controllerへ接続する。
2. matcher/plannerの出力をtyped snapshotとして保存する。
3. 厳密な計画承認後にRemote A2A Taskを開始する。
4. 無料応答はanomaly/final anomalyを通して完了する。
5. payment-required応答はRemote Taskを保存して停止し、まだ決済実行しない。

この段階で無料系と「支払待ちまで」の有料系を試験し、その後に既存payment workflowをbridgeとして接続する。これにより、仲介route復元の問題と決済証跡の問題を分離して検証できる。
