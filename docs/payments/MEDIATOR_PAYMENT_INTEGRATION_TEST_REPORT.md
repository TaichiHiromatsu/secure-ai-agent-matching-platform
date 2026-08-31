# Mediator Payment Integration Test Report — final6 + Cloud Run acceptance

- 実施日: local final6 2026-08-17 JST、Cloud addendum 2026-08-30 JST（実行ログはUTC）
- 対象branch: `codex/ap2-x402-integration`
- 対象: 共有working treeから構築したexact `linux/amd64` image
- image tag: `enterprise-a2a-pf:full-test-final6-20260817-amd64`
- image ID／digest: `sha256:3e4e089643564e00bd6563d08a575fcf2aa2eff94ae60fd1ca518900022a89f0`
- image created: `2026-08-16T17:28:52.879370711Z`
- Cloud Run: **EPHEMERAL DEMO DEPLOYED／NORMAL PAID AND FREE PASS**
- local判定: **LOCAL SIMULATION DEMO CANDIDATE VERIFIED**
- promotion判定: **CLOUD RUN EPHEMERAL DEMO PROMOTED／NOT PRODUCTION PAYMENT**

## 1. 結論

final6 exact imageは、paid、free、refund、privacyのreal Chromium 4 case、canonical regression 3 suite、11 required marker release validatorをすべてPASSした。public mutationは一つのowner-scoped mediation authorityへ統合され、local durable profileではauthoritative mediation sessionとrequest replayがSQLite schema v4へ保存される。

restart検証では `WaitingForPaymentApproval` v2の完全一致復元、exact payment approvalによる`Completed` v5、同一requestのexact replay、二回目container restart後の同一terminal viewを確認した。三つのSQLite DBは`quick_check=ok`、authoritative business count不変、wrong-owner viewはHTTP 200のJSON `null`だった。final3で観測された`InMemoryMediationStore`によるrestart後view消失はfinal6には該当しない。

環境変数なしのraw full pytestは304 PASS／3 FAIL／8 skipで、3 FAILはすべてevaluation-runnerがW&B API keyをnon-TTYで要求する既知問題だった。canonical release契約は各suiteへ`WANDB_DISABLED=true`を設定し、evaluation 17/17を含めてPASSする。

2026-08-30のCloud Run受入では、Vertex ADC、実Firebase login、live external A2Aを使ったpaid／free正常系、paid reload、logout、callback順序、Cloud logを検証し、対象revisionへ100% trafficを切り替えた。これはephemeral simulation demoの正常系受入であり、production durability、実資産移転、official x402／on-chain適合、全139要件release closureを意味しない。refundは最終正常系hotfix後にCloudで再実行しておらず、既存local coverageと区別する。

## 2. Exact candidate binding

| Evidence | Status | SHA-256／binding |
| --- | --- | --- |
| exact image | present | `sha256:3e4e089643564e00bd6563d08a575fcf2aa2eff94ae60fd1ca518900022a89f0` |
| regression manifest | bound | `sha256:7d72de56a96a3f7438b539e0131167e3e7c9acd2c8e0fa204916dbdd7cfd7339` |
| release manifest | bound | `sha256:852aeaba0e024469eb35adfa45a1dd6fabd054484d68aa1b58739ddaf8457f37` |
| `artifacts/regression-result-final6.json` | `PASS` | file `f64da6ec882b3a6a14f27a8df5448ad971c01c208c3f8bcf6070335edfa84ded` |
| `artifacts/browser-evidence-final6.json` | `PASS` | file `1059985e2fac45b8c7c70ed316e2359d1c6da64acc004ebf0207560a3796fa50` |
| `artifacts/ap2-x402-release-validation-final6.json` | `PASS`／`failures={}` | file `4f4aa723d9a5bc02eec4c09d6f097c749f2d6f6652c66f0dc2b0a72573cf96ce` |

regression、browser、release-validationの三artifactは同じexact image digestとrelease manifest digestを持つ。release validatorはregression／browser artifactをread-only入力として通常の11-marker手順で再生成した。optional conformance inputは指定しておらず、`conformanceReportDigest=null`である。

## 3. Canonical regression

source mountなしでexact imageの `/app/scripts/run_regression_manifest.py` を実行した。

| Suite | Collected | Pass | Skip | Fail／error | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| payment-release | 285 | 285 | 0 | 0 | `PASS` |
| evaluation-runner | 17 | 17 | 0 | 0 | `PASS` |
| jury-worker | 13 | 5 | 8 allowed | 0 | `PASS` |

juryの8 skipはGoogle ADK／live credential系としてmanifestに明示され、unexpected skipは0だった。canonical runnerはW&Bをmock／offlineにせず、release契約どおり`WANDB_DISABLED=true`で無効化する。

## 4. 11-marker release validator

`validate_ap2_x402_release.py`をexact image内から実行し、既存final6 regression／browser artifactとexpected image digestを結合した。結果は`status=PASS`、`failures={}`である。

| Marker | Collected | Fail | Error | Skip／xfail |
| --- | ---: | ---: | ---: | ---: |
| spike | 11 | 0 | 0 | 0 |
| unit | 11 | 0 | 0 | 0 |
| contract_ap2 | 17 | 0 | 0 | 0 |
| contract_x402_simulation | 2 | 0 | 0 | 0 |
| integration | 56 | 0 | 0 | 0 |
| security | 84 | 0 | 0 | 0 |
| restart | 41 | 0 | 0 | 0 |
| migration | 5 | 0 | 0 | 0 |
| concurrency | 4 | 0 | 0 | 0 |
| container | 16 | 0 | 0 | 0 |
| browser | 4 | 0 | 0 | 0 |

official x402とon-chainはvalidator outputでも`NOT RUN`であり、simulation境界を維持する。

## 5. Real Chromium evidence

browser artifactはpublic nginx、packaged Chromium／CDP、local deterministic profileを使い、次の4 caseを同じimage digestへ結合した。

| Case | Oracle | Result |
| --- | --- | --- |
| paid | 計画とclosed payment targetを別々に完全一致承認、same Task resume、`Completed`、refresh後維持 | `PASS` |
| free | 計画承認後にpayment recordを作らず`Completed` | `PASS` |
| refund | exact one-shot Merchant fault、`RefundPending`、別承認、相関付き`Refunded` | `PASS` |
| privacy | DOM、console、Resource Timing、network bodyにprivate material 0 | `PASS` |

artifactの`privateMaterialExposed=false`、`completedAfterRefresh=true`、app list=`["payment_user_agent"]`をrelease validatorが再検査した。

## 6. Durable runtimeとrestart

既存runtimeやvolumeには触れず、final6専用containerと次の専用named volumeを使った。

- `enterprise-a2a-final6-payment-data`
- `enterprise-a2a-final6-payment-evidence`
- `enterprise-a2a-final6-ap2-keys`

runtime profileは`APP_ENV=local`、`DEV_MODE=true`、`MEDIATION_STORE_MODE=sqlite`、local deterministic callback／agent、local-only Merchant faultである。readinessはHTTP 200で、target=`explicit-durable-single-host-single-container`、store=`sqlite/local-durable/schema4`、writable／decryptable、schema、worker／outbox、simulation profile checkがすべてtrueだった。

### 6.1 Paused-view restart

対象sessionは`med-773861de-3015-468d-996a-d04751a34f14`。

| Item | Value |
| --- | --- |
| state／version | `WaitingForPaymentApproval` / `2` |
| saved artifact digest | `sha256:795c544b41632a01763ccac544a480e851f173c88fbb87ff2f43b6d7dac91d1d` |
| sorted saved-view digest | `sha256:3637cc151ea9b6e6cfd56a974b58350991b2a8e3edf1c330f112e278f222876e` |
| first restart | restored view equals saved JSON exactly |

復元したexact targetへ承認を送り、同一Taskを再開して`Completed` v5へ到達した。同じrequestをreplayすると新しい副作用を作らず、同じterminal resultをexactに返した。

### 6.2 Terminal-view second restart

二回目のcontainer restart後もreadinessはHTTP 200へ戻り、同じsessionは`Completed` v5だった。compact sorted UTF-8 terminal digestはrestart前後とも次の値で一致した。

```text
sha256:876cbc34199bb854adf0bf2ec34b000887aa5eadee990fe34b64261d1efec44b
```

`jq -cS`の末尾改行込み比較hashも前後とも`2b4ae17c8c10447082af4af95697642390705402e7f4a9caef88edddb4c9f794`だった。

## 7. IsolationとSQLite integrity

別subject `wrong-owner-final6` の有効なsigned assertionで同じviewを問い合わせた結果はHTTP 200のJSON `null`だった。record存在、owner、state、versionを漏らさない。

marketplace、Merchant、evidenceの三DBで`PRAGMA quick_check`はすべて`ok`だった。二回目restart前後でauthoritative business countは不変で、business-count digestは次の値だった。

```text
sha256:b2bdd4a1f1488061c155f77e730a2dd337fa73d02dcd684e1c81f0fb4c47636d
```

主なstable countは次のとおりである。

- marketplace: mediation requests 16、sessions 4、approvals 3、outbox 7、refunds 1、settlements 3、continuations 3、guarantees 3、evidence intents 15。
- Merchant: consumptions 3、guarantees 3、messages 5、requirements 3、tasks 3。
- evidence: evidence 15、access events 6。

`worker_heartbeats`だけが1から2へ増えた。これはcontainer restartで新しいworker identityが起動するoperational rowであり、authoritative business duplicateではない。

## 8. Raw full pytestとW&B分類

exact imageで環境変数、除外、source mountを与えず、literal `/app/.venv/bin/python -m pytest -q` を実行した。

```text
304 passed, 3 failed, 8 skipped, 0 collection errors in 86.82s
```

3 failureはすべて`evaluation-runner/tests/test_cli.py`である。

- `test_cli_generates_artifacts`
- `test_cli_handles_missing_security_dataset`
- `test_cli_handles_missing_agent_card`

いずれも`wandb.init()`がnon-TTYでAPI keyを要求し、`wandb.errors.UsageError: api_key not configured (no-tty)`を返した。payment、browser、restart、security、migration、container、collectionのfailureではない。これは既存evaluation-runnerの未修復問題であり、raw full greenは未達である。

## 9. 139 requirement coverage

統合要件の規範H3は139 IDであり、design coverage YAMLも同じ139 IDをexactly onceで持つ。scopeはRelease-1必須126、future-work 13で排他的である。

ただしfinal6のthree artifactはsuite／browser／markerをimageへ結ぶもので、各126 IDを一件ずつcandidate evidenceへ結ぶ139行のrelease ledgerではない。このためdesign mappingのrequired recordは`PARTIAL`、future-workは`DESIGNED`とし、全139要件のrelease closureをPASSとしない。

## 10. NOT RUN／claim boundary

| Boundary | Status |
| --- | --- |
| 実Firebase credential／ID token exchange | `PASS` — credential値は証跡へ保存しない |
| Vertex ADC／model availability probe | `PASS` — current default service accountで許可model 2件を確認。IAM least-privilegeは未完了 |
| official x402 wallet／facilitator | `NOT RUN` |
| on-chain settlement／refund | `NOT RUN` |
| Cloud Run build／push／revision／tag／traffic | `PASS` — exact deployment evidence参照 |
| candidate-bound conformance report | `NOT RUN` (`conformanceReportDigest=null`) |
| future-work 13の高度recovery／競合／edge matrix | `NOT RUN`／`DESIGNED` |

Cloud Runはephemeral 4項目とVertex ADC 3項目からなるexact 7 non-secret envを使い、`durability=NOT PROVIDED`を表示する。local SQLite v4 restart PASSをCloud Run durabilityへ転用しない。

## 11. Cleanup

検証終了後、次だけを削除し、存在しないことを確認した。

- container `enterprise-a2a-final6-runtime-1`
- volume `enterprise-a2a-final6-payment-data`
- volume `enterprise-a2a-final6-payment-evidence`
- volume `enterprise-a2a-final6-ap2-keys`

exact imageとthree final6 evidence artifactは保持した。このcleanup記録は2026-08-17のlocal final6実行だけを対象とし、後日のCloud Run deployment操作を否定するものではない。

## 12. 最終判定

| Claim | Decision |
| --- | --- |
| local paid/free/refund/privacy simulation | **VERIFIED** |
| local single-host SQLite v4 restart／replay／isolation | **VERIFIED** |
| exact-image canonical regression／browser／release validator | **VERIFIED** |
| raw full repository green | **NO** — known W&B 3 failures |
| all 139 requirements candidate closure | **PARTIAL** — per-ID ledger未完了 |
| Cloud Run demo deployed／observed | **VERIFIED** — ephemeral normal paid/free only |
| production durability | **NOT PROVIDED** |
| official x402／wallet／facilitator／on-chain conformance | **NOT RUN／NOT CONFORMANT** |

final6はlocal simulation baselineとして、後続のCloud Run revisionはephemeral normal-flow demoとして利用できる。いずれもproduction-ready、実決済、official x402適合ではない。per-ID ledger、raw W&B問題、production durability、公式profile、最終hotfix後のCloud refund再実行は未完了である。

## 13. Cloud Run acceptance addendum（2026-09-01 JST）

### 13.1 Exact deployment binding

| Item | Value |
| --- | --- |
| source commit | `69cea4da680f3d61d5d2c05d747e90111ca7a829` |
| image digest | `sha256:81f3f41940c5ecc4a0f11476d84b6e6f5fa772eb930ed8d0a8e84316af88e646` |
| revision | `payment-user-agent-demo-00015-foj` |
| traffic | `100%` |
| candidate evidence | `artifacts/cloud-run-tag-e2e-81f3f41940c5.json`／SHA-256 `46ff561b67ce87ccef44b44ea9dcaef2d93132213b8619ab5c1f036f31849d09`／3001 bytes |
| deployment evidence | `artifacts/cloud-run-deployment-81f3f41940c5.json`／SHA-256 `ba159cc5d73f9216a7f2cac04ca1c4bc954c8c6ffda5e8972e39ede8d982aa1e`／7252 bytes |
| public paid browser evidence | `artifacts/cloud-run-public-paid-81f3f419.json`／SHA-256 `c961bc6f0c367c986026f9d9850f0a4a92acd3729ba41cb94eab5311522798df`／1954 bytes |
| public free browser evidence | `artifacts/cloud-run-public-free-81f3f419.json`／SHA-256 `57c85de9aaae3445266586de3d5ec348c31115e98936c51bb15ea941e74f5eed`／1876 bytes |
| canonical exact-7 environment | SHA-256 `sha256:5b9f4a885039fe3684555a5ec6c359018a5a9b7d394f1c91796dca5ad6dd723c`／actual equals expected |

revisionはsingle container、memory ephemeral store、volumeなし、secret envなし、exact 7 user envでreadyとなった。deployment evidenceは期待allowlistのsource、期待値と実revision値のcanonical配列、両者のdigest、missing／unexpectedの空配列、secret reference 0、API-key-like env 0、`actualMatchesExpected=true`を保持するため、第三者がexact 7を再計算できる。release manifestはpayment release 371件、evaluation runner 17件、jury worker 13件をPASSし、exact image validatorの11 marker suiteは295件をPASSした。candidateと公開後のCloud観測windowはいずれもHTTP 5xxとerror severityが0だった。service account provenanceはdefault Compute、`dedicatedLeastPrivilegeServiceAccount=false`で、今回IAM／bindingを変更していない。

表のsource commitは実際にbuild・deployしたcode commitである。PRの最終headは、このdeploy後に生成した機械可読evidenceと本書の更新commitを追加するため別SHAになる。最終headを同じimageとして扱わず、公開revisionのprovenanceはsource commit、source digest、immutable image digestの組で追跡する。docs/evidenceだけの後続commitはこのrevisionへ再deployしていない。

環境digestは、`actualRevisionEnvironment`を`name`で昇順にし、JSON object keyも整列したcompact JSONを末尾改行なしでSHA-256へ入力する。次の再計算結果が表のcanonical digestと一致する。

```bash
jq -cS '.configuration.environment.actualRevisionEnvironment | sort_by(.name)' \
  artifacts/cloud-run-deployment-81f3f41940c5.json | tr -d '\n' | shasum -a 256
```

### 13.2 Vertex ADCと実ブラウザ正常系

pre-trafficの一時probeで`gemini-2.5-flash`と`gemini-2.5-pro`がいずれもPASSし、probe jobは削除した。probeはcandidateと同じimmutable image、revision identity、default runtime service accountからVertex ADCの`count_tokens`だけを実行した。API key、生成prompt、model outputを使用・保存していない。

fresh headless Chromium／CDPでFirebase reviewer login後に次を確認した。

| Case | Result |
| --- | --- |
| paid | canonical日本語prompt、agent-005、計画承認、固定scenario支払表示、決済承認、`WaitingForPlanApproval -> WaitingForPaymentApproval -> Completed`、厳格なデモ確認Artifact、reload後もCompleted、callback 3 operation | `PASS` |
| free | canonical日付入り日本語prompt、agent-002、計画承認だけ、payment approval／paid dataなし、`WaitingForPlanApproval -> Completed`、callback 1 operation | `PASS` |
| callback order | 各operationで`legacy-callback-before -> transport -> response-persisted -> legacy-callback-after` | `PASS` |
| fresh isolation | paid／freeを別profile・別loginで実行し、各caseでlogoutとcontainer終了時cleanup | `PASS` |
| post-promotion | 通常service URLでpaid／free、readiness、reload、logoutを再実行 | `PASS` |

reason codeは公開されず、reviewer credential、任意の利用者prompt、model output、browser console、network body、screenshotは保存していない。公開デモのcanonical日本語promptだけはrelease manifestで定義された非機密入力としてlocal browser artifactに保持する。

### 13.3 Claim boundaryと残課題

- Cloud Run storeとlive external AgentのTaskStoreはephemeralであり、revision／process再起動後の状態保持を保証しない。
- Payment authorization、仲介保証、settlementはsimulationであり、実資産移転はない。
- official x402 network path、wallet／facilitator、on-chain settlementは`NOT RUN`である。
- refundは既存local browser coverageがあるが、最終正常系hotfix後のCloud revisionでは再実行していない。
- current stable modelで正常系を確認したが、将来のmodel廃止／移行、IAM、quota、latencyの再検証は各releaseで必要である。
- default Compute service accountから専用least-privilege service accountへの移行と不要権限除去は未完了のsecurity debtである。今回のVertex probe PASSをIAM hardening完了とは扱わない。
- 最初のpost-push validatorは、計画表示直後にhostがsleepし、実時間で承認期限を越えたため正しく`Blocked`となった。sleepを抑止したfresh exact-image再実行は全suite PASSであり、Cloudへはその再検証後にだけdeployした。
- release helperは旧100% revisionが同時にtraffic tagを持つCloud Run表現をrollback revisionとして認識できず、変更前に安全停止した。今回のno-traffic deployと明示100% shiftは同じ固定引数を個別実行し、before／afterでrevision、digest、trafficを照合した。helperのtagged-default対応は今後の運用改善事項である。
