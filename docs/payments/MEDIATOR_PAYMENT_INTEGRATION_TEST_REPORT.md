# Mediator Payment Integration Test Report — final6

- 実施日: 2026-08-17 JST（実行ログはUTC）
- 対象branch: `codex/ap2-x402-integration`
- 対象: 共有working treeから構築したexact `linux/amd64` image
- image tag: `enterprise-a2a-pf:full-test-final6-20260817-amd64`
- image ID／digest: `sha256:3e4e089643564e00bd6563d08a575fcf2aa2eff94ae60fd1ca518900022a89f0`
- image created: `2026-08-16T17:28:52.879370711Z`
- Cloud Run: **NOT RUN／未操作**
- local判定: **LOCAL SIMULATION DEMO CANDIDATE VERIFIED**
- promotion判定: **NOT READY FOR CLOUD RUN OR PRODUCTION PROMOTION**

## 1. 結論

final6 exact imageは、paid、free、refund、privacyのreal Chromium 4 case、canonical regression 3 suite、11 required marker release validatorをすべてPASSした。public mutationは一つのowner-scoped mediation authorityへ統合され、local durable profileではauthoritative mediation sessionとrequest replayがSQLite schema v4へ保存される。

restart検証では `WaitingForPaymentApproval` v2の完全一致復元、exact payment approvalによる`Completed` v5、同一requestのexact replay、二回目container restart後の同一terminal viewを確認した。三つのSQLite DBは`quick_check=ok`、authoritative business count不変、wrong-owner viewはHTTP 200のJSON `null`だった。final3で観測された`InMemoryMediationStore`によるrestart後view消失はfinal6には該当しない。

環境変数なしのraw full pytestは304 PASS／3 FAIL／8 skipで、3 FAILはすべてevaluation-runnerがW&B API keyをnon-TTYで要求する既知問題だった。canonical release契約は各suiteへ`WANDB_DISABLED=true`を設定し、evaluation 17/17を含めてPASSする。

ただし実Firebase、Vertex、official x402 wallet／facilitator、on-chain、実Cloud Run、candidate-bound conformance report、126件のrequirement-specific PASS ledgerは未実行または未完了である。したがってlocal simulationのcandidate証跡は成立するが、Cloud Run promotion、production durability、official適合、全139要件release closureは成立しない。

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
| 実Firebase credential／ID token exchange | `NOT RUN` |
| Vertex ADC／IAM／quota／model probe | `NOT RUN` |
| official x402 wallet／facilitator | `NOT RUN` |
| on-chain settlement／refund | `NOT RUN` |
| Cloud Run build／push／revision／tag／traffic | `NOT RUN` |
| candidate-bound conformance report | `NOT RUN` (`conformanceReportDigest=null`) |
| future-work 13の高度recovery／競合／edge matrix | `NOT RUN`／`DESIGNED` |

Cloud Runを使う場合は`EPHEMERAL_CLOUD_RUN_DEMO=true`と`MEDIATION_STORE_MODE=memory`を必須にし、`durability=NOT PROVIDED`を表示する。local SQLite v4 restart PASSをCloud Run durabilityへ転用しない。

## 11. Cleanup

検証終了後、次だけを削除し、存在しないことを確認した。

- container `enterprise-a2a-final6-runtime-1`
- volume `enterprise-a2a-final6-payment-data`
- volume `enterprise-a2a-final6-payment-evidence`
- volume `enterprise-a2a-final6-ap2-keys`

exact imageとthree final6 evidence artifactは保持した。他container、他volume、Cloud Run、code、TEST REPORT以外のruntime stateを変更していない。

## 12. 最終判定

| Claim | Decision |
| --- | --- |
| local paid/free/refund/privacy simulation | **VERIFIED** |
| local single-host SQLite v4 restart／replay／isolation | **VERIFIED** |
| exact-image canonical regression／browser／release validator | **VERIFIED** |
| raw full repository green | **NO** — known W&B 3 failures |
| all 139 requirements candidate closure | **PARTIAL** — per-ID ledger未完了 |
| Cloud Run demo deployed／observed | **NOT RUN** |
| production durability | **NOT PROVIDED** |
| official x402／wallet／facilitator／on-chain conformance | **NOT RUN／NOT CONFORMANT** |

final6はlocal simulation demo candidateとして利用できる。外部gate、per-ID ledger、raw W&B問題を閉じるまでは、Cloud Run promotionまたはproduction-readyを承認しない。
