# AP2 / x402 simulation 実装証跡

- 証跡日: 2026-08-16（Asia/Tokyo）
- 最新Cloud Runイメージ: `sha256:68d6489c9091062e30c31d2b6287fb290c37c6bf94019683aaf4f3c274cc2529`（`linux/amd64`）
- 候補状態: `PASS`（Artifact Registryへpush済み、一時Cloud Run demoへdeploy済み）

## リリースで主張する範囲

この実装は、project-local profile `x402-wire-simulation/1` を使う **AP2 v0.2 Human Present demo** である。x402 について確認しているのは **x402 v0.1 wire-shape test fixture** だけであり、公式 x402 には **NOT CONFORMANT** である。決済 rail は価値を移動しないローカル simulation であり、wallet signature、facilitator verification、実資産移動、blockchain transaction、on-chain transaction hash は存在しない。

耐久性を受け入れ済みとする対象は、単一 host・単一 container の一つの process group で、`marketplace.db`、`paid-agent.db`、`evidence.db` に明示的な永続 POSIX mount を与えた構成である。複数 host／複数 container から SQLite を共有する構成は非対応である。

別サービス `payment-user-agent-demo` を、`EPHEMERAL_CLOUD_RUN_DEMO=true` の一時デモとしてCloud Runへデプロイした。この構成は状態消失の警告を表示し、耐久性を主張しない。Firebase認証後のremote browser full flowとreload後の状態復元までPASSした。

## 固定した依存関係の証跡

| コンポーネント | 固定値 | 結果 |
| --- | --- | --- |
| Python | 3.12 | PASS |
| AP2 SDK | `google-agentic-commerce/AP2@e1ea56db72a6385bce3e5c1112b3a56ce60acb43` | インストールし、リリースで使う経路を検証済み |
| A2A SDK | `a2a-sdk==0.3.19` | PASS |
| Google ADK | `google-adk==1.19.0` | PASS |
| Pydantic | `2.12.5`（AP2 の推移的固定） | PASS |
| cryptography | `46.0.5` | PASS |
| jwcrypto | `1.5.6` | PASS |
| canonical JSON | `rfc8785==0.1.4` | PASS |

リポジトリの互換性 spike は 11/11 PASS。固定した upstream AP2 suite は 186 PASS、2 FAIL だった。2件の失敗は中間 `kb+sd-jwt+kb` の audience／nonce negative test に限られる。固定ライブラリは、このリリースが使う終端 `kb+sd-jwt` 経路で expected-claim validation を行う。リポジトリ側の negative test では、その経路について誤った audience、誤った nonce、leaf signature の改ざん、誤った root issuer がすべて失敗することを確認した。upstream の2件は説明済みの SDK 挙動として記録し、PASS へ昇格していない。

## 実装したセキュリティ境界とワークフロー境界

- ADK の discovery directory に置く root は `payment_user_agent` だけである。これは決定論的で鍵を持たない `BaseAgent` adapter であり、耐久ワークフロー本体の `secure_mediation_agent` は内部依存として動作し、2個目の ADK app として公開しない。
- 認可と状態の正本は耐久 workflow aggregate であり、ADK session の boolean や CLI のローカル状態ではない。
- 計画承認と決済承認は、どちらも text part が一つだけで、その値が完全一致の `承認` である場合に限る。trim、Unicode normalization、part の連結、類義語変換、LLM による意図推定は行わない。
- Firebase login では ID token を same-origin かつ Origin／CSRF 検証付きの POST で交換する。server は project audience、issuer、subject、expiry を検証してから、`Secure`、`HttpOnly`、`SameSite=Strict`、`__Host-` 属性の session cookie を設定する。認証済み request は毎回再検証する。
- 公開する決済対応 route は `/mediation-api/` だけである。旧 `/payment/`、`/paid-agent/`、直接 `/v1/`、内部 route は 404 を返す。
- plan authorization と各 downstream operation は、audience／operation／task／plan に個別に拘束した ES256 capability を使い、それぞれ一度だけ消費する。
- 有料 Merchant は loopback-only の独立 A2A HTTP service `:8005` である。activation と audience／operation／workflow／task／order の capability binding を検証するまで、永続 TaskStore を変更できない。
- supervise された lease 方式の outbox worker が、耐久 handoff と各非終端 workflow checkpoint を再開する。evidence commit と intent acknowledgement の間で crash した場合は、cross-database evidence intent を exact digest で照合する。
- role ごとの鍵は mode `0600` の別 P-256 JWK file とする。耐久起動には永続 key mount が必須であり、使い捨て鍵を生成するのは明示的にラベル付けした一時デモだけである。各署名 artifact は immutable な公開鍵 trust snapshot を参照する。
- 公式 x402 の canonical URI は、無効化した fail-closed marker module にだけ置く。runtime の Agent Card と activation が宣言するのは `urn:secure-a2a:extensions:x402-wire-simulation:v1` だけである。

## 証跡グラフ

immutable evidence store は、計画、plan authorization、capability、元の Merchant Task、requirements、署名済み Checkout、closed Checkout／Payment Mandate、scoped credential、synthetic simulation authorization、selected-profile receipt、AP2 Payment／Checkout Receipt、最終 Task、reconciliation observation、公開鍵 trust snapshot の exact bytes と SHA-256 digest を保持する。

`scripts/verify_ap2_x402_evidence.py` はオフラインで次を再検証する。

1. 署名済み plan authorization と全 scoped capability。
2. Checkout の exact signature と workflow／plan／order／task／amount binding。
3. 終端 closed Checkout／Payment Mandate の signature、audience、nonce、payee、exact Checkout reference。
4. synthetic simulation proof の分類（`walletSigned=false`）、task binding、signature。
5. credential と Mandate／payload／requirements の binding。
6. role を分離した AP2 Receipt の signature と closed reference。
7. 順序付き simulation receipt と、on-chain transaction 形式が存在しないこと。
8. artifact と versioned trust-snapshot digest／`kid` の binding。

## テスト証跡

Python 3.12、`PYTHONDONTWRITEBYTECODE=1` で実行した。

```text
docker run ... enterprise-a2a-pf:ap2-payment-user-final \
  /app/.venv/bin/python -m pytest -p no:cacheprovider -q tests
166 passed
```

既存 payment-marketplace の回帰に加え、依存関係 spike、非完全一致承認の全指定パターンと multipart input、identity forgery／IDOR 境界、Merchant の全 private operation gate、workflow／task／order をまたぐ capability replay、profile／constraint drift、stale version と並行重複、全 durable payment checkpoint、orphan workflow recovery、evidence-intent crash recovery、role ごとの署名済み error receipt、出力の secret scan、無料 flow の回帰、historical v1 migration／restore、実 Chromium による ADK Web 二承認、外部 Merchant A2A、clean process での verifier import、offline evidence verification を検証した。

versioned regression manifest も、source bind mount なしの同一リリースイメージで PASS した。payment release は166件、evaluation runner は17件、jury worker は13件を収集した。jury suite は `GOOGLE_API_KEY` がないため明示 allowlist 済みの8件を skip したが、予期しない skip は0件である。再現性を保つため W&B は無効化した。

イメージ／container の証跡:

```text
docker build --no-cache -t enterprise-a2a-pf:ap2-payment-user-final .  PASS
release image digest                                      sha256:1b743079d533feb3e36a487bf6338799da8cdb3989523c69cdae904a8e9d5c29
/mediation-api/ready                                      11/11 checks ready
request -> 承認 -> 承認                                   completed
officialX402 / wallet / facilitator / onChain             NOT RUN
/payment/* / /paid-agent/*                                404 / 404
external Merchant :8005 / persistent TaskStore            PASS
outbox unfinished / evidence intents pending              0 / 0
container recreate with same DB/evidence/key mounts       completed restored
sanitized v1 three-DB migrate/E2E/recreate                 PASS; legacy rows unchanged
one-command runtime + offline verifier                     PASS
real Chromium ADK Web session                             request -> 承認 -> 承認 -> completed
actual process-death recovery matrix                       12/12 checkpoints PASS
integrated expiry/tamper/revoke/scope attack matrix        PASS; zero unauthorized effects
exact-image release validator v2                           PASS; all 11 marker suites bound
ephemeral local container gate                             PASS; target/durability/warning consistent
```

validator は上記 immutable image digest、release-manifest digest `sha256:701f2b1a66af81b9fe6dacefbb1a713fb35c181571621f8fda85296e363d3495`、regression-manifest digest `sha256:7d72de56a96a3f7438b539e0131167e3e7c9acd2c8e0fa204916dbdd7cfd7339`、組込み suite baseline、marker count、実 Chromium 証跡を相互に拘束する。`--conformance` 使用時は、conformance report内のimage digest、payment test exact count、release-manifest digest、指定したregression artifactのexact byte SHA、browser artifact SHAも完全一致しなければ失敗する。機械可読の出力は `artifacts/regression-result.json`、`artifacts/browser-evidence.json`、`artifacts/ap2-x402-release-validation.json` である。

## 運用コマンド

```text
scripts/provision_ap2_demo_keys.py <persistent-key-directory>
scripts/migrate_ap2_x402_v2.py plan|apply|verify ...
scripts/migrate_ap2_x402_v2.py restore-pre-cutover --manifest ... --confirm RESTORE-PRE-CUTOVER
scripts/verify_ap2_x402_evidence.py <workflow-id> ...
scripts/verify_payment_demo.sh
scripts/run_regression_manifest.py
scripts/validate_ap2_x402_release.py
deploy/run-local.sh
```

耐久ローカルmodeのreadinessは、data／evidence の両 durable-volume marker、三つの schema-v2 DB、worker heartbeat／outbox health、evidence-intent health、role key と permission、trust snapshot、固定した仕様の hash、selected profile の排他性、route isolation、Merchant A2A／TaskStore がすべて正常な場合にだけ成功する。ephemeral demo modeは `target=ephemeral-cloud-run-demo`、`durability=NOT PROVIDED`、state reset warningを返し、durable marker/checkを証拠として返さない。公式 enablement、wallet／facilitator verification、on-chain settlement は `NOT RUN` と報告する。

## Cloud Run 候補の配布拘束（最新）

release path は次の三段階に分離した。

1. `deploy/build-payment-demo-candidate.sh` は Git から可視な file だけで clean context を作り、`--platform linux/amd64 --no-cache --provenance=false --load` で一度だけ build する。source mount なしで組込み regression、実 Chromium、全11 marker、conformance validator を実行し、`artifacts/cloud-run-candidate.json` を `LOCAL_VALIDATED_NOT_PUSHED` として固定する。publish／deploy は行わない。
2. `deploy/push-payment-demo-candidate.sh` は local image ID、`linux/amd64`、source commit／worktree digest、各証跡の exact byte SHA、組込み suite 結果が一致する場合に限り、固定 Artifact Registry repository へ同一候補を push する。Cloud Run は操作しない。
3. `deploy/deploy-payment-demo-cloudrun.sh` は build／push を一切行わず、candidate status `PASS` と固定 repository の `@sha256:` reference が一致する場合にだけ NEW-only preflight 後の deploy を許す。

今回の固定値は次のとおりである。

| binding | value |
| --- | --- |
| source commit | `9730a597a3359f7ecac0f2bf10513a80f9b3c56e` |
| release worktree digest | `sha256:9c8de67d9fc5bbcf5d935b9af5be4991d9d0c550da6eb1a71127f1f8ce01a7f1`（232 files、path-mode-size-bytes-v1） |
| local image / platform | `sha256:68d6489c9091062e30c31d2b6287fb290c37c6bf94019683aaf4f3c274cc2529` / `linux/amd64` |
| regression artifact | `sha256:a5d65ccaa4de9634d1cf36bd1177d440bc4fd7787600d706da1c5d258ac2781e` |
| browser artifact | `sha256:e4fff3b9c823ef0eab5cea4e32a77e31cadb00c7f42064027d7ff7ecc0cbc7e8` |
| conformance report | `sha256:e22cf092ed1c79a087d3c697d8a567504c98ad0c8147fc5da5ef9bf58159d25f` |
| release validation | `sha256:e4bffbefe1c89e785bcdfcb3ce2a6b1d14fc9de3c349dfa87a4b7f9edffd1d30` |
| registry | `asia-northeast1-docker.pkg.dev/gen-lang-client-0585901015/secure-mediation-agent/payment-user-agent-demo@sha256:68d6489c9091062e30c31d2b6287fb290c37c6bf94019683aaf4f3c274cc2529` |

clean-context の実試験で、従来 `*.json` に隠れていた evaluation-runner の schema／prompt JSON も発見したため、公開 Firebase config、AP2/x402 spec manifest、regression／release manifest、conformance／evidence artifact と合わせて exact allowlist にした。`git check-ignore -q` は必須 JSON 全件で non-ignored を返し、clean-context image 内で payment 166、evaluation-runner 17、jury 13、実 Chromium 1、全11 marker が PASS した。CLI と verifier の認証 cookie は server と同じ `__Host-payment-session` に統一し、公開 `/mediation-api/` request の回帰テストを追加した。`deploy/run-local.sh` と関連 start／release script の executable mode も復元した。

Artifact Registryの固定repositoryへexact imageをpushし、新規Cloud Run service `payment-user-agent-demo`、revision `payment-user-agent-demo-00001-77d` をproject `gen-lang-client-0585901015`、region `asia-northeast1`へデプロイした。URLは `https://payment-user-agent-demo-343404053218.asia-northeast1.run.app`。環境は`EPHEMERAL_CLOUD_RUN_DEMO=true,APP_ENV=ephemeral-demo,DEV_MODE=false`、min/max instance 1である。`/health`、auth deployment/configはPASSし、未認証root／list-appsはloginへredirectした。

Firebase Authorized Domainsへ`payment-user-agent-demo-343404053218.asia-northeast1.run.app`を追加し、Emailログイン後に`/dev-ui/?app=payment_user_agent&session=...&userId=user`へredirectすることを確認した。root appは`payment_user_agent`だけで、公開remote browserの依頼→計画承認→決済承認→完了がPASSした。計画画面は未決済表示と1250 USD、決済画面は課金警告、Demo Merchant、`customerTotal=1250`、simulated／`NOT CONFORMANT`を表示した。完了画面はDemo booking confirmed、AP2 evidence、`AP2 v0.2 Human Present demo`、実資産／on-chainなしを表示し、reload後も認証・app選択・完了状態を維持した。revisionの直近30分のerror logは0件だった。

post-deploy gate は Cloud Run revision の `status.imageDigest` が返す full immutable URIを、candidateの `repository/image@sha256:...` full URIと完全一致で比較する。fake gcloud回帰ではexact full URIがexit 0、repositoryだけを変えたURIとdigestだけを変えたURIがともにexit 4となった。この修正を含むsourceから上記candidateを再buildし、全gateを再実行した。

## 意図的に実装・実証していない範囲

- 公式 x402 adapter、canonical URI activation、wallet／facilitator、network／token 選定、TLS conformance、on-chain settlement は **NOT RUN** であり、fail closed とする。
- 耐久 Cloud Run paid workflow は ephemeral storage のため **BLOCKED / NOT RUN**。デプロイ済みなのは状態と鍵が失われ得る一時デモだけである。
- 本番 identity enrollment と KMS／HSM はリリース対象外。identity と key は明示的に demo-only である。
- loopback Merchant 境界を越える外部 partner interoperability は主張しない。

機械可読の受入状態は `docs/ap2_x402_conformance_report.json` を参照する。`PARTIAL` と `NOT_RUN` は、指定された証跡なしに PASS へ昇格してはならない。
