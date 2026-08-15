# AP2 v0.2 / A2A x402 v0.1 統合仲介 — 独立テスト報告

- 実施日時: 2026-08-16 (Asia/Tokyo)
- 対象 branch: `codex/ap2-x402-integration`
- 対象工程: Section 12 Step 9（テストのみ）
- 最新判定: **この文書の末尾にある最新再試験節を参照**。古い判定は履歴として残し、後続節で置換する。
- 初回判定（§1〜§8）: **FAIL — 後続の再試験で置換済み**。§11、§12、§13 の各中間判定も §14 で置換済み。
- テスト担当による実装変更: 各試験節に記載したとおり。初回試験では本書以外の production／test code を変更していない。
- official x402/on-chain: **DISABLED / NOT RUN**
- Cloud Run paid: **BLOCKED / NOT RUN**

## 1. 初回試験の結論（§12で置換済み）

clean Python 3.12 image の build、repository 内 `tests/` の 98 test、主要 marker のうち収集された suite、および独立に構築した明示的 persistent-volume container の HTTP happy path は成功した。container では request → exact `承認` → payment view → exact `承認` → `completed`、副作用一回性、route isolation、simulation label、offline evidence verification、container recreation recovery、非 exact 承認、idempotency tamper、payment rejection、parallel payment approval、CLI を確認した。

ただし release criteria は満たさない。必須 `unit` / `container` marker が 0 collected、browser suite が存在せず、repository 全体の regression collection と evaluation-runner が失敗し、migrated historical volume、全 nonterminal crash、full bypass、replan、role rejection、cross-scope replay、full output redaction の証拠がない。さらに completed workflow に `outbox.status='pending'` が残り、supervisor に outbox worker がなく、readiness は worker/outbox/spec/key/trust/route self-check を確認せず `ready` を返した。これは restart/durable handoff acceptance の実装上の blocker である。

## 2. 環境と固定した基準点

| 項目 | 結果 |
| --- | --- |
| host | macOS; host `python3` 3.9.6、host `uv` なし |
| Docker | client/server 29.5.3 |
| clean image | `enterprise-a2a-pf:ap2-independent-test` |
| image ID | `sha256:ed6631916f5ac966607c31984407b0436d6ba143594585512cc5ce73f2e1494b` |
| Python in image | 3.12 |
| AP2 | `0.1`, Git commit `e1ea56db72a6385bce3e5c1112b3a56ce60acb43` |
| A2A SDK / ADK | `a2a-sdk==0.3.19` / `google-adk==1.19.0` |
| crypto/model/JCS | `cryptography==46.0.5`, `jwcrypto==1.5.6`, `pydantic==2.12.5`, `rfc8785==0.1.4` |
| AP2 spec SHA-256 | `32c3be5011f481d2760e56e7b9935b0842c3da0d5f7d7b8a68402a599f1e6aa3` — PASS |
| x402 spec SHA-256 | `5cdc35ed8c4d7a93bb120f1782fd06e2cc3ef19036684f772e27d0d644c66940` — PASS |

主な exact command:

```bash
docker build --no-cache -t enterprise-a2a-pf:ap2-independent-test .
docker image inspect enterprise-a2a-pf:ap2-independent-test --format '{{.Id}}|{{.RepoDigests}}'
docker run --rm --entrypoint /app/.venv/bin/python enterprise-a2a-pf:ap2-independent-test \
  -c 'import importlib.metadata as m; print({n:m.version(n) for n in ["ap2","a2a-sdk","google-adk","pydantic","cryptography","jwcrypto","rfc8785","pytest"]})'
git clone --filter=blob:none --no-checkout https://github.com/google-agentic-commerce/AP2.git /tmp/ap2-upstream-independent.nAOh2X
git -C /tmp/ap2-upstream-independent.nAOh2X checkout --detach e1ea56db72a6385bce3e5c1112b3a56ce60acb43
shasum -a 256 /tmp/ap2-upstream-independent.nAOh2X/docs/ap2/specification.md
git clone --filter=blob:none --no-checkout https://github.com/google-agentic-commerce/a2a-x402.git /tmp/a2a-x402-upstream-independent.nAOh2X
git -C /tmp/a2a-x402-upstream-independent.nAOh2X checkout --detach 125db5526a965d2325459d1a9df2e274a7e42396
shasum -a 256 /tmp/a2a-x402-upstream-independent.nAOh2X/spec/v0.1/spec.md
```

結果: clean no-cache build PASS（49.3s）。AP2 installed distribution の `direct_url.json` も exact commit を示した。

## 3. テスト収集、skip、xfail、suite結果

全 repository test command は次の read-only source mount と Python 3.12 image を使用した。

```bash
docker run --rm --entrypoint /app/.venv/bin/python \
  -v "$PWD:/work:ro" -w /work -e PYTHONPATH=/work \
  -e PYTHONDONTWRITEBYTECODE=1 enterprise-a2a-pf:ap2-independent-test \
  -m pytest -p no:cacheprovider -q -ra tests
```

結果: **98 passed, 0 skipped, 0 xfailed, 1 dependency warning, 7.77s**。

marker command は上記末尾を `-m <marker> tests` に置換した。

| marker | result | 判定 |
| --- | ---: | --- |
| `spike` | 11 passed, 87 deselected | PASS |
| `unit` | 0 collected, 98 deselected, exit 5 | **FAIL** |
| `contract_ap2` | 10 passed, 88 deselected | PASS for repository-used path |
| `contract_x402_simulation` | 2 passed, 96 deselected | PASS だが coverage は限定的 |
| `integration` | 7 passed, 91 deselected | PASS |
| `security` | 5 passed, 93 deselected | PASS だが static boundary 中心 |
| `restart` | 5 passed, 93 deselected | PASS だが all-state crash ではない |
| `migration` | 3 passed, 95 deselected | PASS だが historical v1 fixture なし |
| `concurrency` | 2 passed, 96 deselected | PASS |
| `container` | 0 collected, 98 deselected, exit 5 | **FAIL** |
| `x402_onchain` | 0 collected, exit 5 | **NOT RUN（意図どおり。PASS へ昇格しない）** |

root 全体の collection:

```bash
docker run --rm --entrypoint /app/.venv/bin/python \
  -v "$PWD:/work:ro" -w /work -e PYTHONPATH=/work \
  enterprise-a2a-pf:ap2-independent-test -m pytest --collect-only -q
```

結果: 124 tests collected の後、`trusted_agent_store/jury-judge-worker/tests/test_e2e_judge_panel.py` が `ModuleNotFoundError: jury_judge_worker.judge_orchestrator` で collection error。

追加 regression:

| command | result |
| --- | --- |
| `... pytest -q -ra trusted_agent_store/evaluation-runner/tests` | **10 passed, 6 failed**。4件は `schemas/response_sample.schema.json` 欠落、2件は compressor expected mismatch |
| `... pytest -q -ra trusted_agent_store/jury-judge-worker/tests` | **collection ERROR**。`judge_orchestrator` 欠落 |
| jury の non-E2E 2 files | 2 passed, 7 skipped。7 skip は `GOOGLE_API_KEY not set` |

したがって「同一 baseline/final regression manifest、collection 減少なし、unexpected skip/xfail なし」は証明されていない。`tests/regression/suite_manifest.json`、`scripts/run_regression_manifest.py`、`scripts/validate_ap2_x402_release.py` は存在しない。

### upstream契約の確認

```bash
docker run --rm --entrypoint /app/.venv/bin/python \
  -v /tmp/ap2-upstream-independent.nAOh2X:/ap2:ro -w /ap2 \
  -e PYTHONPATH=/ap2/code/sdk/python enterprise-a2a-pf:ap2-independent-test \
  -m pytest -p no:cacheprovider -q -ra code/sdk/python/ap2/tests
```

結果: **186 passed, 2 failed**。失敗は upstream `kb_sd_jwt_intermediate_tests.py` の audience mismatch / nonce mismatch negative test。release-used terminal path の repository negative tests は pass したが、upstream 全件 green ではない。

```bash
docker run --rm \
  -v /tmp/a2a-x402-upstream-independent.nAOh2X/python/x402_a2a:/src:ro -w /src \
  python:3.12-slim /bin/sh -c \
  'pip install -q -e . pytest pytest-asyncio a2a-sdk==0.3.19 x402==0.2.0 && pytest -p no:cacheprovider -q -ra tests'
```

結果: **3 passed**。`x402` を unconstrained latest で解決した試行は `x402.types` import error、`a2a-sdk` latest の試行は `TextPart` import errorとなったため、reference fixture は `x402==0.2.0` と repository baseline `a2a-sdk==0.3.19` の組合せで実行した。これは wire/reference unit fixture であり、wallet/facilitator/on-chain execution ではない。

## 4. diff、JSON、shell、静的検査、import確認

| command | result |
| --- | --- |
| `git diff --check` | PASS |
| 全 `*.json` に `jq empty` | PASS |
| 全 `*.sh` に `bash -n` | PASS |
| Python AST parse of `secure_mediation_agent`, paid agent, scripts, auth | `AST_OK=76` |
| 主要 workflow/AP2/profile/merchant module import | `IMPORT_OK=16` |
| canonical x402 URI runtime scan | disabled `secure_mediation_agent/payment_profiles/x402_v01.py` の定数だけ |
| root import-boundary scan | `secure_mediation_agent/agent.py` / subagents から key/repository import なし |
| private PEM source scan | hit 0 |

注意: legacy `external-agents/paid-booking-agent` source には old combined URN、`platform-credit`、`x402Version: 2`、guarantee/payout が残る。current nginx/supervisor path からは公開されないが、新 paid workflow はこの external service を使用せず in-process Merchant serviceを呼ぶため、reviewed external Merchant A2A boundary の実行証拠にはならない。

## 5. 独立した永続volumeのcontainer E2E

### 5.1 証跡path

- persistent test root: `/tmp/ap2-x402-independent-e2e.nAOh2X`
- mediation DB: `/tmp/ap2-x402-independent-e2e.nAOh2X/data/marketplace.db`
- Merchant DB: `/tmp/ap2-x402-independent-e2e.nAOh2X/data/paid-agent.db`
- evidence DB: `/tmp/ap2-x402-independent-e2e.nAOh2X/evidence/evidence.db`
- role keys: `/tmp/ap2-x402-independent-e2e.nAOh2X/keys`（8 files、全て mode `0600`）
- pinned AP2 checkout: `/tmp/ap2-upstream-independent.nAOh2X`
- pinned x402 checkout: `/tmp/a2a-x402-upstream-independent.nAOh2X`

test container `ap2-independent-e2e` は終了・削除済み。DB/evidence/key volume と image は残しており再現可能。

### 5.2 準備、起動、readiness

```bash
mkdir -p /tmp/ap2-x402-independent-e2e.nAOh2X/{data,evidence,keys}
install -m 600 /dev/null /tmp/ap2-x402-independent-e2e.nAOh2X/data/.durable-volume
install -m 600 /dev/null /tmp/ap2-x402-independent-e2e.nAOh2X/evidence/.durable-volume
docker run --rm --entrypoint /app/.venv/bin/python \
  -v /tmp/ap2-x402-independent-e2e.nAOh2X/keys:/keys \
  enterprise-a2a-pf:ap2-independent-test /app/scripts/provision_ap2_demo_keys.py /keys
docker run -d --name ap2-independent-e2e -p 28080:8080 \
  -v /tmp/ap2-x402-independent-e2e.nAOh2X/data:/app/payment-data \
  -v /tmp/ap2-x402-independent-e2e.nAOh2X/evidence:/app/payment-evidence \
  -v /tmp/ap2-x402-independent-e2e.nAOh2X/keys:/run/secrets/ap2-demo:ro \
  -e DEV_MODE=true enterprise-a2a-pf:ap2-independent-test
curl -sS http://127.0.0.1:28080/mediation-api/ready | jq .
```

結果: HTTP 200、schema は marketplace/merchant/evidence 全て v2、durable marker PASS、profile `x402-wire-simulation/1`、rail `simulated`。`officialX402`、wallet、facilitator、onChain は全て **NOT RUN**。

`curl http://127.0.0.1:28080/list-apps` は `['secure_mediation_agent']` のみ。OpenAPI paths は `/health`, `/ready`, workflow create/active/get/message のみ。

### 5.3 正常系、完全一致承認、副作用件数

HTTP calls は全て public `http://127.0.0.1:28080/mediation-api/` を使用した。

1. `POST /v1/workflows` → HTTP 200、`plan_approval_required`, version 1。
   - counts: workflow 1、plan/payment approval 0、Merchant Task 0、payment artifact 0、settlement 0、fulfillment 0。
2. plan pending で text `承認 ` → HTTP 409 `APPROVAL_EXACT_TOKEN_REQUIRED`。
   - all listed business counts unchanged。
3. exact plan `承認` → HTTP 200、`payment_approval_required`, version 4、one task/order、7価格項目と NOT CONFORMANT label。
   - plan approval 1、Merchant Task 1、Checkout artifact 1、payment approval/Mandate/settlement/fulfillment 0。
4. payment pending で `承認します` → HTTP 409 `APPROVAL_EXACT_TOKEN_REQUIRED`。
   - payment approval 0、artifact remains Checkout only、settlement/fulfillment 0。
5. same create idempotency key に changed input → HTTP 409 `IDEMPOTENCY_CONFLICT`。
   - workflow count 1、settlement 0。
6. exact payment `承認` → HTTP 200、`completed`, version 12。
   - plan approval 1、payment approval 1、Merchant Task 1、payment artifacts 8、settlement 1、profile receipt 1、prepare+commit 2、refund 0。
   - simulated balance: customer `98750`, Merchant `1250`。
7. completed state で exact `承認` → HTTP 409 `APPROVAL_NOT_PENDING`。
   - approvals/settlement/receipt/fulfillment unchanged。

primary workflow correlation:

- workflow: `workflow:fed387b4e51441ef8c56c5f666682dd0`
- plan: `plan:c3152fbccc8d49c399a0784d3feea5f4`
- order: `order:66d784581a204eb2b11c64aee6725703`
- task: `task:70dd9d95956e439496c76d07b98b49f1`

### 5.4 迂回防止とroute隔離

caller-supplied forged `X-Verified-Identity` と empty POST body で以下を外側から probeした。

```text
404 /v1/workflows
404 /payment/v1/orders
404 /paid-agent/v1/payout-status-requests
404 /mediation-api/internal/v1/mpp/settle
404 /internal/v1/operator/reconcile
```

before/after で plan approval、Task、settlement、fulfillment は増加しなかった。これは public route bypass の証拠であり、全 typed/internal operation の forged capability/Mandate/credential matrix を代替しない。

### 5.5 拒否、並行実行、CLI

- payment approval pending で exact `拒否`:
  - workflow `cancelled`、payment approval 0、settlement 0、fulfillment 0。
  - Merchant DB に original Task の `payment-rejected` Message が exactly 1。
  - `x402.payment.payload` なし。
- parallel payment exact approvals（different message/idempotency keys, same expected version）:
  - one HTTP 200 `completed`、one HTTP 409 `APPROVAL_NOT_PENDING`。
  - payment approval 1、settlement 1、receipt 1、prepare 1、commit 1。
- actual CLI:

```bash
docker exec ap2-independent-e2e /app/.venv/bin/python /app/user-agent/payment_cli.py \
  --workflow-url http://127.0.0.1:8080/mediation-api \
  --prompt 'CLI独立テスト予約' --plan-approval 承認 --payment-approval 承認
```

結果: plan label → payment label/7 prices → completed/AP2 + NOT CONFORMANT labels を public nginx route 経由で確認。

### 5.6 オフライン検証と再作成

```bash
docker exec ap2-independent-e2e /app/.venv/bin/python \
  /app/scripts/verify_ap2_x402_evidence.py workflow:fed387b4e51441ef8c56c5f666682dd0
```

結果: **PASS**。plan authorization、10 capabilities、original Merchant Task、Checkout JWT、closed Checkout/Payment Mandates、synthetic payload、scoped credential、MPP Payment Receipt、Merchant Checkout Receipt、ordered simulation receipt 1、trust snapshots を再検証。

container を stop/remove し、同 image・同三 DB/key mounts で再作成した。再作成後:

- readiness HTTP 200、v2/v2/v2、official items NOT RUN。
- 同 workflow は `completed`, version 12、同 plan/order/task、artifacts 8、receipt 1。
- settlement 1、receipt 1、balance customer 98750 / Merchant 1250 のまま。
- offline verifier 再実行 PASS。
- three DB の `PRAGMA integrity_check` は全て `ok`。

container log の private-key/JWS/payment-proof pattern scanは hit 0。これは実行した happy/recreate log に限る。

## 6. 初回実行で発見した不具合／リリースブロッカー（修正前の記録）

### B1 — durable outbox recovery が実装されていない

- completed/cancelled を含む独立 run 後も `outbox` は `merchant-task:start pending=4`, `trusted-surface:issue pending=3`。
- `deploy/supervisord.conf` に workflow worker program がない。
- controller は outbox を記録した後、同じ request thread で Merchant/AP2/settlement/commit を同期実行する。
- process crash が `payment_authorizing` 等で起きた場合に pending job を再開する worker がなく、completed 後に stale pending intent も残る。
- 影響: WF-005, RES-004〜008, ACC-020, ACC-022, ACC-032。**release blocker**。

必要な修正: leased outbox worker を実装・superviseし、success時に rowをdoneへ原子的に閉じる。各 nonterminal/outbox/evidence/settle/commit/receipt crash pointを kill/restart testし、same operation IDで復旧・business effect <=1 を証明する。

### B2 — readiness が reviewed fail-closed contract を満たさない

`/ready` は durable marker と三 schema versionだけで ready判定する。pinned spec hash、key permission/kid/trust、Store/Card/profile exclusivity、worker heartbeat、pending evidence/outbox reconciliation、route-isolation self-check を検査しない。実際に worker 不在かつ pending outbox がある状態で HTTP 200 `ready` だった。

必要な修正: SEC-008 / design §13.1 の全 dependency を readiness aggregate に含め、どれか不整合なら 503 にする。

### B3 — payment approval view に expiry がない

独立 HTTP/CLI payment view は Merchant/payee、7価格、instrument、scheme/network/asset/payTo、simulation labelを表示するが、Checkout/requirements/payment approval expiry を表示しない。

必要な修正: authoritative stored requirement expiry を `PublicWorkflowView` rendererへ渡し、UI-004/AP2-013/ACC-008 testで exact表示を固定する。

### B4 — official one-command verification script が stale かつ image にない

```bash
PAYMENT_URL=http://127.0.0.1:28080/mediation-api \
PAID_AGENT_URL=http://127.0.0.1:28080/paid-agent \
PAYMENT_PUBLIC_GATEWAY_URL=http://127.0.0.1:28080 \
/bin/sh scripts/verify_payment_demo.sh
```

結果: `/paid-agent/ready` / legacy Agent Card probe で 404。script は hard-disabled `/payment/` と `/paid-agent/` を success前提にし、現在の CLI に存在しない `--mediator-url` / `--approval` 引数を使い、legacy `run_payment_demo.py` を呼ぶ。さらに Dockerfile は `verify_payment_demo.sh` と `run_payment_demo.py` を imageへ copyしないため、`docker exec ... /app/scripts/verify_payment_demo.sh` は file not found。

必要な修正: script を public `/mediation-api/` の two-approval/rejection/recreate/offline-verifier gateへ更新し、RC imageに含める。legacy routeは404を期待する。

### B5 — external paid Merchant A2A boundary が container runtime にない

supervisor は paid Merchant `:8005` を起動しない。integrated controller は `secure_mediation_agent.merchant.PaidBookingMerchant` を in-process callし、reviewed designの external Merchant A2A HTTP activation/capability/evidence-read boundaryを実行していない。`external-agents/paid-booking-agent` はlegacy v2/guarantee implementationのまま。

必要な修正: reviewed selected-profile Merchant A2A adapter/persistent TaskStoreを実 runtimeへ接続するか、scope/designを正式に再承認する。missing/wrong activation、echo、capability、task/context、evidence grantを network boundary で副作用0件検証する。

### B6 — mandatory release-test topology と regression gate が欠落/失敗

- `unit` / `container` 0 collected。
- real Chromium/browser testと `browser` markerなし。
- clean containerは独立手動PASSだが migrated sanitized v1 volume/container E2Eなし。
- root collection error、evaluation-runner 6 failure、jury full collection error。
- machine-readable ACC validator/regression manifestなし。

必要な修正: reviewed planの marker/file topologyを実装し、required suite 0 collection/skip/xfailをfailする validator、same-image digest browser/container/regression artifactを追加する。historical v1三DB fixtureで false approvalなし、legacy rows不変、reapply/backup/restore/cutover failureを検証する。

### B7 — acceptance negative/security coverage が不足

未実証: all listed non-exact variants/multipart、constraint drift→replan、role別 AP2 Error Receipt、cross-workflow/task/tenant replay、全 internal typed bypass、all-state crash injection、success/failure/timeout/restart output secret scan、free workflow full regression。

必要な修正: ACC-003/005/007/009/016/019/020/027/031 の dedicated negative/zero-side-effect assertionsを追加する。

## 7. 受入条件の網羅状況

`PASS` は本 independent run または収集済み testで必須 Then を確認できたもの、`PARTIAL` は一部のみ、`NOT RUN` は未実行、`FAIL` は観測した実装 defectにより満たさないもの。

| ACC | 状態 | 証跡／不足 |
| --- | --- | --- |
| 001 | PASS | first request、paid side effect 0 |
| 002 | PASS | separate plan approval 1、payment approval/Mandate/settlement 0 |
| 003 | PARTIAL | whitespace / `承認します` は拒否。全指定variant/multipart suiteなし |
| 004 | PASS | completed state exact token→`APPROVAL_NOT_PENDING`、counts不変 |
| 005 | PARTIAL | public bypass 404 PASS。全 typed/internal gate matrixなし |
| 006 | PARTIAL | task/capability modelは動作するが external Merchant A2A runtimeなし |
| 007 | PARTIAL | profile isolation/echo model testsあり。network mismatch zero-effectなし |
| 008 | PARTIAL | dotted PaymentRequired/7 prices/labelあり。payment expiry表示欠落 |
| 009 | NOT RUN | constraint drift→replan scenarioなし |
| 010 | PASS | separate payment approval、closed Mandates/credential/synthetic payload、effect 1 |
| 011 | PASS | payment non-exact→artifact/settlement 0 |
| 012 | PASS | original Task/dotted submitted/v1-shaped simulation payload verified |
| 013 | PASS | offline cross-role binding/signature checks |
| 014 | PASS | completed Task evidence、AP2 receipts、ordered sim receipt、NOT CONFORMANT |
| 015 | PASS | repository settlement failure/history test |
| 016 | PARTIAL | settlement Error Receipt pathあり。Merchant/CP/Network role matrixなし |
| 017 | PASS | clean-process offline verifier and container verifier |
| 018 | PASS | same/changed idempotency input and counts |
| 019 | PARTIAL | nonce constraints/testsあり。cross-scope integrated replay matrixなし |
| 020 | FAIL | completed recreation PASSだが outbox pending/workerなし、all-state recoveryなし |
| 021 | PASS | unknown settlement same external ID/no recharge test |
| 022 | PASS | parallel payment approval independent HTTP test、effects exactly 1 |
| 023 | NOT RUN | real Chromium ADK Webなし |
| 024 | PASS | actual CLI public route two-approval completed |
| 025 | PASS | public internal/legacy routes 404、log/body leakageなし |
| 026 | NOT RUN | sanitized historical v1 fixture/migrated containerなし |
| 027 | NOT RUN | full free matching/planning/orchestration regressionなし |
| 028 | PARTIAL | repository contracts + x402 reference 3 PASS。upstream AP2 2 failures、required fixture/report topology不足 |
| 029 | PASS | project URI/NOT CONFORMANT、official/wallet/facilitator/on-chain NOT RUN |
| 030 | NOT RUN | intended conditional official x402/on-chain; runtime disabled |
| 031 | PARTIAL | source/log happy-path scan PASS。failure/timeout/restart/full output scanなし |
| 032 | FAIL | clean image/volume/recreate PASS。migrated volumeなし、worker/readiness defectあり |
| 033 | PASS | original Task payment-rejected exactly 1、payload/settle/commit 0 |
| 034 | PASS | settle-success/commit-failure/refund immutable evidence tests |
| 035 | PASS | settlement/refund unknown same external ID/no new charge tests |

simulation-only release は ACC-001〜029、031〜035 の全PASSが必要である。上表に FAIL/PARTIAL/NOT RUN が残るため、最終判定は **FAIL**。

## 8. 初回判定（§9の再試験結果により置換済み）

**FAIL — release blockerあり。**

official x402/on-chain ACC-030 は意図どおり **NOT RUN** であり、この FAIL の理由ではない。Cloud Run paid も durability未達のため **BLOCKED / NOT RUN** のままとする。現時点で主張できるのは、explicit persistent volumeを使うsingle-host/single-containerにおける限定的な AP2 v0.2 Human Present simulation happy/rejection/concurrency/recreation demoであり、reviewed release acceptance 完了、official x402 compatibility/conformance、Cloud Run paid readinessは主張できない。

## 9. B1〜B7修正後の再試験（2026-08-16）

§6〜§8 は初回独立runの観測記録として保存する。以下の再試験結果がsimulation-only releaseの最新判定である。

| ブロッカー | 修正内容／証跡 | 状態 |
| --- | --- | --- |
| B1 | leased outbox worker、heartbeat、retry/ack、stable operation IDs、全 payment checkpoint と plan/free/final orphan recovery、exact evidence-intent reconciliation。completed/rejected後 unfinished outbox 0、restart後 duplicate settlement/fulfillment 0。 | PASS |
| B2 | data/evidence markers、schema、worker/outbox、evidence intents、role key permission/kid、trust、spec SHA、profile exclusivity、route isolation、external Merchant/TaskStore の11 fail-closed checks。 | PASS |
| B3 | stored requirement expiryをpayment viewへ `approval expiry (UTC)` として表示し、expiry後のnew quote/Checkout/reapprovalを明示。 | PASS |
| B4 | public `/mediation-api/` のnon-exact/two-approval/rejection/offline verification/restartを行うone-command scriptをimageへ同梱。 | PASS |
| B5 | loopback-only external Merchant A2A `:8005`、persistent TaskStore、activation/capability/workflow/task/order gates、exact replayを実runtimeへ接続。 | PASS（single-container simulation scope） |
| B6 | unit/container/browser markers、real Chromium ADK Web two-approval session、historical v1 three-DB migration/recreate、strict ACC validator、versioned regression manifestを追加。 | PASS |
| B7 | non-exact/multipart、private typed bypass、activation/constraint drift、signed role Error Receipts、cross-scope replay、all-state recovery、output secret scan、free flowのdedicated testsを追加。 | PASS |

再試験コマンドと結果:

```text
clean no-cache image build                         PASS
repository suite in the image                     138 passed, 0 failed, 0 skipped
payment/evaluation/jury regression manifest       138 / 17 / 13 collected; PASS
jury configured no-GOOGLE_API_KEY skips           8 allowed, 0 unexpected
real Chromium + ADK Web root/session              request -> 承認 -> 承認 -> completed
public one-command verifier                       PASS; offline evidence PASS
container remove/recreate, same mounts            ready; existing workflow PASS
sanitized v1 three-DB migrate/E2E/recreate         PASS; v1 rows byte/value unchanged
outbox unfinished / evidence intents pending       0 / 0
container success/failure/restart secret scan      0 findings
```

Machine-readable gateは `docs/ap2_x402_conformance_report.json` と `scripts/validate_ap2_x402_release.py`。required ACCで `PARTIAL` / `FAIL` / unapproved `NOT_RUN`、required markerの0 collection、failure/error、skip/xfailが一つでもあればfailする。

## 10. 当時の最新受入判定（§11、§12で更新）

ACC-001〜029、ACC-031〜035 は全て **PASS**（simulation項目は `PASS_SIMULATION*` としてscopeを保持）。ACC-030 は **NOT_RUN_CONDITIONAL** のままであり、PASSへ昇格していない。

**PASS — explicit durable single-host/single-container simulation-only target。**

このPASSはofficial x402 compatibility/conformance、wallet/facilitator/on-chain settlement、実asset movement、external partner interoperability、またはCloud Run paid readinessを意味しない。official x402/on-chain ACC-030 は **NOT RUN**、current Cloud Run paidは **BLOCKED / NOT RUN** のままである。

## 11. 独立再試験（2026-08-16 02:42 JST）

本節は §9〜§10 の自己申告された remediation/PASS を、現在の worktree と clean no-cache image から独立に再検証した結果である。§1〜§8 の初回 FAIL 履歴は削除せず、本節の判定を最新の独立判定とする。production/test code は変更していない。

### 11.1 clean imageとdigest

```bash
docker build --no-cache -t enterprise-a2a-pf:ap2-independent-retest .
docker image inspect enterprise-a2a-pf:ap2-independent-retest \
  --format 'Id={{.Id}} RepoDigests={{json .RepoDigests}} Created={{.Created}}'
```

結果: build PASS。現在の worktree から得た exact image ID/digest は次である。

```text
sha256:cc2f9c82b70d985a0e081ea487bc30333b9c722ee2bf5ee2ca4cab07d97ae024
```

これは `docs/AP2_X402_IMPLEMENTATION_EVIDENCE.md` と `docs/ap2_x402_conformance_report.json` が記録する `sha256:64e7c2de...` と一致しない。従ってその二文書は現在の final image provenance を示していない。runtime は Python 3.12.14 / Chromium 151.0.7922.137 を含み、one-command verifier は executable だった。一方 `/app/tests` は存在せず、image 内の `test_*.py` は 0 件だった。以下の pytest は image の Python/dependencies に repository source を read-only mount して実行したものであり、literal な embedded suite ではない。

### 11.2 repository suite、必須marker、ACC validator

```bash
docker run --rm --entrypoint /app/.venv/bin/python \
  -v "$PWD:/work:ro" -w /work -e PYTHONPATH=/work \
  -e PYTHONDONTWRITEBYTECODE=1 -e WANDB_DISABLED=true \
  enterprise-a2a-pf:ap2-independent-retest \
  -m pytest -p no:cacheprovider -q -ra tests
```

結果: **138 passed, 0 failed, 0 skipped, 0 xfailed, 3 warnings, 19.39s**。

```bash
docker run --rm --entrypoint /app/.venv/bin/python \
  -v "$PWD:/work:ro" -w /work -e PYTHONPATH=/work \
  -e PYTHONDONTWRITEBYTECODE=1 -e WANDB_DISABLED=true \
  enterprise-a2a-pf:ap2-independent-retest \
  scripts/validate_ap2_x402_release.py \
  --output /tmp/ap2-x402-release-validation.json
```

validator 自体は PASS した。11 required markers は全て nonempty、failure/error/skip/xfail は 0 だった。

| marker | collected | result |
| --- | ---: | --- |
| `spike` | 11 | PASS |
| `unit` | 11 | PASS |
| `contract_ap2` | 17 | PASS |
| `contract_x402_simulation` | 2 | PASS |
| `integration` | 25 | PASS |
| `security` | 31 | PASS |
| `restart` | 15 | PASS |
| `migration` | 4 | PASS |
| `concurrency` | 2 | PASS |
| `container` | 2 | PASS |
| `browser` | 1 | PASS as implemented; acceptance gap は §11.7 |

ただし validator は conformance JSON の ACC status を再検証せず、`PASS*` 文字列を信頼する。build/digest、regression manifest、actual browser操作を release resultへ拘束しないため、この validator PASS だけで release acceptance PASS にはできない。

### 11.3 回帰manifest

```bash
docker run --rm --entrypoint /app/.venv/bin/python \
  -v "$PWD:/work:ro" -w /work -e PYTHONPATH=/work \
  -e PYTHONDONTWRITEBYTECODE=1 -e WANDB_DISABLED=true \
  -e RELEASE_IMAGE_DIGEST=sha256:cc2f9c82b70d985a0e081ea487bc30333b9c722ee2bf5ee2ca4cab07d97ae024 \
  enterprise-a2a-pf:ap2-independent-retest \
  scripts/run_regression_manifest.py --output /tmp/regression-result.json
```

結果: manifest PASS。

| suite | collected | pass/skip | unexpected skip |
| --- | ---: | --- | ---: |
| payment-release | 138 | 138 pass | 0 |
| evaluation-runner | 17 | 17 pass | 0 |
| jury-worker | 13 | 5 pass / 8 allowlisted no-key skip | 0 |

ただし `tests/regression/suite_manifest.json` の payment-release `minimumCollected` は初回 baseline の 98 のままである。現在の 138 から 98 までの collection shrinkage を gate が許すため、「current final collection の減少を拒否する」manifest としては fail-closed ではない。

### 11.4 新規永続volumeのcontainer、Merchant、outbox、readiness

fresh root は `/tmp/ap2-x402-independent-retest.VVlsZm`、public port は `28081`。data/evidence marker と8 role keys（provision 時 mode 0600）を作成し、同じ exact image を明示的三DB/evidence/key mountで起動した。

```bash
docker run -d --name ap2-independent-retest -p 28081:8080 \
  -v /tmp/ap2-x402-independent-retest.VVlsZm/data:/app/payment-data \
  -v /tmp/ap2-x402-independent-retest.VVlsZm/evidence:/app/payment-evidence \
  -v /tmp/ap2-x402-independent-retest.VVlsZm/keys:/run/secrets/ap2-demo:ro \
  -e DEV_MODE=true enterprise-a2a-pf:ap2-independent-retest
curl -sS http://127.0.0.1:28081/mediation-api/ready | jq .
docker exec ap2-independent-retest /app/scripts/verify_payment_demo.sh
```

結果:

- readiness HTTP 200、11/11 checks true、schema v2/v2/v2、worker 1、stale lease/failed/overdue 0、pending evidence intent 0。
- official x402、wallet、facilitator、on-chain は全て **NOT RUN**。
- one-command verifier は non-exact rejection、依頼→exact plan `承認`→expiryを含む payment view→exact payment `承認`→completed、別workflowのpayment rejection、offline evidence verification を実行して PASS。
- workflow `workflow:907934a0d63242f298440beebcc0ff9b` は completed、rejected workflow は `workflow:e2e026a5759c408bacf27b478a511691`。
- loopback `:8005` は ready、simulation-only Agent Card、persistent `sqlite-v2` TaskStore を返した。fresh run後は Merchant Task 2、prepare/commit operation各1、outbox 3件すべて `done`、unfinished 0、evidence intent 34件すべて committed、settlement 1、DB integrity `ok`。
- `:8005/a2a` への unsigned `merchant-task:start` は HTTP 400 `CAPABILITY_MISSING`、Task count は 2→2。

worker PID を `SIGSTOP` して heartbeat を15秒超 stale にすると readiness は HTTP 503、`outboxRecovery=false`, `liveWorkers=0` になり、`SIGCONT` 後は HTTP 200 / 11 checks trueへ戻った。fail-closed readiness と heartbeat は実動確認できた。

テスト中に container 稼働中のSQLiteをmacOS host側 `sqlite3` から同時参照した後、worker が `sqlite3.OperationalError: disk I/O error` で Supervisor のdefault rapid retryを使い切り `FATAL` になった。readiness は503を維持し、同じmountでcontainerをrecreateすると worker/readiness、completed workflow、offline verifierは復旧した。live host/container二重DB accessはsupported runtime操作に含めず acceptance defectには数えないが、volumeをinspectionする際はcontainerを停止すべきである。

### 11.5 route隔離、replay、改ざん、秘匿化、オフライン証跡

forged `X-Verified-Identity` を付けた次のpublic probeは全て404だった。

```text
404 /payment/v1/orders
404 /paid-agent/ready
404 /internal/v1/operator/reconcile
404 /v1/workflows
404 /mediation-api/internal/v1/mpp/settle
```

public APIで別workflow `workflow:4eda13b20af04550ab07a11294b41308` を完了後、同じpayment idempotency key + exact同入力は HTTP 200 / 同じ completed version 12、changed inputは HTTP 409 `IDEMPOTENCY_CONFLICT`。payment approval 1、settlement 1、unfinished outbox 0のままだった。container recreate後も同workflowは completed、offline evidence PASSだった。

```bash
docker run --rm --entrypoint /app/.venv/bin/python \
  -v "$PWD:/work:ro" -w /work -e PYTHONPATH=/work \
  enterprise-a2a-pf:ap2-independent-retest \
  -m pytest -p no:cacheprovider -q -ra \
  tests/resilience/test_outbox_worker_recovery.py \
  tests/security/test_ap2_role_rejections_and_output.py \
  tests/integration/test_external_merchant_a2a.py \
  tests/unit/test_acceptance_negatives.py \
  tests/ap2/test_offline_evidence_chain.py
```

結果: **38 passed**。non-exact/multipart、selected-profile drift、external capability scope、unsigned operations、role Error Receipt、selected output redaction、offline evidence、selected recovery checkpointsがgreen。fresh/migrated container logsを8 keyのprivate `d`値、private-key marker、raw Mandate/credential/proof field名でscanした結果も0 findingsだった。

一方、reviewed TEST-007 の agent/Merchant/product/quantity/amount/currency/fee/network/payTo/task/checkout/nonce/signature/expired/revoked/legacy 全改ざんを、new integrated workflow の全 gateで副作用0として対応付けるmatrixは存在しない。legacy `tests/payment_marketplace` のtamper testsをnew flowの証拠には数えない。

### 11.6 無害化したv1三DBの移行／E2E／再作成

sanitized root は `/tmp/ap2-x402-v1-migration-retest.IZ8sqA`。各DBへschema v1とlegacy rowを作成し、evidenceにはexact bytes `000102` / digest `sha256:legacy` / schema 1を保存した。

```bash
docker run --rm --entrypoint /app/.venv/bin/python \
  -v /tmp/ap2-x402-v1-migration-retest.IZ8sqA:/fixture \
  enterprise-a2a-pf:ap2-independent-retest \
  /app/scripts/migrate_ap2_x402_v2.py apply \
  --marketplace /fixture/data/marketplace.db \
  --merchant /fixture/data/paid-agent.db \
  --evidence /fixture/evidence/evidence.db \
  --backup-dir /fixture/backups
```

`plan` / `apply` / `verify` は全てPASS、schemas v2/v2/v2。migrated containerは11/11 ready、one-command E2E/offline PASS。remove/recreate後もworkflow `workflow:356ccd10832d45b9917d0fbe990b5207` は completed、offline PASSだった。legacy `order-v1` / `task-v1` / `evidence-v1` と上記exact evidence bytes/digest/schemaは migration、E2E、recreate後も不変だった。

### 11.7 browser受入と回復matrixの未達

repository `browser` marker 1件はPASSしたが、`tests/browser/test_adk_web_browser.py` は三つの入力をADK WebのHTTP `/run`へ `httpx.post` し、承認後にChromiumをsession JSON endpointの `--dump-dom` へ一度使うだけである。Chromium/ADK Web UI上で入力、click、refresh/reconnectを行わず、TEST-010が要求する二つの承認ラベル、注意書き、7価格項目、状態復元、final conformance labelの実操作確認をしていない。

Browser control runtimeも `agent.browsers.list() == []` で利用可能browserがなく、本独立runではUI操作を補完できなかった。従って ACC-023 は conformance JSON の `PASS` と異なり **NOT RUN / FAIL（required evidence不足）** と判定する。step 11でactual ADK Webを実操作し、同一root/sessionのrequest→`承認`→価格→`承認`、refresh/reconnect、全required表示を独立に保存できれば、このblockerは解消可能である。

また `tests/resilience/test_outbox_worker_recovery.py` は merchant start、payment authorization、6 payment checkpoints、plan/free orphan、evidence-intentをsame-process fault injectionとdirect `_drain`で検証する。実際のprocess/container killを各nonterminal state、credential/settlement response/fulfillment/Receipt境界で行うTEST-008/ACC-020の全matrixではない。`request_received`、`planning`、`replan_required`、`reconciliation_required`、`refund_required`等の各state restart evidenceもない。このため ACC-020 を全面PASSに昇格できない。

### 11.8 当時の独立判定（§12で置換済み）

**FAIL — simulation-only release acceptance 未達。**

remediationによりB1〜B5の主要runtime behavior、11 marker、138 suite、regression、fresh/migrated container、route isolation、expiry UI renderer、one-command verifier、external Merchant、outbox done/heartbeat/fail-closed readiness、recreate、offline evidenceは大きく改善し、上記実行範囲ではPASSした。

残るrelease blockerは次である。

1. ACC-023 / TEST-010 のactual ADK Web UI操作E2Eがなく、現在のbrowser testはHTTP API test + Chromium JSON dumpで代替している。
2. ACC-020 / TEST-008 の各nonterminal state・各外部境界におけるactual process/container crash/restart matrixが完了していない。
3. integrated new flowに対するTEST-007のfull tamper/expired/revoked/legacy negative matrixが完了していない。
4. release evidence/conformanceのimage digestがcurrent no-cache imageと不一致で、validatorがdigest/manifest/browser evidenceを拘束しない。regression `minimumCollected=98` もcurrent 138 baselineを保護しない。

digest文書の更新はprovenance修正で解消できる。actual browser実演はstep 11で必要なUI項目をすべて独立確認・保存すれば解消可能である。literal embedded suiteをrelease要件とする場合、runtime imageにはtestsがないためdocument修正だけでは解消せず、test image/artifactまたはDocker packagingの変更が必要となる。

official x402/on-chain ACC-030 は意図どおり **DISABLED / NOT RUN**、current Cloud Run paidは **BLOCKED / NOT RUN** のままであり、このFAILの理由ではない。

## 12. `payment_user_agent` の固定イメージによる再試験

実施日: 2026-08-16（Asia/Tokyo）

本節は §11 の blocker を修正した後の再試験であり、§11.7–11.8 の判定を supersede する。Cloud Run deploy は実行していない。

### 12.1 変更不能なイメージと自己試験

`docker build --no-cache` で `enterprise-a2a-pf:ap2-payment-user-final` を作成した。exact image ID は次である。

```text
sha256:713149605701e1b5554326ef2938938756cf6fd7182411b2d3a7b289acc08dc1
```

source treeをbind mountせず、output artifact directoryだけをmountしてimage内のrelease self-testを実行した。

| suite | collected | result |
| --- | ---: | --- |
| payment-release | 166 | 166 pass |
| evaluation-runner | 17 | 17 pass |
| jury-worker | 13 | 5 pass / 8 allowlisted no-key skip |

unexpected skipは0。release manifest digest は `sha256:701f2b1a66af81b9fe6dacefbb1a713fb35c181571621f8fda85296e363d3495`、regression manifest digest は `sha256:7d72de56a96a3f7438b539e0131167e3e7c9acd2c8e0fa204916dbdd7cfd7339` だった。

### 12.2 実 Chromium による ADK Web 受入確認

実ChromiumをCDPで操作し、ADK Web DOMのtextareaへ `browser booking`、exact `承認`、exact `承認` を入力後にrefreshした。`/list-apps` は `payment_user_agent` のみで、plan注意書き、approval expiry、7価格項目、charge warning、completed、simulation / `NOT CONFORMANT`、refresh後のcompleted復元を確認した。machine evidenceは `artifacts/browser-evidence.json` に保存した。

### 12.3 実 process 終了と統合異常系 matrix

test-only failpointは `APP_ENV=test` でのみ有効になり、atomic marker作成後に `os._exit(86)` する。state transitionとMerchant start return、payment submit、settlement、fulfillment commitを含む12 checkpointで実processを終了し、別processでlease/recoveryを再開した結果は **12/12 PASS**。external fulfillment commit直後の再開で失われていたimmutable Merchant Task challengeの復元も修正・検証した。

同じintegrated workflowに対し expired、wrong signing authority/role、tampered、revoked、unsigned、cross-workflow/task/order scope、legacy route absenceを実行し、すべてfail closedかつunauthorized side effect 0だった。

### 12.4 新規／移行済み volume の永続 E2E

fresh root `/tmp/ap2-payment-user-fresh.Z1hblf` を明示的data/evidence/key mountで起動し、readiness 11/11、schema v2/v2/v2、two-approval E2E、offline evidenceを確認した。workflow `workflow:49f6a73888944071be15f63e872f0ae2` はcontainer remove/recreate後も `completed`、offline evidence PASSだった。

sanitized v1 root `/tmp/ap2-payment-user-migrated.fZYDJx` では三DBを `plan` / `apply` / `verify` し、schema v2/v2/v2へmigrationした。migrated containerのreadiness 11/11、two-approval E2E、offline evidenceはPASS。workflow `workflow:5972fb67e9f34632b980bedc8ca4d60e` はremove/recreate後もcompletedだった。`order-v1`、`task-v1`、`evidence-v1` とexact evidence `000102|sha256:legacy|1` はmigration、E2E、recreate後も不変だった。

### 12.5 リリース validator と deployment 境界

release validator v2はexact image digest、frozen manifests、regression/browser artifactsを自己検証し、failures `{}` でPASSした。

| marker | collected |
| --- | ---: |
| spike | 11 |
| unit | 11 |
| contract_ap2 | 17 |
| contract_x402_simulation | 2 |
| integration | 45 |
| security | 44 |
| restart | 27 |
| migration | 4 |
| concurrency | 2 |
| container | 5 |
| browser | 1 |

ephemeral modeを同じexact imageでlocal起動し、warning `EPHEMERAL DEMO: state and keys may reset on restart`、public Firebase config endpoint、unauthenticated rootから`/login`へのredirect、sole ADK app、official x402 / on-chain `NOT RUN` を確認した。新規service `payment-user-agent-demo` 用deploy scriptは用意済みだが実行していない。これはdurabilityを主張しないephemeral demoであり、durable Cloud Run paid releaseではない。

### 12.6 最終判定

**PASS — local durable / explicitly ephemeral simulation demo readiness.**

§11のactual browser、actual process-death recovery、integrated attack matrix、digest/manifest/browser binding、current collection baselineのblockerはすべて解消した。official x402 compatibility、wallet/facilitator、real asset movement、on-chain settlement、durable multi-instance Cloud Run、production identity/KMSは引き続き **NOT RUN / not claimed** である。

## 13. 固定候補の最終独立再試験

実施日: 2026-08-16（Asia/Tokyo）

本節は §12 の自己申告を独立に再試験した履歴であり、過去節は削除しない。試験対象は frozen candidate `enterprise-a2a-pf:ap2-payment-user-final`、exact image ID / RepoDigest `sha256:713149605701e1b5554326ef2938938756cf6fd7182411b2d3a7b289acc08dc1` である。Cloud Run deploy、database service作成、その他cloud resource作成は行っていない。production code/testは修正せず、このreportだけを追記した。

### 13.1 image、組込み自己試験、validator

- `docker image inspect` は要求digestと完全一致した。独立 `docker build --no-cache -t enterprise-a2a-pf:ap2-payment-user-independent-rebuild .` 自体は成功したが、新image IDは `sha256:db06aabc319e8c7d2bd66fa2abdf5fbcd93f02f071e6781f3363aff2cc5854ad` でbit-for-bit一致しなかった。このため依頼で許可された exact frozen candidate 経路を使用した。
- source treeをmountせず、output artifact directory `/tmp/ap2-independent-final.x5kVct` だけをmountしてimage内runnerを実行した。payment-release **166/166 PASS**、evaluation-runner **17/17 PASS**、jury-worker **13 collected / 5 PASS / 8 allowlisted skips**、unexpected skip 0。payment floorを一時manifestで167へ上げたnegative runは同じ166 collectionを検出してexit 1 / FAILとなり、shrink gateもPASSした。
- release manifest digestは `sha256:701f2b1a66af81b9fe6dacefbb1a713fb35c181571621f8fda85296e363d3495`、regression manifest digestは `sha256:7d72de56a96a3f7438b539e0131167e3e7c9acd2c8e0fa204916dbdd7cfd7339`。fresh regression/browser artifactsは両方ともexact image digestとrelease manifest digestへboundされた。
- release validator v2はfresh artifactsとexact digestでPASSし、11 markerは `spike=11`, `unit=11`, `contract_ap2=17`, `contract_x402_simulation=2`, `integration=45`, `security=44`, `restart=27`, `migration=4`, `concurrency=2`, `container=5`, `browser=1`。`--skip-suite-execution`をnon-promotableにし、browser digest mutationを拒否するcontainer binding testsもPASSした。official x402 / on-chainは **NOT RUN** のまま。
- current diffをinspectionし、`git diff --check` はPASS。runtime image内のADK discovery rootには `payment_user_agent` の三fileだけがあり、`secure_mediation_agent` はselectable appとして存在しなかった。

### 13.2 実 Chromium UI と Firebase session

image内browser markerは実system Chromiumをremote-debugging/CDPで起動し、ADK Web DOMのtextareaへ `browser booking` → exact `承認` → exact `承認` を入力し、refresh後もcompletedを復元した。fresh `browser-evidence.json` は `browser=chromium-cdp-real-ui`、`listApps=[payment_user_agent]`、`appSelected=payment_user_agent`、`completedAfterRefresh=true` を記録した。testはplan注意、approval expiry、7価格項目、課金警告、completed、simulation / `NOT CONFORMANT` 表示もassertした。Codex in-app Browser自体はこの環境でavailable browserが0件だったが、release image内の実Chromium/CDP試験が独立実行されているためmain UI acceptanceのblockerには数えない。

Firebase session 4 testとrelease-boundary 6 testは **10/10 PASS**。same-origin + CSRF exchange、Firebase project audience/issuer/subject、`Secure; HttpOnly; SameSite=Strict; __Host-` cookie、requestごとの再verify、logout、caller identity header除去、sole payment app、official adapter fail-closedを確認した。

### 13.3 実 crash と統合攻撃 matrix

`test_actual_process_death_replays_same_operation_without_second_effect` を単独実行し、atomic test-only failpointから `os._exit(86)` する次の **12/12 checkpoint PASS** を確認した: Merchant start return、payment approval required、payment approved、payment submit return、payment submitted、payment verifying、fulfillment preparing、payment settling、settlement return、fulfillment committing、fulfillment commit return、completed。各caseは別processで同operationを再lease/recoverし、authorization caseのsettlementは1、customer/merchant balanceは一回分だけだった。

external Merchant integrated fileは **16/16 PASS**。activation mismatch、workflow/task/order cross-scope replay、全private operationのunsigned bypass、expired、wrong-role authority、signature tamper、revoked capability、legacy/control-plane 4 route absenceを検証し、拒否caseはpersistent Merchant side effect 0だった。

### 13.4 新規／移行済み volume の永続 E2E、Merchant、outbox

Docker-managed persistent volumesでfresh runtimeを起動し、`/mediation-api/ready` はHTTP 200 / 11 checks true / schema v2-v2-v2。image同梱one-command verifierはnon-exact拒否、request → plan `承認` → payment `承認` → completed、別workflowのpayment rejection、offline evidenceをPASSした。workflow `workflow:1aea12122d1d469fa684c45e005462a4` はcontainer remove/recreate後もcompleted / offline PASSだった。runtime `/list-apps` は `["payment_user_agent"]`。loopback `127.0.0.1:8005` MerchantはHTTP 200、`taskStore=sqlite-v2`、simulation-only Cardを返し、listenerもloopbackだけだった。unfinished outbox 0、pending evidence intents 0、settlement attempt 1、三DB `PRAGMA integrity_check=ok`。

sanitized v1三DBではmigration `plan` / `apply` / `verify` がすべてPASSし、schema v2-v2-v2へ移行した。two-approval E2E、rejection、offline verification、container recreateもPASSし、workflow `workflow:436cc51f4e90408faf707a308fcf9d0f` をcompletedで復元した。`order-v1` / `task-v1` / `evidence-v1` とexact evidence `000102|sha256:legacy|1` はmigration、E2E、recreate後も不変、unfinished outbox / pending evidence intentsはいずれも0だった。

補足としてmacOS `/tmp` bind mountを使った最初のfresh試行では、Merchant/worker初回起動がSIGBUSとなりSQLiteがmalformed、one-command verifierはHTTP 500で失敗した。その後readinessはHTTP 503、schemas/merchant check falseへfail closedした。Docker-managed volumesでは上記全gateが安定PASSしたためcandidate logicのmain-flow blockerとは数えないが、Docker Desktop host bind mountをsupported durable storageとして扱う場合はfuture platform qualificationが必要である。

### 13.5 一時 mode と deploy の静的監査

exact imageを `EPHEMERAL_CLOUD_RUN_DEMO=true,APP_ENV=ephemeral-demo,DEV_MODE=false` でlocal起動した。public `/auth/deployment` は `ephemeral=true` と `EPHEMERAL DEMO: state and keys may reset on restart`、official x402 / on-chain `NOT RUN` を返し、unauthenticated rootは `/login` へ302、public Firebase configは200、internal `/list-apps` はpayment appのみだった。`DEV_MODE=true,APP_ENV=ephemeral-demo` はstartup exit 1で拒否された。

一方で同じephemeral runtimeのauthoritative workflow readinessは `target=explicit-durable-single-host-single-container`、両durable marker `PASS`、11 checks trueを返した。startupの第一行もdurable service開始を表示する。public warningは存在するが、readinessが逆のdurability claimを返すため「ephemeral modeは明確にnon-durable」という条件は完全には満たさない。

`deploy/deploy-payment-demo-cloudrun.sh` は `bash -n` PASS、`DEV_MODE=true` refusal、deploy env `DEV_MODE=false`、ephemeral warning、single instanceを設定し、Cloud SQL / Firestore / database等を作るcommandはない。deployは実行していない。しかし `SERVICE_NAME="payment-user-agent-demo"` が固定で、`gcloud run services describe` 等によるpre-existence refusalもunique service生成もない。`gcloud run deploy` は既存同名serviceをupdateできるため、scriptの表示する「NEW service only」はenforceされていない。

### 13.6 証跡 provenance のブロッカー

`docs/ap2_x402_conformance_report.json` はまだrepository 138 pass、release digest `sha256:64e7c2de1e9c2d4c19b49dfc609da124fceef3e79c320adf67dcf7512929dc5d` を記録しており、今回の166 pass / exact `sha256:713149...dc1` と不整合である。さらにvalidatorへこのfileを `--conformance` で渡してもPASSした。validatorはacceptance mapだけを比較し、conformance report自身のimage digest / collection countをboundしていないため、stale machine evidenceをrelease evidenceとして受理できる。

### 13.7 独立判定（後続の修正後再試験があれば置換）

**FAIL — 主要な決済／runtime安全gateはPASSしたが、次のhard blockerを修正して独立再試験するまでcandidateはリリース不可。**

| 分類 | 結果 |
| --- | --- |
| main UI / two exact approvals / refresh | PASS |
| auth bypass / public legacy bypass / identity forgery | PASS (fail closed) |
| duplicate settlement / crash replay / outbox recovery | PASS |
| fresh + migrated persistent E2E / recreate / offline | PASS on Docker-managed volumes |
| hard blocker 1 | Cloud Run demo deploy script can update an existing fixed-name service; NEW-only is not enforced |
| hard blocker 2 | stale 138-test/wrong-digest conformance JSON is accepted by the release validator |
| hard blocker 3 | ephemeral readiness asserts the durable target and durable-marker PASS, contradicting the non-durable warning |
| future qualification | no-cache rebuild is not bit-identical, but exact frozen candidate is available and was used |
| future qualification | macOS host bind-mount SQLite showed SIGBUS/corruption; managed-volume path passed and readiness failed closed |
| intentionally not run | official x402 compatibility, wallet/facilitator, real asset movement, on-chain settlement, durable multi-instance Cloud Run |

Cloud deployは **NOT RUN**、official x402/on-chainは **NOT RUN**。上記三hard blockerの修正後は、exact candidate再freeze、conformance provenance negative test、ephemeral readiness label、NEW-service refusalを重点再試験すればよい。

## 14. §13 の重大ブロッカー修正と最終再固定

実施日: 2026-08-16（Asia/Tokyo）

§13.7 の3件の重大ブロッカーだけを修正し、no-cache rebuild 後の新しい唯一のリリース候補を次の値に固定した。旧 `sha256:713149...dc1` は置換済みの履歴であり、現行候補ではない。Cloud Run へのデプロイは実行していない。

```text
enterprise-a2a-pf:ap2-payment-user-final
sha256:ebb84f014f5a5b7ff86c883aec9791e4b7ff2b550e11a0954e1e443fd086e429
```

### 14.1 Cloud Run の新規サービス限定事前検査

専用スクリプトは project `gen-lang-client-0585901015`、region `asia-northeast1`、service `payment-user-agent-demo` と、デプロイ環境 `EPHEMERAL_CLOUD_RUN_DEMO=true,APP_ENV=ephemeral-demo,DEV_MODE=false` を固定値として自己検証する。引数や上書きは拒否する。build より前に read-only の `gcloud run services list` を対象へ実行し、同名 service が1件でも存在すれば exit 3 で停止する。偽の gcloud が既存 service を返す動作試験では Docker marker が作られず、build／push／deploy が一度も呼ばれないことを確認した。

### 14.2 conformance 証跡の来歴拘束

`docs/ap2_x402_conformance_report.json` を schema v2、payment 166、新しい正確な image digest へ更新した。validator へ `--conformance` を渡すと、acceptance map に加え、conformance 内の image digest、payment の正確な件数、release-manifest digest、browser artifact SHA が、指定・固定済みの証跡と完全に一致することを検証する。古い image、件数 165、誤った manifest SHA、誤った browser SHA の4ケースをそれぞれ拒否する対象試験は PASS した。

```text
release manifest       sha256:701f2b1a66af81b9fe6dacefbb1a713fb35c181571621f8fda85296e363d3495
regression artifact    sha256:4492d97f587263493a8ba364cdb8246bfe6c2ad60535b7b2f8c945a1c3f3cf21
browser artifact       sha256:1631471cd167a767427307ab718dbd1e4e42520a22ebc86cf3b837f071eb14d9
conformance report     sha256:b2a3a1cb5345651ccb98b5442ab3274dd03507e8864cd422bac8ecfc6c1c00e9
release validation     sha256:40a5a66d52e8d999adec9e12b99c93db28aea7bf2cba6b2059b50535d00fd2a5
```

### 14.3 一時デモの耐久性を正確に示す readiness

正確な image を `EPHEMERAL_CLOUD_RUN_DEMO=true,APP_ENV=ephemeral-demo,DEV_MODE=false` でローカル起動した。正本となる workflow `/ready` と公開 `/auth/deployment` は共に `target=ephemeral-cloud-run-demo`、`durability=NOT PROVIDED`、`EPHEMERAL DEMO: state and keys may reset on restart` を返した。workflow response は `dataDurableVolume`、`evidenceDurableVolume`、`durableVolumeMarker`、`evidenceDurableVolumeMarker` を含まず、一時 data／evidence path の書込み可否、worker／outbox、keys、schemas、trust、spec、profile、route、Merchant の runtime health で HTTP 200 を判定した。起動表示も一時 service と明示し、偽の durable marker を作成しない。耐久ローカル mode の既存 target／marker checks は回帰試験で不変を確認した。

### 14.4 固定済み gate と最終判定

source bind mount なしの正確な image による regression は payment **166/166 PASS**、evaluation-runner **17/17 PASS**、jury-worker **13 collected / 5 PASS / allowlist 済み 8 skips**、予期しない skip 0。新規の実 Chromium 証跡は PASS。conformance を指定した完全な validator も failures `{}` で PASS し、marker counts は `spike=11`, `unit=11`, `contract_ap2=17`, `contract_x402_simulation=2`, `integration=45`, `security=44`, `restart=27`, `migration=4`, `concurrency=2`, `container=5`, `browser=1` だった。

**PASS — §13 の3件の重大ブロッカーは解消した。** 公式 x402、wallet／facilitator、実資産、on-chain settlement、耐久 multi-instance Cloud Run は引き続き **NOT RUN / not claimed**。Cloud へのデプロイは **NOT RUN**。

## 15. 新candidateの最終独立release-gate再試験

実施日: 2026-08-16（Asia/Tokyo）

対象は `enterprise-a2a-pf:ap2-payment-user-final`、exact image ID / RepoDigest `sha256:ebb84f014f5a5b7ff86c883aec9791e4b7ff2b550e11a0954e1e443fd086e429`。§13の3 hard blockerだけを独立再試験した。implementationは修正せず、Cloud Run deployは実行していない。

### 15.1 NEW-only deploy preflight — PASS

image内testをsource mountなしで実行し、既存exact serviceを返すfake gcloudではexit 3、Docker build markerなしを確認した。追加の独立probeでは `PROJECT_ID=evil-project`, `REGION=evil-region`, `SERVICE_NAME=evil-service`, `DEPLOY_ENV_VARS=evil-env`, `APP_ENV=evil` を環境から与えても、preflightが使用した値は固定 `gen-lang-client-0585901015 / asia-northeast1 / payment-user-agent-demo` だった。existing service検出でbuild前にexit 3、fake dockerは未呼出し。引数override refusalもembedded testで確認した。

### 15.2 evidence byte binding — FAIL（1 hard blocker残存）

正規artifactではconformance付きfull embedded validatorが `failures={}`、11 marker全PASSとなった。image digest、payment count、release manifest digest、browser artifact byte digestのstale/tamper negative testsもPASSした。

しかしconformance v2が記録する `evidence.regressionArtifactDigest=sha256:4492d97f...cf21` はvalidatorで読まれていない。regression artifactへsemantic gateに影響しないfieldを加え、byte SHAを `sha256:e5fc2d97121c639f9ae2af5d363ffca9916eee277533e610548a714c91825a5f` へ変更した状態で、元のconformance reportを渡してfull validatorを再実行した結果もexit 0、`status=PASS`, `failures={}` だった。markerは全て再実行されPASSした。従ってregression artifactのbyte tamper/stale provenanceを検出できず、要求されたconformance/regression/browser/manifest/imageの完全なbyte bindingは未達である。

### 15.3 actual ephemeral runtime contract — PASS

exact imageを `EPHEMERAL_CLOUD_RUN_DEMO=true,APP_ENV=ephemeral-demo,DEV_MODE=false` で実起動した。workflow `/ready` とpublic `/auth/deployment` は共にHTTP 200、`target=ephemeral-cloud-run-demo`, `durability=NOT PROVIDED`, reset warningを返した。readiness checksはephemeral data/evidence writable、schema、worker/outbox、keys、trust、spec、profile、route、Merchantを実検査してreadyとなった。responseにdurable marker proof fieldはなく、filesystemにも両 `.durable-volume` markerは存在しなかった。startup logもexplicitly ephemeralと表示した。

### 15.4 独立最終判定

**FAIL — deploy NEW-onlyとephemeral durability表示は解消したが、regression artifact byte bindingがないためrelease evidence integrity gateは未達。**

必要な修正はvalidatorがactual regression artifact SHAを計算し、conformance `regressionArtifactDigest` と比較して、不一致をnon-promotableにすること、およびそのtamper negative testの追加である。これはmain payment-flow defectではないが、今回指定されたhard release gateである。公式 x402 / on-chainは引き続き **NOT RUN**、Cloud deployは **NOT RUN**。

## 16. §15 regression artifact byte-binding remediation

実施日: 2026-08-16（Asia/Tokyo）

§15.2の単一hard blockerだけを修正した。validatorはdefault pathを再解決せず、CLIで指定されたexact `--regression-result` pathからbytesを読み、そのSHA-256を `conformance.evidence.regressionArtifactDigest` と完全一致比較する。不一致は `conformanceRegressionArtifactDigest` failureとなり、marker suiteがgreenでもrelease結果は非0／FAILになる。

既存container test内で、regression JSONのobject semanticsを変えずにindent／key orderだけを変更した。`json.loads` equalityを確認した上でbyte SHAが変化し、exact-image validatorがnonzeroかつ上記failure keyを返すことを確認した。test collection baselineは166のまま増えていない。

validator/testをimageへcopyするためno-cache rebuildし、旧 `sha256:ebb84f...e429` をsupersedeして次を唯一のcandidateへfreezeした。

```text
enterprise-a2a-pf:ap2-payment-user-final
sha256:1b743079d533feb3e36a487bf6338799da8cdb3989523c69cdae904a8e9d5c29
```

tracked artifact bytesとconformanceの最終bindingは次である。

```text
release manifest       sha256:701f2b1a66af81b9fe6dacefbb1a713fb35c181571621f8fda85296e363d3495
regression artifact    sha256:9a70562738168c9cce9e2d7a9656fc5271450889a4d299cfd4805224bf188be5
browser artifact       sha256:e0a2d4d82590d3b02d63671428c3edc3956e97bd72065897dbfc1cf30184eb02
conformance report     sha256:c3eb54fe28bd180c9de1303668d09fb6beac82326a8593941bc1217497c559e4
release validation     sha256:52cbd43cd6fc3472c75517407caf4e7014d8ae8842f89a960a7d7ca009e00974
```

source mountなしのexact-image regressionはpayment **166/166 PASS**、evaluation-runner **17/17 PASS**、jury-worker **13 collected / 5 PASS / allowlist済み8 skips**。fresh actual Chromiumは1/1 PASS。tracked regression/browser/conformanceを明示mountしたfull exact-image validatorはfailures `{}`、全11 marker PASS。focused negative testもexact imageで1/1 PASSした。

**PASS — §15のregression artifact byte-binding blockerは解消した。** Cloud deployは **NOT RUN**、公式 x402／on-chainは引き続き **NOT RUN / not claimed**。

## 17. regression byte-binding fix 最終独立再試験

実施日: 2026-08-16（Asia/Tokyo）

対象candidate `enterprise-a2a-pf:ap2-payment-user-final` のimage ID / RepoDigestは要求どおり `sha256:1b743079d533feb3e36a487bf6338799da8cdb3989523c69cdae904a8e9d5c29` と完全一致した。Cloud deployとimplementation変更は行っていない。

current artifact SHAはrelease manifest `sha256:701f2b1a...3495`、regression `sha256:9a705627...8be5`、browser `sha256:e0a2d4d8...eb02`、conformance `sha256:c3eb54fe...59e4`。conformance v2内のimage、manifest、regression、browser digestはすべてcurrent bytesと一致した。この組合せをexact imageへ明示mountしたfull validatorはexit 0、`status=PASS`, `failures={}`。11 markerは全てPASSした。

negative probeでは supplied regression JSONを `jq -cS` で再formatし、JSON semanticsが等しいことをcanonical comparisonで確認した。byte SHAだけが `sha256:9a705627...8be5` から `sha256:2d6f6bf7...c356` へ変化した。exact-image validatorはexit 1、`status=FAIL` となり、`failures.conformanceRegressionArtifactDigest` にtracked original digestを返した。従ってsemantic-equivalent byte tamperもnon-promotableである。

**PASS — §15の最後のhard blockerは独立再試験で解消した。final release evidence gateはPASS。** 公式 x402 / on-chainは引き続き **NOT RUN**、Cloud deployは **NOT RUN**。

## 18A. linux/amd64 Cloud Run local candidate 独立再試験

実施日: 2026-08-16（Asia/Tokyo）

本節は§17をsupersedeするCloud Run向けlocal candidateの独立確認である。対象はtag `enterprise-a2a-pf:payment-user-agent-cloudrun-amd64`、exact image ID / RepoDigest `sha256:a4b65095bfca08f6212b13b4c23a543a15bfacd50713c26906254aea49308552`。`docker image inspect` は `linux/amd64` を返した。類似名の `ap2-payment-user-final` tagは旧arm64候補のままであり、本節では使用していない。push / deploy / implementation変更は行っていない。

### 18A.1 artifact とclean-context binding

current `regression-result.json`、`browser-evidence.json`、`ap2-x402-release-validation.json` はすべてexact `a4b650...8552` とrelease manifest `sha256:701f2b1a...3495` にboundされている。`cloud-run-candidate.json` の `localImageId` も同一で、platformは `linux/amd64`、statusは `LOCAL_VALIDATED_NOT_PUSHED`、registry image/digestは共に `NOT_PUSHED`。candidateに記録された4 artifact byte SHAはcurrent bytesと一致し、conformanceのimage/platform/manifest/regression/browser digestも一致した。

`cloud_run_candidate.py verify-local` はexit 0。clean-context sourceはcommit `9730a597a3359f7ecac0f2bf10513a80f9b3c56e`、232 files、worktree digest `sha256:93201267...37a7` でcandidate記録と完全一致した。必須JSON 11件はすべて存在し、Git-visible clean-contextに含まれ、ignored fileは0。build scriptはこのvisible file setだけを一時contextへmaterializeし、固定 `--platform linux/amd64 --no-cache --provenance=false --load` を使用する契約である。

### 18A.2 source-mount-free embedded execution

exact amd64 imageを `--platform linux/amd64` で実行し、source treeはmountせずoutput directoryだけをmountした。regression結果はpayment **166/166 PASS**、evaluation-runner **17/17 PASS**、jury-worker **13 collected / 5 PASS / allowlist済み8 skips**、unexpected skip 0。

payment suite内の実Chromium/CDP testもPASSし、fresh evidenceは `browser=chromium-cdp-real-ui`、`listApps=[payment_user_agent]`、request → exact `承認` → exact `承認` → refresh、`completedAfterRefresh=true` を記録した。

tracked regression/browser/conformanceを同じexact imageへmountしたfull validatorはexit 0、`status=PASS`, `failures={}`。11 markerは `spike=11`, `unit=11`, `contract_ap2=17`, `contract_x402_simulation=2`, `integration=45`, `security=44`, `restart=27`, `migration=4`, `concurrency=2`, `container=5`, `browser=1` で全件PASS、failure/error/skip/xfailは0だった。container markerにはfixed amd64 build/push/deploy separation、NEW-only preflight、ephemeral readiness、candidate verifier契約も含まれる。

### 18A.3 verdict

**PASS — exact local linux/amd64 candidate、4 artifact binding、全11 marker、payment/evaluator/jury、実Chromium、clean-context必須JSON、platform gateはすべて独立PASS。** registry pushとCloud Run deployは **NOT RUN**。公式 x402 / wallet / facilitator / real asset / on-chain settlementは引き続き **NOT RUN / not claimed**。

## 18. Cloud Run 配布経路レビュー指摘の修正と amd64 local gate

実施日: 2026-08-16（Asia/Tokyo）

§17の候補は決済ロジックの証跡としてはPASSだったが、独立コードレビューで (1) host-native `linux/arm64` build、(2) deploy時の未検証再build、(3)必須JSONのGit管理外、(4) CLI／verifier cookie名不一致、(5)直接実行scriptのmode欠落がrelease blockerとして判明した。本節はその修正後の local release gate である。Cloud Run deployは実行していない。Artifact Registryは固定repositoryの存在をread-only確認しただけで、再レビュー前の指示によりpushを保留した。

### 18.1 clean-context `linux/amd64` exact candidate

`git ls-files --cached --others --exclude-standard` で可視なfileだけを一時build contextへmaterializeし、ignored local fileを利用できない状態で次を実行した。

```text
docker buildx build --platform linux/amd64 --no-cache \
  --provenance=false --load \
  -t enterprise-a2a-pf:payment-user-agent-cloudrun-amd64 <clean-context>
```

最終 image ID は `sha256:a4b65095bfca08f6212b13b4c23a543a15bfacd50713c26906254aea49308552`、`docker image inspect` の platform は `linux/amd64`。source binding は commit `9730a597a3359f7ecac0f2bf10513a80f9b3c56e`、worktree digest `sha256:932012674dbd2b09878ef1f82240ccfe974f6271191c4debc6b0c3ca9d8e37a7`、232 filesである。

初回clean-context試験では、汎用`*.json` ignoreのため evaluation-runner schema／prompt JSON が欠落し、payment 166とjury 13はPASSしたがevaluator 4件がFAILした。この結果を隠さず、必要な6 JSONをexact allowlistとrequired-file gateへ追加してno-cache再buildした。公開Firebase config、spec manifest、regression／release manifest、conformance report、release evidence artifactもexact allowlistでGit-visibleになっている。必須JSONに対する`git check-ignore -q`は全件non-ignoredだった。

### 18.2 source mountなしの埋込み試験

最終exact imageにoutput directoryだけをmountした結果は次のとおり。

| gate | result |
| --- | --- |
| versioned regression | payment 166 PASS / evaluator 17 PASS / jury 13 collected（5 PASS + 8 allowlisted no-key skip）/ unexpected skip 0 |
| real Chromium UI | 1 PASS; `payment_user_agent`のみ、`browser booking` → `承認` → `承認` → refresh後completed |
| required markers | spike 11 / unit 11 / contract_ap2 17 / contract_x402_simulation 2 / integration 45 / security 44 / restart 27 / migration 4 / concurrency 2 / container 5 / browser 1 |
| release validator v2 | `status=PASS`, `failures={}`, 全 marker failure/error/skip 0 |

artifact byte bindingは regression `sha256:0fe2d32c1cd97ae0179de62d72fa2e24e53612d93c8b4403df8d0eb92aadbc32`、browser `sha256:4d9fe30fbedf29a925f6779aa0ad63ba29d0c6625e7859e28c7f96f35f0dc9eb`、conformance `sha256:9de6bc4db5653519d0dbcee71c637a1f07277c81caf506b71662fca42d59770d`、release validation `sha256:cc14e57f9eb0009f2e7196c7008081fd35420560defc05d3c23bd9c9a71d6869`。`artifacts/cloud-run-candidate.json` はこれらとsource binding、platform、組込みsuite summaryを持ち、statusは`LOCAL_VALIDATED_NOT_PUSHED`である。

### 18.3 配布script、認証、mode

- build scriptはpublish／deployしない。push scriptはlocal candidateを再検証後にだけpublishし、deployしない。deploy scriptにはbuild／push commandがなく、status `PASS`、固定repositoryの`@sha256:`、current source／artifact byte digest、platform `linux/amd64`、embedded regression／marker／browser resultが一致しなければfail closedする。
- deploy targetは`gen-lang-client-0585901015/asia-northeast1/payment-user-agent-demo`、environmentは`EPHEMERAL_CLOUD_RUN_DEMO=true,APP_ENV=ephemeral-demo,DEV_MODE=false`に固定し、既存service拒否を維持した。
- CLI／one-command verifierはserverと同じ`__Host-payment-session`を送る。既存回帰testはpublic `https://demo.example/mediation-api/v1/workflows`へのrequestでexact cookie headerを検証する。
- `deploy/run-local.sh`、start scripts、candidate build/push/deploy scripts、one-command verifierは直接実行可能modeである。
- Artifact Registry `projects/gen-lang-client-0585901015/locations/asia-northeast1/repositories/secure-mediation-agent` はDOCKER / STANDARDとread-only確認した。**pushはNOT RUN、Cloud Run deployはNOT RUN**。

**LOCAL PASS — `linux/amd64` candidateのbuild／regression／実browser／full validator／source-artifact bindingまで完了。registry digestはpush後にのみ確定するため、現時点のdeploy gateは意図どおり閉じている。** 公式 x402、wallet／facilitator、実資産、on-chain settlement、耐久multi-instance Cloud Runは引き続き **NOT RUN / not claimed**。

## 19. linux/amd64 local candidate 最終独立判定

実施日: 2026-08-16（Asia/Tokyo）

§18の自己記録を、exact local image `sha256:a4b65095bfca08f6212b13b4c23a543a15bfacd50713c26906254aea49308552` で独立再実行した。指定tagは `enterprise-a2a-pf:payment-user-agent-cloudrun-amd64`、image metadataは `linux/amd64`。旧arm64の `ap2-payment-user-final` tagは試験対象にしていない。

`cloud-run-candidate.json` とcurrent regression/browser/release-validation/conformanceのimage・platform・byte digest・manifest bindingは一致し、`verify-local` はexit 0。clean-context source commit/worktree digest/file countはcandidate記録と一致し、必須JSON 11件はmissing/ignored 0だった。

source mountなしのexact amd64 imageでpayment 166、evaluator 17、jury 13 collectedを再実行してPASS。実Chromium/CDPは `payment_user_agent` のrequest → `承認` → `承認` → refresh後completedをPASS。tracked artifactを使うfull validatorもexit 0、`failures={}` で、11 markerすべてが記録どおりPASSした。

**FINAL INDEPENDENT PASS — local `linux/amd64` candidateと全指定gateはPASS。push / deployは禁止どおりNOT RUN。** registry digest、公式 x402、wallet/facilitator、実資産、on-chain settlementは未実行・未claimのままである。

## 20. Cloud Run revision full URI 比較修正後の再固定

実施日: 2026-08-16（Asia/Tokyo）

§19後の独立レビューで、Cloud Run revisionの`status.imageDigest`はdigest単体ではなく`repository/image@sha256:...`のfull immutable URIを返すのに、deploy scriptがdigest単体と比較して正常deploy後もexit 4になる問題が判明した。post-deploy gateを`revision image == candidate IMAGE_REFERENCE`のfull URI完全一致へ修正した。

fake gcloud回帰は、固定repositoryのexact full URIでexit 0、同じdigestのwrong repositoryと同じrepositoryのwrong digestでそれぞれexit 4を確認した。既存serviceのNEW-only refusalも維持している。Cloud APIはfakeだけを使い、実Cloud Run deployは行っていない。

修正後sourceはcommit`9730a597a3359f7ecac0f2bf10513a80f9b3c56e`、worktree digest `sha256:9c8de67d9fc5bbcf5d935b9af5be4991d9d0c550da6eb1a71127f1f8ce01a7f1`、232 files。Git-visible clean contextから`linux/amd64`をno-cache再buildし、最終local imageを`sha256:68d6489c9091062e30c31d2b6287fb290c37c6bf94019683aaf4f3c274cc2529`へ再固定した。

| gate | result |
| --- | --- |
| regression | payment 166 / evaluator 17 / jury 13 collected、全suite PASS、unexpected skip 0 |
| real Chromium | 1 PASS、`payment_user_agent`二承認とrefresh後completed |
| markers | spike 11 / unit 11 / AP2 17 / x402 simulation 2 / integration 45 / security 44 / restart 27 / migration 4 / concurrency 2 / container 5 / browser 1、全PASS |
| release validator | `status=PASS`, `failures={}` |

新artifact SHAはregression `sha256:a5d65ccaa4de9634d1cf36bd1177d440bc4fd7787600d706da1c5d258ac2781e`、browser `sha256:e4fff3b9c823ef0eab5cea4e32a77e31cadb00c7f42064027d7ff7ecc0cbc7e8`、conformance `sha256:e22cf092ed1c79a087d3c697d8a567504c98ad0c8147fc5da5ef9bf58159d25f`、release validation `sha256:e4bffbefe1c89e785bcdfcb3ce2a6b1d14fc9de3c349dfa87a4b7f9edffd1d30`。candidateは`LOCAL_VALIDATED_NOT_PUSHED`で全byte bindingと一致する。

**LOCAL PASS — full URI比較修正を含む新exact amd64 candidateの全gateを再実行済み。Artifact Registry pushはNOT RUN、Cloud Run deployはNOT RUN。**

## 21. full URI比較修正後candidate 最終独立判定

実施日: 2026-08-16（Asia/Tokyo）

本節は§19の `sha256:a4b65095...8552` 判定をsupersedeし、§20の修正後candidateを独立再試験した最終結果である。対象tag `enterprise-a2a-pf:payment-user-agent-cloudrun-amd64` のexact image ID / RepoDigestは `sha256:68d6489c9091062e30c31d2b6287fb290c37c6bf94019683aaf4f3c274cc2529`、image metadataは `linux/amd64` と完全一致した。旧 `a4b65095...8552` は最終candidateではない。

current artifact SHAはregression `sha256:a5d65ccaa4de9634d1cf36bd1177d440bc4fd7787600d706da1c5d258ac2781e`、browser `sha256:e4fff3b9c823ef0eab5cea4e32a77e31cadb00c7f42064027d7ff7ecc0cbc7e8`、conformance `sha256:e22cf092ed1c79a087d3c697d8a567504c98ad0c8147fc5da5ef9bf58159d25f`、release validation `sha256:e4bffbefe1c89e785bcdfcb3ce2a6b1d14fc9de3c349dfa87a4b7f9edffd1d30`、candidate `sha256:8065004cb0187824978b83643cc7bae1471d53ed7864014a37a423972121e38e`。`cloud-run-candidate.json` の4 artifact byte SHA、image、platform、manifest、conformance evidenceはすべてcurrent bytesと一致し、statusは `LOCAL_VALIDATED_NOT_PUSHED`、registry image/digestは `NOT_PUSHED`。`cloud_run_candidate.py verify-local` はexit 0だった。

clean-context sourceはcommit `9730a597a3359f7ecac0f2bf10513a80f9b3c56e`、232 files、worktree digest `sha256:9c8de67d9fc5bbcf5d935b9af5be4991d9d0c550da6eb1a71127f1f8ce01a7f1` でcandidate記録と一致した。build必須JSON 11件はすべて存在し、`git ls-files --cached --others --exclude-standard` のclean contextで可視、ignored 0、JSON構文正常だった。

source treeをmountせずoutput directoryだけをmountしてexact amd64 imageをfresh実行した。paymentは **166/166 PASS**、evaluation-runnerは **17/17 PASS**、jury-workerは **13 collected / 5 PASS / allowlist済み8 skips**、unexpected skip 0。実Chromium/CDP evidenceも `payment_user_agent`、request → `承認` → `承認` → refresh、`completedAfterRefresh=true` でPASSした。

current regression/browser/conformanceを同じexact imageへread-only mountしたfull validatorはexit 0、`status=PASS`, `failures={}`。全11 markerは `spike=11`, `unit=11`, `contract_ap2=17`, `contract_x402_simulation=2`, `integration=45`, `security=44`, `restart=27`, `migration=4`, `concurrency=2`, `container=5`, `browser=1` で、failure/error/skip/xfailは0だった。

今回の修正点に対するfocused container testもexact image内で **1/1 PASS**。fake gcloudに対しpost-deployのrevision imageがcandidateのfull immutable URIと完全一致すると成功し、同じdigestでもwrong repository、同じrepositoryでもwrong digestならexit 4で拒否することを確認した。実Cloud APIは呼んでいない。

**FINAL INDEPENDENT PASS — `sha256:68d6489c...2529` のlocal `linux/amd64` candidate、artifact/source binding、全11 marker、payment/evaluator/jury、実Chromium、clean-context必須JSON、post-deploy full URI gateはすべて独立PASS。§19の旧candidate判定は本節がsupersedeする。push / deployは指示どおりNOT RUN。** 公式 x402 / wallet / facilitator / real asset / on-chain settlementは引き続き **NOT RUN / not claimed**。

## 22. 一時Cloud Run demoのデプロイ・公開ブラウザ実測

実施日: 2026-08-16（Asia/Tokyo）

§21で固定したexact `linux/amd64` imageをArtifact Registryへpushし、NEW-only guardが確認した新規serviceとしてCloud Runへデプロイした。

| 項目 | 実測値 |
| --- | --- |
| project / region / service | `gen-lang-client-0585901015` / `asia-northeast1` / `payment-user-agent-demo` |
| revision | `payment-user-agent-demo-00001-77d` |
| URL | `https://payment-user-agent-demo-343404053218.asia-northeast1.run.app` |
| registry image | `asia-northeast1-docker.pkg.dev/gen-lang-client-0585901015/secure-mediation-agent/payment-user-agent-demo@sha256:68d6489c9091062e30c31d2b6287fb290c37c6bf94019683aaf4f3c274cc2529` |
| environment | `EPHEMERAL_CLOUD_RUN_DEMO=true`, `APP_ENV=ephemeral-demo`, `DEV_MODE=false` |
| scaling | min 1 / max 1、concurrency 1 |

デプロイ後の実測では`/health`、`/auth/deployment`、`/auth/firebase-config`がPASSし、未認証のrootと`/list-apps`はloginへredirectした。これは起動、公開認証境界、Firebase public config、一時deployment表示のsmoke PASSである。

Firebase Authorized Domainsへ`payment-user-agent-demo-343404053218.asia-northeast1.run.app`を追加し、公開remote browserでEmailログインに成功した。`/dev-ui/?app=payment_user_agent&session=...&userId=user`へredirectし、`payment_user_agent`が選択済みで他のroot appがないことを確認した。

依頼後の「計画の承認」は、まだ決済されない旨と1250 USDを表示した。完全一致の承認後、「決済の承認」は課金警告、Demo Merchant、`customerTotal=1250`、simulated／`NOT CONFORMANT`を表示した。2回目の完全一致承認後、Demo booking confirmed、AP2 evidence、`AP2 v0.2 Human Present demo`、実資産／on-chainなしを表示して完了した。reload後も認証、app選択、完了状態を維持した。revisionの直近30分のerror logは0件だった。

このserviceは`durability=NOT PROVIDED`の一時デモで、revision再起動・置換によりSQLite状態とdemo鍵が失われ得る。耐久Cloud Run paid workflow、複数instance、production identity/KMSは**NOT RUN / not claimed**。公式x402、wallet／facilitator、実資産、on-chain settlementも引き続き**NOT RUN**である。

追加の拒否／期限切れ／replay、長時間運転、負荷、alert、追加partnerは今後のedge caseとして追跡する。これらの未実施を今回のremote browser PASSへ含めない。

**DEPLOYED EPHEMERAL REMOTE BROWSER PASS — exact registry digestで新規revisionが起動し、公開health／auth境界、Firebase認証、二承認、完了、reload後の状態維持を確認済み。**
