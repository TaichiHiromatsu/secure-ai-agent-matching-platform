# AP2 / x402 統合 — 独立コードレビュー

- 実施日: 2026-08-16（Asia/Tokyo）
- 対象ブランチ: `codex/ap2-x402-integration`
- 対象: current worktree の tracked diff と新規実装、`AP2_X402_DOCUMENT_INDEX.md`、統合要件／設計、実装証跡、独立テスト報告の最新節
- 判定対象: imminent ephemeral Cloud Run demo release

## 結論

**FAIL — 現在の専用 deploy script で Cloud Run へリリースしてはならない。**

決済コアの二承認、replay／二重効果防止、loopback Merchant A2A、公開 route 隔離、秘密値非表示、一時 mode の readiness 表示、`payment_user_agent` の単独公開について、今回の範囲で新たな hard defect は確認しなかった。一方、実際の Cloud Run 配布経路に P0 blocker が3件あり、固定済み PASS candidate をそのまま起動できず、clean checkout から同じ release を再現することもできない。

公式 x402、wallet／facilitator、実資産、on-chain settlement は意図どおり **NOT RUN / not claimed** であり、今回の FAIL 理由ではない。

## Release blockers

### P0-1 — Cloud Run 非対応の `linux/arm64` image を生成・配布する

`deploy/deploy-payment-demo-cloudrun.sh:59` は `docker build --no-cache` を platform 指定なしで実行する。レビュー端末は `arm64` で、固定済み候補 `enterprise-a2a-pf:ap2-payment-user-final` も実測で `arm64/linux` だった。

Google Cloud Run の container runtime contract は Linux `x86_64` ABI を要求し、multi-architecture image でも `linux/amd64` を含む必要がある。このため現端末で script を実行して push する単一 architecture image は Cloud Run で起動できない。参照: [Cloud Run container runtime contract](https://docs.cloud.google.com/run/docs/container-contract)

修正条件:

- build／push を `linux/amd64` に固定するか、`linux/amd64` を含む multi-arch manifest を生成する。
- 配布予定の registry digest を使って Cloud Run 起動試験を行い、その exact digest に release evidence を再拘束する。

### P0-2 — deploy script が検証済み固定 image ではなく未検証の再 build を配布する

`deploy/deploy-payment-demo-cloudrun.sh:53-63` は timestamp tag を作り、毎回 `docker build --no-cache` した直後の image を push／deploy する。最終 PASS 証跡が拘束するのは local image ID `sha256:1b743079...e9d5c29` だが、`AP2_X402_TEST_REPORT.md` 自身が no-cache rebuild は bit-for-bit 同一でないと記録している。専用 deploy script には、build 結果の digest 確認、release validator 実行、固定候補との一致確認、registry digest の記録がない。

従って script 実行時に Cloud Run が受け取る image は、文書末尾の PASS 対象ではない。`gcloud run deploy` は tag ではなく exact digest も受け付けるため、現状の「固定 candidate の release evidence」という主張と配布 command が一致していない。

修正条件:

- `linux/amd64` で作った deployable candidate を一度だけ build／test／freeze する。
- Artifact Registry へ同一 candidate を配布し、registry の exact digest を取得する。
- deploy は `...@sha256:<registry-manifest-digest>` を指定し、Cloud Run revision が解決した digest を事後確認する。
- その exact deployable digest に regression／browser／conformance／validator evidence を再拘束する。

### P0-3 — runtime／認証／release gate の必須 JSON が Git 管理外

root `.gitignore:132` の `*.json` により、次の必須 file がすべて ignored かつ untracked である。

| file | 失われる機能 |
| --- | --- |
| `deploy/auth/firebase-config.json` | `/auth/firebase-config` が 503 となり Firebase login が開始できない |
| `secure_mediation_agent/spec_manifest.json` | workflow readiness の `specPins` が false となり `/ready` が 503 |
| `tests/regression/suite_manifest.json` | versioned regression command が再現不能 |
| `tests/release/release_manifest.json` | release validator の既定入力が欠落 |
| `docs/ap2_x402_conformance_report.json` | 機械可読 conformance 証跡と byte binding が欠落 |

現在の local Docker build は ignored file も build context に入るため偶然 PASS するが、branch／commit／clean checkout にはこれらが含まれない。したがって現在の「source bind mount なし」「clean image」「同一証跡で再現可能」という主張は repository state と一致しない。

修正条件:

- 非秘密の必須 JSON を明示的に Git 管理対象へ戻す。Firebase browser config は公開設定であり、秘密鍵として扱わない。
- clean checkout から image build、Firebase config endpoint、ephemeral `/ready`、regression、conformance 付き validator を再実行する。
- tracked file の exact bytes から新しい証跡 digest を再固定する。

## その他の hard command defects

### P1-1 — Firebase 認証時の CLI／one-command verifier が誤った cookie 名を送る

server が設定・検証する cookie 名は `deploy/auth/verify.py:24` の `__Host-payment-session` である。一方、`secure_mediation_agent/workflow/client.py:35-36` と `scripts/verify_ap2_x402_runtime.py:31-34,146-150` は `--session-cookie` の値を `session=<value>` として送る。

そのため `AP2_X402_DEMO_GUIDE.md` が案内する `--session-cookie`／`WORKFLOW_SESSION_COOKIE` は Firebase 有効環境で認証に使えず、`verify_payment_demo.sh` も同じ経路で失敗する。ブラウザの server-owned cookie path 自体は正しい。

修正条件: client と verifier を `__Host-payment-session` に統一し、認証済み public `/mediation-api/` に対する CLI／verifier test を追加する。

### P1-2 — 文書化した local 起動 command に実行権限がない

`deploy/run-local.sh` は current diff で mode `100755 -> 100644` になっている。README、runbook、demo guide はいずれも `./deploy/run-local.sh --no-cache` を案内しているため、clean checkout では `Permission denied` となる。

修正条件: executable bit を復元し、文書記載どおりの直接実行を test する。

## 確認できた PASS 領域

以下は current implementation と固定 candidate の focused test で hard issue を確認しなかった。

- Firebase ID token の固定 project audience／issuer／subject／expiry 検証、same-origin＋CSRF session exchange、`Secure; HttpOnly; SameSite=Strict; __Host-` cookie、request ごとの再検証。
- nginx による client `X-Verified-Identity` の置換、`/payment/`、`/paid-agent/`、`/internal/`、直接 `/v1/` の 404、公開決済 route の `/mediation-api/` 集約。
- plan approval と payment approval の別 state／record／nonce、完全一致の `承認`、payment approval 前 settlement 0。
- stable settlement attempt／idempotency key／CAS／outbox recovery による replay 時の simulated balance 二重反映防止。
- loopback `:8005` Merchant の activation、audience、operation、workflow／task／order capability binding と persistent TaskStore。
- role 別 P-256 private key file、mode `0600`、公開 view／error への raw mandate、credential、proof、private key 非露出。
- ephemeral mode の `target=ephemeral-cloud-run-demo`、`durability=NOT PROVIDED`、reset warning、durable marker 非表示、`DEV_MODE=true` 起動拒否。
- ADK discovery root が `payment_user_agent` 一つで、official x402 profile が fail closed。
- fixed project／region／service、existing service の build 前拒否、single instance／concurrency 1、`DEV_MODE=false`。ただし P0-1／P0-2 のため deploy command 全体は FAIL。

Focused test:

```text
docker run --rm --entrypoint /app/.venv/bin/python \
  enterprise-a2a-pf:ap2-payment-user-final \
  -m pytest -p no:cacheprovider -q \
  tests/security/test_firebase_session.py \
  tests/security/test_release_boundaries.py \
  tests/container/test_release_image_contract.py \
  tests/integration/test_external_merchant_a2a.py \
  tests/workflow/test_concurrency.py

32 passed
```

静的確認は `git diff --check` と対象 shell の `bash -n` が PASS。秘密値 scan では新規 private key／service secret の commit は確認しなかった。Firebase API key は browser 用公開設定だが、現状は P0-3 のとおり未収録である。

## リリース再判定条件

P0-1〜P0-3を修正し、次を同一 `linux/amd64` registry digest に対して再実行するまで FAIL を維持する。

1. clean checkout からの image build。
2. source mount なしの focused/full regression と release validator。
3. actual ephemeral runtime の `/auth/firebase-config`、Firebase session、`/ready`、`payment_user_agent` の二承認 browser flow。
4. registry exact digest 指定の NEW-only Cloud Run deploy と、revision digest／startup／login／主要フローの確認。
5. P1-1 の authenticated CLI／verifier と P1-2 の記載 command の確認。

負荷、長時間運転、追加 partner、非主要 edge case はこの再判定の blocker に追加しない。公式 x402／on-chain は引き続き別スコープの **NOT RUN** とする。

---

## Superseding re-review verdict — 2026-08-16

**PASS for the local pre-push release gate.** 先の FAIL 判定と P0-1〜P0-3／P1-1〜P1-2 は、以下の新しい exact candidate に対して解消した。

- local image ID: `sha256:68d6489c9091062e30c31d2b6287fb290c37c6bf94019683aaf4f3c274cc2529`
- platform: `linux/amd64`
- worktree source digest: `sha256:9c8de67d9fc5bbcf5d935b9af5be4991d9d0c550da6eb1a71127f1f8ce01a7f1`
- candidate status: `LOCAL_VALIDATED_NOT_PUSHED`
- registry push: **NOT RUN**
- Cloud Run deploy: **NOT RUN**
- official x402／wallet／facilitator／on-chain settlement: **NOT RUN / not claimed**

この PASS は、固定 candidate を split release flow の push step へ進めてよいという判定である。Cloud Run に配布済みという判定ではない。deploy step は push 後の `PASS` candidate artifact と fixed repository の immutable `@sha256` がなければ fail closed となる。

### 解消を確認した release blockers

- build は `docker buildx build --platform linux/amd64 --no-cache --load` に固定され、実測 image も `linux/amd64` である。
- build／push／deploy は3つの script に分離された。build は publish/deploy せず、push は deploy せず、deploy は build/push しない。
- build context は `git ls-files --cached --others --exclude-standard` の Git-visible file だけから作られ、Firebase config、spec manifest、regression manifest、release manifest、machine-readable conformance report を含む必須 JSON はすべて non-ignored で context へ収録された。release commit にはこれらの新規 file を必ず含めること。
- CLI、workflow client、black-box verifier の cookie 名は server と同じ `__Host-payment-session` に統一された。
- `deploy/run-local.sh` と release 関連 script の executable bit はすべて復元／付与済みである。
- fixed target は `gen-lang-client-0585901015/asia-northeast1/payment-user-agent-demo` で上書き引数を受け付けず、既存 service を read-only preflight で検出した場合は deploy 前に exit 3 となる。
- deploy は fixed Artifact Registry repository の exact `@sha256` のみを受け付け、registry 実測 digest と比較する。Cloud Run 作成後は ready revision の full `repo@sha256` URI を candidate URI と exact 比較する。wrong repository と wrong digest の両方が exit 4 となる negative test も PASS した。
- deployment env は `EPHEMERAL_CLOUD_RUN_DEMO=true,APP_ENV=ephemeral-demo,DEV_MODE=false` に固定され、single instance／concurrency 1、durability `NOT PROVIDED` である。専用 release scripts に DB resource 作成や durable storage の暗黙 provision はない。

### 独立再検証結果

Candidate binding:

```text
docker inspect: sha256:68d6489c...2529 linux/amd64
source file count: 232
verify-local: PASS
verify-deploy before push: rejected (candidate status must be PASS)
```

Exact image 内の focused tests:

```text
44 passed, 1 dependency warning
```

対象は Firebase session／public boundary、release image contract／NEW-only mock／full revision URI negative cases、payment user agent cookie、external Merchant A2A、concurrency、role rejection／secret output、identity／workflow API である。warning は `requests` の dependency compatibility warning で、test failure ではない。

Versioned evidence:

```text
payment-release: 166 passed
evaluation-runner: 17 passed
jury-worker: 13 passed
release validator: PASS, 11/11 marker suites clean
real Chromium: PASS, only payment_user_agent, completed after refresh
```

Exact image の local ephemeral nginx smoke:

```text
/auth/firebase-config: 200, projectId=mediation-a2a-platform
/auth/csrf: Secure; SameSite=strict; __Host-payment-csrf
/auth/session without same-origin CSRF: 403
/mediation-api/ready without session: 302 -> /login
/payment/, /paid-agent/, /internal/, direct /v1/: 404
/auth/deployment: ephemeral=true, durability=NOT PROVIDED,
                  officialX402=NOT RUN, onChainSettlement=NOT RUN
```

`git diff --check` と release shell の `bash -n` も PASS した。

### 次のステップの境界

1. release commit に current Git-visible 新規 file（特に必須 JSON）を含める。
2. 明示的な publish 承認後に `deploy/push-payment-demo-candidate.sh` でこの exact local candidate を push し、registry digest に拘束された `PASS` artifact を作る。
3. deploy 直前に fixed target が仍旧 NEW であることを確認する。既存なら一切変更せず中止する。
4. deploy 承認後のみ exact registry digest で作成し、revision full image URI、startup、Firebase login、二承認 flow を actual URL で確認する。

以上の Cloud-side 手順は本レビュでは実行していない。したがって、この superseding PASS を「Cloud Run deployment 実績」や「公式 x402／on-chain 成功」と表現してはならない。

---

## 最終 provenance 独立再レビュ — 2026-08-16

**PASS — `sha256:a22c3e696299c3c73dcf2391cba3df16c4e95c9333e72ad3ed8c0a19851a38bc` は、local pre-push provenance gate を通過した exact `linux/amd64` candidate である。**

本節は直前の local candidate `68d6489c...2529` に対する判定を、最終 provenance 修正後の candidate について supersede する。この独立再レビュで Artifact Registry push と Cloud Run deploy は実行していない。

| binding | 独立確認値 |
| --- | --- |
| local image | `sha256:a22c3e696299c3c73dcf2391cba3df16c4e95c9333e72ad3ed8c0a19851a38bc` |
| platform | `linux/amd64` |
| source base commit | `dbd88afc31da9426f159efbeff08be7870dd8c65` |
| worktree source digest | `sha256:c42a61f72c61e357e3ff410b4026125920db305be6df163f140606047c4b741f` |
| source set | 233 files, `path-mode-size-bytes-v1` |
| candidate status | `LOCAL_VALIDATED_NOT_PUSHED` |
| candidate artifact SHA | `sha256:cfe0403e0546542b1d3f7df25b5125856dba79f0e2554c974e599dd5cc3f3cc7` |

### Provenance 設計の判定

- `baseCommit` は40桁 lowercase hex に限定され、`git cat-file ...^{commit}` で commit object の実在、`git merge-base --is-ancestor` で current `HEAD` の祖先性を検証する。実測で candidate base は current `HEAD` の有効な commit ancestor である。
- commit の完全一致は要求せず、current Git-visible release source set の path、executable mode、size、exact bytes を SHA-256 で再計算し、`worktreeDigest`、`fileCount`、`algorithm` を厳密比較する。
- この分離により、artifact、conformance 証跡、レビュ文書など source set 外だけの後続 commit は、base の祖先性と source digest が不変なら許容される。candidate artifact 自身を source digest へ含めないため、release artifact を commit するたびに自己参照で無効化する問題は解消した。
- source byte 変更、executable mode 変更、file count／algorithm 不一致、不正／不存在 commit、non-ancestor base は fail closed となる。tracked source が欠落すれば regular-file check または source digest／file count 不一致で拒否される。
- legacy `source.commit` は同じ base commit として受理される。実 artifact を legacy field へ変換した end-to-end `verify-local` は exit 0。`baseCommit` と legacy `commit` が両方あり値が異なる場合は拒否された。

Exact image 内の `scripts/cloud_run_candidate.py` と provenance test の SHA-256 は host worktree と一致し、mode もそれぞれ `0755`／`0644` で一致した。source mount を使わない focused test は次の結果である。

```text
tests/container/test_cloud_run_candidate_source.py
tests/container/test_release_image_contract.py
tests/container/test_release_validator_binding.py
tests/security/test_release_boundaries.py
tests/security/test_firebase_session.py

22 passed, 1 dependency warning
```

追加した provenance 7回帰は、source-set 外の後続 commit 許容、source bytes／mode 拒否、non-ancestor 拒否、invalid／missing commit 拒否、legacy field 互換をすべて PASS した。warning は従来の `requests` dependency compatibility warning で、test failure ではない。

### Candidate／evidence の再検証

```text
verify-local: PASS
verify-deploy on LOCAL_VALIDATED_NOT_PUSHED: rejected
payment-release: 173 collected, PASS
evaluation-runner: 17 collected, PASS
jury-worker: 13 collected, PASS
real Chromium: PASS, payment_user_agent only, completed after refresh
release validator: PASS, failures={}
marker suites: 11/11 PASS, failure/error/skip-or-xfail 0
```

regression、browser、conformance、release-validation の全 artifact digest は candidate 記録と一致し、すべて exact `a22c3e69...a38bc` image ID へ拘束されている。必須 JSON は全件 Git-visible／non-ignored である。`git diff --check`、release shell の `bash -n`、script executable mode も PASS した。

### Build／push／deploy の fail-closed 境界

- build／push／deploy の各 script は不明な引数をすべて exit 2 で拒否した。
- push は local image ID／`linux/amd64`／`verify-local` を Cloud への書き込み前に要求し、固定 repository の registry digest を得た後も remote platform と final candidate binding を再検証する。push script は Cloud Run を操作しない。
- deploy は `PASS` status と固定 repository の immutable `@sha256` がなければ作成に進まない。fixed project／region／service、`EPHEMERAL_CLOUD_RUN_DEMO=true,APP_ENV=ephemeral-demo,DEV_MODE=false`、single instance／concurrency 1 を保持し、DB／durable storage resource を作成しない。
- NEW-only preflight は既存 fixed service を検出した場合に exit 3 で中止し、後続の deploy を実行しない。wrong revision repository／digest も post-deploy exact URI check で exit 4 となる回帰が PASS している。

現在の証跡には fixed service `payment-user-agent-demo` の既存 revision が記録されている。したがって、現状の NEW-only deploy script はその service を更新せず、意図どおり拒否しなければならない。新 candidate で既存 service を supersede するには、既存 service を触る別の明示的な lifecycle 判断が必要であり、この local PASS はその権限や実行実績を意味しない。

**最終判定:** provenance 修正に hard code blocker は確認しなかった。`a22c3e69...a38bc` は publish 判断に進められる local candidate であるが、push／deploy、新 candidate の remote runtime、公式 x402／wallet／facilitator／実資産／on-chain settlement はすべて **NOT RUN / not claimed** のままである。
