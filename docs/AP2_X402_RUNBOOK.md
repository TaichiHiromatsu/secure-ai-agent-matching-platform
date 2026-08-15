# AP2 / x402 統合決済デモ — 運用手順

本機能は、`AP2 v0.2 Human Present demo` と project-local `x402-wire-simulation/1` を組み合わせた、実資産を移動しない simulation である。公式 x402、wallet／facilitator、on-chain settlement、production identity／KMS を実装済みとは扱わない。

## 1. 構成と公開境界

| 項目 | 現行構成 |
| --- | --- |
| ADK Web の表示アプリ | `payment_user_agent` のみ |
| 内部ワークフロー | `secure_mediation_agent` package の durable workflow |
| 公開決済 API | `/mediation-api/` |
| Merchant | loopback-only A2A service `:8005`、persistent TaskStore |
| 認証 | Firebase Authentication。ローカルに限り固定 demo identity の `DEV_MODE=true` を許可 |
| profile | `x402-wire-simulation/1` / `urn:secure-a2a:extensions:x402-wire-simulation:v1` |
| rail | `exact-simulated` / `demo:local` / `USD` |
| 適合表示 | `x402 v0.1 wire-shape test fixture (NOT CONFORMANT)` |

旧 `/payment/`、`/paid-agent/`、直接 `/v1/`、`/mediation-api/internal/` は公開しない。外側からは 404 または一般化した認可エラーを返す。

## 2. 耐久ローカル環境の起動

```bash
./deploy/run-local.sh --no-cache
curl --fail http://127.0.0.1:8080/mediation-api/ready
```

既定の永続 path は次のとおり。

```text
.local/payment-data/marketplace.db
.local/payment-data/paid-agent.db
.local/payment-evidence/evidence.db
.local/ap2-demo-keys/
```

別 path を使う場合は `PAYMENT_DATA_DIR`、`PAYMENT_EVIDENCE_DIR`、`AP2_KEY_DIR` を明示する。data／evidence directory には `.durable-volume` marker が必要で、role key は permission と `kid` を検証する。anonymous container layer や `/tmp` を耐久環境として扱わない。

最終受入で安定して PASS した永続化経路は Docker-managed volume である。macOS の `/tmp` を host bind mount した独立試験では SQLite の SIGBUS／破損が一度発生し、readiness は 503 へ fail closed した。`deploy/run-local.sh` の host bind mount を耐久用途で使う場合は、利用する Docker Desktop／filesystem の組合せを事前に適格性確認する。これは複数 host 共有を認めるものではない。

ローカル認証を省略する場合だけ `.env` に `DEV_MODE=true` を設定できる。`APP_ENV=local` 以外での `DEV_MODE=true` は起動エラーになる。Firebase を使う場合、ブラウザは same-origin の `/auth/session` で ID token を server-owned session cookie に交換する。

## 3. readiness の確認

`/mediation-api/ready` は次の11領域を fail closed で確認する。

- data／evidence の durable-volume marker
- `marketplace.db`、`paid-agent.db`、`evidence.db` の schema v2
- worker heartbeat、unfinished／stale outbox
- evidence intent
- role key の permission と `kid`
- trust snapshot
- AP2／x402 仕様の固定 hash
- selected profile の排他性
- route isolation
- external Merchant A2A と persistent TaskStore

公式 x402 enablement、wallet、facilitator、on-chain settlement は正常時でも `NOT RUN` と表示される。これらを readiness の PASS と読み替えない。

## 4. 正常系と一括検証

ADK Web では `payment_user_agent` を選び、次の順で操作する。

1. 予約を依頼する。
2. 「計画の承認」を読み、完全一致の `承認` を送る。
3. 「決済の承認」、Merchant、7価格項目、期限、simulation label を読み、別の完全一致 `承認` を送る。
4. `completed`、Artifact、AP2 Checkout／Payment Receipt、simulation receipt の安全な ID／digest を確認する。

自動検証:

```bash
docker exec secure-platform /app/scripts/verify_payment_demo.sh
```

CLI:

```bash
docker exec secure-platform /app/.venv/bin/python \
  /app/user-agent/payment_cli.py \
  --workflow-url http://127.0.0.1:8080/mediation-api \
  --prompt "デモ予約を1件取得して" \
  --plan-approval "承認" \
  --payment-approval "承認"
```

## 5. 状態と回復方針

| 状態／事象 | 正本 | 回復操作 |
| --- | --- | --- |
| 計画または決済の承認待ち | `marketplace.db` の workflow／approval target | 同じ認証主体と session で active workflow を再取得 |
| Merchant Task の応答不明 | `paid-agent.db` の persistent TaskStore | 同じ Task／operation ID で `tasks/get` 相当を照会。新 Task を作らない |
| outbox lease の途中終了 | `marketplace.db.outbox` | lease expiry 後に worker が同じ operation ID と request digest で再開 |
| evidence commit の途中終了 | `evidence_intents_v2` と exact digest | reconciler が evidence DB と照合し、aggregate を進めるか `reconciliation_required` にする |
| settlement 結果不明 | 保存済み attempt／external ID | 新しい charge を作らず、同じ external ID を照会 |
| settle 成功後の Merchant commit failure | immutable payment evidence | `refund_required` に進み、append-only refund record で補償 |
| refund 結果不明 | provider refund ID | 同じ ID だけを照会し、証跡なしに `refunded` としない |

container を再作成するときは、同じ data／evidence／key mount を使う。稼働中の SQLite を host 側から同時に開いて検査しない。検査前に container を停止する。

## 6. オフライン証跡検証

```bash
docker exec secure-platform /app/.venv/bin/python \
  /app/scripts/verify_ap2_x402_evidence.py <workflow-id>
```

このコマンドは plan authorization、scoped capability、Checkout、closed Mandates、credential、synthetic payload、AP2 Receipts、ordered simulation receipt、trust snapshot を公開情報と保存済み exact bytes から再検証する。

## 7. migration と rollback

三DBを v2 へ移行する前に writer を停止し、専用の durable backup directory を用意する。

```text
scripts/migrate_ap2_x402_v2.py plan ...
scripts/migrate_ap2_x402_v2.py apply ...
scripts/migrate_ap2_x402_v2.py verify ...
```

pre-cutover で user／business write が一件もない場合に限り、manifest と checksum を確認して次を使える。

```text
scripts/migrate_ap2_x402_v2.py restore-pre-cutover \
  --manifest ... --confirm RESTORE-PRE-CUTOVER
```

v2 write 受入後は pre-v2 backup へ戻さない。v2 DB／evidence／key／audit を保全し、schema compatibility を検証済みの previous image または forward fix を使う。結果不明の settlement／refund を DB rollback で成功または失敗と推測しない。

## 8. reset

通常の process restart や container recreate では `.local/` の永続 data を削除しない。

デモ reset が必要な場合は、container を停止し、`PAYMENT_DATA_DIR`、`PAYMENT_EVIDENCE_DIR`、`AP2_KEY_DIR` の解決済み path と対象 workflow 件数を確認する。誤削除を避けるため、runbook から recursive delete command は提示しない。保存が必要な証跡を退避してから、新しい空の専用 directory を環境変数で指定する。

## 9. Cloud Run 一時デモ

Cloud Run 一時デモの release は build／push／deploy を混在させない。次の順序を崩してはならない。

```bash
./deploy/build-payment-demo-candidate.sh
jq '{status,platform,localImageId,source,registry,artifacts,embedded}' \
  artifacts/cloud-run-candidate.json
./deploy/push-payment-demo-candidate.sh
./deploy/deploy-payment-demo-cloudrun.sh
```

- build script は Git-visible clean context から `linux/amd64` image を作り、source mount なしの regression、実 Chromium、全 marker validator を通す。registry／Cloud Run は変更しない。
- push script は `LOCAL_VALIDATED_NOT_PUSHED` candidate と loaded image の exact ID／platform／source／artifact binding が一致する場合だけ固定 Artifact Registry repository へ publish する。Cloud Run は変更しない。
- deploy script は build／push を行わない。status `PASS` の tracked candidate が持つ固定 repository の immutable `@sha256:` reference だけを受け付ける。source／artifact／embedded result の不一致は deploy 前に拒否する。
- deploy script は固定 target の read-only NEW-only preflight を行い、既存 `payment-user-agent-demo` service があれば拒否する。成功後も ready revision の image digest が candidate と一致しなければ失敗として扱う。
- provenanceはbuild時HEADを`baseCommit`として記録し、40桁hex、commit objectの存在、current HEADのancestorを確認する。後続commitとの完全一致は要求せず、release sourceの`worktreeDigest`、`fileCount`、`algorithm`を厳密一致させるため、文書／証跡更新は許容しつつsource byte／mode変更は拒否する。

- `EPHEMERAL_CLOUD_RUN_DEMO=true` を必須とし、状態と鍵が reset され得る警告を表示する。
- `DEV_MODE=true` を拒否し、Firebase Authentication を使う。
- exact `linux/amd64` candidateをArtifact Registryへpushし、既存一時demo serviceの最終revisionへ反映済み。初回作成用NEW-only scriptは引き続き同名serviceを拒否する。
- durable Cloud Run paid workflow、複数 instance、production deployment の証拠にはしない。

現在の一時デモ:

| 項目 | 実測値 |
| --- | --- |
| project / region / service | `gen-lang-client-0585901015` / `asia-northeast1` / `payment-user-agent-demo` |
| revision / traffic | `payment-user-agent-demo-00002-nt7` / 100% |
| URL | `https://payment-user-agent-demo-343404053218.asia-northeast1.run.app` |
| immutable image | `asia-northeast1-docker.pkg.dev/gen-lang-client-0585901015/secure-mediation-agent/payment-user-agent-demo@sha256:a22c3e696299c3c73dcf2391cba3df16c4e95c9333e72ad3ed8c0a19851a38bc` |
| environment | `EPHEMERAL_CLOUD_RUN_DEMO=true`, `APP_ENV=ephemeral-demo`, `DEV_MODE=false` |
| scaling | min 1 / max 1、concurrency 1 |
| durability | `NOT PROVIDED`。revision再起動・置換で状態と鍵が失われ得る |

最終revisionでFirebase cookie認証を維持し、Emailログイン後に`/dev-ui/?app=payment_user_agent&session=...&userId=user`へredirectして、単独root appの`payment_user_agent`が選択された。公開remote browserで依頼、計画承認、決済承認、完了までPASSし、reload後も認証・app選択・完了状態を維持した。simulation／`NOT CONFORMANT`表示も維持した。

## 10. 将来課題

主要ブラウザフローの後に追加する non-critical edge case は、独立した issue として優先順位を付けてよい。現在の主な将来課題は次のとおり。

- 外部 partner との interoperability 拡張。
- 追加の異常系、長時間運転、運用 alert、負荷試験。
- 複数 host／複数 instance 用の shared transactional DB／queue。
- production identity enrollment、KMS／HSM、規制・会計・監査要件。
- 公式 x402 の network／asset／wallet／facilitator／TLS／on-chain settlement と ACC-030。

最後の3項目は production または conformance の前提であり、単なるデモ上の小さな edge case ではない。
