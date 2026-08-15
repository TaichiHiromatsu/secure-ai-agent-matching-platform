# 決済機能の運用ガイド

- 対象読者: ローカル運用者、開発者、障害対応者、リリース担当者
- 前提: [エージェント間決済の概要](README.md)、[アーキテクチャ](ARCHITECTURE.md)
- 次に読む文書: [検証ガイド](VERIFICATION.md)、[実演ガイド](DEMO.md)

## 対応する実行形態

| 実行形態 | 用途 | 耐久性 |
| --- | --- | --- |
| 耐久ローカル | 開発、回復、migration、受入試験 | 明示したdata／evidence／key volumeを使う単一host・単一containerに限定 |
| Cloud Run一時デモ | 認証付きの短時間実演 | `NOT PROVIDED`。revision再起動・置換でSQLiteとdemo keyを失い得る |

どちらもproject-local simulationであり、実資産、wallet、facilitator、on-chain settlementを使わない。Cloud Runへ配置したことをproduction決済または耐久構成の証拠にしてはならない。

## 公開境界

| 項目 | 構成 |
| --- | --- |
| 利用者向けapp | `payment_user_agent` |
| 公開決済API | 認証済み`/mediation-api/` |
| Merchant | loopback `127.0.0.1:8005` |
| 旧route | `/payment/`、`/paid-agent/`、`/internal/`、`/v1/`は404 |
| local認証 | `APP_ENV=local`に限り`DEV_MODE=true`を許可 |
| 非local認証 | Firebase Authentication、server-owned session cookie |

外部から送られたidentity headerを信頼しない。nginxのauth subrequestが返した検証済みidentityだけをworkflowへ渡す。

## 耐久ローカル環境

### 起動

リポジトリ直下で実行する。

```bash
./deploy/run-local.sh --no-cache
curl --fail http://127.0.0.1:8080/mediation-api/ready
```

二回目以降、既存imageを使う場合は`--no-build`を指定できる。

```bash
./deploy/run-local.sh --no-build
```

既定の永続path:

```text
.local/payment-data/marketplace.db
.local/payment-data/paid-agent.db
.local/payment-evidence/evidence.db
.local/ap2-demo-keys/
```

別pathを使う場合は、起動前に次を明示する。

```bash
export PAYMENT_DATA_DIR=/absolute/path/to/payment-data
export PAYMENT_EVIDENCE_DIR=/absolute/path/to/payment-evidence
export AP2_KEY_DIR=/absolute/path/to/ap2-demo-keys
./deploy/run-local.sh --no-cache
```

専用directory以外を指定しない。data／evidence directoryの`.durable-volume` marker、key fileのpermission、roleごとの`kid`をreadinessが検証する。anonymous container layerや`/tmp`を耐久環境として扱わない。

macOSのhost bind mountでは、Docker Desktopとfilesystemの組み合わせによってSQLiteのSIGBUSまたは破損が起こり得る。採用するmount方式を事前に適格性確認し、readinessが503を返した状態で処理を続けない。複数hostや複数containerから同じSQLiteを共有しない。

### 認証

ローカルでFirebase loginを省略する場合だけ、`.env`に次を設定できる。

```text
APP_ENV=local
DEV_MODE=true
```

`APP_ENV=local`以外での`DEV_MODE=true`は起動を拒否する。Firebaseを使う場合、ブラウザはsame-originの`/auth/session`でID tokenを交換し、serverが`Secure`、`HttpOnly`、`SameSite=Strict`の`__Host-payment-session`を設定する。

## 準備完了判定

`/mediation-api/ready`は少なくとも次の領域をfail closedで確認する。

- data／evidenceのdurable marker
- `marketplace.db`、`paid-agent.db`、`evidence.db`のschema
- worker heartbeat、unfinished／stale outbox
- cross-DB evidence intent
- role keyのpermissionと`kid`
- public trust snapshot
- AP2／A2A x402仕様の固定hash
- 選択profileの排他性
- public／internal route分離
- Merchant A2Aとpersistent TaskStore

readinessの`officialX402`、wallet、facilitator、on-chain項目が`NOT RUN`であることは正常なsimulation境界であり、公式profileのPASSではない。

Cloud Run一時デモでは、durable markerを偽装せず、target、`durability=NOT PROVIDED`、state reset warningを返す。

## 正常系の確認

画面での実演は[実演ガイド](DEMO.md)を参照する。一括検証は次を使う。

```bash
docker exec secure-platform /app/scripts/verify_payment_demo.sh
```

CLIで同じworkflowを確認する場合:

```bash
docker exec secure-platform /app/.venv/bin/python \
  /app/user-agent/payment_cli.py \
  --workflow-url http://127.0.0.1:8080/mediation-api \
  --prompt "デモ予約を1件取得して" \
  --plan-approval "承認" \
  --payment-approval "承認"
```

Firebase認証を有効にした環境では、有効なsession cookieをCLIへ渡す。内部port `:8004`を認証回避の経路として直接呼ばない。

## 再起動と回復

containerを再作成するときは、同じdata、evidence、key mountを使う。稼働中のSQLiteをhost側から同時に開いて検査しない。整合性を直接調べる場合はwriterとcontainerを停止する。

| 状況 | 正本 | 回復方法 |
| --- | --- | --- |
| 計画または決済の承認待ち | workflowとapproval target | 同じ認証主体とsessionでactive workflowを再取得する |
| Merchant応答が不明 | Merchant TaskStore | 同じTask／operation IDを照会し、新Taskを作らない |
| outbox処理中に停止 | outbox rowとlease | lease expiry後、同じoperation IDとrequest digestで再開する |
| evidence保存中に停止 | evidence intentとdigest | evidence DBのexact artifactと照合してackまたは再試行する |
| settlement結果が不明 | attemptとexternal ID | 新chargeを作らず同じexternal IDをreconcileする |
| settle後のfulfillment失敗 | immutable payment evidence | `refund_required`へ進み、append-onlyな補償recordを作る |
| refund結果が不明 | provider refund ID | 同じIDを照会し、証拠なしに`refunded`へ進めない |

workerが自動回復できない`reconciliation_required`と`refund_required`は、証跡を保全したままoperator手順へ送る。DB rowを直接書き換えて完了扱いにしない。

## オフライン証跡検証

完了したworkflowを、保存済みartifactとpublic keyだけで検証する。

```bash
docker exec secure-platform /app/.venv/bin/python \
  /app/scripts/verify_ap2_x402_evidence.py <workflow-id>
```

検証内容は[検証ガイド](VERIFICATION.md#オフライン証跡検証)を参照する。

## 移行とロールバック

三DBを移行する前にwriterを停止し、専用の永続backup directoryを用意する。

```text
scripts/migrate_ap2_x402_v2.py plan ...
scripts/migrate_ap2_x402_v2.py apply ...
scripts/migrate_ap2_x402_v2.py verify ...
```

`plan`で対象path、schema version、checksum、予定操作を確認してから`apply`する。適用途中の停止と再適用に耐えることを、fixtureのcopyで先に確認する。

pre-cutoverで利用者またはbusiness writeが一件もない場合に限り、manifestとchecksumを確認してpre-cutover backupへ戻せる。

```text
scripts/migrate_ap2_x402_v2.py restore-pre-cutover \
  --manifest ... --confirm RESTORE-PRE-CUTOVER
```

v2 write受入後はpre-v2 backupへ戻さない。v2 DB、evidence、key、auditを保全し、schema compatibilityを確認済みのprevious imageまたはforward fixを使う。結果不明のsettlementやrefundをDB rollbackで成功・失敗へ推測しない。

## 初期化

通常のrestartやcontainer recreationでは永続dataを削除しない。

デモを初期化するときは、containerを停止し、`PAYMENT_DATA_DIR`、`PAYMENT_EVIDENCE_DIR`、`AP2_KEY_DIR`の解決済み絶対path、対象workflow件数、退避対象の証跡を確認する。このガイドでは誤操作を招くrecursive delete commandを提示しない。

必要な証跡を退避した後、新しい空の専用directoryを作り、三つの環境変数へ指定して起動する。既存directoryを再利用する場合は、個別ファイルを手作業で混在させない。

## リリース候補の作成と配布

build、push、deployは別の段階として実行する。

```bash
./deploy/build-payment-demo-candidate.sh
jq '{status,platform,localImageId,source,registry,artifacts,embedded}' \
  artifacts/cloud-run-candidate.json
./deploy/push-payment-demo-candidate.sh
./deploy/deploy-payment-demo-cloudrun.sh
```

### ビルド

build scriptはGitから見えるclean contextで`linux/amd64` imageを一度作り、source mountなしでregression、実Browser、marker suite、conformance validatorを実行する。registryやCloud Runは変更しない。結果を`cloud-run-candidate.json`へ固定する。

### 配布

push scriptはlocal image ID、platform、source digest、file mode、artifact digest、組込みsuiteがcandidateと一致する場合だけ、固定Artifact Registry repositoryへ同じimageをpublishする。Cloud Runは変更しない。

### デプロイ

deploy scriptはbuildもpushも行わず、candidateが持つ固定repositoryのimmutable `@sha256:` referenceだけを受け付ける。現在のscriptはNEW-onlyで、同名serviceが存在する場合は拒否する。override flagはない。

deploy後はready revisionのfull immutable image URIがcandidateと一致することを確認する。不一致は失敗として扱う。

### ソースの来歴

candidateはbuild時のbase commit、release sourceのpath／mode／size／bytes digest、file count、algorithmを記録する。検証時はbase commitの存在とcurrent HEADのancestor関係を確認し、source bytes、mode、count、algorithmは完全一致させる。文書や証跡だけを追加する後続commitは許容できるが、release sourceの変更はcandidate再作成を必要とする。

## Cloud Run一時デモ

Cloud Runでは次を必須とする。

- `EPHEMERAL_CLOUD_RUN_DEMO=true`
- `APP_ENV=ephemeral-demo`
- `DEV_MODE=false`
- Firebase Authentication
- min／max instanceとconcurrencyをdemo用の固定値に制限
- stateとkeyが失われ得る警告の表示

単一instanceでもephemeral filesystemは耐久性を提供しない。durable Cloud Run paid、production identity、KMS、公式A2A x402、実資産、on-chain settlementを主張しない。

現在のURL、revision、traffic、immutable image、remote browser結果は[`artifacts/cloud-run-deployment.json`](../../artifacts/cloud-run-deployment.json)を参照する。candidateの検証結果は[`artifacts/cloud-run-candidate.json`](../../artifacts/cloud-run-candidate.json)を参照する。Markdownへ現在値を複製しない。

## 運用上の禁止事項

- `DEV_MODE=true`を非local環境で使わない。
- legacy／internal routeをnginxから公開しない。
- 公式A2A x402 profileをsimulationへfallbackさせない。
- 稼働中のSQLiteを複数processやhostから直接編集しない。
- settlement結果が不明な状態で新しいchargeを作らない。
- 証跡や鍵を保全せずDBだけをrollbackしない。
- 固定candidateを再buildして同じものとしてpushしない。
- ephemeral Cloud Run demoをproductionまたは耐久決済として扱わない。
