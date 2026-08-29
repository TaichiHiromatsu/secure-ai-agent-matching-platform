# 従来の仲介エージェントへの決済統合：要件定義レビュー

- レビュー日: 2026-08-16（Asia/Tokyo）
- レビュー対象: `docs/payments/MEDIATOR_PAYMENT_INTEGRATION_REQUIREMENTS.md`
- 規範上の正本: `docs/payments/MEDIATOR_PAYMENT_INTEGRATION_HANDOFF.md`
- 事実確認の基準線: `docs/payments/MEDIATOR_PAYMENT_INTEGRATION_CURRENT_STATE.md`
- レビュー種別: 独立要件定義レビュー（設計・実装・試験実施・デプロイは対象外）

## 1. 総合判定

**判定: 不合格（設計着手前にBLOCKERとHIGHの修正が必要）**

| 重要度 | 件数 |
| --- | ---: |
| BLOCKER | 1 |
| HIGH | 2 |
| MEDIUM | 3 |
| LOW | 1 |
| 重大指摘（BLOCKER + HIGH） | 3 |

要件書は正本の中心経路を広くかつ具体的に取り込んでいるが、認可入力のroutingが一意に定まらない点、支払提出に必要なsigned capability等の付与が規範要件から落ちている点、全規範要件をrelease gateへ閉じる仕組みが不足している点により、現状のまま設計入力として承認できない。

## 2. 重要度と合否の基準

- `BLOCKER`: 正本との矛盾または複数の安全上異なる解釈があり、解消前に設計へ進めない。
- `HIGH`: 中心経路、認可、支払、security、release判定の必須条件が欠落または弱体化している。
- `MEDIUM`: 実装・試験の一意な判定を妨げる曖昧さ、または正本の補助的な必須事項の欠落がある。
- `LOW`: 中心機能の安全性は直ちに損なわないが、正本が要求する証跡または運用品質が不足している。
- 各指摘の `合否` は、現行要件文のまま正本適合を宣言できるかを示す。

## 3. 指摘事項

### RQR-BLOCKER-001 承認入力のroutingが正本の優先順位と矛盾し、一意に決まらない

- 重要度: `BLOCKER`
- 対象要件ID: `FR-007`、`OQ-010`、補助的に `SEC-002`、`TEST-003`、`AC-006`
- 合否: **不合格**
- 根拠:
  - 正本9.1は、同一sessionに決済承認待ちが一件ある場合を第一優先、次に計画承認待ち、それ以外を通常依頼とする順序を明示している。
  - `FR-007`は「決済承認待ちが一件ならその対象」「計画承認待ちが一件ならその対象」「複数の保留対象では承認しない」を同時に記載している。一件の決済承認待ちと一件の計画承認待ちが併存すると、最初の二条件と「複数」の条件が同時に成立し、決済へrouteするのか拒否するのか決まらない。
  - `OQ-010`は複数保留時の選択方法を未決事項へ戻しており、正本が固定した認可routeの優先順位自体まで設計判断で変更できるように読める。
  - 承認対象の取り違えは第二承認を別stepへ適用し得るため、UI上の曖昧さではなく認可境界のblockerである。
- 具体的修正案:
  1. `FR-007`に、同一の認証済みsubject、tenant、ADK session、mediation sessionへ束縛した未期限切れpending recordだけを候補とする前提を追加する。
  2. routingを次の排他的な決定表として規範化する。
     - payment pendingが1件なら、それを最優先で決済承認対象とする。
     - payment pendingが0件かつplan pendingが1件なら、計画承認対象とする。
     - 優先対象種別に2件以上ある場合は承認せず、対象の明示選択を要求する。
     - pendingが0件なら新しい通常依頼として `secure_mediator` へ渡す。
  3. 一件のpayment pendingと一件のplan pendingの併存、同種pending複数、別subject／sessionのpending混在、期限切れを `TEST-003` と `AC-006` に明示する。
  4. `OQ-010`は表示・選択UXだけを未決にし、認可routeの優先順位と主体filterは未決対象から外す。

### RQR-HIGH-001 支払提出時のsigned capabilityと必須extensionの付与要件が欠落している

- 重要度: `HIGH`
- 対象要件ID: `FR-009`、`SEC-007`、`TEST-008`、`AC-001`、`REL-009`
- 合否: **不合格**
- 根拠:
  - 正本 `FR-009` は、保存済みremote Taskへの支払提出に「signed capabilityと必要なextension header／metadataを付ける」と明記する。
  - 要件書 `FR-009` は同一TaskとID相関だけを規定し、signed capabilityおよび交渉済みprofileの必須header／metadataを支払messageへ付けることを要求していない。
  - `SEC-007` はsigned capabilityが存在する場合のscopeを規定するが、支払提出で必ず生成・検証・送信する要件にはなっていない。
  - `TEST-008`も正本12.2が要求するcapabilityの実HTTP requestへの使用をassert対象から落としている。このままでは、capabilityなしの支払提出でも `FR-009`、`TEST-008`、`AC-001`を形式上PASSにできる。
- 具体的修正案:
  1. `FR-009`へ「支払提出messageは、plan、step、canonical Agent、操作、remote Task、期限へ限定した検証済みsigned capabilityと、選択済みprofileが必須とするextension header／metadataを含まなければならない」を追加する。
  2. Merchant側も、capabilityの署名、issuer、audience、scope、expiry、task／context bindingとprofile metadataを検証し、欠落・不一致を副作用なしで拒否する要件を追加する。
  3. `TEST-008`、`AC-001`、`REL-009`へ、wire上のcapability／extensionの存在とscope一致、欠落・改ざん時の拒否を機械assertする証跡を追加する。

### RQR-HIGH-002 release closureが全規範要件を包含していない

- 重要度: `HIGH`
- 対象要件ID: `REL-001`、`REL-002`、`REL-012`、19章、21章
- 合否: **不合格**
- 根拠:
  - 1章は、各要件の充足を試験と証跡で判定すると宣言している。
  - 一方、`REL-001`が統合完了の前提として明示するのは `FR-001`〜`FR-015` と `AC-001`〜`AC-013`だけである。
  - `REL-012`は `REL-001`〜`REL-011`だけをrelease closureに使い、21章も製品完了をACとRELの検証だけで表現する。`NFR`、`SEC`、`DATA`、`STATE`、`UI`、`HTTP`、`OPS`、`TEST`、`PRC`および期限到来済み`OQ`が、個別に未達でもrelease判定から漏れ得る。
  - 19章は正本FR／ACから本書への逆引きであり、本書の123個の要件IDそれぞれから試験、AC、証跡、release gateへのforward traceabilityではない。
  - 例えば `OPS-009` のmodel認証・quota・timeout検証や、`SEC-012` のoffline verificationは、現行RELだけを確認する運用では個別の未達が見落とされ得る。
- 具体的修正案:
  1. `REL-001`を、期限到来済みOQを含む本書の全規範要件がPASSまたは明示的に適用外と承認されていることを要求する文へ変更する。範囲外は「適用外」ではなく、18章と `CLAIM-003` の制約として維持する。
  2. `REL-012`と21章を同じclosure規則へ合わせる。
  3. 全要件IDについて、`要件ID -> 試験ID -> AC -> 必要証跡 -> release判定` のforward traceability表を追加し、対応のない規範IDを許さない。
  4. `REL-002`の「対象suite」を、traceability表で必要とされた全suiteと全必須caseの集合として定義する。

### RQR-MEDIUM-001 AP2相関の必須fieldと検証証跡が正本より抽象化されている

- 重要度: `MEDIUM`
- 対象要件ID: `FR-008`、`SEC-012`、`DATA-002`、`DATA-005`、`DATA-008`、`TEST-002`、`REL-009`
- 合否: **不合格**
- 根拠:
  - 正本 `FR-008` は、Intent／Checkout／Payment Mandate／Receiptから、計画承認ID、決済承認ID、nonce、発行時刻を含む列挙fieldを少なくとも相関可能にするよう要求する。
  - `FR-007`は二つの承認recordにnonceと時刻を保持するが、`FR-008`と`DATA-005`はAP2 evidenceから各承認のnonce／発行時刻を相関・検証できることを明記しない。
  - `SEC-012`のnonceと有効期限はAP2 object一般の検証としても読め、二つの承認recordとのbindingを保証しない。`TEST-002`と`REL-009`もfield単位のassertを要求していない。
- 具体的修正案:
  1. `FR-008`または専用DATA要件へ、正本 `FR-008` の全列挙fieldをそのまま最低必須相関fieldとして記載する。
  2. 計画承認nonce／issued-at、決済承認nonce／issued-at、各AP2 object自身のnonce／issued-at／expiryを区別し、直接埋込みまたは改ざん検知可能なimmutable referenceのどちらで結ぶかを `OQ-008` の決定対象にする。
  3. `TEST-002`と `REL-009`で、offline verifierが外部DBの暗黙知なしに必要な署名連鎖と各fieldの一致／不一致を判定できることをassertする。

### RQR-MEDIUM-002 従来のsecurity callback維持とanomaly gate境界が受入可能な単位に分離されていない

- 重要度: `MEDIUM`
- 対象要件ID: `FR-010`、`TEST-006`、`AC-001`、`AC-002`、`REL-008`、`OQ-005`
- 合否: **不合格**
- 根拠:
  - 正本 `FR-010` は「従来のA2A前後のsecurity callback」と `anomaly_detector` の双方を維持する。要件書 `FR-010`は `anomaly_detector` と「必要な決定論的検証」だけを記載し、既存security callbackを維持するか同等性を検証する条件がない。
  - `FR-010`は「外部A2A応答受領後」と「支払要求受領後」を別々に列挙する。支払要求が同じ初回A2A応答で返る正常系に、同一detectorを一回実行するのか、異なる目的で二回実行するのかが定まらない。
  - `TEST-006`、`AC-001`、`REL-008`の「各anomaly gate」にはstableなgate ID、入力、期待副作用件数がなく、実装とreviewerで必要call数がずれる。
- 具体的修正案:
  1. 従来security callbackと新しいanomaly／決定論的gateを別要件として記載し、置換を許すなら、正本の変更承認と同等以上であることの比較証跡を必須にする。
  2. 例として `PRE_A2A_START`、`POST_A2A_RESPONSE`、`POST_PAYMENT_REQUIREMENT`、`PRE_PAYMENT_SUBMIT`、`POST_PAYMENT_RESULT` のstable gate IDを定義し、paid応答時に後二者のどちらか一方か双方かを明示する。
  3. 各gateについて入力schema／digest、呼出回数、`PASS`時に許される次副作用、`BLOCK`／`REVIEW`／timeout時の禁止副作用を `TEST-006`、`AC-001`／`002`、`REL-008`へ対応付ける。

### RQR-MEDIUM-003 restart試験の前提と期待結果が状態ごとに成立していない

- 重要度: `MEDIUM`
- 対象要件ID: `TEST-013`、補助的に `FR-013`、`OPS-003`〜`OPS-005`、`AC-011`
- 合否: **不合格**
- 根拠:
  - `TEST-013`は、計画承認待ち、決済承認待ち、支払後、outbox lease中のすべてについて「同じTaskを一度だけ継続」と一括して要求する。
  - 計画承認待ちでは `FR-004`によりMerchant Taskがまだ存在してはならず、「同じTaskを継続」は成立しない。
  - 正本は同一instanceの子process再起動時のreconciliationと、instance置換時に状態消失を許容する仕様を明確に分けているため、試験期待値も各checkpointで分ける必要がある。
- 具体的修正案:
  1. `TEST-013`をcheckpoint別のcaseへ分割する。
     - 計画承認待ち: 再起動前後のMerchant Taskは0件で、承認待ちを復元し、有効承認後のTask開始が1件だけ。
     - 決済承認待ち: 既存task／contextと支払条件を復元し、承認前の支払提出は0件。
     - 支払後／outbox lease中: 同じtask／contextと冪等性キーで照合し、Task開始とsettlementを増やさない。
     - instance置換／revision更新: 状態消失を許容し、古いcontinuationを成功扱いせず所定の再実行案内を返す。
  2. 各caseに初期record、再起動対象process、期待record数、期待state、期待外部call数を定義する。

### RQR-LOW-001 完了報告に必要なPR URLが証跡要件から落ちている

- 重要度: `LOW`
- 対象要件ID: `PRC-007`、`REL-011`
- 合否: **不合格**
- 根拠:
  - 正本21章は、完了報告へPR URLとdraftではないことを必須証跡として含める。
  - `PRC-007`と`REL-011`は通常PR、説明、テスト証跡、既知課題を要求するが、完了証跡にPR URLまたはPR番号を保存することを明示しない。
- 具体的修正案:
  - `REL-011`へ、PR URL／番号、base／head、head commit SHA、draft=false、対象revision・candidateとの対応、既知課題を完了報告へ記録する要件を追加する。

## 4. 主要観点ごとの確認結果

| 観点 | 判定 | 根拠・備考 |
| --- | --- | --- |
| 正本と現状事実の分離 | 合格 | 1章と4章で正本を規範、CURRENT_STATEを非規範の基準線として明示している。現行証跡の流用も禁止している |
| 要件IDの一意性 | 合格 | `FR`から`OQ`まで123個の要件見出しを確認し、重複IDはなかった |
| 二段階の完全一致承認 | 条件付き不合格 | 完全一致、別digest／nonce／IDは明確だが、RQR-BLOCKER-001のrouting矛盾が残る |
| runtimeのpayment-required判定 | 合格 | `FR-005`、`TEST-001`、`AC-012`で固定flag・自由文判定を禁止し、Task stateと構造化extensionを要求している |
| 同一remote Task／同一step | 条件付き不合格 | ID相関とTask開始1件は明確だが、RQR-HIGH-001の送信認可情報が欠落している |
| AP2相関 | 条件付き不合格 | 主体、plan、step、Agent、Task、quote、支払条件、二承認の結合はあるが、RQR-MEDIUM-001のfield単位検証が不足する |
| x402 fail-closed | 合格 | `SEC-013`、`SEC-014`、`AC-012`でprofile未対応・破損、silent fallback禁止、simulation表示を一意に規定している |
| anomaly各gate／final | 条件付き不合格 | fail-closedとfinal成功禁止は明確だが、RQR-MEDIUM-002のcallback維持とgate単位が不足する |
| 無料経路 | 合格 | `FR-012`、`TEST-007`、`AC-002`で決済workflowを作らず通常仲介とfinal validationを通す |
| subject／session分離 | 合格 | `SEC-001`〜`003`、`DATA-001`、`TEST-005`、`AC-006`で終端間bindingとnegative caseを要求している |
| ephemeral仕様 | 条件付き不合格 | `OPS-003`〜`005`と`AC-011`は正本どおりであり、CURRENT_STATE B-007の耐久化提案を規範へ混入していない。ただしrestart testはRQR-MEDIUM-003の修正が必要 |
| 公開HTTP境界 | 合格 | exact／prefix、認証状態に依存しない404、identity header、loopback、allowlistを `HTTP-001`〜`006`で規定している |
| 固定Cloud Run更新／他service不変更 | 合格 | `OPS-001`、`OPS-007`、`OPS-008`、`REL-005`、`REL-010`で対象固定、immutable digest、traffic、他service非変更を要求している |
| PR／ブラウザ証跡 | 条件付き不合格 | browser evidenceとlocal／Cloud Run E2Eは明確。PR URLのみRQR-LOW-001の不足がある |
| 一次資料再確認 | 合格 | `REL-006`と`OQ-009`で設計前・release前の一次資料、version固定、互換性差分、再評価を要求している |
| 未決事項の設計入力適合性 | 条件付き不合格 | `OQ-001`〜`009`は不変制約と期限を持つ妥当な設計入力。`OQ-010`だけは正本で固定済みの認可優先順位を再度開いており、RQR-BLOCKER-001の修正が必要 |

## 5. 修正後の再レビュー条件

次を満たした差分に対して再レビューする。

1. RQR-BLOCKER-001とRQR-HIGH-001〜002が解消されている。
2. MEDIUM指摘について、要件本文、試験要件、AC、RELの相互整合が取れている。
3. 全規範IDのforward traceabilityに未対応行がない。
4. 正本の必須条件を変更する場合は、要件書側だけで解釈を変えず、正本の明示的な変更承認がある。
5. protected文書の現状記述や実装を、この要件レビューのために変更していない。

## 6. レビュー範囲の記録

- `MEDIATOR_PAYMENT_INTEGRATION_HANDOFF.md`、`MEDIATOR_PAYMENT_INTEGRATION_CURRENT_STATE.md`、`MEDIATOR_PAYMENT_INTEGRATION_REQUIREMENTS.md`を全文確認した。
- 本レビューでは設計、実装、テスト実行、ブラウザ操作、Cloud Run操作、PR更新を行っていない。
- 本レビューで新規作成したファイルは本書だけであり、コード、設定、`README.md`、HANDOFF、CURRENT_STATE、REQUIREMENTSは変更していない。

## 7. 再レビュー（2026-08-16）

### 7.1 再レビュー判定

**最新要件書に対する判定: 不合格（HIGH 1件が未解消）**

- BLOCKER: 0件
- HIGH: 1件
- MEDIUM: 0件
- LOW: 0件
- 重大指摘（BLOCKER + HIGH）: 1件
- 前回7指摘の結果: 解消6件、未解消1件
- この節の判定は、1章の初回レビュー判定を最新要件書について更新するものである。

中心経路の機能・security要件は検証可能な水準へ修正された。しかし、release closureが依存するforward traceability matrixが保存済み要件書に存在しないため、全規範IDの欠落なし検証とrelease判定を実行できない。前回 `RQR-HIGH-002` は未解消であり、合格条件を満たさない。

### 7.2 前回指摘の解消確認

| 前回指摘 | 状態 | 再レビュー根拠 |
| --- | --- | --- |
| `RQR-BLOCKER-001` 承認routing | **解消** | `FR-007`に主体／session／期限filterと排他的な優先順位表が追加された。payment 1件＋plan併存、同種複数、pending 0件を `TEST-003`、`AC-006`が機械assertする。`OQ-010`はUX上の選択方法だけに限定されている |
| `RQR-HIGH-001` signed capability／extension | **解消** | `FR-009`、`SEC-007`、`SEC-015`が支払wireへの付与とMerchantの副作用前検証を必須化し、`TEST-008`、`AC-001`、`REL-009`が存在・scope・改ざん・副作用0件を検証する |
| `RQR-HIGH-002` release closure | **未解消** | `REL-001`、`REL-012`、`REL-013`、`TEST-015`は全規範ID ledgerを要求するよう改善されたが、`REL-002`と`REL-013`が参照する「19.3 forward traceability matrix」が存在しない。19章は19.1と19.2だけで終わる。また19.2は追加済み `SEC-015`／`SEC-016`、`TEST-015`、`REL-013`を範囲へ含めず、21章も製品完了を `REL-001`〜`REL-012`だけで表現している。従って127個の規範IDを試験・AC・証跡へ一意に閉じる基準が未完成である |
| `RQR-MEDIUM-001` AP2必須field | **解消** | `FR-008`、`DATA-002`、`DATA-005`が正本の相関field、二承認のnonce／issued-at、各AP2 objectのfieldを明示し、`TEST-002`、`AC-001`、`REL-009`がoffline一致／不一致を要求する |
| `RQR-MEDIUM-002` callback／gate単位 | **解消** | `FR-010`に5個のstable gate ID、順序、入力、許可副作用、呼出回数が追加され、`SEC-016`で従来security callbackを別防御層として維持する。`TEST-006`、`AC-001`／`002`、`REL-008`も有料・無料の回数を固定している |
| `RQR-MEDIUM-003` restart | **解消** | `TEST-013`と`AC-011`が計画承認待ち、決済承認待ち、支払後／outbox lease中、instance置換を別caseへ分け、Taskが存在しないcheckpointを正しく扱っている |
| `RQR-LOW-001` PR URL | **解消** | `PRC-007`と`REL-011`がPR URL／番号、base／head、head SHA、draft=false、candidate／revision対応、既知課題を完了証跡として要求する |

### 7.3 ID・正本・新規矛盾の確認

- 要件見出しは127個で、ID重複はなかった。
- 各prefix内の番号は連続しており、見出しID自体の欠番はなかった。
- runtime `payment-required`、二段階完全一致承認、同一remote Task、AP2相関、x402 fail-closed、全anomaly gateとfinal、無料経路、主体／session分離、ephemeral境界、公開境界、固定Cloud Run更新、他service非変更、ブラウザ／PR証跡、一次資料再確認について、新たな正本からの弱体化は確認しなかった。
- 新たなBLOCKERは確認しなかった。
- 新たな独立指摘は追加しない。存在しない19.3への参照、19.2の追加ID範囲漏れ、21章の `REL-013`漏れは、いずれも前回 `RQR-HIGH-002` のrelease closure未完成として扱う。

### 7.4 合格に必要な最小修正

1. 実在する19.3節として、127個の全規範IDを `試験ID／判定規則、AC、必要証跡、release判定` へ一回ずつ結ぶforward traceability matrixを追加する。
2. 19.2の範囲を `SEC-001`〜`SEC-016`、`TEST-001`〜`TEST-015`、`REL-001`〜`REL-013`へ更新する。
3. 21章の製品完了判定を `REL-001`〜`REL-013`と全規範ID ledgerのclosureへ一致させる。
4. `TEST-015`で、上記matrixと要件見出しの集合一致、重複0、欠落0、未知ID 0を自動検査できる記載にする。

上記は新しい設計論ではなく、すでに `REL-001`、`REL-002`、`REL-012`、`REL-013`、`TEST-015`が要求しているrelease closureを、文書内で実在・一貫させるための修正である。

## 8. 最終再レビュー（2026-08-16）

### 8.1 最終判定

**最新要件書に対する最終判定: 合格**

- BLOCKER: 0件
- HIGH: 0件
- MEDIUM: 0件
- LOW: 0件
- 重大指摘（BLOCKER + HIGH）: 0件
- 前回7指摘の最終結果: **全7件解消**
- この節の判定は、1章および7章の過去時点の判定を最新要件書について更新する最終判定である。

### 8.2 `RQR-HIGH-002` の解消確認

- 19.2は追加済み要件を含む `SEC-001`〜`SEC-016`、`TEST-001`〜`TEST-015`、`REL-001`〜`REL-013`へ更新されている。
- 19.3に全規範IDのforward traceability matrixが実在し、各行が設計・実装責務、試験ID／判定規則、AC／判定規則、必要証跡、release判定へ結合されている。
- 規範要件見出しは127件、19.3 matrixは127行、matrix内の一意IDは127件であった。見出し集合とmatrix集合の差分は0件、matrix重複は0件であった。
- `TEST-015`は見出し、matrix、candidate ledgerの集合完全一致、重複、欠落、未知ID、証跡なしPASS、期限到来済みOQを自動closure対象としている。
- `REL-002`は19.3が要求する全suiteと必須caseをcandidate imageへ結合し、未実行要件の代替を禁止している。
- `REL-013`は全規範IDを一回ずつ持つ127行ledgerと、試験、AC、証跡、candidate digest、判定者、時刻、4値statusを要求している。
- 21章は19.3の全127 ID、`AC-001`〜`AC-013`、`REL-001`〜`REL-013`、期限到来済みOQを `TEST-015` で閉じる完了判定へ更新されている。

以上により、前回未解消だった `RQR-HIGH-002` は解消した。指定されたclosure範囲にID欠落、重複、正本からの弱体化、新たな矛盾は確認しなかったため、要件定義を合格と判定する。
