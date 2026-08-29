# 決済機能の要件定義

- 対象読者: プロダクト責任者、実装者、テスト担当者、設計レビュアー
- 前提: [エージェント間決済の概要](README.md)
- 次に読む文書: [アーキテクチャ](ARCHITECTURE.md)、[検証ガイド](VERIFICATION.md)

## 文書の位置づけ

この文書は、現在の決済機能が満たすべき不変の要件と受入基準を定義する。実装方法は[アーキテクチャ](ARCHITECTURE.md)、protocol固有の設計は[AP2](AP2.md)と[A2A x402](A2A_X402.md)、現在の試験結果は機械可読な[適合レポート](../ap2_x402_conformance_report.json)を正本とする。

「しなければならない」「してはならない」は必須要件、「してよい」は必須要件を破らない範囲の選択肢を示す。要件IDは設計、テスト、証跡から参照する安定した識別子であり、現在のPASS／FAILを表さない。

## 仕様の基準

| ID | 要件 |
| --- | --- |
| BASE-001 | AP2の規範基準を`google-agentic-commerce/AP2` commit `e1ea56db72a6385bce3e5c1112b3a56ce60acb43`のv0.2資料とcanonical schemaに固定しなければならない。 |
| BASE-002 | A2A x402の規範基準を`google-agentic-commerce/a2a-x402` commit `125db5526a965d2325459d1a9df2e274a7e42396`の`spec/v0.1/spec.md`に固定しなければならない。 |
| BASE-003 | 公式profileのextension URIは完全一致の`https://github.com/google-a2a/a2a-x402/v0.1`とし、simulation profileはこのURIを宣言またはactivationしてはならない。 |
| BASE-004 | AP2、A2A x402、A2A wire、language package、HTTP x402のversionを別々に管理し、相互に読み替えてはならない。 |
| BASE-005 | 固定commitを変更する場合は仕様差分、schema差分、適合表現、移行影響をレビューしなければならない。 |

固定値とcontent hashの機械可読な正本は[`secure_mediation_agent/spec_manifest.json`](../../secure_mediation_agent/spec_manifest.json)とする。

## 対象と非対象

| ID | 要件 |
| --- | --- |
| SCOPE-001 | 必須対象を、一利用者、一tenant、一Merchant、一商品、一数量、一通貨、AP2 Human Present closed-Mandateの有料フローと、その拒否、改ざん、replay、timeout、restart、reconciliation、refund分岐とする。 |
| SCOPE-002 | ADK WebとCLIは同じworkflow serviceを使い、同じ状態、承認境界、error code、最終証跡を返さなければならない。 |
| SCOPE-003 | AP2 Human Not Present、open Mandate、budget、split tender、FX、複数Merchantの一括Checkoutは対象外とする。 |
| SCOPE-004 | production KMS／HSM、本人確認、KYC／AML、PCI／SCA、適合認証、法的な支払保証を実装済みと表示してはならない。 |
| SCOPE-005 | fee policyは`zero-fee-v1`とし、顧客加算、collection cost、commission、payout costを0に固定する。 |
| SCOPE-006 | simulationは実資産、wallet署名、facilitator verification、on-chain transactionを表してはならない。 |
| SCOPE-007 | simulation-only構成は「AP2 v0.2 Human Present demo」と「A2A x402 v0.1 wire-shape test fixture (NOT CONFORMANT)」とのみ表示できる。 |
| SCOPE-008 | 公式A2A x402 profileを有効にするには、対応network／asset／wallet／payTo、公式`exact` payload、facilitator verify／settle、TLS、実transaction hash、ACC-030を満たさなければならない。 |

## 役割と信頼境界

| ID | 要件 |
| --- | --- |
| ROLE-001 | 利用者向けroot agentは`payment_user_agent`一つとし、決済の正本、秘密鍵、直接のrail操作を持たせてはならない。 |
| ROLE-002 | `secure_mediation_agent`のworkflowを状態と実行順序の唯一の正本とし、LLM reasoningを認可判断にしてはならない。 |
| ROLE-003 | Trusted Surfaceは非agenticな決定論的コンポーネントとし、表示した内容と利用者同意をMandateへ結び付けなければならない。 |
| ROLE-004 | Merchant identityをpayee、Checkout issuer、Checkout Receipt issuer、Task endpointへ一貫して結び付けなければならない。 |
| ROLE-005 | Credential ProviderとMPPは、同じdeployableへ配置する場合もissuer、key、検証関数、監査eventを論理的に分離しなければならない。 |
| ROLE-006 | ロールの同居を理由に署名、audience、nonce、期限、取引bindingの検証を省略してはならない。 |
| ROLE-007 | Merchant A2A endpointは選択profileのactivationとaudience／operation-scoped capabilityを副作用前に検証しなければならない。 |
| ROLE-008 | Agent／LLM、一般session、一般artifact storeからprivate key、raw credential、raw proofを読み取れないようにしなければならない。 |

ロールの論理分離を別service／processへの物理分離と表示してはならない。実際の配置は[アーキテクチャ](ARCHITECTURE.md#論理ロールと物理配置)へ記載する。

## 計画と二段階承認

| ID | 要件 |
| --- | --- |
| PLAN-001 | free-form textや`plan_approved` booleanを認可artifactにせず、versionedな計画snapshotをcanonical JSONとして保存しなければならない。 |
| PLAN-002 | 計画はRFC 8785でcanonicalizeし、UTF-8 bytesのSHA-256を`planDigest`として保存しなければならない。 |
| PLAN-003 | 計画に利用者、tenant、session、Agent Card digest、Merchant、skill、商品、数量、金額上限、通貨、fee policy、payment profile、期限を固定しなければならない。 |
| PLAN-004 | 計画作成後に上書きしてはならず、変更時は新しい`planId`、version、digestを作らなければならない。 |
| PLAN-005 | Agent Cardやonboarding recordの後続更新で、承認済み計画の意味を変えてはならない。 |
| PLAN-006 | amountはinteger minor unitsと通貨のdecimalsで扱い、LLMまたは浮動小数点で換算・丸めしてはならない。 |
| PLAN-007 | 計画承認をworkflow、plan digest、tenant、利用者、session、期限、single-use nonceへ結び付けなければならない。 |
| PLAN-008 | downstream capabilityは呼出し先ごとに別ID、audience、operation、workflow、Task、expiry、idempotency scopeを持たなければならない。 |
| PLAN-009 | 商品、数量、金額、通貨、Merchant、profile、期限のいずれかが変われば古い承認を失効させなければならない。 |

| ID | 要件 |
| --- | --- |
| APPROVAL-001 | transport framingを除くuser messageが単一text partでUnicode code point列として完全に`承認`と一致する場合だけ、承認候補として扱わなければならない。 |
| APPROVAL-002 | trim、Unicode normalization、部分一致、LLM intent classificationを承認判定に使ってはならない。 |
| APPROVAL-003 | `plan_approval_required`の`承認`は計画だけを承認し、Mandate、payment payload、settlementを作ってはならない。 |
| APPROVAL-004 | `payment_approval_required`の`承認`は表示済みの決済内容だけを承認し、計画承認を更新してはならない。 |
| APPROVAL-005 | 二つの承認は別ID、intent、nonce、署名対象、期限、監査eventを持たなければならない。 |
| APPROVAL-006 | 承認待ちでない状態の`承認`を拒否し、状態、nonce、外部副作用を変更してはならない。 |
| APPROVAL-007 | `はい`、`yes`、`OK`、`承認します`、前後空白付き`承認`、複数partを承認として扱ってはならない。 |

## ワークフローと外部副作用

| ID | 要件 |
| --- | --- |
| WF-001 | workflow aggregateを利用者依頼から完了までの唯一のauthoritative stateとしなければならない。 |
| WF-002 | workflowはplan、approval、Task、Checkout、Mandate、credential、attempt、fulfillment、Receiptを不変のIDとdigestで相関しなければならない。 |
| WF-003 | Merchant TaskをMerchant payment subflowの正本とし、同じ`taskId`以外への支払提出を拒否しなければならない。 |
| WF-004 | 状態遷移をversion付きcompare-and-setで行い、表外遷移と二重副作用を拒否しなければならない。 |
| WF-005 | 状態更新と外部呼出しintentをtransactional outboxで引き渡さなければならない。 |
| WF-006 | 計画承認前にMerchant Task、Checkout、payment requirement、settlement、fulfillmentを作ってはならない。 |
| WF-007 | 決済承認前にcredential、payment payload、settlement、success Receipt、fulfillment commitを作ってはならない。 |
| WF-008 | Checkoutまたはrequirementsが計画外なら決済承認を表示せず`replan_required`にしなければならない。 |
| WF-009 | settlement前に業務を不可逆にcommitしてはならず、必要な場合はprepare／commit境界を持たなければならない。 |
| WF-010 | settlement成功後のfulfillment failureは`refund_required`、結果不明は`reconciliation_required`にしなければならない。 |
| WF-011 | refundとreconciliationは元のMandate、Receipt、settlement結果を上書きせず、append-onlyなproject-local recordとして保存しなければならない。 |
| WF-012 | payment非対応の無料workflowは、決済承認を要求せず従来どおり実行できなければならない。 |

## AP2

| ID | 要件 |
| --- | --- |
| AP2-001 | Human Presentのclosed Checkout MandateとPayment Mandateを、固定したAP2 schemaとrole別keyで生成・検証しなければならない。 |
| AP2-002 | Checkout MandateをMerchantの署名済みCheckout exact hash、利用者、Merchant、Task、期限へ結び付けなければならない。 |
| AP2-003 | Payment Mandateを金額、通貨、payee、支払手段、Checkout hash、transaction IDへ結び付けなければならない。 |
| AP2-004 | Trusted Surfaceは決済承認と表示内容digestを検証した後だけMandateを発行しなければならない。 |
| AP2-005 | Credential Providerは両Mandate、Checkout、requirements、payee、amountを検証した後だけ取引限定credentialを発行しなければならない。 |
| AP2-006 | MPPはcredential、proof、requirements、workflow、Task、attemptを再検証した後だけ処理結果を作らなければならない。 |
| AP2-007 | MerchantのCheckout ReceiptとMPPのPayment Receiptは別issuer、別意味、別署名対象を持たなければならない。 |
| AP2-008 | AP2 ReceiptとA2A x402のresult historyを同一artifactとして扱ってはならない。 |
| AP2-009 | 計画認可、downstream capability、Merchant Checkout JWT、payment credential、simulation proofをAP2 canonical objectと表示してはならない。 |
| AP2-010 | issuer、key ID、audience、nonce、期間、exact hash、相互参照をofflineで再検証できなければならない。 |
| AP2-011 | AP2 artifactのimmutable bytes、digest、public JWK snapshotを証跡storeへ保存しなければならない。 |

## A2A x402とsimulation

| ID | 要件 |
| --- | --- |
| X402-001 | official profileではcanonical extension URIの宣言、activation request、Merchant echoを完全一致で検証しなければならない。 |
| X402-002 | simulation profileはproject-local URIを使用し、canonical URIとの混在や暗黙fallbackを拒否しなければならない。 |
| X402-003 | PaymentRequiredを元Taskの`input-required` statusとdotted metadataへ結び付けなければならない。 |
| X402-004 | payment submissionは元の`taskId`を持つ新しいMessageとし、requirementsとの相関を検証しなければならない。 |
| X402-005 | scheme、network、asset、payTo、amount、expiryをrequirementsとpayloadで完全照合しなければならない。 |
| X402-006 | 各settlement attemptのsuccess／failureを順序付き履歴へappendし、最終Taskに全履歴を保持しなければならない。 |
| X402-007 | 利用者拒否は元Taskへ一回の`payment-rejected`を保存し、payload、settlement、success Receiptを作ってはならない。 |
| X402-008 | simulation proofを`simulated=true`、`walletSigned=false`として識別し、`sim:`参照を実transaction hashとして扱ってはならない。 |
| X402-009 | official profileに必要なwallet、facilitator、network、asset、TLSがない場合は起動時にfail closedとし、simulationへfallbackしてはならない。 |
| X402-010 | AP2 Payment Receiptとsettlement attemptの対応を検証可能にする一方、両者のschemaとissuerを分けなければならない。 |

## 永続化、再試行、セキュリティ

| ID | 要件 |
| --- | --- |
| RES-001 | idempotency keyをtenant、actor、operation、canonical request hashへ結び付け、同じrequestは同じ結果、異なるrequestはconflictにしなければならない。 |
| RES-002 | approval nonce、capability、payment payloadを別workflow、Task、tenantへreplayできてはならない。 |
| RES-003 | outboxは一意operation ID、期限付きlease、retry、crash後のlease回収を持たなければならない。 |
| RES-004 | 非終端状態からrestartしても同じstate、exact bytes、Task、attempt、履歴を復元しなければならない。 |
| RES-005 | settlement結果が不明なときは新しいchargeを作らず、同じexternal IDを照会しなければならない。 |
| RES-006 | settlement、fulfillment、Receipt、refundのbusiness effectをそれぞれ高々一回にしなければならない。 |
| RES-007 | DBと鍵は明示した永続directoryへ置き、通常restartやcontainer recreationで削除してはならない。 |

| ID | 要件 |
| --- | --- |
| SEC-001 | 公開payment APIは認証済み`/mediation-api/`だけとし、外部identity headerを信頼してはならない。 |
| SEC-002 | internal signer、Trusted Surface、CP、MPP、Merchant control、operator routeをpublic ingressへ公開してはならない。 |
| SEC-003 | secret、private key、raw credential、raw proofをprompt、public view、通常log、error、traceへ出してはならない。 |
| SEC-004 | workflow、evidence、operator操作はtenantとownerまたはoperator権限を検証し、監査eventを残さなければならない。 |
| SEC-005 | demo keyをsource control内のproduction credentialとして扱ってはならない。 |
| SEC-006 | public viewはsimulation、公式profileの無効状態、実資産なしを一貫して表示しなければならない。 |

## 移行と運用

| ID | 要件 |
| --- | --- |
| MIG-001 | schema migrationはversion付きforward migrationとし、既存DBをin-placeで破壊せず、backupと失敗時の回復手順を持たなければならない。 |
| MIG-002 | 空DB、既存fixture、legacy非終端状態、途中失敗、再適用を試験しなければならない。 |
| MIG-003 | legacyの`plan_approved` booleanを署名済み計画承認へ昇格してはならない。 |
| MIG-004 | legacy profile、order、Task、Receiptを新profileへ暗黙変換またはresumeしてはならない。 |
| MIG-005 | 旧public payment routeとpayment-only rootをdefault discoveryから除外しなければならない。 |
| OPS-001 | local durable targetとephemeral Cloud Run demoをreadiness、表示、証跡で区別しなければならない。 |
| OPS-002 | build、push、deployを別段階とし、検証済みimmutable image以外をdeployしてはならない。 |
| OPS-003 | release source、file mode、artifact、manifest、image digestのbindingをfail closedで検証しなければならない。 |
| OPS-004 | resetは対象directoryとworkflow件数を確認し、必要な証跡を退避してから行わなければならない。 |

## UIと表示

| ID | 要件 |
| --- | --- |
| UI-001 | ADK Webで選択する決済対応rootを`payment_user_agent`一つにしなければならない。 |
| UI-002 | 計画画面は、承認対象、金額上限、Merchant、商品、期限、この時点では決済されないことを表示しなければならない。 |
| UI-003 | 決済画面は、商品、数量、Merchant、金額、通貨、fee内訳、期限、simulation、NOT CONFORMANTを表示しなければならない。 |
| UI-004 | 二つの承認対象を別画面または明確に別の状態として表示しなければならない。 |
| UI-005 | 完了画面は業務結果、AP2 evidence参照、simulation、実資産／on-chainなしを表示しなければならない。 |
| UI-006 | refreshまたはrestart後も同じworkflow viewを正本DBから復元しなければならない。 |
| UI-007 | errorは安定したcodeと安全な説明を返し、secretやtenantの存在を漏らしてはならない。 |

## 受入基準

現在の判定はここへ書かない。各IDの状態は[適合レポート](../ap2_x402_conformance_report.json)、試験との対応は[検証ガイド](VERIFICATION.md)を参照する。

| ID | 受入シナリオ |
| --- | --- |
| ACC-001 | 新規有料依頼では計画だけを表示し、Merchant Task、Checkout、settlement、fulfillmentを作らない。 |
| ACC-002 | 一回目の完全一致`承認`では計画承認だけを保存し、決済承認やMandateを作らない。 |
| ACC-003 | 非完全一致の承認語を拒否し、副作用を作らない。 |
| ACC-004 | 承認待ちでない状態の`承認`を拒否し、状態を変えない。 |
| ACC-005 | 段階外のstart／submit／verify／settle／fulfill呼出しを副作用0件で拒否する。 |
| ACC-006 | 計画承認後、選択profileとoperation-scoped capabilityを持つMerchant Taskを一つ作る。 |
| ACC-007 | activationの欠落、不一致、official／simulation URI混在を副作用前に拒否する。 |
| ACC-008 | 計画内のCheckoutと支払条件を`input-required` Taskに保存し、決済画面を表示する。 |
| ACC-009 | Merchant、商品、数量、金額、通貨、payee、profileが計画外なら`replan_required`にする。 |
| ACC-010 | 二回目の完全一致`承認`からclosed Mandateと取引限定credentialを作り、決済効果を高々一回にする。 |
| ACC-011 | 決済画面での非完全一致入力からMandate、payload、settlementを作らない。 |
| ACC-012 | payment Messageを元の`taskId`へ相関し、別Taskへの差替えを拒否する。 |
| ACC-013 | Merchant、Trusted Surface、Credential Provider、MPPが各ロールの署名とbindingを決定論的に検証する。 |
| ACC-014 | simulation成功時、結果履歴、AP2 Receipt、業務Artifactを同じattemptへ相関し、NOT CONFORMANTを表示する。 |
| ACC-015 | settlement失敗時、失敗理由と全attempt履歴を残し、成功や実transactionと表示しない。 |
| ACC-016 | AP2検証失敗時、後続settlementとfulfillmentを開始しない。 |
| ACC-017 | 保存済みartifactだけで計画認可からAP2 Receiptとsimulation結果までoffline検証できる。 |
| ACC-018 | 同じidempotency keyと同じrequestは同じ結果を返し、異なるrequestはconflictにする。 |
| ACC-019 | 使用済みnonce、capability、payloadの別workflow／Task／tenantへのreplayを拒否する。 |
| ACC-020 | 各非終端状態からrestartし、状態とexact bytesを復元して副作用を重複させない。 |
| ACC-021 | settlement timeout時、新しいchargeを作らず同じexternal IDでreconciliationする。 |
| ACC-022 | 並行承認とduplicate Messageでも、承認、settlement、fulfillment、Receiptの効果を各一回以下にする。 |
| ACC-023 | 実ADK Webで依頼、計画承認、決済承認、完了、状態復元を同じroot／sessionで確認する。 |
| ACC-024 | CLIでADK Webと同じ状態、error code、最終相関を確認する。 |
| ACC-025 | public ingressから内部routeとlegacy payment routeを呼べない。 |
| ACC-026 | migrated DBでlegacy booleanや旧profileを新しい認可に使わない。 |
| ACC-027 | payment非対応の無料workflowを決済承認なしで実行できる。 |
| ACC-028 | 固定したAP2 fixtureとA2A x402 fixtureを別suiteで検証し、profile混在を失敗させる。 |
| ACC-029 | simulationではproject-local URIとNOT CONFORMANTを表示し、canonical URIをactivationしない。 |
| ACC-030 | 条件付きの公式profile試験で、wallet、facilitator、network、asset、TLS、実transactionとAP2 Receiptの相関を確認する。未実装なら公式適合を主張しない。 |
| ACC-031 | success／failureのlog、trace、error、prompt、artifactにsecret、raw credential、raw proofを出さない。 |
| ACC-032 | clean imageとmigrated volumeの双方でreadiness、route isolation、二承認、restartを確認する。 |
| ACC-033 | 利用者拒否を元Taskへ一回記録し、payload、settlement、success Receipt、fulfillmentを作らない。 |
| ACC-034 | settlement成功後のfulfillment failureで元証跡を変更せず、冪等なrefund recordを作る。 |
| ACC-035 | settlement／refund結果不明時、同じexternal IDを照会し、証拠なしに成功または返金済みへ進めない。 |

simulation-only releaseではACC-030を条件付き`NOT RUN`としてよいが、その場合はcanonical URIをruntimeで使わず、A2A x402 compatible／conformantを主張してはならない。それ以外の必須基準と既存回帰を満たすことをrelease gateとする。
