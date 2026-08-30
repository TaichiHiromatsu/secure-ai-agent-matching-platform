# AP2の役割と実装

- 対象読者: 決済認可、署名、証跡の設計・実装・レビュー担当者
- 前提: [エージェント間決済の概要](README.md)
- 次に読む文書: [アーキテクチャ](ARCHITECTURE.md)、[検証ガイド](VERIFICATION.md)

## AP2が解決すること

AP2は、エージェントが関わる取引について「誰の意思で、何を、誰から、いくらで購入するのか」を検証可能にする認可・証跡の層である。この実装では、利用者が画面を確認して承認するHuman Presentフローを使う。

AP2だけでA2A通信や決済レールが決まるわけではない。本プロジェクトでは、MerchantとのTask通信をA2A、支払条件の交換をproject-localなA2A x402 fixture、残高処理をローカルsimulationが担当する。

## 固定している仕様

規範基準は[`secure_mediation_agent/spec_manifest.json`](../../secure_mediation_agent/spec_manifest.json)に固定している。

- AP2 repository: `google-agentic-commerce/AP2`
- commit: `e1ea56db72a6385bce3e5c1112b3a56ce60acb43`
- specification: `docs/ap2/specification.md`
- specification SHA-256: `32c3be5011f481d2760e56e7b9935b0842c3da0d5f7d7b8a68402a599f1e6aa3`

参照先:

- [AP2 v0.2 specification](https://github.com/google-agentic-commerce/AP2/blob/e1ea56db72a6385bce3e5c1112b3a56ce60acb43/docs/ap2/specification.md)
- [AP2 flows](https://github.com/google-agentic-commerce/AP2/blob/e1ea56db72a6385bce3e5c1112b3a56ce60acb43/docs/ap2/flows.md)
- [Checkout Mandate](https://github.com/google-agentic-commerce/AP2/blob/e1ea56db72a6385bce3e5c1112b3a56ce60acb43/docs/ap2/checkout_mandate.md)
- [Payment Mandate](https://github.com/google-agentic-commerce/AP2/blob/e1ea56db72a6385bce3e5c1112b3a56ce60acb43/docs/ap2/payment_mandate.md)

`vct`の`mandate.checkout.1`や`mandate.payment.1`はMandate型のversionであり、AP2仕様versionのv0.2とは別の軸である。

## Human Presentの役割

| 論理ロール | 責務 | この実装での配置 |
| --- | --- | --- |
| 利用者 | 計画と決済内容を確認し、それぞれに同意または拒否する | ADK WebまたはCLI |
| Shopping Agent | 利用者の依頼、計画、Merchant Task、AP2証跡を相関する | `secure_mediation_agent`のworkflow |
| Merchant | Checkoutを提示し、仲介の署名付き保証、capability、Task相関、安全なAP2 digest要約を照合して業務を完了する | loopback HTTPの別プロセス |
| Trusted Surface | 決済内容を利用者へ表示し、同意後にclosed Mandateを発行する | workflow内の決定論的コンポーネント |
| Credential Provider | Payment Mandateを検証し、取引限定credentialを発行する | workflow内の決定論的コンポーネント |
| Merchant Payment Processor | credential、proof、支払条件の結び付きを検証する | workflow内の決定論的コンポーネント |

ロールごとにissuer、key ID、署名鍵、検証関数、監査対象を分けるが、配置は完全な物理分離ではない。プロセスとrouteの実態は[アーキテクチャ](ARCHITECTURE.md#論理ロールと物理配置)が正本である。

## 証跡チェーン

正常系では、次のartifactを同じworkflow、plan、Merchant Task、Checkout、settlement attemptへ結び付ける。

```text
署名済み計画認可（project-local）
  └─ Merchant Checkout JWT（project-local）
      ├─ AP2 closed Checkout Mandate
      └─ AP2 closed Payment Mandate
          └─ payment credential（project-local）
              └─ payment proof（simulationではproject-local）
                  ├─ AP2 Payment Receipt
                  └─ AP2 Checkout Receipt
```

AP2のartifactとproject-local artifactを混同しない。

| Artifact | 分類 | 主な意味 |
| --- | --- | --- |
| 計画認可とdownstream capability | project-local | 計画承認を、呼出し先・操作・workflowへ限定する |
| Merchant Checkout JWT | project-local | Merchantが提示した商品、数量、金額、通貨、期限 |
| Checkout Mandate | AP2 | 利用者が確認したCheckoutへのコミットメント |
| Payment Mandate | AP2 | 金額、通貨、payee、支払手段をCheckoutへ結び付けた認可 |
| payment credential | project-local | Credential Providerが検証後に発行する取引限定credential |
| Checkout／Payment Receipt | AP2 | Merchant／MPPが署名する処理結果の証跡 |
| settlement receipt | project-local fixture | ローカルsimulationの試行結果。AP2 Receiptではない |

同じJSON fieldを持つだけではAP2 Mandateとは扱わない。署名、issuer、audience、nonce、期間、presentation、信頼鍵、Checkoutとの参照関係をロールごとに検証する。

## Mandateの生成と検証

### Checkout Mandate

Trusted SurfaceはMerchantが署名したCheckoutを表示対象として受け取り、Checkoutのexact bytesまたはdigestをclosed Checkout Mandateへ結び付ける。raw Mandateは仲介の信頼境界内で検証し、Merchantへ送らない。Merchantは支払提出時に、署名付きsimulation保証に含まれる安全なMandate digest要約と、保存済みCheckout／Task相関を照合する。

主な検証対象:

- Shopping Agent、Merchant、利用者の識別子
- Checkoutのissuer、audience、nonce、有効期間
- 商品、数量、金額、通貨
- Checkoutのexact hash
- 同意時に表示した内容のdigest

### Payment Mandate

Payment Mandateは、支払額、通貨、payee、支払手段、transaction IDをCheckoutへ結び付ける。Credential ProviderはMandate単体ではなく、元のCheckout、Merchant、accepted payment requirementsと合わせて検証する。

このdemoのpayment instrumentとcredentialはproject-localであり、公式wallet credentialやon-chain signatureではない。`secure-payment-credential+jwt`をAP2のcanonical credential wire formatとして主張しない。

## 決済承認との結び付き

計画承認後、MerchantからCheckoutと支払条件が返ると、workflowは利用者向けの決済表示を作成する。二回目の完全一致`承認`は、その表示内容のdigest、workflow、plan、Task、Checkout、有効期間に結び付けて保存する。

Trusted SurfaceがMandateを発行できるのは、この決済承認が現在の`payment_approval_required`状態と完全に一致するときだけである。価格、payee、通貨、Task、Checkout、期限のいずれかが変われば、古い承認を再利用しない。

このデモのworked exampleでは、固定シナリオ「デモ東京ベイホテル、2026年9月12日〜14日、2名」と、宿泊代を含まない予約手配サービス料`12.50 USD`、payee `demo-merchant`、versioned terms（simulation限定、実予約・実在庫hold・実課金・実送金・法的保証はすべてfalse）をCheckoutと支払表示へ含める。シナリオのcanonical digestはpayment requirementsとCheckoutの双方へ結び付け、利用者の承認はその両digestを参照する。Mandateはこの表示への認可証跡であり、部屋の確保、実予約、実hold、実送金を証明しない。

## Credential ProviderとMPP

Credential Providerは次を決定論的に検証する。

- Checkout MandateとPayment Mandateの署名および有効期間
- Checkout hash、金額、通貨、payeeの一致
- accepted payment requirementsとの一致
- workflow、plan、Task、approval、nonceの相関
- simulation proofがsimulationとして識別されていること

検証後に発行するcredentialは、対象取引、使用先、使用操作、期限、proof digestへ限定する。

Merchant Payment Processorは、credentialとproofを再検証し、支払条件、Mandate、settlement attemptが同じ取引を指すことを確認する。ロールが同じdeployable内にあることを理由に、この検証を省略しない。

Human approval、AP2 Mandate、pre-payment authorization envelopeは別artifactである。いずれも実決済レールのholdではない。現在のsimulationにreal rail holdは実装されていない。Mandate、envelope、保証、台帳効果をLLM／orchestratorや外部Merchantに生成させない。

## Receipt

AP2 ReceiptとA2A x402の結果履歴は役割が異なる。

- Checkout Receipt: Merchantが、Checkoutに対応する業務結果を署名する。
- Payment Receipt: MPPが、Payment Mandateに対応する決済処理結果を署名する。
- A2A x402のreceipt history: Task上で各settlement attemptの結果を順序付きで運ぶ。AP2 Receiptそのものではない。

成功時は、Payment Receipt、業務のcommit、Checkout Receipt、最終A2A Taskの順で証跡を確定する。業務commitが失敗した場合は成功を返さず、元のAP2 Receiptを変更せずにproject-localなrefund／reconciliation recordを追加する。

拒否や検証失敗では、後続のsettlementとfulfillmentを開始しない。roleに応じたError Receiptを作る場合も、成功Receiptへ読み替えない。

## 正規化と保存

署名対象と参照対象は、決められたcanonical bytesとSHA-256 digestで固定する。表示用文字列、LLM出力、mutable statusを署名の正本にしない。

証跡DBには、artifact本体のimmutable bytes、digest、issuer、kind、workflow／plan／Task／attemptの参照、検証時に使う公開JWK snapshotを保存する。秘密鍵、raw credential、raw proofはUI、LLM prompt、一般ログへ出さない。

完了後は[`secure_mediation_agent/ap2/evidence_verifier.py`](../../secure_mediation_agent/ap2/evidence_verifier.py)で、保存済みartifactだけから署名と参照グラフをoffline検証できる。

## 実装対応

| 責務 | 実装 |
| --- | --- |
| Trusted Surface | `secure_mediation_agent/ap2/trusted_surface.py` |
| Credential Provider | `secure_mediation_agent/ap2/credential_provider.py` |
| MPP検証 | `secure_mediation_agent/ap2/mpp.py` |
| Receipt生成・検証 | `secure_mediation_agent/ap2/receipts.py` |
| 署名鍵とロール | `secure_mediation_agent/ap2/keys.py` |
| 共通検証 | `secure_mediation_agent/ap2/verification.py` |
| offline証跡検証 | `secure_mediation_agent/ap2/evidence_verifier.py` |
| workflow上の発行順序 | `secure_mediation_agent/workflow/controller.py` |

## 主張できる範囲

この実装は、固定したAP2 v0.2資料を基準にしたHuman Present demoである。closed Checkout／Payment Mandate、ロール別の署名検証、Checkout／Payment Receipt、offline evidence chainを実装している。

一方、次は主張しない。

- AP2全体への正式なconformance
- Human Not Present、自律購入、open Mandate
- production-gradeなTrusted Surface、Credential Provider、MPPの物理分離
- production identity、KMS／HSM、法令・業界標準への適合
- AP2がA2A transportやsettlement protocolを定義しているという主張

現在の受入状態と再検証方法は[検証ガイド](VERIFICATION.md)を参照する。
