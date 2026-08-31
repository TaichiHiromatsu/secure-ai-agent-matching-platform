# 仲介エージェント決済デモ：これ一枚で実演

## 1. 目的と制約

このページは、公開Cloud Run環境で有料・無料の正常系を短時間で実演するための正本である。入口は[決済デモ](https://payment-user-agent-demo-kzeuhywicq-an.a.run.app)。稼働revision、image、環境境界は[deployment evidence](../../artifacts/cloud-run-deployment-81f3f41940c5.json)で確認できる。

- 公式x402、wallet、facilitator、on-chain処理、実資産の移動は実行しない。A2A x402部分はproject-local simulationで、公式profileには **NOT CONFORMANT**。
- 状態はephemeral（一時的）で、revisionの再起動・置換後の保持は保証しない。
- refundはローカル自動testの対象だが、最終hotfix後のCloud環境では **NOT RUN**。この実演は正常系に限定する。
- screenshot、credential、利用者prompt、model outputは証跡に保存しない。

詳細仕様は[概要](README.md)、[完全sequence](mediator-payment-integration-design/03_MEDIATION_FLOW.md#fig-flow-01)、[検証ガイド](VERIFICATION.md)を参照する。

## 2. まず貼り付けるプロンプト

### Paid（有料）

```text
有料の外部エージェントに、デモ予約商品を1件シミュレーション購入し、デモの予約確認を発行するよう依頼してください。
```

このシナリオでは、外部Booking Agentが東京出張向けの架空商品「デモ東京ベイホテル」（2026年9月12日〜14日、2名）の予約手配simulationを行う。`12.50 USD`は宿泊代ではなく、デモホテル予約手配サービス料である。

### Free（無料）

```text
東京で2026年9月12日から9月14日まで、2名で宿泊できるホテル候補を検索してください。
```

| ケース | 承認回数 | 画面上の期待state | 金額表示 | session |
| --- | ---: | --- | --- | --- |
| Paid | 2回（計画、決済） | `WaitingForPlanApproval` → `WaitingForPaymentApproval` → `Completed` | `12.50 USD` | 最初のfresh session |
| Free | 1回（計画） | `WaitingForPlanApproval` → `Completed` | なし | Paid後にログアウトし、再ログインしたfresh session |

承認入力は単一textの完全一致 `承認` だけを使う。前後の空白、`承認します`、連打は避け、各送信後はstateが変わるまで待つ。

## 3. 準備

1. 上記の公開URLを開く。
2. 共有済みreviewer credentialsを、別途案内された安全な経路から取得してログインする。この文書には値を記載しない。
3. `payment_user_agent` とメッセージ入力欄が表示されたことを確認する。

## 4. Paid正常系

1. §2のPaidプロンプトを入力して送る。
   - 期待表示: 計画と `WaitingForPlanApproval`。
   - この時点では決済処理を開始しない。
2. 計画を読み、完全一致の `承認` を送る。
   - 期待表示: `12.50 USD` と `WaitingForPaymentApproval`。
   - 外部Agentが同一Taskを`input-required`として返したpayment request（支払要求）を表示している。
3. 金額・通貨・受取人・条件を読み、完全一致の `承認` をもう一度送る。
   - 期待表示: `Completed` と業務結果。
   - 条件にはホテル、日程、人数、予約手配サービス料、宿泊代を含まないこと、実予約ではないことが含まれる。決定論的handlerがAP2 Payment Mandateを作り、仲介のpayment authorityがsimulation保証を発行する。Merchantが保証等を検証して`working`を返し、仲介のrailが同期simulationを記録する。そのreceiptをMerchantが検証し、同じTaskで業務を完了する。
   - 完了結果は、固定シナリオと同一Taskに結び付いた「デモ予約確認（シミュレーション）」である。`SIMULATED / NOT A REAL BOOKING`とデモ参照番号を表示し、実際の宿泊には使えない。
4. 必要ならreloadし、同じ稼働revision内で`Completed`が復元されることを確認する。これは耐久保存の証明ではない。
5. Paid完了後にログアウトする。Freeは同じ会話を流用せず、再ログインしてfresh sessionを作る。

## 5. Free正常系

1. §2のFreeプロンプトを入力して送る。
   - 期待表示: 計画と `WaitingForPlanApproval`。
2. 計画を読み、完全一致の `承認` を1回だけ送る。
   - 期待表示: 決済承認画面を経ず、`Completed`と検索結果。
   - Payment Mandate、simulation保証、settlementは生成しない。二回目の承認も求めない。
3. 完了を確認したらログアウトする。

## 6. 30秒で説明する仕組み

| 用語 | このデモでの意味 |
| --- | --- |
| Merchant | 有料の業務を履行する外部Agentで、payee（受取人）は`demo-merchant`。Merchant自身は決済やsettlementを行わない。 |
| payment request | Merchantが同一Taskを`input-required`にして返す支払要求。`amount`、`currency`、`payee`、`terms`を含み、利用者の決済承認前に画面へ表示する。 |
| AP2 | 利用者が支払条件を承認した事実と、決定論的handlerが作るPayment Mandateを認可証跡として扱う。LLMは承認やMandateを作成・変更できない。 |
| A2A x402 | 仲介とMerchantがsimulation保証とsettlement receiptを同一Task上で交換するproject-local simulation。公式profileには **NOT CONFORMANT**。 |

仲介はworkflow、payment authority、SQLite simulation railのownerだが、payeeではない。railは`demo-customer`から`demo-merchant`への同期simulationを記録するだけで、承認画面にも固定条件として示すとおり、実予約、実在庫hold、実課金、実送金、後日精算、法的保証はない。

Free検索とPaid手配は説明上の連続した物語だが、現行デモでは別sessionであり、検索結果を自動で引き継がない。Paidは固定シナリオを改めて指定して実行する。将来はBooking Agentの固定fixtureを実在庫APIへ差し替えられるが、在庫・価格の再確認、実決済、取消、個人情報、法令対応は本デモの範囲外である。

> 利用者は仲介画面で計画と支払条件を別々に承認します。AP2はその意思をMandateという証跡にし、仲介は外部Merchantへsimulation保証を渡します。Merchantが保証を検証した後、仲介railが実資産を動かさない同期simulationを記録し、Merchantはreceiptを確認して業務を完了します。無料では計画承認だけで、Mandateも保証もありません。

## 7. 詰まったときの確認

- stateが変わらない: 数秒待ち、送信を連打しない。承認入力が完全一致か確認する。
- Paidで金額が出ない: §2のPaidプロンプトを使ったか、fresh sessionか確認する。
- Freeで決済承認が出た: ログアウトして再ログインし、fresh sessionで§2のFreeプロンプトからやり直す。
- reload後に消えた: Cloud Runの状態はephemeralで、revision再起動・置換をまたぐ復元は保証外。
- サービス異常が疑われる: [運用ガイド](OPERATIONS.md)と[検証レポート](MEDIATOR_PAYMENT_INTEGRATION_TEST_REPORT.md)でreadinessと既知の境界を確認する。
