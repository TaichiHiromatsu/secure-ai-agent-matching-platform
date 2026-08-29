# 仲介エージェント決済デモ：これ一枚で実演

## 1. 目的とデモの制約

このページは、公開Cloud Run環境で有料・無料の正常系を短時間で実演するための正本である。現行revision時点の入口は[決済デモ](https://payment-user-agent-demo-kzeuhywicq-an.a.run.app)。稼働revision、image、環境境界は[この受入時点のdeployment evidence](../../artifacts/cloud-run-deployment-399750d686a8.json)で確認できる。

- AP2は、利用者の支払意思と証跡を扱う。このデモではHuman Present（利用者が画面にいる）フローをsimulationする。
- A2A x402は、エージェント間の支払要求交換を模したproject-local fixtureであり、公式profileには **NOT CONFORMANT**。公式x402、wallet、facilitator、on-chain処理は実行しない。
- 実資産・実送金・法的な支払保証はない。状態はephemeral（一時的）で、revisionの再起動・置換後の保持は保証しない。
- refundはローカル自動testの対象だが、最終hotfix後のCloud環境では **NOT RUN**。今回の実演は正常系に限定する。
- screenshot、credential、利用者prompt、model outputは証跡に保存しない。

詳細仕様は[概要](README.md)、[アーキテクチャ](ARCHITECTURE.md)、[検証ガイド](VERIFICATION.md)を参照する。

## 2. 準備

1. 上記の公開URLを開く。
2. 共有済みのreviewer credentialsを、別途案内された安全な経路から取得してログインする。この文書には値を記載しない。
3. 画面が入力待ちになったことを確認する。

入力上の注意:

- promptは以下のcodeをそのまま使う。
- 承認入力は単一textの完全一致 `承認` のみ。前後の空白、`承認します`、連打は避ける。
- 各送信後はstateが変わるまで待つ。処理中に再送しない。

## 3. Paid正常系（承認2回）

| 手順 | 操作 | 画面上の期待結果 | 裏側で起きること |
| --- | --- | --- | --- |
| 1 | `paid payment booking` を送る | 計画と `WaitingForPlanApproval` | 仲介エージェントが外部Agent候補と依頼範囲を提示する。まだ決済しない。 |
| 2 | 完全一致の `承認` を1回送る | `12.50 USD` と `WaitingForPaymentApproval` | 仲介が外部Agentへ同一Taskで依頼し、支払条件を受け取る。表示額はminor units `1250`、小数桁 `2` をUSD表記したもの。 |
| 3 | 金額・条件を読み、完全一致の `承認` をもう1回送る | 最終state `Completed` と業務結果 | 決定論的routerがLLMを介さずapprove handlerを呼び、AP2 Payment Mandate（利用者の支払意思の証跡）を生成する。仲介側の決定論的payment workflow／railがsimulation authorizationとsettlementを処理し、payment authorityがsigned simulation guaranteeを発行する。外部Agentはその保証、capability、Task相関、AP2安全要約を検証し、業務を履行して同じTaskを完了する。 |
| 4 | ブラウザをreloadする | `Completed` が復元される | 同じ稼働revision内の一時状態から復元する。耐久保存を示すものではない。 |

承認の責務はLLMにない。公開rootの決定論的routerとTrusted Surface（利用者同意を確定する信頼境界）が承認を識別し、専用handlerがMandateを作る。payment authority（仲介側の決済権限コンポーネント）だけがsimulation精算保証を作る。

Paid完了を確認したらログアウトする。Freeは同じ会話を流用せず、再ログインしたfresh sessionで始める。

## 4. Free正常系（承認1回）

| 手順 | 操作 | 画面上の期待結果 | 裏側で起きること |
| --- | --- | --- | --- |
| 1 | `hotel search` を送る | 計画と `WaitingForPlanApproval` | 仲介エージェントが無料の外部Agent呼出しを計画する。 |
| 2 | 完全一致の `承認` を1回送る | 決済承認画面を経ず、最終state `Completed` と検索結果 | 仲介が外部Agentの同一Taskを完了まで運ぶ。無料なのでPayment Mandate、simulation精算保証、決済処理は生成しない。 |

FreeをPaidへfallbackさせないこと、二回目の `承認` を求めないことが確認点である。

## 5. Paid / Freeの比較

| 項目 | Paid | Free |
| --- | --- | --- |
| exact prompt | `paid payment booking` | `hotel search` |
| 承認回数 | 2回（計画、決済） | 1回（計画） |
| 主要state | `WaitingForPlanApproval` → `WaitingForPaymentApproval` → `Completed` | `WaitingForPlanApproval` → `Completed` |
| AP2 Payment Mandate | あり | なし |
| 仲介のpayment workflow／rail | simulation authorizationとsettlementを実行 | 決済処理なし |
| 仲介のsimulation精算保証 | あり | なし |
| 外部Agent | signed simulation guarantee、capability、Task相関、AP2安全要約を検証し、業務履行後に同一Taskを完了 | 業務履行後に同一Taskを完了 |

## 6. 30秒トークトラック

> 利用者が支払う相手は仲介エージェントで、外部Agentは仲介からsimulation上の後日精算保証を受け取ります。有料では、まず依頼計画を承認し、12.50 USDの条件を見てもう一度承認します。二回目はLLMではなく決定論的handlerが処理し、AP2 Payment Mandateという支払意思の証跡を作ります。仲介側の決定論的workflowがsimulation authorizationとsettlementを行い、外部Agentはsigned guarantee、権限、Task相関、安全なAP2要約を検証して業務を履行します。無料では計画承認だけで、Mandateも精算保証もありません。x402部分は支払要求交換のsimulationで、公式・on-chain・実送金ではありません。

## 7. 詰まったときの確認と終了

- stateが変わらない: 数秒待ち、送信を連打しない。入力が完全一致の `承認` か確認する。
- Paidで金額が出ない: 最初のpromptが完全一致の `paid payment booking` か、新しいsessionか確認する。
- Freeで決済承認が出た: ログアウトし、fresh sessionで完全一致の `hotel search` からやり直す。
- reload後に消えた: Cloud Runの状態はephemeral。revision再起動・置換をまたぐ復元は保証外である。
- サービス異常が疑われる: [運用ガイド](OPERATIONS.md)と[検証レポート](MEDIATOR_PAYMENT_INTEGRATION_TEST_REPORT.md)でreadinessと既知の境界を確認する。

実演終了後は画面のログアウト操作を実行し、ログイン状態が解除されたことを確認する。
