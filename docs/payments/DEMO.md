# 決済デモの実演ガイド

- 対象読者: 実演者、評価者
- 前提: [エージェント間決済の概要](README.md)
- 次に読む文書: [運用ガイド](OPERATIONS.md)、[検証ガイド](VERIFICATION.md)

## このデモで示すこと

利用者が`payment_user_agent`へ有料タスクを依頼すると、内部workflowが次の二段階で同意を得る。

1. どのMerchantへ何を依頼するかという計画を承認する。この時点では決済しない。
2. Merchantが返した具体的な価格と支払条件を、別の意思表示として承認する。

このデモは **AP2 v0.2 Human Present demo** である。A2A x402はproject-localなwire-shape fixtureで **NOT CONFORMANT**。実wallet、facilitator、blockchain、実資産、on-chain transactionは使用しない。

## 実演前の準備

環境の起動とreadiness確認は[運用ガイド](OPERATIONS.md#耐久ローカル環境)に従う。実演者は次を確認する。

- `/mediation-api/ready`が対象環境に応じた正常状態である。
- ADK Webで選択できるroot appが`payment_user_agent`だけである。
- Firebaseを使う環境ではログインできる。
- Cloud Runを使う場合は[deployment observation](../../artifacts/cloud-run-deployment.json)から現在のURLとephemeral境界を確認する。

## 5分の台本

### 0:00–1:00 — AP2とA2A x402の役割を説明する

次のように説明する。

> AP2は、誰が何をいくらで支払うことを承認したかをMandateと署名済みReceiptで証明します。A2A x402は、支払条件と支払結果をA2A Task上で交換します。今回はAP2のHuman Presentフローを実装し、x402部分は実資産を動かさないwire-shape fixtureとして確認します。

`payment_user_agent`は画面用の薄いadapterであり、状態、認可、鍵、Merchant呼出しの正本は内部workflowにあることも伝える。

### 1:00–2:00 — 計画を承認する

次の依頼を送る。

> 信頼済みの予約エージェントを使い、デモ予約を1件取得してください。支払総額が12.50 USDを超える場合は止めてください。

「計画の承認」画面で次を確認する。

- 選択したMerchantと商品
- 数量と金額上限
- 計画の有効期限
- この承認ではまだ決済されないという注意
- project-localなsimulation profile

次の一語だけを送る。

> 承認

`はい`、`yes`、`承認します`、前後空白付き`承認`は承認にならない、と説明する。

### 2:00–3:30 — 決済を承認する

「決済の承認」画面で次を確認する。

- Merchant／payee
- order IDとTask ID
- 商品、数量、通貨、期限
- 商品価格、顧客加算、collection cost、顧客総額、commission、Merchant受取額、payout cost
- `x402-wire-simulation/1`、`exact-simulated`、`demo:local`
- simulated／`NOT CONFORMANT`
- この承認後にsimulation settlementが始まるという注意

内容を確認し、計画承認とは別の意思表示として、再び次を送る。

> 承認

### 3:30–5:00 — 完了と復元を確認する

完了画面で次を確認する。

- `completed`
- Merchantの業務Artifact
- AP2 Checkout／Payment Receiptの安全なIDまたはdigest
- simulation resultの参照
- `AP2 v0.2 Human Present demo`
- 実資産とon-chain transactionがないこと

raw Mandate、credential、private key、raw proofが表示されないことも確認する。

ブラウザを再読み込みし、同じworkflowが`completed`として復元されることを示す。Cloud Run一時デモでは、これは同じrevisionが動いている間のUI復元を示すだけで、revision置換後の耐久性を示さない。

## CLIでの確認

同じ二承認フローをCLIで再現する。

```bash
docker exec secure-platform /app/.venv/bin/python \
  /app/user-agent/payment_cli.py \
  --workflow-url http://127.0.0.1:8080/mediation-api \
  --prompt "デモ予約を1件取得して" \
  --plan-approval "承認" \
  --payment-approval "承認"
```

Firebase環境では有効なsession cookieを渡す。内部portを直接呼んで認証を省略しない。

## 想定Q&A

### なぜ`承認`が二回必要か

最初は「どのMerchantへ何を依頼するか」という計画、二回目は「この価格と支払条件で支払うか」という決済への同意だからである。二つは別のID、nonce、署名対象、監査eventとして保存する。

### AP2とA2A x402はどちらか一方でよいのか

扱う問題が異なる。AP2は利用者の意思と取引証跡、A2A x402はTask上の支払交換を扱う。一般にはどちらかを別方式へ置き換えられるが、この設計では責務を分離して組み合わせている。

### A2A x402に準拠しているか

準拠していない。v0.1のdotted metadata、Task相関、結果履歴に似た形をproject-local fixtureで検証している。canonical URI、wallet、facilitator、on-chain settlementは`NOT RUN`である。

### AP2に完全準拠しているか

AP2 v0.2 Human Presentのclosed MandateとReceiptを使うdemoである。AP2全体の正式なconformance、Human Not Present、production-gradeなtrust deploymentは主張しない。

### 実際のお金は動くか

動かない。ローカルSQLiteのsimulation balanceだけを更新し、transaction参照も`sim:`で始まる。実transaction hashや法的な支払保証は生成しない。

### 各AP2ロールは別サービスか

論理的には鍵、issuer、検証責務を分けるが、すべてが別サービスではない。Merchantはloopbackの別プロセスで、Trusted Surface、Credential Provider、MPPは現在のdemoでは同じdeployable内のcomponentである。

### Cloud Runで再起動しても状態は残るか

保証しない。一時デモはephemeral filesystemを使い、revision再起動または置換で状態とdemo keyを失い得る。耐久性の受入対象は明示した永続volumeを使うローカル構成である。

## 実演後の検証

主要フローの後、必要に応じて一括verifierとoffline evidence verifierを実行する。手順とartifactの読み方は[検証ガイド](VERIFICATION.md)を参照する。

拒否、改ざん、replay、期限切れ、process death、migration、refund、reconciliationは自動testの対象であり、5分の台本へ詰め込まない。公式A2A x402、実資産、耐久Cloud Run、production identity／KMSは未実装境界であり、単なる追加デモ項目として扱わない。
