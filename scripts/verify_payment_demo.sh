#!/bin/sh
# Minimal, repeatable smoke test for the all-in-one payment demo container.
set -eu

python_bin="${PAYMENT_E2E_PYTHON:-/app/.venv/bin/python}"
payment_url="${PAYMENT_URL:-http://127.0.0.1:8004}"
merchant_url="${PAID_AGENT_URL:-http://127.0.0.1:8005}"
public_url="${PAYMENT_PUBLIC_GATEWAY_URL:-http://127.0.0.1:8080}"

echo "[demo] readiness"
curl --fail --silent --show-error "$payment_url/ready" >/dev/null
curl --fail --silent --show-error "$merchant_url/ready" >/dev/null
curl --fail --silent --show-error "$public_url/payment/.well-known/agent-card.json" >/dev/null
curl --fail --silent --show-error "$public_url/paid-agent/.well-known/agent-card.json" >/dev/null

echo "[demo] public route isolation"
for probe in \
  "$public_url/payment/internal/v1/payouts" \
  "$public_url/paid-agent/v1/payout-status-requests" \
  "$public_url/paid-agent/v1/fulfillments/not-public/not-public"
do
  status="$(curl --silent --output /dev/null --write-out '%{http_code}' --request POST "$probe")"
  [ "$status" = "404" ] || {
    echo "sensitive route was not blocked: $probe returned $status" >&2
    exit 1
  }
done

echo "[demo] user agent: request -> 承認 -> mediated payment"
"$python_bin" /app/user-agent/payment_cli.py \
  --mediator-url "$payment_url" \
  --prompt "デモ予約を1件取得して" \
  --approval "承認"

echo "[demo] happy+payout, failure+refund, timeout+reconcile"
"$python_bin" /app/scripts/run_payment_demo.py \
  --payment-url "$payment_url" \
  --merchant-url "$merchant_url"
