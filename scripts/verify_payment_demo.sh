#!/bin/sh
# One-command RC gate: public route, exact approvals, rejection, offline evidence.
set -eu

python_bin="${PAYMENT_E2E_PYTHON:-/app/.venv/bin/python}"
script_root="${PAYMENT_SCRIPT_ROOT:-/app/scripts}"
if [ -n "${WORKFLOW_SESSION_COOKIE:-}" ]; then
  set -- --session-cookie "$WORKFLOW_SESSION_COOKIE" "$@"
fi

exec "$python_bin" "$script_root/verify_ap2_x402_runtime.py" \
  --public-url "${PAYMENT_PUBLIC_URL:-http://127.0.0.1:8080/mediation-api}" \
  --gateway-url "${PAYMENT_GATEWAY_URL:-http://127.0.0.1:8080}" \
  "$@"
