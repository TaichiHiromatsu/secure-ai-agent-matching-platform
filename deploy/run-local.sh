#!/bin/bash
# Run the approved explicit durable single-host/single-container simulation.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

NO_BUILD=false
NO_CACHE=false
for arg in "$@"; do
    case "$arg" in
        --no-build) NO_BUILD=true ;;
        --no-cache) NO_CACHE=true ;;
        *) echo "Unknown option: $arg"; exit 2 ;;
    esac
done

IMAGE_NAME="${LOCAL_IMAGE_NAME:-enterprise-a2a-pf:ap2-simulation}"
PAYMENT_DATA_DIR="${PAYMENT_DATA_DIR:-$PWD/.local/payment-data}"
PAYMENT_EVIDENCE_DIR="${PAYMENT_EVIDENCE_DIR:-$PWD/.local/payment-evidence}"
AP2_KEY_DIR="${AP2_KEY_DIR:-$PWD/.local/ap2-demo-keys}"

mkdir -p "$PAYMENT_DATA_DIR" "$PAYMENT_EVIDENCE_DIR" "$AP2_KEY_DIR"
chmod 700 "$PAYMENT_DATA_DIR" "$PAYMENT_EVIDENCE_DIR" "$AP2_KEY_DIR"
install -m 600 /dev/null "$PAYMENT_DATA_DIR/.durable-volume"
install -m 600 /dev/null "$PAYMENT_EVIDENCE_DIR/.durable-volume"

if [ "$NO_BUILD" = false ]; then
    if [ "$NO_CACHE" = true ]; then
        docker build --no-cache -t "$IMAGE_NAME" .
    else
        docker build -t "$IMAGE_NAME" .
    fi
fi

docker run --rm \
    --entrypoint /app/.venv/bin/python \
    -v "$AP2_KEY_DIR:/keys" \
    "$IMAGE_NAME" \
    /app/scripts/provision_ap2_demo_keys.py /keys

docker stop secure-platform 2>/dev/null || true
docker rm secure-platform 2>/dev/null || true

docker run -d \
    --name secure-platform \
    -p 8080:8080 \
    -v "$PAYMENT_DATA_DIR:/app/payment-data" \
    -v "$PAYMENT_EVIDENCE_DIR:/app/payment-evidence" \
    -v "$AP2_KEY_DIR:/run/secrets/ap2-demo:ro" \
    -e APP_ENV=local \
    --env-file "${ENV_FILE:-.env}" \
    "$IMAGE_NAME"

echo "Started explicit durable simulation: http://localhost:8080"
echo "Profile: x402-wire-simulation/1 (NOT CONFORMANT; no real/on-chain payment)"
echo "Logs: docker logs -f secure-platform"
