#!/bin/bash
set -e

if [ "${DEV_MODE:-false}" = "true" ] && [ "${APP_ENV:-}" != "local" ]; then
    echo "Refusing startup: DEV_MODE=true is local-only (APP_ENV=local required)"
    exit 1
fi

if [ "${EPHEMERAL_CLOUD_RUN_DEMO:-false}" = "true" ]; then
    echo "Starting explicitly ephemeral Cloud Run demo services..."
    echo "EPHEMERAL DEMO: state and keys may reset on restart"
    echo "Target: ephemeral-cloud-run-demo; durability: NOT PROVIDED"
    echo "Official x402 and on-chain settlement: NOT RUN"
    mkdir -p /app/payment-data /app/payment-evidence /app/ephemeral/ap2-demo
    chmod 700 /app/payment-data /app/payment-evidence /app/ephemeral/ap2-demo
    export AP2_DEMO_KEY_DIR=/app/ephemeral/ap2-demo
    /app/.venv/bin/python /app/scripts/provision_ap2_demo_keys.py "$AP2_DEMO_KEY_DIR"
else
    echo "Starting explicit durable single-host simulation services..."
    if [ ! -f /app/payment-data/.durable-volume ]; then
        echo "Refusing paid startup: /app/payment-data is not an explicit durable mount"
        exit 1
    fi
    if [ ! -f /app/payment-evidence/.durable-volume ]; then
        echo "Refusing paid startup: /app/payment-evidence is not an explicit durable mount"
        exit 1
    fi
    if [ ! -d /run/secrets/ap2-demo ]; then
        echo "Refusing paid startup: persistent AP2 role-key mount is missing"
        exit 1
    fi
    export AP2_DEMO_KEY_DIR=/run/secrets/ap2-demo
fi

# Ensure data directory exists
mkdir -p /app/trusted_agent_store/data

# Start supervisord (manages nginx + all services)
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
