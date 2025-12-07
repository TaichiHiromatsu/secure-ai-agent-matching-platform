#!/bin/bash
set -e

echo "Starting all services..."

# Ensure data directory exists
mkdir -p /app/trusted_agent_store/data

# Start supervisord (manages nginx + all services)
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
