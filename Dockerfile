# Explicit single-host/single-container simulation image.
# Current Cloud Run deployment is intentionally blocked because its writable
# filesystem is ephemeral and does not satisfy the durable SQLite profile.

FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN pip install --no-cache-dir uv

# ============================================
# Install all dependencies via pyproject.toml (single source of truth)
# ============================================
COPY pyproject.toml uv.lock ./

# Install main dependencies with uv first
RUN uv sync --frozen

# ============================================
# Trusted Agent Store dependencies (evaluation-runner, jury-judge-worker)
# ============================================
COPY trusted_agent_store/evaluation-runner /app/evaluation-runner
COPY trusted_agent_store/third_party /app/third_party
COPY trusted_agent_store/jury-judge-worker /app/jury-judge-worker

RUN uv pip install -e /app/evaluation-runner
RUN UV_HTTP_TIMEOUT=300 uv pip install --no-cache-dir -r /app/jury-judge-worker/requirements.txt
RUN uv pip install -e /app/jury-judge-worker

FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    curl \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/evaluation-runner /app/evaluation-runner
COPY --from=builder /app/third_party /app/third_party
COPY --from=builder /app/jury-judge-worker /app/jury-judge-worker

# Copy trusted_agent_store application
COPY trusted_agent_store/app /app/trusted_agent_store/app
COPY trusted_agent_store/static /app/trusted_agent_store/static
RUN mkdir -p /app/trusted_agent_store/data/agents /app/data/agents
COPY trusted_agent_store/data/agents/registered-agents.json /app/trusted_agent_store/data/agents/
COPY trusted_agent_store/data/agents/registered-agents.json /app/data/agents/

# ============================================
# Secure Mediation Agent & External Agents
# ============================================
# The authoritative workflow is importable internal code.  Only the separate
# payment_user_agent directory is placed beneath ADK's discovery root.
COPY secure_mediation_agent /app/internal/secure_mediation_agent
COPY secure_mediation_agent /app/secure_mediation_agent
COPY payment_user_agent /app/payment-apps/payment_user_agent
COPY payment_user_agent /app/payment_user_agent
COPY user-agent ./user-agent
COPY external-agents ./external-agents
COPY scripts/migrate_ap2_x402_v2.py /app/scripts/migrate_ap2_x402_v2.py
COPY scripts/provision_ap2_demo_keys.py /app/scripts/provision_ap2_demo_keys.py
COPY scripts/run_regression_manifest.py /app/scripts/run_regression_manifest.py
COPY scripts/verify_ap2_x402_evidence.py /app/scripts/verify_ap2_x402_evidence.py
COPY scripts/verify_ap2_x402_runtime.py /app/scripts/verify_ap2_x402_runtime.py
COPY scripts/verify_payment_demo.sh /app/scripts/verify_payment_demo.sh
COPY scripts/validate_ap2_x402_release.py /app/scripts/validate_ap2_x402_release.py
COPY scripts/cloud_run_candidate.py /app/scripts/cloud_run_candidate.py
# The exact release image is self-testing: no source bind mount is needed.
COPY pyproject.toml /app/pyproject.toml
COPY Dockerfile /app/Dockerfile
COPY tests /app/tests
COPY trusted_agent_store/evaluation-runner/tests /app/trusted_agent_store/evaluation-runner/tests
COPY trusted_agent_store/jury-judge-worker/tests /app/trusted_agent_store/jury-judge-worker/tests
RUN chmod +x /app/scripts/*.py /app/scripts/*.sh /app/user-agent/payment_cli.py

# ============================================
# Firebase Authentication
# ============================================
COPY deploy/auth /app/auth
# Source-layout deployment files are retained for exact-image contract tests.
COPY deploy /app/deploy
RUN mkdir -p /app/static
COPY deploy/auth/login.html /app/static/login.html
RUN chmod 644 /app/static/login.html

# firebase-admin is installed via pyproject.toml dependencies

# ============================================
# Configuration files
# ============================================
COPY deploy/nginx.conf /etc/nginx/nginx.conf
COPY deploy/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY deploy/start.sh /app/start.sh
COPY deploy/start-nginx.sh /app/start-nginx.sh
RUN chmod +x /app/start.sh /app/start-nginx.sh

# Create required directories
RUN mkdir -p /var/log/nginx /var/log/supervisor /app/logs /app/payment-data /app/payment-evidence \
    && chmod 700 /app/payment-data /app/payment-evidence

# Set environment variables
ENV PYTHONPATH=/app/internal:/app:/app/trusted_agent_store:/app/jury-judge-worker:/app/evaluation-runner/src
ENV DATABASE_URL=sqlite:////app/trusted_agent_store/data/agent_store.db
ENV PAYMENT_MARKETPLACE_DB=/app/payment-data/marketplace.db
ENV PAYMENT_MERCHANT_DB=/app/payment-data/paid-agent.db
ENV PAYMENT_EVIDENCE_DB=/app/payment-evidence/evidence.db
ENV PAYMENT_DURABLE_VOLUME_MARKER=/app/payment-data/.durable-volume
ENV PAYMENT_EVIDENCE_VOLUME_MARKER=/app/payment-evidence/.durable-volume

# Cloud Run uses port 8080
EXPOSE 8080

# Start supervisord (manages nginx + all services)
CMD ["/app/start.sh"]
