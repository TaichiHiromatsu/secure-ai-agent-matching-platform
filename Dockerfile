# Multi-service container for Cloud Run
# Combines: secure_mediation_agent, trusted_agent_store, external-agents
# Uses Nginx as reverse proxy, supervisord for process management

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    nginx \
    supervisor \
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

# Copy trusted_agent_store application
COPY trusted_agent_store/app /app/trusted_agent_store/app
COPY trusted_agent_store/static /app/trusted_agent_store/static
RUN mkdir -p /app/trusted_agent_store/data/agents /app/data/agents
COPY trusted_agent_store/data/agents/registered-agents.json /app/trusted_agent_store/data/agents/
COPY trusted_agent_store/data/agents/registered-agents.json /app/data/agents/

# ============================================
# Secure Mediation Agent & External Agents
# ============================================
COPY secure_mediation_agent ./secure_mediation_agent/secure_mediation_agent
COPY user-agent ./user-agent
# The existing ADK Web scans /app/secure_mediation_agent for agent packages.
# Include the demo user agent there so it appears in the same chat UI selector.
COPY user-agent ./secure_mediation_agent/payment_user_agent
COPY external-agents ./external-agents
COPY scripts ./scripts
RUN chmod +x /app/scripts/verify_payment_demo.sh /app/user-agent/payment_cli.py

# ============================================
# Firebase Authentication
# ============================================
COPY deploy/auth /app/auth
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
    && chmod 700 /app/payment-evidence

# Set environment variables
ENV PYTHONPATH=/app:/app/secure_mediation_agent:/app/trusted_agent_store:/app/jury-judge-worker:/app/evaluation-runner/src
ENV DATABASE_URL=sqlite:////app/trusted_agent_store/data/agent_store.db
ENV PAYMENT_MARKETPLACE_DB=/app/payment-data/marketplace.db
ENV PAYMENT_EVIDENCE_DB=/app/payment-evidence/evidence.db

# Cloud Run uses port 8080
EXPOSE 8080

# Start supervisord (manages nginx + all services)
CMD ["/app/start.sh"]
