FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY external-agents ./external-agents

# Install dependencies
RUN uv sync --frozen

# Expose port
EXPOSE 8003

# Run the agent using adk api_server command
# Fix agent.json URL before starting if AGENT_HOST is set
CMD ["sh", "-c", "\
if [ -n \"$AGENT_HOST\" ] && [ -n \"$AGENT_PORT\" ]; then \
  sed -i \"s|\\\"url\\\": \\\"http://[^:]*:[0-9]*|\\\"url\\\": \\\"http://$AGENT_HOST:$AGENT_PORT|g\" /app/external-agents/trusted-agents/hotel_agent/agent.json; \
  echo 'Updated agent.json URL to use AGENT_HOST=$AGENT_HOST'; \
fi && \
cd /app && uv run adk api_server --a2a --host 0.0.0.0 --port 8003 external-agents/trusted-agents/"]
