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
EXPOSE 8004

# Run the agent using adk api_server command
CMD ["sh", "-c", "cd /app && uv run adk api_server --a2a --host 0.0.0.0 --port 8004 external-agents/trusted-agents/"]
