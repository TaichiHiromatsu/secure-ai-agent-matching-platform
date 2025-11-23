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
COPY secure-mediation-agent ./secure-mediation-agent
COPY user-agent ./user-agent

# Install dependencies
RUN uv sync --frozen

# Expose port
EXPOSE 8000

# Run the mediation agent from secure-mediation-agent directory
CMD ["sh", "-c", "cd /app/secure-mediation-agent && uv run --directory /app adk web . --port 8000 --reload"]
