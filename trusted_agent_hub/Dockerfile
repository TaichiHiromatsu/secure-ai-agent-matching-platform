FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY trusted_agent_hub/requirements.txt .
COPY trusted_agent_hub/evaluation-runner /app/evaluation-runner
COPY trusted_agent_hub/third_party /app/third_party

# Copy jury-judge-worker for Judge Panel (Agents-as-a-Judge)
COPY trusted_agent_hub/jury-judge-worker /app/jury-judge-worker

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e /app/evaluation-runner
RUN pip install --no-cache-dir -r /app/jury-judge-worker/requirements.txt
RUN pip install -e /app/jury-judge-worker

# Copy application code
COPY trusted_agent_hub/app /app/app

# Copy static files
COPY trusted_agent_hub/static /app/static

# Create data directory and copy initial data
RUN mkdir -p /app/data/agents
COPY trusted_agent_hub/data/agents/registered-agents.json /app/data/agents/

# Set environment variables
ENV PYTHONPATH=/app:/app/jury-judge-worker
ENV DATABASE_URL=sqlite:////app/data/agent_store.db

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
