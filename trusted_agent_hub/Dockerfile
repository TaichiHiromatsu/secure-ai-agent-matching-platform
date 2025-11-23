FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY trusted_agent_hub/requirements.txt .
COPY trusted_agent_hub/sandbox-runner /app/sandbox-runner
COPY trusted_agent_hub/third_party /app/third_party

# Copy inspect-worker for Judge Panel (Agents-as-a-Judge)
COPY trusted_agent_hub/inspect-worker /app/inspect-worker

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e /app/sandbox-runner
RUN pip install --no-cache-dir -r /app/inspect-worker/requirements.txt

# Copy application code
COPY trusted_agent_hub/app /app/app

# Create data and static directories
RUN mkdir -p data static

# Set environment variables
ENV PYTHONPATH=/app:/app/inspect-worker
ENV DATABASE_URL=sqlite:////app/data/agent_store.db

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
