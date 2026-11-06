# FraudGuard AI — Production Dockerfile
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered streaming logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies for C-extensions (gcc/g++ for building libraries if required, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure data directory exists for SQLite history persistence
RUN mkdir -p /app/data

# Expose API port (documentary; Render sets PORT environment dynamically)
EXPOSE 8000

# Container healthcheck querying /health endpoint dynamically
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Launch production ASGI server with single high-throughput worker (prevents duplicating model memory in free-tier container)
CMD ["sh", "-c", "exec uvicorn src.api.fraud_api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

