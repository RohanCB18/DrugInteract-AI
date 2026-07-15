# ── Backend Dockerfile ────────────────────────────────────────────────────
# Multi-stage: install deps in a builder, copy to slim runtime

FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for RDKit and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Runtime ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory to /app
WORKDIR /app

# Set Python Path so that 'backend' module is discoverable
ENV PYTHONPATH=/app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy backend files into /app/backend
COPY . /app/backend

# Create folders inside backend
RUN mkdir -p /app/backend/data_cache /app/backend/checkpoints

# Expose FastAPI port
EXPOSE 8000

# Start uvicorn referencing the discoverable backend module
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
