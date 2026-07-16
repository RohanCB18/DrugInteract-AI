# ── Backend Dockerfile ────────────────────────────────────────────────────
# Single-stage build to prevent disk space exhaustion during intermediate copies
# ─────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# Set Python Path so that 'backend' module is discoverable
ENV PYTHONPATH=/app

# System deps for RDKit and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only PyTorch first to prevent downloading 3.5GB of CUDA packages
RUN pip install --no-cache-dir torch==2.6.0 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files into /app/backend
COPY . /app/backend

# Create folders inside backend
RUN mkdir -p /app/backend/data_cache /app/backend/checkpoints

# Expose FastAPI port
EXPOSE 8000

# Start uvicorn referencing the discoverable backend module
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
