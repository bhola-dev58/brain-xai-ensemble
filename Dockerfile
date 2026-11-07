# ============================================================
#  BrainTumorXAI — Production Hardened Dockerfile
#  Base: python:3.12-slim (matches host runtime, required by numpy>=2.3)
#  Security: Runs as unprivileged non-root user (appuser:appgroup)
#  Workstation: FastAPI PACS + Tri-Ensemble REST API on port 8000
# ============================================================

FROM python:3.12-slim

# System-level dependencies required by OpenCV and headless imaging
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged application user and group
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

# Set working directory
WORKDIR /app

# Copy dependency manifest first for layer caching optimization
COPY requirements.txt .

# Install Python dependencies (no pip cache to keep container lean)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, assets, scripts, and baseline test images
COPY src/ ./src/
COPY static/ ./static/
COPY scripts/ ./scripts/
COPY test_images/ ./test_images/
COPY app.py .

# Ensure storage directories exist with appropriate non-root ownership
RUN mkdir -p models data reports/temp && \
    chown -R appuser:appgroup /app

# Download and bake Phase 3 Tri-Ensemble model weights into container image
RUN python scripts/download_models.py && \
    chown -R appuser:appgroup /app/models

# Declare persistent storage mount points (SQLite db, reports)
VOLUME ["/app/data", "/app/reports"]

# Switch to non-root user for process isolation
USER appuser

# Expose HTTP port (8000 default; overridden via $PORT in cloud environments)
EXPOSE 8000

# Runtime Environment Variables (Optimized for low-memory container instances)
ENV CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    MALLOC_ARENA_MAX=2 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

# Container liveness and model-readiness healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c 'import urllib.request, sys; sys.exit(0 if urllib.request.urlopen("http://localhost:8000/api/health").getcode() == 200 else 1)'

# Ensure model artifacts exist and launch Uvicorn ASGI server
CMD ["sh", "-c", "python scripts/download_models.py && uvicorn src.api:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
