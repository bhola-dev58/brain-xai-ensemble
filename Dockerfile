# ============================================================
#  BrainTumorXAI — Dockerfile
#  CPU-only inference (no CUDA required)
#  Runs Gradio dashboard on port 7860
#  Base: python:3.12-slim (matches host runtime, required by numpy>=2.3)
# ============================================================

FROM python:3.12-slim

# System-level dependencies required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Working directory inside the container
WORKDIR /app

# Copy dependency manifest first (layer-caching optimization)
COPY requirements.txt .

# Install Python dependencies (no cache to keep image lean)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and assets
COPY src/ ./src/
COPY static/ ./static/
COPY scripts/ ./scripts/
COPY test_images/ ./test_images/
COPY app.py .

# Ensure runtime directories exist
RUN mkdir -p models data reports

# Copy models if present in the build context (e.g. local builds)
COPY [m]odels* ./models/

# Initialize baseline models if directory is empty (ensures cloud builds without git models succeed)
RUN python scripts/init_models.py

# Expose default HTTP port (8000 for FastAPI + PACS workstation; supports cloud $PORT overrides)
EXPOSE 8000

# Force CPU-only execution inside container and disable output buffering
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Launch Uvicorn serving FastAPI (mounts static PACS UI at / and REST API at /api/*)
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
