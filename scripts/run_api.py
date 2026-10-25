"""
Launcher script for the BrainTumorXAI FastAPI PACS Workstation and REST server.

Serves both the interactive PACS Clinical Diagnostic Studio and REST endpoints.

Usage:
    ./venv/bin/python scripts/run_api.py

Then open:
    http://localhost:8000/docs   — Interactive Swagger UI
    http://localhost:8000/redoc  — ReDoc documentation
    GET  http://localhost:8000/api/health
    POST http://localhost:8000/api/predict  (upload file=<mri.jpg>)
"""
import os
import sys

# Allow imports from the project root (e.g., src.inference, src.config)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from src.config import HOST, PORT

if __name__ == "__main__":
    print(f"Starting BrainTumorXAI REST API server on {HOST}:{PORT}...")
    uvicorn.run(
        "src.api:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
