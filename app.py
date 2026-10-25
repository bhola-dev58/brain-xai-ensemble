"""
BrainTumorXAI — Production Application Entrypoint
Launches the FastAPI PACS Clinical Diagnostic Studio workstation.

Usage:
    python app.py
    # or
    python scripts/run_api.py

Access:
    Web Workstation: http://localhost:8000
    Interactive API Docs: http://localhost:8000/docs
    ReDoc Documentation: http://localhost:8000/redoc
"""
import os
import sys
import uvicorn
from src.config import HOST, PORT

if __name__ == "__main__":
    host = os.environ.get("HOST", HOST)
    port = int(os.environ.get("PORT", PORT))
    print(f"Starting BrainTumorXAI Clinical Workstation on http://{host}:{port}...")
    uvicorn.run("src.api:app", host=host, port=port, reload=False)
