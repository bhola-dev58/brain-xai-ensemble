"""
tests/test_logging_observability.py
===================================
Focused test suite for structured logging, runtime observability,
request tracing, failure handling, and privacy preservation.
"""

import io
import logging
import uuid
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from src.logger import configure_logging, logger, request_id_ctx
from src.api import app
from src.inference import is_ensemble_ready, get_model_status, init_models
import src.inference as inference_mod


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def dummy_png_bytes():
    """Generates a minimal valid RGB PNG image in-memory."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[20:80, 20:80] = [120, 120, 120]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── 1. Duplicate Handlers Prevention ──────────────────────────────────────────
def test_configure_logging_prevents_duplicate_handlers():
    """Verifies that calling configure_logging multiple times does not duplicate handlers."""
    log_inst = configure_logging()
    initial_handlers = [h for h in log_inst.handlers if getattr(h, "_is_btxai_handler", False)]
    assert len(initial_handlers) == 1

    # Call configure_logging multiple times
    configure_logging("DEBUG")
    configure_logging("INFO")
    configure_logging("WARNING")

    updated_handlers = [h for h in log_inst.handlers if getattr(h, "_is_btxai_handler", False)]
    assert len(updated_handlers) == 1, "Duplicate handlers were added on re-configuration"


# ── 2. Model Initialization Failure Logging ──────────────────────────────────
def test_model_initialization_failure_logged(caplog, monkeypatch):
    """Verifies that model initialization failures are captured in structured server logs."""
    # Point models to a non-existent path
    monkeypatch.setattr(inference_mod, "CONVNEXT_PATH", "/non/existent/convnext.keras")

    with caplog.at_level(logging.WARNING, logger="BrainTumorXAI"):
        # Reset model to force reload check
        monkeypatch.setattr(inference_mod, "model_convnext", None)
        inference_mod.init_models(force_reload=True)

    # Check that a warning or error was logged
    log_text = caplog.text
    assert "not found" in log_text.lower() or "failed to load" in log_text.lower()


# ── 3. Unexpected API Errors Produce Controlled Responses ────────────────────
def test_unexpected_api_error_produces_controlled_response_and_logs(client, caplog, monkeypatch):
    """
    Verifies that unhandled server exceptions produce a controlled 500 response
    with request_id and no leaked tracebacks or server paths.
    """
    def faulty_ensemble_check():
        raise RuntimeError("Simulated internal database connection drop: /var/db/secret.db")

    monkeypatch.setattr("src.api.is_ensemble_ready", faulty_ensemble_check)

    with caplog.at_level(logging.ERROR, logger="BrainTumorXAI"):
        response = client.get("/api/health", headers={"X-Request-ID": "test-req-err-12345"})

    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "An internal server error occurred while processing the request."
    assert data["request_id"] == "test-req-err-12345"
    assert response.headers.get("X-Request-ID") == "test-req-err-12345"

    # Ensure sensitive server paths / internal error is NOT exposed to client
    assert "secret.db" not in response.text
    assert "Traceback" not in response.text

    # Verify that the server log recorded the unhandled error with the request ID
    assert "500" in caplog.text
    assert "Simulated internal database connection drop" in caplog.text


# ── 4. Request Timing and Status Logging ──────────────────────────────────────
def test_request_timing_and_id_propagation(client, caplog):
    """
    Verifies that requests record duration and status, and propagate X-Request-ID.
    """
    custom_req_id = "req-audit-" + uuid.uuid4().hex[:8]

    with caplog.at_level(logging.INFO, logger="BrainTumorXAI"):
        resp = client.get("/api/health", headers={"X-Request-ID": custom_req_id})

    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID") == custom_req_id

    # Verify that log message contains method, path, status, and duration
    log_text = caplog.text
    assert "GET /api/health" in log_text
    assert "status=200" in log_text
    assert "duration=" in log_text


# ── 5. Sensitive Values Excluded from Logs ────────────────────────────────────
def test_sensitive_values_excluded_from_logs(client, dummy_png_bytes, caplog):
    """
    Verifies that patient names, MRNs, raw image bytes, and query parameters
    are not leaked into the operational logs.
    """
    patient_name = "JaneDoeSecretName"
    patient_id = "MRN-Confidential-98765"

    with caplog.at_level(logging.INFO, logger="BrainTumorXAI"):
        client.post(
            "/api/report",
            files={"file": ("test_mri_scan.png", dummy_png_bytes, "image/png")},
            data={
                "patient_id": patient_id,
                "patient_name": patient_name,
                "patient_age": "50",
                "patient_gender": "Female",
            },
        )

    log_text = caplog.text

    # Patient identifiable information must NOT appear in the log output
    assert patient_name not in log_text, "Patient name leaked in server logs!"
    assert patient_id not in log_text, "Patient ID/MRN leaked in server logs!"
    # Raw image bytes representation must not appear
    assert "b'\\x89PNG" not in log_text


# ── 6. Readiness Accurately Reflects Model Availability ───────────────────────
def test_readiness_accurately_reflects_model_availability(client, monkeypatch):
    """
    Verifies /api/health accurately distinguishes ready vs degraded states
    without triggering expensive inference.
    """
    # 1. When fully loaded
    monkeypatch.setattr(inference_mod, "model_convnext", object())
    monkeypatch.setattr(inference_mod, "model_inc", object())
    monkeypatch.setattr(inference_mod, "model_dense", object())

    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["ready"] is True
    assert all(data["models_active"].values())

    # 2. When one model is missing
    monkeypatch.setattr(inference_mod, "model_dense", None)
    resp_degraded = client.get("/api/health")
    assert resp_degraded.status_code == 200
    data_degraded = resp_degraded.json()
    assert data_degraded["status"] == "degraded"
    assert data_degraded["ready"] is False
    assert data_degraded["models_active"]["DenseNet121"] is False


# ── 7. Existing Prediction and Report Workflows Functional ────────────────────
def test_prediction_and_report_workflows_functional(client, dummy_png_bytes):
    """
    Verifies that the prediction endpoint remains functional and includes
    the analysis_id and traceable request context.
    """
    resp = client.post(
        "/api/predict",
        files={"file": ("mri_sample.png", dummy_png_bytes, "image/png")}
    )
    # The endpoint should return a 200 if models are loaded or 503 if unloaded, but valid JSON structure
    assert resp.status_code in (200, 503)
    assert "X-Request-ID" in resp.headers

    if resp.status_code == 200:
        data = resp.json()
        assert "analysis_id" in data
        assert "class_name" in data
        assert "confidence" in data
