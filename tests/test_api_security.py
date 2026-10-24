"""
tests/test_api_security.py
==========================
Security audit and vulnerability regression test suite for BrainTumorXAI.

Covers:
  1. Oversized uploads are rejected (HTTP 413).
  2. Unsupported or malformed files are rejected safely (HTTP 415 / 422).
  3. Malicious filenames / path traversal cannot escape the intended directory.
  4. Invalid requests produce controlled responses without stack trace leakage.
  5. Sensitive endpoints enforce the intended access policy when API key is set.
  6. Unauthorized users cannot access protected records or reports (HTTP 401).
  7. Database operations remain safe against SQL injection attempts.
  8. Temporary files are cleaned up after failed and successful processing.
  9. Existing legitimate requests continue to work as expected.
 10. Health and telemetry checks remain publicly available.
"""

import io
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api import app
from src.database import init_db, log_scan, get_scan_by_analysis_id, get_scans_paginated
from src.report_generator import generate_pdf_report
from src.security import (
    validate_and_read_upload,
    detect_file_format,
    validate_image_content,
    validate_image_dimensions,
    sanitize_filename,
    sanitize_patient_metadata,
    validate_analysis_id,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_png_bytes():
    """Generates a small valid 100x100 RGB PNG in memory."""
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_jpeg_bytes():
    """Generates a small valid 100x100 RGB JPEG in memory."""
    img = Image.new("RGB", (100, 100), color=(64, 64, 64))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestSecurityUploadValidation:
    """Tests 1 & 2: Upload size and format safety."""

    def test_oversized_upload_rejected_with_413(self, client, monkeypatch):
        """1. Oversized uploads are rejected with HTTP 413."""
        # Lower MAX_UPLOAD_SIZE_BYTES artificially for quick test
        from src import security
        monkeypatch.setattr(security, "MAX_UPLOAD_SIZE_BYTES", 1024 * 50)  # 50 KB limit
        monkeypatch.setattr(security, "MAX_UPLOAD_SIZE_MB", 1)

        huge_payload = b"\x89PNG\r\n\x1a\n" + b"A" * (1024 * 60)  # 60 KB payload
        response = client.post(
            "/api/predict",
            files={"file": ("test_huge.png", huge_payload, "image/png")},
        )
        assert response.status_code == 413
        assert "exceeds the maximum allowed size" in response.json()["detail"]

    def test_empty_file_rejected_with_400(self, client):
        """Empty uploads produce HTTP 400."""
        response = client.post(
            "/api/predict",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_unsupported_file_format_rejected_with_415(self, client):
        """2. Unsupported file types (e.g. executable/text) are rejected safely."""
        fake_payload = b"GIF89aThisIsFakeGIFOrTextContentNotSupported"
        response = client.post(
            "/api/predict",
            files={"file": ("malicious.sh", b"#!/bin/bash\nrm -rf /", "text/plain")},
        )
        assert response.status_code == 415
        assert "Unsupported file type" in response.json()["detail"]

    def test_corrupt_image_rejected_safely(self, client):
        """Malformed image data produces a controlled 422 response without server crash."""
        # Starts with PNG magic bytes but followed by corrupt junk
        corrupt_png = b"\x89PNG\r\n\x1a\n" + b"CorruptGarbageDataNotAValidPng"
        response = client.post(
            "/api/predict",
            files={"file": ("corrupt.png", corrupt_png, "image/png")},
        )
        assert response.status_code == 422
        assert "corrupted" in response.json()["detail"].lower() or "decode" in response.json()["detail"].lower()

    def test_decompression_bomb_dimension_protection(self):
        """Dimensions outside safe bounds (10 to 5000) are rejected."""
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            validate_image_dimensions(height=5001, width=100)
        assert exc_info.value.status_code == 422

        with pytest.raises(HTTPException) as exc_info2:
            validate_image_dimensions(height=5, width=100)
        assert exc_info2.value.status_code == 422


class TestPathTraversalAndSanitization:
    """Tests 3 & 4: Path traversal prevention and input sanitization."""

    def test_malicious_filename_cannot_escape_directory(self):
        """3. Path traversal sequences in filenames are stripped."""
        evil_name = "../../../../etc/passwd"
        safe = sanitize_filename(evil_name)
        assert "/" not in safe
        assert ".." not in safe
        assert "passwd" in safe

        windows_evil = "..\\..\\windows\\system32\\cmd.exe"
        safe_win = sanitize_filename(windows_evil)
        assert "\\" not in safe_win
        assert "cmd.exe" in safe_win

    def test_report_generator_rejects_path_traversal(self):
        """Report generator raises ValueError if output_pdf_path escapes reports/."""
        mock_result = {
            "is_valid": True,
            "class_name": "Glioma",
            "confidence": 95.0,
            "is_tumor": True,
            "severity": "Moderate",
            "tumor_percentage": 4.5,
            "location": "Frontal Lobe",
            "uncertainty": 0.1,
            "inference_time": 0.05,
            "avg_pred": [0.05, 0.90, 0.03, 0.02],
            "classes": ["No Tumor", "Glioma", "Meningioma", "Pituitary"],
            "gradcam_overlay": np.zeros((100, 100, 3), dtype=np.uint8),
            "gradcam_pp_overlay": np.zeros((100, 100, 3), dtype=np.uint8),
            "segmentation_img": np.zeros((100, 100, 3), dtype=np.uint8),
            "analysis_id": "ANL-TESTTRAVERSAL1",
            "timestamp": "2026-10-24 12:00:00",
        }

        with pytest.raises(ValueError) as exc_info:
            generate_pdf_report(
                diag_result=mock_result,
                output_pdf_path="/etc/evil_report.pdf",
            )
        assert "Path traversal detected" in str(exc_info.value)

    def test_metadata_sanitization_bounds_inputs(self):
        """4. Patient metadata inputs are length-bounded and stripped of control characters."""
        long_name = "Dr. " + ("A" * 500)
        long_id = "PT-" + ("9" * 300) + "/../../../"
        safe_pid, safe_name, safe_age, safe_gender = sanitize_patient_metadata(
            patient_id=long_id,
            patient_name=long_name,
            patient_age="45 years old and extra string",
            patient_gender="Male\x00Injected",
        )
        assert len(safe_pid) <= 64
        assert "/" not in safe_pid
        assert len(safe_name) <= 100
        assert "\x00" not in safe_gender

    def test_invalid_analysis_id_format_rejected(self):
        """Invalid analysis ID strings produce HTTP 400."""
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            validate_analysis_id("../../etc/passwd")
        assert exc.value.status_code == 400

        with pytest.raises(HTTPException):
            validate_analysis_id("INVALID_PREFIX-1234")


class TestAccessControlAndAuthentication:
    """Tests 5 & 6: API key enforcement and unauthorized access protection."""

    def test_sensitive_endpoints_require_key_when_configured(self, client, sample_png_bytes, monkeypatch):
        """5 & 6. When BRAIN_XAI_API_KEY is configured, unauthenticated requests are rejected with 401."""
        monkeypatch.setenv("BRAIN_XAI_API_KEY", "super-secret-clinical-key-9988")

        # Request without key
        resp = client.post(
            "/api/predict",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
        )
        assert resp.status_code == 401
        assert "Authentication required" in resp.json()["detail"]

        # Request with incorrect key
        resp_wrong = client.post(
            "/api/predict",
            headers={"X-API-Key": "wrong-key"},
            files={"file": ("test.png", sample_png_bytes, "image/png")},
        )
        assert resp_wrong.status_code == 401
        assert "Invalid API key" in resp_wrong.json()["detail"]

        # History endpoint also protected
        resp_hist = client.get("/api/history")
        assert resp_hist.status_code == 401

        # Reports endpoint also protected
        resp_rep = client.get("/api/reports/ANL-NONEXISTENT")
        assert resp_rep.status_code == 401

    def test_authenticated_requests_succeed_with_api_key(self, client, sample_png_bytes, monkeypatch):
        """Requests with valid API key succeed."""
        test_key = "valid-secret-key-123"
        monkeypatch.setenv("BRAIN_XAI_API_KEY", test_key)

        with patch("src.api.predict_tumor_logic") as mock_pred:
            mock_pred.return_value = {
                "is_valid": True,
                "class_name": "Glioma",
                "confidence": 92.5,
                "is_tumor": True,
                "severity": "Moderate",
                "severity_color": "#f59e0b",
                "tumor_percentage": 3.8,
                "location": "Right Frontal",
                "uncertainty": 0.2,
                "inference_time": 0.04,
                "avg_pred": [0.05, 0.925, 0.015, 0.01],
                "classes": ["No Tumor", "Glioma", "Meningioma", "Pituitary"],
                "gradcam_overlay": np.zeros((50, 50, 3), dtype=np.uint8),
                "gradcam_pp_overlay": np.zeros((50, 50, 3), dtype=np.uint8),
                "segmentation_img": np.zeros((50, 50, 3), dtype=np.uint8),
                "analysis_id": "ANL-AUTHKEYSUCCESS1",
                "timestamp": "2026-10-24 14:00:00",
            }

            # Header X-API-Key
            resp = client.post(
                "/api/predict",
                headers={"X-API-Key": test_key},
                files={"file": ("test.png", sample_png_bytes, "image/png")},
            )
            assert resp.status_code == 200
            assert resp.json()["class_name"] == "Glioma"

            # Bearer token format
            resp_bearer = client.post(
                "/api/predict",
                headers={"Authorization": f"Bearer {test_key}"},
                files={"file": ("test.png", sample_png_bytes, "image/png")},
            )
            assert resp_bearer.status_code == 200

    def test_public_health_and_models_endpoints_accessible_without_key(self, client, monkeypatch):
        """10. Public liveness and telemetry endpoints do not require an API key even when set."""
        monkeypatch.setenv("BRAIN_XAI_API_KEY", "super-secret-clinical-key-9988")

        res_health = client.get("/api/health")
        assert res_health.status_code == 200
        assert "status" in res_health.json()

        res_models = client.get("/api/models")
        assert res_models.status_code == 200
        assert "architecture" in res_models.json()


class TestSQLInjectionAndDatabaseSafety:
    """Test 7: Database operations remain safe against injection."""

    def test_sql_injection_in_analysis_id_handled_safely(self):
        """7. Parameterized queries neutralize injection attempts."""
        evil_analysis_id = "ANL-' OR '1'='1"
        # Should return None, not dump table
        record = get_scan_by_analysis_id(evil_analysis_id)
        assert record is None

    def test_sql_injection_in_log_scan_handled_safely(self):
        """Values containing SQL statements are inserted as literal strings."""
        evil_pid = "PT-'; DROP TABLE scans; --"
        row_id = log_scan(
            predicted_class="Glioma",
            confidence_pct=90.0,
            is_tumor=True,
            severity="Low",
            tumor_area_pct=1.0,
            location="Frontal",
            inference_time=0.1,
            patient_id=evil_pid,
            analysis_id="ANL-SQLTESTSAFE1",
        )
        assert row_id > 0

        # Verify scans table still exists and record is retrievable
        retrieved = get_scan_by_analysis_id("ANL-SQLTESTSAFE1")
        assert retrieved is not None
        assert retrieved["patient_id"] == evil_pid


class TestTemporaryAssetCleanup:
    """Test 8: Cleanup of temporary report assets."""

    def test_temporary_files_cleaned_up_on_report_failure(self):
        """8. If report generation encounters an error, temp snapshot images are removed."""
        temp_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "temp")
        )
        analysis_id = "ANL-FAILCLEANUP1"

        # Pass invalid diag_result
        with pytest.raises(ValueError):
            generate_pdf_report(
                diag_result={"is_valid": False, "error": "Simulated failure"},
                analysis_id=analysis_id,
            )

        # Confirm no temporary files with this analysis_id exist
        if os.path.exists(temp_dir):
            matching = [f for f in os.listdir(temp_dir) if analysis_id in f]
            assert len(matching) == 0


class TestLegitimateRequestsCompatibility:
    """Test 9: Existing legitimate requests continue to work."""

    def test_legitimate_prediction_request_succeeds(self, client, sample_png_bytes):
        """9. Valid prediction request succeeds in unauthenticated dev mode."""
        with patch("src.api.predict_tumor_logic") as mock_pred:
            mock_pred.return_value = {
                "is_valid": True,
                "class_name": "Meningioma",
                "confidence": 96.0,
                "is_tumor": True,
                "severity": "Moderate",
                "severity_color": "#f59e0b",
                "tumor_percentage": 5.2,
                "location": "Left Parietal",
                "uncertainty": 0.15,
                "inference_time": 0.035,
                "avg_pred": [0.01, 0.02, 0.96, 0.01],
                "classes": ["No Tumor", "Glioma", "Meningioma", "Pituitary"],
                "gradcam_overlay": np.zeros((64, 64, 3), dtype=np.uint8),
                "gradcam_pp_overlay": np.zeros((64, 64, 3), dtype=np.uint8),
                "segmentation_img": np.zeros((64, 64, 3), dtype=np.uint8),
                "analysis_id": "ANL-LEGIT12345",
                "timestamp": "2026-10-24 15:30:00",
            }

            resp = client.post(
                "/api/predict",
                files={"file": ("brain_scan.png", sample_png_bytes, "image/png")},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["class_name"] == "Meningioma"
            assert "analysis_id" in data
            assert "gradcam_overlay_b64" in data

    def test_history_endpoint_returns_paginated_scans(self, client):
        """GET /api/history returns valid pagination payload."""
        resp = client.get("/api/history?limit=10&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "scans" in data
        assert isinstance(data["scans"], list)
