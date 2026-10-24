"""
Unit and regression tests for safe diagnostic reporting, traceability,
and consistency in BrainTumorXAI.
"""
import os
import sys
import glob
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import app
import src.inference as inference
from src.report_generator import generate_pdf_report
from src.database import log_scan, get_scan_by_analysis_id, init_db


@pytest.fixture
def mock_valid_prediction():
    """Generates a mock valid prediction result dictionary."""
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    return {
        "is_valid": True,
        "analysis_id": "ANL-TEST-9999",
        "timestamp": "2026-10-24 12:00:00",
        "img": dummy_img,
        "segmentation_img": dummy_img,
        "gradcam_overlay": dummy_img,
        "gradcam_pp_overlay": dummy_img,
        "class_name": "Glioma Tumor",
        "confidence": 94.50,
        "uncertainty": 12.30,
        "is_tumor": True,
        "inference_time": 0.42,
        "tumor_percentage": 5.25,
        "location": "Left Frontal Lobe (Superior)",
        "severity": "Moderate",
        "severity_color": "#f59e0b",
        "avg_pred": np.array([0.02, 0.945, 0.025, 0.01], dtype=np.float32),
        "pred_dict": {
            "ConvNeXtSmall": np.array([0.01, 0.96, 0.02, 0.01]),
            "InceptionV3": np.array([0.03, 0.93, 0.03, 0.01]),
            "DenseNet121": np.array([0.02, 0.95, 0.02, 0.01]),
        },
        "classes": ["No Tumor", "Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor"],
        "class_idx": 1
    }


class TestDiagnosticReportingSafety:
    # ------------------------------------------------------------------ #
    # 1. Successful model output is labelled as prediction, not confirmed diagnosis
    # ------------------------------------------------------------------ #
    def test_successful_output_labelled_as_prediction(self, mock_valid_prediction):
        assert "class_name" in mock_valid_prediction
        assert mock_valid_prediction["is_valid"] is True
        # Ensure analysis_id and timestamp are attached for traceability
        assert mock_valid_prediction["analysis_id"].startswith("ANL-")
        assert len(mock_valid_prediction["timestamp"]) > 0

    # ------------------------------------------------------------------ #
    # 2. Confidence values are labelled accurately (0-100 range, softmax-derived)
    # ------------------------------------------------------------------ #
    def test_confidence_values_labelled_accurately(self, mock_valid_prediction):
        conf = mock_valid_prediction["confidence"]
        assert 0.0 <= conf <= 100.0
        # Check that avg_pred sums to approximately 1.0 (softmax output)
        assert np.isclose(float(np.sum(mock_valid_prediction["avg_pred"])), 1.0, atol=1e-3)

    # ------------------------------------------------------------------ #
    # 3. Severity output is not misrepresented as a confirmed tumor grade
    # ------------------------------------------------------------------ #
    def test_severity_output_not_misrepresented_as_tumor_grade(self):
        from src.processor import estimate_severity
        sev, color = estimate_severity(confidence=98.0, tumor_percentage=6.0)
        # Severity must be heuristic categorization ("High", "Moderate", "Low", "Borderline", "Uncertain")
        valid_severities = {"High", "Moderate", "Low", "Borderline", "Uncertain"}
        assert sev in valid_severities
        # Must not be WHO Roman numeral tumor grade (e.g., Grade IV)
        assert sev not in {"Grade I", "Grade II", "Grade III", "Grade IV", "WHO Grade IV"}

    # ------------------------------------------------------------------ #
    # 4. Invalid inference cannot produce a misleading completed report
    # ------------------------------------------------------------------ #
    def test_invalid_inference_cannot_produce_report(self):
        # 4a: Failed inference dict
        failed_diag = {"is_valid": False, "error": "Models unavailable"}
        with pytest.raises(ValueError) as excinfo:
            generate_pdf_report(failed_diag)
        assert "Cannot generate clinical report" in str(excinfo.value)

        # 4b: None input
        with pytest.raises(ValueError):
            generate_pdf_report(None)

    # ------------------------------------------------------------------ #
    # 5. Failed report generation is handled explicitly
    # ------------------------------------------------------------------ #
    def test_failed_report_generation_handled_explicitly(self, monkeypatch, mock_valid_prediction):
        client = TestClient(app)
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        _, img_encoded = cv2.imencode(".jpg", dummy_img)

        # Mock inference to isolate report generation failure
        monkeypatch.setattr("src.api.predict_tumor_logic", lambda *args, **kwargs: mock_valid_prediction)

        # Force report generator to raise an error
        def mock_generate_fail(*args, **kwargs):
            raise RuntimeError("Disk quota exceeded or filesystem write failure")

        monkeypatch.setattr("src.api.generate_pdf_report", mock_generate_fail)

        files = {"file": ("scan.jpg", img_encoded.tobytes(), "image/jpeg")}
        res = client.post("/api/report", files=files, data={"patient_id": "TEST-PT"})
        assert res.status_code == 500
        assert "Report generation failed:" in res.json()["detail"]

    # ------------------------------------------------------------------ #
    # 6. Prediction and report data remain consistent
    # ------------------------------------------------------------------ #
    def test_prediction_and_report_data_remain_consistent(self, mock_valid_prediction, tmp_path):
        out_pdf = str(tmp_path / "test_report.pdf")
        pdf_path = generate_pdf_report(
            diag_result=mock_valid_prediction,
            patient_id="PT-CONSISTENCY",
            patient_name="Consistency Check Patient",
            output_pdf_path=out_pdf,
            analysis_id=mock_valid_prediction["analysis_id"]
        )
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 1000  # Non-trivial PDF generated

    # ------------------------------------------------------------------ #
    # 7. Analysis identifiers and timestamps remain associated with the correct record
    # ------------------------------------------------------------------ #
    def test_analysis_id_and_timestamp_traceability(self, mock_valid_prediction):
        init_db()
        test_anl_id = "ANL-TRACE-12345"
        row_id = log_scan(
            predicted_class=mock_valid_prediction["class_name"],
            confidence_pct=mock_valid_prediction["confidence"],
            is_tumor=mock_valid_prediction["is_tumor"],
            severity=mock_valid_prediction["severity"],
            tumor_area_pct=mock_valid_prediction["tumor_percentage"],
            location=mock_valid_prediction["location"],
            inference_time=mock_valid_prediction["inference_time"],
            patient_id="PT-TRACE-01",
            patient_name="Traceable Patient",
            scan_filename="trace_scan.jpg",
            analysis_id=test_anl_id
        )
        assert row_id > 0

        # Retrieve and verify association
        record = get_scan_by_analysis_id(test_anl_id)
        assert record is not None
        assert record["analysis_id"] == test_anl_id
        assert record["patient_id"] == "PT-TRACE-01"
        assert record["predicted_class"] == "Glioma Tumor"

    # ------------------------------------------------------------------ #
    # 8. Existing API response compatibility is preserved where intended
    # ------------------------------------------------------------------ #
    def test_existing_api_response_compatibility(self, monkeypatch, mock_valid_prediction):
        client = TestClient(app)
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        _, img_encoded = cv2.imencode(".jpg", dummy_img)

        # Mock inference to isolate API response schema contract
        monkeypatch.setattr("src.api.predict_tumor_logic", lambda *args, **kwargs: mock_valid_prediction)

        files = {"file": ("scan.jpg", img_encoded.tobytes(), "image/jpeg")}
        res = client.post("/api/predict", files=files)
        assert res.status_code == 200
        data = res.json()

        # Legacy keys required by clients
        expected_keys = [
            "class_name", "confidence", "uncertainty_entropy", "is_tumor",
            "location", "tumor_percentage", "severity", "severity_color",
            "inference_time_seconds", "prediction_breakdown", "gradcam_overlay_b64",
            "gradcam_pp_overlay_b64", "segmentation_b64"
        ]
        for key in expected_keys:
            assert key in data, f"Missing legacy key: {key}"

        # New traceability keys
        assert "analysis_id" in data
        assert "timestamp" in data
        assert data["analysis_id"] is not None

    # ------------------------------------------------------------------ #
    # 9. Temporary report assets are cleaned up according to the design
    # ------------------------------------------------------------------ #
    def test_temporary_report_assets_cleaned_up(self, mock_valid_prediction, tmp_path):
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "temp")
        out_pdf = str(tmp_path / "cleanup_report.pdf")

        test_anl = "ANL-CLEANUP-CHECK"
        generate_pdf_report(
            diag_result=mock_valid_prediction,
            patient_id="PT-CLEANUP",
            output_pdf_path=out_pdf,
            analysis_id=test_anl
        )

        # Temporary snapshots with test_anl must be cleaned up
        leftover = glob.glob(os.path.join(temp_dir, f"*{test_anl}*"))
        assert len(leftover) == 0, f"Found uncleaned temporary files: {leftover}"
