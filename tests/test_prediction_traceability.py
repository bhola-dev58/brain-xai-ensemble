"""
tests/test_prediction_traceability.py
=====================================
Targeted regression tests for Task 12: Traceable and Reproducible Prediction Records.
Validates:
  1. Unique analysis IDs (format, randomness, retention).
  2. Consistent UTC timestamps.
  3. Model and pipeline provenance (architectures, weights, SHA-256 digests).
  4. Missing model-version metadata handling.
  5. Successful analysis persistence and retrieval.
  6. Failed inference never recorded as a successful prediction.
  7. Consistency between analysis history and generated reports.
  8. Database failure handling and resilience.
  9. Privacy: sensitive info excluded from ordinary logs and public error responses.
 10. Backward compatibility with existing database records and API consumers.
 11. Traceability retrieval endpoint (/api/provenance/{analysis_id}).
"""

import io
import json
import os
import re
import sqlite3
import numpy as np
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from src.api import app
from src.config import PIPELINE_VERSION, CLASSES
import src.database as db_mod
from src.database import (
    init_db,
    log_scan,
    get_scan_by_analysis_id,
    get_scan_provenance,
    get_scans_paginated,
    get_scan_count,
)
from src.inference import (
    predict_tumor_logic,
    get_model_provenance,
    _compute_file_sha256,
)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Provides a fresh isolated SQLite database for each test."""
    test_db_path = str(tmp_path / "test_patient_history.db")
    monkeypatch.setattr(db_mod, "DB_PATH", test_db_path)
    init_db()
    return test_db_path


@pytest.fixture
def client():
    return TestClient(app)


# =====================================================================
# 1. Unique Analysis IDs
# =====================================================================
class TestAnalysisIdentifierIntegrity:
    def test_unique_analysis_id_format_and_uniqueness(self):
        """Auto-generated analysis IDs must follow the format ANL-[A-F0-9]{10} and be unique."""
        ids = set()
        pattern = re.compile(r"^ANL-[A-F0-9]{10}$")

        for _ in range(25):
            res = predict_tumor_logic(None)
            aid = res["analysis_id"]
            assert pattern.match(aid), f"Analysis ID {aid} does not match expected pattern"
            ids.add(aid)

        assert len(ids) == 25, "Analysis IDs collided; expected 25 distinct identifiers"

    def test_custom_analysis_id_retained(self):
        """When an analysis ID is explicitly provided, it is preserved throughout the pipeline."""
        custom_id = "ANL-CUSTOM-TRACE-99"
        res = predict_tumor_logic(None, analysis_id=custom_id)
        assert res["analysis_id"] == custom_id


# =====================================================================
# 2. Consistent Timestamps
# =====================================================================
class TestTimestampConsistency:
    def test_timestamp_uses_documented_utc_format(self):
        """Analysis timestamps must use documented UTC format 'YYYY-MM-DD HH:MM:SS UTC'."""
        res = predict_tumor_logic(None)
        ts_str = res["timestamp"]

        assert ts_str.endswith(" UTC"), f"Timestamp {ts_str} does not include UTC timezone label"
        # Parse the datetime portion and verify it is close to current UTC time
        raw_dt = ts_str.replace(" UTC", "")
        dt = datetime.strptime(raw_dt, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        now_utc = datetime.now(timezone.utc)
        diff_seconds = abs((now_utc - dt).total_seconds())
        assert diff_seconds < 10, f"Timestamp {ts_str} deviated by {diff_seconds}s from current UTC time"


# =====================================================================
# 3. Model and Pipeline Provenance
# =====================================================================
class TestModelAndPipelineProvenance:
    def test_model_provenance_contains_all_architectures_and_weights(self):
        """Provenance record must accurately detail pipeline version, architectures, and voting weights."""
        prov = get_model_provenance()

        assert prov["pipeline_version"] == PIPELINE_VERSION
        assert prov["classes"] == CLASSES
        assert "models" in prov

        models = prov["models"]
        for arch in ("ConvNeXtSmall", "InceptionV3", "DenseNet121"):
            assert arch in models
            m_info = models[arch]
            assert "artifact" in m_info
            assert "weight" in m_info
            assert "input_size" in m_info
            assert m_info["weight"] > 0.0

        # Check weight sum equals 1.0 (45% + 35% + 20%)
        total_w = sum(m["weight"] for m in models.values())
        assert pytest.approx(total_w, 1e-4) == 1.0

    def test_model_provenance_sanitizes_internal_file_paths(self):
        """Artifact identifiers must be basenames only, preventing leak of server directories."""
        prov = get_model_provenance()
        for arch, m_info in prov["models"].items():
            artifact_name = m_info["artifact"]
            assert "/" not in artifact_name, f"Internal path separator detected in artifact name: {artifact_name}"
            assert "\\" not in artifact_name


# =====================================================================
# 4. Missing Model Metadata Handling
# =====================================================================
class TestMissingModelMetadata:
    def test_compute_file_sha256_handles_missing_file_safely(self):
        """Non-existent weight file returns None safely without raising an exception."""
        digest = _compute_file_sha256("/path/to/missing_model_weights.keras")
        assert digest is None


# =====================================================================
# 5. Successful Analysis Persistence and Retrieval
# =====================================================================
class TestSuccessfulAnalysisPersistence:
    def test_log_scan_persists_full_provenance_and_retrieves_cleanly(self, temp_db):
        """Completed analyses store pipeline version and provenance JSON, retrievable by analysis ID."""
        test_aid = "ANL-PERSIST-001"
        prov_dict = get_model_provenance()

        row_id = log_scan(
            predicted_class="Glioma Tumor",
            confidence_pct=94.5,
            is_tumor=True,
            severity="High",
            tumor_area_pct=6.2,
            location="Left Frontal Lobe",
            inference_time=0.42,
            patient_id="PT-PERSIST-01",
            patient_name="Persisted Patient",
            scan_filename="brain_scan.dcm",
            input_type="DICOM",
            analysis_id=test_aid,
            status="COMPLETED",
            pipeline_version=PIPELINE_VERSION,
            model_provenance=prov_dict,
        )
        assert row_id > 0

        # Direct row check
        record = get_scan_by_analysis_id(test_aid)
        assert record is not None
        assert record["status"] == "COMPLETED"
        assert record["pipeline_version"] == PIPELINE_VERSION
        assert record["error_code"] is None

        # Provenance helper check
        prov_record = get_scan_provenance(test_aid)
        assert prov_record is not None
        assert prov_record["analysis_id"] == test_aid
        assert prov_record["status"] == "COMPLETED"
        assert isinstance(prov_record["model_provenance"], dict)
        assert prov_record["model_provenance"]["pipeline_version"] == PIPELINE_VERSION


# =====================================================================
# 6. Failed Inference Never Recorded as Successful
# =====================================================================
class TestFailedInferenceHonesty:
    def test_failed_inference_never_records_as_success(self, temp_db):
        """
        Failed operations logged with status='FAILED' must never have is_tumor=True
        or be misrepresented as successful predictions.
        """
        fail_aid = "ANL-FAIL-HONEST-01"
        row_id = log_scan(
            predicted_class="Glioma Tumor",  # attempt to erroneously pass tumor class
            confidence_pct=99.0,
            is_tumor=True,
            severity="High",
            tumor_area_pct=10.0,
            location="Frontal",
            inference_time=0.0,
            analysis_id=fail_aid,
            status="FAILED",
            error_code="INVALID_INPUT",
        )
        assert row_id > 0

        rec = get_scan_by_analysis_id(fail_aid)
        assert rec["status"] == "FAILED"
        assert rec["error_code"] == "INVALID_INPUT"
        # Critical safety invariants for failed analyses:
        assert rec["is_tumor"] == 0
        assert rec["confidence_pct"] == 0.0

    def test_api_predict_logs_failed_request_honestly(self, client, temp_db, monkeypatch):
        """API /api/predict logs failed inference attempts as status='FAILED' in the database."""
        fail_aid = "ANL-FAIL-API-TEST"
        monkeypatch.setattr(
            "src.api.predict_tumor_logic",
            lambda *args, **kwargs: {
                "is_valid": False,
                "status": "FAILED",
                "code": "MODELS_UNAVAILABLE",
                "error": "Model weights missing or inactive.",
                "analysis_id": fail_aid,
                "timestamp": "2026-10-24 19:30:00 UTC",
                "pipeline_version": PIPELINE_VERSION,
            }
        )

        import cv2
        dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".png", dummy_img)
        resp = client.post("/api/predict", files={"file": ("scan.png", encoded.tobytes(), "image/png")})
        assert resp.status_code == 503

        # Scans table should contain the failed record
        scans = get_scans_paginated(limit=5)
        failed_records = [s for s in scans if s.get("status") == "FAILED"]
        assert len(failed_records) >= 1
        assert failed_records[0]["analysis_id"] == fail_aid
        assert failed_records[0]["error_code"] == "MODELS_UNAVAILABLE"
        assert failed_records[0]["is_tumor"] == 0


# =====================================================================
# 7. Consistency Between Analysis History and Generated Reports
# =====================================================================
class TestHistoryAndReportConsistency:
    def test_report_generation_records_matching_analysis_id_and_provenance(self, client, temp_db, monkeypatch, tmp_path):
        """When a PDF report is generated, the database record links to the exact PDF and analysis ID."""
        test_aid = "ANL-RPT-CONSIST-01"
        mock_pred = {
            "is_valid": True,
            "status": "COMPLETED",
            "analysis_id": test_aid,
            "timestamp": "2026-10-24 18:00:00 UTC",
            "pipeline_version": PIPELINE_VERSION,
            "provenance": get_model_provenance(),
            "class_name": "Pituitary Tumor",
            "confidence": 91.2,
            "uncertainty": 0.22,
            "is_tumor": True,
            "inference_time": 0.35,
            "tumor_percentage": 3.8,
            "location": "Sellar / Suprasellar",
            "severity": "Moderate",
            "severity_color": "#f59e0b",
            "img": np.zeros((64, 64, 3), dtype=np.uint8),
            "gradcam_overlay": np.zeros((64, 64, 3), dtype=np.uint8),
            "gradcam_pp_overlay": np.zeros((64, 64, 3), dtype=np.uint8),
            "segmentation_img": np.zeros((64, 64, 3), dtype=np.uint8),
            "classes": CLASSES,
            "avg_pred": [0.05, 0.02, 0.02, 0.91],
        }

        monkeypatch.setattr("src.api.predict_tumor_logic", lambda *args, **kwargs: mock_pred)

        dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)
        import cv2
        _, encoded = cv2.imencode(".png", dummy_img)
        files = {"file": ("scan.png", encoded.tobytes(), "image/png")}
        data = {
            "patient_id": "PT-CONSIST-99",
            "patient_name": "Consistent Patient",
            "analysis_id": test_aid,
        }

        res = client.post("/api/report", files=files, data=data)
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/pdf"

        # Verify DB entry matches the report exactly
        rec = get_scan_by_analysis_id(test_aid)
        assert rec is not None
        assert rec["analysis_id"] == test_aid
        assert rec["predicted_class"] == "Pituitary Tumor"
        assert rec["pdf_path"] is not None
        assert os.path.exists(rec["pdf_path"])


# =====================================================================
# 8. Database Failure Handling
# =====================================================================
class TestDatabaseFailureHandling:
    def test_log_scan_returns_negative_on_sqlite_error(self, monkeypatch):
        """When SQLite encounters an error, log_scan returns -1 without raising an uncaught exception."""
        def mock_error_conn():
            raise sqlite3.OperationalError("Database disk image is malformed")

        monkeypatch.setattr(db_mod, "_get_connection", mock_error_conn)
        row_id = log_scan(
            predicted_class="Glioma Tumor",
            confidence_pct=95.0,
            is_tumor=True,
            severity="High",
            tumor_area_pct=5.0,
            location="Frontal",
            inference_time=0.2,
        )
        assert row_id == -1


# =====================================================================
# 9. Sensitive Information Protection in Logs & Errors
# =====================================================================
class TestPrivacyInProvenanceAndLogs:
    def test_provenance_retrieval_does_not_contain_phi(self, temp_db):
        """The provenance endpoint returns technical metadata only, without patient demographics."""
        test_aid = "ANL-PHI-FREE-01"
        log_scan(
            predicted_class="No Tumor",
            confidence_pct=98.0,
            is_tumor=False,
            severity="Low",
            tumor_area_pct=0.0,
            location="N/A",
            inference_time=0.15,
            patient_id="SECRET-PATIENT-ID-999",
            patient_name="Top Secret Patient",
            analysis_id=test_aid,
            model_provenance=get_model_provenance(),
        )

        prov = get_scan_provenance(test_aid)
        prov_json_str = json.dumps(prov)

        assert "SECRET-PATIENT-ID" not in prov_json_str
        assert "Top Secret Patient" not in prov_json_str


# =====================================================================
# 10. Backward Compatibility with Existing Database Records
# =====================================================================
class TestBackwardCompatibility:
    def test_legacy_database_migration_adds_new_columns(self, tmp_path, monkeypatch):
        """Legacy tables missing new provenance columns are migrated cleanly on init_db()."""
        legacy_db = str(tmp_path / "legacy_v1_patient_history.db")
        monkeypatch.setattr(db_mod, "DB_PATH", legacy_db)

        # Create old schema table missing analysis_id, status, error_code, pipeline_version, model_provenance
        conn = sqlite3.connect(legacy_db)
        conn.execute("""
            CREATE TABLE scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence_pct REAL NOT NULL,
                is_tumor INTEGER NOT NULL
            )
        """)
        conn.execute("INSERT INTO scans (timestamp, predicted_class, confidence_pct, is_tumor) VALUES ('2026-01-01', 'Glioma Tumor', 90.0, 1)")
        conn.commit()
        conn.close()

        # Run init_db which must apply schema migrations
        init_db()

        conn = sqlite3.connect(legacy_db)
        cursor = conn.execute("PRAGMA table_info(scans)")
        cols = {r[1] for r in cursor.fetchall()}
        conn.close()

        for expected_col in ("analysis_id", "status", "error_code", "pipeline_version", "model_provenance"):
            assert expected_col in cols, f"Missing migrated column: {expected_col}"

        # Existing record still readable
        legacy_rows = get_scans_paginated()
        assert len(legacy_rows) == 1
        assert legacy_rows[0]["predicted_class"] == "Glioma Tumor"
        assert legacy_rows[0]["status"] == "COMPLETED"

    def test_api_predict_maintains_all_legacy_keys(self, client, monkeypatch):
        """API /api/predict continues to return all keys expected by existing UI/API consumers."""
        mock_pred = {
            "is_valid": True,
            "status": "COMPLETED",
            "pipeline_version": PIPELINE_VERSION,
            "provenance": get_model_provenance(),
            "analysis_id": "ANL-LEGACY-KEY-CHECK",
            "timestamp": "2026-10-24 19:00:00 UTC",
            "class_name": "Glioma Tumor",
            "confidence": 95.0,
            "uncertainty": 0.15,
            "is_tumor": True,
            "inference_time": 0.25,
            "tumor_percentage": 4.5,
            "location": "Right Parietal",
            "severity": "High",
            "severity_color": "#ef4444",
            "img": np.zeros((32, 32, 3), dtype=np.uint8),
            "gradcam_overlay": np.zeros((32, 32, 3), dtype=np.uint8),
            "gradcam_pp_overlay": np.zeros((32, 32, 3), dtype=np.uint8),
            "segmentation_img": np.zeros((32, 32, 3), dtype=np.uint8),
            "classes": CLASSES,
            "avg_pred": [0.02, 0.95, 0.02, 0.01],
        }
        monkeypatch.setattr("src.api.predict_tumor_logic", lambda *args, **kwargs: mock_pred)

        import cv2
        _, encoded = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
        res = client.post("/api/predict", files={"file": ("scan.jpg", encoded.tobytes(), "image/jpeg")})
        assert res.status_code == 200
        data = res.json()

        legacy_keys = [
            "class_name", "confidence", "uncertainty_entropy", "is_tumor",
            "location", "tumor_percentage", "severity", "severity_color",
            "inference_time_seconds", "prediction_breakdown", "gradcam_overlay_b64",
            "gradcam_pp_overlay_b64", "segmentation_b64"
        ]
        for k in legacy_keys:
            assert k in data, f"Legacy response key missing: {k}"

        # Traceability additions
        assert data["analysis_id"] == "ANL-LEGACY-KEY-CHECK"
        assert data["status"] == "COMPLETED"
        assert data["pipeline_version"] == PIPELINE_VERSION
        assert "provenance" in data


# =====================================================================
# 11. Traceability API Endpoint (/api/provenance/{analysis_id})
# =====================================================================
class TestProvenanceEndpoint:
    def test_get_provenance_endpoint_success_and_404(self, client, temp_db):
        """GET /api/provenance/{analysis_id} returns 200 with provenance or 404 if not found."""
        test_aid = "ANL-PROV-EP-01"
        log_scan(
            predicted_class="Meningioma Tumor",
            confidence_pct=88.5,
            is_tumor=True,
            severity="Moderate",
            tumor_area_pct=4.0,
            location="Parasagittal",
            inference_time=0.3,
            analysis_id=test_aid,
            model_provenance=get_model_provenance(),
        )

        # 1. Existing ID returns 200
        res200 = client.get(f"/api/provenance/{test_aid}")
        assert res200.status_code == 200
        body = res200.json()
        assert body["analysis_id"] == test_aid
        assert body["status"] == "COMPLETED"
        assert body["model_provenance"]["pipeline_version"] == PIPELINE_VERSION

        # 2. Non-existent ID returns 404
        res404 = client.get("/api/provenance/ANL-NONEXISTENT")
        assert res404.status_code == 404
        assert "not found" in res404.json()["detail"].lower()
