"""
tests/test_reporting_and_db_resilience.py
=========================================
Tests for database schema migration, query resilience, report generator
fallback execution, temporary file cleanup, and unicode handling.
"""

import os
import sqlite3
import numpy as np
import pytest

from src.database import (
    init_db,
    log_scan,
    get_scan_by_analysis_id,
    get_scans_paginated,
    get_scan_count,
    _get_connection,
)
import src.database as db_mod
from src.report_generator import (
    generate_pdf_report,
    _generate_matplotlib_pdf,
    _save_temp_image,
)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Provides an isolated SQLite database path for testing schema and queries."""
    db_file = str(tmp_path / "test_patient_history.db")
    monkeypatch.setattr(db_mod, "DB_PATH", db_file)
    init_db()
    return db_file


# ------------------------------------------------------------------ #
# 1. Database Schema Migration and Query Robustness
# ------------------------------------------------------------------ #
class TestDatabaseResilience:
    def test_schema_migration_adds_missing_analysis_id_column(self, tmp_path, monkeypatch):
        """When an older schema table exists without analysis_id, init_db() migrates it safely."""
        legacy_db = str(tmp_path / "legacy_patient_history.db")
        monkeypatch.setattr(db_mod, "DB_PATH", legacy_db)

        # Create old schema table without analysis_id
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
        conn.commit()
        conn.close()

        # Run init_db which must detect missing column and ALTER TABLE
        init_db()

        # Verify analysis_id column was added
        conn = sqlite3.connect(legacy_db)
        cursor = conn.execute("PRAGMA table_info(scans)")
        cols = [r[1] for r in cursor.fetchall()]
        conn.close()

        assert "analysis_id" in cols

    def test_database_insertion_with_special_characters(self, temp_db):
        """log_scan safely handles patient names and filenames with quotes, apostrophes, and unicode."""
        row_id = log_scan(
            predicted_class="Glioma Tumor",
            confidence_pct=95.5,
            is_tumor=True,
            severity="High",
            tumor_area_pct=5.2,
            location="Left Frontal",
            inference_time=0.35,
            patient_id="PT-SPECIAL-01",
            patient_name="O'Connor \"The Specialist\" Müller & Co",
            scan_filename="scan's_image;--drop.jpg",
            analysis_id="ANL-SPEC-12345"
        )
        assert row_id > 0

        # Retrieve and verify exact string fidelity without corruption
        rec = get_scan_by_analysis_id("ANL-SPEC-12345")
        assert rec is not None
        assert rec["patient_name"] == "O'Connor \"The Specialist\" Müller & Co"
        assert rec["scan_filename"] == "scan's_image;--drop.jpg"

    def test_empty_database_query_handling(self, temp_db):
        """Querying an empty database returns empty list and zero count without errors."""
        assert get_scan_count() == 0
        assert get_scans_paginated(limit=10, offset=0) == []
        assert get_scan_by_analysis_id("ANL-NONEXISTENT") is None


# ------------------------------------------------------------------ #
# 2. Report Generation Robustness and Fallback Execution
# ------------------------------------------------------------------ #
class TestReportGenerationRobustness:
    def test_matplotlib_fallback_pdf_generation(self, tmp_path):
        """Directly tests the Matplotlib fallback PDF generator to ensure it renders a valid PDF."""
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_diag = {
            "img": dummy_img,
            "gradcam_overlay": dummy_img,
            "segmentation_img": dummy_img,
            "classes": ["No Tumor", "Glioma", "Meningioma", "Pituitary"],
            "avg_pred": [0.05, 0.85, 0.05, 0.05],
            "class_name": "Glioma",
            "confidence": 85.0,
            "location": "Left Frontal Lobe",
            "tumor_percentage": 4.5,
        }

        out_pdf = str(tmp_path / "matplotlib_fallback_report.pdf")
        pdf_res = _generate_matplotlib_pdf(
            diag_result=dummy_diag,
            patient_id="PT-FALLBACK-01",
            patient_name="Fallback Patient",
            patient_age="42",
            patient_gender="Male",
            pdf_path=out_pdf,
            analysis_id="ANL-FALLBACK-99",
            timestamp="2026-10-24 14:00:00"
        )

        assert os.path.exists(pdf_res)
        assert os.path.getsize(pdf_res) > 500  # Non-empty PDF
        # Verify PDF magic header
        with open(pdf_res, "rb") as f:
            header = f.read(4)
        assert header == b"%PDF"

    def test_report_generation_with_unicode_patient_metadata(self, tmp_path):
        """Report generation safely handles international patient names with accents."""
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_diag = {
            "is_valid": True,
            "analysis_id": "ANL-UNICODE-01",
            "timestamp": "2026-10-24 12:00:00",
            "img": dummy_img,
            "gradcam_overlay": dummy_img,
            "segmentation_img": dummy_img,
            "classes": ["No Tumor", "Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor"],
            "avg_pred": [0.02, 0.94, 0.02, 0.02],
            "pred_dict": {
                "ConvNeXtSmall": [0.01, 0.95, 0.02, 0.02],
                "InceptionV3": [0.03, 0.93, 0.02, 0.02],
                "DenseNet121": [0.02, 0.94, 0.02, 0.02],
            },
            "class_name": "Glioma Tumor",
            "confidence": 94.0,
            "uncertainty": 10.0,
            "is_tumor": True,
            "location": "Left Temporal Lobe",
            "tumor_percentage": 3.8,
            "severity": "Moderate",
        }

        out_pdf = str(tmp_path / "unicode_report.pdf")
        pdf_path = generate_pdf_report(
            diag_result=dummy_diag,
            patient_id="PT-UNICODE",
            patient_name="José González-Müller",
            patient_age="58",
            patient_gender="M",
            output_pdf_path=out_pdf,
            analysis_id="ANL-UNICODE-01"
        )

        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 1000

    def test_temp_image_creation_and_cleanup(self, tmp_path):
        """_save_temp_image creates a file that can be cleanly removed."""
        dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
        temp_file = _save_temp_image(dummy_img, "test_snapshot_cleanup.png")
        assert os.path.exists(temp_file)
        os.remove(temp_file)
        assert not os.path.exists(temp_file)
