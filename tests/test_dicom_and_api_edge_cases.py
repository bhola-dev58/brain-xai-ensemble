"""
tests/test_dicom_and_api_edge_cases.py
======================================
Regression tests for DICOM medical imaging conversion, metadata extraction,
and API parameter/boundary validation.
"""

import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.api import app
from src.dicom_reader import (
    dicom_to_rgb,
    extract_dicom_metadata,
    _apply_windowing,
    _normalize_to_uint8,
    _parse_age,
)


@pytest.fixture
def client():
    return TestClient(app)


# ------------------------------------------------------------------ #
# 1. DICOM Image Normalization and Conversion
# ------------------------------------------------------------------ #
class TestDicomReaderConversions:
    def test_normalize_to_uint8_handles_16bit_range(self):
        """16-bit CT/MRI HU pixel values must be properly normalized to [0, 255] uint8."""
        arr_16 = np.array([0, 1024, 2048, 4096], dtype=np.uint16)
        out = _normalize_to_uint8(arr_16)
        assert out.dtype == np.uint8
        assert out[0] == 0
        assert out[-1] == 255

    def test_normalize_to_uint8_handles_flat_array(self):
        """Flat array with arr_min == arr_max must return an all-zero uint8 array without div-by-zero."""
        flat = np.full((10, 10), 500, dtype=np.int32)
        out = _normalize_to_uint8(flat)
        assert out.dtype == np.uint8
        assert np.all(out == 0)

    def test_apply_windowing_clips_and_normalizes(self):
        """DICOM soft-tissue window center and width must properly clip pixel values."""
        ds = MagicMock()
        ds.WindowCenter = 50.0
        ds.WindowWidth = 100.0  # Window: [0, 100]

        raw_pixels = np.array([-50, 0, 50, 100, 200], dtype=np.float32)
        windowed = _apply_windowing(raw_pixels, ds)
        assert windowed.dtype == np.uint8
        assert windowed[0] == 0    # clipped to 0
        assert windowed[-1] == 255  # clipped to 100 -> normalized to 255

    def test_dicom_to_rgb_monochrome2(self):
        """Standard MONOCHROME2 dataset converts to 3-channel RGB uint8 array."""
        ds = MagicMock()
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.pixel_array = np.full((64, 64), 100, dtype=np.uint16)

        rgb = dicom_to_rgb(ds)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8

    def test_dicom_to_rgb_monochrome1_inversion(self):
        """MONOCHROME1 (0 is white, 255 is black) must invert pixels so 0 is dark background."""
        ds = MagicMock()
        ds.PhotometricInterpretation = "MONOCHROME1"
        # Pixel array with black background (high value in MONOCHROME1)
        ds.pixel_array = np.array([[0, 255]], dtype=np.uint8)

        rgb = dicom_to_rgb(ds)
        assert rgb.shape == (1, 2, 3)
        # After inversion, original 0 becomes 255, original 255 becomes 0
        assert rgb[0, 0, 0] == 255
        assert rgb[0, 1, 0] == 0

    def test_dicom_to_rgb_missing_pixel_data_raises_value_error(self):
        """DICOM dataset missing pixel_array raises AttributeError or ValueError."""
        ds = MagicMock(spec=[])  # no pixel_array
        with pytest.raises((AttributeError, ValueError)):
            dicom_to_rgb(ds)


# ------------------------------------------------------------------ #
# 2. DICOM Metadata Extraction and Parsing
# ------------------------------------------------------------------ #
class TestDicomMetadataExtraction:
    def test_parse_age_formats(self):
        """Parses standard DICOM 4-character age strings (e.g. '045Y', '006M')."""
        assert _parse_age("045Y") == "045"
        assert _parse_age("006M") == "006 months"
        assert _parse_age("52") == "52"
        assert _parse_age(None) == "N/A"

    def test_extract_dicom_metadata_fields(self):
        """Correctly extracts and formats clinical metadata fields from DICOM dataset."""
        ds = MagicMock()
        ds.PatientID = "PT-MR-9001"
        ds.PatientName = "DOE^JOHN"
        ds.PatientAge = "035Y"
        ds.PatientSex = "M"
        ds.StudyDate = "20261024"
        ds.Modality = "MR"
        ds.SliceThickness = "4.0"
        ds.PixelSpacing = ["0.5", "0.5"]
        ds.InstitutionName = "General Hospital"

        meta = extract_dicom_metadata(ds)
        assert meta["patient_id"] == "PT-MR-9001"
        assert meta["patient_name"] == "DOE^JOHN"
        assert meta["patient_age"] == "035"
        assert meta["patient_gender"] == "Male"
        assert meta["study_date"] == "24/10/2026"
        assert meta["modality"] == "MR"
        assert meta["slice_thickness"] == "4.00 mm"
        assert meta["pixel_spacing"] == "0.50 x 0.50 mm"


# ------------------------------------------------------------------ #
# 3. API Boundary and Validation Tests
# ------------------------------------------------------------------ #
class TestApiBoundariesAndValidation:
    def test_history_endpoint_validation_bounds(self, client):
        """Pagination parameters strictly enforce bounds: limit (1-100) and offset (>=0)."""
        # limit < 1 fails validation
        r1 = client.get("/api/history?limit=0")
        assert r1.status_code == 422

        # limit > 100 fails validation
        r2 = client.get("/api/history?limit=101")
        assert r2.status_code == 422

        # offset < 0 fails validation
        r3 = client.get("/api/history?offset=-1")
        assert r3.status_code == 422

        # Valid pagination succeeds
        r4 = client.get("/api/history?limit=25&offset=0")
        assert r4.status_code == 200

    def test_predict_endpoint_rejects_unsupported_text_upload(self, client):
        """Uploading plain text or executable file as MRI image returns 415 or 422."""
        fake_txt = io.BytesIO(b"Not an image file. Plain text payload.")
        res = client.post(
            "/api/predict",
            files={"file": ("fake.png", fake_txt, "image/png")}
        )
        assert res.status_code in (415, 422)
        assert "unsupported" in res.text.lower() or "signature" in res.text.lower() or "decode" in res.text.lower()

    def test_reports_endpoint_rejects_path_traversal_attempts(self, client):
        """Malformed analysis identifiers with path traversal symbols are rejected with 400 Bad Request."""
        res = client.get("/api/reports/ANL-..-passwd")
        assert res.status_code == 400
        assert "Invalid analysis identifier format" in res.json()["detail"]

    def test_reports_endpoint_returns_404_for_unknown_valid_id(self, client):
        """Valid format but non-existent analysis ID returns controlled 404."""
        res = client.get("/api/reports/ANL-NONEXIST99")
        assert res.status_code == 404
        assert "Report not found" in res.json()["detail"]
