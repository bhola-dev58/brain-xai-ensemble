"""
tests/test_mri_input_validation.py
==================================
Comprehensive regression tests for MRI input validation and preprocessing:
1. Valid grayscale input conversion.
2. Valid RGB input processing.
3. Empty and corrupted files.
4. Unsupported image types.
5. Unexpected dimensions and channel counts.
6. NaN or infinite pixel values.
7. Very small and very large input rejection.
8. DICOM decoding failure and multi-frame handling.
9. Correct tensor shape and dtype.
10. Correct normalization range [0.0, 1.0] and model-specific preprocessing.
11. Preprocessing failure cannot return a valid diagnosis.
12. Existing valid-input behavior remains compatible.
"""

import io
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException
from PIL import Image

from src.api import app, _read_upload_as_numpy
from src.config import CLASSES, IMG_SIZE_CLASSIFY, IMG_SIZE_INCEPTION
from src.dicom_reader import dicom_to_rgb, _normalize_to_uint8
from src.inference import predict_tumor_logic, is_ensemble_ready
from src.processor import extract_brain_region, create_segmentation
from src.security import (
    validate_image_content,
    validate_image_dimensions,
    detect_file_format,
)
import src.inference as inference_mod


@pytest.fixture
def mock_ensemble():
    """Provides mocked models for deterministic unit testing without requiring heavy weights."""
    mock_model_224 = MagicMock()
    mock_model_224.input_shape = (None, 224, 224, 3)
    mock_model_224.predict.return_value = np.array([[0.05, 0.85, 0.05, 0.05]], dtype=np.float32)

    mock_model_299 = MagicMock()
    mock_model_299.input_shape = (None, 299, 299, 3)
    mock_model_299.predict.return_value = np.array([[0.05, 0.85, 0.05, 0.05]], dtype=np.float32)

    with patch.object(inference_mod, "model_convnext", mock_model_224), \
         patch.object(inference_mod, "model_inc", mock_model_299), \
         patch.object(inference_mod, "model_dense", mock_model_224), \
         patch.object(inference_mod, "_grad_model_dense", None), \
         patch.object(inference_mod, "_grad_model_convnext", None):
        yield {
            "convnext": mock_model_224,
            "inception": mock_model_299,
            "densenet": mock_model_224,
        }


# ------------------------------------------------------------------ #
# 1 & 2. Grayscale, RGB, and Channel Conversion
# ------------------------------------------------------------------ #
class TestChannelStandardization:
    def test_valid_grayscale_2d_input_converted_to_rgb(self, mock_ensemble):
        """A 2D grayscale MRI slice (H, W) must be safely converted to 3-channel RGB uint8."""
        gray_img = np.full((128, 128), 150, dtype=np.uint8)
        res = predict_tumor_logic(gray_img)

        assert res["is_valid"] is True
        assert res["img"].ndim == 3
        assert res["img"].shape[2] == 3
        assert res["gradcam_overlay"].shape[2] == 3
        assert res["segmentation_img"].shape[2] == 3

    def test_valid_rgb_input_processed_directly(self, mock_ensemble):
        """A standard 3-channel RGB image (H, W, 3) must be processed directly."""
        rgb_img = np.full((128, 128, 3), 150, dtype=np.uint8)
        res = predict_tumor_logic(rgb_img)

        assert res["is_valid"] is True
        assert res["img"].shape[2] == 3
        assert res["confidence"] > 0

    def test_valid_rgba_input_alpha_channel_dropped(self, mock_ensemble):
        """A 4-channel RGBA input must have its alpha channel stripped to preserve RGB tensor shape."""
        rgba_img = np.full((128, 128, 4), 200, dtype=np.uint8)
        res = predict_tumor_logic(rgba_img)

        assert res["is_valid"] is True
        assert res["img"].shape[2] == 3

    def test_single_channel_3d_input_converted_to_rgb(self, mock_ensemble):
        """A 3D image with a single channel (H, W, 1) must be converted to (H, W, 3)."""
        single_ch = np.full((128, 128, 1), 100, dtype=np.uint8)
        res = predict_tumor_logic(single_ch)

        assert res["is_valid"] is True
        assert res["img"].shape[2] == 3


# ------------------------------------------------------------------ #
# 3 & 4. Empty, Corrupted, and Unsupported Files
# ------------------------------------------------------------------ #
class TestFileFormatAndIntegrityRejection:
    def test_empty_file_upload_rejected_with_400(self):
        """Uploading a 0-byte file must be rejected with HTTP 400."""
        client = TestClient(app)
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        resp = client.post("/api/predict", files=files)
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_corrupted_image_file_rejected_with_422(self):
        """A file with valid image magic bytes but corrupted payload must be rejected with HTTP 422."""
        corrupt_jpeg = b"\xff\xd8\xff\xe0" + b"CORRUPTED_BYTES_NOT_VALID_JPEG" * 10
        client = TestClient(app)
        files = {"file": ("scan.jpg", corrupt_jpeg, "image/jpeg")}
        resp = client.post("/api/predict", files=files)
        assert resp.status_code == 422
        assert "corrupted" in resp.json()["detail"].lower()

    def test_unsupported_file_type_rejected_with_415(self):
        """A file with an unsupported signature (e.g. PDF or GIF) must be rejected with HTTP 415."""
        pdf_bytes = b"%PDF-1.5 fake pdf content"
        client = TestClient(app)
        files = {"file": ("scan.pdf", pdf_bytes, "application/pdf")}
        resp = client.post("/api/predict", files=files)
        assert resp.status_code == 415
        assert "unsupported" in resp.json()["detail"].lower()

    def test_empty_array_rejected_by_inference(self):
        """Passing an empty array (0 pixels) to predict_tumor_logic must return is_valid=False."""
        empty_arr = np.zeros((0, 0, 3), dtype=np.uint8)
        res = predict_tumor_logic(empty_arr)
        assert res["is_valid"] is False
        assert res["code"] == "INVALID_INPUT"


# ------------------------------------------------------------------ #
# 5 & 6. Unexpected Dimensions and Non-Finite Pixels
# ------------------------------------------------------------------ #
class TestDimensionAndValueSafety:
    def test_1d_array_rejected(self):
        """1D array input must be rejected without crashing."""
        res = predict_tumor_logic(np.array([1, 2, 3, 4], dtype=np.uint8))
        assert res["is_valid"] is False
        assert res["code"] == "INVALID_INPUT"

    def test_4d_array_rejected(self):
        """4D batch array passed to single-slice logic must be rejected with INVALID_INPUT."""
        res = predict_tumor_logic(np.zeros((2, 128, 128, 3), dtype=np.uint8))
        assert res["is_valid"] is False
        assert res["code"] == "INVALID_INPUT"

    def test_unsupported_channel_count_rejected(self):
        """5-channel image array must be rejected."""
        res = predict_tumor_logic(np.zeros((128, 128, 5), dtype=np.uint8))
        assert res["is_valid"] is False
        assert res["code"] == "INVALID_INPUT"

    def test_nan_pixels_rejected(self):
        """Image containing NaN values must be rejected to prevent silent false diagnoses."""
        nan_img = np.full((128, 128, 3), np.nan, dtype=np.float32)
        res = predict_tumor_logic(nan_img)
        assert res["is_valid"] is False
        assert res["code"] == "INVALID_INPUT"
        assert "non-finite" in res["error"].lower()

    def test_infinite_pixels_rejected(self):
        """Image containing Inf values must be rejected."""
        inf_img = np.full((128, 128, 3), np.inf, dtype=np.float32)
        res = predict_tumor_logic(inf_img)
        assert res["is_valid"] is False
        assert res["code"] == "INVALID_INPUT"
        assert "non-finite" in res["error"].lower()

    def test_dimensions_too_small_rejected(self):
        """Images smaller than 10x10 must be rejected."""
        tiny_img = np.zeros((8, 8, 3), dtype=np.uint8)
        res = predict_tumor_logic(tiny_img)
        assert res["is_valid"] is False
        assert res["code"] == "INVALID_INPUT"
        assert "too small" in res["error"].lower()

    def test_dimensions_too_large_rejected(self):
        """Images larger than 5000x5000 must be rejected."""
        huge_img = np.zeros((5001, 100, 3), dtype=np.uint8)
        res = predict_tumor_logic(huge_img)
        assert res["is_valid"] is False
        assert res["code"] == "INVALID_INPUT"
        assert "exceed maximum" in res["error"].lower()


# ------------------------------------------------------------------ #
# 7 & 8. DICOM Decoding and Multi-Frame Edge Cases
# ------------------------------------------------------------------ #
class TestDicomEdgeCases:
    def test_dicom_missing_pixel_array_raises_value_error(self):
        """A DICOM dataset with missing pixel_array must raise ValueError in dicom_to_rgb."""
        class CorruptDicom:
            @property
            def pixel_array(self):
                raise AttributeError("PixelData missing or corrupted transfer syntax")

        with pytest.raises(ValueError, match="Failed to read or decompress"):
            dicom_to_rgb(CorruptDicom())

    def test_dicom_with_nan_pixels_raises_value_error(self):
        """A DICOM dataset with NaN pixel values must raise ValueError."""
        mock_ds = MagicMock()
        mock_ds.pixel_array = np.array([[np.nan, 100.0], [200.0, 300.0]], dtype=np.float32)
        mock_ds.PhotometricInterpretation = "MONOCHROME2"
        with pytest.raises(ValueError, match="non-finite"):
            dicom_to_rgb(mock_ds)

    def test_dicom_multi_frame_volume_extracts_central_slice(self):
        """A 3D multi-frame DICOM (Frames, H, W) must select the central slice and return (H, W, 3)."""
        mock_ds = MagicMock()
        # 5 frames of 64x64 slices
        frames = np.zeros((5, 64, 64), dtype=np.uint16)
        frames[2, 30:35, 30:35] = 1000  # distinct feature in center frame
        mock_ds.pixel_array = frames
        mock_ds.PhotometricInterpretation = "MONOCHROME2"

        rgb = dicom_to_rgb(mock_ds)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8
        # Center frame has the peak intensity
        assert rgb[32, 32, 0] > 0

    def test_dicom_monochrome1_inversion(self):
        """MONOCHROME1 DICOM images must be inverted so bone/tissue is bright and air is dark."""
        mock_ds = MagicMock()
        # Dark pixel in MONOCHROME1 means high absorption / tissue
        arr = np.full((32, 32), 0, dtype=np.uint8)
        mock_ds.pixel_array = arr
        mock_ds.PhotometricInterpretation = "MONOCHROME1"

        rgb = dicom_to_rgb(mock_ds)
        assert rgb.shape == (32, 32, 3)
        # 0 inverted is 255
        assert rgb[0, 0, 0] == 255

    def test_normalize_to_uint8_flat_array(self):
        """A completely uniform pixel array must normalize to all zeros without ZeroDivisionError."""
        flat_arr = np.full((50, 50), 42, dtype=np.float32)
        norm = _normalize_to_uint8(flat_arr)
        assert norm.shape == (50, 50)
        assert np.all(norm == 0)


# ------------------------------------------------------------------ #
# 9 & 10. Model Tensor Shape, Dtype, and Normalization
# ------------------------------------------------------------------ #
class TestTensorPreprocessingContract:
    def test_model_input_tensors_match_training_specifications(self, mock_ensemble):
        """
        Verify that model inputs sent to DenseNet121, ConvNeXtSmall, and InceptionV3
        strictly match training specifications:
        - DenseNet121: (1, 224, 224, 3), float32, in range [0.0, 1.0]
        - ConvNeXtSmall: (1, 224, 224, 3), float32, in range [0.0, 1.0]
        - InceptionV3: (1, 299, 299, 3), float32, in range [0.0, 1.0]
        """
        test_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        res = predict_tumor_logic(test_img)
        assert res["is_valid"] is True

        # Check ConvNeXt call argument
        cnx_call_args = mock_ensemble["convnext"].predict.call_args[0][0]
        assert cnx_call_args.shape == (1, 224, 224, 3)
        assert cnx_call_args.dtype == np.float32
        assert 0.0 <= cnx_call_args.min() and cnx_call_args.max() <= 1.0

        # Check InceptionV3 call argument
        inc_call_args = mock_ensemble["inception"].predict.call_args[0][0]
        assert inc_call_args.shape == (1, 299, 299, 3)
        assert inc_call_args.dtype == np.float32
        assert 0.0 <= inc_call_args.min() and inc_call_args.max() <= 1.0


# ------------------------------------------------------------------ #
# 11 & 12. Preprocessing Failures & Diagnostic Safety
# ------------------------------------------------------------------ #
class TestPreprocessingSafetyContract:
    def test_invalid_input_never_produces_diagnosis_or_confidence(self):
        """When input validation fails, inference must NEVER fabricate a prediction, class, or confidence score."""
        for bad_input in [None, "invalid_str", np.zeros((0, 0, 3)), np.full((50, 50), np.nan)]:
            res = predict_tumor_logic(bad_input)
            assert res["is_valid"] is False
            assert "class_name" not in res
            assert "confidence" not in res
            assert "segmentation_img" not in res
            assert res["code"] == "INVALID_INPUT"

    def test_extract_brain_region_resilience_to_non_finite_values(self):
        """extract_brain_region must gracefully return original input if non-finite values are encountered."""
        nan_arr = np.full((64, 64, 3), np.nan, dtype=np.float32)
        res = extract_brain_region(nan_arr)
        assert res is nan_arr

    def test_create_segmentation_resilience_to_non_finite_heatmap(self):
        """create_segmentation must handle NaN/Inf heatmaps without crashing, returning an empty tumor mask."""
        orig_img = np.full((64, 64, 3), 128, dtype=np.uint8)
        nan_heatmap = np.full((32, 32), np.nan, dtype=np.float32)
        overlay, mask, pct = create_segmentation(nan_heatmap, orig_img)
        assert overlay.shape == orig_img.shape
        assert mask.shape == (64, 64)
        assert pct == 0.0
