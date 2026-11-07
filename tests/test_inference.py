"""
Integration tests for src/inference.py

Loads the real model weights and runs a full pipeline prediction
using one of the provided test scans. This validates end-to-end
correctness after any code changes.

NOTE: These tests will take ~30–60 seconds because they load
      the TensorFlow models into memory on the first run.
"""
import os
import numpy as np
import pytest
import cv2

# Ensure project root is in path when running tests from any CWD
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import predict_tumor_logic, is_ensemble_ready, init_models


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
_TEST_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "test_images",
)

@pytest.fixture(scope="module")
def real_mri_scan():
    """Loads a real MRI scan once per test module for efficiency."""
    path = os.path.join(_TEST_IMAGES_DIR, "Tr-me_0025.jpg")
    if not os.path.exists(path):
        pytest.skip(f"Test image not found: {path}")
    img = cv2.imread(path)
    assert img is not None, f"cv2.imread returned None for: {path}"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="module")
def prediction_result(real_mri_scan):
    """Runs inference once and caches result for all tests in this module."""
    if not is_ensemble_ready():
        init_models()
    if not is_ensemble_ready():
        pytest.skip("Tri-Ensemble model weights not available in models/ — skipping live weight inference tests.")
    return predict_tumor_logic(real_mri_scan)


# ------------------------------------------------------------------ #
# Output shape & type validation
# ------------------------------------------------------------------ #
class TestPredictTumorLogic:
    def test_valid_flag_is_true(self, prediction_result):
        assert prediction_result["is_valid"] is True

    def test_class_name_is_valid(self, prediction_result):
        valid_classes = ["No Tumor", "Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor"]
        assert prediction_result["class_name"] in valid_classes

    def test_confidence_is_percentage(self, prediction_result):
        conf = prediction_result["confidence"]
        assert 0.0 <= conf <= 100.0, f"Confidence out of range: {conf}"

    def test_is_tumor_is_boolean(self, prediction_result):
        assert isinstance(prediction_result["is_tumor"], bool)

    def test_inference_time_is_positive(self, prediction_result):
        assert prediction_result["inference_time"] > 0

    def test_avg_pred_sums_to_one(self, prediction_result):
        total = float(np.sum(prediction_result["avg_pred"]))
        assert abs(total - 1.0) < 1e-4, f"Softmax probabilities don't sum to 1: {total}"

    def test_four_class_probabilities(self, prediction_result):
        assert len(prediction_result["avg_pred"]) == 4

    def test_gradcam_overlay_is_rgb_image(self, prediction_result):
        overlay = prediction_result["gradcam_overlay"]
        assert isinstance(overlay, np.ndarray)
        assert overlay.ndim == 3
        assert overlay.shape[2] == 3

    def test_segmentation_image_is_rgb(self, prediction_result):
        seg = prediction_result["segmentation_img"]
        assert isinstance(seg, np.ndarray)
        assert seg.ndim == 3
        assert seg.shape[2] == 3

    def test_severity_is_valid_string(self, prediction_result):
        valid_severities = {"High", "Moderate", "Low", "Borderline", "Uncertain"}
        assert prediction_result["severity"] in valid_severities

    def test_severity_color_is_hex(self, prediction_result):
        color = prediction_result["severity_color"]
        assert color.startswith("#"), f"Expected hex color, got: {color}"

    def test_class_idx_matches_class_name(self, prediction_result):
        classes = prediction_result["classes"]
        idx = prediction_result["class_idx"]
        assert classes[idx] == prediction_result["class_name"]


# ------------------------------------------------------------------ #
# Edge case: None input
# ------------------------------------------------------------------ #
class TestEdgeCases:
    def test_none_input_returns_invalid(self):
        result = predict_tumor_logic(None)
        assert result["is_valid"] is False
        assert "error" in result
        assert len(result["error"]) > 0

    def test_none_input_error_is_string(self):
        result = predict_tumor_logic(None)
        assert isinstance(result["error"], str)

    def test_float_image_is_normalized(self):
        """Passing a float image should be safely clipped and cast to uint8."""
        if not is_ensemble_ready():
            pytest.skip("Tri-Ensemble model weights not available in models/ — skipping live weight inference tests.")
        float_img = np.random.uniform(0, 1, (224, 224, 3)).astype(np.float32) * 255
        result = predict_tumor_logic(float_img)
        assert result["is_valid"] is True
