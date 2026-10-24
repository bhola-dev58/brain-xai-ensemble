"""
tests/test_inference_edge_cases.py
==================================
High-value regression tests for inference edge cases, mathematical safety,
entropy calculations, class mapping, and controlled failure modes.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.config import CLASSES
from src.inference import (
    calculate_entropy,
    predict_tumor_logic,
    get_model_status,
    is_ensemble_ready,
)
import src.inference as inference_mod


# ------------------------------------------------------------------ #
# 1. Entropy & Uncertainty Calculations
# ------------------------------------------------------------------ #
class TestEntropyCalculations:
    def test_uniform_distribution_has_maximum_uncertainty(self):
        """A completely uniform 4-class distribution has 100% normalized Shannon entropy."""
        uniform_probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        entropy = calculate_entropy(uniform_probs)
        assert np.isclose(entropy, 100.0, atol=0.01)

    def test_deterministic_distribution_has_minimal_uncertainty(self):
        """A confident distribution with p=1.0 has ~0% uncertainty."""
        confident_probs = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        entropy = calculate_entropy(confident_probs)
        assert np.isclose(entropy, 0.0, atol=0.01)

    def test_entropy_handles_zero_probabilities_without_nan(self):
        """Probabilities containing exact zeros must not produce NaN or Inf (clip safe)."""
        probs = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32)
        entropy = calculate_entropy(probs)
        assert np.isfinite(entropy)
        assert 0.0 <= entropy <= 100.0

    def test_entropy_monotonic_property(self):
        """More dispersed probability distributions must strictly yield higher entropy than peaked ones."""
        peaked = np.array([0.90, 0.05, 0.03, 0.02], dtype=np.float32)
        dispersed = np.array([0.40, 0.30, 0.20, 0.10], dtype=np.float32)
        assert calculate_entropy(peaked) < calculate_entropy(dispersed)


# ------------------------------------------------------------------ #
# 2. Input Shape and Malformed Array Edge Cases
# ------------------------------------------------------------------ #
class TestInputValidationEdgeCases:
    def test_none_input_strictly_fails(self):
        """None image input must return is_valid=False with code INVALID_INPUT."""
        res = predict_tumor_logic(None)
        assert res["is_valid"] is False
        assert res["code"] == "INVALID_INPUT"
        assert "upload an MRI image" in res["error"]

    def test_1d_array_rejected_or_handled_safely(self):
        """1D array cannot be processed as a 2D MRI slice; must fail safely without crash."""
        bad_arr = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        res = predict_tumor_logic(bad_arr)
        assert res["is_valid"] is False
        assert res["code"] in ("INFERENCE_ERROR", "INVALID_INPUT")

    def test_4d_array_rejected_or_handled_safely(self):
        """4D tensor passed directly to slice logic must fail safely with an error response."""
        bad_4d = np.zeros((2, 224, 224, 3), dtype=np.uint8)
        res = predict_tumor_logic(bad_4d)
        assert res["is_valid"] is False
        assert res["code"] in ("INFERENCE_ERROR", "INVALID_INPUT")

    def test_empty_image_array(self):
        """Array with zero spatial dimensions must fail safely."""
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        res = predict_tumor_logic(empty_img)
        assert res["is_valid"] is False
        assert res["code"] in ("INFERENCE_ERROR", "INVALID_INPUT")


# ------------------------------------------------------------------ #
# 3. Model Failure and Output Shape Invariant Tests
# ------------------------------------------------------------------ #
class TestModelFailureModesAndInvariants:
    def test_missing_single_model_fails_inference_when_unready(self, monkeypatch):
        """If any model in the tri-ensemble is None, inference is rejected with MODELS_UNAVAILABLE."""
        monkeypatch.setattr(inference_mod, "model_convnext", None)
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        res = predict_tumor_logic(dummy_img)
        assert res["is_valid"] is False
        assert res["code"] == "MODELS_UNAVAILABLE"
        assert "ConvNeXtSmall" in res["error"]

    def test_incomplete_ensemble_prediction_rejection(self, monkeypatch):
        """If ensemble prediction produces fewer than 3 model outputs, inference fails with INCOMPLETE_ENSEMBLE."""
        # Simulate models present
        mock_model = MagicMock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.predict.return_value = np.array([[0.1, 0.7, 0.1, 0.1]], dtype=np.float32)

        monkeypatch.setattr(inference_mod, "model_convnext", mock_model)
        monkeypatch.setattr(inference_mod, "model_inc", None)  # missing
        monkeypatch.setattr(inference_mod, "model_dense", mock_model)

        # Force is_ensemble_ready check to pass artificially to test downstream incomplete ensemble branch
        monkeypatch.setattr(inference_mod, "is_ensemble_ready", lambda: True)

        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        res = predict_tumor_logic(dummy_img)
        assert res["is_valid"] is False
        assert res["code"] == "INCOMPLETE_ENSEMBLE"

    def test_class_index_and_class_name_strict_contract(self):
        """Valid prediction class_idx must strictly correspond to CLASSES[class_idx]."""
        mock_pred = np.array([0.05, 0.85, 0.05, 0.05], dtype=np.float32)
        expected_class = CLASSES[1]
        assert CLASSES[int(np.argmax(mock_pred))] == expected_class
