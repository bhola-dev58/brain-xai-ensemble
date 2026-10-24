"""
tests/test_ensemble_and_uncertainty.py
======================================
Regression and unit tests for Task 14:
Auditing and hardening ensemble prediction, weighting, TTA, and uncertainty indicators.

1. Correct output shape and class ordering.
2. Invalid, non-finite, or out-of-range model outputs.
3. Probability-vector validation and softmax conversion.
4. Correct ensemble aggregation with deterministic sample outputs.
5. Missing or failed model behavior (no silent partial ensembles).
6. Correct handling of model weights.
7. Test-Time Augmentation (TTA) aggregation.
8. Failed inference not represented as a valid prediction.
9. Consistency of API and report confidence/uncertainty wording.
10. Shannon entropy calculation, boundaries, and limitations.
"""

import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.config import (
    CLASSES,
    CONVNEXT_VOTE_WEIGHT,
    INCEPTION_VOTE_WEIGHT,
    DENSENET_VOTE_WEIGHT,
    EFFNET_VOTE_WEIGHT,
)
from src.inference import (
    calculate_entropy,
    _validate_model_output,
    _generate_tta_variants,
    _predict_single_model,
    predict_tumor_logic,
)
from src.api import app


# ----------------------------------------------------------------------
# 1. Output shape and class ordering
# ----------------------------------------------------------------------
class TestClassOrderingAndOutputShape:
    def test_class_count_and_order(self):
        """Classes list must have exactly 4 classes in the fixed clinical sequence."""
        assert len(CLASSES) == 4
        assert CLASSES[0] == "No Tumor"
        assert CLASSES[1] == "Glioma Tumor"
        assert CLASSES[2] == "Meningioma Tumor"
        assert CLASSES[3] == "Pituitary Tumor"

    def test_output_vector_length(self, monkeypatch):
        """Ensemble output must produce a 4-element probability vector."""
        mock_pred = np.array([0.70, 0.10, 0.10, 0.10], dtype=np.float32)

        class MockModel:
            input_shape = (None, 224, 224, 3)
            def predict(self, x, verbose=0):
                return np.array([mock_pred])

        monkeypatch.setattr("src.inference.model_convnext", MockModel())
        monkeypatch.setattr("src.inference.model_inc", MockModel())
        monkeypatch.setattr("src.inference.model_dense", MockModel())
        monkeypatch.setattr("src.inference.is_ensemble_ready", lambda: True)

        img = np.full((100, 100, 3), 120, dtype=np.uint8)
        res = predict_tumor_logic(img)

        assert res["is_valid"] is True
        assert len(res["avg_pred"]) == 4
        assert res["avg_pred"].shape == (4,)


# ----------------------------------------------------------------------
# 2. Invalid, non-finite, or out-of-range model outputs
# ----------------------------------------------------------------------
class TestModelOutputValidation:
    def test_valid_probabilities_preserved(self):
        """Valid normalized probabilities pass through intact."""
        raw = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        out = _validate_model_output(raw, "TestModel")
        assert out is not None
        assert np.isclose(np.sum(out), 1.0, atol=1e-5)
        np.testing.assert_allclose(out, raw, atol=1e-5)

    def test_non_finite_nan_inf_rejected(self):
        """Vectors with NaN or Inf return None."""
        raw_nan = np.array([np.nan, 0.3, 0.2, 0.1], dtype=np.float32)
        assert _validate_model_output(raw_nan, "TestModel") is None

        raw_inf = np.array([np.inf, 0.3, 0.2, 0.1], dtype=np.float32)
        assert _validate_model_output(raw_inf, "TestModel") is None

    def test_invalid_length_rejected(self):
        """Output vectors with length != 4 return None."""
        raw_short = np.array([0.5, 0.5], dtype=np.float32)
        assert _validate_model_output(raw_short, "TestModel") is None

        raw_long = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        assert _validate_model_output(raw_long, "TestModel") is None

    def test_none_input_rejected(self):
        """None input returns None."""
        assert _validate_model_output(None, "TestModel") is None

    def test_unnormalized_logits_converted_via_softmax(self):
        """Raw unnormalized logits (including negative numbers) are converted to a valid probability distribution."""
        raw_logits = np.array([2.5, -1.0, 0.5, 1.0], dtype=np.float32)
        out = _validate_model_output(raw_logits, "TestModel")
        assert out is not None
        assert np.isclose(np.sum(out), 1.0, atol=1e-5)
        assert np.all(out >= 0.0)
        # 2.5 was the highest logit -> index 0 must have highest probability
        assert np.argmax(out) == 0


# ----------------------------------------------------------------------
# 3. Correct ensemble aggregation with deterministic sample outputs
# ----------------------------------------------------------------------
class TestDeterministicEnsembleAggregation:
    def test_exact_weighted_voting_calculation(self, monkeypatch):
        """Ensemble probabilities must match the analytic weighted average of 0.45/0.35/0.20."""
        p_cnx = np.array([0.80, 0.10, 0.05, 0.05], dtype=np.float32)
        p_inc = np.array([0.60, 0.20, 0.10, 0.10], dtype=np.float32)
        p_den = np.array([0.50, 0.30, 0.10, 0.10], dtype=np.float32)

        class ModelCnx:
            input_shape = (None, 224, 224, 3)
            def predict(self, x, verbose=0):
                return np.array([p_cnx])

        class ModelInc:
            input_shape = (None, 299, 299, 3)
            def predict(self, x, verbose=0):
                return np.array([p_inc])

        class ModelDen:
            input_shape = (None, 224, 224, 3)
            def predict(self, x, verbose=0):
                return np.array([p_den])

        monkeypatch.setattr("src.inference.model_convnext", ModelCnx())
        monkeypatch.setattr("src.inference.model_inc", ModelInc())
        monkeypatch.setattr("src.inference.model_dense", ModelDen())
        monkeypatch.setattr("src.inference.is_ensemble_ready", lambda: True)

        img = np.full((100, 100, 3), 120, dtype=np.uint8)
        res = predict_tumor_logic(img)

        assert res["is_valid"] is True

        # Expected weights: 0.45 * cnx + 0.35 * inc + 0.20 * den
        expected = 0.45 * p_cnx + 0.35 * p_inc + 0.20 * p_den
        np.testing.assert_allclose(res["avg_pred"], expected, atol=1e-4)

        # Class 0 has highest probability (0.670)
        assert res["class_idx"] == 0
        assert res["class_name"] == "No Tumor"
        assert pytest.approx(res["confidence"], 1e-2) == 67.0


# ----------------------------------------------------------------------
# 4. Missing or failed model behavior (no silent partial ensembles)
# ----------------------------------------------------------------------
class TestMissingOrFailedModels:
    def test_missing_model_fails_explicitly(self, monkeypatch):
        """If one model is missing, predict_tumor_logic rejects with MODELS_UNAVAILABLE."""
        monkeypatch.setattr("src.inference.model_convnext", None)
        monkeypatch.setattr("src.inference.model_inc", MagicMock())
        monkeypatch.setattr("src.inference.model_dense", MagicMock())
        monkeypatch.setattr("src.inference.is_ensemble_ready", lambda: False)

        img = np.full((50, 50, 3), 100, dtype=np.uint8)
        res = predict_tumor_logic(img)

        assert res["is_valid"] is False
        assert res["code"] == "MODELS_UNAVAILABLE"

    def test_runtime_failure_of_single_model_fails_explicitly(self, monkeypatch):
        """If a model raises an exception during prediction, ensemble rejects with INCOMPLETE_ENSEMBLE."""
        class FailingModel:
            input_shape = (None, 224, 224, 3)
            def predict(self, x, verbose=0):
                raise RuntimeError("GPU memory allocation error")

        class GoodModel:
            input_shape = (None, 224, 224, 3)
            def predict(self, x, verbose=0):
                return np.array([[0.25, 0.25, 0.25, 0.25]])

        monkeypatch.setattr("src.inference.model_convnext", FailingModel())
        monkeypatch.setattr("src.inference.model_inc", GoodModel())
        monkeypatch.setattr("src.inference.model_dense", GoodModel())
        monkeypatch.setattr("src.inference.is_ensemble_ready", lambda: True)

        img = np.full((50, 50, 3), 100, dtype=np.uint8)
        res = predict_tumor_logic(img)

        assert res["is_valid"] is False
        assert res["code"] == "INCOMPLETE_ENSEMBLE"
        assert "ConvNeXtSmall" in res["error"]

    def test_nan_output_from_single_model_fails_explicitly(self, monkeypatch):
        """If one model outputs NaNs, ensemble rejects rather than silently poisoning the average."""
        class NanModel:
            input_shape = (None, 224, 224, 3)
            def predict(self, x, verbose=0):
                return np.array([[np.nan, 0.0, 0.0, 0.0]])

        class GoodModel:
            input_shape = (None, 224, 224, 3)
            def predict(self, x, verbose=0):
                return np.array([[0.25, 0.25, 0.25, 0.25]])

        monkeypatch.setattr("src.inference.model_convnext", GoodModel())
        monkeypatch.setattr("src.inference.model_inc", NanModel())
        monkeypatch.setattr("src.inference.model_dense", GoodModel())
        monkeypatch.setattr("src.inference.is_ensemble_ready", lambda: True)

        img = np.full((50, 50, 3), 100, dtype=np.uint8)
        res = predict_tumor_logic(img)

        assert res["is_valid"] is False
        assert res["code"] == "INCOMPLETE_ENSEMBLE"
        assert "InceptionV3" in res["error"]


# ----------------------------------------------------------------------
# 5. Handling of model weights
# ----------------------------------------------------------------------
class TestModelWeightsIntegrity:
    def test_weights_sum_to_one(self):
        """Ensemble voting weights must sum exactly to 1.0."""
        total = CONVNEXT_VOTE_WEIGHT + INCEPTION_VOTE_WEIGHT + DENSENET_VOTE_WEIGHT
        assert pytest.approx(total, 1e-6) == 1.0

    def test_weights_strictly_positive(self):
        """Active backbone weights must be positive and follow clinical distribution."""
        assert CONVNEXT_VOTE_WEIGHT == 0.45
        assert INCEPTION_VOTE_WEIGHT == 0.35
        assert DENSENET_VOTE_WEIGHT == 0.20
        assert EFFNET_VOTE_WEIGHT == 0.00


# ----------------------------------------------------------------------
# 6. Test-Time Augmentation (TTA) aggregation
# ----------------------------------------------------------------------
class TestTTAAggregation:
    def test_generate_tta_variants(self):
        """5-pass TTA creates 5 deterministic geometric variants of input."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        img[20:40, 20:40] = 200  # asymmetry to distinguish flip/rot
        variants = _generate_tta_variants(img, num_passes=5)

        assert len(variants) == 5
        for v in variants:
            assert v.shape == img.shape
            assert v.dtype == np.uint8

        # Pass 0 must be original
        np.testing.assert_array_equal(variants[0], img)
        # Pass 1 is flipped horizontally
        np.testing.assert_array_equal(variants[1], cv2.flip(img, 1))

    def test_predict_single_model_with_tta(self):
        """_predict_single_model with TTA averages across passes and normalizes to 1.0."""
        call_count = 0
        def mock_predict(inp, verbose=0):
            nonlocal call_count
            call_count += 1
            # Return slightly varying predictions per pass
            return np.array([[0.6 + 0.01 * call_count, 0.2, 0.1, 0.1 - 0.01 * call_count]])

        mock_model = MagicMock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.predict = mock_predict

        img = np.full((100, 100, 3), 100, dtype=np.uint8)
        pred = _predict_single_model(mock_model, img, (224, 224), "TestModel", use_tta=True, tta_passes=5)

        assert pred is not None
        assert call_count == 5
        assert np.isclose(np.sum(pred), 1.0, atol=1e-5)
        assert len(pred) == 4

    def test_predict_tumor_logic_with_tta(self, monkeypatch):
        """predict_tumor_logic with use_tta=True passes tta_enabled flag in response."""
        p_mock = np.array([0.1, 0.8, 0.05, 0.05], dtype=np.float32)
        class MockModel:
            input_shape = (None, 224, 224, 3)
            def predict(self, x, verbose=0):
                return np.array([p_mock])

        monkeypatch.setattr("src.inference.model_convnext", MockModel())
        monkeypatch.setattr("src.inference.model_inc", MockModel())
        monkeypatch.setattr("src.inference.model_dense", MockModel())
        monkeypatch.setattr("src.inference.is_ensemble_ready", lambda: True)

        img = np.full((80, 80, 3), 120, dtype=np.uint8)
        res = predict_tumor_logic(img, use_tta=True, tta_passes=5)

        assert res["is_valid"] is True
        assert res["tta_enabled"] is True
        assert res["tta_passes"] == 5


# ----------------------------------------------------------------------
# 7. Shannon Entropy / Uncertainty outputs and limits
# ----------------------------------------------------------------------
class TestShannonEntropyCalculations:
    def test_one_hot_distribution_has_zero_entropy(self):
        """Deterministic one-hot probability vector has exactly 0.0% normalized entropy."""
        p_deterministic = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        entropy = calculate_entropy(p_deterministic)
        assert entropy == 0.0

    def test_uniform_distribution_has_100_percent_entropy(self):
        """Uniform distribution across 4 classes has exactly 100.0% normalized entropy."""
        p_uniform = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        entropy = calculate_entropy(p_uniform)
        assert pytest.approx(entropy, 1e-2) == 100.0

    def test_intermediate_entropy_monotonicity(self):
        """Higher concentration produces strictly lower entropy than diffuse distribution."""
        p_concentrated = np.array([0.90, 0.04, 0.03, 0.03], dtype=np.float32)
        p_diffuse = np.array([0.40, 0.30, 0.20, 0.10], dtype=np.float32)

        e_conc = calculate_entropy(p_concentrated)
        e_diff = calculate_entropy(p_diffuse)

        assert 0.0 < e_conc < e_diff < 100.0

    def test_entropy_handles_non_finite_safely(self):
        """Non-finite or empty inputs return 0.0 without throwing exceptions."""
        assert calculate_entropy(np.array([np.nan, 0.5, 0.5, 0.0])) == 0.0
        assert calculate_entropy(None) == 0.0
        assert calculate_entropy(np.array([])) == 0.0


# ----------------------------------------------------------------------
# 8. Consistency of API Confidence and Uncertainty Wording
# ----------------------------------------------------------------------
class TestAPIConfidenceWording:
    def test_api_predict_returns_metric_descriptors(self, monkeypatch):
        """API response must contain explicit uncalibrated confidence and dispersion descriptors."""
        p_mock = np.array([0.05, 0.90, 0.03, 0.02], dtype=np.float32)
        class MockModel:
            input_shape = (None, 224, 224, 3)
            def predict(self, x, verbose=0):
                return np.array([p_mock])

        monkeypatch.setattr("src.inference.model_convnext", MockModel())
        monkeypatch.setattr("src.inference.model_inc", MockModel())
        monkeypatch.setattr("src.inference.model_dense", MockModel())
        monkeypatch.setattr("src.inference.is_ensemble_ready", lambda: True)

        client = TestClient(app)
        import cv2
        img = np.full((100, 100, 3), 100, dtype=np.uint8)
        _, encoded = cv2.imencode(".png", img)

        res = client.post("/api/predict", files={"file": ("brain.png", encoded.tobytes(), "image/png")})
        assert res.status_code == 200
        data = res.json()

        assert "confidence" in data
        assert data["confidence_metric"] == "Ensemble Softmax Probability (Uncalibrated)"
        assert "uncertainty_entropy" in data
        assert data["uncertainty_metric"] == "Normalized Shannon Entropy of Softmax Distribution"
        assert "individual_model_breakdown" in data
