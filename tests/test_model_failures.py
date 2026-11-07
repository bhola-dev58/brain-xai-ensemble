"""
Unit and regression tests for model initialization, availability,
failure handling, and health/readiness reporting in BrainTumorXAI.
"""
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.inference as inference
from src.api import app


@pytest.fixture
def dummy_image():
    """Returns a dummy uint8 RGB image simulating an MRI scan."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Draw simple circular pattern for brain contouring
    import cv2
    cv2.circle(img, (112, 112), 80, (200, 200, 200), -1)
    return img


@pytest.fixture
def restore_models():
    """Saves and restores real models in src.inference after tests that mutate them."""
    orig_convnext = inference.model_convnext
    orig_inc = inference.model_inc
    orig_dense = inference.model_dense
    orig_grad_convnext = inference._grad_model_convnext
    orig_grad_dense = inference._grad_model_dense
    yield
    inference.model_convnext = orig_convnext
    inference.model_inc = orig_inc
    inference.model_dense = orig_dense
    inference._grad_model_convnext = orig_grad_convnext
    inference._grad_model_dense = orig_grad_dense


class TestModelAvailabilityAndFailures:
    # ------------------------------------------------------------------ #
    # 1. All three models are available
    # ------------------------------------------------------------------ #
    def test_all_three_models_available(self, restore_models, dummy_image):
        """When all models are loaded, ensemble is declared ready and inference succeeds."""
        if not inference.is_ensemble_ready():
            inference.init_models()
        if not inference.is_ensemble_ready():
            pytest.skip("Tri-Ensemble model weights not available in models/ — requires trained weights.")
        assert inference.is_ensemble_ready() is True
        status = inference.get_model_status()
        assert status["ConvNeXtSmall"] is True
        assert status["InceptionV3"] is True
        assert status["DenseNet121"] is True

        result = inference.predict_tumor_logic(dummy_image)
        assert result["is_valid"] is True
        assert "class_name" in result
        assert "confidence" in result

    # ------------------------------------------------------------------ #
    # 2. All three models are unavailable
    # ------------------------------------------------------------------ #
    def test_all_three_models_unavailable(self, restore_models, dummy_image):
        """When all models are None, returns structured failure without fake predictions."""
        inference.model_convnext = None
        inference.model_inc = None
        inference.model_dense = None

        assert inference.is_ensemble_ready() is False
        status = inference.get_model_status()
        assert not any(status.values())

        result = inference.predict_tumor_logic(dummy_image)
        assert result["is_valid"] is False
        assert result["code"] == "MODELS_UNAVAILABLE"
        assert "Inference unavailable" in result["error"]
        assert "class_name" not in result
        assert "avg_pred" not in result

    # ------------------------------------------------------------------ #
    # 3. Only ConvNeXtSmall is unavailable
    # ------------------------------------------------------------------ #
    def test_only_convnext_unavailable(self, restore_models, dummy_image):
        """Partial ensemble with missing ConvNeXtSmall is rejected."""
        inference.model_convnext = None
        assert inference.is_ensemble_ready() is False
        assert inference.get_model_status()["ConvNeXtSmall"] is False

        result = inference.predict_tumor_logic(dummy_image)
        assert result["is_valid"] is False
        assert result["code"] == "MODELS_UNAVAILABLE"
        assert "ConvNeXtSmall" in result["error"]

    # ------------------------------------------------------------------ #
    # 4. Only InceptionV3 is unavailable
    # ------------------------------------------------------------------ #
    def test_only_inception_unavailable(self, restore_models, dummy_image):
        """Partial ensemble with missing InceptionV3 is rejected."""
        inference.model_inc = None
        assert inference.is_ensemble_ready() is False
        assert inference.get_model_status()["InceptionV3"] is False

        result = inference.predict_tumor_logic(dummy_image)
        assert result["is_valid"] is False
        assert result["code"] == "MODELS_UNAVAILABLE"
        assert "InceptionV3" in result["error"]

    # ------------------------------------------------------------------ #
    # 5. Only DenseNet121 is unavailable
    # ------------------------------------------------------------------ #
    def test_only_densenet_unavailable(self, restore_models, dummy_image):
        """Partial ensemble with missing DenseNet121 is rejected."""
        inference.model_dense = None
        assert inference.is_ensemble_ready() is False
        assert inference.get_model_status()["DenseNet121"] is False

        result = inference.predict_tumor_logic(dummy_image)
        assert result["is_valid"] is False
        assert result["code"] == "MODELS_UNAVAILABLE"
        assert "DenseNet121" in result["error"]

    # ------------------------------------------------------------------ #
    # 6. A required model artifact is missing
    # ------------------------------------------------------------------ #
    def test_missing_model_artifact(self, restore_models, monkeypatch):
        """If an artifact path does not exist, init_models fails safely."""
        real_exists = os.path.exists

        def mock_exists(path):
            if "densenet" in str(path).lower():
                return False
            return real_exists(path)

        monkeypatch.setattr(os.path, "exists", mock_exists)
        success = inference.init_models(force_reload=True)
        assert success is False
        assert inference.model_dense is None
        assert inference.is_ensemble_ready() is False

    # ------------------------------------------------------------------ #
    # 7. Model loading raises an exception
    # ------------------------------------------------------------------ #
    def test_model_loading_exception(self, restore_models, monkeypatch):
        """If load_model raises an error, init_models handles it without crashing."""
        import tensorflow as tf

        def mock_load_model(*args, **kwargs):
            raise RuntimeError("Corrupted HDF5 / Keras artifact header")

        monkeypatch.setattr(tf.keras.models, "load_model", mock_load_model)
        success = inference.init_models(force_reload=True)
        assert success is False
        assert inference.is_ensemble_ready() is False
        assert inference.model_convnext is None

    # ------------------------------------------------------------------ #
    # 8. Invalid input still returns the expected controlled response
    # ------------------------------------------------------------------ #
    def test_invalid_none_input(self):
        """None input returns is_valid=False with code INVALID_INPUT."""
        result = inference.predict_tumor_logic(None)
        assert result["is_valid"] is False
        assert result["error"] == "Please upload an MRI image first."
        assert result["code"] == "INVALID_INPUT"

    # ------------------------------------------------------------------ #
    # 9. A failed inference cannot be marked as a valid diagnosis
    # ------------------------------------------------------------------ #
    def test_failed_inference_never_valid(self, restore_models, dummy_image, monkeypatch):
        """Exceptions during prediction return is_valid=False and generic error."""
        # Force an unexpected exception during model prediction
        mock_model = MagicMock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.predict.side_effect = ValueError("CUDA OOM or tensor execution failure")

        inference.model_convnext = mock_model
        inference.model_inc = mock_model
        inference.model_dense = mock_model

        result = inference.predict_tumor_logic(dummy_image)
        assert result["is_valid"] is False
        assert result["code"] == "INFERENCE_ERROR"
        # Must not leak filesystem paths or stack traces
        assert "An internal error occurred during model inference." in result["error"]
        assert "CUDA" not in result["error"]
        assert "class_name" not in result

    # ------------------------------------------------------------------ #
    # 10. Health and readiness reporting reflects model availability
    # ------------------------------------------------------------------ #
    def test_health_reporting_ready_and_degraded(self, restore_models, dummy_image):
        """API /api/health distinguishes liveness from readiness, and /api/predict returns 503."""
        if not inference.is_ensemble_ready():
            inference.init_models()
        if not inference.is_ensemble_ready():
            pytest.skip("Tri-Ensemble model weights not available in models/ — requires trained weights.")
        client = TestClient(app)

        # 10a: Healthy state
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["ready"] is True
        assert data["models_active"]["ConvNeXtSmall"] is True
        assert data["models_active"]["InceptionV3"] is True
        assert data["models_active"]["DenseNet121"] is True

        # 10b: Degraded state (one model missing)
        inference.model_convnext = None
        res_deg = client.get("/api/health")
        assert res_deg.status_code == 200
        data_deg = res_deg.json()
        assert data_deg["status"] == "degraded"
        assert data_deg["ready"] is False
        assert data_deg["models_active"]["ConvNeXtSmall"] is False
        assert data_deg["models_active"]["InceptionV3"] is True

        # 10c: Predict fails with 503 when models are unavailable
        import cv2
        _, img_encoded = cv2.imencode(".jpg", dummy_image)
        files = {"file": ("test.jpg", img_encoded.tobytes(), "image/jpeg")}
        predict_res = client.post("/api/predict", files=files)
        assert predict_res.status_code == 503
        assert "Inference unavailable" in predict_res.json()["detail"]
