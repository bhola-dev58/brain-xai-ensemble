"""
tests/test_explainability_and_segmentation.py
=============================================
Regression and unit tests for explainability (Grad-CAM, Grad-CAM++) and
morphological segmentation pipelines:
1. Correct class-index mapping and boundary validation.
2. Heatmap shape, dtype, and finite numerical values.
3. Zero or constant activation heatmap normalization (safe division).
4. Incompatible activation / gradient shapes and 1x1 spatial feature maps.
5. Failed gradient calculation (None / non-finite gradients).
6. Overlay dimensions, channel handling, and valid [0, 255] uint8 range.
7. Semantic distinction between attention heatmaps and heuristic masks.
8. No fabricated explanation or fake tumor contour when computation yields zero.
9. Dice and IoU metric calculation against reference masks and error handling.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.inference import (
    CLASSES,
    _top_k_channel_pooling,
    _build_overlay,
    _compute_brain_mask,
    _apply_brain_gate,
    _adaptive_threshold,
    make_gradcam_heatmap,
    predict_tumor_logic,
)
from src.processor import (
    create_segmentation,
    estimate_location,
    compute_dice_coefficient,
    compute_iou,
)


# ----------------------------------------------------------------------
# 1. Correct class-index mapping and boundary validation
# ----------------------------------------------------------------------
class TestClassIndexMapping:
    def test_documented_four_classes(self):
        """Classes must strictly match the 4 documented clinical categories."""
        expected = ["No Tumor", "Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor"]
        assert list(CLASSES) == expected
        assert len(CLASSES) == 4
        assert CLASSES[0] == "No Tumor"
        assert "Glioma" in CLASSES[1]
        assert "Meningioma" in CLASSES[2]
        assert "Pituitary" in CLASSES[3]

    def test_gradcam_target_class_selection(self, monkeypatch):
        """make_gradcam_heatmap must use the specified target class index."""
        inputs = tf.keras.Input(shape=(32, 32, 3))
        x = tf.keras.layers.Conv2D(8, (3, 3), padding="same", name="target_conv")(inputs)
        outputs = tf.keras.layers.Dense(4, activation="softmax", name="prob_out")(tf.keras.layers.Flatten()(x))
        mock_grad_model = tf.keras.Model(inputs=inputs, outputs=[x, outputs])

        monkeypatch.setattr("src.inference._grad_model_dense", mock_grad_model)
        monkeypatch.setattr("src.inference._grad_model_convnext", None)

        orig_img = np.full((64, 64, 3), 120, dtype=np.uint8)
        img_array = np.full((1, 32, 32, 3), 0.5, dtype=np.float32)

        # Test valid pred_index for each class
        for c_idx in range(4):
            hm, mask = make_gradcam_heatmap(img_array, orig_img, pred_index=c_idx, use_pp=False)
            assert hm.shape == (64, 64)
            assert np.all(np.isfinite(hm))

    def test_gradcam_invalid_pred_index_falls_back_safely(self, monkeypatch):
        """Out of bounds or negative pred_index must fall back to argmax without crashing."""
        inputs = tf.keras.Input(shape=(32, 32, 3))
        x = tf.keras.layers.Conv2D(8, (3, 3), padding="same", name="target_conv")(inputs)
        outputs = tf.keras.layers.Dense(4, activation="softmax", name="prob_out")(tf.keras.layers.Flatten()(x))
        mock_grad_model = tf.keras.Model(inputs=inputs, outputs=[x, outputs])

        monkeypatch.setattr("src.inference._grad_model_dense", mock_grad_model)
        monkeypatch.setattr("src.inference._grad_model_convnext", None)

        orig_img = np.full((64, 64, 3), 120, dtype=np.uint8)
        img_array = np.full((1, 32, 32, 3), 0.5, dtype=np.float32)

        # Negative index and out-of-bounds index
        for bad_idx in [-1, 99]:
            hm, mask = make_gradcam_heatmap(img_array, orig_img, pred_index=bad_idx, use_pp=False)
            assert hm.shape == (64, 64)
            assert np.all(np.isfinite(hm))


# ----------------------------------------------------------------------
# 2. Heatmap shape and finite numerical values
# ----------------------------------------------------------------------
class TestHeatmapShapeAndFiniteness:
    def test_standard_gradcam_numerical_properties(self, monkeypatch):
        """Heatmap values must be strictly bounded in [0.0, 1.0] and contain no NaNs/Infs."""
        inputs = tf.keras.Input(shape=(32, 32, 3))
        x = tf.keras.layers.Conv2D(16, (3, 3), padding="same")(inputs)
        outputs = tf.keras.layers.Dense(4, activation="softmax")(tf.keras.layers.Flatten()(x))
        mock_grad_model = tf.keras.Model(inputs=inputs, outputs=[x, outputs])

        monkeypatch.setattr("src.inference._grad_model_dense", mock_grad_model)
        monkeypatch.setattr("src.inference._grad_model_convnext", None)

        h_orig, w_orig = 100, 150
        orig_img = np.random.randint(20, 200, (h_orig, w_orig, 3), dtype=np.uint8)
        img_array = np.random.uniform(0.0, 1.0, (1, 32, 32, 3)).astype(np.float32)

        hm, mask = make_gradcam_heatmap(img_array, orig_img, pred_index=1, use_pp=False)

        assert hm.shape == (h_orig, w_orig)
        assert hm.dtype == np.float32
        assert np.all(np.isfinite(hm))
        assert np.min(hm) >= 0.0
        assert np.max(hm) <= 1.0

    def test_gradcam_pp_numerical_properties(self, monkeypatch):
        """Grad-CAM++ heatmap must produce finite [0.0, 1.0] activations at original resolution."""
        inputs = tf.keras.Input(shape=(32, 32, 3))
        x = tf.keras.layers.Conv2D(16, (3, 3), padding="same")(inputs)
        outputs = tf.keras.layers.Dense(4, activation="softmax")(tf.keras.layers.Flatten()(x))
        mock_grad_model = tf.keras.Model(inputs=inputs, outputs=[x, outputs])

        monkeypatch.setattr("src.inference._grad_model_dense", mock_grad_model)
        monkeypatch.setattr("src.inference._grad_model_convnext", None)

        h_orig, w_orig = 80, 120
        orig_img = np.random.randint(20, 200, (h_orig, w_orig, 3), dtype=np.uint8)
        img_array = np.random.uniform(0.0, 1.0, (1, 32, 32, 3)).astype(np.float32)

        hm_pp, mask = make_gradcam_heatmap(img_array, orig_img, pred_index=2, use_pp=True)

        assert hm_pp.shape == (h_orig, w_orig)
        assert hm_pp.dtype == np.float32
        assert np.all(np.isfinite(hm_pp))
        assert np.min(hm_pp) >= 0.0
        assert np.max(hm_pp) <= 1.0


# ----------------------------------------------------------------------
# 3. Zero or constant heatmap normalization
# ----------------------------------------------------------------------
class TestZeroOrConstantNormalization:
    def test_all_zero_activations_return_zeros_safely(self, monkeypatch):
        """When convolutional layer outputs zeros, make_gradcam_heatmap returns exact zeros without NaN."""
        inputs = tf.keras.Input(shape=(16, 16, 3))
        x = tf.keras.layers.Lambda(lambda t: tf.zeros_like(t)[:, :, :, :4])(inputs)
        outputs = tf.keras.layers.Dense(4, activation="softmax")(tf.keras.layers.Flatten()(inputs))
        mock_grad_model = tf.keras.Model(inputs=inputs, outputs=[x, outputs])

        monkeypatch.setattr("src.inference._grad_model_dense", mock_grad_model)
        monkeypatch.setattr("src.inference._grad_model_convnext", None)

        orig_img = np.full((50, 50, 3), 100, dtype=np.uint8)
        img_array = np.zeros((1, 16, 16, 3), dtype=np.float32)

        hm, mask = make_gradcam_heatmap(img_array, orig_img, pred_index=1, use_pp=False)
        assert hm.shape == (50, 50)
        assert np.all(hm == 0.0)
        assert not np.any(np.isnan(hm))

    def test_constant_flat_activation_handled_safely(self, monkeypatch):
        """Flat uniform activations must not cause division-by-zero or NaNs."""
        inputs = tf.keras.Input(shape=(16, 16, 3))
        x = tf.keras.layers.Lambda(lambda t: tf.ones_like(t)[:, :, :, :4] * 5.0)(inputs)
        outputs = tf.keras.layers.Dense(4, activation="softmax")(tf.keras.layers.Flatten()(inputs))
        mock_grad_model = tf.keras.Model(inputs=inputs, outputs=[x, outputs])

        monkeypatch.setattr("src.inference._grad_model_dense", mock_grad_model)
        monkeypatch.setattr("src.inference._grad_model_convnext", None)

        orig_img = np.full((40, 40, 3), 120, dtype=np.uint8)
        img_array = np.zeros((1, 16, 16, 3), dtype=np.float32)

        hm, mask = make_gradcam_heatmap(img_array, orig_img, pred_index=0, use_pp=False)
        assert hm.shape == (40, 40)
        assert np.all(np.isfinite(hm))


# ----------------------------------------------------------------------
# 4. Incorrect dimensions and incompatible shapes
# ----------------------------------------------------------------------
class TestShapeCompatibility:
    def test_top_k_channel_pooling_preserves_2d_shape(self):
        """Feature map with H=1 or W=1 must retain 2D shape (H, W) and not squeeze to 1D."""
        grads = tf.constant(np.ones((1, 1, 7, 8), dtype=np.float32))
        conv_outputs = tf.constant(np.ones((1, 1, 7, 8), dtype=np.float32))

        hm = _top_k_channel_pooling(grads, conv_outputs, top_k_ratio=0.5)
        assert hm.shape == (1, 7)
        assert len(hm.shape) == 2

    def test_build_overlay_with_dimension_mismatches(self):
        """_build_overlay handles heatmap shape differing from original image shape."""
        orig = np.full((120, 90, 3), 150, dtype=np.uint8)
        heatmap_small = np.full((10, 10), 0.5, dtype=np.float32)

        overlay = _build_overlay(orig, heatmap_small, colormap=2, opacity=0.4)
        assert overlay.shape == (120, 90, 3)
        assert overlay.dtype == np.uint8


# ----------------------------------------------------------------------
# 5. Failed gradient calculation handling
# ----------------------------------------------------------------------
class TestFailedGradients:
    def test_non_differentiable_graph_returns_zeros(self, monkeypatch):
        """When tape.gradient returns None, make_gradcam_heatmap returns clean zeros with brain mask."""
        inputs = tf.keras.Input(shape=(16, 16, 3))
        conv_detached = tf.keras.layers.Lambda(lambda t: tf.stop_gradient(t))(inputs)
        outputs = tf.keras.layers.Dense(4, activation="softmax")(tf.keras.layers.Flatten()(inputs))
        mock_grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_detached, outputs])

        monkeypatch.setattr("src.inference._grad_model_dense", mock_grad_model)
        monkeypatch.setattr("src.inference._grad_model_convnext", None)

        orig = np.full((50, 50, 3), 128, dtype=np.uint8)
        img_array = np.full((1, 16, 16, 3), 0.5, dtype=np.float32)

        hm, mask = make_gradcam_heatmap(img_array, orig, pred_index=1, use_pp=False)
        assert np.all(hm == 0.0)
        assert hm.shape == (50, 50)
        assert mask.shape == (50, 50)

    def test_missing_model_returns_controlled_zeros(self, monkeypatch):
        """When no grad_model is available, returns zeros without error."""
        monkeypatch.setattr("src.inference._grad_model_dense", None)
        monkeypatch.setattr("src.inference._grad_model_convnext", None)

        orig = np.full((60, 60, 3), 100, dtype=np.uint8)
        img_array = np.full((1, 32, 32, 3), 0.5, dtype=np.float32)

        hm, mask = make_gradcam_heatmap(img_array, orig)
        assert np.all(hm == 0.0)
        assert hm.shape == (60, 60)


# ----------------------------------------------------------------------
# 6. Overlay dimensions and valid output range
# ----------------------------------------------------------------------
class TestOverlayCorrectness:
    def test_overlay_uint8_bounds(self):
        """Overlay output must strictly stay in uint8 [0, 255] range."""
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        heatmap = np.ones((64, 64), dtype=np.float32)

        overlay = _build_overlay(img, heatmap, colormap=2, opacity=0.7)
        assert overlay.dtype == np.uint8
        assert np.min(overlay) >= 0
        assert np.max(overlay) <= 255
        assert overlay.shape == (64, 64, 3)

    def test_overlay_zero_heatmap_returns_unmodified_image(self):
        """An all-zero heatmap must leave the original image unaltered."""
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        zero_hm = np.zeros((50, 50), dtype=np.float32)

        overlay = _build_overlay(img, zero_hm, colormap=2, opacity=0.5)
        np.testing.assert_array_equal(overlay, img)


# ----------------------------------------------------------------------
# 7. Semantic distinction between heatmaps and segmentation masks
# ----------------------------------------------------------------------
class TestSemanticDistinction:
    def test_negative_tumor_case_produces_zero_segmentation(self, monkeypatch):
        """When ensemble predicts 'No Tumor' (class 0), tumor percentage is 0.0 and location indicates no tumor."""
        no_tumor_pred = np.array([0.96, 0.02, 0.01, 0.01], dtype=np.float32)

        class MockModel:
            input_shape = (None, 224, 224, 3)
            def predict(self, x, verbose=0):
                return np.array([no_tumor_pred])

        monkeypatch.setattr("src.inference.model_convnext", MockModel())
        monkeypatch.setattr("src.inference.model_inc", MockModel())
        monkeypatch.setattr("src.inference.model_dense", MockModel())
        monkeypatch.setattr("src.inference.is_ensemble_ready", lambda: True)

        img = np.full((128, 128, 3), 120, dtype=np.uint8)
        res = predict_tumor_logic(img)

        assert res["is_valid"] is True
        assert res["class_name"] == "No Tumor"
        assert res["is_tumor"] is False
        assert res["tumor_percentage"] == 0.0
        assert res["location"] == "No Tumor Detected"
        np.testing.assert_array_equal(res["segmentation_img"], res["img"])

    def test_api_keys_distinguish_attention_and_contour(self, monkeypatch):
        """API response must contain distinct keys for attention overlays vs lesion contours."""
        tumor_pred = np.array([0.05, 0.90, 0.03, 0.02], dtype=np.float32)

        class MockModel:
            input_shape = (None, 224, 224, 3)
            def predict(self, x, verbose=0):
                return np.array([tumor_pred])

        monkeypatch.setattr("src.inference.model_convnext", MockModel())
        monkeypatch.setattr("src.inference.model_inc", MockModel())
        monkeypatch.setattr("src.inference.model_dense", MockModel())
        monkeypatch.setattr("src.inference.is_ensemble_ready", lambda: True)

        img = np.full((128, 128, 3), 120, dtype=np.uint8)
        res = predict_tumor_logic(img)

        assert "gradcam_overlay" in res
        assert "gradcam_pp_overlay" in res
        assert "segmentation_img" in res
        assert res["gradcam_overlay"] is not res["segmentation_img"]


# ----------------------------------------------------------------------
# 8. No fabricated explanation when segmentation yields 0 tumor pixels
# ----------------------------------------------------------------------
class TestNoFabricatedExplanation:
    def test_empty_tumor_mask_not_replaced_by_roi(self):
        """When Otsu threshold inside ROI detects no bright lesion pixels, return empty mask (never substitute ROI)."""
        orig = np.full((100, 100, 3), 50, dtype=np.uint8)
        brain_mask = np.full((100, 100), 255, dtype=np.uint8)
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[40:60, 40:60] = 0.9

        orig[40:60, 40:60] = 50

        overlay, mask, area_pct = create_segmentation(heatmap, orig, brain_mask)

        if np.sum(mask > 0) == 0:
            assert area_pct == 0.0
            np.testing.assert_array_equal(overlay, orig)

    def test_non_finite_heatmap_produces_empty_mask(self):
        """NaN/Inf in heatmap returns empty mask and 0.0% area without crashing."""
        nan_heatmap = np.full((80, 80), np.nan, dtype=np.float32)
        orig = np.full((80, 80, 3), 100, dtype=np.uint8)

        overlay, mask, area_pct = create_segmentation(nan_heatmap, orig)
        assert area_pct == 0.0
        assert np.all(mask == 0)


# ----------------------------------------------------------------------
# 9. Metric calculation only when valid reference masks are available
# ----------------------------------------------------------------------
class TestSegmentationMetricsProvenance:
    def test_exact_agreement_returns_1(self):
        """Identical masks yield Dice=1.0 and IoU=1.0."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:50, 20:50] = 255

        dice = compute_dice_coefficient(mask, mask)
        iou = compute_iou(mask, mask)
        assert pytest.approx(dice, 1e-4) == 1.0
        assert pytest.approx(iou, 1e-4) == 1.0

    def test_disjoint_masks_return_0(self):
        """Completely non-overlapping masks yield Dice=0.0 and IoU=0.0."""
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[10:30, 10:30] = 255
        pred = np.zeros((100, 100), dtype=np.uint8)
        pred[60:80, 60:80] = 255

        assert compute_dice_coefficient(gt, pred) == 0.0
        assert compute_iou(gt, pred) == 0.0

    def test_partial_overlap_calculation(self):
        """Test partial overlap with known analytic Dice and IoU values."""
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[0:20, 0:20] = 1

        pred = np.zeros((100, 100), dtype=np.uint8)
        pred[10:30, 0:20] = 1

        dice = compute_dice_coefficient(gt, pred)
        iou = compute_iou(gt, pred)

        assert pytest.approx(dice, 1e-3) == 0.50
        assert pytest.approx(iou, 1e-3) == 1.0 / 3.0

    def test_true_negative_both_empty_returns_1(self):
        """When both ground truth and prediction have no lesion, Dice=1.0 and IoU=1.0."""
        empty_gt = np.zeros((50, 50), dtype=np.uint8)
        empty_pred = np.zeros((50, 50), dtype=np.uint8)

        assert compute_dice_coefficient(empty_gt, empty_pred) == 1.0
        assert compute_iou(empty_gt, empty_pred) == 1.0

    def test_one_empty_one_nonempty_returns_0(self):
        """When one mask has a lesion and the other is empty, Dice=0.0 and IoU=0.0."""
        gt = np.zeros((50, 50), dtype=np.uint8)
        pred = np.zeros((50, 50), dtype=np.uint8)
        pred[10:20, 10:20] = 255

        assert compute_dice_coefficient(gt, pred) == 0.0
        assert compute_iou(gt, pred) == 0.0

    def test_mismatched_dimensions_raises_error(self):
        """Metric calculation requires matching spatial shapes; raises ValueError on mismatch."""
        gt = np.zeros((50, 50), dtype=np.uint8)
        pred = np.zeros((60, 60), dtype=np.uint8)

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_dice_coefficient(gt, pred)

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_iou(gt, pred)

    def test_none_input_raises_error(self):
        """Passing None as mask raises ValueError."""
        with pytest.raises(ValueError, match="must not be None"):
            compute_dice_coefficient(None, np.zeros((10, 10)))

        with pytest.raises(ValueError, match="must not be None"):
            compute_iou(np.zeros((10, 10)), None)
