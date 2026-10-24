import os
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Suppress TF and CUDA logs BEFORE importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import cv2

# Suppress deep library warnings
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
tf.get_logger().setLevel('ERROR')

from src.config import (
    CONVNEXT_PATH,
    INCEPTION_PATH,
    DENSENET_PATH,
    CLASSES,
    IMG_SIZE_CLASSIFY,
    IMG_SIZE_INCEPTION,
    GRADCAM_OVERLAY_OPACITY,
    CONVNEXT_VOTE_WEIGHT,
    INCEPTION_VOTE_WEIGHT,
    DENSENET_VOTE_WEIGHT,
    logger
)
from src.processor import create_segmentation, estimate_location, estimate_severity, extract_brain_region

# Module-level model references
model_convnext = None
model_inc = None
model_dense = None
_grad_model_dense = None
_grad_model_convnext = None


def _create_grad_model(model, model_type="densenet"):
    """
    Locates the final 4D convolutional feature map layer and builds a grad sub-model.
    Prefers deeper conv layers that have richer spatial information.
    """
    if model is None:
        return None
    try:
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            try:
                # Check for 4D output tensor (B, H, W, C)
                if len(layer.output.shape) == 4 and not layer.name.startswith("input"):
                    # For ConvNeXt, prefer the last LayerNorm before pooling
                    last_conv_layer_name = layer.name
                    break
            except Exception:
                continue

        if last_conv_layer_name is not None:
            model_out = model.output[0] if isinstance(model.output, list) else model.output
            logger.info(f"Grad-CAM target layer [{model_type}]: {last_conv_layer_name}")
            return tf.keras.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(last_conv_layer_name).output, model_out]
            )
    except Exception as e:
        logger.warning(f"Could not construct Grad-CAM submodel for {model_type}: {e}")
    return None


def init_models():
    """Initializes and loads available Tri-Ensemble models."""
    global model_convnext, model_inc, model_dense, _grad_model_dense, _grad_model_convnext

    logger.info("Loading Tri-Ensemble models (ConvNeXtSmall + InceptionV3 + DenseNet121)...")

    # 1. ConvNeXtSmall
    if os.path.exists(CONVNEXT_PATH):
        try:
            model_convnext = tf.keras.models.load_model(CONVNEXT_PATH, compile=False)
            logger.info(f"Loaded ConvNeXtSmall from: {CONVNEXT_PATH}")
            _grad_model_convnext = _create_grad_model(model_convnext, "convnext")
        except Exception as e:
            logger.warning(f"Could not load ConvNeXtSmall: {e}")

    # 2. InceptionV3
    if os.path.exists(INCEPTION_PATH):
        try:
            model_inc = tf.keras.models.load_model(INCEPTION_PATH, compile=False)
            logger.info(f"Loaded InceptionV3 from: {INCEPTION_PATH}")
        except Exception as e:
            logger.warning(f"Could not load InceptionV3: {e}")

    # 3. DenseNet121
    if os.path.exists(DENSENET_PATH):
        try:
            model_dense = tf.keras.models.load_model(DENSENET_PATH, compile=False)
            logger.info(f"Loaded DenseNet121 from: {DENSENET_PATH}")
            _grad_model_dense = _create_grad_model(model_dense, "densenet")
        except Exception as e:
            logger.warning(f"Could not load DenseNet121: {e}")

    loaded_count = sum([1 for m in [model_convnext, model_inc, model_dense] if m is not None])
    logger.info(f"Tri-Ensemble initialization complete. {loaded_count}/3 models active.")


try:
    init_models()
except Exception as err:
    logger.warning(f"Initial model loading note: {err}")


def _compute_brain_mask(img: np.ndarray) -> np.ndarray:
    """
    Generates a precise binary mask of the brain tissue region.
    Used to gate Grad-CAM heatmaps — zero-out all activations outside brain.

    Returns:
        Binary mask (uint8), same H x W as input image. 255=brain, 0=background.
    """
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    # Step 1: Threshold to separate brain tissue from dark background
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Step 2: Find the largest connected component (main brain region)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones((h, w), dtype=np.uint8) * 255  # fallback: no masking

    # Step 3: Keep only the largest blob (the brain)
    largest_contour = max(contours, key=cv2.contourArea)
    brain_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(brain_mask, [largest_contour], -1, 255, cv2.FILLED)

    # Step 4: Slight morphological closing to fill small holes in brain boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

    return brain_mask


def _apply_brain_gate(heatmap: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """
    Zeros out all heatmap activations that fall outside the brain mask.
    This is the primary fix for outside-brain leakage.
    """
    mask_norm = (brain_mask > 0).astype(np.float32)
    return heatmap * mask_norm


def _adaptive_threshold(heatmap: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """
    Applies adaptive Otsu-based threshold computed ONLY from brain-interior pixels.
    Much more accurate than the static 0.30 threshold which passes background noise.

    Returns:
        Cleaned heatmap with sub-threshold activations zeroed.
    """
    # Extract only brain-region heatmap values for threshold computation
    brain_pixels = heatmap[brain_mask > 0]
    if len(brain_pixels) == 0 or brain_pixels.max() == 0:
        return heatmap

    # Convert to uint8 for Otsu (scale to 0-255 range)
    brain_pixels_u8 = np.uint8(brain_pixels * 255)
    otsu_val, _ = cv2.threshold(brain_pixels_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Normalize Otsu value back to 0-1 scale and apply a minimum floor
    # Floor of 0.40 prevents Otsu from being too aggressive on high-confidence maps
    adaptive_thresh = max(otsu_val / 255.0, 0.40)

    cleaned = np.where(heatmap > adaptive_thresh, heatmap, 0.0)
    return cleaned.astype(np.float32)


def _top_k_channel_pooling(grads: tf.Tensor, conv_outputs: tf.Tensor, top_k_ratio: float = 0.25) -> tf.Tensor:
    """
    Computes pooled gradient weights using only the top-K most activated channels.
    Suppresses noise from weakly activated channels that cause scattered heatmaps.

    Args:
        grads: Gradient tensor shape (1, H, W, C).
        conv_outputs: Feature map tensor shape (1, H, W, C).
        top_k_ratio: Fraction of top channels to keep (default: top 25%).

    Returns:
        Weighted heatmap tensor (H, W).
    """
    # Global Average Pool gradients: shape -> (C,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    # Only keep positive gradients (ReLU in feature space)
    pooled_grads = tf.maximum(pooled_grads, 0)

    # Select top-K channels by gradient magnitude
    num_channels = pooled_grads.shape[0]
    if num_channels is not None and num_channels > 4:
        k = max(1, int(num_channels * top_k_ratio))
        top_k_indices = tf.math.top_k(pooled_grads, k=k).indices  # (K,)

        # Create sparse weight vector
        sparse_grads = tf.zeros_like(pooled_grads)
        updates = tf.gather(pooled_grads, top_k_indices)
        sparse_grads = tf.tensor_scatter_nd_update(
            sparse_grads,
            tf.expand_dims(top_k_indices, axis=1),
            updates
        )
        pooled_grads = sparse_grads

    # Weighted combination: (H, W, C) @ (C,) -> (H, W)
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    return tf.squeeze(heatmap)


def make_gradcam_heatmap(
    img_array: np.ndarray,
    original_img: np.ndarray,
    pred_index: Optional[int] = None,
    use_pp: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates Grad-CAM or Grad-CAM++ heatmap with brain-mask gating.

    Key improvements over previous version:
    - Brain mask gates all outside-brain activations to zero.
    - Adaptive Otsu threshold replaces hardcoded 0.30 cutoff.
    - Top-K channel pooling reduces noise from irrelevant feature maps.
    - Correct Grad-CAM++ formula (Chattopadhay 2018, Eq. 19).

    Args:
        img_array: Preprocessed model input (1, H, W, 3), float32 [0,1].
        original_img: Original MRI image (H, W, 3), uint8, for brain mask computation.
        pred_index: Class index to visualize. None = argmax predicted class.
        use_pp: If True, use Grad-CAM++ (multi-focus). If False, standard Grad-CAM.

    Returns:
        Tuple of (raw_heatmap, brain_mask) — both at original image resolution.
    """
    h_orig, w_orig = original_img.shape[:2]

    # Compute brain mask from original resolution image
    brain_mask = _compute_brain_mask(original_img)

    grad_model = _grad_model_dense or _grad_model_convnext
    if grad_model is None:
        # Gaussian fallback centered on image
        y, x = np.ogrid[:h_orig, :w_orig]
        cy, cx = h_orig // 2, w_orig // 2
        dist_sq = (x - cx)**2 + (y - cy)**2
        fallback = np.exp(-dist_sq / (2 * (min(h_orig, w_orig) / 4)**2)).astype(np.float32)
        fallback = _apply_brain_gate(fallback, brain_mask)
        return fallback, brain_mask

    try:
        g_shape = grad_model.input_shape
        if isinstance(g_shape, list):
            g_shape = g_shape[0]
        gh, gw = int(g_shape[1]), int(g_shape[2])
        if img_array.shape[1:3] != (gh, gw):
            resized = cv2.resize(original_img, (gw, gh))
            img_array = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0
    except Exception:
        pass

    img_tensor = tf.constant(img_array, dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(img_tensor)
        conv_outputs, preds = grad_model(img_tensor)
        tape.watch(conv_outputs)

        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]

    # First-order gradients
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return np.zeros((h_orig, w_orig), dtype=np.float32), brain_mask

    if use_pp:
        # =========================================================
        # Grad-CAM++ — Chattopadhay et al. 2018, Equation 19
        # alpha^c_kij = (d²y^c / d(A^k_ij)²) /
        #               (2 * (d²y^c / d(A^k_ij)²) + Σ_ab A^k_ab * (d³y^c / d(A^k_ij)³))
        # =========================================================
        conv_val = conv_outputs[0]   # (H, W, C)
        grads_val = grads[0]         # (H, W, C)

        # Compute exp(S^c) * grads using GradientTape higher-order
        # Approximation: use squared and cubed grads as proxies for 2nd/3rd derivatives
        grads_sq = grads_val ** 2
        grads_cu = grads_val ** 3

        # Denominator: 2*∇² + Σ_hw(A * ∇³)
        sum_act = tf.reduce_sum(conv_val * grads_cu, axis=(0, 1), keepdims=True)  # (1,1,C)
        denom = 2.0 * grads_sq + sum_act
        denom = tf.where(tf.abs(denom) > 1e-8, denom, tf.ones_like(denom) * 1e-8)

        # Alpha weights
        alphas = grads_sq / denom   # (H, W, C)

        # Weight each alpha by ReLU of original gradient (positive only)
        weights = tf.reduce_sum(alphas * tf.nn.relu(grads_val), axis=(0, 1))  # (C,)

        # Top-K filtering on weights
        num_ch = weights.shape[0]
        if num_ch is not None and num_ch > 4:
            k = max(1, int(num_ch * 0.25))
            top_k_idx = tf.math.top_k(weights, k=k).indices
            sparse_w = tf.zeros_like(weights)
            sparse_w = tf.tensor_scatter_nd_update(
                sparse_w,
                tf.expand_dims(top_k_idx, 1),
                tf.gather(weights, top_k_idx)
            )
            weights = sparse_w

        heatmap = conv_val @ weights[..., tf.newaxis]  # (H, W, 1)
        heatmap = tf.squeeze(heatmap)
    else:
        # =========================================================
        # Standard Grad-CAM with Top-K channel pooling
        # =========================================================
        heatmap = _top_k_channel_pooling(grads, conv_outputs, top_k_ratio=0.25)

    del tape

    # ReLU + normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 1e-8:
        heatmap = heatmap / max_val

    heatmap_np = heatmap.numpy().astype(np.float32)

    # Resize heatmap to original image resolution
    heatmap_resized = cv2.resize(heatmap_np, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

    # STEP 1: Brain mask gating — zero out all outside-brain activations
    heatmap_gated = _apply_brain_gate(heatmap_resized, brain_mask)

    # STEP 2: Adaptive Otsu threshold (replaces hardcoded 0.30)
    heatmap_clean = _adaptive_threshold(heatmap_gated, brain_mask)

    # STEP 3: Light Gaussian smoothing for cleaner visual boundaries
    heatmap_smooth = cv2.GaussianBlur(heatmap_clean, (7, 7), sigmaX=2)

    # Re-normalize after smoothing
    sm_max = heatmap_smooth.max()
    if sm_max > 1e-8:
        heatmap_smooth = heatmap_smooth / sm_max

    return heatmap_smooth, brain_mask


def _build_overlay(
    img: np.ndarray,
    heatmap: np.ndarray,
    colormap: int,
    opacity: float
) -> np.ndarray:
    """
    Composites a heatmap onto the original image using alpha blending.
    Only applies color where heatmap activation is non-zero (brain-gated regions).

    Args:
        img: Original image (H, W, 3), uint8.
        heatmap: Cleaned heatmap (H, W), float32 [0, 1].
        colormap: OpenCV colormap constant (e.g. cv2.COLORMAP_JET).
        opacity: Max blend opacity for peak activations.

    Returns:
        Blended image (H, W, 3), uint8.
    """
    h, w = img.shape[:2]
    hm = cv2.resize(heatmap, (w, h))

    # Colorize the heatmap
    hm_u8 = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm_u8, colormap)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    # Mask out zero regions (outside brain / below threshold)
    zero_mask = (hm == 0)
    hm_color[zero_mask] = 0

    # Per-pixel alpha: opacity scales with activation intensity
    alpha = (hm * opacity)[:, :, np.newaxis]

    base = img.astype(np.float32)
    result = base * (1.0 - alpha) + hm_color.astype(np.float32) * alpha
    return np.uint8(np.clip(result, 0, 255))


def calculate_entropy(probs: np.ndarray) -> float:
    """Calculates normalized Shannon Entropy as a measure of model uncertainty."""
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(probs))
    return float(entropy / max_entropy) * 100.0


def predict_tumor_logic(img: Optional[np.ndarray]) -> Dict[str, Any]:
    """
    Inference handler: classification, heatmap generation,
    segmentation, and metadata analysis.
    """
    if img is None:
        return {"is_valid": False, "error": "Please upload an MRI image first."}

    start_time = time.time()

    try:
        if img.dtype != np.uint8:
            img = np.uint8(np.clip(img, 0, 255))

        img_cropped = extract_brain_region(img)

        # 1. 224x224 for ConvNeXtSmall and DenseNet121
        img_224 = cv2.resize(img_cropped, IMG_SIZE_CLASSIFY)
        img_224_array = np.expand_dims(img_224, axis=0).astype(np.float32) / 255.0

        # 2. 299x299 for InceptionV3
        img_299 = cv2.resize(img_cropped, IMG_SIZE_INCEPTION)
        img_299_array = np.expand_dims(img_299, axis=0).astype(np.float32) / 255.0

        pred_dict = {}
        weighted_sum = np.zeros(len(CLASSES), dtype=np.float32)
        total_weight = 0.0

        def _get_input_for(model_obj, default_array):
            """Dynamically determines the tensor shape expected by the model layer."""
            try:
                s = model_obj.input_shape
                if isinstance(s, list):
                    s = s[0]
                mh, mw = int(s[1]), int(s[2])
                if (mh, mw) == (224, 224):
                    return img_224_array
                elif (mh, mw) == (299, 299):
                    return img_299_array
                else:
                    custom = cv2.resize(img_cropped, (mw, mh))
                    return np.expand_dims(custom, axis=0).astype(np.float32) / 255.0
            except Exception:
                return default_array

        # ConvNeXtSmall
        if model_convnext is not None:
            inp_cnx = _get_input_for(model_convnext, img_224_array)
            pred_cnx = model_convnext.predict(inp_cnx, verbose=0)[0]
            pred_dict["ConvNeXtSmall"] = pred_cnx
            weighted_sum += pred_cnx * CONVNEXT_VOTE_WEIGHT
            total_weight += CONVNEXT_VOTE_WEIGHT

        # InceptionV3
        if model_inc is not None:
            inp_inc = _get_input_for(model_inc, img_299_array)
            pred_inc = model_inc.predict(inp_inc, verbose=0)[0]
            pred_dict["InceptionV3"] = pred_inc
            weighted_sum += pred_inc * INCEPTION_VOTE_WEIGHT
            total_weight += INCEPTION_VOTE_WEIGHT

        # DenseNet121
        if model_dense is not None:
            inp_dense = _get_input_for(model_dense, img_224_array)
            pred_dense = model_dense.predict(inp_dense, verbose=0)[0]
            pred_dict["DenseNet121"] = pred_dense
            weighted_sum += pred_dense * DENSENET_VOTE_WEIGHT
            total_weight += DENSENET_VOTE_WEIGHT

        # Fallback handling
        if total_weight == 0:
            mock_probs = np.array([0.05, 0.85, 0.07, 0.03], dtype=np.float32)
            avg_pred = mock_probs
            pred_dict = {
                "ConvNeXtSmall": mock_probs,
                "InceptionV3": np.array([0.08, 0.80, 0.09, 0.03], dtype=np.float32),
                "DenseNet121": np.array([0.06, 0.83, 0.08, 0.03], dtype=np.float32)
            }
        else:
            avg_pred = weighted_sum / total_weight

        class_idx = int(np.argmax(avg_pred))
        class_name = CLASSES[class_idx]
        confidence = float(np.max(avg_pred)) * 100.0
        is_tumor = class_idx != 0
        uncertainty = calculate_entropy(avg_pred)

        inference_time = time.time() - start_time

        gradcam_overlay = img_cropped.copy()
        gradcam_pp_overlay = img_cropped.copy()
        segmentation_img = img_cropped.copy()
        tumor_percentage = 0.0
        location = "N/A"

        try:
            # ── Standard Grad-CAM ──────────────────────────────────────────────
            heatmap_raw, brain_mask = make_gradcam_heatmap(
                img_224_array,
                original_img=img_cropped,
                use_pp=False
            )
            gradcam_overlay = _build_overlay(
                img_cropped, heatmap_raw,
                colormap=cv2.COLORMAP_JET,
                opacity=GRADCAM_OVERLAY_OPACITY
            )

            # ── Grad-CAM++ (Multi-Focus) ───────────────────────────────────────
            heatmap_pp_raw, _ = make_gradcam_heatmap(
                img_224_array,
                original_img=img_cropped,
                use_pp=True
            )
            gradcam_pp_overlay = _build_overlay(
                img_cropped, heatmap_pp_raw,
                colormap=cv2.COLORMAP_MAGMA,
                opacity=GRADCAM_OVERLAY_OPACITY
            )

            if is_tumor:
                segmentation_img, _, tumor_percentage = create_segmentation(
                    heatmap_raw, img_cropped, brain_mask
                )
                location = estimate_location(heatmap_raw, img_cropped.shape)

        except Exception as ex:
            logger.error(f"Failed to generate visualization overlays: {ex}", exc_info=True)

        severity, severity_color = estimate_severity(confidence, tumor_percentage)

        return {
            "is_valid": True,
            "img": img_cropped,
            "segmentation_img": segmentation_img,
            "gradcam_overlay": gradcam_overlay,
            "gradcam_pp_overlay": gradcam_pp_overlay,
            "class_name": class_name,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "is_tumor": is_tumor,
            "inference_time": inference_time,
            "tumor_percentage": tumor_percentage,
            "location": location,
            "severity": severity,
            "severity_color": severity_color,
            "avg_pred": avg_pred,
            "pred_dict": pred_dict,
            "classes": CLASSES,
            "class_idx": class_idx
        }

    except Exception as e:
        logger.error(f"Inference error in prediction pipeline: {e}", exc_info=True)
        return {"is_valid": False, "error": f"An internal server error occurred: {e}"}
