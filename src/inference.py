import os
import time
from typing import Dict, Any, Optional, Tuple, List
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

import hashlib
from src.config import (
    PIPELINE_VERSION,
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
from src.processor import create_segmentation, estimate_location, estimate_severity, extract_brain_region, extract_brain_region_bounds

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


def is_ensemble_ready() -> bool:
    """
    Returns True only when all three required ensemble backbones
    (ConvNeXtSmall, InceptionV3, DenseNet121) are fully loaded and operational.
    """
    return (
        model_convnext is not None
        and model_inc is not None
        and model_dense is not None
    )


def get_model_status() -> Dict[str, bool]:
    """
    Returns live availability status for each individual ensemble model.
    """
    return {
        "ConvNeXtSmall": model_convnext is not None,
        "InceptionV3": model_inc is not None,
        "DenseNet121": model_dense is not None,
    }


_PROVENANCE_HASH_CACHE: Dict[str, Optional[str]] = {}


def _compute_file_sha256(path: str, chunk_size: int = 65536) -> Optional[str]:
    """Computes SHA-256 digest for an artifact file, returning hex string or None if file missing."""
    if not path or not os.path.exists(path):
        return None
    if path in _PROVENANCE_HASH_CACHE:
        return _PROVENANCE_HASH_CACHE[path]
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        digest = h.hexdigest()
        _PROVENANCE_HASH_CACHE[path] = digest
        return digest
    except Exception as e:
        logger.warning(f"Could not compute artifact digest for {path}: {e}")
        return None


def get_model_provenance() -> Dict[str, Any]:
    """
    Returns complete architecture, artifact digest, and weighting provenance for the active ensemble.
    File paths are sanitized to basenames to prevent leaking internal server directories.
    """
    return {
        "pipeline_version": PIPELINE_VERSION,
        "models": {
            "ConvNeXtSmall": {
                "artifact": os.path.basename(CONVNEXT_PATH) if os.path.exists(CONVNEXT_PATH) else "UNKNOWN",
                "sha256": _compute_file_sha256(CONVNEXT_PATH),
                "weight": CONVNEXT_VOTE_WEIGHT,
                "input_size": list(IMG_SIZE_CLASSIFY),
            },
            "InceptionV3": {
                "artifact": os.path.basename(INCEPTION_PATH) if os.path.exists(INCEPTION_PATH) else "UNKNOWN",
                "sha256": _compute_file_sha256(INCEPTION_PATH),
                "weight": INCEPTION_VOTE_WEIGHT,
                "input_size": list(IMG_SIZE_INCEPTION),
            },
            "DenseNet121": {
                "artifact": os.path.basename(DENSENET_PATH) if os.path.exists(DENSENET_PATH) else "UNKNOWN",
                "sha256": _compute_file_sha256(DENSENET_PATH),
                "weight": DENSENET_VOTE_WEIGHT,
                "input_size": list(IMG_SIZE_CLASSIFY),
            },
        },
        "classes": CLASSES,
    }


def init_models(force_reload: bool = False) -> bool:
    """
    Initializes and loads required Tri-Ensemble models.
    Does not reload if models are already loaded unless force_reload=True.
    Returns True if all required models loaded successfully, False otherwise.
    """
    global model_convnext, model_inc, model_dense, _grad_model_dense, _grad_model_convnext

    if not force_reload and is_ensemble_ready():
        return True

    logger.info("Loading Tri-Ensemble models (ConvNeXtSmall + InceptionV3 + DenseNet121)...")

    # 1. ConvNeXtSmall
    if model_convnext is None or force_reload:
        if os.path.exists(CONVNEXT_PATH):
            try:
                model_convnext = tf.keras.models.load_model(CONVNEXT_PATH, compile=False)
                logger.info(f"Loaded ConvNeXtSmall from: {CONVNEXT_PATH}")
                _grad_model_convnext = _create_grad_model(model_convnext, "convnext")
            except Exception as e:
                logger.error(f"Failed to load ConvNeXtSmall from {CONVNEXT_PATH}: {e}")
                model_convnext = None
                _grad_model_convnext = None
        else:
            logger.warning(f"ConvNeXtSmall weights file not found: {CONVNEXT_PATH}")
            model_convnext = None
            _grad_model_convnext = None

    # 2. InceptionV3
    if model_inc is None or force_reload:
        if os.path.exists(INCEPTION_PATH):
            try:
                model_inc = tf.keras.models.load_model(INCEPTION_PATH, compile=False)
                logger.info(f"Loaded InceptionV3 from: {INCEPTION_PATH}")
            except Exception as e:
                logger.error(f"Failed to load InceptionV3 from {INCEPTION_PATH}: {e}")
                model_inc = None
        else:
            logger.warning(f"InceptionV3 weights file not found: {INCEPTION_PATH}")
            model_inc = None

    # 3. DenseNet121
    if model_dense is None or force_reload:
        if os.path.exists(DENSENET_PATH):
            try:
                model_dense = tf.keras.models.load_model(DENSENET_PATH, compile=False)
                logger.info(f"Loaded DenseNet121 from: {DENSENET_PATH}")
                _grad_model_dense = _create_grad_model(model_dense, "densenet")
            except Exception as e:
                logger.error(f"Failed to load DenseNet121 from {DENSENET_PATH}: {e}")
                model_dense = None
                _grad_model_dense = None
        else:
            logger.warning(f"DenseNet121 weights file not found: {DENSENET_PATH}")
            model_dense = None
            _grad_model_dense = None

    status = get_model_status()
    loaded_count = sum(status.values())
    logger.info(f"Tri-Ensemble initialization complete: {loaded_count}/3 models active. Ready: {is_ensemble_ready()}")
    return is_ensemble_ready()


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
    pooled_grads = tf.maximum(pooled_grads, 0.0)

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

    # Weighted combination: (H, W, C) @ (C, 1) -> (H, W, 1)
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    # Extract spatial channels without squeezing spatial dims (preserves (H, W) even if H=1 or W=1)
    return heatmap[:, :, 0]


def make_gradcam_heatmap(
    img_array: np.ndarray,
    original_img: np.ndarray,
    pred_index: Optional[int] = None,
    use_pp: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates Grad-CAM or Grad-CAM++ heatmap with brain-mask gating.

    Key improvements:
    - Target class validation ensures heatmap matches winning ensemble class.
    - Brain mask gates all outside-brain activations to zero.
    - Adaptive Otsu threshold replaces hardcoded cutoffs.
    - Top-K channel pooling reduces noise from irrelevant feature maps.
    - Chattopadhay et al. (2018) Grad-CAM++ formulation.
    - Strict non-finite checks and dimension safeguards; no fabricated output.

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
        logger.warning("No grad_model available for explainability generation.")
        return np.zeros((h_orig, w_orig), dtype=np.float32), brain_mask

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

        num_classes = preds.shape[-1]
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        elif not isinstance(pred_index, (int, np.integer)) or pred_index < 0 or pred_index >= num_classes:
            logger.warning(f"Invalid pred_index {pred_index} for model with {num_classes} classes. Falling back to argmax.")
            pred_index = int(tf.argmax(preds[0]))

        class_channel = preds[:, pred_index]

    # First-order gradients
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None or not tf.reduce_all(tf.math.is_finite(grads)):
        del tape
        logger.warning("Grad-CAM backpropagation yielded None or non-finite gradients.")
        return np.zeros((h_orig, w_orig), dtype=np.float32), brain_mask

    if use_pp:
        # =========================================================
        # Grad-CAM++ — Chattopadhay et al. 2018, Equation 19
        # alpha^c_kij = (d²y^c / d(A^k_ij)²) /
        #               (2 * (d²y^c / d(A^k_ij)²) + Σ_ab A^k_ab * (d³y^c / d(A^k_ij)³))
        # =========================================================
        conv_val = conv_outputs[0]   # (H, W, C)
        grads_val = grads[0]         # (H, W, C)

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
        heatmap = heatmap[:, :, 0]
    else:
        # =========================================================
        # Standard Grad-CAM with Top-K channel pooling
        # =========================================================
        heatmap = _top_k_channel_pooling(grads, conv_outputs, top_k_ratio=0.25)

    del tape

    # ReLU + sanitize non-finite values + normalize
    heatmap = tf.maximum(heatmap, 0.0)
    heatmap = tf.where(tf.math.is_finite(heatmap), heatmap, tf.zeros_like(heatmap))
    max_val = tf.reduce_max(heatmap)
    if max_val > 1e-8:
        heatmap = heatmap / max_val
    else:
        heatmap = tf.zeros_like(heatmap)

    heatmap_np = heatmap.numpy().astype(np.float32)
    if not np.any(heatmap_np > 1e-8):
        return np.zeros((h_orig, w_orig), dtype=np.float32), brain_mask

    # Resize heatmap to original image resolution with clamp against interpolation ringing
    heatmap_resized = cv2.resize(heatmap_np, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

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
    else:
        heatmap_smooth = np.zeros_like(heatmap_smooth)

    return heatmap_smooth.astype(np.float32), brain_mask


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
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    base = img.copy()
    if base.ndim == 2:
        base = np.stack([base, base, base], axis=-1)
    elif base.ndim == 3 and base.shape[2] == 1:
        base = np.concatenate([base, base, base], axis=-1)
    elif base.ndim == 3 and base.shape[2] == 4:
        base = base[:, :, :3]

    if heatmap is None or not isinstance(heatmap, np.ndarray) or heatmap.size == 0 or np.all(heatmap <= 0):
        return base.astype(np.uint8)

    hm_clean = np.nan_to_num(heatmap.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    hm_clean = np.clip(hm_clean, 0.0, 1.0)
    hm = cv2.resize(hm_clean, (w, h))

    # Colorize the heatmap
    hm_u8 = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm_u8, colormap)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    # Mask out zero regions (outside brain / below threshold)
    zero_mask = (hm == 0)
    hm_color[zero_mask] = 0

    # Per-pixel alpha: opacity scales with activation intensity
    alpha = (hm * opacity)[:, :, np.newaxis]

    base_f = base.astype(np.float32)
    result = base_f * (1.0 - alpha) + hm_color.astype(np.float32) * alpha
    return np.uint8(np.clip(result, 0, 255))


def calculate_entropy(probs: np.ndarray) -> float:
    """
    Calculates normalized Shannon Entropy (0.0 to 100.0%) of a discrete probability vector.
    
    Dispersion metric:
    - 0.0% = maximum concentration (deterministic one-hot prediction).
    - 100.0% = uniform distribution across all K classes (maximum dispersion / ambiguity).
    
    IMPORTANT CLINICAL CAVEAT:
    Shannon entropy measures the mathematical dispersion of the model's softmax outputs across classes.
    It does NOT represent calibrated Bayesian uncertainty, aleatoric sensor noise, or
    epistemic out-of-distribution clinical uncertainty.
    """
    if probs is None or not isinstance(probs, np.ndarray) or probs.size == 0:
        return 0.0
    if not np.all(np.isfinite(probs)):
        return 0.0

    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    p_sum = np.sum(p)
    if p_sum <= 0:
        return 0.0
    p = p / p_sum

    positive_p = p[p > 1e-12]
    if len(positive_p) <= 1:
        return 0.0

    entropy = -np.sum(positive_p * np.log2(positive_p))
    k = len(p)
    max_entropy = np.log2(k) if k > 1 else 1.0
    if max_entropy <= 0:
        return 0.0

    norm_entropy = float(np.clip((entropy / max_entropy) * 100.0, 0.0, 100.0))
    return round(norm_entropy, 4)


def _validate_model_output(pred: Any, model_name: str) -> Optional[np.ndarray]:
    """
    Validates and normalizes an individual model prediction vector.
    
    Ensures:
    - Output is a 1D numpy array of length matching CLASSES (4).
    - Contains only finite numerical values (no NaN or Inf).
    - Represents a valid probability distribution (non-negative, sums to 1.0).
      Applies numerically stable softmax if unnormalized logits are detected.
      
    Returns:
        1D float32 numpy array of length 4 summing to 1.0, or None if invalid.
    """
    if pred is None:
        logger.error(f"Model output validation failed [{model_name}]: None output.")
        return None

    try:
        arr = np.asarray(pred, dtype=np.float32)
    except Exception as e:
        logger.error(f"Model output validation failed [{model_name}]: could not convert to float32: {e}")
        return None

    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 1 or arr.shape[0] != len(CLASSES):
        logger.error(
            f"Model output validation failed [{model_name}]: invalid shape {arr.shape}. "
            f"Expected 1D vector of length {len(CLASSES)}."
        )
        return None

    if not np.all(np.isfinite(arr)):
        logger.error(f"Model output validation failed [{model_name}]: non-finite values (NaN/Inf) detected.")
        return None

    # Check if already a valid probability vector (non-negative, sums to ~1.0)
    is_prob = np.all(arr >= 0.0) and (0.95 <= float(np.sum(arr)) <= 1.05)
    if not is_prob:
        shift_arr = arr - np.max(arr)
        exp_arr = np.exp(shift_arr)
        sum_exp = np.sum(exp_arr)
        if sum_exp <= 0 or not np.isfinite(sum_exp):
            logger.error(f"Model output validation failed [{model_name}]: numerical instability during softmax.")
            return None
        arr = exp_arr / sum_exp
    else:
        s = float(np.sum(arr))
        if s > 0:
            arr = arr / s

    return arr.astype(np.float32)


def _generate_tta_variants(img_rgb: np.ndarray, num_passes: int = 5) -> List[np.ndarray]:
    """
    Generates deterministic, clinically plausible 2D MRI test-time augmentations.
    
    Pass 0: Original input image.
    Pass 1: Horizontal reflection (sagittal/axial symmetry).
    Pass 2: Subtle clockwise rotation (+5 degrees).
    Pass 3: Subtle counter-clockwise rotation (-5 degrees).
    Pass 4: Subtle isotropic magnification (1.04x zoom).
    """
    variants = [img_rgb]
    if num_passes <= 1:
        return variants

    h, w = img_rgb.shape[:2]

    # Pass 1: Horizontal flip
    if len(variants) < num_passes:
        variants.append(cv2.flip(img_rgb, 1))

    # Pass 2: Mild +5 deg rotation
    if len(variants) < num_passes:
        M_pos = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 5.0, 1.0)
        variants.append(cv2.warpAffine(img_rgb, M_pos, (w, h), borderMode=cv2.BORDER_REFLECT_101))

    # Pass 3: Mild -5 deg rotation
    if len(variants) < num_passes:
        M_neg = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -5.0, 1.0)
        variants.append(cv2.warpAffine(img_rgb, M_neg, (w, h), borderMode=cv2.BORDER_REFLECT_101))

    # Pass 4: Mild zoom (1.04x)
    if len(variants) < num_passes:
        zoom_factor = 1.04
        zh, zw = int(h * zoom_factor), int(w * zoom_factor)
        iz = cv2.resize(img_rgb, (zw, zh))
        sh, sw = (zh - h) // 2, (zw - w) // 2
        variants.append(iz[sh:sh + h, sw:sw + w])

    return variants[:num_passes]


def _predict_single_model(
    model_obj: Any,
    img_cropped: np.ndarray,
    target_size: Tuple[int, int],
    model_name: str,
    use_tta: bool = False,
    tta_passes: int = 5
) -> Optional[np.ndarray]:
    """
    Runs model inference for a single backbone with shape adaptation,
    optional multi-pass TTA, and strict output validation.
    """
    if model_obj is None:
        return None

    try:
        mw, mh = target_size
        try:
            s = model_obj.input_shape
            if isinstance(s, list):
                s = s[0]
            if len(s) >= 3 and s[1] is not None and s[2] is not None:
                mh, mw = int(s[1]), int(s[2])
        except Exception:
            pass

        if not use_tta:
            resized = cv2.resize(img_cropped, (mw, mh))
            inp = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0
            raw_pred = model_obj.predict(inp, verbose=0)
            if isinstance(raw_pred, list):
                raw_pred = raw_pred[0]
            if hasattr(raw_pred, "numpy"):
                raw_pred = raw_pred.numpy()
            return _validate_model_output(raw_pred[0], model_name)

        # Multi-Pass TTA
        variants = _generate_tta_variants(img_cropped, num_passes=tta_passes)
        pass_preds = []
        for var in variants:
            resized = cv2.resize(var, (mw, mh))
            inp = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0
            raw = model_obj.predict(inp, verbose=0)
            if isinstance(raw, list):
                raw = raw[0]
            if hasattr(raw, "numpy"):
                raw = raw.numpy()
            v = _validate_model_output(raw[0], model_name)
            if v is not None:
                pass_preds.append(v)

        if not pass_preds:
            logger.error(f"All TTA passes failed validation for {model_name}.")
            return None

        # Mean of valid TTA passes
        mean_pred = np.mean(pass_preds, axis=0).astype(np.float32)
        s = float(np.sum(mean_pred))
        if s > 0:
            mean_pred = mean_pred / s
        return mean_pred

    except Exception as e:
        logger.error(f"Prediction execution failed for {model_name}: {e}", exc_info=True)
        return None


def predict_tumor_logic(
    img: Optional[np.ndarray],
    analysis_id: Optional[str] = None,
    use_tta: bool = False,
    tta_passes: int = 5,
    colormap: str = "jet"
) -> Dict[str, Any]:
    """
    Inference handler: classification, heatmap generation,
    segmentation, and metadata analysis.
    """
    import uuid
    from datetime import datetime, timezone

    analysis_id = analysis_id or f"ANL-{uuid.uuid4().hex[:10].upper()}"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _fail_response(error_msg: str, code: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        resp = {
            "is_valid": False,
            "status": "FAILED",
            "error": error_msg,
            "code": code,
            "error_code": code,
            "pipeline_version": PIPELINE_VERSION,
            "analysis_id": analysis_id,
            "timestamp": timestamp,
        }
        if extra:
            resp.update(extra)
        return resp

    # 1. Input presence and type validation
    if img is None:
        logger.warning(f"Inference rejected: analysis_id={analysis_id} reason='Empty image input'")
        return _fail_response("Please upload an MRI image first.", "INVALID_INPUT")

    if not isinstance(img, np.ndarray):
        logger.warning(f"Inference rejected: analysis_id={analysis_id} reason='Invalid input type {type(img)}'")
        return _fail_response(f"Invalid image input type {type(img).__name__}. Expected numpy.ndarray.", "INVALID_INPUT")

    if img.size == 0:
        logger.warning(f"Inference rejected: analysis_id={analysis_id} reason='Image array has 0 pixels'")
        return _fail_response("Image array is empty (0 pixels).", "INVALID_INPUT")

    # 2. Non-finite values (NaN / Inf) check
    if not np.all(np.isfinite(img)):
        logger.warning(f"Inference rejected: analysis_id={analysis_id} reason='Non-finite pixel values (NaN/Inf)'")
        return _fail_response("Image contains non-finite pixel values (NaN or Inf). Cannot safely perform clinical inference.", "INVALID_INPUT")

    # 3. Dimension and channel checks
    if img.ndim < 2 or img.ndim > 3:
        logger.warning(f"Inference rejected: analysis_id={analysis_id} reason='Invalid array ndim {img.ndim}'")
        return _fail_response(f"Invalid image array dimension: {img.shape}. Expected 2D (H, W) or 3D (H, W, C).", "INVALID_INPUT")

    if img.ndim == 3 and img.shape[2] not in (1, 3, 4):
        logger.warning(f"Inference rejected: analysis_id={analysis_id} reason='Invalid channel count {img.shape[2]}'")
        return _fail_response(f"Unsupported channel count {img.shape[2]}. Expected 1 (grayscale), 3 (RGB), or 4 (RGBA).", "INVALID_INPUT")

    h, w = img.shape[:2]
    if h < 10 or w < 10:
        logger.warning(f"Inference rejected: analysis_id={analysis_id} reason='Dimensions too small ({w}x{h})'")
        return _fail_response(f"Image dimensions ({w}x{h}) are too small for clinical inference (minimum 10x10).", "INVALID_INPUT")

    if h > 5000 or w > 5000:
        logger.warning(f"Inference rejected: analysis_id={analysis_id} reason='Dimensions too large ({w}x{h})'")
        return _fail_response(f"Image dimensions ({w}x{h}) exceed maximum allowable size (5000x5000).", "INVALID_INPUT")

    # 4. Check model availability
    if not is_ensemble_ready():
        status = get_model_status()
        missing = [m for m, active in status.items() if not active]
        msg = f"Inference unavailable: Ensemble requires all 3 models, but missing or inactive: {', '.join(missing)}."
        logger.error(f"Inference rejected: analysis_id={analysis_id} reason='{msg}'")
        return _fail_response(msg, "MODELS_UNAVAILABLE", {"models_active": status})

    logger.info(f"Inference started: analysis_id={analysis_id} tta_enabled={use_tta}")
    start_time = time.time()

    try:
        # Standardize pixel values to uint8 [0, 255]
        if img.dtype in (np.float32, np.float64):
            if img.max() <= 1.0 and img.min() >= 0.0:
                img_u8 = (img * 255.0).astype(np.uint8)
            else:
                img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img_u8 = img.copy()

        # Standardize color channels to 3-channel RGB uint8
        if img_u8.ndim == 2:
            img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)
        elif img_u8.ndim == 3 and img_u8.shape[2] == 1:
            img_rgb = np.concatenate([img_u8, img_u8, img_u8], axis=-1)
        elif img_u8.ndim == 3 and img_u8.shape[2] == 4:
            img_rgb = img_u8[:, :, :3]
        else:
            img_rgb = img_u8

        img_cropped, (bx1, by1, bx2, by2) = extract_brain_region_bounds(img_rgb)
        if img_cropped.ndim == 2:
            img_cropped = np.stack([img_cropped, img_cropped, img_cropped], axis=-1)
        elif img_cropped.ndim == 3 and img_cropped.shape[2] == 1:
            img_cropped = np.concatenate([img_cropped, img_cropped, img_cropped], axis=-1)
        elif img_cropped.ndim == 3 and img_cropped.shape[2] == 4:
            img_cropped = img_cropped[:, :, :3]

        # 224x224 input tensor for Grad-CAM
        img_224 = cv2.resize(img_cropped, IMG_SIZE_CLASSIFY)
        img_224_array = np.expand_dims(img_224, axis=0).astype(np.float32) / 255.0

        pred_dict = {}
        weighted_sum = np.zeros(len(CLASSES), dtype=np.float32)
        total_weight = 0.0

        # ConvNeXtSmall (45% vote)
        if model_convnext is not None:
            pred_cnx = _predict_single_model(
                model_convnext, img_cropped, IMG_SIZE_CLASSIFY, "ConvNeXtSmall", use_tta, tta_passes
            )
            if pred_cnx is not None:
                pred_dict["ConvNeXtSmall"] = pred_cnx
                weighted_sum += pred_cnx * CONVNEXT_VOTE_WEIGHT
                total_weight += CONVNEXT_VOTE_WEIGHT

        # InceptionV3 (35% vote)
        if model_inc is not None:
            pred_inc = _predict_single_model(
                model_inc, img_cropped, IMG_SIZE_INCEPTION, "InceptionV3", use_tta, tta_passes
            )
            if pred_inc is not None:
                pred_dict["InceptionV3"] = pred_inc
                weighted_sum += pred_inc * INCEPTION_VOTE_WEIGHT
                total_weight += INCEPTION_VOTE_WEIGHT

        # DenseNet121 (20% vote)
        if model_dense is not None:
            pred_dense = _predict_single_model(
                model_dense, img_cropped, IMG_SIZE_CLASSIFY, "DenseNet121", use_tta, tta_passes
            )
            if pred_dense is not None:
                pred_dict["DenseNet121"] = pred_dense
                weighted_sum += pred_dense * DENSENET_VOTE_WEIGHT
                total_weight += DENSENET_VOTE_WEIGHT

        # Strict ensemble policy: all 3 models must successfully participate
        if len(pred_dict) == 0:
            msg = "An internal error occurred during model inference."
            logger.error(f"Inference rejected: analysis_id={analysis_id} reason='All models failed during execution.'")
            return _fail_response(msg, "INFERENCE_ERROR")
        elif len(pred_dict) < 3 or total_weight == 0.0:
            missing = [m for m in ("ConvNeXtSmall", "InceptionV3", "DenseNet121") if m not in pred_dict]
            msg = f"Ensemble inference failed: incomplete or invalid model predictions. Missing or failed: {', '.join(missing)}."
            logger.error(f"Inference rejected: analysis_id={analysis_id} reason='{msg}'")
            return _fail_response(msg, "INCOMPLETE_ENSEMBLE", {"valid_models": list(pred_dict.keys())})

        avg_pred = weighted_sum / total_weight
        # Re-normalize to guarantee exact sum to 1.0 without precision drift
        avg_pred = np.clip(avg_pred, 0.0, 1.0)
        s_sum = float(np.sum(avg_pred))
        if s_sum > 0:
            avg_pred = avg_pred / s_sum

        class_idx = int(np.argmax(avg_pred))
        class_name = CLASSES[class_idx]
        confidence = float(np.max(avg_pred)) * 100.0
        is_tumor = class_idx != 0
        uncertainty = calculate_entropy(avg_pred)

        inference_time = time.time() - start_time

        h_orig, w_orig = img_rgb.shape[:2]
        brain_mask_full = _compute_brain_mask(img_rgb)

        gradcam_overlay = img_rgb.copy()
        gradcam_pp_overlay = img_rgb.copy()
        segmentation_img = img_rgb.copy()
        tumor_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        tumor_percentage = 0.0
        location = "N/A"
        heatmap_raw = None
        heatmap_pp_raw = None

        cm_map = {
            "jet": cv2.COLORMAP_JET,
            "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
            "inferno": getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET),
            "viridis": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET),
        }
        cm_key = str(colormap).lower().strip()
        cm_code = cm_map.get(cm_key, cv2.COLORMAP_JET)
        cm_pp_code = cv2.COLORMAP_MAGMA if cm_key == "jet" else cm_code

        try:
            # ── Standard Grad-CAM ──────────────────────────────────────────────
            heatmap_raw_crop, _ = make_gradcam_heatmap(
                img_224_array,
                original_img=img_cropped,
                pred_index=class_idx,
                use_pp=False
            )
            heatmap_raw = np.zeros((h_orig, w_orig), dtype=np.float32)
            if heatmap_raw_crop is not None and (by2 > by1) and (bx2 > bx1):
                hm_h, hm_w = by2 - by1, bx2 - bx1
                if heatmap_raw_crop.shape[:2] != (hm_h, hm_w):
                    heatmap_raw_crop = cv2.resize(heatmap_raw_crop, (hm_w, hm_h))
                heatmap_raw[by1:by2, bx1:bx2] = heatmap_raw_crop
            heatmap_raw = _apply_brain_gate(heatmap_raw, brain_mask_full)

            gradcam_overlay = _build_overlay(
                img_rgb, heatmap_raw,
                colormap=cm_code,
                opacity=GRADCAM_OVERLAY_OPACITY
            )

            # ── Grad-CAM++ (Multi-Focus) ───────────────────────────────────────
            heatmap_pp_raw_crop, _ = make_gradcam_heatmap(
                img_224_array,
                original_img=img_cropped,
                pred_index=class_idx,
                use_pp=True
            )
            heatmap_pp_raw = np.zeros((h_orig, w_orig), dtype=np.float32)
            if heatmap_pp_raw_crop is not None and (by2 > by1) and (bx2 > bx1):
                hm_h, hm_w = by2 - by1, bx2 - bx1
                if heatmap_pp_raw_crop.shape[:2] != (hm_h, hm_w):
                    heatmap_pp_raw_crop = cv2.resize(heatmap_pp_raw_crop, (hm_w, hm_h))
                heatmap_pp_raw[by1:by2, bx1:bx2] = heatmap_pp_raw_crop
            heatmap_pp_raw = _apply_brain_gate(heatmap_pp_raw, brain_mask_full)

            gradcam_pp_overlay = _build_overlay(
                img_rgb, heatmap_pp_raw,
                colormap=cm_pp_code,
                opacity=GRADCAM_OVERLAY_OPACITY
            )

            if is_tumor:
                segmentation_img, tumor_mask, tumor_percentage = create_segmentation(
                    heatmap_raw, img_rgb, brain_mask_full
                )
                location = estimate_location(heatmap_raw, img_rgb.shape)
            else:
                segmentation_img = img_rgb.copy()
                tumor_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
                location = "No Tumor Detected"

        except Exception as ex:
            logger.warning(f"Explainability generation partial failure: analysis_id={analysis_id} error={ex}")
            gradcam_overlay = img_rgb.copy()
            gradcam_pp_overlay = img_rgb.copy()
            segmentation_img = img_rgb.copy()
            tumor_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

        severity, severity_color = estimate_severity(confidence, tumor_percentage)
        logger.info(
            f"Inference completed: analysis_id={analysis_id} class='{class_name}' "
            f"confidence={confidence:.2f}% is_tumor={is_tumor} tta={use_tta} duration={inference_time:.4f}s"
        )

        return {
            "is_valid": True,
            "status": "COMPLETED",
            "pipeline_version": PIPELINE_VERSION,
            "provenance": get_model_provenance(),
            "analysis_id": analysis_id,
            "timestamp": timestamp,
            "tta_enabled": use_tta,
            "tta_passes": tta_passes if use_tta else 1,
            "colormap": cm_key,
            "img": img_rgb,
            "segmentation_img": segmentation_img,
            "tumor_mask": tumor_mask,
            "gradcam_overlay": gradcam_overlay,
            "gradcam_pp_overlay": gradcam_pp_overlay,
            "heatmap_raw": heatmap_raw,
            "heatmap_pp_raw": heatmap_pp_raw,
            "class_name": class_name,
            "confidence": confidence,
            "confidence_metric": "Ensemble Softmax Probability (Uncalibrated)",
            "uncertainty": uncertainty,
            "uncertainty_metric": "Normalized Shannon Entropy of Softmax Distribution",
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
        logger.error(f"Inference pipeline failure: analysis_id={analysis_id} error={e}", exc_info=True)
        return _fail_response("An internal error occurred during model inference.", "INFERENCE_ERROR")
