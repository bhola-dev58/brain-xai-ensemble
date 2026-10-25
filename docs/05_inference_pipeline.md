# 05 — Inference Pipeline

This document explains every step of the inference pipeline from raw image input to final output dictionary, as implemented in `src/inference.py` and `src/processor.py`.

---

## Entry Point

All inference goes through a single function:

```python
# src/inference.py
result = predict_tumor_logic(img)   # img: numpy array (H, W, 3), uint8
```

This function is called by the FastAPI application (`src/api.py`) to serve both the web PACS workstation and REST endpoints.

---

## Step 1: Input Validation and Normalization

```python
if img is None:
    return {"is_valid": False, "error": "Please upload an MRI image first."}

if img.dtype != np.uint8:
    img = np.uint8(np.clip(img, 0, 255))
```

- Rejects `None` inputs immediately with a user-friendly error message.
- Accepts float or integer images and safely clips + casts them to `uint8`.

---

## Step 2: Preprocessing for Model Input

```python
img_resized = cv2.resize(img, IMG_SIZE_CLASSIFY)          # (224, 224)
img_array = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
# Final shape: (1, 224, 224, 3)
```

Both models require a **4D tensor** with pixel values normalized to `[0.0, 1.0]`.

---

## Step 3: Ensemble Classification (Soft Voting)

```python
pred_dense = model_dense.predict(img_array, verbose=0)    # (1, 4) float32
pred_inc   = model_inc.predict(img_array, verbose=0)       # (1, 4) float32
avg_pred   = (pred_dense + pred_inc) / 2.0                # element-wise avg
```

Each model independently outputs a 4-element softmax probability vector. The vectors are averaged to produce the ensemble prediction.

```python
class_idx  = int(np.argmax(avg_pred))     # Index of the highest probability class
class_name = CLASSES[class_idx]           # e.g., "Glioma Tumor"
confidence = float(np.max(avg_pred)) * 100  # Percentage (0–100)
is_tumor   = class_idx != 0              # False if "No Tumor"
```

**Class index mapping:**

| `class_idx` | `class_name` |
|---|---|
| 0 | No Tumor |
| 1 | Glioma Tumor |
| 2 | Meningioma Tumor |
| 3 | Pituitary Tumor |

---

## Step 4: Grad-CAM Heatmap Generation

Implemented in `make_gradcam_heatmap()`:

```python
with tf.GradientTape() as tape:
    tape.watch(img_tensor)
    conv_outputs, preds = _grad_model(img_tensor)
    tape.watch(conv_outputs)
    class_channel = preds[:, pred_index]

grads = tape.gradient(class_channel, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Global Average Pooling of gradients
heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]  # Weighted sum of activation maps
heatmap = tf.squeeze(heatmap)
heatmap = tf.maximum(heatmap, 0)           # ReLU — only positive activations matter
heatmap = heatmap / tf.reduce_max(heatmap) # Normalize to [0, 1]
```

**Key concepts:**
- `conv_outputs` — the feature maps from the last 4D convolutional layer of DenseNet121
- `grads` — how much each feature map activation influenced the class output
- `pooled_grads` — global importance weight for each feature map channel
- Final heatmap — weighted combination: channels that drove the classification get high values

The result is a 2D numpy array (heatmap), usually much smaller than the input image (e.g., 7×7) — it gets resized back to the original resolution.

---

## Step 5: Heatmap Overlay on Original Image

```python
heatmap_resized = cv2.resize(heatmap_raw, (img.shape[1], img.shape[0]))

# Discard weak background activations below threshold
heatmap_cleaned = np.where(heatmap_resized > GRADCAM_CLEAN_THRESHOLD, heatmap_resized, 0)
# GRADCAM_CLEAN_THRESHOLD = 0.30

heatmap_uint8 = np.uint8(255 * heatmap_cleaned)
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

# Remove color tint from zero-activation pixels (avoid blue wash over normal areas)
heatmap_color[heatmap_cleaned == 0] = [0, 0, 0]

# Alpha-blended overlay (varying transparency based on activation strength)
alpha = (heatmap_cleaned * GRADCAM_OVERLAY_OPACITY)[:, :, np.newaxis]
# GRADCAM_OVERLAY_OPACITY = 0.65
gradcam_overlay = np.uint8(img * (1.0 - alpha) + heatmap_color * alpha)
```

**Color scale:** The JET colormap maps activation values:
- Blue → low activation (background)
- Green/Yellow → medium activation
- Red → highest activation (model focus)

---

## Step 6: Tumor Segmentation (if tumor detected)

Only executed when `is_tumor == True`. Implemented in `processor.create_segmentation()`:

### 6a. Resize and normalize heatmap
```python
hm = cv2.resize(heatmap_raw, (w, h)).astype(np.float32)
hm = hm / hm.max()    # Normalize to [0, 1]
```

### 6b. Create ROI mask from high-confidence activations
```python
roi_mask = np.uint8(hm > GRADCAM_ROI_THRESHOLD) * 255
# GRADCAM_ROI_THRESHOLD = 0.70 — only top 30% activations

# Morphological closing to fill holes in ROI
k_roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, k_roi)
```

### 6c. Keep only largest connected blob
```python
contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_roi = max(contours, key=cv2.contourArea)
```
Removes scattered attention points and keeps only the primary tumor region.

### 6d. Otsu threshold inside ROI
```python
roi_pixels = gray[roi_mask > 0]
otsu_thresh, _ = cv2.threshold(roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
tumor_mask = np.uint8((gray > otsu_thresh) & (roi_mask > 0)) * 255
```
Otsu's method finds the optimal intensity threshold that separates the bright tumor pixels from the darker surrounding tissue — computed only within the Grad-CAM-guided ROI, not the whole image.

### 6e. Morphological cleanup
```python
k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, k_clean, iterations=1)   # Remove noise
tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, k_clean, iterations=2)  # Fill holes
```

### 6f. Gaussian blur + re-threshold for smooth edges
```python
tumor_mask = cv2.GaussianBlur(tumor_mask, (7, 7), 0)
_, tumor_mask = cv2.threshold(tumor_mask, 127, 255, cv2.THRESH_BINARY)
```

### 6g. Fallback check
```python
area_pct = (np.sum(tumor_mask > 0) / (h * w)) * 100
if area_pct == 0 or area_pct > GRADCAM_MAX_TUMOR_PCT_FALLBACK:  # > 25%
    tumor_mask = roi_mask    # Fall back to raw ROI mask
```
If the result is empty or implausibly large (>25% of scan area), the system falls back to using the raw ROI mask directly.

### 6h. Semi-transparent red overlay
```python
red_layer[:, :] = COLOR_TUMOR_SEGMENT   # [220, 40, 40] — deep red
opacity_base = 1.0 - SEGMENTATION_OPACITY  # 0.65
result[tumor_px] = result[tumor_px] * opacity_base + red_layer[tumor_px] * SEGMENTATION_OPACITY
```

---

## Step 7: Location Estimation

Implemented in `processor.estimate_location()`:

```python
y_center, x_center = np.unravel_index(np.argmax(hm), hm.shape)

v_pos = "Superior" if y_center < h/2 else "Inferior"
h_pos = "Left" if x_center < w/2 else "Right"

if y_center < h * 0.4:
    lobe = "Frontal Lobe"
elif y_center < h * 0.7:
    lobe = "Parietal Lobe"
else:
    lobe = "Occipital Lobe"

return f"{h_pos} {lobe} ({v_pos})"
# e.g., "Left Frontal Lobe (Superior)"
```

**Mapping logic:**
- Top 40% of image → Frontal Lobe
- Middle 40-70% → Parietal Lobe
- Bottom 30% → Occipital Lobe
- Left half → Left hemisphere; Right half → Right hemisphere
- Top half → Superior; Bottom half → Inferior

---

## Step 8: Severity Estimation

Implemented in `processor.estimate_severity()`:

```python
if confidence > 95 and tumor_percentage > 5:
    return "High", "#ef4444"       # Red
elif confidence > 80 and tumor_percentage > 3:
    return "Moderate", "#f59e0b"   # Amber
elif confidence > 60:
    return "Low", "#22c55e"        # Green
else:
    return "Uncertain", "#6b7280"  # Gray
```

| Severity | Confidence Threshold | Area Threshold | Color |
|---|---|---|---|
| High | > 95% | > 5% | 🔴 Red |
| Moderate | > 80% | > 3% | 🟡 Amber |
| Low | > 60% | any | 🟢 Green |
| Uncertain | ≤ 60% | any | ⚫ Gray |

---

## Final Return Dictionary

```python
return {
    "is_valid":         True,
    "img":              img,                  # Original image (numpy)
    "segmentation_img": segmentation_img,     # Segmented overlay image
    "gradcam_overlay":  gradcam_overlay,      # Heatmap overlay image
    "class_name":       class_name,           # e.g., "Glioma Tumor"
    "confidence":       confidence,           # e.g., 87.34 (float, 0-100)
    "is_tumor":         is_tumor,             # bool
    "inference_time":   inference_time,       # float (seconds)
    "tumor_percentage": tumor_percentage,     # float (0-100)
    "location":         location,             # e.g., "Left Frontal Lobe (Superior)"
    "severity":         severity,             # e.g., "High"
    "severity_color":   severity_color,       # e.g., "#ef4444"
    "avg_pred":         avg_pred[0],          # numpy array shape (4,)
    "classes":          CLASSES,              # list of 4 class names
    "class_idx":        class_idx             # int 0–3
}
```

---

## Error Handling

The pipeline has two levels of error handling:

1. **Outer try/except** — catches any critical failure in preprocessing or model prediction. Returns `{"is_valid": False, "error": "..."}`.
2. **Inner try/except** (around Grad-CAM + segmentation) — if visualization fails, the main classification result is still returned. The overlay images fall back to copies of the original image.

This ensures the system **always returns a classification result** even if visualization generation fails.
