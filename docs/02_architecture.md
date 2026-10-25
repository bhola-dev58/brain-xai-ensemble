# 02 — System Architecture

## High-Level Architecture Diagram

```
                        ┌──────────────────────────────────────┐
                        │           USER / CLINICIAN            │
                        └─────────────┬────────────────────────┘
                                      │  Uploads CE-MRI Image
                        ┌─────────────▼──────────────────────┐
                        │       APPLICATION ENTRY POINT      │
                        │                                    │
                        │  ┌──────────────────────────────┐  │
                        │  │  FastAPI Workstation & REST  │  │
                        │  │  (app.py / scripts/run_api)  │  │
                        │  │  Port 8000                   │  │
                        │  └──────────────┬───────────────┘  │
                        └─────────────────┼──────────────────┘
                                          │  Calls predict_tumor_logic()
                                             │  Calls predict_tumor_logic()
                        ┌────────────────────▼───────────────────────────┐
                        │               src/inference.py                  │
                        │                                                  │
                        │  1. Preprocess: resize(224,224) + /255           │
                        │  2. Ensemble Classification                       │
                        │     ┌────────────┐   ┌─────────────┐            │
                        │     │ DenseNet121│   │ InceptionV3 │            │
                        │     │  37.4 MB   │   │  102.4 MB   │            │
                        │     └──────┬─────┘   └──────┬──────┘            │
                        │            └────── avg ──────┘                   │
                        │               (Soft Voting)                      │
                        │  3. class_idx = argmax(avg_pred)                 │
                        │  4. Grad-CAM backprop on DenseNet121             │
                        └────────────────────┬───────────────────────────┘
                                             │
                        ┌────────────────────▼───────────────────────────┐
                        │               src/processor.py                  │
                        │                                                  │
                        │  5. create_segmentation()                        │
                        │     - ROI mask from heatmap (τ=0.70)            │
                        │     - Otsu threshold inside ROI                  │
                        │     - Morphological cleanup (OPEN + CLOSE)       │
                        │     - Red overlay on tumor pixels                │
                        │  6. estimate_location()                          │
                        │     - Peak activation → lobe + hemisphere        │
                        │  7. estimate_severity()                          │
                        │     - confidence + area → severity level         │
                        └────────────────────┬───────────────────────────┘
                                             │
                        ┌────────────────────▼───────────────────────────┐
                        │              OUTPUT BUNDLE (dict)                │
                        │                                                  │
                        │  ✔ class_name     ✔ confidence                  │
                        │  ✔ gradcam_overlay ✔ segmentation_img           │
                        │  ✔ tumor_percentage ✔ location                  │
                        │  ✔ severity        ✔ inference_time             │
                        │  ✔ avg_pred (all 4 class probabilities)         │
                        └────────────────────────────────────────────────┘
```

---

## Module Dependency Map

```
app.py / scripts/run_api.py
  └── src/api.py              (FastAPI app + static PACS UI mount + REST API)
        └── src/inference.py  (model loading + predict_tumor_logic)
              ├── src/processor.py    (segmentation, location, severity)
              ├── src/report_generator.py (PDF generation)
              ├── src/database.py     (patient history persistence)
              └── src/config.py       (all constants + logger)

scripts/train_segmentation.py
  └── src/config.py           (IMG_SIZE_TRAIN, logger)
  └── (standalone — loads dataset, trains U-Net, saves model)

tests/
  ├── test_processor.py       → imports src/processor.py  (no models needed)
  └── test_inference.py       → imports src/inference.py  (loads real models)
```

---

## File-by-File Responsibilities

### `app.py` — Main Entrypoint
- Launches the unified FastAPI Clinical Diagnostic Studio workstation on `0.0.0.0:8000`.
- Serves both the interactive web PACS interface and the RESTful API endpoints.

---

### `src/config.py` — Central Configuration
- Defines all file paths, class names, image sizes, thresholds, and color codes.
- Sets up the root `BrainTumorXAI` logger used across all modules.
- Acts as a single source of truth — all other modules import from here.
- See [08 — Configuration](./08_configuration.md) for the full reference.

---

### `src/inference.py` — Inference Core
- Loads `densenet_best.h5` and `inception_best.h5` at **module import time** (once only).
- Dynamically finds the last 4D convolutional layer in DenseNet121 to build the Grad-CAM sub-model.
- `make_gradcam_heatmap()` — runs GradientTape backpropagation to produce a 2D activation heatmap.
- `predict_tumor_logic()` — the main inference handler: preprocesses input, runs ensemble, generates overlays, and returns a unified result dictionary.

---

### `src/processor.py` — Image Processing
- `create_segmentation()` — converts a raw Grad-CAM heatmap into a morphologically cleaned binary tumor mask and applies a semi-transparent red overlay.
- `estimate_location()` — maps the heatmap peak coordinate to a human-readable anatomical description.
- `estimate_severity()` — classifies severity into 4 levels based on confidence and tumor area.

---


---

### `src/api.py` — FastAPI REST Backend
- Defines `GET /api/health` (liveness check) and `POST /api/predict` (full inference).
- `_ndarray_to_base64()` — encodes numpy image arrays to base64 PNG strings for JSON transport.
- `_read_upload_as_numpy()` — decodes uploaded image bytes to a uint8 RGB numpy array.
- CORS is enabled for all origins (suitable for separate frontend clients).
- See [06 — API Reference](./06_api_reference.md) for full endpoint documentation.

---

### `scripts/run_api.py` — API & Workstation Server Launcher
- Runs the FastAPI app using Uvicorn on port 8000, serving both the REST API and the PACS Clinical Diagnostic Studio.

### `scripts/train_segmentation.py` — U-Net Training Script
- Loads image-mask pairs from `brain-tumor-2d-dataset/`.
- Trains a 4-layer U-Net with Dice + BCE combined loss.
- Saves the result to `models/segmentation_model.h5`.
- Run this once if you want a dedicated segmentation model (optional — the main pipeline uses Grad-CAM-guided morphological segmentation instead).

---

## Data Flow: Single Prediction Request

```
User uploads image
        │
        ▼
cv2.resize → (224, 224)          # Fixed input size for both models
        │
        ▼
np.expand_dims + /255            # Shape: (1, 224, 224, 3), float32 [0,1]
        │
        ├──► DenseNet121.predict()   → pred_dense: shape (1, 4)
        ├──► InceptionV3.predict()   → pred_inc:   shape (1, 4)
        │
        ▼
avg_pred = (pred_dense + pred_inc) / 2.0    # Soft voting
class_idx = argmax(avg_pred)
confidence = max(avg_pred) * 100
        │
        ▼
make_gradcam_heatmap(img_array)
  → GradientTape: gradient of class_channel w.r.t. last conv layer
  → pooled_grads = mean(grads, axis=(0,1,2))
  → heatmap = conv_outputs @ pooled_grads
  → normalize to [0, 1]
        │
        ▼
cv2.resize(heatmap, original_img_size)
clean: zero out activations < 0.30 (GRADCAM_CLEAN_THRESHOLD)
colorize: cv2.applyColorMap(..., COLORMAP_JET)
alpha-blend: overlay = img*(1-alpha) + heatmap_color*alpha
        │
        ▼ (if tumor detected)
create_segmentation(heatmap_raw, img)
  → ROI mask: heatmap > 0.70
  → Largest contour in ROI
  → Otsu threshold on grayscale pixels inside ROI
  → Morphological OPEN + CLOSE cleanup
  → Red overlay (opacity 0.35) on tumor pixels
  → area_pct = tumor_pixels / total_pixels * 100
        │
        ▼
estimate_location(heatmap_raw, img.shape)
  → peak y,x → vertical + horizontal zone + brain lobe
        │
        ▼
estimate_severity(confidence, tumor_percentage)
  → Returns (severity_label, hex_color)
        │
        ▼
Return result dict with all outputs
```
