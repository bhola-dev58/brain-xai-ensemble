# 08 — Configuration Reference

All project-wide constants, file paths, model settings, and color codes are centralized in `src/config.py`. Every other module imports from here — no magic numbers or hardcoded paths appear elsewhere.

---

## File: `src/config.py`

### Logging

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BrainTumorXAI")
```

- Logger name: `BrainTumorXAI`
- All modules import and use the same `logger` instance.
- Log format: `2025-01-01 10:30:00 [INFO] BrainTumorXAI: message here`

---

### File Paths

```python
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR      = os.path.join(BASE_DIR, "models")
DENSENET_PATH   = os.path.join(MODELS_DIR, "densenet_best.h5")
INCEPTION_PATH  = os.path.join(MODELS_DIR, "inception_best.h5")
```

| Constant | Resolved Path | Description |
|---|---|---|
| `BASE_DIR` | `/path/to/Brain_Tumor_Project/` | Project root directory |
| `MODELS_DIR` | `<BASE_DIR>/models/` | Directory containing `.h5` weight files |
| `DENSENET_PATH` | `<MODELS_DIR>/densenet_best.h5` | DenseNet121 trained weights |
| `INCEPTION_PATH` | `<MODELS_DIR>/inception_best.h5` | InceptionV3 trained weights |

Paths are computed dynamically relative to `config.py`, so the project can run from any working directory.

---

### Classification Settings

```python
CLASSES = ['No Tumor', 'Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']
IMG_SIZE_CLASSIFY = (224, 224)   # Inference input resolution
IMG_SIZE_TRAIN    = (256, 256)   # U-Net training input resolution
```

| Constant | Value | Used In |
|---|---|---|
| `CLASSES` | List of 4 class names | `inference.py`, `api.py` |
| `IMG_SIZE_CLASSIFY` | `(224, 224)` | `inference.py` — resizes input before model prediction |
| `IMG_SIZE_TRAIN` | `(256, 256)` | `train_segmentation.py` — U-Net input size |

---

### Grad-CAM Thresholds

| Constant | Value | Purpose |
|---|---|---|
| `GRADCAM_CLEAN_THRESHOLD` | `0.30` | Zeroes out heatmap activations below 30% — removes weak background noise from the JET overlay |
| `GRADCAM_ROI_THRESHOLD` | `0.70` | Only the top 30% activation area is used to define the tumor ROI for segmentation |
| `GRADCAM_MAX_TUMOR_PCT_FALLBACK` | `25.0` | If the segmented tumor area exceeds 25% of the scan, fall back to using the raw ROI mask (prevents implausible over-segmentation) |
| `GRADCAM_OVERLAY_OPACITY` | `0.65` | Transparency of the Grad-CAM heatmap overlay on the original image (0=invisible, 1=fully opaque) |

**Tuning guidance:**
- Increase `GRADCAM_CLEAN_THRESHOLD` (e.g., `0.40`) for cleaner overlays but less detail.
- Decrease `GRADCAM_ROI_THRESHOLD` (e.g., `0.50`) to expand the segmentation area.
- Increase `GRADCAM_OVERLAY_OPACITY` (e.g., `0.80`) for a stronger heatmap overlay.

---

### Segmentation Settings

| Constant | Value | Purpose |
|---|---|---|
| `SEGMENTATION_OPACITY` | `0.35` | Transparency of the red tumor overlay on the segmentation image (35% red, 65% original) |

---

### Color Palette

```python
COLOR_TUMOR_SEGMENT   = [220, 40, 40]    # Deep Red (RGB list — used with OpenCV/NumPy)
COLOR_SEVERITY_HIGH   = "#ef4444"        # Red hex
COLOR_SEVERITY_MODERATE = "#f59e0b"      # Amber hex
COLOR_SEVERITY_LOW    = "#22c55e"        # Green hex
COLOR_SEVERITY_UNCERTAIN = "#6b7280"     # Gray hex
```

| Constant | Color | Used For |
|---|---|---|
| `COLOR_TUMOR_SEGMENT` | Deep Red `[220,40,40]` | Red fill applied to tumor pixels in segmentation overlay |
| `COLOR_SEVERITY_HIGH` | `#ef4444` | Severity text and confidence bar when High |
| `COLOR_SEVERITY_MODERATE` | `#f59e0b` | Severity text and confidence bar when Moderate |
| `COLOR_SEVERITY_LOW` | `#22c55e` | Severity text and confidence bar when Low |
| `COLOR_SEVERITY_UNCERTAIN` | `#6b7280` | Severity text when confidence is too low to classify |

---

## Changing Configuration

All thresholds and colors can be changed by editing only `src/config.py`. No other file needs to be modified.

Example — make the Grad-CAM overlay more transparent:
```python
# Before
GRADCAM_OVERLAY_OPACITY = 0.65

# After
GRADCAM_OVERLAY_OPACITY = 0.40   # More transparent overlay
```

Example — change tumor overlay color to cyan:
```python
# Before
COLOR_TUMOR_SEGMENT = [220, 40, 40]

# After
COLOR_TUMOR_SEGMENT = [0, 200, 200]   # Cyan
```
