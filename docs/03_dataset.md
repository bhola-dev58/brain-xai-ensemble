# 03 — Dataset

## Overview

The models were trained and evaluated on the **Brain Tumor MRI Dataset** — a publicly available CE-MRI scan collection containing four classes. The dataset is stored locally in the `brain-tumor-2d-dataset/` directory.

---

## Directory Structure

```
brain-tumor-2d-dataset/
├── image/
│   ├── 0/          # No Tumor images
│   ├── 1/          # Glioma Tumor images
│   ├── 2/          # Meningioma Tumor images
│   └── 3/          # Pituitary Tumor images
│
└── mask/           # Ground truth binary masks (used for U-Net training only)
    ├── 1/          # Glioma masks
    ├── 2/          # Meningioma masks
    └── 3/          # Pituitary masks
```

> **Note:** The `mask/` directory does **not** contain a class `0/` folder — no-tumor scans do not have segmentation masks since there is nothing to segment.

---

## Class Distribution

| Class Index | Class Name | Typical Count | Notes |
|---|---|---|---|
| `0` | No Tumor | ~395 (val) | Normal brain scan |
| `1` | Glioma Tumor | ~194 (val) | Most aggressive |
| `2` | Meningioma Tumor | ~299 (val) | Benign but variable |
| `3` | Pituitary Tumor | ~298 (val) | Often well-defined boundary |

The **70/30 split** is applied: 70% training, 30% validation.

---

## Preprocessing Pipeline

Preprocessing is handled by Keras `ImageDataGenerator`. The same configuration is used for both models during training and validation evaluation.

```python
ImageDataGenerator(
    rescale          = 1./255,      # Normalize pixel values to [0, 1]
    rotation_range   = 20,          # Random rotation up to ±20 degrees
    horizontal_flip  = True,        # Random horizontal mirroring
    zoom_range       = 0.05,        # Random zoom up to ±5%
    validation_split = 0.30         # 30% reserved for validation
)
```

### Image Resize

All images are resized to **(224, 224)** before being passed to either model.

```python
flow_from_directory(
    dataset_path,
    target_size = (224, 224),
    batch_size  = 10,
    class_mode  = 'categorical',
    subset      = 'training'  # or 'validation'
)
```

### Normalization at Inference

During inference (not training), preprocessing is done directly in `inference.py`:

```python
img_resized = cv2.resize(img, (224, 224))
img_array = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
```

---

## Dataset Augmentation Strategy

Augmentation is applied **only to training images**. It is intentionally conservative to avoid distorting anatomical features that are clinically meaningful:

| Augmentation | Value | Rationale |
|---|---|---|
| `rescale` | 1/255 | Standardize to [0,1] float range |
| `rotation_range` | 20° | Handle slight head tilt in scans |
| `horizontal_flip` | True | Simulate left/right brain symmetry |
| `zoom_range` | 5% | Handle minor zoom variance in different MRI machines |

> **Not used:** Shear, width/height shift, brightness adjustments — these are avoided to maintain anatomical validity.

---

## Mask Dataset (for U-Net Training)

The mask dataset has a specific naming convention:

- Image: `brain-tumor-2d-dataset/image/1/Tr-gl_0001.jpg`
- Mask:  `brain-tumor-2d-dataset/mask/1/Tr-gl_0001_m.jpg`

The suffix `_m` before the extension marks the mask counterpart. The `train_segmentation.py` script handles this automatically:

```python
base, ext = os.path.splitext(fname)
mask_path = os.path.join(mask_dir, f"{base}_m{ext}")
```

Masks are:
- Read as grayscale (`cv2.IMREAD_GRAYSCALE`)
- Resized to **(256, 256)** (note: different from classification size of 224)
- Binarized: `mask = (mask > 127).astype(np.float32)`

---

## Test Images

The `test_images/` directory contains pre-selected MRI samples included directly in the PACS workstation and test suite as quick-load benchmark examples:

| File | Class |
|---|---|
| `Tr-me_0025.jpg` | Meningioma |
| `Tr-me_0070.jpg` | Meningioma |
| `Tr-me_0080.jpg` | Meningioma |
| `Tr-pi_0050.jpg` | Pituitary |

These files are also used by `tests/test_inference.py` as the integration test input image (`Tr-me_0025.jpg`).

---

## Supported Input Format (at Inference)

At runtime, the system accepts:

- **JPEG** (`.jpg`, `.jpeg`)
- **PNG** (`.png`)
- Any bit depth that OpenCV can decode
- Color (RGB) or grayscale — grayscale is handled gracefully in `processor.py`
- Any resolution — resized internally to 224×224

**Not supported:** DICOM (`.dcm`), NIfTI (`.nii`) — raw hospital formats must be converted to JPEG/PNG before upload.
