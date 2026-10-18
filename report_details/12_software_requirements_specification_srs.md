# Software Requirements Specification (SRS)

## 1. Introduction and Operational Purpose

This document provides a formal Software Requirements Specification (SRS) for the **Explainable Brain Tumor Detection and Segmentation** system. It specifies the functional requirements, non-functional performance guarantees, interface definitions, and hardware/software constraints governing the design, implementation, and deployment of the CAD platform.

---

## 2. Functional Requirements (FR)

### FR-01: Multi-Format Medical Image Ingestion
- **Description:** The system must accept 2D MRI slices in standard digital formats (JPEG, PNG, TIFF) and raw clinical hospital DICOM files (`.dcm`).
- **Input Parameters:** File stream (multipart upload or file path).
- **Processing:** Extract pixel data, validate non-zero dimensions, auto-convert multi-channel representations to normalized grayscale luminance, and parse DICOM tags.
- **Output:** Normalized 3D NumPy array `(H, W, 3)` of `uint8` values [0, 255] and extracted patient metadata dictionary.

### FR-02: Automated Intracranial Skull Stripping
- **Description:** The system must identify the brain parenchyma and remove non-cerebral background, calvarial bone artifacts, and external scanner annotations.
- **Processing:** Apply Gaussian filter (sigma = 1.5), Otsu binary thresholding, external contour detection, and extreme Cartesian coordinate extraction.
- **Output:** Tightly cropped intracranial scan `I_cropped` and binary parenchyma mask `Mask_brain`.

### FR-03: Dual-Model Soft-Voting Ensemble Classification
- **Description:** The system must independently evaluate input scans using DenseNet121 and InceptionV3 backbones and fuse their predictions via soft voting.
- **Processing:** Feed resized inputs `(224 x 224 x 3)` to DenseNet121 and `(299 x 299 x 3)` to InceptionV3; compute the arithmetic or weighted mean of their 4-class softmax probability vectors.
- **Output:** Predicted class label `c_hat in {No Tumor, Glioma, Meningioma, Pituitary}`, confidence score (0.0% to 100.0%), and binary flag `is_tumor`.

### FR-04: Diagnostic Uncertainty Quantification
- **Description:** The system must compute the normalized Shannon entropy of the ensemble probability distribution to alert clinicians to ambiguous scans.
- **Output:** Normalized uncertainty metric (0.0% to 100.0%) and automated safety flag if uncertainty exceeds 40.0%.

### FR-05: Brain-Gated Visual Explainability (Grad-CAM & Grad-CAM++)
- **Description:** The system must generate visual class activation heatmaps highlighting the exact pathological regions driving the classification.
- **Processing:** Execute gradient backpropagation via TensorFlow `GradientTape` from the predicted class logit to the final convolutional feature maps (`relu` in DenseNet121); apply Top-K (25%) channel pooling; multiply by `Mask_brain` to guarantee zero outside-brain activation; apply adaptive Otsu thresholding.
- **Output:** High-resolution Jet colormap overlay (Standard Grad-CAM) and Magma colormap overlay (Grad-CAM++).

### FR-06: Weakly-Supervised Morphological Segmentation
- **Description:** For scans classified as containing a tumor, the system must extract pixel-level boundaries without requiring ground-truth mask supervision.
- **Processing:** Threshold the top 30% of intracranial heatmap activations (`tau >= 0.45`); perform Otsu thresholding on grayscale pixels within the ROI; execute morphological opening (1 iteration) and closing (3 iterations) using a 5x5 elliptical structuring element; isolate the largest connected component.
- **Output:** Binary tumor mask `Mask_tumor`, alpha-blended medical red overlay image, and relative intracranial tumor burden percentage `rho_tumor`.

### FR-07: Clinical Severity Assessment & Anatomical Localization
- **Description:** The system must evaluate the clinical urgency and anatomical location of the detected lesion.
- **Processing:** Apply the deterministic Severity Matrix linking confidence score and `rho_tumor` into tiers (`High`, `Moderate`, `Low`, `Borderline`, `Uncertain`); map peak activation coordinates to cerebral lobes (Frontal, Parietal, Occipital) and hemispheres (Left, Right).
- **Output:** Severity status badge with hex color code and anatomical location string.

### FR-08: Persistent Database Logging and PDF Radiology Report Export
- **Description:** Every successful diagnostic evaluation must be logged for auditability, and an official PDF report must be generated on demand.
- **Processing:** Store scan metadata, confidence, class, location, severity, and timestamp in SQLite (`patient_history.db`); compile a two-page clinical PDF report via ReportLab containing patient demographics, numerical findings, and a quad-view visual panel.

---

## 3. Non-Functional Requirements (NFR)

| ID | Requirement Category | Metric Specification | Implementation Guarantee |
|---|---|---|---|
| **NFR-01** | **Inference Latency** | End-to-end processing time < 250 milliseconds per 2D slice on GPU (or < 1.2s on modern multi-core CPU). | Lightweight convolutional backbones, vectorization in NumPy/OpenCV, and pre-compiled model inference graphs. |
| **NFR-02** | **Diagnostic Accuracy** | Overall multi-class classification accuracy > 95.0%; recall on malignant Gliomas > 98.0%. | 3-Phase progressive fine-tuning, balanced class weighting, and ensemble soft voting. |
| **NFR-03** | **Memory Footprint** | System operational footprint < 4.0 GB RAM; combined model storage on disk < 300 MB. | Compact parameter count (DenseNet121 ~7M, InceptionV3 ~21M parameters). |
| **NFR-04** | **Data Privacy & Compliance** | Strict anonymization of Protected Health Information (PHI) compliant with HIPAA / GDPR guidelines. | DICOM parser de-identifies patient records; local SQLite database runs entirely on-premise without external telemetry. |
| **NFR-05** | **System Portability** | Cross-platform execution across standard Linux (Ubuntu 20.04/22.04 LTS), Windows 10/11, and macOS. | Containerized Python environment with deterministic dependency pinning. |
| **NFR-06** | **Fault Tolerance** | Zero unhandled crashes on corrupt, truncated, non-square, or completely dark MRI scans. | Try-catch exception wrappers returning structured error JSON and safe fallback masks. |

---

## 4. Hardware and Computational Environment

### 4.1 Development and Training Specifications
- **Hardware Platform:** Google Colaboratory Cloud GPU Environment.
- **Accelerator:** NVIDIA Tesla T4 GPU (16 GB GDDR6 VRAM) or NVIDIA A100-SXM4 (40 GB VRAM).
- **Host Memory:** 16 GB to 32 GB High-Speed System RAM.
- **Persistent Storage:** Cloud Storage / Local SSD with >= 50 GB allocated space.

### 4.2 Target Clinical Deployment Workstation (Minimum Prerequisites)
- **Processor:** Intel Core i5 / AMD Ryzen 5 (6 cores, 3.2 GHz or higher).
- **RAM:** 8 GB DDR4 RAM minimum (16 GB recommended for concurrent multi-user PACS queues).
- **Graphics Card:** NVIDIA GTX 1660 / RTX 3050 (4 GB VRAM) for real-time GPU acceleration, or CPU fallback mode.
- **Operating System:** Linux Ubuntu 20.04+, Windows 10/11 (64-bit).

---

## 5. Software Stack and Library Dependencies

```text
[Frontend Presentation Layer]
├── Gradio (v4.x)               ── Interactive Clinical Studio UI
└── Custom CSS (assets/)        ── Medical dark-mode design system

[Application & API Layer]
├── Python (v3.10 / v3.12)      ── Core application runtime
├── FastAPI (v0.110+)           ── High-performance asynchronous REST API
├── Uvicorn (v0.28+)            ── ASGI production web server
└── SQLite3                     ── Serverless local case history database

[Deep Learning & Vision Layer]
├── TensorFlow (v2.10 - v2.16)  ── Neural network execution & GradientTape
├── Keras (v2.10 - v3.x)        ── Model graph compilation & callbacks
├── OpenCV (cv2, v4.8+)         ── Contour extraction, Otsu, morphology
└── NumPy (v1.24 - v1.26)       ── High-performance tensor manipulations

[Medical Imaging & Reporting Layer]
├── PyDICOM (v2.4+)             ── Native hospital DICOM (.dcm) parser
├── ReportLab (v4.1+)           ── Programmatic PDF radiology report engine
└── Pillow (PIL, v10.0+)        ── Digital image format handling
```
