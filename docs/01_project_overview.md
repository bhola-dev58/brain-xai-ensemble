# 01 — Project Overview

## What Is BrainTumorXAI?

**BrainTumorXAI** is a full-stack, production-ready medical imaging system that performs **automatic brain tumor detection, classification, and segmentation** from Contrast-Enhanced MRI (CE-MRI) scans. The system is built around the principle of **Explainable AI (XAI)**: every prediction is accompanied by a Grad-CAM visualization that visually explains *what* the model saw and *why* it made its decision.

The project was developed as a Final Year Major Project at **CMR Institute of Technology**, inspired by and built upon the methodology of:

> K. M. Hosny, M. A. Mohammed, R. A. Salama, and A. M. Elshewey, *"Explainable ensemble deep learning-based model for brain tumor detection and classification,"* Neural Computing and Applications, vol. 37, pp. 1289–1306, 2025.

---

## Project Goals

| Goal | Description |
|---|---|
| **Accurate Classification** | Classify brain MRI scans into 4 categories: No Tumor, Glioma, Meningioma, Pituitary |
| **Explainability (XAI)** | Generate Grad-CAM heatmaps to highlight the regions the model focused on |
| **Automated Segmentation** | Extract the tumor boundary from the heatmap without a separate segmentation model |
| **Clinical Usability** | Provide a clean, easy-to-use dashboard and REST API suitable for hospital workflows |
| **Lightweight Deployment** | CPU-only inference, no GPU required — runnable on standard hospital hardware |

---

## Core Novelty

What makes BrainTumorXAI different from typical classification-only systems:

1. **Weighted Soft-Voting Ensemble (95.59% Accuracy)** — Two complementary backbones (InceptionV3 at native $299\times299$ + DenseNet121 at $224\times224$) combine multi-scale receptive fields with dense feature reuse, minimizing false positives (99.33% healthy scan precision).

2. **Automated Brain Region Isolation** — Preprocessing automatically identifies and crops the cerebrum boundary using contour analysis, removing extraneous background noise and skull artifacts before feature extraction.

3. **Grad-CAM Guided Segmentation** — Most systems treat classification and segmentation as two completely independent tasks requiring two separate trained models. BrainTumorXAI reuses the classification network's Grad-CAM activation map to guide a lightweight morphological segmentation pipeline — eliminating the cost and complexity of a dedicated segmentation model.

4. **Anatomical Location Estimation** — The peak activation coordinate is mapped to brain anatomy (Frontal, Parietal, Occipital lobe × Left/Right × Superior/Inferior) and shown to the clinician directly.

5. **Clinical Severity & Borderline Flagging** — Combines classification confidence and tumor area percentage into a 5-level clinical indicator (High / Moderate / Low / Borderline / Uncertain) to flag low-confidence (<55%) predictions for manual radiological review.


---

## Problem Statement

Brain tumors are one of the most life-threatening neurological conditions. Early, accurate diagnosis directly determines survival outcomes. Current challenges:

- Manual MRI reading is time-consuming and subject to inter-radiologist variability.
- Standard deep learning classifiers are "black boxes" — clinicians cannot verify the reasoning behind a prediction.
- Dedicated segmentation models (e.g., U-Net) require paired mask datasets, expensive GPU training, and complex pipelines.

BrainTumorXAI addresses all three by combining ensemble classification with XAI-driven segmentation in a single, interpretable, CPU-runnable system.

---

## Tumor Classes

The system identifies 4 mutually exclusive categories:

| Class | Description |
|---|---|
| **No Tumor** | Normal brain scan — no tumor detected |
| **Glioma Tumor** | Arises from glial cells; most common and aggressive type |
| **Meningioma Tumor** | Grows from the meninges; typically benign but can cause pressure |
| **Pituitary Tumor** | Located at the pituitary gland; affects hormone regulation |

---

## System Outputs Per Scan

When a user uploads an MRI scan, the system produces:

1. **Classification** — Predicted tumor class (or "No Tumor")
2. **Confidence Score** — Ensemble prediction probability (0–100%)
3. **Prediction Breakdown** — Per-class probability bar chart across all 4 classes
4. **Grad-CAM Heatmap Overlay** — Color-coded visualization of model attention
5. **Segmentation Mask Overlay** — Red region marking the estimated tumor boundary
6. **Tumor Area %** — Area of the segmented region as a percentage of total scan area
7. **Anatomical Location** — Brain lobe and hemisphere (e.g., "Left Frontal Lobe (Superior)")
8. **Severity Level** — High / Moderate / Low / Uncertain
9. **Inference Time** — Processing time in seconds

---

## Technology Stack Summary

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow 2.21, Keras |
| Models | DenseNet121, InceptionV3 (pre-trained ImageNet weights, fine-tuned) |
| Explainability | Grad-CAM (via TensorFlow GradientTape) |
| Computer Vision | OpenCV 4.x, NumPy |
| Web PACS Workstation | Vanilla HTML5 / CSS3 / ES6+ |
| REST API | FastAPI 0.136 + Uvicorn |
| Testing | pytest |
| Containerization | Docker (CPU-only, Python 3.12-slim base) |
