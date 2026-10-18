# Project Executive Summary and Identity

## 1. Academic and Institutional Context

- **Project Title:** Explainable Brain Tumor Detection and Segmentation
- **Degree:** Bachelor of Engineering (B.E.) in Computer Science & Engineering
- **University:** Visvesvaraya Technological University (VTU), Belagavi, Karnataka, India
- **Course Code:** BCS685 - Project Phase 1 (Academic Year 2025-2026)
- **Institution:** Department of Computer Science and Engineering, CMR Institute of Technology (CMRIT), Bangalore - 560037
- **Project Guidance:**
  - Guide: Mrs. Anjali Gupta (Assistant Professor, Dept. of CSE, CMRIT)
  - Head of Department: Dr. Prem Kumar Ramesh (Professor & HOD, Dept. of CSE, CMRIT)
  - Principal: Dr. Sanjay Jain (Principal, CMRIT, Bangalore)
- **Student Team:**
  - Deeptanshu Dutta (USN: 1CR23CS050)
  - Abinash Mishra (USN: 1CR23CS014)
  - Bhola Yadav (USN: 1CR23CS044)
  - Aman Kumar Pathak (USN: 1CR23CS029)

---

## 2. Project Abstract and Executive Overview

The precise and timely diagnosis of brain tumors using Magnetic Resonance Imaging (MRI) is a highly critical, yet complex and time-consuming task for neuro-oncologists. While automated Computer-Aided Diagnosis (CAD) systems powered by Deep Learning have significantly accelerated this process, their inherent "black-box" nature fundamentally restricts clinical adoption: medical practitioners cannot responsibly act on opaque predictions without verifiable visual justifications.

To bridge this critical gap, this project develops a robust, transparent, and clinically deployable framework titled **"Explainable Brain Tumor Detection and Segmentation"**. 

Before the subsequent exploration of modern ConvNeXt backbones, the core architecture was established through an ensemble hybridization of two complementary convolutional neural networks:
1. **DenseNet121:** Capitalizes on dense inter-layer connectivity, facilitating maximum feature reuse, strong backpropagation gradient flow, and high-fidelity spatial activations ideal for visual explainability.
2. **InceptionV3:** Employs multi-scale factorized convolutions (1x1, 3x3, and asymmetric 1x7 / 7x1 kernels) to capture diverse tumor morphologies across variable receptive fields.

The framework operates on multi-class Contrast-Enhanced MRI (CE-MRI) scans, classifying images across four diagnostic categories:
- Class 0: Healthy Tissue / No Tumor
- Class 1: Glioma Tumor
- Class 2: Meningioma Tumor
- Class 3: Pituitary Tumor

Beyond raw multi-class classification, the defining contribution is the dual-layer Explainable Artificial Intelligence (XAI) and morphological segmentation pipeline:
- **Global Visual Saliency:** Gradient-weighted Class Activation Mapping (Grad-CAM and Grad-CAM++) extracts gradient signals from the final convolutional layers to generate heatmaps highlighting tumor pathology.
- **Skull-Artifact Suppression:** An automated brain-tissue contour mask strictly restricts heatmap activations to intracranial tissue, eliminating false activations on skull bones, dura mater, and scanner background artifacts.
- **Morphological Tumor Segmentation:** By coupling Otsu's adaptive thresholding within the Grad-CAM region of interest (ROI) with morphological opening and closing operators, the system delineates pixel-level tumor boundaries without requiring manual slice annotations during inference.
- **Clinical Severity Assessment:** An automated decision matrix combines classification confidence and intracranial tumor area percentage to compute severity (High, Moderate, Low, Borderline, Uncertain) and anatomical location (Cerebral Lobes and Hemispheres).

---

## 3. Key Milestone Performance (Pre-ConvNeXt Era)

The project progressed through systematic optimization stages:

| Development Stage | Primary Backbone(s) | Preprocessing / Strategy | Accuracy | Macro F1-Score |
|---|---|---|---|---|
| Stage 1: Baseline Frozen Backbones | DenseNet121 (77.30%), InceptionV3 (75.65%) | Standard Resizing (224x224), Soft Voting | 78.17% | 77.85% |
| Stage 2: Initial Report Prototype | DenseNet121 + InceptionV3 Hybrid Head | Kaggle Augmentation (8,177 images), Colab GPU | 97.63% | 97.63% |
| Stage 3: Phase 3 Full Fine-Tuning | DenseNet121 (224x224) + InceptionV3 (299x299) | OpenCV Brain Cropping, Label Smoothing 0.03 | 95.59% | 95.65% |
| Stage 4: Test-Time Augmentation (TTA) | 5 to 10 Pass Multiscale TTA Ensemble | Multi-angle affine sampling + Soft Voting | 95.82% - 98.88% | 95.85% - 98.80% |

---

## 4. Architectural Innovations and Core Value Proposition

1. **Dual-Model Feature Complementarity:** Dense feature aggregation from DenseNet121 compensates for InceptionV3's multi-scale receptive field bias, reducing false positives between parasellar meningiomas and pituitary adenomas.
2. **Clinically Trustworthy Explainability:** Saliency maps are not merely visual ornaments; they are mathematically gated by anatomical brain masks and used directly to seed the downstream segmentation engine.
3. **End-to-End Hospital Integration:** The system includes full DICOM (.dcm) medical image ingestion, automated window leveling, SQLite patient history logging, and automated generation of formal, downloadable PDF radiology reports.
