# Master Technical Documentation: Explainable Brain Tumor CAD Framework

Welcome to the exhaustive technical documentation repository for the **Explainable Brain Tumor Detection and Segmentation** framework.

This directory compiles full algorithmic, mathematical, architectural, and clinical details established during the major project phase prior to ConvNeXtSmall exploration, synthesized directly from the official college project report (`Report Body_merged.pdf`) and the underlying project codebase.

---

## Complete Directory Navigation and Chapter Map

| File | Document Title | Primary Topics and Highlights |
|---|---|---|
| [00_executive_summary_and_project_identity.md](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/00_executive_summary_and_project_identity.md) | **Executive Summary and Identity** | VTU/CMRIT academic credentials, student team, guide details, project abstract, core innovations, and milestone summary. |
| [01_clinical_background_and_problem_formulation.md](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/01_clinical_background_and_problem_formulation.md) | **Clinical Background & Problem Formulation** | Neuro-oncological disease burden, 4 target classes (Glioma, Meningioma, Pituitary, Normal), manual MRI bottlenecks, and the black-box AI dilemma. |
| [02_dataset_architecture_and_preprocessing.md](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/02_dataset_architecture_and_preprocessing.md) | **Dataset Architecture & Preprocessing** | Cheng Figshare / Sartaj Kaggle CE-MRI split counts, OpenCV skull cropping, intensity normalization formulas, augmentation parameters, and balanced class weights. |
| [03_deep_learning_architectures_and_transfer_learning.md](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/03_deep_learning_architectures_and_transfer_learning.md) | **Deep Learning Backbones & Transfer Learning** | DenseNet121 dense connectivity equations, InceptionV3 factorized convolutions, custom dual pooling classification head, 3-Phase training strategy, and Adam optimizer math. |
| [04_ensemble_learning_and_decision_fusion.md](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/04_ensemble_learning_and_decision_fusion.md) | **Ensemble Learning & Decision Fusion** | Unweighted vs weighted soft-voting formulas, Test-Time Augmentation (TTA) mechanics, Shannon Entropy, and normalized prediction uncertainty quantification. |
| [05_explainable_ai_gradcam_and_lime.md](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/05_explainable_ai_gradcam_and_lime.md) | **Explainable AI (Grad-CAM, Grad-CAM++, LIME)** | GradientTape backpropagation, alpha importance weights, Top-K channel pooling, Grad-CAM++ higher-order derivatives, Brain-Mask anatomical gating, and LIME super-pixel theory. |
| [06_morphological_segmentation_pipeline.md](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/06_morphological_segmentation_pipeline.md) | **Morphological Segmentation Pipeline** | Weakly supervised tumor boundary extraction, dynamic 70th percentile ROI thresholding, intra-ROI Otsu thresholding math, morphological opening/closing, and intracranial area % calculation. |
| [07_clinical_severity_and_anatomical_localization.md](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/07_clinical_severity_and_anatomical_localization.md) | **Clinical Severity & Anatomical Localization** | Deterministic severity decision matrix (High, Moderate, Low, Borderline, Uncertain), Cartesian lobe/hemisphere mapping, DICOM ingestion, SQLite schema, and PDF reports. |
| [08_evaluation_metrics_and_benchmark_results.md](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/08_evaluation_metrics_and_benchmark_results.md) | **Evaluation Metrics & Benchmark Results** | Complete definitions for all 8 metrics (Accuracy, Precision, Recall, Specificity, F1, Balanced Accuracy, Dice, IoU), 4x4 confusion matrix, per-class reports, and literature comparison (10 papers). |
| [09_algorithms_and_pseudocode.md](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/09_algorithms_and_pseudocode.md) | **Complete Algorithms and Pseudocode** | Six formal pseudocode algorithms covering the entire pipeline from raw image ingestion to clinical PDF generation. |
| [10_system_implementation_and_ui_workflow.md](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/10_system_implementation_and_ui_workflow.md) | **System Implementation & UI Workflow** | Complete Gradio Diagnostic Studio layout (6 tabs), FastAPI REST backend specifications, base64 overlays, and end-to-end execution trace. |
| [11_research_gaps_future_scope_and_references.md](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/11_research_gaps_future_scope_and_references.md) | **Research Gaps, Future Scope & References** | 4 critical research gaps, modern ConvNeXt transition roadmap, 3D scan aggregation, multi-parametric MRI, and formal IEEE bibliographic references ([1] to [8]). |

---

## Key Mathematical Reference Guide

### 1. Balanced Class Weights
```text
Weight(c) = N_total / [ N_classes * N_samples(c) ]
```

### 2. Dual-Backbone Soft Voting
```text
P_ensemble(c) = 0.5 * P_DenseNet(c) + 0.5 * P_Inception(c)
Winning_Class = argmax_{c} [ P_ensemble(c) ]
Confidence_Percentage = max_{c} [ P_ensemble(c) ] * 100.0%
```

### 3. Grad-CAM Neuron Importance Weights
```text
alpha_k^c = [ 1 / (u * v) ] * sum_{i=1}^u sum_{j=1}^v [ d(y^c) / d(A^k_{i, j}) ]
L_GradCAM^c = ReLU( sum_{k=1}^K [ alpha_k^c * A^k ] )
```

### 4. Brain-Mask Anatomical Gating
```text
HM_gated(x, y) = HM_resized(x, y) * Mask_brain(x, y)
```

### 5. Otsu Optimal Thresholding (Maximizing Between-Class Variance)
```text
sigma_B^2(t) = omega_0(t) * omega_1(t) * [ mu_0(t) - mu_1(t) ]^2
t_opt = argmax_{0 <= t < 255} [ sigma_B^2(t) ]
```

### 6. Intracranial Relative Tumor Burden (rho_tumor)
```text
rho_tumor = [ Tumor_Pixels / Brain_Pixels ] * 100.0%
```

### 7. Dice Similarity Coefficient and IoU
```text
Dice = [ 2 * TP ] / [ 2 * TP + FP + FN ]
IoU  = [ TP ] / [ TP + FP + FN ]
```

### 8. Normalized Shannon Prediction Uncertainty
```text
Entropy = - sum_{c=0}^3 [ P(c) * log2(P(c)) ]
Uncertainty_Percentage = [ Entropy / log2(4) ] * 100.0%
```
