# Master Technical Documentation: Explainable Brain Tumor CAD Framework

Welcome to the exhaustive technical documentation repository for the **Explainable Brain Tumor Detection and Segmentation** framework.

This directory compiles full algorithmic, mathematical, architectural, and clinical details established during the major project phase prior to ConvNeXtSmall exploration, synthesized directly from the official college project report (`Report Body_merged.pdf`) and the underlying project codebase.

---

## Complete Directory Navigation and Chapter Map (16 Modular Chapters)

| Chapter | Document Title | Primary Topics and Highlights |
|---|---|---|
| [00](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/00_executive_summary_and_project_identity.md) | **Executive Summary and Identity** | VTU/CMRIT academic credentials, student team, guide details, project abstract, core innovations, and milestone summary. |
| [01](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/01_clinical_background_and_problem_formulation.md) | **Clinical Background & Problem Formulation** | Neuro-oncological disease burden, 4 target classes (Glioma, Meningioma, Pituitary, Normal), manual MRI bottlenecks, and the black-box AI dilemma. |
| [02](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/02_dataset_architecture_and_preprocessing.md) | **Dataset Architecture & Preprocessing** | Cheng Figshare / Sartaj Kaggle CE-MRI split counts, OpenCV skull cropping, intensity normalization formulas, augmentation parameters, and balanced class weights. |
| [03](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/03_deep_learning_architectures_and_transfer_learning.md) | **Deep Learning Backbones & Transfer Learning** | DenseNet121 dense connectivity equations, InceptionV3 factorized convolutions, custom dual pooling classification head, 3-Phase training strategy, and Adam optimizer math. |
| [04](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/04_ensemble_learning_and_decision_fusion.md) | **Ensemble Learning & Decision Fusion** | Unweighted vs weighted soft-voting formulas, Test-Time Augmentation (TTA) mechanics, Shannon Entropy, and normalized prediction uncertainty quantification. |
| [05](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/05_explainable_ai_gradcam_and_lime.md) | **Explainable AI (Grad-CAM, Grad-CAM++, LIME)** | GradientTape backpropagation, alpha importance weights, Top-K channel pooling, Grad-CAM++ higher-order derivatives, Brain-Mask anatomical gating, and LIME super-pixel theory. |
| [06](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/06_morphological_segmentation_pipeline.md) | **Morphological Segmentation Pipeline** | Weakly supervised tumor boundary extraction, dynamic 70th percentile ROI thresholding, intra-ROI Otsu thresholding math, morphological opening/closing, and intracranial area % calculation. |
| [07](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/07_clinical_severity_and_anatomical_localization.md) | **Clinical Severity & Anatomical Localization** | Deterministic severity decision matrix (High, Moderate, Low, Borderline, Uncertain), Cartesian lobe/hemisphere mapping, DICOM ingestion, SQLite schema, and PDF reports. |
| [08](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/08_evaluation_metrics_and_benchmark_results.md) | **Evaluation Metrics & Benchmark Results** | Complete definitions for all 8 metrics (Accuracy, Precision, Recall, Specificity, F1, Balanced Accuracy, Dice, IoU), 4x4 confusion matrix, per-class reports, and literature comparison (10 papers). |
| [09](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/09_algorithms_and_pseudocode.md) | **Complete Algorithms and Pseudocode** | Six formal pseudocode algorithms covering the entire pipeline from raw image ingestion to clinical PDF generation. |
| [10](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/10_system_implementation_and_ui_workflow.md) | **System Implementation & UI Workflow** | Complete Gradio Diagnostic Studio layout (6 tabs), FastAPI REST backend specifications, base64 overlays, and end-to-end execution trace. |
| [11](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/11_research_gaps_future_scope_and_references.md) | **Research Gaps, Future Scope & References** | 4 critical research gaps, modern ConvNeXt transition roadmap, 3D scan aggregation, multi-parametric MRI, and formal IEEE bibliographic references ([1] to [8]). |
| [12](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/12_software_requirements_specification_srs.md) | **Software Requirements Specification (SRS)** | Complete functional requirements (FR-01 to FR-08), non-functional requirements (NFR-01 to NFR-06), hardware profiles, and dependency graph. |
| [13](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/13_testing_verification_and_quality_assurance.md) | **Testing, Verification & Quality Assurance** | 3-tier testing hierarchy (Unit, Integration, Full-Stack), 12 formal test cases matrix, boundary checks, and defensive programming exception guards. |
| [14](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/14_clinical_case_studies_and_qualitative_xai.md) | **Clinical Case Studies & Qualitative XAI** | Detailed qualitative audits of 4 real patient cases (Glioma, Meningioma, Pituitary, Normal brain), necrosis handling, and Clever Hans artifact avoidance. |
| [15](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/report_details/15_mathematical_proofs_and_derivations_appendix.md) | **Mathematical Proofs & Derivations Appendix** | Rigorous mathematical derivations: DenseNet gradient flow, Taylor series Grad-CAM/Grad-CAM++ alpha weights, Otsu variance equivalence, Dice/IoU algebra, and Shannon entropy bounds. |

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
Dice = [ 2 * IoU ] / [ 1 + IoU ]
IoU  = Dice / [ 2 - Dice ]
```

### 8. Normalized Shannon Prediction Uncertainty
```text
Entropy = - sum_{c=0}^3 [ P(c) * log2(P(c)) ]
Uncertainty_Percentage = [ Entropy / log2(4) ] * 100.0%
```
