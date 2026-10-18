# System Implementation and UI Workflow

## 1. System Architecture and Clinical Integration

To ensure the CAD framework is not merely a theoretical notebook script but an operational medical software suite, the project implements a complete dual-tier deployment architecture:
1. **Interactive Clinical Diagnostic Studio:** Built using Gradio with a custom clinical aesthetic theme (`src/dashboard.py`).
2. **High-Throughput REST API Backend:** Built using FastAPI (`src/api.py`) for automated integration into hospital PACS (Picture Archiving and Communication System) networks.

---

## 2. Interactive Clinical Diagnostic Studio (Gradio Dashboard)

The web dashboard is organized into six dedicated clinical workflow tabs:

```
┌────────────────────────────────────────────────────────────────────────┐
│               BrainTumorXAI Clinical Diagnostic Studio                 │
├─────────────┬─────────────┬─────────────┬────────────┬─────────┬───────┤
│ Tab 1: XAI  │ Tab 2:      │ Tab 3: PDF  │ Tab 4:     │ Tab 5:  │ Tab 6:│
│ Diagnostics │ Ensemble    │ Report      │ DICOM      │ Patient │ Tech  │
│ & Overlays  │ Analytics   │ Generation  │ Ingestion  │ History │ Docs  │
└─────────────┴─────────────┴─────────────┴────────────┴─────────┴───────┘
```

### 2.1 Tab 1: Clinical Diagnostics & Saliency Overlays
- **Input Panel:** Allows drag-and-drop upload of 2D MRI slices (JPG, PNG, TIFF) or selection from curated clinical test presets.
- **Diagnostic Badges:**
  - **Predicted Class:** Formatted in large, bold clinical typography (e.g., `Glioma Tumor`).
  - **Confidence Gauge:** Percentage bar indicating model certainty (e.g., `97.85%`).
  - **Uncertainty Indicator:** Normalized Shannon entropy metric (e.g., `12.4% Uncertainty`).
  - **Severity Alert Badge:** Color-coded status badge (`High`, `Moderate`, `Low`, `Borderline`, `Uncertain`).
  - **Anatomical Location:** E.g., `Left Frontal Lobe (Superior)`.
  - **Intracranial Tumor Area:** E.g., `6.42% of cerebral parenchyma`.
- **Quad-View Visual Evidence Display:**
  1. *Original Scan:* Raw patient MRI slice.
  2. *Brain Extraction:* Skull-stripped intracranial parenchyma.
  3. *Grad-CAM Heatmap:* Jet-colormap saliency map highlighting the primary focus.
  4. *Grad-CAM++ Heatmap:* Magma-colormap multi-focal saliency map.
  5. *Morphological Segmentation:* Medical-red alpha-blended overlay delineating the tumor border.

### 2.2 Tab 2: Multi-Model Ensemble Analytics
- **Probability Breakdown:** Interactive horizontal bar charts comparing the individual softmax probability vectors of DenseNet121, InceptionV3, and the Ensemble consensus.
- **Consensus Assessment:** Quantifies model agreement. If models diverge significantly, a visual warning prompts secondary radiologist review.

### 2.3 Tab 3: Clinical PDF Report Export
- **Patient Information Form:** Inputs for Patient ID, Name, Age, Gender, and Referring Physician.
- **Dynamic PDF Compilation:** Triggers ReportLab to assemble an official two-page radiological diagnostic report containing patient demographics, numerical findings, four-panel visual evidence, and a legal CAD disclaimer.
- **Download Action:** Generates a direct PDF download link.

### 2.4 Tab 4: Native Hospital DICOM Ingestion
- **Direct Upload of `.dcm` Files:** Ingests raw clinical scans directly from hospital MRI machines.
- **Automated Metadata Extraction:** Populates patient demographics (Age, Gender, Study Date, Slice Thickness, Pixel Spacing) without manual typing.
- **Automated Window Leveling:** Converts 12-bit/16-bit CT/MRI dynamic ranges into normalized 8-bit RGB tensors for immediate inference.

### 2.5 Tab 5: Patient Case History (SQLite Audit Ledger)
- **Persistent Scan Records:** Queries `data/patient_history.db` to show chronological historical records.
- **Search and Filtering:** Filter records by Patient ID, diagnostic class, or severity tier.
- **Clinical Quality Auditing:** Displays statistical aggregates (total scans, tumor detection rate, average confidence).

### 2.6 Tab 6: Architecture and Citations
- Embedded documentation detailing neural network architectures, growth rates, factorized convolutions, and literature references.

---

## 3. High-Throughput REST API (FastAPI)

For programmatic integration into hospital enterprise networks, the framework exposes four REST endpoints via FastAPI (`src/api.py`):

### 3.1 Endpoint Specifications

#### `GET /api/health`
- **Purpose:** Liveness and readiness probe for Kubernetes / cloud orchestration.
- **Response:**
  ```json
  {
    "status": "healthy",
    "timestamp": 1776535200.0,
    "models_loaded": {
      "DenseNet121": true,
      "InceptionV3": true,
      "GradCAM": true
    }
  }
  ```

#### `GET /api/models`
- **Purpose:** Telemetry detailing active backbones, input resolutions, and ensemble voting weights.

#### `POST /api/predict`
- **Input:** Multipart form upload of an MRI image file (`image/jpeg` or `image/png`).
- **Response:**
  ```json
  {
    "is_valid": true,
    "class_name": "Glioma Tumor",
    "class_idx": 1,
    "confidence": 98.42,
    "uncertainty": 8.15,
    "severity": "High",
    "severity_color": "#EF4444",
    "tumor_percentage": 7.12,
    "location": "Left Frontal Lobe (Superior)",
    "inference_time_ms": 142.5,
    "gradcam_base64": "data:image/jpeg;base64,...",
    "segmentation_base64": "data:image/jpeg;base64,...",
    "probabilities": {
      "No Tumor": 0.0012,
      "Glioma": 0.9842,
      "Meningioma": 0.0114,
      "Pituitary": 0.0032
    }
  }
  ```

#### `POST /api/report`
- **Input:** Image file alongside form fields (`patient_id`, `patient_name`, `patient_age`, `patient_gender`).
- **Response:** Binary streaming response (`application/pdf`) downloading the formal radiology report document.

---

## 4. End-to-End Processing Execution Trace

```
1. [User / Hospital PACS]
   └─ Uploads MRI slice via Web UI or POST /api/predict

2. [Data Validation & Ingestion]
   ├─ Check image format, channel depth, and non-empty pixels
   └─ If DICOM (.dcm): extract metadata tags and apply linear windowing

3. [Automated Skull Stripping (OpenCV)]
   ├─ Gaussian smoothing (sigma = 1.5)
   ├─ Find largest external contour (intracranial perimeter)
   ├─ Crop bounding extremities (min_x, max_x, min_y, max_y)
   └─ Generate binary Mask_brain (255 = brain tissue, 0 = skull/void)

4. [Multi-Scale Resizing & Normalization]
   ├─ Resize to (224, 224, 3) for DenseNet121 -> Scale [0.0, 1.0]
   └─ Resize to (299, 299, 3) for InceptionV3 -> Scale [0.0, 1.0]

5. [Dual-Model Soft Voting Inference]
   ├─ P_DenseNet = DenseNet121.predict(X_224)
   ├─ P_Inception = InceptionV3.predict(X_299)
   ├─ P_ensemble = 0.5 * P_DenseNet + 0.5 * P_Inception
   ├─ Winning Class = argmax(P_ensemble)
   └─ Confidence = max(P_ensemble) * 100%

6. [Explainable AI (Grad-CAM & Grad-CAM++)]
   ├─ GradientTape backpropagation from predicted class logit
   ├─ Top-K channel pooling (top 25% highest activation channels)
   ├─ Element-wise multiplication with Mask_brain (Anatomical Gating)
   └─ Adaptive Otsu thresholding on brain interior activations

7. [Morphological Segmentation Pipeline]
   ├─ Extract candidate ROI (top 30% brain activations, tau >= 0.45)
   ├─ Intra-ROI Otsu adaptive thresholding
   ├─ Morphological opening (noise removal) & closing (bridging holes)
   ├─ Largest connected component filtering
   ├─ Calculate relative brain area percentage (rho_tumor)
   └─ Alpha-blended semi-transparent medical red overlay

8. [Clinical Severity & Localization Engine]
   ├─ Map peak Grad-CAM coordinate to lobe and hemisphere
   └─ Apply severity rules (High, Moderate, Low, Borderline, Uncertain)

9. [Audit Logging & Report Export]
   ├─ Insert complete record into SQLite (patient_history.db)
   └─ Compile two-page ReportLab PDF report for download
```
