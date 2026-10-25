# BrainTumorXAI — Persistent Project Context

This file serves as the concise technical index of the existing application. Agents should consult this file to understand verified architecture, file locations, and implementation decisions without re-analyzing the repository.

For in-depth architectural diagrams and module workflows, refer to [`docs/02_architecture.md`](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/docs/02_architecture.md).

---

## 1. Application Overview & Capabilities

**BrainTumorXAI** is a clinical-grade medical imaging diagnostic studio and PACS workstation for brain tumor detection, classification, segmentation, and explainability from multi-sequence MRI scans (axial, coronal, sagittal; CE-T1w, T2w, FLAIR).

### Key Features
- **Heterogeneous Tri-Ensemble Classification**: Soft-voting ensemble combining ConvNeXt-Small (45%), InceptionV3 (35%), and DenseNet121 (20%) classifying into 4 categories: *No Tumor*, *Glioma Tumor*, *Meningioma Tumor*, and *Pituitary Tumor*.
- **Explainable AI (XAI)**: Anatomically constrained Brain-Masked Grad-CAM and Grad-CAM++ with second-order gradient saliency and Hadamard cerebral gating to eliminate skull-edge artifacts.
- **Quad-View Saliency Viewport**: Multi-canvas rendering displaying Original Scan, Grad-CAM, Grad-CAM++, and Morphological Segmentation with real-time dynamic colormaps (*Jet*, *Turbo*, *Inferno*, *Viridis*) and adjustable ROI cutoff thresholds.
- **Automated Clinical Reporting**: PDF diagnostic report synthesis using ReportLab with clinical triage severity grading, multi-planar scans, tumor area (mm²), anatomical location (lobe and hemisphere), and provenance metadata.
- **Patient History & PACS Storage**: Case provenance and audit logging stored in a local SQLite database (`data/patient_history.db`).
- **DICOM & Standard Image Ingestion**: Support for DICOM (.dcm), PNG, JPG, and JPEG inputs with sanitization and dimension validation.

---

## 2. Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI (Python 3.10+) | Asynchronous REST API, static asset serving, request security |
| **ASGI Server** | Uvicorn | High-performance HTTP server running on port 8000 |
| **Deep Learning** | TensorFlow 2.x, Keras | Model inference, Grad-CAM backpropagation, tensor operations |
| **Image Processing** | OpenCV (`cv2`), NumPy, SciPy | Brain boundary contour extraction, Otsu thresholding, morphology |
| **PDF Synthesis** | ReportLab | Clinical triage PDF report rendering |
| **Database** | SQLite (`sqlite3`) | Patient audit records, scan metadata, and analysis logging |
| **Frontend Workstation**| Vanilla HTML5, CSS3, ES6+ | Single-page clinical PACS dashboard (zero JS/CSS frameworks) |
| **Styling Architecture** | Handwritten standard CSS | Strict adherence to [`.agents/rules/uiux-designing-rule.md`](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/.agents/rules/uiux-designing-rule.md) |
| **Testing** | pytest, pytest-cov | Automated unit, edge-case, and security integration testing |

---

## 3. Directory & File Index

```
Brain_Tumor_Project/
├── app.py                      # Main entrypoint: launches uvicorn server on HOST:PORT
├── scripts/
│   ├── run_api.py              # Alternative API startup script
│   └── evaluate_segmentation.py# Segmentation metric benchmarks (Dice, IoU)
├── src/
│   ├── __init__.py
│   ├── api.py                  # FastAPI application routes, middleware, request handling
│   ├── config.py               # Central environment paths, ensemble weights, thresholds
│   ├── database.py             # SQLite data access layer for patient cases & scan records
│   ├── dicom_reader.py         # DICOM header parsing, windowing, and slice extraction
│   ├── inference.py            # Model loading, Tri-Ensemble soft voting, Grad-CAM/++ calculation
│   ├── logger.py               # Structured contextual logging with request IDs
│   ├── processor.py            # Skull-stripping, brain crop bounds, segmentation, severity
│   ├── report_generator.py     # ReportLab PDF synthesis routines
│   └── security.py             # File upload sanitization, path traversal defenses, API keys
├── static/
│   ├── index.html              # PACS single-page workstation structure
│   ├── css/
│   │   └── style.css           # Manual CSS design system, dark/light themes, components
│   └── js/
│       └── app.js              # Viewport canvas drawing, colormaps, API interaction, UI state
├── models/                     # Saved Keras/H5 model weights (convnext, inception, densenet)
├── data/
│   └── patient_history.db      # SQLite production database
├── reports/                    # Generated clinical PDF reports
├── tests/                      # 234 comprehensive test cases covering full stack
└── docs/                       # Comprehensive documentation (see docs/02_architecture.md)
```

---

## 4. Backend Endpoints

All endpoints are hosted on `http://localhost:8000`:

| Method | Endpoint | Description | Auth / Security |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Liveness probe and model readiness status | Public |
| `GET` | `/api/models` | Active model telemetry and ensemble weights | Public |
| `POST` | `/api/predict` | MRI upload → ensemble classification, Grad-CAM, segmentation | Validated upload |
| `POST` | `/api/report` | Generates clinical PDF report for a diagnostic case | Validated payload |
| `GET` | `/api/reports/{id}` | Downloads generated PDF report by analysis ID | Path sanitized |
| `GET` | `/api/history` | Paginated patient scan history records | Query validated |
| `GET` | `/api/history/{id}` | Detailed case provenance and metrics for specific analysis | Path sanitized |
| `GET` | `/*` | Static file serving (`static/index.html`, CSS, JS, assets) | StaticFiles mount |

---

## 5. Machine Learning & Inference Pipeline

1. **Preprocessing (`src/processor.py`)**:
   - `extract_brain_region_bounds(img)`: Detects outer skull contour using Otsu thresholding, strips background, and returns cropped brain along with bounding box `(x1, y1, x2, y2)` in original image coordinates.
2. **Tri-Ensemble Soft Voting (`src/inference.py`)**:
   - Resizes input to model-specific dimensions: ConvNeXt (224x224), InceptionV3 (299x299), DenseNet121 (224x224).
   - Computes weighted ensemble probability:
     $$\mathbf{P}_{\text{ensemble}} = 0.45 \cdot \mathbf{P}_{\text{ConvNeXt}} + 0.35 \cdot \mathbf{P}_{\text{Inception}} + 0.20 \cdot \mathbf{P}_{\text{DenseNet}}$$
   - Extracts predicted class and confidence margin.
3. **Brain-Masked Saliency (`src/inference.py`)**:
   - Computes Grad-CAM and Grad-CAM++ feature maps on the top convolutional layer of DenseNet121 (`conv5_block16_concat`).
   - Rescales saliency map to cropped brain dimensions, applies Hadamard tissue gating to zero out non-brain pixels, and remaps the result back into full original image coordinates `(h_orig, w_orig)`.
4. **Segmentation & Area (`src/processor.py`)**:
   - Applies ROI cutoff ($\tau = 0.70$) on the gated saliency map, Otsu thresholding within the ROI, and morphological cleanup (`MORPH_OPEN` + `MORPH_CLOSE`).
   - Calculates tumor area percentage and physical footprint in mm².

---

## 6. Frontend Architecture & Styling

- **Layout**: Navigation Rail (`.nav-rail`) on the left, PACS Sidebar (`.pacs-sidebar`) for metadata and studies, Main Stage with Quad-View Viewport (`.quad-viewport-grid`), and Tab views for Findings, Reports, and Settings.
- **Dual-Layer Canvas Architecture**:
  - Anatomical MRI scan renders as the base layer.
  - Heatmaps and segmentation masks render on transparent overlay canvases (`blendHeatmapToCanvas`, `renderSegmentationToCanvas`).
  - Colormaps (*Jet*, *Turbo*, *Inferno*, *Viridis*) and opacity thresholds are rendered dynamically in client-side JavaScript without server re-computation.
- **Styling Rules**: Strict manual CSS policy in `static/css/style.css` using CSS custom properties (`--bg-primary`, `--clr-accent`, etc.). No CSS frameworks or inline styles.

---

## 7. Testing & Verification

- **Test Command**: `./venv/bin/pytest tests/ -q`
- **Suite Scope**: 234 automated tests across API security, DICOM reading, ensemble uncertainty, explainability, segmentation, model failures, and report generation.
- **Historical Baseline**: 234/234 passing (verified 2026-10-25). When implementing code changes, execute targeted tests (`pytest tests/<test_file>.py -q`) to establish current status.
