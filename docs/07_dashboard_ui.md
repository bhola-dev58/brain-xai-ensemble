# 07 — Clinical Diagnostic Studio (UI/UX)

The interactive workstation is built as a high-performance **Clinical Diagnostic Studio** using modern semantic HTML5, zero-framework hand-crafted vanilla CSS, and vanilla ES6+ JavaScript, served by FastAPI. It is launched via `app.py` or `scripts/run_api.py`.

---

## Launching the Workstation

```bash
# Standard Entrypoint
python app.py

# Or via scripts
python scripts/run_api.py
```

- **Local Workstation URL**: `http://localhost:8000` (or `http://127.0.0.1:8000`)
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **Liveness & Readiness Healthcheck**: `http://localhost:8000/api/health`

---

## Architecture & Design System

The workstation uses a clinical medical imaging design system adhering to WCAG 2.2 AA standards:
- **Palette**: Obsidian Charcoal surfaces (`#080B0F`), Elevated Panels (`#11161C`), Subtle Borders (`#252C35`).
- **Brand Accent**: Electric Clinical Amber (`#FF7A00`) in dark mode and Radiant Warm Amber (`#EA580C`) in light mode.
- **Theme Support**: Seamless Dark/Light mode toggle with persistent state in `localStorage` and OS preference auto-detection.
- **Strict Manual CSS**: 100% handwritten modular CSS in `static/css/style.css` without utility framework bloat (no Tailwind, no Bootstrap). Zero inline `style` attributes.

---

## Layout Overview

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  HEADER: Brand Logo + Clinical Diagnostic Studio + Badges + Theme Toggle + Telemetry    │
└────────────────────────────────────────────────────────────────────────────────────────┘
┌──────────────┬──────────────────┬──────────────────────────────────────────────────────┐
│  NAV RAIL    │  PACS CONTROLS   │  WORKSTATION STAGE (Tabs & Diagnostic Grid)          │
│              │  (Collapsible)   │                                                      │
│  [Dashboard] │  - Window Presets│  [Tab: Clinical Diagnostics & XAI]                   │
│  [Scans]     │  - WW / WL       │  ┌──────────────┬──────────────────┬──────────────┐  │
│  [Reports]   │  - Sharpening    │  │ 1. Upload    │ 2. Quad Viewport │ 3. Findings  │  │
│  [Analytics] │  - PHI Masking   │  │    & Samples │    (Raw, CAM,    │    & Metrics │  │
│  [Patients]  │  - TTA Ensemble  │  │              │     CAM++, Seg)  │              │  │
│  [Settings]  │  - Crosshair     │  └──────────────┴──────────────────┴──────────────┘  │
│  [Help]      │                  │  [Tab: Analytics] [Tab: PDF Export] [Tab: History]   │
└──────────────┴──────────────────┴──────────────────────────────────────────────────────┘
```

---

## Key Workstation Modules

### 1. Diagnostic Studio Stage
- **Drag & Drop Zone**: Instant drag-and-drop file ingestion supporting JPG, PNG, and DICOM slices up to 25MB.
- **Sample Benchmark Buttons**: Clickable representative cases (High Grade Glioma, Extra-Axial Meningioma, Pituitary Adenoma, Healthy Control).
- **PACS Quad-View Viewport**:
  - Panel A: Raw CE-MRI axial scan with aspect-ratio containment.
  - Panel B: Brain-gated Grad-CAM saliency overlay.
  - Panel C: Higher-order multi-focal Grad-CAM++ activation map.
  - Panel D: Weakly-supervised morphological tumor delineation contour.
  - Interactive blend opacity slider (0–100%) and synchronized crosshair coordinate tracker HUD (`X: --, Y: --`).
- **Findings & Clinical Metrics**:
  - SVG Circular Confidence Gauge with weighted consensus breakdown.
  - Clinical Verdict Banner with severity grading and latency tracking.
  - 4-Card Biomarkers: Estimated Location, Tumor Tissue Area (%), Severity Grading, Uncertainty Entropy (%).
  - Multi-class posterior probability distribution bars.

### 2. Tri-Ensemble Analytics Tab
- Model voting consensus comparison:
  - ConvNeXt-Small (45% weight): 7×7 depthwise macro-spatial convolutional backbones.
  - InceptionV3 (35% weight): Multi-scale asymmetric factorized kernels.
  - DenseNet121 (20% weight): Dense connectivity preventing gradient vanishing.

### 3. Clinical Report PDF Export Tab
- Form fields for Patient MRN, Name, Age, and Gender.
- Instant server-side PDF generation via ReportLab including all 4 XAI saliency overlays.

### 4. Patient Audit History Tab
- Case audit log table with responsive horizontal scrolling.
