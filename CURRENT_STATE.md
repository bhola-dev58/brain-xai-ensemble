# BrainTumorXAI — Current State Checkpoint

**Last Updated**: 2026-10-25  
**Current Branch**: `main`  
**Overall Status**: Stable / Operational / Production-Ready Local Workstation  

---

## 1. Verified Implemented Features

All listed features have been cross-verified against the active codebase and automated test suite:

- [x] **Heterogeneous Tri-Ensemble Classification Engine**:
  - Models: ConvNeXt-Small (45%), InceptionV3 (35%), DenseNet121 (20%).
  - Soft-voting ensemble with confidence estimation and fallback mechanisms (`src/inference.py`).
  - 4 classes supported: No Tumor, Glioma, Meningioma, Pituitary.
- [x] **Anatomically Constrained Explainability (XAI)**:
  - Brain-Masked Grad-CAM and Grad-CAM++ with Hadamard tissue gating (`src/inference.py`).
  - Full-resolution coordinate re-mapping preserving scan geometry (`src/processor.py`).
- [x] **Multi-Layer Quad-View Saliency Viewport**:
  - Independent transparent overlay canvases for heatmap and segmentation rendering (`static/js/app.js`).
  - Dynamic client-side colormap switching (*Jet*, *Turbo*, *Inferno*, *Viridis*).
  - Real-time interactive ROI cutoff threshold slider.
  - Ghosting-free anatomical underlay alignment.
- [x] **Morphological Lesion Boundary & Triage**:
  - Weakly-supervised lesion segmentation with Otsu thresholding and morphological opening/closing (`src/processor.py`).
  - Quantitative tumor area measurement (mm² and percentage of brain area).
  - Automated anatomical location (lobe + hemisphere) and severity classification.
- [x] **Clinical PDF Diagnostic Report Generation**:
  - Automated multi-planar report generation using ReportLab (`src/report_generator.py`).
  - Patient demographics, scan metadata, confidence breakdowns, and XAI figures.
- [x] **Data Persistence & Audit Logging**:
  - SQLite case history logging with SHA-256 provenance hashes (`src/database.py`).
  - Paginated patient history view in Settings (`static/js/app.js`).
- [x] **PACS Workstation Interface**:
  - Streamlined modern header with live engine status telemetry and theme toggle (`static/index.html`).
  - Strict compliance with [`.agents/rules/uiux-designing-rule.md`](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/.agents/rules/uiux-designing-rule.md).
- [x] **API Security & Validation**:
  - File upload sanitization, magic-byte MIME type validation, dimension limits (`src/security.py`).
  - Rate limiting and optional API key enforcement.

---

## 2. Recently Completed Work

- **CI Regression Fix (2026-11-06)**:
  - Root cause from Actions job `127958931599`: `tests/test_prediction_traceability.py::TestModelAndPipelineProvenance::test_model_provenance_sanitizes_internal_file_paths` failed because missing artifact placeholder `"N/A"` contains `/`.
  - `src/inference.py`: updated missing provenance artifact fallback from `"N/A"` to `"UNKNOWN"` for ConvNeXt, Inception, and DenseNet entries to preserve path-sanitization invariant.
  - Validation (2026-11-06): `python -m pytest tests/test_prediction_traceability.py -q` → `15 passed`; `python -m flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics` → `0`.

- **HiDevs Audit Improvements (2026-11-06)**:
  - `requirements-dev.txt`: added `httpx2>=0.28.0` and `pip-audit>=2.7.0` as dev dependencies.
  - `pytest.ini`: added `--cov=src --cov-report=term-missing --cov-report=xml:coverage.xml` to `addopts`.
  - `tests/test_logger.py`: **NEW** — 13 unit tests covering `src/logger.py` (100% module coverage achieved).
  - `tests/test_database_edge_cases.py`: **NEW** — 27 unit tests covering `src/database.py` edge cases (init idempotency, FAILED status overrides, provenance serialisation, pagination, ordering).
  - `.env.example`: **NEW** — comprehensive environment variable template documenting all 14 configurable variables with safe defaults.
  - `pyproject.toml`: **NEW** — PEP 517 project metadata and optional dev dependency group.
  - `.github/workflows/ci.yml`: enhanced with `--cov` coverage reporting, artifact upload, and second `security-audit` job using `pip-audit`.
- **Commit `1963cd6`**: `style: declutter interface header and retain essential telemetry controls`
  - Removed static non-functional marketing badges (`SOTA Tri-Ensemble`, `Grad-CAM`, `PDF Clinical Reports`).
  - Removed placeholder profile badge (`Dr. Alex Johnson`).
  - Removed redundant header export button (canonical PDF report action lives in the Reports tab).
  - Streamlined theme toggle into a sleek icon button and compacted telemetry indicator to `"Engine Online"`.
- **Commit `f4b6e5f`**: `fix: resolve dual-layer viewport misalignment and image ghosting`
  - Updated `extract_brain_region_bounds()` in `src/processor.py` to return bounding coordinates `(x1, y1, x2, y2)`.
  - Remapped Grad-CAM, Grad-CAM++, and segmentation masks back to full original image space `(h_orig, w_orig)`.
  - Serialized `tumor_mask_b64` in `src/api.py` and rendered transparent overlays in `static/js/app.js`.
- **Commit `59f456a`**: `feat: add dynamic colormap and roi cutoff rendering to viewport`
  - Interactive colormap selection and threshold updates rendered entirely on canvas without re-querying the API.

---

## 3. Historical Test Baseline

> [!NOTE]
> **Historical Record**: The test outcome below was verified on **2026-10-25 at 21:36 local time**. It documents the baseline health of the codebase at that checkpoint and must **not** be presented as real-time proof of subsequent or uncommitted code changes until targeted tests are executed.

- **Verification Date**: 2026-10-25 21:36:14 (local)
- **Command Executed**: `./venv/bin/pytest tests/ -q`
- **Recorded Output**:
  ```text
  ======================= 234 passed, 1 warning in 40.66s ========================
  ```
- **Recorded Pass Rate**: 100% (234/234 tests passing at checkpoint).
- **Verified Test Modules**:
  - `test_api_security.py` (17 tests)
  - `test_deployment_config.py` (7 tests)
  - `test_dicom_and_api_edge_cases.py` (12 tests)
  - `test_dicom_study_series.py` (17 tests)
  - `test_ensemble_and_uncertainty.py` (21 tests)
  - `test_explainability_and_segmentation.py` (24 tests)
  - `test_full_stack.py` (1 test)
  - `test_inference.py` (15 tests)
  - `test_inference_edge_cases.py` (11 tests)
  - `test_logging_observability.py` (7 tests)
  - `test_model_failures.py` (10 tests)
  - `test_mri_input_validation.py` (24 tests)
  - `test_prediction_traceability.py` (15 tests)
  - `test_processor.py` (20 tests)
  - `test_processor_edge_cases.py` (18 tests)
  - `test_reporting_and_db_resilience.py` (6 tests)
  - `test_reporting_safety.py` (9 tests)

---

## 4. Known Bugs, Limitations & Constraints

1. **Volumetric DICOM MPR**: The current DICOM loader handles single-slice DICOMs and parses multi-frame metadata, but 3D multi-planar reconstruction (MPR) volume rendering in the browser is not yet implemented.
2. **EffNet Weight Inactive**: `EFFNET_VOTE_WEIGHT` is set to `0.00` in `src/config.py` as ConvNeXt, Inception, and DenseNet yield the optimal ensemble accuracy.
3. **Running Process Guardrail**: Do not execute commands that attempt to start the server (`python app.py` / `python scripts/run_api.py`) as the server is already managed externally.
4. **Git Push Guardrail**: Never run `git push` directly; print the command for the user.

---

## 5. Next Actionable Development Task

- **Pending Work Reference**: All pending work and future tasks are tracked exclusively in [`TASKS.md`](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/TASKS.md).
- **Mandatory Agent Check**: Before planning or proposing any development task, the agent MUST consult [`TASKS.md`](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/TASKS.md) Section 2 ("Pending Tasks") and select an existing tracked item. The agent must never invent new development tasks outside `TASKS.md` without explicit user confirmation.
- **Current Active Queue Item**: Task 2.2 — Batch MRI Upload & Triage Queue (see `TASKS.md`).
