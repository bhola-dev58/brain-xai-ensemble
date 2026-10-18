# Testing, Verification, and Quality Assurance

## 1. Quality Assurance Strategy in Medical Software

Medical artificial intelligence systems demand rigorous verification protocols to ensure diagnostic safety, algorithmic stability, and clinical reliability. An unhandled exception during acute clinical triage or a miscalculated tumor mask due to image quantization can compromise patient safety.

To achieve production-grade reliability, the framework implements a structured **3-Tier Testing Architecture** backed by automated test suites in the `tests/` directory:

```
[Tier 1: Unit Testing]        ── Isolated testing of mathematical modules,
                                  image filters, and severity matrices (test_processor.py)
            │
            ▼
[Tier 2: Integration Testing] ── End-to-end model inference, gradient tape execution,
                                  and overlay generation (test_inference.py)
            │
            ▼
[Tier 3: System Testing]      ── Full-stack configuration, DICOM ingestion, SQLite logging,
                                  and PDF report generation (test_full_stack.py)
```

---

## 2. Testing Framework and Test Suites

### 2.1 Tier 1: Unit Testing (`tests/test_processor.py`)
- **Execution Methodology:** Tests are fully decoupled from heavyweight TensorFlow neural network models, allowing instant sub-second execution on continuous integration (CI) runners.
- **Mock Data Fixtures:** Synthetic NumPy arrays representing simulated MRI scans, blank arrays, and extreme noise patterns are generated programmatically.
- **Tested Modules:**
  - `extract_brain_region()`: Validates that skull-stripping returns valid bounding arrays and correctly handles `None` inputs.
  - `create_segmentation()`: Verifies that Otsu morphological segmentation produces a 3-tuple `(result_img, tumor_mask, area_pct)` with spatial dimensions matching the input.
  - `estimate_location()`: Tests anatomical lobe and hemisphere string mapping for arbitrary peak coordinates.
  - `estimate_severity()`: Tests all five clinical severity tiers against borderline numeric thresholds.

### 2.2 Tier 2: Integration Testing (`tests/test_inference.py`)
- **Execution Methodology:** Loads real pre-trained weights (`models/densenet_full_best.keras` and `models/inception_full_best.keras`) and validates end-to-end inference using real clinical test scans from `test_images/`.
- **Verified Invariants:**
  - Output dictionary contains all mandatory keys: `class_name`, `confidence`, `uncertainty`, `is_tumor`, `gradcam_overlay`, `segmentation_img`, `tumor_percentage`, `location`, `severity`.
  - Softmax probabilities across all 4 classes sum to 1.0 (within floating-point tolerance: `sum(P) == 1.0 +/- 1e-5`).
  - Grad-CAM heatmap dimensions strictly match the original image dimensions `(H, W)`.
  - Brain-mask gating successfully forces all outside-skull pixels to zero intensity.

### 2.3 Tier 3: Full-Stack Verification (`tests/test_full_stack.py`)
- **Execution Methodology:** Simulates a complete hospital workflow from configuration loading, through inference and SQLite database logging, to binary PDF generation.
- **Verified Invariants:**
  - Global configuration invariants: Exactly 4 diagnostic classes; ensemble voting weights strictly sum to 1.0 (`sum(W) == 1.0`).
  - ReportLab PDF generator builds a non-empty, valid PDF file (`len(file_bytes) > 10,000` bytes) with embedded images and tables.
  - SQLite database persists records with proper primary keys and timestamps.

---

## 3. Formal Test Cases and Verification Matrix

| Test Case ID | Target Module | Test Scenario & Objective | Input Condition | Expected Result | Verification Status |
|---|---|---|---|---|---|
| **TC-01** | `inference.py` | Full Pipeline Inference on Glioma Scan | High-grade glioma axial MRI slice (`Tr-gl_0010.jpg`) | Predicted class is "Glioma Tumor"; Confidence > 90%; `is_tumor == True`; Valid red overlay. | **PASS** |
| **TC-02** | `inference.py` | Normal Healthy Brain Verification | Normal healthy scan without lesion | Predicted class is "No Tumor"; `is_tumor == False`; Tumor percentage == 0.0%; No false red overlay. | **PASS** |
| **TC-03** | `processor.py` | Null / None Input Handling | Input image is `None` | Graceful handling: returns structured error dict `{"is_valid": False, "error": "..."}`; No uncaught crash. | **PASS** |
| **TC-04** | `processor.py` | Non-Square Aspect Ratio Scan | Rectangular image `(384 x 512 x 3)` | Successfully resized, processed, and restored to original aspect ratio without spatial distortion. | **PASS** |
| **TC-05** | `processor.py` | Completely Black / Blank Scan | Zero-intensity tensor `zeros((256, 256, 3))` | Bounding contour fallback triggered; returns empty tumor mask; `area_pct == 0.0%`. | **PASS** |
| **TC-06** | `inference.py` | Brain-Gating Saliency Integrity | High-contrast skull bone intensity spike | Outside-skull pixels in Grad-CAM map are strictly 0.0; Heatmap confined inside cerebral parenchyma. | **PASS** |
| **TC-07** | `processor.py` | Segmentation Area Guard Limit | Degenerate mask covering >30% brain | Safety fallback triggered: defaults to core 70th-percentile ROI to prevent runaway segmentation. | **PASS** |
| **TC-08** | `processor.py` | Severity Matrix Boundary (Borderline) | Confidence = 54.9%, Area = 2.0% | Correctly assigned severity tier "Borderline" with Amber color code `#F59E0B`. | **PASS** |
| **TC-09** | `processor.py` | Severity Matrix Boundary (High) | Confidence = 96.0%, Area = 6.2% | Correctly assigned severity tier "High" with Medical Red color code `#EF4444`. | **PASS** |
| **TC-10** | `dicom_reader.py` | DICOM Metadata Parsing & Windowing | Standard MR DICOM slice (`scan.dcm`) | Extracted Patient ID, Age, Modality ("MR"); Window leveled to clean 8-bit RGB image. | **PASS** |
| **TC-11** | `report_generator.py`| PDF Radiology Report Generation | Diagnostic result bundle + Patient details | Generates valid, downloadable 2-page PDF with header, quad-view panels, and disclaimer. | **PASS** |
| **TC-12** | `database.py` | SQLite Persistence & Concurrency | 50 consecutive diagnostic scan records | All 50 records inserted with incrementing IDs; verified via `get_scan_count() == 50`. | **PASS** |

---

## 4. Edge-Case Robustness and Defensive Programming Guards

The codebase enforces multiple layers of defensive programming to maintain 100% uptime in clinical environments:

1. **Floating-Point Clipping Guard:**
   ```python
   if img.dtype != np.uint8:
       img = np.uint8(np.clip(img, 0, 255))
   ```
   Prevents OpenCV buffer overflow exceptions when upstream frameworks deliver float32 or float64 arrays.

2. **Zero-Division Stabilizer in Metric Calculations:**
   ```python
   dice = (2 * TP) / (2 * TP + FP + FN + 1e-7)
   ```
   Prevents runtime `ZeroDivisionError` crashes on blank negative slices where TP = FP = FN = 0.

3. **Fallback Saliency Heatmap:**
   If the neural network graph fails to build a gradient tape model (e.g., during headless model inspection), the system smoothly falls back to an anatomically weighted Gaussian prior centered on the intracranial vault, preventing UI freeze.

4. **Brain Area Calculation Safeguard:**
   ```python
   brain_area_px = np.sum(brain_mask > 0)
   if brain_area_px == 0:
       brain_area_px = h * w
   ```
   Guarantees that tumor percentage calculations never encounter an invalid zero-area denominator even if skull-stripping encounters an unrecognizable scan.
