# Clinical Severity and Anatomical Localization

## 1. Automated Clinical Severity Rule Matrix

To translate numeric probabilities and spatial masks into actionable clinical alerts, the system implements a deterministic **Clinical Severity Rule Matrix**. The logic combines diagnostic confidence with relative intracranial tumor burden (rho_tumor) to classify the patient's condition into standardized severity tiers.

### 1.1 Decision Rules Formulation

```
IF Confidence < 55.0%:
    Severity = "Borderline" (Color: #F59E0B, Amber/Yellow)

ELSE IF Confidence > 95.0% AND Tumor_Percentage > 5.0%:
    Severity = "High" (Color: #EF4444, Medical Red)

ELSE IF Confidence > 80.0% AND Tumor_Percentage > 3.0%:
    Severity = "Moderate" (Color: #F97316, Orange)

ELSE IF Confidence > 60.0%:
    Severity = "Low" (Color: #10B981, Emerald Green)

ELSE:
    Severity = "Uncertain" (Color: #6B7280, Slate Gray)
```

### 1.2 Clinical Tiers Characterization

| Severity Tier | Color Hex | Clinical Profile | Recommended Hospital Workflow |
|---|---|---|---|
| **High** | `#EF4444` | High-confidence neoplasm (>95%) with substantial intracranial mass effect (>5% of brain volume). | Immediate STAT escalation to neurosurgical oncology team for surgical craniotomy / decompression planning. |
| **Moderate** | `#F97316` | Strong diagnostic consensus (>80%) with intermediate tumor expansion (>3% brain volume). | Priority outpatient oncology referral, contrast volumetric MRI staging, and biopsy scheduling. |
| **Low** | `#10B981` | Moderate-to-high confidence (>60%) with small, focal lesion (<=3% brain volume). | Routine clinical follow-up; monitoring for indolent micro-adenomas or small incidental meningiomas. |
| **Borderline** | `#F59E0B` | Weak ensemble consensus (<55%); diagnostic ambiguity between competing categories. | Multidisciplinary Tumor Board review; secondary evaluation with 3.0 Tesla high-field MRI. |
| **Uncertain** | `#6B7280` | Conflicting model feature attributions with elevated Shannon entropy. | Technical scan re-acquisition to rule out motion blur or scanner phase artifacts. |

---

## 2. Anatomical Coordinate Mapping and Localization

Beyond reporting the tumor category, neurosurgeons require the anatomical coordinates of the lesion within the intracranial vault. The system estimates anatomical position based on the Cartesian coordinates of the global peak activation in the Grad-CAM saliency map.

### 2.1 Peak Saliency Coordinate Identification
Let HM(y, x) denote the brain-gated Grad-CAM activation map resized to image height H and width W. The peak coordinates (y_peak, x_peak) are extracted via:

```
(y_peak, x_peak) = argmax_{0 <= y < H , 0 <= x < W} [ HM(y, x) ]
```

### 2.2 Bilateral Hemisphere Determination
In axial view neuroimaging, the sagittal midline separates the left and right cerebral hemispheres:

```
Hemisphere = "Left"    if x_peak < (W / 2.0)
Hemisphere = "Right"   if x_peak >= (W / 2.0)
```

### 2.3 Craniocaudal (Superior vs. Inferior) Aspect
```
Vertical_Aspect = "Superior"   if y_peak < (H / 2.0)
Vertical_Aspect = "Inferior"   if y_peak >= (H / 2.0)
```

### 2.4 Cerebral Lobe Partitioning
Based on standard axial neuro-anatomical templates:

```
IF y_peak < (0.40 * H):
    Lobe = "Frontal Lobe"
ELSE IF y_peak < (0.70 * H):
    Lobe = "Parietal Lobe"
ELSE:
    Lobe = "Occipital Lobe"
```

### 2.5 Formatted Localization String
The combined anatomical descriptor is assembled:

```
Location_Descriptor = Hemisphere + " " + Lobe + " (" + Vertical_Aspect + ")"
```

**Example Outputs:**
- `"Left Frontal Lobe (Superior)"`
- `"Right Parietal Lobe (Inferior)"`
- `"Left Occipital Lobe (Superior)"`

---

## 3. Clinical DICOM Ingestion and Metadata Processing

To operate seamlessly within Hospital Information Systems (HIS) and Picture Archiving and Communication Systems (PACS), the platform includes a native DICOM (.dcm) reader (`src/dicom_reader.py`).

### 3.1 Metadata Extraction
The module parses standard NEMA/DICOM tags:
- **Patient Identifier:** Tag `(0010, 0020)` -> Patient ID
- **Patient Demographics:** Tag `(0010, 0010)` -> Name, `(0010, 0030)` -> Age, `(0010, 0040)` -> Sex
- **Acquisition Parameters:** Tag `(0008, 0060)` -> Modality (MR), `(0018, 0050)` -> Slice Thickness, `(0028, 0030)` -> Pixel Spacing (mm per pixel)

### 3.2 Dynamic Window Leveling
Raw DICOM pixel data frequently uses 12-bit or 16-bit signed integer representations. To convert these into visible contrast without clipping pathological features, linear window leveling is applied:

```
Window_Center = ds.WindowCenter
Window_Width  = ds.WindowWidth

Lower_Bound = Window_Center - 0.5 - [ (Window_Width - 1) / 2.0 ]
Upper_Bound = Window_Center - 0.5 + [ (Window_Width - 1) / 2.0 ]

IF Pixel <= Lower_Bound:
    Pixel_out = 0.0
ELSE IF Pixel > Upper_Bound:
    Pixel_out = 255.0
ELSE:
    Pixel_out = [ (Pixel - (Window_Center - 0.5)) / (Window_Width - 1) + 0.5 ] * 255.0
```

---

## 4. SQLite Patient History Database Engine

Every diagnostic prediction is automatically archived within a local SQLite database (`data/patient_history.db`) via `src/database.py`.

### 4.1 Schema Definition
```sql
CREATE TABLE IF NOT EXISTS scans (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    patient_id      TEXT    NOT NULL DEFAULT 'N/A',
    patient_name    TEXT    NOT NULL DEFAULT 'Anonymous',
    patient_age     TEXT    NOT NULL DEFAULT 'N/A',
    patient_gender  TEXT    NOT NULL DEFAULT 'N/A',
    scan_filename   TEXT    NOT NULL DEFAULT 'N/A',
    input_type      TEXT    NOT NULL DEFAULT 'IMAGE',
    predicted_class TEXT    NOT NULL,
    confidence_pct  REAL    NOT NULL,
    is_tumor        INTEGER NOT NULL,
    severity        TEXT    NOT NULL DEFAULT 'N/A',
    tumor_area_pct  REAL    NOT NULL DEFAULT 0.0,
    location        TEXT    NOT NULL DEFAULT 'N/A',
    inference_time  REAL    NOT NULL DEFAULT 0.0,
    pdf_path        TEXT    DEFAULT NULL
);
```

This persistent audit trail enables retrospective radiological cohort studies, clinical validation, and longitudinal monitoring of patient tumor progression.

---

## 5. Automated PDF Radiology Report Generation

The reporting engine (`src/report_generator.py`) compiles the diagnostic results into a standardized, two-page clinical PDF report using ReportLab:

1. **Header Block:** Institutional branding (CMRIT / Neuro-Oncology CAD Unit), report generation timestamp, and unique scan accession ID.
2. **Patient Demographics Table:** Patient ID, Name, Age, Sex, Modality (MR), Slice thickness.
3. **Primary Diagnostic Findings Table:** Predicted Class, Classification Confidence %, Relative Tumor Burden (%), Anatomical Location, and Severity Tier with colored visual badge.
4. **Visual Evidence Quad-View:**
   - Panel A: Original Input MRI Scan.
   - Panel B: Skull-Stripped Cerebral Parenchyma.
   - Panel C: Intracranial-Gated Grad-CAM Heatmap (Jet Colormap).
   - Panel D: Weakly-Supervised Morphological Tumor Segmentation (Red Alpha Overlay).
5. **Class Probability Distribution:** Individual prediction vectors for all 4 classes across DenseNet121, InceptionV3, and Ensemble consensus.
6. **Medico-Legal Disclaimer:** Standard regulatory CAD notice stating that the report serves as an algorithmic decision-support aid and must be verified by a licensed neuro-radiologist before clinical intervention.
