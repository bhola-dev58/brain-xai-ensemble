"""
src/dicom_reader.py
====================
DICOM (.dcm) medical image reader for Brain XAI Ensemble.

Converts raw DICOM pixel data into a standard RGB numpy array
compatible with the existing `predict_tumor_logic()` inference pipeline.

Also extracts embedded patient metadata (Patient ID, Name, Age, Sex,
Study Date, Modality, Slice Thickness, Pixel Spacing) for auto-filling
the PDF report form.

Supported Photometric Interpretations:
  - MONOCHROME1 / MONOCHROME2 (standard grayscale brain MRI)
  - RGB
  - YBR_FULL / YBR_FULL_422
  - PALETTE COLOR (with LUT lookup)
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Normalizes any numeric array to uint8 range [0, 255].
    Handles 12-bit, 16-bit, and float DICOM pixel arrays gracefully.
    """
    arr = arr.astype(np.float32)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max == arr_min:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)


def _apply_windowing(pixel_array: np.ndarray, ds) -> np.ndarray:
    """
    Applies DICOM Window Center / Window Width (VOI LUT) if present.
    This is critical for brain MRI where soft-tissue windowing is needed.
    """
    try:
        wc = float(ds.WindowCenter) if hasattr(ds, "WindowCenter") else None
        ww = float(ds.WindowWidth)  if hasattr(ds, "WindowWidth")  else None
        if wc is not None and ww is not None and ww > 0:
            low  = wc - ww / 2.0
            high = wc + ww / 2.0
            arr = pixel_array.astype(np.float32)
            arr = np.clip(arr, low, high)
            return _normalize_to_uint8(arr)
    except Exception:
        pass  # Fall through to plain normalization
    return _normalize_to_uint8(pixel_array)


def dicom_to_rgb(ds) -> np.ndarray:
    """
    Converts a DICOM dataset's pixel array into a 3-channel RGB uint8 numpy array.

    Args:
        ds: A `pydicom.Dataset` object with PixelData loaded.

    Returns:
        RGB numpy array of shape (H, W, 3), dtype uint8.

    Raises:
        ValueError: If pixel data cannot be extracted or decoded.
    """
    import pydicom.pixel_data_handlers.util as pdh_util

    pixel_array = ds.pixel_array  # Raw decoded pixel data

    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2").strip()

    # ── Grayscale Formats ────────────────────────────────────────────────────
    if photometric in ("MONOCHROME1", "MONOCHROME2"):
        gray = _apply_windowing(pixel_array, ds)
        if photometric == "MONOCHROME1":
            gray = 255 - gray  # Invert: bright = air, dark = tissue
        return np.stack([gray, gray, gray], axis=-1)

    # ── Standard RGB ─────────────────────────────────────────────────────────
    if photometric == "RGB":
        if pixel_array.dtype != np.uint8:
            pixel_array = _normalize_to_uint8(pixel_array)
        return pixel_array

    # ── YBR Formats (common in JPEG-compressed DICOM) ────────────────────────
    if photometric in ("YBR_FULL", "YBR_FULL_422"):
        try:
            rgb = pdh_util.convert_color_space(pixel_array, photometric, "RGB")
            return rgb.astype(np.uint8)
        except Exception as e:
            logger.warning(f"YBR conversion failed ({e}), falling back to grayscale treatment.")
            gray = _normalize_to_uint8(pixel_array[..., 0] if pixel_array.ndim == 3 else pixel_array)
            return np.stack([gray, gray, gray], axis=-1)

    # ── Palette Color ────────────────────────────────────────────────────────
    if photometric == "PALETTE COLOR":
        try:
            rgb = pdh_util.apply_color_lut(pixel_array, ds)
            return rgb.astype(np.uint8)
        except Exception as e:
            logger.warning(f"Palette LUT application failed ({e}), normalizing raw array.")

    # ── Fallback: treat as grayscale ─────────────────────────────────────────
    logger.warning(f"Unhandled PhotometricInterpretation '{photometric}', treating as grayscale.")
    if pixel_array.ndim == 3:
        pixel_array = pixel_array[..., 0]
    gray = _normalize_to_uint8(pixel_array)
    return np.stack([gray, gray, gray], axis=-1)


def _safe_str(value, default: str = "N/A") -> str:
    """Safely converts a DICOM tag value to a plain Python string."""
    if value is None:
        return default
    try:
        s = str(value).strip()
        return s if s else default
    except Exception:
        return default


def _parse_age(raw_age: str) -> str:
    """
    Converts DICOM Patient Age strings (e.g., '045Y', '006M') to human-readable form.
    Returns the plain number string for compatibility with the PDF report age field.
    """
    if not raw_age or raw_age == "N/A":
        return "N/A"
    raw = raw_age.strip().upper()
    if raw.endswith("Y") and raw[:-1].isdigit():
        return raw[:-1]  # e.g., "045Y" -> "45"
    if raw.endswith("M") and raw[:-1].isdigit():
        return f"{raw[:-1]} months"
    if raw.endswith("D") and raw[:-1].isdigit():
        return f"{raw[:-1]} days"
    return raw


def extract_dicom_metadata(ds) -> Dict[str, str]:
    """
    Extracts human-readable patient and acquisition metadata from a DICOM dataset.

    Returns a dictionary with the following keys (all strings):
        patient_id, patient_name, patient_age, patient_sex,
        study_date, modality, slice_thickness, pixel_spacing, institution
    """
    raw_age = _safe_str(getattr(ds, "PatientAge", None))
    patient_sex_raw = _safe_str(getattr(ds, "PatientSex", None))
    sex_map = {"M": "Male", "F": "Female", "O": "Other"}
    patient_gender = sex_map.get(patient_sex_raw.upper(), patient_sex_raw)

    # Pixel Spacing: [row_spacing, col_spacing] in mm
    px_spacing = getattr(ds, "PixelSpacing", None)
    if px_spacing and len(px_spacing) >= 2:
        pixel_spacing_str = f"{float(px_spacing[0]):.2f} x {float(px_spacing[1]):.2f} mm"
    else:
        pixel_spacing_str = "N/A"

    # Slice Thickness in mm
    slice_th = getattr(ds, "SliceThickness", None)
    slice_thickness_str = f"{float(slice_th):.2f} mm" if slice_th else "N/A"

    # Study Date: YYYYMMDD -> DD/MM/YYYY
    raw_date = _safe_str(getattr(ds, "StudyDate", None))
    if len(raw_date) == 8 and raw_date.isdigit():
        study_date = f"{raw_date[6:8]}/{raw_date[4:6]}/{raw_date[0:4]}"
    else:
        study_date = raw_date

    return {
        "patient_id":       _safe_str(getattr(ds, "PatientID", None)),
        "patient_name":     _safe_str(getattr(ds, "PatientName", None)),
        "patient_age":      _parse_age(raw_age),
        "patient_gender":   patient_gender,
        "study_date":       study_date,
        "modality":         _safe_str(getattr(ds, "Modality", None)),
        "slice_thickness":  slice_thickness_str,
        "pixel_spacing":    pixel_spacing_str,
        "institution":      _safe_str(getattr(ds, "InstitutionName", None)),
    }


def read_dicom(filepath: str) -> Dict[str, Any]:
    """
    Reads a DICOM (.dcm) file and returns a dictionary containing:
      - "image"    : RGB numpy array (H, W, 3) uint8, ready for inference.
      - "metadata" : Dict of extracted patient/acquisition metadata strings.
      - "filename" : Basename of the original file.
      - "success"  : True if reading succeeded.
      - "error"    : Error message string if success is False, else None.

    Args:
        filepath: Absolute or relative path to the .dcm file.

    Returns:
        Dictionary as described above.
    """
    import pydicom
    result: Dict[str, Any] = {
        "image":    None,
        "metadata": {},
        "filename": os.path.basename(filepath) if filepath else "unknown.dcm",
        "success":  False,
        "error":    None,
    }

    if not filepath or not os.path.exists(filepath):
        result["error"] = f"DICOM file not found: {filepath}"
        logger.error(result["error"])
        return result

    try:
        ds = pydicom.dcmread(filepath, force=True)
    except Exception as e:
        result["error"] = f"Failed to read DICOM file '{os.path.basename(filepath)}': {e}"
        logger.error(result["error"])
        return result

    try:
        rgb_image = dicom_to_rgb(ds)
    except Exception as e:
        result["error"] = f"Failed to decode DICOM pixel data: {e}"
        logger.error(result["error"])
        return result

    result["image"]   = rgb_image
    result["metadata"] = extract_dicom_metadata(ds)
    result["success"] = True
    logger.info(
        f"DICOM loaded: {result['filename']} | "
        f"Shape={rgb_image.shape} | "
        f"Patient={result['metadata'].get('patient_id', 'N/A')}"
    )
    return result


# ── Optional standalone test ────────────────────────────────────────────────────
import os  # noqa: E402  (import at module level, repeated here for clarity)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 -m src.dicom_reader <path_to_file.dcm>")
        sys.exit(1)
    result = read_dicom(sys.argv[1])
    if result["success"]:
        print(f"Image shape: {result['image'].shape}")
        print("Metadata:")
        for k, v in result["metadata"].items():
            print(f"  {k:20s}: {v}")
    else:
        print(f"Error: {result['error']}")
        sys.exit(1)
