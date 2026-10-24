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
    if not np.all(np.isfinite(arr)):
        raise ValueError("DICOM pixel array contains non-finite values (NaN or Inf).")
    arr = arr.astype(np.float32)
    arr_min, arr_max = float(arr.min()), float(arr.max())
    if arr_max <= arr_min:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)


def _apply_modality_rescale(pixel_array: np.ndarray, ds: Any) -> np.ndarray:
    """
    Applies DICOM Modality LUT transformation (Rescale Slope and Rescale Intercept)
    to convert raw stored pixel values to standard modality units (e.g. Hounsfield units or MR signal).

    Formula: PV = SV * RescaleSlope + RescaleIntercept
    Avoids double-rescaling if pixel_array has already been transformed.
    """
    try:
        slope = getattr(ds, "RescaleSlope", None)
        intercept = getattr(ds, "RescaleIntercept", None)
        if slope is not None or intercept is not None:
            s = float(slope) if slope is not None else 1.0
            i = float(intercept) if intercept is not None else 0.0
            if s != 1.0 or i != 0.0:
                arr = pixel_array.astype(np.float32)
                return arr * s + i
    except Exception as e:
        logger.debug(f"Modality rescaling skipped: {e}")
    return pixel_array


def _apply_windowing(pixel_array: np.ndarray, ds) -> np.ndarray:
    """
    Applies DICOM Window Center / Window Width (VOI LUT) in rescaled modality space.
    This is critical for brain MRI where soft-tissue windowing is needed.
    """
    rescaled = _apply_modality_rescale(pixel_array, ds)
    try:
        wc_raw = getattr(ds, "WindowCenter", None)
        ww_raw = getattr(ds, "WindowWidth", None)

        if hasattr(wc_raw, "__iter__") and not isinstance(wc_raw, (str, bytes)):
            wc_raw = list(wc_raw)[0]
        if hasattr(ww_raw, "__iter__") and not isinstance(ww_raw, (str, bytes)):
            ww_raw = list(ww_raw)[0]

        wc = float(wc_raw) if wc_raw is not None else None
        ww = float(ww_raw) if ww_raw is not None else None
        if wc is not None and ww is not None and ww > 0:
            low = wc - ww / 2.0
            high = wc + ww / 2.0
            arr = rescaled.astype(np.float32)
            arr = np.clip(arr, low, high)
            return _normalize_to_uint8(arr)
    except Exception:
        pass  # Fall through to plain normalization
    return _normalize_to_uint8(rescaled)


def dicom_to_rgb(ds) -> np.ndarray:
    """
    Converts a DICOM dataset's pixel array into a 3-channel RGB uint8 numpy array.

    Args:
        ds: A `pydicom.Dataset` object with PixelData loaded.

    Returns:
        RGB numpy array of shape (H, W, 3), dtype uint8.

    Raises:
        ValueError: If pixel data cannot be extracted, decompressed, or decoded.
    """
    import pydicom.pixel_data_handlers.util as pdh_util

    try:
        pixel_array = ds.pixel_array  # Raw decoded pixel data
    except Exception as e:
        raise ValueError(f"Failed to read or decompress DICOM pixel data: {e}")

    if not isinstance(pixel_array, np.ndarray):
        try:
            pixel_array = np.asarray(pixel_array)
        except Exception:
            raise ValueError("DICOM pixel data could not be converted to a valid numpy array.")

    if pixel_array is None or pixel_array.size == 0:
        raise ValueError("DICOM contains empty pixel data.")

    if not np.all(np.isfinite(pixel_array)):
        raise ValueError("DICOM pixel data contains non-finite values (NaN or Inf).")

    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric is not None:
        photometric = str(photometric).strip()
    else:
        photometric = "MONOCHROME2"

    # Handle dimension squeezing / multi-frame slice selection
    if pixel_array.ndim == 3 and pixel_array.shape[0] == 1:
        pixel_array = np.squeeze(pixel_array, axis=0)
    elif pixel_array.ndim == 3 and pixel_array.shape[-1] == 1:
        pixel_array = np.squeeze(pixel_array, axis=-1)
    elif pixel_array.ndim == 3 and pixel_array.shape[0] > 1 and photometric in ("MONOCHROME1", "MONOCHROME2"):
        # Multi-frame volume: extract central representative slice for 2D analysis
        mid_idx = pixel_array.shape[0] // 2
        logger.info(
            f"Multi-frame DICOM detected ({pixel_array.shape[0]} frames). "
            f"Extracting central slice index {mid_idx} for 2D cross-sectional inference."
        )
        pixel_array = pixel_array[mid_idx]
    elif pixel_array.ndim > 3:
        raise ValueError(f"Unsupported DICOM pixel array dimensions: {pixel_array.shape}")

    if pixel_array.shape[0] == 0 or pixel_array.shape[1] == 0:
        raise ValueError(
            f"DICOM image dimensions ({pixel_array.shape[1]}x{pixel_array.shape[0]}) are empty."
        )

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


def compute_slice_spatial_position(ds: Any) -> Optional[float]:
    """
    Calculates the spatial distance of a slice along the scan acquisition normal vector.
    Uses DICOM Patient-Based Coordinate System (PS 3.3 C.7.6.2.1):
      Normal N = (Row Direction Cosines) x (Column Direction Cosines)
      Distance = ImagePositionPatient . N

    Args:
        ds: pydicom.Dataset object.

    Returns:
        Float position along the slice normal in millimeters, or None if metadata missing.
    """
    try:
        iop = getattr(ds, "ImageOrientationPatient", None)
        ipp = getattr(ds, "ImagePositionPatient", None)
        if iop and len(iop) >= 6 and ipp and len(ipp) >= 3:
            r = np.array([float(iop[0]), float(iop[1]), float(iop[2])], dtype=np.float64)
            c = np.array([float(iop[3]), float(iop[4]), float(iop[5])], dtype=np.float64)
            normal = np.cross(r, c)
            norm_mag = np.linalg.norm(normal)
            if norm_mag > 1e-6:
                normal = normal / norm_mag
                pos = np.array([float(ipp[0]), float(ipp[1]), float(ipp[2])], dtype=np.float64)
                return float(np.dot(pos, normal))
    except Exception as e:
        logger.debug(f"Could not compute spatial normal position: {e}")

    # Fallback to SliceLocation if orientation vectors not available
    try:
        loc = getattr(ds, "SliceLocation", None)
        if loc is not None:
            return float(loc)
    except Exception:
        pass

    return None


def extract_dicom_metadata(ds, deidentify: bool = False) -> Dict[str, Any]:
    """
    Extracts comprehensive DICOM metadata encompassing Patient, Study, Series,
    Instance, and Spatial Orientation tags.

    Args:
        ds: pydicom.Dataset object.
        deidentify: If True, masks patient identifiable information (PHI).

    Returns:
        Structured dictionary containing study, series, instance, and spatial fields.
    """
    raw_pid = _safe_str(getattr(ds, "PatientID", None))
    raw_pname = _safe_str(getattr(ds, "PatientName", None))
    raw_age = _safe_str(getattr(ds, "PatientAge", None))
    patient_sex_raw = _safe_str(getattr(ds, "PatientSex", None))
    sex_map = {"M": "Male", "F": "Female", "O": "Other"}
    patient_gender = sex_map.get(patient_sex_raw.upper(), patient_sex_raw)

    if deidentify:
        # Mask PHI to prevent accidental disclosure in logs or public responses
        patient_id = "ANON-" + (raw_pid[:4] if len(raw_pid) >= 4 else "XXXX")
        patient_name = "Anonymous Patient"
    else:
        patient_id = raw_pid
        patient_name = raw_pname

    # Pixel Spacing: [row_spacing, col_spacing] in mm
    px_spacing = getattr(ds, "PixelSpacing", None)
    if px_spacing and len(px_spacing) >= 2:
        pixel_spacing_str = f"{float(px_spacing[0]):.2f} x {float(px_spacing[1]):.2f} mm"
        pixel_spacing_tuple = (float(px_spacing[0]), float(px_spacing[1]))
    else:
        pixel_spacing_str = "N/A"
        pixel_spacing_tuple = None

    # Slice Thickness in mm
    slice_th = getattr(ds, "SliceThickness", None)
    slice_thickness_str = f"{float(slice_th):.2f} mm" if slice_th is not None else "N/A"

    # Study Date: YYYYMMDD -> DD/MM/YYYY
    raw_date = _safe_str(getattr(ds, "StudyDate", None))
    if len(raw_date) == 8 and raw_date.isdigit():
        study_date = f"{raw_date[6:8]}/{raw_date[4:6]}/{raw_date[0:4]}"
    else:
        study_date = raw_date

    # Study / Series / Instance Hierarchy
    study_uid = _safe_str(getattr(ds, "StudyInstanceUID", None), default="N/A")
    series_uid = _safe_str(getattr(ds, "SeriesInstanceUID", None), default="N/A")
    sop_uid = _safe_str(getattr(ds, "SOPInstanceUID", None), default="N/A")
    study_desc = _safe_str(getattr(ds, "StudyDescription", None), default="N/A")
    series_desc = _safe_str(getattr(ds, "SeriesDescription", None), default="N/A")
    series_num = getattr(ds, "SeriesNumber", None)
    series_number_str = str(series_num) if series_num is not None else "N/A"
    instance_num = getattr(ds, "InstanceNumber", None)
    instance_number_int = int(instance_num) if instance_num is not None and str(instance_num).isdigit() else None

    # Spatial Geometry
    slice_loc = getattr(ds, "SliceLocation", None)
    slice_loc_float = float(slice_loc) if slice_loc is not None else None

    ipp = getattr(ds, "ImagePositionPatient", None)
    ipp_tuple = tuple(float(x) for x in ipp) if ipp and len(ipp) >= 3 else None

    iop = getattr(ds, "ImageOrientationPatient", None)
    iop_tuple = tuple(float(x) for x in iop) if iop and len(iop) >= 6 else None

    num_frames = getattr(ds, "NumberOfFrames", None)
    num_frames_int = int(num_frames) if num_frames is not None and str(num_frames).isdigit() else 1

    spatial_pos = compute_slice_spatial_position(ds)

    # Modality LUT parameters
    rescale_slope = getattr(ds, "RescaleSlope", None)
    rescale_slope_val = float(rescale_slope) if rescale_slope is not None else None
    rescale_intercept = getattr(ds, "RescaleIntercept", None)
    rescale_intercept_val = float(rescale_intercept) if rescale_intercept is not None else None

    # Single-Slice Diagnostic Indicator
    inst_label = f"#{instance_number_int}" if instance_number_int is not None else "N/A"
    slice_info = (
        f"Single 2D Slice (Instance {inst_label}) - Series: {series_desc if series_desc != 'N/A' else series_number_str}"
    )

    return {
        # Core patient demographics
        "patient_id": patient_id,
        "patient_name": patient_name,
        "patient_age": _parse_age(raw_age),
        "patient_gender": patient_gender,
        "study_date": study_date,
        "modality": _safe_str(getattr(ds, "Modality", None), default="MR"),
        "slice_thickness": slice_thickness_str,
        "pixel_spacing": pixel_spacing_str,
        "pixel_spacing_mm": pixel_spacing_tuple,
        "institution": _safe_str(getattr(ds, "InstitutionName", None)),

        # Study & Series Architecture
        "study_instance_uid": study_uid,
        "study_description": study_desc,
        "series_instance_uid": series_uid,
        "series_number": series_number_str,
        "series_description": series_desc,
        "sop_instance_uid": sop_uid,
        "instance_number": instance_number_int,

        # Spatial orientation & geometry
        "slice_location": slice_loc_float,
        "spatial_position_mm": spatial_pos,
        "image_position_patient": ipp_tuple,
        "image_orientation_patient": iop_tuple,
        "number_of_frames": num_frames_int,
        "rescale_slope": rescale_slope_val,
        "rescale_intercept": rescale_intercept_val,

        # Diagnostic scope & safety disclaimer
        "is_single_slice": True,
        "slice_context": slice_info,
        "is_volumetric_study": False,
    }


def group_and_sort_dicom_series(
    items: list,
    deidentify: bool = False
) -> Dict[str, Any]:
    """
    Groups DICOM objects into Study -> Series hierarchy and sorts slices
    using physical spatial geometry along the acquisition normal vector.

    Args:
        items: List of filepaths (str) or loaded pydicom.Dataset objects.
        deidentify: If True, strips PHI from metadata.

    Returns:
        Structured dictionary containing grouped studies, series, slice counts,
        spatial ordering methods, and safety disclaimers.
    """
    import pydicom

    studies: Dict[str, Dict[str, Any]] = {}
    total_slices = 0

    for item in items:
        try:
            if isinstance(item, str):
                if not os.path.exists(item):
                    continue
                ds = pydicom.dcmread(item, force=True)
                source_ref = os.path.basename(item)
            else:
                ds = item
                source_ref = getattr(ds, "filename", "in-memory.dcm")

            meta = extract_dicom_metadata(ds, deidentify=deidentify)
            study_uid = meta["study_instance_uid"]
            series_uid = meta["series_instance_uid"]

            if study_uid not in studies:
                studies[study_uid] = {
                    "study_instance_uid": study_uid,
                    "study_description": meta["study_description"],
                    "study_date": meta["study_date"],
                    "patient_id": meta["patient_id"],
                    "series": {},
                }

            study_dict = studies[study_uid]
            if series_uid not in study_dict["series"]:
                study_dict["series"][series_uid] = {
                    "series_instance_uid": series_uid,
                    "series_number": meta["series_number"],
                    "series_description": meta["series_description"],
                    "modality": meta["modality"],
                    "slices": [],
                    "is_complete_volume": False,
                    "ordering_method": "none",
                }

            slice_entry = {
                "sop_instance_uid": meta["sop_instance_uid"],
                "instance_number": meta["instance_number"],
                "spatial_position_mm": meta["spatial_position_mm"],
                "slice_location": meta["slice_location"],
                "source": source_ref,
                "metadata": meta,
            }
            study_dict["series"][series_uid]["slices"].append(slice_entry)
            total_slices += 1

        except Exception as e:
            logger.warning(f"Skipping unreadable DICOM item in series grouping: {e}")

    # Sort slices within each series along the normal vector or fallback
    total_series_count = 0
    for study in studies.values():
        total_series_count += len(study["series"])
        for series in study["series"].values():
            slices = series["slices"]
            if not slices:
                continue

            # Check if all slices have valid spatial positions along the normal vector
            has_spatial = all(s["spatial_position_mm"] is not None for s in slices)
            if has_spatial and len(slices) > 1:
                slices.sort(key=lambda s: s["spatial_position_mm"])
                series["ordering_method"] = "spatial_normal"
            elif all(s["instance_number"] is not None for s in slices):
                slices.sort(key=lambda s: s["instance_number"])
                series["ordering_method"] = "instance_number"
            elif all(s["slice_location"] is not None for s in slices):
                slices.sort(key=lambda s: s["slice_location"])
                series["ordering_method"] = "slice_location"
            else:
                slices.sort(key=lambda s: s["source"])
                series["ordering_method"] = "filename_fallback"

            series["slice_count"] = len(slices)

    return {
        "studies": studies,
        "total_studies": len(studies),
        "total_series": total_series_count,
        "total_slices": total_slices,
        "clinical_scope": "2D Cross-Sectional MRI Slices - Not a 3D Reconstructed Volume",
        "volumetric_reconstruction_supported": False,
    }


def read_dicom(filepath: str, deidentify: bool = False) -> Dict[str, Any]:
    """
    Reads a DICOM (.dcm) file and returns a dictionary containing:
      - "image"       : RGB numpy array (H, W, 3) uint8, ready for inference.
      - "metadata"    : Dict of extracted patient, study, series, and spatial metadata.
      - "filename"    : Basename of the original file.
      - "success"     : True if reading succeeded.
      - "error"       : Error message string if success is False, else None.
      - "slice_context": Explanatory label indicating single-slice diagnostic scope.

    Args:
        filepath: Absolute or relative path to the .dcm file.
        deidentify: If True, masks patient identifiable information in returned metadata.

    Returns:
        Dictionary as described above.
    """
    import pydicom
    result: Dict[str, Any] = {
        "image":         None,
        "metadata":      {},
        "filename":      os.path.basename(filepath) if filepath else "unknown.dcm",
        "success":       False,
        "error":         None,
        "is_single_slice": True,
        "slice_context": "Single 2D Slice - Not a Volumetric Study",
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

    meta = extract_dicom_metadata(ds, deidentify=deidentify)
    result["image"] = rgb_image
    result["metadata"] = meta
    result["slice_context"] = meta.get("slice_context", result["slice_context"])
    result["success"] = True

    # Use de-identified or masked patient ID for operational logging
    log_pid = meta.get("patient_id", "N/A")
    if not deidentify and len(log_pid) > 4:
        log_pid = log_pid[:4] + "***"

    logger.info(
        f"DICOM loaded: {result['filename']} | "
        f"Shape={rgb_image.shape} | "
        f"Series={meta.get('series_number', 'N/A')} | "
        f"Instance={meta.get('instance_number', 'N/A')} | "
        f"Patient={log_pid}"
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
