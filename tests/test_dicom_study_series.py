"""
tests/test_dicom_study_series.py
================================
Targeted regression tests for DICOM study/series parsing, modality rescaling,
spatial normal slice ordering, multi-frame handling, and de-identification.
"""

import numpy as np
import pytest
import pydicom

from src.dicom_reader import (
    _apply_modality_rescale,
    _apply_windowing,
    _normalize_to_uint8,
    compute_slice_spatial_position,
    dicom_to_rgb,
    extract_dicom_metadata,
    group_and_sort_dicom_series,
    read_dicom,
)


class MockDicomDataset:
    """Lightweight test fixture mimicking a pydicom Dataset."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# =====================================================================
# 1. Modality Rescale (RescaleSlope / RescaleIntercept) & Windowing
# =====================================================================
class TestDicomModalityRescaleAndWindowing:
    def test_modality_rescale_slope_and_intercept_calculation(self):
        """
        Tests standard DICOM Modality LUT calculation:
          Modality Value = Stored Value * RescaleSlope + RescaleIntercept
        """
        ds = MockDicomDataset(RescaleSlope="2.0", RescaleIntercept="-1000.0")
        raw_pixels = np.array([500, 1000, 1500], dtype=np.int16)

        rescaled = _apply_modality_rescale(raw_pixels, ds)
        # Expected: 500*2 - 1000 = 0; 1000*2 - 1000 = 1000; 1500*2 - 1000 = 2000
        np.testing.assert_allclose(rescaled, [0.0, 1000.0, 2000.0])

    def test_modality_rescale_noop_when_absent_or_identity(self):
        """When slope is 1.0 and intercept is 0.0 (or absent), raw pixels are unchanged."""
        ds_none = MockDicomDataset()
        raw = np.array([10, 20, 30], dtype=np.uint16)
        assert np.array_equal(_apply_modality_rescale(raw, ds_none), raw)

        ds_identity = MockDicomDataset(RescaleSlope=1.0, RescaleIntercept=0.0)
        assert np.array_equal(_apply_modality_rescale(raw, ds_identity), raw)

    def test_windowing_applies_modality_rescale_before_clipping(self):
        """
        WindowCenter (40) and WindowWidth (80) define window [0, 80] in rescaled units.
        Stored value 520 with slope=2, intercept=-1000 gives 40 (midpoint of [0, 80]).
        After windowing and normalization to uint8, 40 should normalize to ~128.
        """
        ds = MockDicomDataset(
            RescaleSlope=2.0,
            RescaleIntercept=-1000.0,
            WindowCenter=40.0,
            WindowWidth=80.0,
        )
        # Stored 500 -> 0 (min of window), 520 -> 40 (mid), 540 -> 80 (max of window)
        pixels = np.array([[500, 520, 540]], dtype=np.int16)
        windowed = _apply_windowing(pixels, ds)

        assert windowed.shape == (1, 3)
        assert windowed.dtype == np.uint8
        assert windowed[0, 0] == 0
        assert 126 <= windowed[0, 1] <= 129  # exact or near midpoint (127.5)
        assert windowed[0, 2] == 255

    def test_windowing_supports_sequence_or_multivalue(self):
        """DICOM WindowCenter and WindowWidth can be sequences / MultiValue objects."""
        ds = MockDicomDataset(
            WindowCenter=[50.0, 100.0],
            WindowWidth=[100.0, 200.0],
        )
        pixels = np.array([0, 50, 100], dtype=np.float32)
        windowed = _apply_windowing(pixels, ds)
        assert windowed[0] == 0
        assert 126 <= windowed[1] <= 129
        assert windowed[2] == 255


# =====================================================================
# 2. Spatial Normal Slice Ordering along Acquisition Vector
# =====================================================================
class TestSliceSpatialGeometryAndOrdering:
    def test_compute_slice_spatial_position_axial(self):
        """
        In standard axial orientation:
          Row Direction Cosines: [1, 0, 0] (Patient Left)
          Column Direction Cosines: [0, 1, 0] (Patient Posterior)
          Normal = Row x Col = [0, 0, 1] (Patient Superior)
          Distance = IPP . Normal = z coordinate
        """
        ds = MockDicomDataset(
            ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
            ImagePositionPatient=[12.5, -45.0, 37.8],
        )
        pos = compute_slice_spatial_position(ds)
        assert pos is not None
        assert pytest.approx(pos, 1e-4) == 37.8

    def test_compute_slice_spatial_position_sagittal(self):
        """
        In sagittal orientation:
          Row Direction Cosines: [0, 1, 0]
          Column Direction Cosines: [0, 0, 1]
          Normal = [1, 0, 0]
          Distance = IPP . Normal = x coordinate
        """
        ds = MockDicomDataset(
            ImageOrientationPatient=[0, 1, 0, 0, 0, 1],
            ImagePositionPatient=[-24.6, 10.0, 50.0],
        )
        pos = compute_slice_spatial_position(ds)
        assert pos is not None
        assert pytest.approx(pos, 1e-4) == -24.6

    def test_compute_slice_spatial_position_fallback_to_slice_location(self):
        """When IOP or IPP is missing, falls back to SliceLocation."""
        ds = MockDicomDataset(SliceLocation="18.5")
        pos = compute_slice_spatial_position(ds)
        assert pos == 18.5

    def test_compute_slice_spatial_position_returns_none_when_no_spatial_data(self):
        """Returns None safely when no spatial coordinate tags exist."""
        ds = MockDicomDataset()
        assert compute_slice_spatial_position(ds) is None


# =====================================================================
# 3. Study / Series Grouping, Isolation, and Sorting
# =====================================================================
class TestStudySeriesGroupingAndSorting:
    def test_group_and_sort_same_series_by_spatial_normal(self):
        """
        Slices belonging to the same series provided in scrambled order
        must be sorted monotonically along the normal vector.
        """
        study_uid = "1.2.840.10008.1"
        series_uid = "1.2.840.10008.1.1"

        # Three axial slices: z = 30.0, z = -10.0, z = 10.0
        s1 = MockDicomDataset(
            StudyInstanceUID=study_uid,
            SeriesInstanceUID=series_uid,
            SOPInstanceUID="sop.1",
            ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
            ImagePositionPatient=[0, 0, 30.0],
            InstanceNumber=3,
            filename="slice_z30.dcm",
        )
        s2 = MockDicomDataset(
            StudyInstanceUID=study_uid,
            SeriesInstanceUID=series_uid,
            SOPInstanceUID="sop.2",
            ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
            ImagePositionPatient=[0, 0, -10.0],
            InstanceNumber=1,
            filename="slice_z_neg10.dcm",
        )
        s3 = MockDicomDataset(
            StudyInstanceUID=study_uid,
            SeriesInstanceUID=series_uid,
            SOPInstanceUID="sop.3",
            ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
            ImagePositionPatient=[0, 0, 10.0],
            InstanceNumber=2,
            filename="slice_z10.dcm",
        )

        # Scrambled input order: s1 (30), s2 (-10), s3 (10)
        result = group_and_sort_dicom_series([s1, s2, s3])

        assert result["total_studies"] == 1
        assert result["total_series"] == 1
        assert result["total_slices"] == 3
        assert result["volumetric_reconstruction_supported"] is False
        assert "Not a 3D Reconstructed Volume" in result["clinical_scope"]

        series = result["studies"][study_uid]["series"][series_uid]
        assert series["ordering_method"] == "spatial_normal"
        assert series["slice_count"] == 3

        sorted_positions = [s["spatial_position_mm"] for s in series["slices"]]
        assert sorted_positions == [-10.0, 10.0, 30.0]
        assert [s["sop_instance_uid"] for s in series["slices"]] == ["sop.2", "sop.3", "sop.1"]

    def test_group_and_sort_strictly_isolates_different_series(self):
        """
        Slices from different series (e.g., T1 and T2 acquisitions)
        must never be grouped into the same series volume.
        """
        study_uid = "1.2.840.10008.2"
        series_t1 = "1.2.840.10008.2.1"
        series_t2 = "1.2.840.10008.2.2"

        slice_t1 = MockDicomDataset(
            StudyInstanceUID=study_uid,
            SeriesInstanceUID=series_t1,
            SeriesDescription="T1 Axial Post-Contrast",
            SOPInstanceUID="sop.t1.1",
            InstanceNumber=1,
            filename="t1_01.dcm",
        )
        slice_t2 = MockDicomDataset(
            StudyInstanceUID=study_uid,
            SeriesInstanceUID=series_t2,
            SeriesDescription="T2 FLAIR Axial",
            SOPInstanceUID="sop.t2.1",
            InstanceNumber=1,
            filename="t2_01.dcm",
        )

        result = group_and_sort_dicom_series([slice_t1, slice_t2])
        assert result["total_studies"] == 1
        assert result["total_series"] == 2
        study_series = result["studies"][study_uid]["series"]
        assert series_t1 in study_series
        assert series_t2 in study_series
        assert len(study_series[series_t1]["slices"]) == 1
        assert len(study_series[series_t2]["slices"]) == 1

    def test_group_and_sort_fallback_to_instance_number(self):
        """When spatial normal coordinates are missing, falls back to InstanceNumber."""
        study_uid = "1.2.3.5"
        series_uid = "1.2.3.5.1"
        s1 = MockDicomDataset(
            StudyInstanceUID=study_uid,
            SeriesInstanceUID=series_uid,
            SOPInstanceUID="sop.inst3",
            InstanceNumber=3,
            filename="slice_c.dcm",
        )
        s2 = MockDicomDataset(
            StudyInstanceUID=study_uid,
            SeriesInstanceUID=series_uid,
            SOPInstanceUID="sop.inst1",
            InstanceNumber=1,
            filename="slice_a.dcm",
        )
        result = group_and_sort_dicom_series([s1, s2])
        series = result["studies"][study_uid]["series"][series_uid]
        assert series["ordering_method"] == "instance_number"
        assert [s["instance_number"] for s in series["slices"]] == [1, 3]


# =====================================================================
# 4. Multi-Frame DICOM Support & Fallback
# =====================================================================
class TestMultiFrameDicom:
    def test_multiframe_dicom_extracts_central_slice(self):
        """
        For multi-frame objects (N frames), extracts the central 2D slice
        to provide a representative cross-section for 2D classification models.
        """
        # 5 frames of 32x32 pixels, where frame 2 (the center) has distinct values
        frames = np.zeros((5, 32, 32), dtype=np.uint16)
        frames[2, :, :] = 1000  # Central slice

        ds = MockDicomDataset(
            pixel_array=frames,
            PhotometricInterpretation="MONOCHROME2",
        )
        rgb = dicom_to_rgb(ds)
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8
        # Because all pixels in central frame are 1000, flat normalization yields 0
        assert np.all(rgb == 0)

    def test_unsupported_high_dimensional_pixel_array_raises_value_error(self):
        """4D or higher dimensional arrays raise a clear ValueError."""
        ds = MockDicomDataset(
            pixel_array=np.zeros((2, 5, 32, 32), dtype=np.uint16),
            PhotometricInterpretation="MONOCHROME2",
        )
        with pytest.raises(ValueError, match="Unsupported DICOM pixel array dimensions"):
            dicom_to_rgb(ds)


# =====================================================================
# 5. Metadata De-identification & Single-Slice Scope Safety
# =====================================================================
class TestDicomMetadataAndPrivacy:
    def test_metadata_extraction_includes_single_slice_context(self):
        """Extracted metadata must explicitly specify single-slice evaluation scope."""
        ds = MockDicomDataset(
            PatientID="PAT-00123",
            PatientName="DOE^JANE",
            StudyInstanceUID="study.1",
            SeriesInstanceUID="series.1",
            SeriesDescription="T1 Axial SE",
            InstanceNumber=7,
        )
        meta = extract_dicom_metadata(ds, deidentify=False)
        assert meta["patient_id"] == "PAT-00123"
        assert meta["patient_name"] == "DOE^JANE"
        assert meta["is_single_slice"] is True
        assert meta["is_volumetric_study"] is False
        assert "Single 2D Slice" in meta["slice_context"]
        assert "Instance #7" in meta["slice_context"]

    def test_deidentify_masks_phi_fields(self):
        """When deidentify=True, patient name is anonymized and patient ID is masked."""
        ds = MockDicomDataset(
            PatientID="CLINICAL-SECRET-999",
            PatientName="CONFIDENTIAL^PERSON",
        )
        meta = extract_dicom_metadata(ds, deidentify=True)
        assert meta["patient_name"] == "Anonymous Patient"
        assert meta["patient_id"].startswith("ANON-")
        assert "CONFIDENTIAL" not in meta["patient_name"]
        assert "SECRET" not in meta["patient_id"]


# =====================================================================
# 6. Error Handling and Corrupt Data
# =====================================================================
class TestDicomErrorHandling:
    def test_read_dicom_non_existent_file_returns_controlled_error(self):
        """Non-existent DICOM file returns failure dict without raising an exception."""
        result = read_dicom("/path/to/non_existent_scan_12345.dcm")
        assert result["success"] is False
        assert result["image"] is None
        assert "not found" in result["error"].lower()

    def test_dicom_to_rgb_rejects_non_finite_pixels(self):
        """Pixel array with NaNs or Infs is rejected with ValueError."""
        corrupt_pixels = np.array([[0.0, np.nan], [1.0, 2.0]])
        ds = MockDicomDataset(
            pixel_array=corrupt_pixels,
            PhotometricInterpretation="MONOCHROME2",
        )
        with pytest.raises(ValueError, match="non-finite values"):
            dicom_to_rgb(ds)
