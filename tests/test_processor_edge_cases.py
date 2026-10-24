"""
tests/test_processor_edge_cases.py
==================================
Comprehensive unit tests for image preprocessing, brain region segmentation,
morphological contour sizing, and spatial localization heuristics.
"""

import numpy as np
import pytest

from src.processor import (
    extract_brain_region,
    create_segmentation,
    estimate_location,
    estimate_severity,
)


# ------------------------------------------------------------------ #
# 1. Brain Region Extraction Edge Cases
# ------------------------------------------------------------------ #
class TestExtractBrainRegionEdgeCases:
    def test_grayscale_2d_input_cropped_safely(self):
        """A 2D grayscale image (H, W) is cropped safely without errors."""
        gray = np.full((120, 120), 128, dtype=np.uint8)
        out = extract_brain_region(gray)
        assert out.ndim == 2
        assert out.dtype == np.uint8

    def test_rgba_4_channel_input_cropped_safely(self):
        """A 4-channel RGBA image (H, W, 4) is cropped safely."""
        rgba = np.full((120, 120, 4), 200, dtype=np.uint8)
        out = extract_brain_region(rgba)
        assert out is not None
        assert out.ndim == 3

    def test_all_black_blank_image(self):
        """An all-black image (zeros) must not crash contour search and return a valid array."""
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        out = extract_brain_region(blank)
        assert out is not None
        assert out.shape == (100, 100, 3)

    def test_all_white_image(self):
        """An all-white image (saturated) must not crash and return valid bounded array."""
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        out = extract_brain_region(white)
        assert out is not None
        assert out.shape == (100, 100, 3)

    def test_extreme_aspect_ratio_image(self):
        """An image with extreme aspect ratio (e.g. 20x300) must be processed safely."""
        extreme = np.zeros((20, 300, 3), dtype=np.uint8)
        extreme[5:15, 20:280] = 180
        out = extract_brain_region(extreme)
        assert out is not None
        assert out.ndim == 3


# ------------------------------------------------------------------ #
# 2. Morphological Lesion Segmentation Edge Cases
# ------------------------------------------------------------------ #
class TestCreateSegmentationEdgeCases:
    def test_binary_mask_invariant(self):
        """Returned segmentation mask must strictly contain only binary values: 0 and 255."""
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[30:70, 30:70] = 0.85  # strong activation
        orig = np.full((100, 100, 3), 120, dtype=np.uint8)
        brain_mask = np.full((100, 100), 255, dtype=np.uint8)

        overlay, mask, area_pct = create_segmentation(heatmap, orig, brain_mask)
        unique_vals = set(np.unique(mask))
        assert unique_vals.issubset({0, 255})
        assert area_pct > 0.0

    def test_all_zero_heatmap_produces_zero_area(self):
        """When activation heatmap is zero everywhere, estimated tumor percentage must be exactly 0.0."""
        zero_heatmap = np.zeros((100, 100), dtype=np.float32)
        orig = np.full((100, 100, 3), 120, dtype=np.uint8)
        brain_mask = np.full((100, 100), 255, dtype=np.uint8)

        overlay, mask, area_pct = create_segmentation(zero_heatmap, orig, brain_mask)
        assert area_pct == 0.0
        assert np.all(mask == 0)
        assert overlay.shape == orig.shape

    def test_segmentation_output_dimensions_match_input(self):
        """Output overlay and mask must strictly match the spatial dimensions of the input image."""
        h, w = 143, 217
        heatmap = np.random.uniform(0, 1, (h, w)).astype(np.float32)
        orig = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        brain_mask = np.full((h, w), 255, dtype=np.uint8)

        overlay, mask, area_pct = create_segmentation(heatmap, orig, brain_mask)
        assert overlay.shape == (h, w, 3)
        assert mask.shape == (h, w)
        assert 0.0 <= area_pct <= 100.0


# ------------------------------------------------------------------ #
# 3. Spatial Localization Heuristics
# ------------------------------------------------------------------ #
class TestEstimateLocationEdgeCases:
    def test_parietal_activation(self):
        """Focal activation in the middle height maps to Parietal Lobe."""
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[45:55, 45:55] = 1.0
        loc = estimate_location(heatmap, (100, 100))
        assert "Parietal Lobe" in loc

    def test_superior_left_frontal_activation(self):
        """Focal activation in the upper-left quadrant maps to Left Frontal (Superior)."""
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[10:30, 10:30] = 1.0
        loc = estimate_location(heatmap, (100, 100))
        assert "Left Frontal" in loc
        assert "Superior" in loc

    def test_inferior_right_occipital_activation(self):
        """Focal activation in the lower-right quadrant maps to Right Occipital (Inferior)."""
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[75:95, 75:95] = 1.0
        loc = estimate_location(heatmap, (100, 100))
        assert "Right Occipital" in loc
        assert "Inferior" in loc

    def test_superior_right_frontal_activation(self):
        """Focal activation in upper-right quadrant maps to Right Frontal (Superior)."""
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[10:30, 70:90] = 1.0
        loc = estimate_location(heatmap, (100, 100))
        assert "Right Frontal" in loc
        assert "Superior" in loc

    def test_blank_or_flat_heatmap_fallback(self):
        """Flat uniform heatmap falls back safely without raising exceptions."""
        flat = np.full((100, 100), 0.5, dtype=np.float32)
        loc = estimate_location(flat, (100, 100))
        assert isinstance(loc, str)
        assert len(loc) > 0


# ------------------------------------------------------------------ #
# 4. Severity Estimation Clinical Boundary Guards
# ------------------------------------------------------------------ #
class TestEstimateSeverityBoundaries:
    def test_high_confidence_with_large_tumor_is_high(self):
        """Confidence > 95% with tumor area > 5% must classify as High severity."""
        sev, color = estimate_severity(confidence=96.0, tumor_percentage=6.0)
        assert sev == "High"
        assert color.startswith("#")

    def test_high_confidence_with_zero_tumor_area_safeguard(self):
        """CRITICAL: High confidence with 0% tumor area must NOT be marked High (defaults to Low)."""
        sev, color = estimate_severity(confidence=98.0, tumor_percentage=0.0)
        assert sev == "Low"
        assert sev != "High"

    def test_moderate_confidence_and_area(self):
        """Confidence > 80% with area > 3% must classify as Moderate."""
        sev, _ = estimate_severity(confidence=85.0, tumor_percentage=4.0)
        assert sev == "Moderate"

    def test_low_confidence_boundary(self):
        """Confidence between 60% and 80% classifies as Low."""
        sev, _ = estimate_severity(confidence=70.0, tumor_percentage=1.0)
        assert sev == "Low"

    def test_borderline_confidence_boundary(self):
        """Confidence < 55% classifies as Borderline."""
        sev, _ = estimate_severity(confidence=50.0, tumor_percentage=0.5)
        assert sev == "Borderline"
