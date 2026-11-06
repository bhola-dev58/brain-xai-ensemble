"""
tests/test_download_models.py
=============================
Unit tests for the automated model downloader utility (scripts/download_models.py).
"""

import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from scripts.download_models import ensure_models, download_file, MODELS


def test_ensure_models_skips_when_all_exist():
    """ensure_models should skip downloads if all files exist and are sufficiently large."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create dummy files with sufficient size
        for fname, meta in MODELS.items():
            fpath = os.path.join(tmp_dir, fname)
            with open(fpath, "wb") as f:
                f.seek(meta["min_size_bytes"] + 10)
                f.write(b"\0")

        with patch("scripts.download_models.download_file") as mock_dl:
            ready = ensure_models(models_dir=tmp_dir)
            assert ready is True
            mock_dl.assert_not_called()


def test_download_file_fails_if_size_below_minimum():
    """download_file should reject files smaller than the minimum expected size."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = os.path.join(tmp_dir, "test_model.keras")

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "100"}
        mock_resp.read.side_effect = [b"short data", b""]
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            success = download_file("https://dummy.url", dest, min_size=5000, max_retries=1)
            assert success is False
            assert not os.path.exists(dest)
