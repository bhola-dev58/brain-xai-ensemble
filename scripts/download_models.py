#!/usr/bin/env python3
"""
scripts/download_models.py
==========================
Automated model weight downloader for BrainTumorXAI.

Downloads Phase 3 Tri-Ensemble weights from the official GitHub Release
into the configured models directory (e.g. /app/models) if missing.

Guarantees:
  - Idempotent: Skips download if valid model files already exist.
  - Atomic: Downloads to a temporary file before renaming to prevent corrupt partial models.
  - Safe: Standard library only (no external dependencies required).
"""

import os
import sys
import time
import urllib.error
import urllib.request

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import MODELS_DIR, logger

RELEASE_TAG = "v3.0.0"
BASE_URL = f"https://github.com/bhola-dev58/brain-xai-ensemble/releases/download/{RELEASE_TAG}"

MODELS = {
    "densenet_full_best.keras": {
        "url": f"{BASE_URL}/densenet_full_best.keras",
        "min_size_bytes": 50_000_000,   # ~61 MB
    },
    "inception_full_best.keras": {
        "url": f"{BASE_URL}/inception_full_best.keras",
        "min_size_bytes": 200_000_000,  # ~240 MB
    },
    "convnext_full_best.keras": {
        "url": f"{BASE_URL}/convnext_full_best.keras",
        "min_size_bytes": 450_000_000,  # ~533 MB
    },
}


def download_file(url: str, dest_path: str, min_size: int, max_retries: int = 3) -> bool:
    """
    Downloads a large model weight file with chunked streaming and atomic rename.
    """
    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)

    filename = os.path.basename(dest_path)
    temp_path = f"{dest_path}.part"

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[ModelDownloader] Downloading {filename} (attempt {attempt}/{max_retries})...")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "BrainTumorXAI-ModelDownloader/3.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                total_size = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1 MB

                with open(temp_path, "wb") as f_out:
                    start_time = time.time()
                    last_log = start_time
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_log > 5.0 or downloaded == total_size:
                            pct = (downloaded / total_size * 100) if total_size else 0
                            mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            print(f"  -> {filename}: {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)")
                            last_log = now

            actual_size = os.path.getsize(temp_path)
            if actual_size < min_size:
                raise ValueError(
                    f"Downloaded file {filename} is too small ({actual_size} bytes < {min_size} bytes)."
                )

            # Atomic move
            os.replace(temp_path, dest_path)
            print(f"[ModelDownloader] [OK] Successfully verified and saved: {filename} ({actual_size / 1024 / 1024:.1f} MB)")
            return True

        except Exception as err:
            print(f"[ModelDownloader] Error downloading {filename}: {err}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            if attempt < max_retries:
                time.sleep(3 * attempt)
            else:
                print(f"[ModelDownloader] [FAILED] Exhausted retries for {filename}.")
                return False

    return False


def ensure_models(models_dir: str = MODELS_DIR) -> bool:
    """
    Verifies that all required Tri-Ensemble models are present in models_dir.
    Downloads any missing models automatically.
    """
    os.makedirs(models_dir, exist_ok=True)
    print("=================================================================")
    print("  BrainTumorXAI — Automated Model Artifact Provisioning")
    print(f"  Target Directory: {models_dir}")
    print("=================================================================")

    all_success = True
    for filename, meta in MODELS.items():
        dest = os.path.join(models_dir, filename)
        if os.path.exists(dest) and os.path.getsize(dest) >= meta["min_size_bytes"]:
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f" [EXISTS] {filename:<26} ({size_mb:.1f} MB) — skipping download.")
        else:
            ok = download_file(meta["url"], dest, meta["min_size_bytes"])
            if not ok:
                all_success = False

    print("-----------------------------------------------------------------")
    if all_success:
        print(" [READY] All Tri-Ensemble model artifacts verified and operational.")
    else:
        print(" [WARNING] One or more model artifacts failed to download.")
    print("=================================================================")
    return all_success


if __name__ == "__main__":
    success = ensure_models()
    sys.exit(0 if success else 1)
