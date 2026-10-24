"""
src/security.py
===============
Security, validation, and access control module for BrainTumorXAI.

Responsibilities:
  - Enforce configurable file upload size limits to prevent memory exhaustion (DoS).
  - Verify file magic bytes to ensure only genuine, supported medical images and DICOM files are processed.
  - Sanitize filenames, patient metadata, and identifiers to prevent path traversal and injection.
  - Provide constant-time API key authentication for sensitive medical endpoints when configured.
"""

import io
import os
import re
import secrets
from typing import Optional, Tuple
from fastapi import HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from PIL import Image

from src.config import logger

# ── Configuration Defaults ────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "5000"))
MIN_IMAGE_DIMENSION = 10

API_KEY_HEADER_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


# ── Supported File Signatures (Magic Bytes) ──────────────────────────────────
# Mapping of format name to predicate checking initial bytes
def detect_file_format(header_bytes: bytes) -> Optional[str]:
    """
    Inspects leading bytes of a file to verify known image / DICOM magic numbers.
    Returns format string ('JPEG', 'PNG', 'BMP', 'TIFF', 'DICOM') or None.
    """
    if len(header_bytes) < 4:
        return None

    # JPEG: starts with FF D8 FF
    if header_bytes.startswith(b"\xff\xd8\xff"):
        return "JPEG"

    # PNG: starts with 89 50 4E 47 0D 0A 1A 0A
    if header_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"

    # BMP: starts with 'BM'
    if header_bytes.startswith(b"BM"):
        return "BMP"

    # TIFF: starts with 'II*\x00' (little-endian) or 'MM\x00*' (big-endian)
    if header_bytes.startswith(b"II*\x00") or header_bytes.startswith(b"MM\x00*"):
        return "TIFF"

    # DICOM: standard preamble has 128 bytes then 'DICM'
    if len(header_bytes) >= 132 and header_bytes[128:132] == b"DICM":
        return "DICOM"

    # DICOM without 128-byte preamble (starts directly with DICM)
    if header_bytes.startswith(b"DICM"):
        return "DICOM"

    return None


# ── File Upload Size Validation ───────────────────────────────────────────────
async def validate_and_read_upload(file, max_size_bytes: Optional[int] = None) -> bytes:
    """
    Reads an uploaded file in 64 KB chunks, verifying that total bytes do not
    exceed `max_size_bytes` (or MAX_UPLOAD_SIZE_BYTES). Prevents Denial-of-Service (DoS)
    via unbounded memory allocation.

    Raises:
        HTTPException(413): If the file exceeds the allowed upload limit.
        HTTPException(400): If the file is completely empty.
    """
    if max_size_bytes is None:
        max_size_bytes = MAX_UPLOAD_SIZE_BYTES
    total_bytes = 0
    chunks = []
    chunk_size = 64 * 1024  # 64 KB

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_size_bytes:
            limit_mb = max_size_bytes // (1024 * 1024)
            logger.warning(
                f"Rejected upload exceeding size limit: {total_bytes} bytes > {max_size_bytes} bytes"
            )
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds the maximum allowed size of {limit_mb} MB.",
            )
        chunks.append(chunk)

    if total_bytes == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty (0 bytes). Please upload a valid brain MRI scan.",
        )

    return b"".join(chunks)


# ── Content Validation ────────────────────────────────────────────────────────
def validate_image_content(content: bytes) -> str:
    """
    Validates that raw file bytes correspond to a supported medical image format.
    Checks magic bytes and verifies image integrity using PIL.

    Returns:
        Detected format name ('JPEG', 'PNG', 'DICOM', etc.).

    Raises:
        HTTPException(415): If file signature is unsupported.
        HTTPException(422): If file content is corrupt or cannot be parsed as an image.
    """
    detected_format = detect_file_format(content[:136])
    if not detected_format:
        logger.warning("Rejected upload with unsupported magic bytes / format.")
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported file type. Please upload a valid JPEG, PNG, or DICOM (.dcm) MRI scan."
            ),
        )

    # If it's a standard image (not DICOM), verify structural integrity via PIL
    if detected_format != "DICOM":
        try:
            with Image.open(io.BytesIO(content)) as img:
                img.verify()
        except Exception as e:
            logger.warning(f"Image integrity verification failed: {e}")
            raise HTTPException(
                status_code=422,
                detail="Uploaded image file is corrupted or could not be decoded.",
            )
    else:
        # Verify DICOM header integrity
        try:
            import pydicom
            pydicom.dcmread(io.BytesIO(content), stop_before_pixels=True, force=True)
        except Exception as e:
            logger.warning(f"DICOM header verification failed: {e}")
            raise HTTPException(
                status_code=422,
                detail="Uploaded DICOM file is corrupted or has an unreadable header structure.",
            )

    return detected_format


def validate_image_dimensions(height: int, width: int) -> None:
    """
    Checks that decoded image dimensions fall within reasonable diagnostic bounds.
    Protects against decompression bombs (huge pixel buffers).
    """
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=422,
            detail=f"Image dimensions ({width}x{height}) are too small for clinical analysis (minimum {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}).",
        )
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=422,
            detail=f"Image dimensions ({width}x{height}) exceed maximum allowable size ({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}).",
        )


# ── Input Sanitization ────────────────────────────────────────────────────────
def sanitize_filename(filename: Optional[str]) -> str:
    """
    Strips directory traversal sequences, path separators, and control characters from filenames.
    """
    if not filename:
        return "scan_upload"
    # Take basename only
    base = os.path.basename(filename)
    # Strip null bytes and non-printable characters
    cleaned = re.sub(r"[^\w\s\.-]", "_", base)
    # Truncate to reasonable length
    return cleaned[:128] or "scan_upload"


def sanitize_patient_metadata(
    patient_id: Optional[str],
    patient_name: Optional[str],
    patient_age: Optional[str],
    patient_gender: Optional[str],
) -> Tuple[str, str, str, str]:
    """
    Sanitizes and bounds patient metadata fields to prevent excessive lengths
    or format-string / layout injection in reports.
    """
    # Patient ID: alphanumeric, dash, underscore only, max 64 chars
    raw_pid = (patient_id or "PT-8942").strip()
    safe_pid = re.sub(r"[^A-Za-z0-9_-]", "_", raw_pid)[:64] or "ANON"

    # Patient Name: printable text, max 100 chars
    raw_name = (patient_name or "Anonymous Patient").strip()
    safe_name = re.sub(r"[^\w\s\.,'-]", "", raw_name)[:100] or "Anonymous Patient"

    # Patient Age: max 10 chars, digits or N/A
    raw_age = (patient_age or "45").strip()
    safe_age = re.sub(r"[^0-9A-Za-z\s/-]", "", raw_age)[:10] or "N/A"

    # Patient Gender: constrained or sanitized
    raw_gender = (patient_gender or "Unspecified").strip()
    safe_gender = re.sub(r"[^A-Za-z\s/-]", "", raw_gender)[:20] or "Unspecified"

    return safe_pid, safe_name, safe_age, safe_gender


def validate_analysis_id(analysis_id: Optional[str]) -> Optional[str]:
    """
    Validates that an analysis ID matches expected format (e.g. ANL-...).
    Prevents path traversal or SQL parameter manipulation.
    """
    if not analysis_id:
        return None
    cleaned = analysis_id.strip()
    if not re.match(r"^ANL-[A-Za-z0-9_-]{4,64}$", cleaned):
        raise HTTPException(
            status_code=400,
            detail="Invalid analysis identifier format. Expected 'ANL-<id>'.",
        )
    return cleaned


# ── Authentication and Authorization ──────────────────────────────────────────
def get_configured_api_key() -> Optional[str]:
    """Retrieves the configured API key from environment variables, if any."""
    return os.getenv("BRAIN_XAI_API_KEY") or os.getenv("API_KEY")


def require_api_key(
    request: Request,
    header_key: Optional[str] = Security(_api_key_header),
) -> Optional[str]:
    """
    FastAPI dependency that enforces API key authentication when configured.

    Behavior:
      - If `BRAIN_XAI_API_KEY` (or `API_KEY`) is set in the environment:
          Accepts key via `X-API-Key` header or `Authorization: Bearer <key>`.
          Validates key using constant-time comparison.
          Rejects with 401 Unauthorized if missing or invalid.
      - If no API key is configured in the environment:
          Permits request to proceed (local development mode).
    """
    expected_key = get_configured_api_key()

    # If no key configured in environment, allow unauthenticated access (dev mode)
    if not expected_key:
        return None

    # Check X-API-Key header first
    provided_key = header_key

    # Check Authorization: Bearer <token> fallback
    if not provided_key:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            provided_key = auth_header[7:].strip()

    if not provided_key:
        logger.warning(f"Unauthorized API request to {request.url.path}: missing API key.")
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please provide a valid API key via X-API-Key or Authorization: Bearer header.",
        )

    # Constant-time comparison to prevent timing side-channel attacks
    if not secrets.compare_digest(provided_key, expected_key):
        logger.warning(f"Unauthorized API request to {request.url.path}: invalid API key.")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key provided.",
        )

    return provided_key
