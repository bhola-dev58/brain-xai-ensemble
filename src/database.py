"""
src/database.py
===============
SQLite patient case history database module for Brain XAI Ensemble.

Responsibilities:
  - Initialize the local SQLite database and schema on first run.
  - Log every successful prediction as a structured scan record.
  - Retrieve, count, and clear scan history records.

Database file location: data/patient_history.db
"""

import sqlite3
import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

# ── Path & Logger Configuration ───────────────────────────────────────────────
from src.config import DB_PATH, logger

# ── Schema ──────────────────────────────────────────────────────────────────────
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS scans (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT    NOT NULL,
    patient_id        TEXT    NOT NULL DEFAULT 'N/A',
    patient_name      TEXT    NOT NULL DEFAULT 'Anonymous',
    patient_age       TEXT    NOT NULL DEFAULT 'N/A',
    patient_gender    TEXT    NOT NULL DEFAULT 'N/A',
    scan_filename     TEXT    NOT NULL DEFAULT 'N/A',
    input_type        TEXT    NOT NULL DEFAULT 'IMAGE',
    predicted_class   TEXT    NOT NULL DEFAULT 'N/A',
    confidence_pct    REAL    NOT NULL DEFAULT 0.0,
    is_tumor          INTEGER NOT NULL DEFAULT 0,
    severity          TEXT    NOT NULL DEFAULT 'N/A',
    tumor_area_pct    REAL    NOT NULL DEFAULT 0.0,
    location          TEXT    NOT NULL DEFAULT 'N/A',
    inference_time    REAL    NOT NULL DEFAULT 0.0,
    pdf_path          TEXT    DEFAULT NULL,
    analysis_id       TEXT    DEFAULT NULL,
    status            TEXT    NOT NULL DEFAULT 'COMPLETED',
    error_code        TEXT    DEFAULT NULL,
    pipeline_version  TEXT    DEFAULT NULL,
    model_provenance  TEXT    DEFAULT NULL
);
"""


def _get_connection() -> sqlite3.Connection:
    """Returns a new SQLite connection with row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Creates the database file and scans table if they do not already exist.
    Performs backward-compatible column migration if needed.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    try:
        with _get_connection() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            # Backward-compatible column migration
            cursor = conn.execute("PRAGMA table_info(scans)")
            columns = [row["name"] for row in cursor.fetchall()]
            if "analysis_id" not in columns:
                conn.execute("ALTER TABLE scans ADD COLUMN analysis_id TEXT DEFAULT NULL")
            if "status" not in columns:
                conn.execute("ALTER TABLE scans ADD COLUMN status TEXT NOT NULL DEFAULT 'COMPLETED'")
            if "error_code" not in columns:
                conn.execute("ALTER TABLE scans ADD COLUMN error_code TEXT DEFAULT NULL")
            if "pipeline_version" not in columns:
                conn.execute("ALTER TABLE scans ADD COLUMN pipeline_version TEXT DEFAULT NULL")
            if "model_provenance" not in columns:
                conn.execute("ALTER TABLE scans ADD COLUMN model_provenance TEXT DEFAULT NULL")
            conn.commit()
        logger.info(f"Patient history database initialized at: {DB_PATH}")
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def log_scan(
    predicted_class: str,
    confidence_pct: float,
    is_tumor: bool,
    severity: str,
    tumor_area_pct: float,
    location: str,
    inference_time: float,
    patient_id: str = "N/A",
    patient_name: str = "Anonymous",
    patient_age: str = "N/A",
    patient_gender: str = "N/A",
    scan_filename: str = "N/A",
    input_type: str = "IMAGE",
    pdf_path: Optional[str] = None,
    analysis_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    status: str = "COMPLETED",
    error_code: Optional[str] = None,
    pipeline_version: Optional[str] = None,
    model_provenance: Optional[Any] = None,
) -> int:
    """
    Inserts one analysis record into the database.
    Can record completed predictions or failed inferences.

    Args:
        predicted_class:  Predicted tumor class name (e.g., "Glioma").
        confidence_pct:   Ensemble confidence percentage (0–100).
        is_tumor:         True if a tumor was detected.
        severity:         Severity heuristic string (e.g., "Moderate").
        tumor_area_pct:   Estimated tumor tissue area as a percentage.
        location:         Estimated brain region description.
        inference_time:   Wall-clock inference duration in seconds.
        patient_id:       Patient identifier string.
        patient_name:     Patient full name.
        patient_age:      Patient age string.
        patient_gender:   Patient gender string.
        scan_filename:    Original filename of the uploaded scan.
        input_type:       "IMAGE" for standard formats or "DICOM" for .dcm files.
        pdf_path:         Absolute path to the generated PDF report, if any.
        analysis_id:      Traceable unique identifier for the analysis.
        timestamp:        Creation timestamp (ISO/UTC).
        status:           "COMPLETED" or "FAILED".
        error_code:       Standardized error category for failed analyses.
        pipeline_version: Pipeline version identifier.
        model_provenance: Architecture names, weights, and artifact digests.

    Returns:
        The row ID (integer) of the inserted record.
    """
    resolved_ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # If failed, ensure it is honestly represented
    if status == "FAILED":
        predicted_class = "FAILED" if predicted_class in ("N/A", "") else predicted_class
        confidence_pct = 0.0
        is_tumor = False

    prov_str = (
        json.dumps(model_provenance)
        if isinstance(model_provenance, (dict, list))
        else (str(model_provenance) if model_provenance is not None else None)
    )

    sql = """
        INSERT INTO scans (
            timestamp, patient_id, patient_name, patient_age, patient_gender,
            scan_filename, input_type, predicted_class, confidence_pct, is_tumor,
            severity, tumor_area_pct, location, inference_time, pdf_path, analysis_id,
            status, error_code, pipeline_version, model_provenance
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        with _get_connection() as conn:
            cursor = conn.execute(sql, (
                resolved_ts, patient_id, patient_name, patient_age, patient_gender,
                scan_filename, input_type, predicted_class, round(confidence_pct, 2),
                int(is_tumor), severity, round(tumor_area_pct, 2),
                location, round(inference_time, 3), pdf_path, analysis_id,
                status, error_code, pipeline_version, prov_str
            ))
            conn.commit()
            row_id = cursor.lastrowid
            logger.info(f"Scan logged to DB: ID={row_id}, analysis_id={analysis_id}, status={status}, class={predicted_class}")
            return row_id
    except sqlite3.Error as e:
        logger.error(f"Failed to log scan to database: {e}")
        return -1


def get_scan_provenance(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves full provenance metadata for a given analysis ID."""
    scan = get_scan_by_analysis_id(analysis_id)
    if not scan:
        return None
    prov_raw = scan.get("model_provenance")
    parsed_prov = None
    if prov_raw:
        try:
            parsed_prov = json.loads(prov_raw)
        except Exception:
            parsed_prov = prov_raw
    return {
        "analysis_id": scan.get("analysis_id"),
        "timestamp": scan.get("timestamp"),
        "status": scan.get("status", "COMPLETED"),
        "error_code": scan.get("error_code"),
        "pipeline_version": scan.get("pipeline_version"),
        "predicted_class": scan.get("predicted_class"),
        "confidence_pct": scan.get("confidence_pct"),
        "model_provenance": parsed_prov,
    }


def get_scan_by_analysis_id(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a specific scan record by its unique analysis ID."""
    try:
        with _get_connection() as conn:
            cursor = conn.execute("SELECT * FROM scans WHERE analysis_id = ? LIMIT 1", (analysis_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve scan by analysis_id {analysis_id}: {e}")
        return None


def get_all_scans() -> List[Dict[str, Any]]:
    """
    Retrieves all scan records, ordered newest first.

    Returns:
        A list of dictionaries, one per row.
    """
    try:
        with _get_connection() as conn:
            cursor = conn.execute("SELECT * FROM scans ORDER BY id DESC")
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve scan history: {e}")
        return []


def get_scans_paginated(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Retrieves a paginated list of scan records, ordered newest first.
    Uses parameterized SQL to guarantee safety against injection attacks.
    """
    safe_limit = max(1, min(limit, 200))
    safe_offset = max(0, offset)
    try:
        with _get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM scans ORDER BY id DESC LIMIT ? OFFSET ?",
                (safe_limit, safe_offset)
            )
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve paginated scan history: {e}")
        return []


def get_scan_count() -> int:
    """Returns the total number of scan records in the database."""
    try:
        with _get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM scans")
            return cursor.fetchone()[0]
    except sqlite3.Error as e:
        logger.error(f"Failed to count scans: {e}")
        return 0


def clear_all_scans() -> int:
    """
    Deletes all records from the scans table.

    Returns:
        Number of rows deleted.
    """
    try:
        with _get_connection() as conn:
            cursor = conn.execute("DELETE FROM scans")
            conn.commit()
            deleted = cursor.rowcount
            logger.info(f"Cleared {deleted} scan records from patient history database.")
            return deleted
    except sqlite3.Error as e:
        logger.error(f"Failed to clear scan history: {e}")
        return 0


# ── Auto-initialize on module import ───────────────────────────────────────────
try:
    init_db()
except Exception as _db_init_err:
    logger.warning(f"Database could not be initialized on import: {_db_init_err}")
