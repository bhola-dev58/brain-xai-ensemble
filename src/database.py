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
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# ── Path Configuration ──────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_BASE_DIR, "data", "patient_history.db")

# ── Schema ──────────────────────────────────────────────────────────────────────
_CREATE_TABLE_SQL = """
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
"""


def _get_connection() -> sqlite3.Connection:
    """Returns a new SQLite connection with row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Creates the database file and scans table if they do not already exist.
    Safe to call multiple times (idempotent).
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    try:
        with _get_connection() as conn:
            conn.execute(_CREATE_TABLE_SQL)
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
) -> int:
    """
    Inserts one completed scan diagnosis record into the database.

    Args:
        predicted_class:  Predicted tumor class name (e.g., "Glioma").
        confidence_pct:   Ensemble confidence percentage (0–100).
        is_tumor:         True if a tumor was detected.
        severity:         Severity grade string (e.g., "Moderate").
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

    Returns:
        The row ID (integer) of the inserted record.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sql = """
        INSERT INTO scans (
            timestamp, patient_id, patient_name, patient_age, patient_gender,
            scan_filename, input_type, predicted_class, confidence_pct, is_tumor,
            severity, tumor_area_pct, location, inference_time, pdf_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        with _get_connection() as conn:
            cursor = conn.execute(sql, (
                timestamp, patient_id, patient_name, patient_age, patient_gender,
                scan_filename, input_type, predicted_class, round(confidence_pct, 2),
                int(is_tumor), severity, round(tumor_area_pct, 2),
                location, round(inference_time, 3), pdf_path
            ))
            conn.commit()
            row_id = cursor.lastrowid
            logger.info(f"Scan logged to DB: ID={row_id}, class={predicted_class}, conf={confidence_pct:.1f}%")
            return row_id
    except sqlite3.Error as e:
        logger.error(f"Failed to log scan to database: {e}")
        return -1


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
