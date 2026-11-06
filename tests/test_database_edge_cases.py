"""
tests/test_database_edge_cases.py
==================================
Edge-case and resilience tests for src/database.py

Covers:
  1. Database initialises cleanly to a custom temp path.
  2. init_db() is safe to call multiple times (idempotent).
  3. log_scan() returns a positive row ID on success.
  4. log_scan() with FAILED status zeroes confidence and forces is_tumor=False.
  5. log_scan() correctly serialises dict model_provenance to JSON string.
  6. log_scan() correctly serialises list model_provenance to JSON string.
  7. log_scan() with None provenance stores NULL without crashing.
  8. get_scan_by_analysis_id() returns None for a non-existent ID.
  9. get_scan_by_analysis_id() returns the correct record for an existing ID.
 10. get_scans_paginated() respects limit and offset.
 11. get_scan_count() increments accurately after insertions.
 12. clear_all_scans() removes all records and returns correct count.
 13. get_scan_provenance() returns None for unknown analysis_id.
 14. get_scan_provenance() correctly parses JSON provenance from stored record.
 15. get_all_scans() returns newest-first ordering.
"""
import json
import os
import tempfile

import pytest

import src.database as db_mod
from src.database import (
    init_db,
    log_scan,
    get_scan_by_analysis_id,
    get_scans_paginated,
    get_scan_count,
    clear_all_scans,
    get_all_scans,
    get_scan_provenance,
)


# ------------------------------------------------------------------ #
# Fixture: isolated temp database for each test
# ------------------------------------------------------------------ #
@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """
    Redirect every test to its own throwaway SQLite database so tests
    never interfere with the production database or each other.
    """
    custom_db = str(tmp_path / "test_history.db")
    monkeypatch.setattr(db_mod, "DB_PATH", custom_db)
    # Patch config.DB_PATH too so _get_connection picks up the right path
    import src.config as cfg
    monkeypatch.setattr(cfg, "DB_PATH", custom_db)
    init_db()
    yield custom_db


def _insert_scan(analysis_id="ANL-EDGETEST-0001", status="COMPLETED", **overrides):
    """Helper: insert a minimal scan record and return the row ID."""
    defaults = dict(
        predicted_class="Glioma Tumor",
        confidence_pct=88.5,
        is_tumor=True,
        severity="Moderate",
        tumor_area_pct=3.2,
        location="Right Frontal Lobe",
        inference_time=0.045,
        patient_id="PT-TEST",
        analysis_id=analysis_id,
        status=status,
    )
    defaults.update(overrides)
    return log_scan(**defaults)


# ------------------------------------------------------------------ #
# 1 & 2. Initialisation
# ------------------------------------------------------------------ #
class TestDatabaseInitialisation:
    def test_db_file_is_created_on_init(self, isolated_db):
        """The SQLite file must exist after init_db()."""
        assert os.path.exists(isolated_db)

    def test_init_db_is_idempotent(self, isolated_db):
        """Calling init_db() multiple times must not raise or corrupt the schema."""
        init_db()
        init_db()
        # Still able to query
        count = get_scan_count()
        assert count >= 0


# ------------------------------------------------------------------ #
# 3. log_scan success path
# ------------------------------------------------------------------ #
class TestLogScanSuccess:
    def test_log_scan_returns_positive_row_id(self):
        """A successful insertion must return a positive integer row ID."""
        row_id = _insert_scan(analysis_id="ANL-ROWID-001")
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_log_scan_stored_values_match_input(self):
        """Stored values must exactly match what was inserted."""
        aid = "ANL-MATCH-001"
        _insert_scan(
            analysis_id=aid,
            predicted_class="Pituitary Tumor",
            confidence_pct=92.0,
            is_tumor=True,
            severity="Mild",
            location="Sellar Region",
            tumor_area_pct=1.5,
            inference_time=0.03,
            patient_id="PT-MATCH",
        )
        record = get_scan_by_analysis_id(aid)
        assert record is not None
        assert record["predicted_class"] == "Pituitary Tumor"
        assert record["patient_id"] == "PT-MATCH"
        assert record["severity"] == "Mild"
        assert record["is_tumor"] == 1  # SQLite stores bool as int


# ------------------------------------------------------------------ #
# 4. FAILED status overrides
# ------------------------------------------------------------------ #
class TestFailedScanOverrides:
    def test_failed_status_forces_confidence_to_zero(self):
        """When status='FAILED', confidence_pct must be stored as 0.0 regardless of input."""
        aid = "ANL-FAIL-001"
        _insert_scan(analysis_id=aid, status="FAILED", confidence_pct=75.0)
        record = get_scan_by_analysis_id(aid)
        assert record["confidence_pct"] == 0.0

    def test_failed_status_forces_is_tumor_to_false(self):
        """When status='FAILED', is_tumor must be stored as 0 (False)."""
        aid = "ANL-FAIL-002"
        _insert_scan(analysis_id=aid, status="FAILED", is_tumor=True)
        record = get_scan_by_analysis_id(aid)
        assert record["is_tumor"] == 0


# ------------------------------------------------------------------ #
# 5, 6, 7. model_provenance serialisation
# ------------------------------------------------------------------ #
class TestProvenanceSerialisation:
    def test_dict_provenance_serialised_to_json_string(self):
        """A dict passed as model_provenance must be JSON-serialised in the DB."""
        aid = "ANL-PROV-DICT"
        prov = {"convnext": "v3", "inception": "v2"}
        _insert_scan(analysis_id=aid, model_provenance=prov)
        record = get_scan_by_analysis_id(aid)
        stored = record["model_provenance"]
        assert stored is not None
        parsed = json.loads(stored)
        assert parsed == prov

    def test_list_provenance_serialised_to_json_string(self):
        """A list passed as model_provenance must be JSON-serialised."""
        aid = "ANL-PROV-LIST"
        prov = ["ConvNeXt-Small", "InceptionV3", "DenseNet121"]
        _insert_scan(analysis_id=aid, model_provenance=prov)
        record = get_scan_by_analysis_id(aid)
        parsed = json.loads(record["model_provenance"])
        assert parsed == prov

    def test_none_provenance_stored_as_null(self):
        """None provenance must result in NULL stored in the DB without error."""
        aid = "ANL-PROV-NONE"
        _insert_scan(analysis_id=aid, model_provenance=None)
        record = get_scan_by_analysis_id(aid)
        assert record["model_provenance"] is None


# ------------------------------------------------------------------ #
# 8 & 9. get_scan_by_analysis_id
# ------------------------------------------------------------------ #
class TestGetScanByAnalysisId:
    def test_returns_none_for_nonexistent_id(self):
        """Querying a non-existent analysis_id must return None."""
        result = get_scan_by_analysis_id("ANL-DOESNOTEXIST-9999")
        assert result is None

    def test_returns_correct_record_for_existing_id(self):
        """Must return the exact record that was previously inserted."""
        aid = "ANL-FETCH-001"
        _insert_scan(analysis_id=aid, predicted_class="No Tumor")
        record = get_scan_by_analysis_id(aid)
        assert record is not None
        assert record["analysis_id"] == aid
        assert record["predicted_class"] == "No Tumor"


# ------------------------------------------------------------------ #
# 10. get_scans_paginated
# ------------------------------------------------------------------ #
class TestGetScansPaginated:
    def test_limit_is_respected(self):
        """get_scans_paginated(limit=2) must return at most 2 records."""
        for i in range(5):
            _insert_scan(analysis_id=f"ANL-PAGE-{i:04d}")
        results = get_scans_paginated(limit=2, offset=0)
        assert len(results) == 2

    def test_offset_skips_records(self):
        """Offset must skip the correct number of records."""
        for i in range(4):
            _insert_scan(analysis_id=f"ANL-OFFSET-{i:04d}")
        all_results = get_scans_paginated(limit=10, offset=0)
        offset_results = get_scans_paginated(limit=10, offset=2)
        assert len(offset_results) == len(all_results) - 2

    def test_zero_results_when_offset_exceeds_total(self):
        """Offset beyond total count must return an empty list."""
        _insert_scan(analysis_id="ANL-OFFSET-OVER")
        results = get_scans_paginated(limit=10, offset=1000)
        assert results == []

    def test_limit_clamped_to_max_200(self):
        """Limit values > 200 must be clamped to 200 without raising."""
        # Insert only 3 records; clamped limit=200 should still return 3
        for i in range(3):
            _insert_scan(analysis_id=f"ANL-CLAMP-{i:04d}")
        results = get_scans_paginated(limit=9999, offset=0)
        assert len(results) == 3


# ------------------------------------------------------------------ #
# 11. get_scan_count
# ------------------------------------------------------------------ #
class TestGetScanCount:
    def test_count_starts_at_zero_on_fresh_db(self):
        """Fresh isolated DB must have 0 records."""
        assert get_scan_count() == 0

    def test_count_increments_after_insertions(self):
        """get_scan_count() must accurately reflect inserted records."""
        _insert_scan(analysis_id="ANL-COUNT-001")
        _insert_scan(analysis_id="ANL-COUNT-002")
        assert get_scan_count() == 2


# ------------------------------------------------------------------ #
# 12. clear_all_scans
# ------------------------------------------------------------------ #
class TestClearAllScans:
    def test_clear_returns_number_of_deleted_rows(self):
        """clear_all_scans() must return the count of deleted rows."""
        _insert_scan(analysis_id="ANL-CLR-001")
        _insert_scan(analysis_id="ANL-CLR-002")
        deleted = clear_all_scans()
        assert deleted == 2

    def test_clear_empties_the_table(self):
        """After clear_all_scans(), get_scan_count() must return 0."""
        _insert_scan(analysis_id="ANL-CLR-003")
        clear_all_scans()
        assert get_scan_count() == 0

    def test_clear_on_empty_table_returns_zero(self):
        """Calling clear on an already-empty table must return 0 without error."""
        result = clear_all_scans()
        assert result == 0


# ------------------------------------------------------------------ #
# 13 & 14. get_scan_provenance
# ------------------------------------------------------------------ #
class TestGetScanProvenance:
    def test_provenance_returns_none_for_unknown_id(self):
        """Must return None for a non-existent analysis_id."""
        result = get_scan_provenance("ANL-PROV-UNKNOWN-9999")
        assert result is None

    def test_provenance_dict_is_parsed_from_json(self):
        """Stored JSON provenance must be deserialised back to a dict."""
        aid = "ANL-PROV-PARSE-001"
        prov = {"architecture": "Tri-Ensemble", "version": "3.0.0"}
        _insert_scan(analysis_id=aid, model_provenance=prov)
        result = get_scan_provenance(aid)
        assert result is not None
        assert result["analysis_id"] == aid
        assert result["model_provenance"] == prov

    def test_provenance_contains_required_keys(self):
        """Returned provenance dict must contain all required metadata keys."""
        aid = "ANL-PROV-KEYS-001"
        _insert_scan(analysis_id=aid)
        result = get_scan_provenance(aid)
        for key in ("analysis_id", "timestamp", "status", "predicted_class", "model_provenance"):
            assert key in result


# ------------------------------------------------------------------ #
# 15. get_all_scans ordering
# ------------------------------------------------------------------ #
class TestGetAllScansOrdering:
    def test_scans_returned_newest_first(self):
        """Records must be returned in descending insertion order (newest first)."""
        _insert_scan(analysis_id="ANL-ORD-001")
        _insert_scan(analysis_id="ANL-ORD-002")
        _insert_scan(analysis_id="ANL-ORD-003")
        all_scans = get_all_scans()
        ids = [s["analysis_id"] for s in all_scans]
        # Newest (last inserted) must be first
        assert ids[0] == "ANL-ORD-003"
        assert ids[-1] == "ANL-ORD-001"

