"""
tests/test_logger.py
====================
Unit tests for src/logger.py

Covers:
  1. Logger is returned and has correct name.
  2. configure_logging() is idempotent (no duplicate handlers).
  3. LOG_LEVEL environment variable is respected.
  4. RequestContextFilter injects request_id into log records.
  5. RequestContextFilter works when request_id is None (no context set).
  6. Log records include the request_id field when context is active.
  7. Invalid LOG_LEVEL falls back to INFO.
  8. Logger does not propagate to root logger (prevents duplicate console output).
"""
import logging
import os

import pytest

from src.logger import configure_logging, request_id_ctx, RequestContextFilter


# ------------------------------------------------------------------ #
# 1. Logger identity
# ------------------------------------------------------------------ #
class TestLoggerIdentity:
    def test_configure_logging_returns_logger(self):
        """configure_logging must return a logging.Logger instance."""
        result = configure_logging()
        assert isinstance(result, logging.Logger)

    def test_logger_name_is_braintumorxai(self):
        """Logger name must be 'BrainTumorXAI'."""
        lgr = configure_logging()
        assert lgr.name == "BrainTumorXAI"

    def test_logger_does_not_propagate_to_root(self):
        """Logger must NOT propagate to root to prevent duplicate log output."""
        lgr = configure_logging()
        assert lgr.propagate is False


# ------------------------------------------------------------------ #
# 2. Idempotency — no duplicate handlers on re-import
# ------------------------------------------------------------------ #
class TestIdempotency:
    def test_multiple_calls_do_not_add_duplicate_handlers(self):
        """Calling configure_logging() multiple times must not accumulate handlers."""
        lgr = configure_logging()
        handler_count_before = len(lgr.handlers)
        configure_logging()
        configure_logging()
        handler_count_after = len(lgr.handlers)
        assert handler_count_after == handler_count_before

    def test_logger_has_at_least_one_handler(self):
        """Logger must have at least one stream handler after configuration."""
        lgr = configure_logging()
        assert len(lgr.handlers) >= 1


# ------------------------------------------------------------------ #
# 3. LOG_LEVEL env variable
# ------------------------------------------------------------------ #
class TestLogLevel:
    def test_explicit_level_debug_is_applied(self):
        """Passing 'DEBUG' level must set logger level to DEBUG."""
        lgr = configure_logging(level="DEBUG")
        assert lgr.level == logging.DEBUG

    def test_explicit_level_warning_is_applied(self):
        """Passing 'WARNING' must set logger level to WARNING."""
        lgr = configure_logging(level="WARNING")
        assert lgr.level == logging.WARNING
        # Restore to INFO for subsequent tests
        configure_logging(level="INFO")

    def test_env_var_log_level_respected(self, monkeypatch):
        """LOG_LEVEL env variable must be respected when no explicit level is passed."""
        monkeypatch.setenv("LOG_LEVEL", "ERROR")
        lgr = configure_logging()
        assert lgr.level == logging.ERROR
        # Restore
        configure_logging(level="INFO")

    def test_invalid_log_level_falls_back_to_info(self):
        """An unrecognised level string must fall back to INFO without raising."""
        lgr = configure_logging(level="NONEXISTENT_LEVEL")
        # getattr(logging, 'NONEXISTENT_LEVEL', logging.INFO) → returns INFO
        assert lgr.level == logging.INFO


# ------------------------------------------------------------------ #
# 4 & 5. RequestContextFilter
# ------------------------------------------------------------------ #
class TestRequestContextFilter:
    def _make_record(self):
        """Creates a minimal LogRecord for filter testing."""
        return logging.LogRecord(
            name="BrainTumorXAI",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )

    def test_filter_returns_true_always(self):
        """RequestContextFilter.filter() must always return True (never suppress records)."""
        f = RequestContextFilter()
        record = self._make_record()
        result = f.filter(record)
        assert result is True

    def test_filter_injects_empty_request_id_when_no_context(self):
        """When no request_id is set, record.request_id must be an empty string."""
        # Ensure no context is set
        token = request_id_ctx.set(None)
        try:
            f = RequestContextFilter()
            record = self._make_record()
            f.filter(record)
            assert record.request_id == ""
        finally:
            request_id_ctx.reset(token)

    def test_filter_injects_request_id_when_context_set(self):
        """When request_id is set via context var, record.request_id must include it."""
        test_id = "test-req-abc123"
        token = request_id_ctx.set(test_id)
        try:
            f = RequestContextFilter()
            record = self._make_record()
            f.filter(record)
            assert test_id in record.request_id
        finally:
            request_id_ctx.reset(token)

    def test_filter_format_uses_bracket_notation(self):
        """The injected request_id must follow ' [req_id=<id>]' format."""
        test_id = "req-98765"
        token = request_id_ctx.set(test_id)
        try:
            f = RequestContextFilter()
            record = self._make_record()
            f.filter(record)
            assert record.request_id == f" [req_id={test_id}]"
        finally:
            request_id_ctx.reset(token)


# ------------------------------------------------------------------ #
# 6. Log record emission (smoke test — no crash)
# ------------------------------------------------------------------ #
class TestLogEmission:
    def test_logger_can_emit_info_message_without_error(self, caplog):
        """Emitting an INFO log via the configured logger must not raise any exception."""
        lgr = configure_logging()
        with caplog.at_level(logging.INFO, logger="BrainTumorXAI"):
            lgr.info("BrainTumorXAI test log emission — OK")

    def test_logger_can_emit_warning_without_error(self, caplog):
        """Emitting a WARNING must not raise."""
        lgr = configure_logging()
        with caplog.at_level(logging.WARNING, logger="BrainTumorXAI"):
            lgr.warning("BrainTumorXAI test warning — OK")

    def test_logger_can_emit_error_without_error(self, caplog):
        """Emitting an ERROR must not raise."""
        lgr = configure_logging()
        with caplog.at_level(logging.ERROR, logger="BrainTumorXAI"):
            lgr.error("BrainTumorXAI test error — OK")

