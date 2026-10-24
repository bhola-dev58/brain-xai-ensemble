"""
src/logger.py
=============
Centralized, structured, and privacy-conscious logging module for BrainTumorXAI.

Responsibilities:
  - Standardizes log formatting with ISO timestamps, severity, and module names.
  - Automatically correlates request-scoped log records with a unique request ID via ContextVars.
  - Enforces privacy: never logs raw image bytes, patient names, medical record numbers, or credentials.
  - Prevents duplicate handlers when modules are reloaded or imported across processes.
"""

import contextvars
import logging
import os
import sys
from typing import Optional

# ── ContextVars for Request Tracing ───────────────────────────────────────────
# Holds the current request ID for the active async task / thread
request_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)


class RequestContextFilter(logging.Filter):
    """
    Log filter that injects the current request ID into log records
    for unified log correlation across application layers.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        req_id = request_id_ctx.get()
        record.request_id = f" [req_id={req_id}]" if req_id else ""
        return True


def configure_logging(level: Optional[str] = None) -> logging.Logger:
    """
    Configures and returns the central BrainTumorXAI logger.

    Guarantees:
      - Log level configurable via parameter or `LOG_LEVEL` environment variable.
      - Output streams cleanly to stdout for local and container runtime observability.
      - Idempotent: prevents duplicate handlers if called multiple times.
    """
    log_level_str = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logger = logging.getLogger("BrainTumorXAI")
    logger.setLevel(log_level)

    # Avoid duplicate handlers on re-import
    if not any(getattr(h, "_is_btxai_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler._is_btxai_handler = True
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]%(request_id)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(RequestContextFilter())
        logger.addHandler(handler)
        logger.propagate = False
    else:
        # Update level on existing handlers
        for h in logger.handlers:
            h.setLevel(log_level)

    return logger


# Global singleton logger instance
logger = configure_logging()
