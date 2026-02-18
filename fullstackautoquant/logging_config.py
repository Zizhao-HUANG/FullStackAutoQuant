"""Centralized logging configuration for FullStackAutoQuant.

Usage:
    from fullstackautoquant.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Starting trading run", extra={"date": "2026-01-01"})

All modules should use this instead of print() for production observability.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


class _JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production ingestion (ELK/Datadog/etc)."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Include any extra fields passed via extra={...}
        for key in ("date", "symbol", "side", "volume", "price", "status", "module_name"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class _ReadableFormatter(logging.Formatter):
    """Human-readable formatter for development / TTY output."""

    FMT = "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT)


_CONFIGURED = False


def configure_logging(
    *,
    level: str | None = None,
    json_output: bool | None = None,
) -> None:
    """Configure the root logger for the fullstackautoquant package.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR). Defaults to
               env var FSAQ_LOG_LEVEL or "INFO".
        json_output: If True, output structured JSON lines. Defaults to
                     env var FSAQ_LOG_JSON == "1", or auto-detect non-TTY.
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return

    if level is None:
        level = os.getenv("FSAQ_LOG_LEVEL", "INFO").upper()
    if json_output is None:
        env_json = os.getenv("FSAQ_LOG_JSON", "").strip()
        if env_json:
            json_output = env_json == "1"
        else:
            # Auto: use JSON when stdout is not a TTY (e.g. Docker, cron, CI)
            json_output = not sys.stdout.isatty()

    root_logger = logging.getLogger("fullstackautoquant")
    root_logger.setLevel(getattr(logging, level, logging.INFO))

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter: logging.Formatter
        if json_output:
            formatter = _JSONFormatter()
        else:
            formatter = _ReadableFormatter()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate output
    root_logger.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the fullstackautoquant namespace.

    Automatically calls configure_logging() on first use.

    Args:
        name: Typically __name__ of the calling module.
    """
    configure_logging()
    return logging.getLogger(name)


def reset_logging() -> None:
    """Reset logging configuration (primarily for testing)."""
    global _CONFIGURED  # noqa: PLW0603
    root_logger = logging.getLogger("fullstackautoquant")
    root_logger.handlers.clear()
    _CONFIGURED = False
