"""Shared constants and utilities for the dashboard export pipeline."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Dashboard display constants
# ---------------------------------------------------------------------------
ANNUALIZATION_DAYS = 252
RISK_FREE_ANNUAL = 0.02
TOP_K_DISPLAY = 30  # Number of top signals to include in dashboard
CONFIDENCE_BINS = [0.96, 0.97, 0.98, 0.99, 1.00]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(step: str, msg: str) -> None:
    """Structured log line with timestamp and step label."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{step}] {msg}")
