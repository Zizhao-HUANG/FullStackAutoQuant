#!/usr/bin/env python3
"""Check if today (or a given date) is an A-share trading day.

Exit code:
    0 — trading day (pipeline should proceed)
    1 — NOT a trading day (pipeline should skip)

Usage:
    python scripts/check_trading_day.py              # check today (Beijing time)
    python scripts/check_trading_day.py 2026-05-01   # check specific date
    python scripts/check_trading_day.py --quiet       # suppress output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.dashboard_export.trading_calendar import is_trading_day


def main() -> int:
    ap = argparse.ArgumentParser(description="Check if a date is an A-share trading day")
    ap.add_argument("date", nargs="?", default=None, help="Date to check (YYYY-MM-DD). Default: today (Beijing time)")
    ap.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    args = ap.parse_args()

    result = is_trading_day(args.date)

    if not args.quiet:
        date_label = args.date or "today"
        status = "TRADING DAY ✓" if result else "NOT a trading day ✗"
        print(f"[check_trading_day] {date_label} → {status}")

    # Exit code: 0 = trading day, 1 = not
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
