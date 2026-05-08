"""A-share trading calendar helper using Tushare ``trade_cal`` API.

Provides a single public function :func:`is_trading_day` that checks
whether a given date is a trading day on the Shanghai Stock Exchange
(SSE).  Results are cached to a local JSON file so that the Tushare
API is called at most once per calendar month.

Usage::

    from scripts.dashboard_export.trading_calendar import is_trading_day

    if not is_trading_day("2026-05-01"):
        print("Market closed today")

The ``TUSHARE`` environment variable must be set (same token used by
the rest of the pipeline).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.dashboard_export.constants import REPO_ROOT, log

# Local cache file — avoids redundant Tushare API calls.
# Stores {month_key: {date_str: bool}} mappings.
_CACHE_FILE = REPO_ROOT / "output" / "history" / "trading_calendar_cache.json"


def _load_cache() -> dict[str, Any]:
    """Load the calendar cache from disk."""
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    """Persist the calendar cache to disk."""
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _fetch_month_calendar(year: int, month: int) -> dict[str, bool]:
    """Fetch trading calendar for a full month from Tushare.

    Returns a dict mapping ``YYYY-MM-DD`` → ``True``/``False``
    (is_open).
    """
    import tushare as ts

    token = os.getenv("TUSHARE") or os.getenv("TS_TOKEN") or ""
    if not token:
        raise RuntimeError(
            "TUSHARE env var is required for trading calendar check. "
            "Set it in .env or export it before running the pipeline."
        )

    pro = ts.pro_api(token)

    start_date = f"{year}{month:02d}01"
    # End date: last day of month (use 31, Tushare handles overflow)
    end_date = f"{year}{month:02d}31"

    df = pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date)

    result: dict[str, bool] = {}
    for _, row in df.iterrows():
        cal_date = str(row["cal_date"])  # e.g. "20260501"
        is_open = int(row["is_open"]) == 1
        # Normalise to YYYY-MM-DD
        formatted = f"{cal_date[:4]}-{cal_date[4:6]}-{cal_date[6:8]}"
        result[formatted] = is_open

    return result


def is_trading_day(date_str: str | None = None) -> bool:
    """Check whether *date_str* is an A-share (SSE) trading day.

    Args:
        date_str: Date in ``YYYY-MM-DD`` format.  Defaults to today
            (Beijing time, UTC+8).

    Returns:
        ``True`` if the date is a trading day, ``False`` otherwise.

    Raises:
        RuntimeError: If the ``TUSHARE`` env var is not set.
    """
    if date_str is None:
        # Use Beijing time (UTC+8) to determine "today"
        import datetime as _dt

        utc_now = _dt.datetime.now(_dt.timezone.utc)
        beijing_offset = _dt.timedelta(hours=8)
        beijing_now = utc_now + beijing_offset
        date_str = beijing_now.strftime("%Y-%m-%d")

    # Parse year/month for cache key
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        log("WARN", f"Invalid date format: {date_str!r}, expected YYYY-MM-DD")
        return False

    month_key = f"{dt.year}-{dt.month:02d}"

    # Check cache first
    cache = _load_cache()
    if month_key in cache and date_str in cache[month_key]:
        return cache[month_key][date_str]

    # Cache miss — fetch from Tushare
    log("INFO", f"Fetching SSE trading calendar for {month_key} from Tushare")
    try:
        month_data = _fetch_month_calendar(dt.year, dt.month)
    except Exception as exc:
        log("WARN", f"Failed to fetch trading calendar: {exc}")
        # Fallback: weekdays are trading days (imprecise but safe-ish)
        is_weekday = dt.weekday() < 5
        log("WARN", f"Falling back to weekday check: {date_str} → {'weekday' if is_weekday else 'weekend'}")
        return is_weekday

    # Update cache
    cache[month_key] = month_data
    _save_cache(cache)

    return month_data.get(date_str, False)
