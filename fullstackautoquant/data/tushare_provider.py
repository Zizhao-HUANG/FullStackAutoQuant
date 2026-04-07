"""Tushare -> Qlib minimal data provider.

Replaces the Docker/Dolt pipeline by pulling only ~200 trading days of CSI300
OHLCV data directly from Tushare, converting to Qlib binary format. Total data
volume: ~3 MB instead of ~5 GB.

Price adjustment:
    Tushare pro.daily() returns UNADJUSTED prices. The existing pipeline
    (qlib_update.sh -> normalize.py) applies backward adjustment so that
    Alpha158 temporal features (Ref($close,60), Corr, etc.) see continuous
    price series across splits and dividends. This module does the same:

        adjusted = raw * adj_factor / latest_adj_factor

Qlib binary format (verified against qlib 0.9.7, file_storage.py:299-310):
    Each field is stored as <instrument>/<field>.day.bin:
      bytes[0:4] = start_index (float32, little-endian) -- calendar offset
      bytes[4:]  = data values (float32[], little-endian)

adj_factor strategies:
    - "boundary_probe" (default): 2 calls for boundary dates, then per-stock
      only for stocks whose factor changed. Best when few corporate actions.
    - "by_date": Fetch adj_factor by trade_date for all dates (~200 calls),
      then pivot locally. Always exactly N_dates calls, predictable.

Parallel fetch:
    Per-stock adj_factor calls use ThreadPoolExecutor(max_workers=3) with
    the thread-safe rate limiter to reduce wall-clock time by ~3x.

Usage:
    from fullstackautoquant.data.tushare_provider import build_minimal_qlib_data
    build_minimal_qlib_data(token="...", n_days=200)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from fullstackautoquant.logging_config import get_logger
from fullstackautoquant.resilience import RateLimiter, retry_on_exception

logger = get_logger(__name__)

# Qlib field names (without $ prefix) that go into binary
_PRICE_FIELDS = ["open", "high", "low", "close"]
_ALL_FIELDS = _PRICE_FIELDS + ["volume", "factor"]

# Cache directory for API responses (TTL-based)
_CACHE_DIR = Path(os.path.expanduser("~/.cache/fullstackautoquant"))


# ---------------------------------------------------------------------------
# Disk cache for API responses (TTL-based)
# ---------------------------------------------------------------------------


def _cache_path(namespace: str, key: str) -> Path:
    """Build a deterministic cache file path."""
    h = hashlib.md5(key.encode()).hexdigest()[:12]  # noqa: S324
    return _CACHE_DIR / namespace / f"{h}.json"


def _read_cache(namespace: str, key: str, ttl_seconds: int = 86400) -> list[str] | None:
    """Read cached data if it exists and is within TTL.

    Returns None if cache miss or expired.
    """
    path = _cache_path(namespace, key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        cached_at = data.get("cached_at", 0)
        if time.time() - cached_at > ttl_seconds:
            logger.debug("Cache expired: %s/%s", namespace, key)
            return None
        return data.get("value")
    except Exception:
        return None


def _write_cache(namespace: str, key: str, value: list[str]) -> None:
    """Write data to disk cache with timestamp."""
    path = _cache_path(namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"cached_at": time.time(), "key": key, "value": value}
    path.write_text(json.dumps(data, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Tushare API wrappers (rate-limited + retried)
# ---------------------------------------------------------------------------


def _get_pro(token: str):
    """Initialize and return Tushare pro API handle."""
    import tushare as ts

    ts.set_token(token)
    return ts.pro_api()


@retry_on_exception(max_retries=3, base_delay=2.0, description="tushare:index_weight")
def fetch_csi300_constituents(pro, date: str, *, cache_ttl: int = 86400) -> list[str]:
    """Get current CSI300 constituent stock codes from Tushare.

    Results are cached to disk with a configurable TTL (default: 1 day).
    Constituents change quarterly, so daily caching is safe.

    Args:
        pro: Tushare pro API instance.
        date: Reference date YYYYMMDD.
        cache_ttl: Cache time-to-live in seconds (default 86400 = 1 day).

    Returns:
        Sorted list of ts_codes, e.g. ['000001.SZ', '000002.SZ', ...].
    """
    cache_key = f"csi300_{date}"
    cached = _read_cache("index_weight", cache_key, ttl_seconds=cache_ttl)
    if cached is not None:
        logger.info("Using cached CSI300 constituents (%d stocks, date=%s)", len(cached), date)
        return cached

    df = pro.index_weight(index_code="399300.SZ", start_date=date, end_date=date)
    if df.empty:
        # Fallback: try last month
        fallback = (pd.Timestamp(date) - pd.Timedelta(days=30)).strftime("%Y%m%d")
        df = pro.index_weight(index_code="399300.SZ", start_date=fallback, end_date=date)
    codes = sorted(df["con_code"].unique().tolist())
    logger.info("Fetched %d CSI300 constituents as of %s", len(codes), date)

    # Save to cache
    _write_cache("index_weight", cache_key, codes)
    return codes


def fetch_daily_ohlcv(
    pro,
    stocks: list[str],
    start_date: str,
    end_date: str,
    limiter: RateLimiter | None = None,
    *,
    adj_strategy: str = "boundary_probe",
    parallel_workers: int = 3,
) -> pd.DataFrame:
    """Fetch OHLCV + adj_factor, apply backward price adjustment.

    Optimization: Tushare daily() accepts comma-separated ts_codes.
    With 6000 rows/request and ~200 days, we batch ~30 stocks per call,
    reducing ~300 daily() calls to ~10.

    Args:
        pro: Tushare pro API instance.
        stocks: List of ts_codes to fetch.
        start_date: Start date YYYYMMDD.
        end_date: End date YYYYMMDD.
        limiter: Rate limiter instance (default: 6 calls/sec).
        adj_strategy: Strategy for adj_factor fetch:
            - "boundary_probe" (default): 2 boundary calls + per-stock for
              stocks with corporate actions. Best when few stocks have events.
            - "by_date": Fetch adj_factor by trade_date for all dates
              (~N_dates calls). Always predictable call count.
        parallel_workers: Number of parallel threads for per-stock adj_factor
            fetch (default: 3). Only used with "boundary_probe" strategy.
            Set to 1 to disable parallelism.

    Returns:
        DataFrame with MultiIndex['datetime','instrument'] and columns
        [open, high, low, close, volume, factor] -- prices are BACKWARD-ADJUSTED.
    """
    if limiter is None:
        # Tushare 2000-point accounts: 500 req/min for daily, adj_factor.
        # 6/sec = 360/min, leaves 28% headroom.
        limiter = RateLimiter(calls_per_second=6.0)

    # --- Phase 1: Batch-fetch daily OHLCV ---
    # 6000 rows / ~200 days = 30 stocks per batch (conservative)
    batch_size = max(1, 6000 // max(200, _approx_trading_days(start_date, end_date)))
    daily_frames: list[pd.DataFrame] = []
    for batch_start in range(0, len(stocks), batch_size):
        batch = stocks[batch_start : batch_start + batch_size]
        batch_key = ",".join(batch)
        logger.info(
            "Fetching daily batch %d-%d/%d (%d stocks)",
            batch_start + 1,
            min(batch_start + batch_size, len(stocks)),
            len(stocks),
            len(batch),
        )
        limiter.wait()
        df_batch = _fetch_batch_daily(pro, batch_key, start_date, end_date)
        if not df_batch.empty:
            daily_frames.append(df_batch)

    if not daily_frames:
        raise RuntimeError("No OHLCV data fetched from Tushare. Check token and date range.")

    df_all_daily = pd.concat(daily_frames, ignore_index=True)

    # --- Phase 2: Fetch adj_factor (strategy-dependent) ---
    fetched_codes = sorted(df_all_daily["ts_code"].unique())
    trade_dates_sorted = sorted(df_all_daily["trade_date"].unique())
    first_td, last_td = trade_dates_sorted[0], trade_dates_sorted[-1]

    if adj_strategy == "by_date":
        # Strategy B (#8): Fetch adj_factor by trade_date for ALL dates.
        # Always exactly N_dates calls, regardless of corporate action frequency.
        full_adj_by_code = _fetch_adj_by_date_all(
            pro, trade_dates_sorted, limiter, parallel_workers
        )
        adj_last_map: dict[str, float] = {}
        for code in fetched_codes:
            if code in full_adj_by_code and not full_adj_by_code[code].empty:
                last_row = full_adj_by_code[code].sort_values("trade_date").iloc[-1]
                adj_last_map[code] = float(last_row["adj_factor"])
            else:
                adj_last_map[code] = 1.0
        full_adj = full_adj_by_code
    else:
        # Strategy A (default): Boundary probe + per-stock for changed stocks.
        limiter.wait()
        df_adj_first = _fetch_adj_by_date(pro, first_td)
        limiter.wait()
        df_adj_last = _fetch_adj_by_date(pro, last_td)

        # Build lookup: ts_code -> (first_factor, last_factor)
        adj_first = (
            dict(zip(df_adj_first["ts_code"], df_adj_first["adj_factor"], strict=False))
            if not df_adj_first.empty
            else {}
        )
        adj_last_map = (
            dict(zip(df_adj_last["ts_code"], df_adj_last["adj_factor"], strict=False))
            if not df_adj_last.empty
            else {}
        )

        # Identify stocks needing full adj_factor history (factor changed = corporate action)
        needs_full = []
        for code in fetched_codes:
            f0 = adj_first.get(code)
            f1 = adj_last_map.get(code)
            if f0 is None or f1 is None or abs(f0 - f1) > 1e-6:
                needs_full.append(code)

        logger.info(
            "adj_factor boundary probe: %d/%d stocks have constant factor, " "%d need full history",
            len(fetched_codes) - len(needs_full),
            len(fetched_codes),
            len(needs_full),
        )

        # Fetch full per-stock adj_factor (parallel with ThreadPoolExecutor, #10)
        full_adj = _fetch_adj_parallel(
            pro, needs_full, start_date, end_date, limiter, parallel_workers
        )

    # --- Phase 3: Apply backward price adjustment ---
    frames: list[pd.DataFrame] = []
    for code in fetched_codes:
        df_d = df_all_daily[df_all_daily["ts_code"] == code].copy()
        df_d = df_d.sort_values("trade_date")

        if code in full_adj and not full_adj[code].empty:
            # Stock had corporate action: use full per-date adj_factor
            df_d = df_d.merge(
                full_adj[code][["trade_date", "adj_factor"]], on="trade_date", how="left"
            )
            df_d["adj_factor"] = df_d["adj_factor"].ffill().bfill().fillna(1.0)
        else:
            # Constant factor: use the last date's factor (ratio = 1.0, no price adjustment)
            df_d["adj_factor"] = adj_last_map.get(code, 1.0)

        # Backward price adjustment: price * adj_factor / latest_adj_factor
        latest_factor = df_d["adj_factor"].iloc[-1]
        if latest_factor != 0 and not np.isnan(latest_factor):
            ratio = df_d["adj_factor"] / latest_factor
            for col in _PRICE_FIELDS:
                if col in df_d.columns:
                    df_d[col] = df_d[col] * ratio

        # Normalize to Qlib convention
        instrument = _ts_code_to_qlib(code)
        df_d["datetime"] = pd.to_datetime(df_d["trade_date"])
        df_d["instrument"] = instrument
        df_d = df_d.rename(columns={"vol": "volume", "adj_factor": "factor"})
        frames.append(df_d[["datetime", "instrument"] + _ALL_FIELDS])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.set_index(["datetime", "instrument"]).sort_index()
    logger.info("Fetched OHLCV: %s, date range %s -> %s", combined.shape, start_date, end_date)
    return combined


def _approx_trading_days(start: str, end: str) -> int:
    """Rough estimate of trading days between two YYYYMMDD dates."""
    delta = (pd.Timestamp(end) - pd.Timestamp(start)).days
    return max(1, int(delta / 1.4))


@retry_on_exception(max_retries=3, base_delay=1.0, description="tushare:daily")
def _fetch_batch_daily(pro, ts_codes_csv: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV for multiple comma-separated ts_codes in one call."""
    return pro.daily(ts_code=ts_codes_csv, start_date=start, end_date=end)


@retry_on_exception(max_retries=3, base_delay=1.0, description="tushare:adj_factor")
def _fetch_one_adj(pro, ts_code: str, start: str, end: str) -> pd.DataFrame:
    return pro.adj_factor(ts_code=ts_code, start_date=start, end_date=end)


@retry_on_exception(max_retries=3, base_delay=1.0, description="tushare:adj_factor_date")
def _fetch_adj_by_date(pro, trade_date: str) -> pd.DataFrame:
    """Fetch adj_factor for ALL stocks on a single trade_date (one API call)."""
    return pro.adj_factor(ts_code="", trade_date=trade_date)


def _fetch_adj_parallel(
    pro,
    codes: list[str],
    start_date: str,
    end_date: str,
    limiter: RateLimiter,
    max_workers: int = 3,
) -> dict[str, pd.DataFrame]:
    """Fetch per-stock adj_factor in parallel using ThreadPoolExecutor (#10).

    Uses the thread-safe RateLimiter to stay within Tushare rate limits.
    With max_workers=3 and 6 calls/sec limiter, achieves ~3x speedup
    over sequential fetching.

    Args:
        pro: Tushare pro API instance.
        codes: List of ts_codes needing full adj_factor history.
        start_date: Start date YYYYMMDD.
        end_date: End date YYYYMMDD.
        limiter: Thread-safe rate limiter.
        max_workers: Number of parallel threads (default 3).

    Returns:
        Dict mapping ts_code -> adj_factor DataFrame.
    """
    if not codes:
        return {}

    if max_workers <= 1:
        # Sequential fallback
        result: dict[str, pd.DataFrame] = {}
        for i, code in enumerate(codes, 1):
            if i % 20 == 0 or i == len(codes):
                logger.info("Fetching full adj_factor: %d/%d (%s)", i, len(codes), code)
            limiter.wait()
            result[code] = _fetch_one_adj(pro, code, start_date, end_date)
        return result

    # Parallel fetch
    logger.info(
        "Fetching adj_factor for %d stocks in parallel (workers=%d)",
        len(codes),
        max_workers,
    )
    result = {}
    completed = 0

    def _fetch_single(code: str) -> tuple[str, pd.DataFrame]:
        limiter.wait()
        return code, _fetch_one_adj(pro, code, start_date, end_date)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_single, code): code for code in codes}
        for future in as_completed(futures):
            code, df = future.result()
            result[code] = df
            completed += 1
            if completed % 20 == 0 or completed == len(codes):
                logger.info(
                    "adj_factor parallel progress: %d/%d",
                    completed,
                    len(codes),
                )

    return result


def _fetch_adj_by_date_all(
    pro,
    trade_dates: list[str],
    limiter: RateLimiter,
    max_workers: int = 3,
) -> dict[str, pd.DataFrame]:
    """Fetch adj_factor by trade_date for all dates, then pivot by stock (#8).

    Strategy: for each trading date, call adj_factor(trade_date=date) which
    returns all stocks' factors for that date. Then group by ts_code to get
    per-stock factor timeseries.

    This is always exactly N_dates calls, regardless of corporate action
    frequency. More predictable than boundary_probe when many stocks have
    corporate actions.

    Args:
        pro: Tushare pro API instance.
        trade_dates: Sorted list of trade dates YYYYMMDD.
        limiter: Thread-safe rate limiter.
        max_workers: Number of parallel threads.

    Returns:
        Dict mapping ts_code -> adj_factor DataFrame (columns: trade_date, adj_factor).
    """
    logger.info(
        "Fetching adj_factor by-date for %d dates (workers=%d)",
        len(trade_dates),
        max_workers,
    )

    all_frames: list[pd.DataFrame] = []
    completed = 0

    def _fetch_date(td: str) -> pd.DataFrame:
        limiter.wait()
        return _fetch_adj_by_date(pro, td)

    if max_workers <= 1:
        for td in trade_dates:
            df = _fetch_date(td)
            if not df.empty:
                all_frames.append(df)
            completed += 1
            if completed % 50 == 0 or completed == len(trade_dates):
                logger.info("adj_factor by-date: %d/%d", completed, len(trade_dates))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_date, td): td for td in trade_dates}
            for future in as_completed(futures):
                df = future.result()
                if not df.empty:
                    all_frames.append(df)
                completed += 1
                if completed % 50 == 0 or completed == len(trade_dates):
                    logger.info("adj_factor by-date: %d/%d", completed, len(trade_dates))

    if not all_frames:
        logger.warning("No adj_factor data fetched by-date")
        return {}

    combined = pd.concat(all_frames, ignore_index=True)

    # Pivot: group by ts_code
    result: dict[str, pd.DataFrame] = {}
    for code, group in combined.groupby("ts_code"):
        result[code] = (
            group[["trade_date", "adj_factor"]].sort_values("trade_date").reset_index(drop=True)
        )

    logger.info(
        "adj_factor by-date complete: %d dates -> %d stocks",
        len(trade_dates),
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Qlib binary format -- native writer (no external dump_bin dependency)
# ---------------------------------------------------------------------------


def _ts_code_to_qlib(ts_code: str) -> str:
    """Convert Tushare code '000001.SZ' -> Qlib instrument 'SZ000001'."""
    parts = ts_code.split(".")
    return f"{parts[1]}{parts[0]}" if len(parts) == 2 else ts_code


def _build_calendar_index(dates: list[pd.Timestamp]) -> dict[pd.Timestamp, int]:
    """Map each calendar date to its integer index (0-based)."""
    return {d: i for i, d in enumerate(sorted(dates))}


def dump_qlib_binary_native(df: pd.DataFrame, qlib_dir: Path) -> None:
    """Write Qlib binary data directly -- no dependency on qlib.scripts.dump_bin.

    Qlib binary format (file_storage.py:299-310):
        <instrument>/<field>.day.bin:
          [0:4]  start_index as float32 (calendar offset)
          [4:]   data values as float32[]

    This function writes:
        features/  -- per-instrument per-field .bin files
        calendars/day.txt
        instruments/csi300.txt + instruments/all.txt
    """
    qlib_dir = Path(qlib_dir)

    # Build global calendar
    all_dates = sorted(df.index.get_level_values("datetime").unique())
    cal_idx = _build_calendar_index(all_dates)

    # Write calendars/day.txt
    cal_dir = qlib_dir / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_lines = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in all_dates]
    (cal_dir / "day.txt").write_text("\n".join(cal_lines) + "\n")

    # Write per-instrument binary features
    feat_dir = qlib_dir / "features"
    instruments = df.index.get_level_values("instrument").unique()
    inst_ranges: dict[str, tuple[str, str]] = {}

    for inst in sorted(instruments):
        sub = df.xs(inst, level="instrument").sort_index()
        inst_dir = feat_dir / inst.lower()
        inst_dir.mkdir(parents=True, exist_ok=True)

        inst_dates = sub.index
        start_ts = pd.Timestamp(inst_dates.min())
        end_ts = pd.Timestamp(inst_dates.max())
        inst_ranges[inst] = (start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))

        start_cal_idx = cal_idx[start_ts]

        # For each field, write a .bin file
        for field in _ALL_FIELDS:
            if field not in sub.columns:
                continue

            # Build dense array spanning [start_date, end_date] in calendar space
            values = []
            for d in all_dates:
                if d < start_ts or d > end_ts:
                    continue
                if d in sub.index:
                    values.append(float(sub.loc[d, field]) if d in sub.index else np.nan)
                else:
                    values.append(np.nan)

            # Qlib format: first float32 = start_index, then data values
            bin_path = inst_dir / f"{field}.day.bin"
            data_array = np.array([start_cal_idx] + values, dtype="<f")
            data_array.tofile(str(bin_path))

    # Write instruments/csi300.txt and instruments/all.txt
    inst_dir = qlib_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for inst in sorted(inst_ranges):
        s, e = inst_ranges[inst]
        lines.append(f"{inst}\t{s}\t{e}")
    inst_text = "\n".join(lines) + "\n"
    (inst_dir / "csi300.txt").write_text(inst_text)
    (inst_dir / "all.txt").write_text(inst_text)  # Qlib also needs all.txt

    logger.info(
        "Wrote native Qlib binary: %d stocks, %d days, %d fields -> %s",
        len(instruments),
        len(all_dates),
        len(_ALL_FIELDS),
        qlib_dir,
    )


# ---------------------------------------------------------------------------
# Incremental data detection and merging (#11)
# ---------------------------------------------------------------------------


def _detect_existing_data(qlib_dir: Path) -> pd.Timestamp | None:
    """Detect the last available trading date from existing Qlib binary data.

    Reads calendars/day.txt to find the most recent date. Returns None if
    no existing data or calendar file is missing/empty.
    """
    cal_path = qlib_dir / "calendars" / "day.txt"
    if not cal_path.exists():
        return None

    try:
        lines = cal_path.read_text().strip().split("\n")
        lines = [line.strip() for line in lines if line.strip()]
        if not lines:
            return None
        last_date = pd.Timestamp(lines[-1])
        logger.info(
            "Detected existing Qlib data: %d calendar days, last=%s",
            len(lines),
            last_date.strftime("%Y-%m-%d"),
        )
        return last_date
    except Exception as exc:
        logger.warning("Failed to parse existing calendar: %s", exc)
        return None


def _load_existing_binary(qlib_dir: Path) -> pd.DataFrame | None:
    """Load existing Qlib binary data into a DataFrame.

    Reads all instrument binary files and reconstructs the DataFrame.
    Returns None if data cannot be loaded.
    """
    cal_path = qlib_dir / "calendars" / "day.txt"
    feat_dir = qlib_dir / "features"

    if not cal_path.exists() or not feat_dir.exists():
        return None

    try:
        # Read calendar
        cal_lines = cal_path.read_text().strip().split("\n")
        cal_dates = [pd.Timestamp(line.strip()) for line in cal_lines if line.strip()]
        if not cal_dates:
            return None

        # Read instruments
        inst_dir = qlib_dir / "instruments"
        inst_file = (
            inst_dir / "csi300.txt" if (inst_dir / "csi300.txt").exists() else inst_dir / "all.txt"
        )
        if not inst_file.exists():
            return None

        frames = []
        for line in inst_file.read_text().strip().split("\n"):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            inst = parts[0]
            inst_feat_dir = feat_dir / inst.lower()
            if not inst_feat_dir.exists():
                continue

            # Read one field to determine date range
            close_bin = inst_feat_dir / "close.day.bin"
            if not close_bin.exists():
                continue

            data = np.fromfile(str(close_bin), dtype="<f")
            if len(data) < 2:
                continue
            start_idx = int(data[0])
            n_values = len(data) - 1

            # Reconstruct dates for this instrument
            inst_dates = cal_dates[start_idx : start_idx + n_values]
            if not inst_dates:
                continue

            # Read all fields
            row_data = {"datetime": inst_dates, "instrument": [inst] * len(inst_dates)}
            for field in _ALL_FIELDS:
                bin_path = inst_feat_dir / f"{field}.day.bin"
                if bin_path.exists():
                    field_data = np.fromfile(str(bin_path), dtype="<f")
                    row_data[field] = field_data[1:].tolist()  # Skip header
                else:
                    row_data[field] = [np.nan] * len(inst_dates)

            frames.append(pd.DataFrame(row_data))

        if not frames:
            return None

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.set_index(["datetime", "instrument"]).sort_index()
        logger.info(
            "Loaded existing binary data: %s, %d instruments, %d dates",
            combined.shape,
            combined.index.get_level_values("instrument").nunique(),
            combined.index.get_level_values("datetime").nunique(),
        )
        return combined

    except Exception as exc:
        logger.warning("Failed to load existing binary data: %s", exc)
        return None


def _merge_incremental_data(existing: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    """Merge existing binary data with newly fetched incremental data.

    New data takes precedence for overlapping dates (handles re-adjustments).
    """
    # Remove overlapping dates from existing data (new data has correct adj)
    new_dates = new_data.index.get_level_values("datetime").unique()
    existing_filtered = existing[~existing.index.get_level_values("datetime").isin(new_dates)]

    merged = pd.concat([existing_filtered, new_data]).sort_index()

    logger.info(
        "Merged incremental data: existing=%d rows, new=%d rows, merged=%d rows",
        len(existing),
        len(new_data),
        len(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_minimal_qlib_data(
    token: str,
    n_days: int = 200,
    qlib_dir: str | None = None,
    ref_date: str | None = None,
    force_full: bool = False,
) -> Path:
    """Main entry: Tushare -> minimal Qlib binary data (~3 MB).

    Supports incremental mode: if existing Qlib binary data is detected,
    only fetches new dates since the last available date. Falls back to
    full fetch if no existing data or if force_full=True.

    Args:
        token: Tushare pro API token.
        n_days: Number of trading days to fetch (default 200).
        qlib_dir: Target Qlib data directory. Defaults to ~/.qlib/qlib_data/cn_data.
        ref_date: Reference date YYYY-MM-DD for constituent lookup.
                  Defaults to today.
        force_full: If True, always do a full fetch (ignore existing data).

    Returns:
        Path to the Qlib binary data directory.
    """
    qlib_path = Path(qlib_dir or os.path.expanduser("~/.qlib/qlib_data/cn_data"))
    pro = _get_pro(token)

    # Determine date range
    if ref_date is None:
        ref_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    end_ts = pd.Timestamp(ref_date)
    end_str = end_ts.strftime("%Y%m%d")

    # Check for existing data (incremental mode)
    existing_last_date = None if force_full else _detect_existing_data(qlib_path)

    if existing_last_date is not None and existing_last_date >= end_ts:
        logger.info(
            "Existing data already covers target date %s (last=%s). Skipping fetch.",
            end_ts.strftime("%Y-%m-%d"),
            existing_last_date.strftime("%Y-%m-%d"),
        )
        return qlib_path

    if existing_last_date is not None:
        # Incremental mode: fetch only new dates
        # Start from 1 day before last existing date (overlap for adj_factor continuity)
        incr_start_ts = existing_last_date - pd.Timedelta(days=2)
        incr_start_str = incr_start_ts.strftime("%Y%m%d")
        gap_days = (end_ts - existing_last_date).days

        # If gap is too large (> n_days * 1.5 calendar days), fall back to full fetch
        if gap_days > n_days * 1.5:
            logger.warning(
                "Gap too large (%d calendar days > %d threshold). Falling back to full fetch.",
                gap_days,
                int(n_days * 1.5),
            )
            existing_last_date = None  # Force full fetch
        else:
            logger.info(
                "Incremental mode: fetching %d new calendar days (%s -> %s)",
                gap_days,
                incr_start_str,
                end_str,
            )

    if existing_last_date is not None:
        # --- Incremental fetch ---
        # 1. Fetch CSI300 constituents
        stocks = fetch_csi300_constituents(pro, end_str)
        if not stocks:
            raise RuntimeError("No CSI300 constituents returned from Tushare")

        # 2. Fetch new OHLCV data (small date range)
        limiter = RateLimiter(calls_per_second=6.0)
        new_ohlcv = fetch_daily_ohlcv(pro, stocks, incr_start_str, end_str, limiter)

        # 3. Load existing data, merge, and rewrite
        existing_data = _load_existing_binary(qlib_path)
        if existing_data is not None:
            merged = _merge_incremental_data(existing_data, new_ohlcv)
            # Trim to keep only the last n_days worth of calendar days
            all_dates = sorted(merged.index.get_level_values("datetime").unique())
            if len(all_dates) > n_days:
                cutoff_date = all_dates[-n_days]
                merged = merged[merged.index.get_level_values("datetime") >= cutoff_date]
                logger.info(
                    "Trimmed to %d calendar days (cutoff=%s)",
                    n_days,
                    cutoff_date.strftime("%Y-%m-%d"),
                )
            dump_qlib_binary_native(merged, qlib_path)
        else:
            # Existing data couldn't be loaded, just write new data
            dump_qlib_binary_native(new_ohlcv, qlib_path)

        n_dates = new_ohlcv.index.get_level_values("datetime").nunique()
        n_stocks = new_ohlcv.index.get_level_values("instrument").nunique()
        logger.info(
            "Incremental update done: +%d new days, %d stocks -> %s",
            n_dates,
            n_stocks,
            qlib_path,
        )
    else:
        # --- Full fetch (original behavior) ---
        # Approximate: 1 trading day ~ 1.4 calendar days
        start_ts = end_ts - pd.Timedelta(days=int(n_days * 1.5))
        start_str = start_ts.strftime("%Y%m%d")

        # 1. Fetch CSI300 constituents
        stocks = fetch_csi300_constituents(pro, end_str)
        if not stocks:
            raise RuntimeError("No CSI300 constituents returned from Tushare")

        # 2. Fetch OHLCV + adj_factor (with backward price adjustment)
        limiter = RateLimiter(calls_per_second=6.0)
        ohlcv = fetch_daily_ohlcv(pro, stocks, start_str, end_str, limiter)

        # 3. Write native Qlib binary (no dependency on qlib.scripts.dump_bin)
        dump_qlib_binary_native(ohlcv, qlib_path)

        n_dates = ohlcv.index.get_level_values("datetime").nunique()
        n_stocks = ohlcv.index.get_level_values("instrument").nunique()
        logger.info(
            "Minimal Qlib data ready: %d days x %d stocks -> %s",
            n_dates,
            n_stocks,
            qlib_path,
        )

    return qlib_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli_main() -> int:
    """Command-line wrapper for build_minimal_qlib_data."""
    import argparse

    ap = argparse.ArgumentParser(description="Fetch CSI300 data from Tushare -> Qlib binary")
    ap.add_argument("--token", default=None, help="Tushare API token (default: $TUSHARE env var)")
    ap.add_argument("--n-days", type=int, default=200, help="Trading days to fetch (default 200)")
    ap.add_argument(
        "--qlib-dir",
        default=os.path.expanduser("~/.qlib/qlib_data/cn_data"),
        help="Target Qlib data directory",
    )
    ap.add_argument("--ref-date", default=None, help="Reference date YYYY-MM-DD (default: today)")
    ap.add_argument("--force-full", action="store_true", help="Force full fetch (ignore existing)")
    args = ap.parse_args()

    token = args.token or os.getenv("TUSHARE") or os.getenv("TS_TOKEN")
    if not token:
        print(
            "[ERROR] Tushare token required. Pass --token or set TUSHARE env var.",
            file=__import__("sys").stderr,
        )
        return 1

    build_minimal_qlib_data(
        token=token,
        n_days=args.n_days,
        qlib_dir=args.qlib_dir,
        ref_date=args.ref_date,
        force_full=args.force_full,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
