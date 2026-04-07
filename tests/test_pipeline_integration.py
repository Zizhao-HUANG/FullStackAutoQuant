"""Integration test: End-to-End Lite Pipeline with mock Tushare API.

Covers (#14):
  1. Uses a mock Tushare API (synthetic responses) to avoid real API calls
  2. Runs pipeline steps: fetch -> binary write -> verify
  3. Asserts output has expected instruments with non-degenerate data
  4. Validates binary file format (header + float32 values)

This test is CI-runnable (no network, no credentials required).
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fullstackautoquant.data.tushare_provider import (
    _ALL_FIELDS,
    dump_qlib_binary_native,
    fetch_csi300_constituents,
    fetch_daily_ohlcv,
)
from fullstackautoquant.resilience import RateLimiter

# ---------------------------------------------------------------------------
# Mock data generators
# ---------------------------------------------------------------------------


def _generate_mock_constituents(n: int = 10) -> list[str]:
    """Generate N mock CSI300 constituent codes."""
    codes = []
    for i in range(n):
        suffix = "SZ" if i < n // 2 else "SH"
        num = f"{i + 1:06d}"
        codes.append(f"{num}.{suffix}")
    return sorted(codes)


def _generate_mock_daily(ts_codes: list[str], start: str, end: str) -> pd.DataFrame:
    """Generate synthetic daily OHLCV data for given stocks and date range."""
    dates = pd.bdate_range(start, end)
    rows = []
    for code in ts_codes:
        np.random.seed(hash(code) % 2**31)
        base_price = np.random.uniform(10, 100)
        for _i, d in enumerate(dates):
            # Random walk for realistic-ish prices
            price = base_price + np.random.randn() * 2
            base_price = max(1.0, price)
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": d.strftime("%Y%m%d"),
                    "open": round(base_price * 0.99, 2),
                    "high": round(base_price * 1.02, 2),
                    "low": round(base_price * 0.97, 2),
                    "close": round(base_price, 2),
                    "vol": round(np.random.uniform(10000, 500000), 0),
                    "amount": round(np.random.uniform(1e6, 1e8), 0),
                }
            )
    return pd.DataFrame(rows)


def _generate_mock_adj_factor(ts_codes: list[str], start: str, end: str) -> pd.DataFrame:
    """Generate synthetic adj_factor data (mostly constant, some with splits)."""
    dates = pd.bdate_range(start, end)
    rows = []
    for code in ts_codes:
        # ~20% of stocks have a split/dividend mid-window
        has_event = hash(code) % 5 == 0
        for i, d in enumerate(dates):
            factor = 1.0
            if has_event and i > len(dates) // 2:
                factor = 1.1  # Simulate dividend
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": d.strftime("%Y%m%d"),
                    "adj_factor": factor,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pro():
    """Create a mock Tushare pro API that returns synthetic data."""
    pro = MagicMock()

    mock_codes = _generate_mock_constituents(10)

    # Mock index_weight
    def mock_index_weight(**kwargs):
        return pd.DataFrame({"con_code": mock_codes})

    pro.index_weight = MagicMock(side_effect=mock_index_weight)

    # Mock daily
    def mock_daily(**kwargs):
        ts_code = kwargs.get("ts_code", "")
        start = kwargs.get("start_date", "20250101")
        end = kwargs.get("end_date", "20250301")
        codes = [c.strip() for c in ts_code.split(",") if c.strip()]
        return _generate_mock_daily(codes, start, end)

    pro.daily = MagicMock(side_effect=mock_daily)

    # Mock adj_factor (by ts_code)
    def mock_adj_factor(**kwargs):
        ts_code = kwargs.get("ts_code", "")
        trade_date = kwargs.get("trade_date", "")
        start = kwargs.get("start_date", "20250101")
        end = kwargs.get("end_date", "20250301")

        if trade_date:
            # Date-based query: return all stocks for that date
            return _generate_mock_adj_factor(mock_codes, trade_date, trade_date)
        elif ts_code:
            return _generate_mock_adj_factor([ts_code], start, end)
        return pd.DataFrame()

    pro.adj_factor = MagicMock(side_effect=mock_adj_factor)

    return pro


@pytest.fixture
def fast_limiter():
    """Rate limiter with no delay for testing."""
    return RateLimiter(calls_per_second=1000.0)


# ---------------------------------------------------------------------------
# #14: Integration Tests
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """Full pipeline integration test with mock Tushare API."""

    def test_fetch_and_write_binary(self, mock_pro, fast_limiter, tmp_path):
        """Step 1-3: Fetch mock data, write Qlib binary, verify format."""
        # Step 1: Fetch constituents
        constituents = fetch_csi300_constituents(mock_pro, "20250301", cache_ttl=0)
        assert len(constituents) == 10

        # Step 2: Fetch OHLCV with backward adjustment
        ohlcv = fetch_daily_ohlcv(mock_pro, constituents, "20250101", "20250301", fast_limiter)
        assert not ohlcv.empty
        assert ohlcv.index.names == ["datetime", "instrument"]

        n_instruments = ohlcv.index.get_level_values("instrument").nunique()
        assert n_instruments == 10

        # Step 3: Write to Qlib binary
        qlib_dir = tmp_path / "qlib_data"
        dump_qlib_binary_native(ohlcv, qlib_dir)

        # Verify directory structure
        assert (qlib_dir / "calendars" / "day.txt").exists()
        assert (qlib_dir / "instruments" / "csi300.txt").exists()
        assert (qlib_dir / "instruments" / "all.txt").exists()

        # Verify binary files exist for each instrument
        inst_text = (qlib_dir / "instruments" / "csi300.txt").read_text().strip()
        inst_lines = inst_text.split("\n")
        assert len(inst_lines) == 10

        for line in inst_lines:
            parts = line.split("\t")
            inst = parts[0].lower()
            for field in _ALL_FIELDS:
                bin_path = qlib_dir / "features" / inst / f"{field}.day.bin"
                assert bin_path.exists(), f"Missing {bin_path}"

    def test_binary_format_integrity(self, mock_pro, fast_limiter, tmp_path):
        """Verify each written binary file has valid Qlib format."""
        constituents = fetch_csi300_constituents(mock_pro, "20250301", cache_ttl=0)
        ohlcv = fetch_daily_ohlcv(mock_pro, constituents, "20250101", "20250301", fast_limiter)

        qlib_dir = tmp_path / "qlib_data"
        dump_qlib_binary_native(ohlcv, qlib_dir)

        # Read calendar to know expected number of dates
        cal_lines = (qlib_dir / "calendars" / "day.txt").read_text().strip().split("\n")
        n_dates = len(cal_lines)
        assert n_dates > 0

        # Check each binary file
        for field in _ALL_FIELDS:
            inst_dirs = list((qlib_dir / "features").iterdir())
            assert len(inst_dirs) > 0

            for inst_dir in inst_dirs:
                if not inst_dir.is_dir():
                    continue
                bin_path = inst_dir / f"{field}.day.bin"
                if not bin_path.exists():
                    continue

                raw_bytes = bin_path.read_bytes()

                # Must have at least 8 bytes (1 header + 1 value)
                assert len(raw_bytes) >= 8, f"File too small: {bin_path}"

                # Total bytes must be multiple of 4 (all float32)
                assert len(raw_bytes) % 4 == 0, f"Not aligned to float32: {bin_path}"

                # First 4 bytes = start_index as float32
                start_idx = struct.unpack("<f", raw_bytes[:4])[0]
                assert start_idx >= 0, f"Negative start_index: {bin_path}"
                assert start_idx == int(start_idx), f"Non-integer start_index: {bin_path}"

                # Data values: should be finite floats (or NaN for missing)
                data = np.frombuffer(raw_bytes[4:], dtype="<f")
                # At least some values should be non-NaN
                assert np.any(~np.isnan(data)), f"All NaN in {bin_path}"

    def test_non_degenerate_ohlcv(self, mock_pro, fast_limiter, tmp_path):
        """Verify fetched data is non-degenerate: varied prices, reasonable volumes."""
        constituents = fetch_csi300_constituents(mock_pro, "20250301", cache_ttl=0)
        ohlcv = fetch_daily_ohlcv(mock_pro, constituents, "20250101", "20250301", fast_limiter)

        # Check non-degeneracy
        for col in ["open", "high", "low", "close"]:
            assert ohlcv[col].std() > 0.01, f"{col} has near-zero std"
            assert ohlcv[col].nunique() > 10, f"{col} has too few unique values"

        # Volume should be positive
        assert (ohlcv["volume"] > 0).all(), "Volume must be positive"

        # OHLC relationship: high >= low for each row
        assert (ohlcv["high"] >= ohlcv["low"]).all(), "High must >= Low"

    def test_incremental_fetch_simulation(self, mock_pro, fast_limiter, tmp_path):
        """Simulate incremental fetch: write initial data, then append new dates."""
        constituents = fetch_csi300_constituents(mock_pro, "20250301", cache_ttl=0)

        # First fetch: Jan-Feb
        ohlcv1 = fetch_daily_ohlcv(mock_pro, constituents, "20250101", "20250228", fast_limiter)
        qlib_dir = tmp_path / "qlib_data"
        dump_qlib_binary_native(ohlcv1, qlib_dir)

        cal1 = (qlib_dir / "calendars" / "day.txt").read_text().strip().split("\n")
        n_dates_1 = len(cal1)

        # Second fetch: Feb-Mar (overlapping Feb dates)
        ohlcv2 = fetch_daily_ohlcv(mock_pro, constituents, "20250220", "20250315", fast_limiter)

        # Import merge function
        from fullstackautoquant.data.tushare_provider import (
            _load_existing_binary,
            _merge_incremental_data,
        )

        existing = _load_existing_binary(qlib_dir)
        assert existing is not None

        merged = _merge_incremental_data(existing, ohlcv2)
        dump_qlib_binary_native(merged, qlib_dir)

        cal2 = (qlib_dir / "calendars" / "day.txt").read_text().strip().split("\n")
        n_dates_2 = len(cal2)

        # Should have more dates after merge
        assert n_dates_2 >= n_dates_1, "Merged data should have at least as many dates"


class TestCalendarConsistency:
    """Verify calendar and binary data are consistent."""

    def test_calendar_dates_match_binary_data(self, mock_pro, fast_limiter, tmp_path):
        """Every date in the binary data must appear in the calendar."""
        constituents = fetch_csi300_constituents(mock_pro, "20250301", cache_ttl=0)
        ohlcv = fetch_daily_ohlcv(mock_pro, constituents, "20250101", "20250301", fast_limiter)

        qlib_dir = tmp_path / "qlib_data"
        dump_qlib_binary_native(ohlcv, qlib_dir)

        # Parse calendar dates
        cal_lines = (qlib_dir / "calendars" / "day.txt").read_text().strip().split("\n")
        cal_dates = set(pd.Timestamp(line.strip()) for line in cal_lines)

        # All dates in original data should be in calendar
        data_dates = set(ohlcv.index.get_level_values("datetime").unique())
        missing = data_dates - cal_dates
        assert len(missing) == 0, f"Dates in data but not in calendar: {missing}"

    def test_instrument_count_consistent(self, mock_pro, fast_limiter, tmp_path):
        """Number of instruments in csi300.txt matches feature directories."""
        constituents = fetch_csi300_constituents(mock_pro, "20250301", cache_ttl=0)
        ohlcv = fetch_daily_ohlcv(mock_pro, constituents, "20250101", "20250301", fast_limiter)

        qlib_dir = tmp_path / "qlib_data"
        dump_qlib_binary_native(ohlcv, qlib_dir)

        # Count from instruments file
        inst_lines = (qlib_dir / "instruments" / "csi300.txt").read_text().strip().split("\n")
        n_from_file = len(inst_lines)

        # Count from feature directories
        feat_dirs = [d for d in (qlib_dir / "features").iterdir() if d.is_dir()]
        n_from_dirs = len(feat_dirs)

        assert (
            n_from_file == n_from_dirs
        ), f"Instrument count mismatch: file={n_from_file}, dirs={n_from_dirs}"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
