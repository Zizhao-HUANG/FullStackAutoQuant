"""Tests for tushare_provider.py -- core logic validation.

Covers:
  - ts_code <-> Qlib instrument conversion
  - Approximate trading-day estimation
  - Backward price adjustment math
  - Qlib binary format (header + float32)
  - Boundary probe logic for adj_factor optimization
  - Binary writer verification (#4)
  - Incremental data detection (#11)
  - Disk cache for API responses (#9)
"""

from __future__ import annotations

import json
import struct

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fullstackautoquant.data.tushare_provider import (
    _approx_trading_days,
    _detect_existing_data,
    _load_existing_binary,
    _merge_incremental_data,
    _read_cache,
    _ts_code_to_qlib,
    _write_cache,
    dump_qlib_binary_native,
)

# ---------------------------------------------------------------------------
# #12a: _ts_code_to_qlib
# ---------------------------------------------------------------------------


class TestTsCodeToQlib:
    def test_sz_code(self):
        assert _ts_code_to_qlib("000001.SZ") == "SZ000001"

    def test_sh_code(self):
        assert _ts_code_to_qlib("600000.SH") == "SH600000"

    def test_no_dot_passthrough(self):
        assert _ts_code_to_qlib("UNKNOWN") == "UNKNOWN"

    def test_bj_code(self):
        assert _ts_code_to_qlib("430047.BJ") == "BJ430047"


# ---------------------------------------------------------------------------
# #12b: _approx_trading_days
# ---------------------------------------------------------------------------


class TestApproxTradingDays:
    def test_one_year(self):
        # ~365 calendar days / 1.4 ~ 260
        result = _approx_trading_days("20250101", "20251231")
        assert 240 <= result <= 280

    def test_one_month(self):
        result = _approx_trading_days("20250101", "20250131")
        assert 15 <= result <= 25

    def test_same_day(self):
        result = _approx_trading_days("20250101", "20250101")
        assert result >= 1  # must not be zero


# ---------------------------------------------------------------------------
# #12c: Backward price adjustment math
# ---------------------------------------------------------------------------


class TestBackwardAdjustment:
    """Verify: adjusted = raw * adj_factor[date] / adj_factor[latest_date]"""

    def test_no_split_trivial(self):
        """If adj_factor is constant, prices are unchanged."""
        raw_close = np.array([10.0, 11.0, 12.0])
        adj_factor = np.array([1.5, 1.5, 1.5])
        latest = adj_factor[-1]
        adjusted = raw_close * adj_factor / latest
        np.testing.assert_array_almost_equal(adjusted, raw_close)

    def test_2_for_1_split(self):
        """2-for-1 split: adj_factor halves at the split point.
        Before split: raw=20, factor=0.5 -> adjusted = 20*0.5/1.0 = 10
        After split:  raw=10, factor=1.0 -> adjusted = 10*1.0/1.0 = 10
        The adjusted series should be continuous."""
        raw_close = np.array([20.0, 20.0, 10.0, 10.0])
        adj_factor = np.array([0.5, 0.5, 1.0, 1.0])
        latest = adj_factor[-1]
        adjusted = raw_close * adj_factor / latest
        np.testing.assert_array_almost_equal(adjusted, [10.0, 10.0, 10.0, 10.0])

    def test_dividend_ex_date(self):
        """Cash dividend increases adj_factor slightly on ex-date.
        Pre-dividend:  raw=50, factor=1.0 -> adjusted = 50*1.0/1.02 ~ 49.02
        Post-dividend: raw=49, factor=1.02 -> adjusted = 49*1.02/1.02 = 49.0
        """
        raw = np.array([50.0, 49.0])
        adj = np.array([1.0, 1.02])
        latest = adj[-1]
        result = raw * adj / latest
        assert abs(result[0] - 49.0196) < 0.01
        assert abs(result[1] - 49.0) < 0.001

    def test_preserves_returns(self):
        """Backward adjustment preserves returns between non-event days."""
        raw = np.array([100.0, 105.0, 102.0])
        adj = np.array([2.0, 2.0, 2.0])  # constant factor
        latest = adj[-1]
        adjusted = raw * adj / latest
        raw_ret = np.diff(raw) / raw[:-1]
        adj_ret = np.diff(adjusted) / adjusted[:-1]
        np.testing.assert_array_almost_equal(raw_ret, adj_ret)


# ---------------------------------------------------------------------------
# #12d: Qlib binary format
# ---------------------------------------------------------------------------


class TestQlibBinaryFormat:
    def test_basic_write_read(self, tmp_path):
        """Write binary data and verify format: float32 header + float32 values."""
        dates = pd.date_range("2025-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0, 13.0, 14.0],
                "high": [10.5, 11.5, 12.5, 13.5, 14.5],
                "low": [9.5, 10.5, 11.5, 12.5, 13.5],
                "close": [10.2, 11.2, 12.2, 13.2, 14.2],
                "volume": [1000, 2000, 3000, 4000, 5000],
                "factor": [1.0, 1.0, 1.0, 1.0, 1.0],
            },
            index=pd.MultiIndex.from_arrays(
                [dates, ["SH600000"] * 5], names=["datetime", "instrument"]
            ),
        )

        out_dir = tmp_path / "qlib_data"
        dump_qlib_binary_native(df, out_dir)

        # Verify calendar
        cal = (out_dir / "calendars" / "day.txt").read_text().strip().split("\n")
        assert len(cal) == 5

        # Verify binary file format: first float32 = start_index (0), then 5 values
        bin_path = out_dir / "features" / "sh600000" / "close.day.bin"
        assert bin_path.exists()
        data = np.fromfile(str(bin_path), dtype="<f")
        assert len(data) == 6  # 1 header + 5 values
        assert data[0] == 0.0  # start_index
        np.testing.assert_array_almost_equal(data[1:], [10.2, 11.2, 12.2, 13.2, 14.2], decimal=1)

    def test_multi_instrument(self, tmp_path):
        """Verify multiple instruments produce separate directories."""
        dates = pd.date_range("2025-01-01", periods=3, freq="B")
        rows = []
        for inst in ["SH600000", "SZ000001"]:
            for i, d in enumerate(dates):
                rows.append(
                    {
                        "datetime": d,
                        "instrument": inst,
                        "open": 10.0 + i,
                        "high": 11.0 + i,
                        "low": 9.0 + i,
                        "close": 10.5 + i,
                        "volume": 1000 * (i + 1),
                        "factor": 1.0,
                    }
                )
        df = pd.DataFrame(rows).set_index(["datetime", "instrument"])
        out_dir = tmp_path / "qlib_data"
        dump_qlib_binary_native(df, out_dir)

        assert (out_dir / "features" / "sh600000" / "close.day.bin").exists()
        assert (out_dir / "features" / "sz000001" / "close.day.bin").exists()

        # Instruments file should have 2 entries
        inst_txt = (out_dir / "instruments" / "csi300.txt").read_text().strip()
        assert len(inst_txt.split("\n")) == 2


# ---------------------------------------------------------------------------
# #4: Binary Writer Verification (bitwise format correctness)
# ---------------------------------------------------------------------------


class TestQlibBinaryVerification:
    """Verify our native binary writer produces Qlib-compatible output."""

    def test_header_is_float32_start_index(self, tmp_path):
        """The first 4 bytes of each .bin file must be the start calendar index as float32."""
        dates = pd.date_range("2025-03-01", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "open": range(10, 20),
                "high": range(11, 21),
                "low": range(9, 19),
                "close": np.linspace(10.0, 19.0, 10),
                "volume": [1000] * 10,
                "factor": [1.0] * 10,
            },
            index=pd.MultiIndex.from_arrays(
                [dates, ["SH600000"] * 10], names=["datetime", "instrument"]
            ),
        )

        out_dir = tmp_path / "qlib_verify"
        dump_qlib_binary_native(df, out_dir)

        for field in ["open", "high", "low", "close", "volume", "factor"]:
            bin_path = out_dir / "features" / "sh600000" / f"{field}.day.bin"
            assert bin_path.exists(), f"Missing {field}.day.bin"

            raw = bin_path.read_bytes()
            # First 4 bytes = start_index as float32
            start_idx = struct.unpack("<f", raw[:4])[0]
            assert start_idx == 0.0, f"{field}: start_index should be 0, got {start_idx}"

            # Remaining bytes should be exactly 10 float32 values
            n_values = (len(raw) - 4) // 4
            assert n_values == 10, f"{field}: expected 10 values, got {n_values}"

    def test_values_are_little_endian_float32(self, tmp_path):
        """All data values must be stored as little-endian float32."""
        dates = pd.date_range("2025-01-01", periods=3, freq="B")
        close_values = [100.5, 200.25, 300.75]
        df = pd.DataFrame(
            {
                "open": close_values,
                "high": close_values,
                "low": close_values,
                "close": close_values,
                "volume": [1000, 2000, 3000],
                "factor": [1.0, 1.0, 1.0],
            },
            index=pd.MultiIndex.from_arrays(
                [dates, ["SZ000001"] * 3], names=["datetime", "instrument"]
            ),
        )

        out_dir = tmp_path / "qlib_endian"
        dump_qlib_binary_native(df, out_dir)

        bin_path = out_dir / "features" / "sz000001" / "close.day.bin"
        data = np.fromfile(str(bin_path), dtype="<f")

        # Verify values match (accounting for float32 precision)
        for i, expected in enumerate(close_values):
            actual = float(data[i + 1])  # +1 to skip header
            assert abs(actual - expected) < 0.01, f"Value {i}: expected {expected}, got {actual}"

    def test_calendar_file_format(self, tmp_path):
        """Calendar day.txt must have one YYYY-MM-DD per line."""
        dates = pd.date_range("2025-06-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "open": [1.0] * 5,
                "high": [1.0] * 5,
                "low": [1.0] * 5,
                "close": [1.0] * 5,
                "volume": [100] * 5,
                "factor": [1.0] * 5,
            },
            index=pd.MultiIndex.from_arrays(
                [dates, ["SH600000"] * 5], names=["datetime", "instrument"]
            ),
        )

        out_dir = tmp_path / "qlib_cal"
        dump_qlib_binary_native(df, out_dir)

        cal_text = (out_dir / "calendars" / "day.txt").read_text()
        lines = [line for line in cal_text.strip().split("\n") if line.strip()]
        assert len(lines) == 5, f"Expected 5 calendar lines, got {len(lines)}"

        # Each line must be valid YYYY-MM-DD format
        for line in lines:
            ts = pd.Timestamp(line.strip())
            assert ts.strftime("%Y-%m-%d") == line.strip()

    def test_instrument_file_format(self, tmp_path):
        """Instrument files must have TAB-separated: INSTRUMENT\tSTART\tEND."""
        dates = pd.date_range("2025-01-01", periods=5, freq="B")
        rows = []
        for inst in ["SH600000", "SZ000001", "SZ000002"]:
            for _i, d in enumerate(dates):
                rows.append(
                    {
                        "datetime": d,
                        "instrument": inst,
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.0,
                        "close": 10.5,
                        "volume": 1000,
                        "factor": 1.0,
                    }
                )
        df = pd.DataFrame(rows).set_index(["datetime", "instrument"])

        out_dir = tmp_path / "qlib_inst"
        dump_qlib_binary_native(df, out_dir)

        for fname in ["csi300.txt", "all.txt"]:
            inst_text = (out_dir / "instruments" / fname).read_text()
            lines = [line for line in inst_text.strip().split("\n") if line.strip()]
            assert len(lines) == 3, f"{fname}: expected 3 instruments, got {len(lines)}"
            for line in lines:
                parts = line.split("\t")
                assert len(parts) == 3, f"Expected 3 TAB-separated fields, got {len(parts)}: {line}"
                # Validate date format
                pd.Timestamp(parts[1])
                pd.Timestamp(parts[2])

    def test_roundtrip_write_read(self, tmp_path):
        """Write binary data, read it back, verify values match original."""
        dates = pd.date_range("2025-01-01", periods=7, freq="B")
        np.random.seed(42)
        orig_df = pd.DataFrame(
            {
                "open": np.random.uniform(10, 50, 7),
                "high": np.random.uniform(50, 100, 7),
                "low": np.random.uniform(1, 10, 7),
                "close": np.random.uniform(10, 50, 7),
                "volume": np.random.uniform(1000, 5000, 7),
                "factor": [1.0] * 7,
            },
            index=pd.MultiIndex.from_arrays(
                [dates, ["SH600000"] * 7], names=["datetime", "instrument"]
            ),
        )

        out_dir = tmp_path / "qlib_roundtrip"
        dump_qlib_binary_native(orig_df, out_dir)

        # Read back
        loaded = _load_existing_binary(out_dir)
        assert loaded is not None, "Failed to load binary data back"

        # Verify shape
        assert loaded.shape == orig_df.shape, f"Shape mismatch: {loaded.shape} vs {orig_df.shape}"

        # Verify values (with float32 precision tolerance)
        for col in ["open", "high", "low", "close", "volume", "factor"]:
            np.testing.assert_array_almost_equal(
                loaded[col].values.astype(np.float32),
                orig_df[col].values.astype(np.float32),
                decimal=3,
                err_msg=f"Column {col} values mismatch after roundtrip",
            )


# ---------------------------------------------------------------------------
# #12e: Boundary probe logic
# ---------------------------------------------------------------------------


class TestBoundaryProbeLogic:
    """Test the adj_factor optimization: detect constant vs changed factors."""

    def test_constant_factor_detection(self):
        """Stock with identical first/last adj_factor needs no full fetch."""
        adj_first = {"000001.SZ": 15.0, "600000.SH": 22.5}
        adj_last = {"000001.SZ": 15.0, "600000.SH": 22.5}
        needs_full = []
        for code in ["000001.SZ", "600000.SH"]:
            f0 = adj_first.get(code)
            f1 = adj_last.get(code)
            if f0 is None or f1 is None or abs(f0 - f1) > 1e-6:
                needs_full.append(code)
        assert len(needs_full) == 0

    def test_changed_factor_detection(self):
        """Stock with split between boundaries needs full fetch."""
        adj_first = {"000001.SZ": 15.0}
        adj_last = {"000001.SZ": 30.0}  # 2:1 split
        needs_full = []
        for code in ["000001.SZ"]:
            f0 = adj_first.get(code)
            f1 = adj_last.get(code)
            if f0 is None or f1 is None or abs(f0 - f1) > 1e-6:
                needs_full.append(code)
        assert needs_full == ["000001.SZ"]

    def test_missing_from_boundary(self):
        """Stock missing from boundary date (e.g., newly listed) needs full fetch."""
        adj_first = {}  # not in first date
        adj_last = {"000001.SZ": 15.0}
        needs_full = []
        for code in ["000001.SZ"]:
            f0 = adj_first.get(code)
            f1 = adj_last.get(code)
            if f0 is None or f1 is None or abs(f0 - f1) > 1e-6:
                needs_full.append(code)
        assert needs_full == ["000001.SZ"]


# ---------------------------------------------------------------------------
# #11: Incremental data detection and merging
# ---------------------------------------------------------------------------


class TestIncrementalDataDetection:
    """Test incremental data fetch logic."""

    def test_detect_no_existing_data(self, tmp_path):
        """No calendar file -> returns None."""
        result = _detect_existing_data(tmp_path / "nonexistent")
        assert result is None

    def test_detect_empty_calendar(self, tmp_path):
        """Empty calendar file -> returns None."""
        cal_dir = tmp_path / "calendars"
        cal_dir.mkdir(parents=True)
        (cal_dir / "day.txt").write_text("")
        assert _detect_existing_data(tmp_path) is None

    def test_detect_valid_calendar(self, tmp_path):
        """Valid calendar -> returns last date."""
        cal_dir = tmp_path / "calendars"
        cal_dir.mkdir(parents=True)
        (cal_dir / "day.txt").write_text("2025-01-01\n2025-01-02\n2025-01-03\n")
        result = _detect_existing_data(tmp_path)
        assert result == pd.Timestamp("2025-01-03")

    def test_load_and_roundtrip(self, tmp_path):
        """Write Qlib binary, load it back, verify structure."""
        dates = pd.date_range("2025-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0, 13.0, 14.0],
                "high": [10.5, 11.5, 12.5, 13.5, 14.5],
                "low": [9.5, 10.5, 11.5, 12.5, 13.5],
                "close": [10.2, 11.2, 12.2, 13.2, 14.2],
                "volume": [1000, 2000, 3000, 4000, 5000],
                "factor": [1.0, 1.0, 1.0, 1.0, 1.0],
            },
            index=pd.MultiIndex.from_arrays(
                [dates, ["SH600000"] * 5], names=["datetime", "instrument"]
            ),
        )

        out_dir = tmp_path / "qlib_data"
        dump_qlib_binary_native(df, out_dir)

        loaded = _load_existing_binary(out_dir)
        assert loaded is not None
        assert loaded.shape[0] == 5
        assert loaded.index.get_level_values("instrument").nunique() == 1

    def test_merge_incremental(self):
        """Merge existing + new data, new data takes precedence on overlap."""
        dates_old = pd.date_range("2025-01-01", periods=5, freq="B")
        # Use dates that start AFTER all old dates end (Jan 8 onward)
        dates_new = pd.date_range("2025-01-08", periods=5, freq="B")

        old_df = pd.DataFrame(
            {
                "close": [10.0, 11.0, 12.0, 13.0, 14.0],
                "open": [10.0] * 5,
                "high": [11.0] * 5,
                "low": [9.0] * 5,
                "volume": [1000] * 5,
                "factor": [1.0] * 5,
            },
            index=pd.MultiIndex.from_arrays(
                [dates_old, ["SH600000"] * 5], names=["datetime", "instrument"]
            ),
        )

        new_df = pd.DataFrame(
            {
                "close": [15.0, 16.0, 17.0, 18.0, 19.0],
                "open": [15.0] * 5,
                "high": [16.0] * 5,
                "low": [14.0] * 5,
                "volume": [2000] * 5,
                "factor": [1.0] * 5,
            },
            index=pd.MultiIndex.from_arrays(
                [dates_new, ["SH600000"] * 5], names=["datetime", "instrument"]
            ),
        )

        merged = _merge_incremental_data(old_df, new_df)
        # Should have all unique dates from both (no overlap)
        n_dates = merged.index.get_level_values("datetime").nunique()
        assert n_dates == 10

    def test_merge_with_overlap(self):
        """Overlapping dates: new data takes precedence."""
        dates = pd.date_range("2025-01-01", periods=5, freq="B")

        old_df = pd.DataFrame(
            {
                "close": [10.0, 11.0, 12.0, 13.0, 14.0],
                "open": [10.0] * 5,
                "high": [11.0] * 5,
                "low": [9.0] * 5,
                "volume": [1000] * 5,
                "factor": [1.0] * 5,
            },
            index=pd.MultiIndex.from_arrays(
                [dates, ["SH600000"] * 5], names=["datetime", "instrument"]
            ),
        )

        # New data overlaps last 2 dates
        new_dates = dates[3:]  # Last 2 dates
        new_df = pd.DataFrame(
            {
                "close": [99.0, 99.0],
                "open": [99.0] * 2,
                "high": [99.0] * 2,
                "low": [99.0] * 2,
                "volume": [9999] * 2,
                "factor": [1.0] * 2,
            },
            index=pd.MultiIndex.from_arrays(
                [new_dates, ["SH600000"] * 2], names=["datetime", "instrument"]
            ),
        )

        merged = _merge_incremental_data(old_df, new_df)
        assert merged.index.get_level_values("datetime").nunique() == 5
        # Last 2 values should be from new data
        last_close = merged.xs("SH600000", level="instrument")["close"].iloc[-1]
        assert last_close == 99.0


# ---------------------------------------------------------------------------
# #9: Disk cache tests
# ---------------------------------------------------------------------------


class TestDiskCache:
    """Test TTL-based disk cache for API responses."""

    def test_write_and_read(self, tmp_path, monkeypatch):
        """Write to cache and read back."""
        monkeypatch.setattr("fullstackautoquant.data.tushare_provider._CACHE_DIR", tmp_path)
        _write_cache("test_ns", "key1", ["a", "b", "c"])
        result = _read_cache("test_ns", "key1", ttl_seconds=3600)
        assert result == ["a", "b", "c"]

    def test_expired_cache(self, tmp_path, monkeypatch):
        """Expired cache returns None."""
        monkeypatch.setattr("fullstackautoquant.data.tushare_provider._CACHE_DIR", tmp_path)
        _write_cache("test_ns", "key2", ["x"])

        # Manually set cached_at to the past
        from fullstackautoquant.data.tushare_provider import _cache_path

        path = _cache_path("test_ns", "key2")
        data = json.loads(path.read_text())
        data["cached_at"] = time.time() - 7200  # 2 hours ago
        path.write_text(json.dumps(data))

        result = _read_cache("test_ns", "key2", ttl_seconds=3600)
        assert result is None  # Expired

    def test_missing_cache(self, tmp_path, monkeypatch):
        """Missing cache returns None."""
        monkeypatch.setattr("fullstackautoquant.data.tushare_provider._CACHE_DIR", tmp_path)
        result = _read_cache("test_ns", "nonexistent", ttl_seconds=3600)
        assert result is None


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
