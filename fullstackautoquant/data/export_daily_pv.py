#!/usr/bin/env python3
"""
Exports daily OHLCV (optionally $factor) data from Qlib to HDF5 file `daily_pv.h5`(key='data').

Design principles (strictly equivalent to original implementation, for audit):
- Uses official Qlib Data API:D.features(..., freq='day')
- Index processing: swaplevel().sort_index(), yielding MultiIndex ['datetime','instrument'] ascending order
- Persistence: pandas to_hdf(key='data'), no forward/backward fill or extra processing
- Default: export-as-is (all instruments, fields include $factor)

Note: downstream factor scripts use pd.read_hdf('daily_pv.h5', key='data') to read data, typically depending on at least '$close'.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Sequence

import pandas as pd
import qlib
from qlib.data import D


def _parse_fields(csv_fields: str | None) -> List[str]:
    if csv_fields is None:
    # Default "as-is" export: OHLCV + $factor
        return ["$open", "$high", "$low", "$close", "$volume", "$factor"]
    return [f.strip() for f in csv_fields.split(",") if f.strip()]


def _export_daily_pv(
    instruments: str,
    start_time: str | None,
    end_time: str | None,
    fields: List[str],
    provider_uri: str,
    region: str,
) -> pd.DataFrame:
    # Initialize Qlib (same provider path and region as other scripts)
    qlib.init(provider_uri=provider_uri, region=region)

    # Get instrument universe
    inst = D.instruments(instruments)

    # Same as original generate.py: fetch raw data via D.features, no extra fill
    df = D.features(inst, fields, start_time=start_time, end_time=end_time, freq="day")

    # Swap levels, sort ascending: yields MultiIndex['datetime','instrument']
    df = df.swaplevel().sort_index()

    return df


def _assert_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in exported data: {missing}")


def _assert_last_day_coverage(df: pd.DataFrame, expect_instruments: int | None) -> None:
    if expect_instruments is None or df.empty:
        return
    # Use last trading day from export itself for overwrite check(avoid external calendar read discrepancies)
    last_day = df.index.get_level_values("datetime").max()
    last_cnt = df.xs(last_day, level="datetime", drop_level=False).groupby("datetime", group_keys=False).size()
    # last_cnt is Series[datetime]->count, take only the last item
    last_value = int(last_cnt.iloc[-1]) if len(last_cnt) else 0
    if last_value != expect_instruments:
        raise RuntimeError(
            f"CSI300 coverage on last day != {expect_instruments}: got {last_value} on {pd.Timestamp(last_day).date()}"
        )


def _save_hdf(df: pd.DataFrame, out_path: str) -> None:
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Same as original: key='data', default fixed format
    df.to_hdf(str(out), key="data", format="table")


def _maybe_copy(out_path: str, copy_to: Sequence[str] | None) -> None:
    if not copy_to:
        return
    src = Path(out_path).expanduser().resolve()
    for target in copy_to:
        dst_path = Path(target).expanduser().resolve()
        if dst_path.is_dir():
            dst_file = dst_path / "daily_pv.h5"
        else:
            # Allow giving .h5 file path directly
            if not dst_path.suffix:
                # No extension and not a directory, treat as directory
                dst_path.mkdir(parents=True, exist_ok=True)
                dst_file = dst_path / "daily_pv.h5"
            else:
                dst_file = dst_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst_file))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export daily OHLCV from Qlib to daily_pv.h5 for factor calculation")
    parser.add_argument("--instruments", default="all", help="Qlib instruments, e.g., all, csi300 (pool/time filtering is handled by downstream workflow)")
    parser.add_argument("--fields", default=None, help="Comma separated fields, e.g., '$open,$high,$low,$close,$volume'")
    parser.add_argument("--start", default="", help="Start date (inclusive). Empty means use all available data from the beginning")
    parser.add_argument("--end", default="auto", help="End date (inclusive). 'auto' means use all available data")
    parser.add_argument("--provider_uri", default="~/.qlib/qlib_data/cn_data", help="Qlib provider uri")
    parser.add_argument("--region", default="cn", help="Qlib region, default cn")
    parser.add_argument("--out", default="./daily_pv.h5", help="Output HDF5 path")
    parser.add_argument(
        "--expect_instruments",
        type=int,
        default=0,
        help="Expected cross-section size on the last trading day for coverage assertion; 0 (default) disables the check",
    )
    parser.add_argument(
        "--copy_to",
        nargs="*",
        default=None,
        help="Optional directories or file paths to also copy the generated daily_pv.h5 to",
    )

    args = parser.parse_args()

    fields = _parse_fields(args.fields)
    # Downstream factors typically require '$close'; warn but don't enforce to keep "as-is" export
    if "$close" not in fields:
        print("[WARN] '$close' not in fields; some factor implementations may fail.")

    end_time = None if args.end == "auto" else args.end
    start_time = None if not args.start else args.start

    df = _export_daily_pv(
        instruments=args.instruments,
        start_time=start_time,
        end_time=end_time,
        fields=fields,
        provider_uri=os.path.expanduser(args.provider_uri),
        region=args.region,
    )

    _assert_required_columns(df, required=["$close"])
    _assert_last_day_coverage(df, expect_instruments=(None if args.expect_instruments == 0 else args.expect_instruments))

    _save_hdf(df, args.out)
    _maybe_copy(args.out, args.copy_to)

    print(f"Exported daily_pv.h5 with shape={df.shape} → {Path(args.out).resolve()}")
    if args.copy_to:
        print(f"Also copied to: {[str(Path(p).expanduser().resolve()) for p in args.copy_to]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


