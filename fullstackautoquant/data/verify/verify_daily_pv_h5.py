#!/usr/bin/env python3
"""
Verify daily_pv.h5 file structure and basic health (consistent with RD Agent conventions):
- Can be read by pd.read_hdf(path, key='data')
- Index is MultiIndex('datetime','instrument') in ascending order
- Columns contain at least --require fields (default check '$close')
- Unique index, reasonable time range
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_required(csv_fields: str | None) -> list[str]:
    if not csv_fields:
        return ["$close"]
    return [f.strip() for f in csv_fields.split(",") if f.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate daily_pv.h5 structure and basic health")
    parser.add_argument("--path", required=True, help="Path to daily_pv.h5")
    parser.add_argument("--require", default="$close", help="Comma-separated required columns, e.g., '$open,$high,$low,$close,$volume,$factor'")
    args = parser.parse_args()

    path = Path(args.path).expanduser().resolve()
    require_cols = parse_required(args.require)

    print(f"[INFO] Loading HDF5: {path}")
    try:
        df = pd.read_hdf(str(path), key="data")
    except Exception as e:
        print(f"[ERROR] Failed to read HDF5 with key='data': {e}")
        return 1

    print(f"[INFO] Shape={df.shape}")

    # 1) Index is MultiIndex('datetime','instrument')
    if not isinstance(df.index, pd.MultiIndex):
        print("[ERROR] Index is not a pandas.MultiIndex")
        return 1
    if list(df.index.names) != ["datetime", "instrument"]:
        print(f"[ERROR] Index names mismatch: got {df.index.names}, expect ['datetime','instrument']")
        return 1

    # 2) Index ascending order
    if not df.index.is_monotonic_increasing:
        print("[ERROR] Index is not monotonically increasing (ascending)")
        return 1

    # 3) Unique index
    if not df.index.is_unique:
        print("[ERROR] Index is not unique")
        return 1

    # 4) Required columns exist
    missing = [c for c in require_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns: {missing}")
        return 1

    # 5) Basic time range
    try:
        first_day = df.index.get_level_values("datetime").min()
        last_day = df.index.get_level_values("datetime").max()
        print(f"[INFO] Date range: {first_day} → {last_day}")
    except Exception as e:
        print(f"[WARN] Failed to compute date range: {e}")

    # 6) Report last day cross-section size (advisory only)
    try:
        last = df.xs(last_day, level="datetime", drop_level=False)
        cs = last.groupby("datetime", group_keys=False).size().iloc[-1]
        print(f"[INFO] Last day cross-section size: {int(cs)}")
    except Exception:
        pass

    print("[OK] daily_pv.h5 validated successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


