#!/usr/bin/env python3

"""
Verify that the built inference feature parquet satisfies training consistency requirements:
  - Index: MultiIndex ['datetime', 'instrument']
  - Columns: MultiIndex with only ('feature', name) groups, count = 22
  - Feature order: 20 Alpha158 features (fixed order) + 2 custom factors
  - Value range: [-3, 3] (RobustZScoreNorm + clip)
  - Missing: none
  - Latest trading day: instrument count = 300 (CSI300)

Usage:
  python -m fullstackautoquant.data.verify.verify_parquet_ready_for_infer \
    --path /path/to/features_ready_infer_auto.parquet \
    --combined_factors /path/to/combined_factors_df.parquet
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

ALPHA158_NAMES: list[str] = [
    "RESI5",
    "WVMA5",
    "RSQR5",
    "KLEN",
    "RSQR10",
    "CORR5",
    "CORD5",
    "CORR10",
    "ROC60",
    "RESI10",
    "VSTD5",
    "RSQR60",
    "CORR60",
    "WVMA60",
    "STD5",
    "RSQR20",
    "CORD60",
    "CORD10",
    "CORR20",
    "KLOW",
]


def load_new_factor_names(cf_path: str) -> list[str]:
    df = pd.read_parquet(cf_path)
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples([("feature", str(c)) for c in df.columns])
    return [c[1] for c in df.columns if c[0] == "feature"]


def verify(path: str, cf_path: str | None) -> int:
    df = pd.read_parquet(path)
    issues: list[str] = []

    # index check
    if not isinstance(df.index, pd.MultiIndex) or df.index.names != ["datetime", "instrument"]:
        issues.append("index structure invalid (expect MultiIndex ['datetime','instrument'])")

    # columns check
    if not isinstance(df.columns, pd.MultiIndex):
        issues.append("columns not MultiIndex")
    else:
        if not all(c[0] == "feature" for c in df.columns):
            issues.append("columns contain non-feature groups")
        if len(df.columns) != 22:
            issues.append(f"feature count {len(df.columns)} != 22")

        # order check
        if cf_path:
            new_factors = load_new_factor_names(cf_path)
        else:
            # fallback: take the last two columns directly
            new_factors = [df.columns[-2][1], df.columns[-1][1]]
        expected_order = ALPHA158_NAMES + [n for n in new_factors if n not in ALPHA158_NAMES]
        actual_order = [c[1] for c in df.columns]
        if expected_order != actual_order:
            issues.append("feature order mismatch with training (Alpha158+new factors)")

    # values and missing
    if len(df) == 0:
        issues.append("dataframe is empty")
    else:
        vmin = float(np.nanmin(df.values))
        vmax = float(np.nanmax(df.values))
        nan_count = int(np.isnan(df.values).sum())
        if nan_count != 0:
            issues.append(f"nan_count={nan_count}")
        if not (vmin >= -3 - 1e-8 and vmax <= 3 + 1e-8):
            issues.append(f"value_range=[{vmin:.6f},{vmax:.6f}] out of [-3,3]")

        # last day instrument coverage
        last_dt = df.index.get_level_values("datetime").max()
        last_inst = df.loc[(last_dt, slice(None)), :].index.get_level_values("instrument").nunique()
        if last_inst != 300:
            issues.append(f"last_day instruments={last_inst} != 300 (CSI300)")

    ok = len(issues) == 0
    print("OK" if ok else "FAIL")
    if len(df) > 0:
        print(
            "summary:",
            {
                "rows": len(df),
                "cols": len(df.columns),
                "first_dt": str(df.index.get_level_values("datetime").min()),
                "last_dt": str(df.index.get_level_values("datetime").max()),
            },
        )
    for m in issues:
        print("ISSUE:", m)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="parquet path to verify")
    ap.add_argument(
        "--combined_factors", default=None, help="combined_factors_df.parquet path for order check"
    )
    args = ap.parse_args()
    return verify(args.path, args.combined_factors)


if __name__ == "__main__":
    raise SystemExit(main())
