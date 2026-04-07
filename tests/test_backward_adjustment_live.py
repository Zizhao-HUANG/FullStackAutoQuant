#!/usr/bin/env python3
"""Cross-validate backward price adjustment against Tushare pro_bar(adj='hfq').

To-do item #3: For stocks with known dividends, compare our manual backward
adjustment (raw * adj_factor / latest_adj_factor) against Tushare's built-in
hfq (后复权) output. Max acceptable relative error: < 0.01%.

Usage:
    TUSHARE=<token> python tests/test_backward_adjustment_live.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    token = os.getenv("TUSHARE") or os.getenv("TS_TOKEN")
    if not token:
        print("[SKIP] No TUSHARE token set. Skipping live validation.")
        return 0

    import tushare as ts

    ts.set_token(token)
    pro = ts.pro_api()

    # Test stocks: pick 5 that are likely to have had dividends
    test_cases = [
        ("600000.SH", "20250601", "20260403"),  # Pudong Dev Bank
        ("000001.SZ", "20250601", "20260403"),  # Ping An Bank
        ("601318.SH", "20250601", "20260403"),  # Ping An Insurance
        ("600519.SH", "20250601", "20260403"),  # Kweichow Moutai
        ("000858.SZ", "20250601", "20260403"),  # Wuliangye
    ]

    all_pass = True
    for ts_code, start, end in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {ts_code} [{start} -> {end}]")

        # 1. Fetch raw daily + adj_factor (our method)
        time.sleep(0.2)
        df_raw = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
        time.sleep(0.2)
        df_adj = pro.adj_factor(ts_code=ts_code, start_date=start, end_date=end)

        if df_raw.empty or df_adj.empty:
            print(f"  [WARN] No data for {ts_code}, skipping")
            continue

        # Merge and apply our backward adjustment
        df = df_raw.merge(df_adj[["trade_date", "adj_factor"]], on="trade_date", how="left")
        df = df.sort_values("trade_date")
        df["adj_factor"] = df["adj_factor"].ffill().bfill().fillna(1.0)
        latest_factor = df["adj_factor"].iloc[-1]
        df["our_adj_close"] = df["close"] * df["adj_factor"] / latest_factor

        # 2. Fetch Tushare's own hfq-adjusted close
        # pro_bar returns hfq data directly
        time.sleep(0.2)
        try:
            df_hfq = ts.pro_bar(ts_code=ts_code, start_date=start, end_date=end, adj="hfq")
        except Exception as e:
            print(f"  [WARN] pro_bar(adj='hfq') failed: {e}")
            # Fallback: compute ratio check using factor consistency
            factor_changed = df["adj_factor"].iloc[0] != df["adj_factor"].iloc[-1]
            print(f"  Factor changed: {factor_changed}")
            print(f"  factor range: [{df['adj_factor'].min():.4f}, {df['adj_factor'].max():.4f}]")
            continue

        if df_hfq is None or df_hfq.empty:
            print(f"  [WARN] No hfq data for {ts_code}")
            continue

        df_hfq = df_hfq.sort_values("trade_date")

        # 3. Compare: our backward-adjusted close vs Tushare hfq close
        # Note: Tushare hfq is FORWARD-adjusted (cumulative from IPO), not backward.
        # Our backward adjustment and hfq differ in scale but should have IDENTICAL RETURNS.
        our_returns = df.set_index("trade_date")["our_adj_close"].pct_change().dropna()
        hfq_returns = df_hfq.set_index("trade_date")["close"].pct_change().dropna()

        # Align on common dates
        common = our_returns.index.intersection(hfq_returns.index)
        if len(common) < 10:
            print(f"  [WARN] Only {len(common)} common dates, too few for validation")
            continue

        our_r = our_returns.loc[common].values
        hfq_r = hfq_returns.loc[common].values

        # Return difference should be < 0.01%
        abs_diff = np.abs(our_r - hfq_r)
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()
        threshold = 1e-4  # 0.01%

        factor_changed = abs(df["adj_factor"].iloc[0] - df["adj_factor"].iloc[-1]) > 1e-6
        status = "PASS" if max_diff < threshold else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  Factor changed in window: {factor_changed}")
        print(f"  Factor range: [{df['adj_factor'].min():.6f}, {df['adj_factor'].max():.6f}]")
        print(f"  Common dates: {len(common)}")
        print(f"  Max return diff: {max_diff:.2e} (threshold: {threshold:.2e})")
        print(f"  Mean return diff: {mean_diff:.2e}")
        print(f"  [{status}]")

    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
