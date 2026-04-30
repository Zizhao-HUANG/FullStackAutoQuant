#!/usr/bin/env python3
"""Snapshot current portfolio positions from GM Trade API.

Called by the daily pipeline (run_full_pipeline.sh) to capture a daily
snapshot of real portfolio holdings. This enables the dashboard to show
actual portfolio returns instead of simulated ones.

Usage:
    python scripts/snapshot_portfolio.py
    python scripts/snapshot_portfolio.py --account-id <id>
    python scripts/snapshot_portfolio.py --config configs/trading.yaml
    python scripts/snapshot_portfolio.py --date 2026-04-30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.dashboard_export.portfolio import snapshot_positions


def main() -> int:
    ap = argparse.ArgumentParser(description="Snapshot portfolio positions from GM Trade API")
    ap.add_argument("--account-id", default=None, help="Override GM account ID")
    ap.add_argument("--config", default=None, help="Trading config YAML path")
    ap.add_argument("--date", default=None, help="Override snapshot date (YYYY-MM-DD)")
    args = ap.parse_args()

    result = snapshot_positions(
        account_id=args.account_id,
        config_path=args.config,
        date_override=args.date,
    )

    if result is None:
        print("[WARN] Portfolio snapshot failed — no data saved", file=sys.stderr)
        # Non-fatal: return 0 so pipeline continues
        return 0

    print(f"[OK] Snapshot: date={result['date']}, NAV={result['nav']:,.2f}, "
          f"positions={result['position_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
