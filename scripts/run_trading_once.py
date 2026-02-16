#!/usr/bin/env python3
"""Run one-shot trading pipeline.

Executes the full trading cycle once:
    Signal parsing → Risk evaluation → Strategy → Order placement

Usage:
    python scripts/run_trading_once.py \\
        --csv output/ranked_scores.csv \\
        --config configs/trading.auto.local.yaml

    # With order placement (live trading):
    python scripts/run_trading_once.py \\
        --csv output/ranked_scores.csv \\
        --config configs/trading.auto.local.yaml \\
        --place
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one-shot trading pipeline")
    p.add_argument("--csv", required=True, help="Path to ranked_scores.csv")
    p.add_argument("--config", default=None, help="Path to trading config YAML")
    p.add_argument("--place", action="store_true", help="Actually place orders (default: dry-run)")
    p.add_argument("--override-buy", action="store_true", help="Override buy block for this run")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        print("Run inference first: make inference")
        sys.exit(1)

    print("=" * 60)
    print("FullStackAutoQuant — One-Shot Trading Pipeline")
    print("=" * 60)
    print()
    print(f"  Signals:    {csv_path}")
    print(f"  Config:     {args.config or '(default)'}")
    print(f"  Mode:       {'LIVE' if args.place else 'DRY-RUN'}")
    print(f"  Override:   {args.override_buy}")
    print()

    from fullstackautoquant.trading.run_once import run_pipeline

    result = run_pipeline(
        csv_path=str(csv_path),
        config_path=args.config,
        place_orders=args.place,
        override_buy=args.override_buy,
    )

    print()
    print("Pipeline complete.")
    if result:
        print(f"  Orders generated: {result.get('n_orders', 0)}")
        print(f"  Buy orders:       {result.get('n_buy', 0)}")
        print(f"  Sell orders:      {result.get('n_sell', 0)}")
    if not args.place:
        print("\n  (Dry-run mode — no orders placed. Use --place to execute.)")


if __name__ == "__main__":
    main()
