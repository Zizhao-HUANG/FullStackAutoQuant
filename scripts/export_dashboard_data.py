#!/usr/bin/env python3
"""Dashboard data export — CLI entry point.

Generates structured JSON for the FullStackAutoQuant Live Dashboard.
All heavy logic is in the ``scripts.dashboard_export`` subpackage.

Output structure:
    dashboard/data/
    ├── meta/           — system_info.json, trading_config.json
    ├── signals/        — latest.json + history/signals_YYYY-MM-DD.json
    ├── performance/    — summary.json, daily_stats.json, equity_curve.json
    ├── risk/           — latest.json, confidence_distribution.json
    ├── trading/        — latest_targets.json, latest_orders.json, execution_history.json
    └── dashboard_data.json  — combined master file (all-in-one for frontend)

Usage:
    python scripts/export_dashboard_data.py
    python scripts/export_dashboard_data.py --out-dir docs/dashboard/data
    python scripts/export_dashboard_data.py --pretty
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for ``scripts.dashboard_export`` imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.dashboard_export.assembler import build_master_json, build_subdirectory_files
from scripts.dashboard_export.confidence import build_confidence_distribution
from scripts.dashboard_export.constants import REPO_ROOT, log
from scripts.dashboard_export.health import build_system_health
from scripts.dashboard_export.metadata import extract_system_info, extract_trading_config
from scripts.dashboard_export.performance import build_equity_curve, compute_performance
from scripts.dashboard_export.scores import collect_and_parse_scores
from scripts.dashboard_export.trading import extract_trading_logs
from scripts.dashboard_export.writer import write_json


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export dashboard data from existing CSV/JSON pipeline artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "dashboard" / "data"),
        help="Output directory for dashboard data (default: dashboard/data/)",
    )
    ap.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON files (adds indentation)",
    )
    ap.add_argument(
        "--master-only",
        action="store_true",
        help="Only generate the combined master JSON (skip subdirectory files)",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    pretty = args.pretty
    t0 = time.time()

    log("START", f"Exporting dashboard data → {out_dir}")

    # ── Step 1: Collect and parse all score CSVs ──────────────────
    log("1/7", "Collecting ranked scores CSVs")
    all_scores = collect_and_parse_scores()
    if not all_scores:
        return 1

    # ── Step 2: Compute performance metrics ───────────────────────
    log("2/7", "Computing performance metrics")
    performance = compute_performance(all_scores)

    # ── Step 3: Build equity curve ────────────────────────────────
    log("3/7", "Building simulated equity curve")
    equity_curve = build_equity_curve(all_scores)

    # ── Step 4: Analyze confidence distribution ───────────────────
    log("4/7", "Analyzing MC Dropout confidence distribution")
    confidence = build_confidence_distribution(all_scores)

    # ── Step 5: Extract system metadata ───────────────────────────
    log("5/7", "Extracting system metadata and trading config")
    system_info = extract_system_info()
    trading_config = extract_trading_config()

    # ── Step 6: Extract trading logs ──────────────────────────────
    log("6/7", "Extracting trading logs")
    trading_logs = extract_trading_logs()
    system_health = build_system_health(all_scores)

    # ── Step 7: Write output files ────────────────────────────────
    log("7/7", "Writing output files")

    if not args.master_only:
        build_subdirectory_files(
            out_dir, all_scores, performance, equity_curve,
            confidence, system_info, trading_config, trading_logs, pretty,
        )

    # Master combined JSON (always generated)
    master = build_master_json(
        all_scores, performance, equity_curve, confidence,
        system_info, trading_config, trading_logs, system_health,
    )
    write_json(master, out_dir / "dashboard_data.json", pretty)

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    total_files = sum(1 for _ in Path(out_dir).rglob("*") if _.is_file())
    total_size = sum(f.stat().st_size for f in Path(out_dir).rglob("*") if f.is_file())

    log("DONE", f"Export complete in {elapsed:.1f}s")
    log("DONE", f"  Output directory: {out_dir}")
    log("DONE", f"  Total files: {total_files}")
    log("DONE", f"  Total size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    log("DONE", f"  Trading days covered: {len(all_scores)}")
    log("DONE", f"  Date range: {all_scores[0].date} → {all_scores[-1].date}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
