#!/usr/bin/env python3
"""Lightweight daily orchestrator: Tushare → Qlib → Factors → Inference.

Replaces the entire Docker/Dolt pipeline with a single command:

    python scripts/run_daily_lite.py --date auto

Steps:
  1. Fetch ~200 trading days from Tushare → minimal Qlib binary (~3 MB)
  2. Export daily_pv.h5 (reuses existing export_daily_pv.py)
  3. Run factor synthesis (reuses existing factor_synthesis.py)
  4. Run inference with cached normalizer (uses norm_params.pkl)

Prerequisites:
  - TUSHARE env var set (or in .env)
  - weights/norm_params.pkl (one-time: python scripts/extract_norm_cache.py)
  - weights/params.pkl or weights/state_dict_cpu.pt (model weights)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PKG = REPO_ROOT / "fullstackautoquant"
DATA_DIR = PKG / "data"
MODEL_DIR = PKG / "model"


def _log(step: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{step}] {msg}")


def _run(cmd: list[str], desc: str) -> None:
    """Run a subprocess, fail fast on error."""
    _log(desc, f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"{desc} failed (exit {result.returncode})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Lightweight daily inference (Tushare-only)")
    ap.add_argument("--date", default="auto", help="Target date YYYY-MM-DD or 'auto'")
    ap.add_argument("--n-days", type=int, default=200, help="Trading days to fetch (default 200)")
    ap.add_argument(
        "--norm-cache",
        default=str(REPO_ROOT / "weights" / "norm_params.pkl"),
        help="Path to cached normalizer params",
    )
    ap.add_argument(
        "--params",
        default=None,
        help="Model weights path (auto-detect from weights/ if not set)",
    )
    ap.add_argument("--out", default=str(REPO_ROOT / "output" / "ranked_scores.csv"))
    ap.add_argument("--provider_uri", default="~/.qlib/qlib_data/cn_data")
    ap.add_argument("--skip-fetch", action="store_true", help="Skip Tushare fetch (data exists)")
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv()

    tushare_token = os.getenv("TUSHARE") or os.getenv("TS_TOKEN")
    if not tushare_token and not args.skip_fetch:
        print("[ERROR] TUSHARE env var required. Set it in .env or shell.", file=sys.stderr)
        return 1

    norm_cache = Path(args.norm_cache)
    if not norm_cache.exists():
        print(
            f"[ERROR] Normalizer cache not found: {norm_cache}\n"
            "Run once: python scripts/extract_norm_cache.py",
            file=sys.stderr,
        )
        return 1

    provider_uri = os.path.expanduser(args.provider_uri)
    daily_pv = DATA_DIR / "daily_pv.h5"

    # Auto-detect model weights
    params = Path(args.params) if args.params else None
    if params is None:
        for candidate in ["state_dict_cpu.pt", "params.pkl"]:
            p = REPO_ROOT / "weights" / candidate
            if p.exists():
                params = p
                break
    if params is None or not params.exists():
        print("[ERROR] Model weights not found in weights/", file=sys.stderr)
        return 1

    t0 = time.time()

    # ---- Step 1: Fetch from Tushare → Qlib binary ----
    if not args.skip_fetch:
        _log("1/4", "Fetching data from Tushare")
        from fullstackautoquant.data.tushare_provider import build_minimal_qlib_data

        ref_date = None if args.date == "auto" else args.date
        build_minimal_qlib_data(
            token=tushare_token,
            n_days=args.n_days,
            qlib_dir=provider_uri,
            ref_date=ref_date,
        )
    else:
        _log("1/4", "Skipped (--skip-fetch)")

    # ---- Step 2: Export Qlib → daily_pv.h5 ----
    _log("2/4", "Exporting daily_pv.h5")
    _run(
        [
            sys.executable,
            str(DATA_DIR / "export_daily_pv.py"),
            "--instruments", "csi300",
            "--end", "auto",
            "--provider_uri", provider_uri,
            "--out", str(daily_pv),
        ],
        "export_daily_pv",
    )

    # ---- Step 3: Factor synthesis ----
    _log("3/4", "Synthesizing custom factors")
    _run(
        [
            sys.executable,
            str(DATA_DIR / "factor_synthesis.py"),
            "--workspace", str(MODEL_DIR),
            "--provider_uri", provider_uri,
        ],
        "factor_synthesis",
    )

    # ---- Step 4: Inference with cached normalizer ----
    _log("4/4", "Running inference (cached normalizer)")
    _run(
        [
            sys.executable,
            "-m", "fullstackautoquant.model.inference",
            "--date", args.date,
            "--combined_factors", str(MODEL_DIR / "combined_factors_df.parquet"),
            "--params", str(params),
            "--out", args.out,
            "--norm-cache", str(norm_cache),
            "--provider_uri", provider_uri,
        ],
        "inference",
    )

    elapsed = time.time() - t0
    _log("DONE", f"Pipeline complete in {elapsed:.0f}s → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
