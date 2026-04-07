#!/usr/bin/env python3
"""One-time extraction of normalizer parameters from a full-data Qlib setup.

Run ONCE while the full Qlib binary data is still available, then commit
the output file to version control alongside model weights:

    python scripts/extract_norm_cache.py \
        --provider_uri ~/.qlib/qlib_data/cn_data \
        --task-config configs/task_rendered.yaml \
        --out weights/norm_params.pkl

Produces weights/norm_params.pkl (~1 KB) containing 22 medians + 22 MAD-scales.
After this, the 5 GB Qlib binary store is no longer needed for daily inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root without install
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract and cache RobustZScoreNorm parameters")
    ap.add_argument(
        "--provider_uri",
        default="~/.qlib/qlib_data/cn_data",
        help="Qlib provider URI with full historical data",
    )
    ap.add_argument(
        "--task-config",
        default=str(REPO_ROOT / "configs" / "task_rendered.yaml"),
        help="Path to task_rendered.yaml",
    )
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "weights" / "norm_params.pkl"),
        help="Output pickle path for cached normalizer params",
    )
    ap.add_argument("--region", default="cn", help="Qlib region (default: cn)")
    args = ap.parse_args()

    import os

    import qlib

    from fullstackautoquant.model.norm_cache import extract_norm_params, save_norm_cache
    from fullstackautoquant.model.task_config import (
        build_handler_from_task,
        get_training_time_range,
        load_task_config,
    )

    # 1. Initialize Qlib with full data
    provider_uri = os.path.expanduser(args.provider_uri)
    print(f"Initializing Qlib with full data: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=args.region)

    # 2. Build handler (this triggers RobustZScoreNorm.fit on the 17-year window)
    task_cfg = load_task_config(Path(args.task_config))
    training_range = get_training_time_range(task_cfg)

    # We need a combined_factors_df.parquet to build the handler
    # Look for it in the standard location
    factors_path = REPO_ROOT / "fullstackautoquant" / "model" / "combined_factors_df.parquet"
    if not factors_path.exists():
        print(f"[ERROR] Cannot find {factors_path}. Run factor_synthesis.py first.", file=sys.stderr)
        return 1

    print(
        f"Building handler with fit window [{training_range['start_time']}, "
        f"{training_range['end_time']}]..."
    )
    handler = build_handler_from_task(
        task_cfg=task_cfg,
        combined_factors_path=factors_path,
        start_time=training_range["start_time"],
        end_time=training_range["end_time"],
    )

    # 3. Extract and save
    params = extract_norm_params(handler)
    out_path = Path(args.out)
    save_norm_cache(params, out_path)

    # 4. Print summary
    print(f"\n{'='*60}")
    print(f"Normalizer cache saved:  {out_path}")
    print(f"  Features: {len(params['median'])}")
    print(f"  Fit window: [{params['fit_start']}, {params['fit_end']}]")
    print(f"  File size: {out_path.stat().st_size:,} bytes")
    print(f"{'='*60}")
    print("\nThis file should be committed alongside model weights.")
    print("The full Qlib binary data is no longer needed for daily inference.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
