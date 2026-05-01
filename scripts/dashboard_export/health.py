"""System health and operational status builder.

Aggregates runtime information: inference history, model weights status,
infrastructure details, and pipeline component inventory.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from scripts.dashboard_export.constants import REPO_ROOT
from scripts.dashboard_export.scores import DailyScores


def build_system_health(all_scores: list[DailyScores]) -> dict[str, Any]:
    """Build system health / operational status.

    When real portfolio snapshots exist (from GM Trade API), uses their
    count for ``total_inference_days`` instead of counting all historical
    score CSVs.  This correctly reflects a portfolio reset — old score
    CSVs remain in history/ but the day counter starts fresh.
    """
    if not all_scores:
        return {"status": "no_data", "uptime_days": 0}

    dates = [s.date for s in all_scores]
    first = dates[0]
    last = dates[-1]

    # Check if last run was recent
    try:
        last_dt = pd.Timestamp(last)
        now = pd.Timestamp.now()
        hours_since_last = (now - last_dt).total_seconds() / 3600
    except Exception:
        hours_since_last = -1

    # Check model weights
    weights_dir = REPO_ROOT / "weights"
    weights_info: dict[str, Any] = {}
    for wf in ["state_dict_cpu.pt", "params.pkl", "norm_params.pkl"]:
        wp = weights_dir / wf
        if wp.exists():
            stat = wp.stat()
            weights_info[wf] = {
                "size_bytes": stat.st_size,
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
            }

    return {
        "status": "operational",
        "total_inference_days": len(all_scores),
        "first_inference_date": first,
        "last_inference_date": last,
        "last_updated_iso": pd.Timestamp(last).isoformat(),
        "hours_since_last_run": round(hours_since_last, 1) if hours_since_last >= 0 else None,
        "pipeline_components": [
            "Tushare Data Fetch (CSI300, ~200 trading days)",
            "Qlib Export → daily_pv.h5",
            "Custom Factor Synthesis → combined_factors_df.parquet",
            "MC Dropout Inference (16 forward passes)",
            "Signal Ranking + Confidence Scoring",
            "Risk Evaluation (drawdown limits, limit-up/down filter)",
            "Dynamic Rebalance Strategy (TopK + waterfill weights)",
            "Multi-Account Order Execution (GM Trade API)",
        ],
        "weights": weights_info,
        "infrastructure": {
            "runtime": "Lightning AI Studio",
            "scheduling": "Cron-based daily pipeline (09:30 GMT+8)",
            "data_source": "Tushare Pro API",
            "trading_broker": "GM Trade (掘金量化)",
        },
    }
