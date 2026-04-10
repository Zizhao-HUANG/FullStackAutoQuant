"""MC Dropout confidence distribution analysis.

Analyzes prediction confidence patterns across all inference days,
producing per-day histograms and overall distribution statistics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.dashboard_export.constants import CONFIDENCE_BINS
from scripts.dashboard_export.scores import DailyScores


def build_confidence_distribution(all_scores: list[DailyScores]) -> dict[str, Any]:
    """Analyze MC Dropout confidence patterns across all days."""
    if not all_scores:
        return {"bins": [], "history": []}

    all_confs: list[float] = []
    history: list[dict[str, Any]] = []

    for day in all_scores:
        confs = [s.get("confidence", 0) for s in day.instruments if "confidence" in s]
        all_confs.extend(confs)

        if confs:
            # Histogram of confidence bins
            conf_arr = np.array(confs)
            bin_counts: dict[str, int] = {}
            for i in range(len(CONFIDENCE_BINS) - 1):
                lo, hi = CONFIDENCE_BINS[i], CONFIDENCE_BINS[i + 1]
                label = f"{lo:.2f}-{hi:.2f}"
                count = int(((conf_arr >= lo) & (conf_arr < hi)).sum())
                bin_counts[label] = count
            # Last bin includes upper bound
            last_label = f"≥{CONFIDENCE_BINS[-1]:.2f}"
            bin_counts[last_label] = int((conf_arr >= CONFIDENCE_BINS[-1]).sum())

            history.append({
                "date": day.date,
                "mean": round(float(np.mean(conf_arr)), 6),
                "std": round(float(np.std(conf_arr)), 6),
                "min": round(float(np.min(conf_arr)), 6),
                "max": round(float(np.max(conf_arr)), 6),
                "bins": bin_counts,
            })

    # Overall distribution
    if all_confs:
        overall_arr = np.array(all_confs)
        overall_stats: dict[str, Any] = {
            "total_predictions": len(all_confs),
            "mean": round(float(np.mean(overall_arr)), 6),
            "std": round(float(np.std(overall_arr)), 6),
            "percentiles": {
                f"p{p}": round(float(np.percentile(overall_arr, p)), 6)
                for p in [1, 5, 25, 50, 75, 95, 99]
            },
        }
    else:
        overall_stats = {}

    return {"overall": overall_stats, "history": history}
