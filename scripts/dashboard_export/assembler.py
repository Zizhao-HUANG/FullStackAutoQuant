"""Master JSON assembly and per-subdirectory file generation.

Combines all collector outputs into the unified dashboard_data.json
and generates individual files for each subdirectory category.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.dashboard_export.constants import TOP_K_DISPLAY, log
from scripts.dashboard_export.performance import PerformanceSummary
from scripts.dashboard_export.scores import DailyScores
from scripts.dashboard_export.writer import write_json


# ---------------------------------------------------------------------------
# Per-day signal and stats builders
# ---------------------------------------------------------------------------
def _build_daily_signal_file(day: DailyScores) -> dict[str, Any]:
    """Build a per-day signal file for individual export."""
    return {
        "date": day.date,
        "universe": "CSI300",
        "total_instruments": day.total_count,
        "statistics": {
            "positive_count": day.positive_count,
            "negative_count": day.negative_count,
            "mean_score": day.mean_score,
            "median_score": day.median_score,
            "std_score": day.std_score,
            "min_score": day.min_score,
            "max_score": day.max_score,
            "mean_confidence": day.mean_confidence,
            "min_confidence": day.min_confidence,
            "max_confidence": day.max_confidence,
            "score_quantiles": day.score_quantiles,
        },
        "instruments": day.instruments,
    }


def _build_daily_stats_summary(all_scores: list[DailyScores]) -> list[dict[str, Any]]:
    """Build per-day statistics array for the performance subdirectory."""
    stats: list[dict[str, Any]] = []
    for day in all_scores:
        top_k_scores = [s["score"] for s in day.instruments[:TOP_K_DISPLAY]]
        top_k_confs = [
            s.get("confidence", 0)
            for s in day.instruments[:TOP_K_DISPLAY]
            if "confidence" in s
        ]
        stats.append({
            "date": day.date,
            "total_instruments": day.total_count,
            "positive_count": day.positive_count,
            "negative_count": day.negative_count,
            "positive_ratio": round(day.positive_count / max(day.total_count, 1), 4),
            "mean_score": day.mean_score,
            "median_score": day.median_score,
            "std_score": day.std_score,
            "score_spread_top_bottom": round(day.max_score - day.min_score, 8),
            "top_k_mean_score": round(float(np.mean(top_k_scores)), 8) if top_k_scores else 0,
            "top_k_mean_confidence": round(float(np.mean(top_k_confs)), 6) if top_k_confs else 0,
            "mean_confidence": day.mean_confidence,
            "score_quantiles": day.score_quantiles,
        })
    return stats


# ---------------------------------------------------------------------------
# Subdirectory file generation
# ---------------------------------------------------------------------------
def build_subdirectory_files(
    out_dir: Path,
    all_scores: list[DailyScores],
    performance: PerformanceSummary,
    equity_curve: list[dict[str, Any]],
    confidence: dict[str, Any],
    system_info: dict[str, Any],
    trading_config: dict[str, Any],
    trading_logs: dict[str, Any],
    pretty: bool = False,
) -> None:
    """Generate all categorized subdirectory files."""
    # meta/
    write_json(system_info, out_dir / "meta" / "system_info.json", pretty)
    write_json(trading_config, out_dir / "meta" / "trading_config.json", pretty)
    log("  ✓", "meta/ — system_info.json, trading_config.json")

    # signals/
    latest_day = all_scores[-1]
    write_json(
        _build_daily_signal_file(latest_day),
        out_dir / "signals" / "latest.json",
        pretty,
    )
    for day in all_scores:
        write_json(
            _build_daily_signal_file(day),
            out_dir / "signals" / "history" / f"signals_{day.date}.json",
            pretty,
        )
    log("  ✓", f"signals/ — latest.json + {len(all_scores)} history files")

    # performance/
    write_json(asdict(performance), out_dir / "performance" / "summary.json", pretty)
    write_json(
        _build_daily_stats_summary(all_scores),
        out_dir / "performance" / "daily_stats.json",
        pretty,
    )
    write_json(equity_curve, out_dir / "performance" / "equity_curve.json", pretty)
    log("  ✓", "performance/ — summary.json, daily_stats.json, equity_curve.json")

    # risk/
    risk_data: dict[str, Any] = {"latest": trading_logs.get("latest_risk_state")}
    write_json(risk_data, out_dir / "risk" / "latest.json", pretty)
    write_json(confidence, out_dir / "risk" / "confidence_distribution.json", pretty)
    log("  ✓", "risk/ — latest.json, confidence_distribution.json")

    # trading/
    if trading_logs.get("latest_targets"):
        write_json(
            trading_logs["latest_targets"],
            out_dir / "trading" / "latest_targets.json",
            pretty,
        )
    if trading_logs.get("latest_orders"):
        write_json(
            trading_logs["latest_orders"],
            out_dir / "trading" / "latest_orders.json",
            pretty,
        )
    if trading_logs.get("execution_history"):
        write_json(
            trading_logs["execution_history"],
            out_dir / "trading" / "execution_history.json",
            pretty,
        )
    log("  ✓", "trading/ — latest_targets, latest_orders, execution_history")


# ---------------------------------------------------------------------------
# Master JSON assembly
# ---------------------------------------------------------------------------
def build_master_json(
    all_scores: list[DailyScores],
    performance: PerformanceSummary,
    equity_curve: list[dict[str, Any]],
    confidence: dict[str, Any],
    system_info: dict[str, Any],
    trading_config: dict[str, Any],
    trading_logs: dict[str, Any],
    system_health: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the combined master dashboard_data.json."""
    latest = all_scores[-1] if all_scores else None

    return {
        "_meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generator": "scripts/export_dashboard_data.py",
            "version": "1.0.0",
            "description": "FullStackAutoQuant Dashboard — comprehensive dataset for live visualization",
        },
        "system_health": system_health,
        "system_info": system_info,
        "trading_config": trading_config,
        "performance": asdict(performance) if performance.total_trading_days > 0 else {},
        "equity_curve": equity_curve,
        "latest_signals": {
            "date": latest.date if latest else None,
            "top_k": latest.instruments[:TOP_K_DISPLAY] if latest else [],
            "statistics": {
                "total_count": latest.total_count,
                "positive_count": latest.positive_count,
                "negative_count": latest.negative_count,
                "mean_score": latest.mean_score,
                "mean_confidence": latest.mean_confidence,
                "score_quantiles": latest.score_quantiles,
            } if latest else {},
            "all_instruments": latest.instruments if latest else [],
        },
        "confidence_analysis": confidence,
        "trading": trading_logs,
        "daily_stats": _build_daily_stats_summary(all_scores),
    }
