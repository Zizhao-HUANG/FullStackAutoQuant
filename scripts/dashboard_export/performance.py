"""Performance metrics computation and equity curve generation.

Computes aggregated signal-quality metrics (Sharpe, IR, max drawdown, etc.)
and builds a simulated equity curve from daily top-K predicted scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scripts.dashboard_export.constants import ANNUALIZATION_DAYS, RISK_FREE_ANNUAL, TOP_K_DISPLAY
from scripts.dashboard_export.scores import DailyScores


@dataclass
class PerformanceSummary:
    """Aggregated performance metrics across all available days."""

    total_trading_days: int
    first_date: str
    last_date: str
    # Signal quality metrics (simulated long-short)
    mean_daily_long_short_spread: float
    long_short_sharpe: float
    long_short_information_ratio: float
    # Top-K portfolio simulation (equal-weight top-30)
    topk_mean_daily_score: float
    topk_cumulative_score: float
    topk_annualized_score: float
    topk_score_volatility: float
    topk_sharpe_ratio: float
    topk_max_drawdown: float
    topk_calmar_ratio: float
    topk_win_rate: float
    # Score distribution stability
    mean_spread_top_bottom: float
    mean_positive_ratio: float
    avg_confidence: float


def compute_performance(all_scores: list[DailyScores]) -> PerformanceSummary:
    """Compute aggregated performance metrics from daily scores."""
    if not all_scores:
        return PerformanceSummary(
            total_trading_days=0, first_date="", last_date="",
            mean_daily_long_short_spread=0, long_short_sharpe=0,
            long_short_information_ratio=0, topk_mean_daily_score=0,
            topk_cumulative_score=0, topk_annualized_score=0,
            topk_score_volatility=0, topk_sharpe_ratio=0,
            topk_max_drawdown=0, topk_calmar_ratio=0, topk_win_rate=0,
            mean_spread_top_bottom=0, mean_positive_ratio=0, avg_confidence=0,
        )

    n_days = len(all_scores)
    first_date = all_scores[0].date
    last_date = all_scores[-1].date

    # Daily metrics
    daily_topk_scores: list[float] = []
    daily_spreads: list[float] = []
    daily_pos_ratios: list[float] = []
    daily_confidences: list[float] = []

    for day in all_scores:
        top_k = [s["score"] for s in day.instruments[:TOP_K_DISPLAY]]
        bottom_k = [s["score"] for s in day.instruments[-TOP_K_DISPLAY:]]
        topk_mean = float(np.mean(top_k)) if top_k else 0.0
        bottomk_mean = float(np.mean(bottom_k)) if bottom_k else 0.0
        daily_topk_scores.append(topk_mean)
        daily_spreads.append(topk_mean - bottomk_mean)
        daily_pos_ratios.append(day.positive_count / max(day.total_count, 1))
        daily_confidences.append(day.mean_confidence)

    topk_arr = np.array(daily_topk_scores)
    spread_arr = np.array(daily_spreads)

    # Sharpe-like computation on predicted scores
    rf_daily = RISK_FREE_ANNUAL / ANNUALIZATION_DAYS
    topk_mean = float(np.mean(topk_arr))
    topk_std = float(np.std(topk_arr, ddof=1)) if n_days > 1 else 1e-9
    topk_sharpe = (
        (topk_mean - rf_daily) * np.sqrt(ANNUALIZATION_DAYS) / max(topk_std, 1e-9)
    )

    # Simulated cumulative score (sum of daily top-K mean predicted returns)
    topk_cumulative = float(np.sum(topk_arr))
    topk_annualized = topk_cumulative * (ANNUALIZATION_DAYS / max(n_days, 1))

    # Max drawdown on cumulative top-K score curve
    cumsum = np.cumsum(topk_arr)
    peak = np.maximum.accumulate(cumsum)
    drawdown = cumsum - peak
    max_dd = float(np.min(drawdown)) if len(drawdown) else 0.0

    calmar = topk_annualized / abs(max_dd) if max_dd < -1e-9 else float("inf")
    win_rate = float(np.mean(topk_arr > 0)) if n_days else 0.0

    # Long-short spread metrics
    spread_mean = float(np.mean(spread_arr))
    spread_std = float(np.std(spread_arr, ddof=1)) if n_days > 1 else 1e-9
    ls_sharpe = spread_mean * np.sqrt(ANNUALIZATION_DAYS) / max(spread_std, 1e-9)
    ls_ir = spread_mean / max(spread_std, 1e-9)

    return PerformanceSummary(
        total_trading_days=n_days,
        first_date=first_date,
        last_date=last_date,
        mean_daily_long_short_spread=round(spread_mean, 8),
        long_short_sharpe=round(float(ls_sharpe), 4),
        long_short_information_ratio=round(float(ls_ir), 4),
        topk_mean_daily_score=round(topk_mean, 8),
        topk_cumulative_score=round(topk_cumulative, 6),
        topk_annualized_score=round(topk_annualized, 6),
        topk_score_volatility=round(topk_std, 8),
        topk_sharpe_ratio=round(float(topk_sharpe), 4),
        topk_max_drawdown=round(max_dd, 6),
        topk_calmar_ratio=round(float(calmar), 4) if np.isfinite(calmar) else None,
        topk_win_rate=round(win_rate, 4),
        mean_spread_top_bottom=round(spread_mean, 8),
        mean_positive_ratio=round(float(np.mean(daily_pos_ratios)), 4),
        avg_confidence=round(float(np.mean(daily_confidences)), 6),
    )


def build_equity_curve(all_scores: list[DailyScores]) -> list[dict[str, Any]]:
    """Build a simulated equity curve from daily top-K predicted scores."""
    if not all_scores:
        return []

    equity = 1.0
    peak = 1.0
    curve: list[dict[str, Any]] = []

    for day in all_scores:
        top_k = [s["score"] for s in day.instruments[:TOP_K_DISPLAY]]
        daily_return = float(np.mean(top_k)) if top_k else 0.0
        equity *= (1.0 + daily_return)
        peak = max(peak, equity)
        dd = (equity / peak) - 1.0

        curve.append({
            "date": day.date,
            "equity": round(equity, 6),
            "daily_return": round(daily_return, 8),
            "cumulative_return": round(equity - 1.0, 6),
            "drawdown": round(dd, 6),
        })

    return curve
