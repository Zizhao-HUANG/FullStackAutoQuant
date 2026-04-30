"""Performance metrics computation and equity curve generation.

Computes aggregated signal-quality metrics (Sharpe, IR, max drawdown, etc.)
and builds an equity curve from daily data.

Key design:
  - **Real portfolio data** (preferred): When GM Trade API position
    snapshots are available (via ``portfolio.py``), the equity curve
    and portfolio-level metrics (Sharpe, win rate, max drawdown) are
    computed from actual daily NAV changes.
  - **Fallback**: When no real data exists, uses universe-wide mean
    prediction scores as a proxy.  This is clearly labeled as
    ``source: simulated_signal`` in the output.
  - Top-K metrics are always tracked for alpha signal characterisation.
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
    # Top-K alpha signal tracking
    topk_mean_daily_score: float
    topk_cumulative_score: float
    topk_annualized_score: float
    # Portfolio simulation (based on universe-wide mean scores)
    topk_score_volatility: float
    topk_sharpe_ratio: float
    topk_max_drawdown: float
    topk_calmar_ratio: float
    topk_win_rate: float
    # Score distribution stability
    mean_spread_top_bottom: float
    mean_positive_ratio: float
    avg_confidence: float
    # Latest day specific metrics (change every day)
    latest_topk_alpha: float
    latest_daily_return: float
    latest_confidence: float
    latest_positive_ratio: float
    cumulative_return: float


def compute_performance(all_scores: list[DailyScores]) -> PerformanceSummary:
    """Compute aggregated performance metrics from daily scores.

    Uses universe-wide mean scores for portfolio simulation (Sharpe, win rate,
    max drawdown) so that metrics are realistic and vary meaningfully each day.
    """
    if not all_scores:
        return PerformanceSummary(
            total_trading_days=0, first_date="", last_date="",
            mean_daily_long_short_spread=0, long_short_sharpe=0,
            long_short_information_ratio=0, topk_mean_daily_score=0,
            topk_cumulative_score=0, topk_annualized_score=0,
            topk_score_volatility=0, topk_sharpe_ratio=0,
            topk_max_drawdown=0, topk_calmar_ratio=0, topk_win_rate=0,
            mean_spread_top_bottom=0, mean_positive_ratio=0, avg_confidence=0,
            latest_topk_alpha=0, latest_daily_return=0,
            latest_confidence=0, latest_positive_ratio=0,
            cumulative_return=0,
        )

    n_days = len(all_scores)
    first_date = all_scores[0].date
    last_date = all_scores[-1].date
    latest = all_scores[-1]

    # Daily metrics
    daily_topk_scores: list[float] = []
    daily_universe_returns: list[float] = []
    daily_spreads: list[float] = []
    daily_pos_ratios: list[float] = []
    daily_confidences: list[float] = []

    for day in all_scores:
        top_k = [s["score"] for s in day.instruments[:TOP_K_DISPLAY]]
        bottom_k = [s["score"] for s in day.instruments[-TOP_K_DISPLAY:]]
        topk_mean = float(np.mean(top_k)) if top_k else 0.0
        bottomk_mean = float(np.mean(bottom_k)) if bottom_k else 0.0
        daily_topk_scores.append(topk_mean)
        daily_universe_returns.append(day.mean_score)
        daily_spreads.append(topk_mean - bottomk_mean)
        daily_pos_ratios.append(day.positive_count / max(day.total_count, 1))
        daily_confidences.append(day.mean_confidence)

    topk_arr = np.array(daily_topk_scores)
    universe_arr = np.array(daily_universe_returns)
    spread_arr = np.array(daily_spreads)

    rf_daily = RISK_FREE_ANNUAL / ANNUALIZATION_DAYS

    # ── Top-K alpha tracking (characterises signal strength) ──────
    topk_mean = float(np.mean(topk_arr))
    topk_cumulative = float(np.sum(topk_arr))
    topk_annualized = topk_cumulative * (ANNUALIZATION_DAYS / max(n_days, 1))

    # ── Portfolio metrics from real GM Trade API data only ──────────
    from scripts.dashboard_export.portfolio import compute_real_performance

    real_perf = compute_real_performance()

    if real_perf is not None:
        sharpe = real_perf["sharpe_ratio"]
        win_rate = real_perf["win_rate"]
        max_dd = real_perf["max_drawdown"]
        cumulative_return = real_perf["cumulative_return"]
        calmar = real_perf.get("calmar_ratio") or float("inf")
        latest_daily_return = real_perf["latest_daily_return"]
    else:
        # No real portfolio data yet — show zeros, never fake it
        sharpe = 0.0
        win_rate = 0.0
        max_dd = 0.0
        cumulative_return = 0.0
        calmar = 0.0
        latest_daily_return = 0.0

    uni_std_for_vol = float(np.std(universe_arr, ddof=1)) if n_days > 1 else 1e-9

    # ── Long-short spread metrics ─────────────────────────────────
    spread_mean = float(np.mean(spread_arr))
    spread_std = float(np.std(spread_arr, ddof=1)) if n_days > 1 else 1e-9
    ls_sharpe = spread_mean * np.sqrt(ANNUALIZATION_DAYS) / max(spread_std, 1e-9)
    ls_ir = spread_mean / max(spread_std, 1e-9)

    # ── Latest day specific (changes every single day) ────────────
    latest_topk = [s["score"] for s in latest.instruments[:TOP_K_DISPLAY]]
    latest_topk_alpha = float(np.mean(latest_topk)) if latest_topk else 0.0
    latest_pos_ratio = latest.positive_count / max(latest.total_count, 1)

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
        topk_score_volatility=round(uni_std_for_vol, 8),
        topk_sharpe_ratio=round(float(sharpe), 4),
        topk_max_drawdown=round(max_dd, 6),
        topk_calmar_ratio=round(float(calmar), 4) if np.isfinite(calmar) else None,
        topk_win_rate=round(win_rate, 4),
        mean_spread_top_bottom=round(spread_mean, 8),
        mean_positive_ratio=round(float(np.mean(daily_pos_ratios)), 4),
        avg_confidence=round(float(np.mean(daily_confidences)), 6),
        # Per-day values
        latest_topk_alpha=round(latest_topk_alpha, 8),
        latest_daily_return=round(latest_daily_return, 8),
        latest_confidence=round(latest.mean_confidence, 6),
        latest_positive_ratio=round(latest_pos_ratio, 4),
        cumulative_return=round(cumulative_return, 6),
    )


def build_equity_curve(all_scores: list[DailyScores]) -> list[dict[str, Any]]:
    """Build equity curve from real GM Trade API portfolio snapshots.

    Returns actual daily NAV-based returns.  Each point includes nav,
    total_market_value, and available_cash.

    Returns an empty list if no portfolio snapshots are available yet.
    """
    from scripts.dashboard_export.portfolio import build_real_equity_curve

    curve = build_real_equity_curve()
    for pt in curve:
        pt["source"] = "real_portfolio"
    return curve
