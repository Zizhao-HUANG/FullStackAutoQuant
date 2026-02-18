from __future__ import annotations

from math import isfinite

import numpy as np
import pandas as pd

from .components.records import BacktestSummary

ANNUALIZATION_DAYS = 252
MIN_PERIODS_FOR_ANNUALIZATION = 21
RISK_FREE_ANNUAL = 0.02


def compute_summary(equity_df: pd.DataFrame) -> BacktestSummary:
    if equity_df.empty:
        return BacktestSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    equity_series = equity_df["equity"].astype(float)
    total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1.0
    daily_returns = equity_df["daily_return"].replace([np.inf, -np.inf], np.nan).astype(float)
    daily_returns = daily_returns.fillna(0.0)
    periods = int(daily_returns.count())
    cumulative = float(np.prod(1.0 + daily_returns)) if periods else 1.0
    annualized = float("nan")
    if periods >= MIN_PERIODS_FOR_ANNUALIZATION and cumulative > 0:
        annualized = cumulative ** (ANNUALIZATION_DAYS / periods) - 1.0
    volatility = float("nan")
    if periods > 1 and periods >= MIN_PERIODS_FOR_ANNUALIZATION:
        volatility = daily_returns.std(ddof=1) * np.sqrt(ANNUALIZATION_DAYS)
    mean_daily = daily_returns.mean() if periods else 0.0
    max_dd = max_drawdown(equity_df["equity"].to_numpy())
    risk_free_daily = RISK_FREE_ANNUAL / ANNUALIZATION_DAYS
    sharpe = float("nan")
    if periods >= MIN_PERIODS_FOR_ANNUALIZATION and volatility and isfinite(volatility) and volatility > 1e-9:
        excess_return = (mean_daily - risk_free_daily) * ANNUALIZATION_DAYS
        sharpe = excess_return / volatility
    calmar = float("nan")
    if periods >= MIN_PERIODS_FOR_ANNUALIZATION and max_dd < 0 and isfinite(annualized):
        calmar = annualized / abs(max_dd)

    return BacktestSummary(
        total_return=float(total_return),
        annualized_return=float(annualized) if isfinite(annualized) else float("nan"),
        max_drawdown=float(max_dd),
        sharpe=float(sharpe) if isfinite(sharpe) else float("nan"),
        volatility=float(volatility) if isfinite(volatility) else float("nan"),
        calmar=float(calmar) if isfinite(calmar) else float("nan"),
    )


def max_drawdown(equity: np.ndarray) -> float:
    peak = -np.inf
    max_dd = 0.0
    for val in equity:
        peak = max(peak, val)
        if peak <= 0:
            continue
        drawdown = val / peak - 1.0
        max_dd = min(max_dd, drawdown)
    return float(max_dd)


def enrich_equity_curve(equity_df: pd.DataFrame) -> pd.DataFrame:
    if equity_df.empty:
        return equity_df
    equity_df = equity_df.sort_index()
    equity_df["daily_return"] = equity_df["equity"].pct_change().fillna(0.0)
    return equity_df
