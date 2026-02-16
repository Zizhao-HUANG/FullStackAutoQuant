"""Backtest utility functions: metrics, portfolio helpers, parameter overrides."""

from __future__ import annotations

import datetime as dt
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


def compute_annualized_return(daily_returns: Sequence[float], trading_days: int = 252) -> float:
    if not daily_returns:
        return 0.0
    arr = np.array(list(daily_returns), dtype=float)
    mean = np.nanmean(arr)
    return float((1 + mean) ** trading_days - 1)


def compute_volatility(daily_returns: Sequence[float], trading_days: int = 252) -> float:
    if len(daily_returns) <= 1:
        return 0.0
    arr = np.array(list(daily_returns), dtype=float)
    std = np.nanstd(arr, ddof=1)
    return float(std * np.sqrt(trading_days))


def compute_max_drawdown(equity_curve: Sequence[float]) -> float:
    peak = -np.inf
    max_dd = 0.0
    for val in equity_curve:
        peak = max(peak, val)
        if peak <= 0:
            continue
        dd = val / peak - 1.0
        max_dd = min(max_dd, dd)
    return float(max_dd)


def align_to_trading_days(
    calendar: List[dt.date],
    series: pd.Series,
    fill_method: str | None = "ffill",
) -> pd.Series:
    idx = pd.Index(calendar, name=series.index.name or "date")
    aligned = series.reindex(idx)
    if fill_method == "ffill":
        aligned = aligned.ffill()
    elif fill_method == "bfill":
        aligned = aligned.bfill()
    return aligned


__all__ = [
    "compute_annualized_return",
    "compute_volatility",
    "compute_max_drawdown",
    "align_to_trading_days",
]


