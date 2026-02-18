"""Backtest data structure definitions."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from math import isfinite
from typing import Any

import pandas as pd


@dataclass(slots=True)
class TradeRecord:
    date: dt.date
    symbol: str
    side: str
    volume: float
    price: float
    fee: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "symbol": self.symbol,
            "side": self.side,
            "volume": self.volume,
            "price": self.price,
            "fee": self.fee,
        }


@dataclass(slots=True)
class PositionSnapshot:
    date: dt.date
    symbol: str
    shares: float
    market_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "symbol": self.symbol,
            "shares": self.shares,
            "market_value": self.market_value,
        }


@dataclass(slots=True)
class BacktestSummary:
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe: float
    volatility: float
    calmar: float

    def to_dict(self) -> dict[str, float | None]:
        def _normalise(value: float) -> float | None:
            return value if isfinite(value) else None

        return {
            "total_return": _normalise(self.total_return),
            "annualized_return": _normalise(self.annualized_return),
            "max_drawdown": _normalise(self.max_drawdown),
            "sharpe": _normalise(self.sharpe),
            "volatility": _normalise(self.volatility),
            "calmar": _normalise(self.calmar),
        }


@dataclass(slots=True)
class DailyEquity:
    date: dt.date
    cash: float
    market_value: float
    equity: float
    daily_return: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "cash": self.cash,
            "market_value": self.market_value,
            "equity": self.equity,
            "daily_return": self.daily_return,
        }


@dataclass(slots=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    summary: BacktestSummary
    metadata: dict[str, Any]

    def with_metadata(self, extra: dict[str, Any]) -> BacktestResult:
        combined = dict(self.metadata)
        combined.update(extra)
        return BacktestResult(
            equity_curve=self.equity_curve,
            trades=self.trades,
            positions=self.positions,
            summary=self.summary,
            metadata=combined,
        )


@dataclass(slots=True)
class BacktestIntermediate:
    equities: list[DailyEquity]
    trades: list[TradeRecord]
    snapshots: list[PositionSnapshot]
    risk_records: list[dict[str, Any]]
    signal_records: list[dict[str, Any]]
    manual_decisions: list[dict[str, Any]]

    def equity_curve_df(self) -> pd.DataFrame:
        return (
            pd.DataFrame([rec.to_dict() for rec in self.equities]).set_index("date")
            if self.equities
            else pd.DataFrame()
        )

    def to_result(self, summary: BacktestSummary, metadata: dict[str, Any]) -> BacktestResult:
        equity_df = self.equity_curve_df()
        trades_df = (
            pd.DataFrame([rec.to_dict() for rec in self.trades]) if self.trades else pd.DataFrame()
        )
        positions_df = (
            pd.DataFrame([snap.to_dict() for snap in self.snapshots])
            if self.snapshots
            else pd.DataFrame()
        )
        result_metadata = {
            **metadata,
            "risk_records": self.risk_records,
            "signal_records": self.signal_records,
            "manual_decisions": self.manual_decisions,
        }
        return BacktestResult(equity_df, trades_df, positions_df, summary, result_metadata)
