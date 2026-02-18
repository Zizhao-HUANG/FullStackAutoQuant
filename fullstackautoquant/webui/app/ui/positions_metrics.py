"""Position metrics: returns, weights, P&L derived fields."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd
from app.quotes import apply_quotes_to_market_value
from app.ui.positions_data import digits_only_symbol


@dataclass(frozen=True)
class PositionMetrics:
    """Encapsulates position metrics and display table."""

    raw: pd.DataFrame
    display: pd.DataFrame
    totals: dict[str, float]


class PositionMetricsCalculator:
    """Compute position table derived metrics."""

    def __init__(self, quotes: Mapping[str, Mapping[str, object]]):
        self._quotes = quotes

    def compute(self, df: pd.DataFrame) -> PositionMetrics:
        enriched = self._compute_indicators(df)
        display = self._build_display(enriched)
        totals = self._summarize(enriched)
        return PositionMetrics(raw=enriched, display=display, totals=totals)

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        enriched = df.copy()
        enriched = apply_quotes_to_market_value(enriched, self._quotes)
        enriched = _fill_numeric(enriched, ["qty", "cost_price", "market_value"])

        price_series = enriched["symbol"].map(lambda x: self._quotes.get(x, {}).get("price"))
        price_numeric = pd.to_numeric(price_series, errors="coerce")
        pre_close = enriched["symbol"].map(lambda x: self._quotes.get(x, {}).get("pre_close"))
        pre_close_numeric = pd.to_numeric(pre_close, errors="coerce")

        enriched["floating_pnl"] = (
            enriched["market_value"] - enriched["cost_price"] * enriched["qty"]
        ).round(4)
        cost_mask = enriched["cost_price"] > 0
        enriched.loc[cost_mask, "return_pct"] = (
            enriched.loc[cost_mask, "market_value"]
            / (enriched.loc[cost_mask, "cost_price"] * enriched.loc[cost_mask, "qty"])
            - 1
        ).round(6)
        enriched["return_pct"].fillna(0.0, inplace=True)
        enriched["daily_pnl"] = (price_numeric - pre_close_numeric) * enriched["qty"]
        enriched["daily_pnl"] = enriched["daily_pnl"].fillna(0.0).round(4)

        total_market_value = enriched["market_value"].sum()
        enriched["weight"] = (
            enriched["market_value"] / total_market_value if total_market_value else 0.0
        )
        return enriched

    def _build_display(self, df: pd.DataFrame) -> pd.DataFrame:
        display_df = df.copy()
        display_df["Name"] = display_df["symbol"].map(lambda x: self._quotes.get(x, {}).get("name", ""))
        display_df["Live Price"] = _to_numeric(display_df["symbol"], self._quotes, "price")
        column_mapping = {
            "symbol": "Symbol",
            "qty": "Qty",
            "cost_price": "Cost Price",
            "market_value": "Live Value",
            "floating_pnl": "Floating P&L",
            "daily_pnl": "Daily P&L",
            "weight": "Weight",
            "return_pct": "Return %",
        }
        display_df = display_df.rename(columns=column_mapping)
        if "Symbol" in display_df.columns:
            display_df["Symbol"] = display_df["Symbol"].map(digits_only_symbol)

        display_df["Qty"] = pd.to_numeric(display_df["Qty"], errors="coerce").fillna(0.0)
        display_df["Cost Price"] = pd.to_numeric(display_df["Cost Price"], errors="coerce").fillna(0.0)
        display_df["Live Value"] = pd.to_numeric(display_df["Live Value"], errors="coerce").fillna(0.0)
        display_df["Live Price"] = display_df["Live Price"].round(4)
        display_df["Floating P&L"] = display_df["Floating P&L"].map(lambda v: round(float(v), 2) if pd.notna(v) else 0.0)
        display_df["Daily P&L"] = display_df["Daily P&L"].map(lambda v: round(float(v), 2) if pd.notna(v) else 0.0)
        display_df["Weight"] = display_df["Weight"].map(
            lambda v: f"{float(v):.2%}" if pd.notna(v) and v not in ("", None) else ""
        )
        display_df["Return %"] = display_df["Return %"].map(
            lambda v: f"{float(v):.2%}" if pd.notna(v) and v not in ("", None) else ""
        )
        return display_df

    def _summarize(self, df: pd.DataFrame) -> dict[str, float]:
        summary_total_market_value = float(df["market_value"].sum())
        summary_total_cost = float((df["cost_price"] * df["qty"]).sum())
        summary_total_pnl = summary_total_market_value - summary_total_cost
        summary_daily_pnl = float(df["daily_pnl"].sum())
        return {
            "market_value": summary_total_market_value,
            "cost": summary_total_cost,
            "pnl": summary_total_pnl,
            "daily_pnl": summary_daily_pnl,
        }


def _fill_numeric(df: pd.DataFrame, columns) -> pd.DataFrame:
    converted = df.copy()
    for column in columns:
        converted[column] = pd.to_numeric(converted[column], errors="coerce").fillna(0.0)
    return converted


def _to_numeric(symbols: pd.Series, quotes: Mapping[str, Mapping[str, object]], key: str) -> pd.Series:
    return pd.to_numeric(symbols.map(lambda x: quotes.get(x, {}).get(key)), errors="coerce").fillna(0.0)
