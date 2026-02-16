"""Generate trade plan from strategy targets and current positions."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .database import Database


@dataclass
class PlanResult:
    date: dt.date
    table: pd.DataFrame
    fallback_price: bool


def generate_plan(
    db: Database,
    targets: pd.DataFrame,
    prices: Dict[str, float],
    fallback_price: Optional[float] = None,
    fallback_price_map: Optional[Dict[str, float]] = None,
) -> PlanResult:
    positions = db.get_positions()
    pos_df = pd.DataFrame([
        {
            "symbol": p.symbol,
            "qty_current": p.qty,
            "market_value_current": p.market_value,
        }
        for p in positions
    ])

    if pos_df.empty:
        pos_df = pd.DataFrame(columns=["symbol", "qty_current", "market_value_current"])

    targets_df = targets.copy()
    if "symbol" not in targets_df.columns:
        raise ValueError("targets is missing symbol column")
    targets_df["target_qty"] = pd.to_numeric(targets_df.get("target_qty", 0.0), errors="coerce").fillna(0.0)
    if "weight" not in targets_df.columns:
        targets_df["weight"] = 0.0

    merged = targets_df.merge(pos_df, on="symbol", how="outer")
    merged["target_qty"] = merged["target_qty"].fillna(0.0)
    merged["weight"] = merged["weight"].fillna(0.0)
    merged["qty_current"] = merged["qty_current"].fillna(0.0)
    merged["market_value_current"] = merged["market_value_current"].fillna(0.0)

    merged["delta_qty"] = merged["target_qty"] - merged["qty_current"]

    latest_prices = pd.Series(prices)
    merged["last_price"] = merged["symbol"].map(latest_prices)
    used_fallback = False
    if fallback_price_map:
        for symbol, price in fallback_price_map.items():
            mask = merged["symbol"].eq(symbol) & merged["last_price"].isna()
            if mask.any():
                merged.loc[mask, "last_price"] = float(price)
                used_fallback = True
    if fallback_price is not None:
        missing_mask = merged["last_price"].isna()
        if missing_mask.any():
            merged.loc[missing_mask, "last_price"] = float(fallback_price)
            used_fallback = True
    merged["last_price"] = merged["last_price"].fillna(0.0)
    merged["delta_value"] = merged["delta_qty"] * merged["last_price"]
    merged["target_value"] = merged["target_qty"] * merged["last_price"]
    merged["current_value"] = merged["qty_current"] * merged["last_price"]

    date = dt.date.today()
    result_rows = merged[[
        "symbol",
        "target_qty",
        "delta_qty",
        "qty_current",
        "last_price",
        "delta_value",
        "target_value",
        "current_value",
        "weight",
    ]]

    db.upsert_plans(result_rows.to_dict("records"), date)

    return PlanResult(
        date=date,
        table=result_rows,
        fallback_price=used_fallback,
    )

