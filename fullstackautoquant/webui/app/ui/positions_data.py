"""Position data conversion: parsing, normalizing, comparing position DataFrames."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from app.database import Database

_POSITION_ALIASES: dict[str, str] = {
    "symbol": "symbol",
    "Symbol": "symbol",
    "Symbol (symbol)": "symbol",
    "Stock Code": "symbol",
    "code": "symbol",
    "qty": "qty",
    "Volume": "qty",
    "Qty": "qty",
    "Position Qty (qty)": "qty",
    "volume": "qty",
    "shares": "qty",
    "cost": "cost_price",
    "Cost Price": "cost_price",
    "Cost Price (cost)": "cost_price",
    "cost_price": "cost_price",
    "market_value": "market_value",
    "Market Value": "market_value",
    "Market Value (value)": "market_value",
    "value": "market_value",
}


@dataclass(frozen=True)
class PositionDiffResult:
    """Describe differences between imported and current positions."""

    diff: pd.DataFrame


def rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename columns by mapping, keep original if key not found."""
    cols = {col: mapping.get(col, col) for col in df.columns}
    return df.rename(columns=cols)


def standardize_positions_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize broker import format to unified fields."""
    renamed: dict[str, str] = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in _POSITION_ALIASES:
            renamed[col] = _POSITION_ALIASES[key]
    normalized = df.rename(columns=renamed)
    required = ["symbol", "qty", "cost_price", "market_value"]
    for field in required:
        if field not in normalized.columns:
            normalized[field] = 0.0
    normalized = normalized[required]
    normalized["symbol"] = normalized["symbol"].astype(str).str.strip()
    numeric_cols = ["qty", "cost_price", "market_value"]
    for col in numeric_cols:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce").fillna(0.0)
    normalized = normalized[normalized["symbol"] != ""].reset_index(drop=True)
    return normalized


def load_positions_from_upload(uploaded) -> pd.DataFrame:
    """Parse position data from uploaded file."""
    name = uploaded.name.lower()
    uploaded.seek(0)
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        text = uploaded.getvalue().decode("utf-8")
        data = json.loads(text)
        if isinstance(data, dict) and "positions" in data:
            data = data["positions"]
        df = pd.DataFrame(data)
    return standardize_positions_dataframe(df)


def get_positions_dataframe(db: Database) -> pd.DataFrame:
    """Read current positions from database as DataFrame."""
    positions = db.get_positions()
    data = [
        {
            "symbol": p.symbol,
            "qty": float(p.qty or 0.0),
            "cost_price": float(p.cost_price or 0.0),
            "market_value": float(p.market_value or 0.0),
        }
        for p in positions
    ]
    df = pd.DataFrame(data)
    if df.empty:
        df = pd.DataFrame(columns=["symbol", "qty", "cost_price", "market_value"])
    return df


def compute_position_diff(current_df: pd.DataFrame, new_df: pd.DataFrame) -> PositionDiffResult:
    """Compare imported vs current positions."""
    merged = current_df.merge(new_df, on="symbol", how="outer", suffixes=("_current", "_new"))
    for col in [
        "qty_current",
        "cost_price_current",
        "market_value_current",
        "qty_new",
        "cost_price_new",
        "market_value_new",
    ]:
        if col not in merged.columns:
            merged[col] = 0.0
    merged = merged.fillna(0.0)
    merged["delta_qty"] = merged["qty_new"] - merged["qty_current"]
    merged["delta_value"] = merged["market_value_new"] - merged["market_value_current"]
    diff_df = merged.sort_values("symbol").reset_index(drop=True)
    return PositionDiffResult(diff=diff_df)


def fill_missing_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert columns to numeric and fill missing values."""
    converted = df.copy()
    for column in columns:
        converted[column] = pd.to_numeric(converted[column], errors="coerce").fillna(0.0)
    return converted


def to_payload(db: Database | None) -> tuple[list[dict[str, Any]], list[str]]:
    """Generate GM symbol payload for workflow."""
    if db is None:
        return [], []
    df = get_positions_dataframe(db)
    aggregated: dict[str, int] = {}
    skipped: list[str] = []
    for row in df.itertuples(index=False):
        qty = float(getattr(row, "qty", 0.0) or 0.0)
        shares = int(round(qty))
        if shares <= 0:
            continue
        raw_symbol = getattr(row, "symbol", "")
        gm_symbol = to_gm_symbol(raw_symbol)
        if gm_symbol is None:
            skipped.append(str(raw_symbol))
            continue
        aggregated[gm_symbol] = aggregated.get(gm_symbol, 0) + shares
    payload = [{"symbol": sym, "shares": qty} for sym, qty in aggregated.items() if qty > 0]
    return payload, skipped


def normalize_symbols(symbols: Iterable[str]) -> list[str]:
    """Deduplicate and clean symbols."""
    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if symbol is None:
            continue
        sym = str(symbol).strip()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        normalized.append(sym)
    return normalized


def strip_exchange_prefix(symbol: str) -> str:
    """Remove exchange prefix SHSE./SZSE."""
    if symbol is None:
        return ""
    sym = str(symbol).strip().upper()
    if sym.startswith("SHSE.") or sym.startswith("SZSE."):
        return sym.split(".", 1)[1]
    return sym


def digits_only_symbol(symbol: str) -> str:
    """Numeric-only stock code for display (strip prefix and non-digits)."""
    if symbol is None:
        return ""
    sym = strip_exchange_prefix(symbol)
    digits = "".join(ch for ch in sym if ch.isdigit())
    if digits:
        return digits
    return "".join(ch for ch in str(symbol).strip() if ch.isdigit())


def to_gm_symbol(symbol: str) -> str | None:
    """Convert to GM symbol encoding."""
    if symbol is None:
        return None
    sym = str(symbol).strip().upper()
    if not sym:
        return None
    if sym.startswith("SHSE.") or sym.startswith("SZSE."):
        return sym
    if sym.startswith("SH") and len(sym) == 8 and sym[2:].isdigit():
        return f"SHSE.{sym[2:]}"
    if sym.startswith("SZ") and len(sym) == 8 and sym[2:].isdigit():
        return f"SZSE.{sym[2:]}"
    if sym.isdigit() and len(sym) == 6:
        prefix = "SHSE." if sym.startswith("6") else "SZSE."
        return f"{prefix}{sym}"
    return None
