from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd


def normalize_symbol(symbol: object) -> str:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return ""
    if sym.startswith("SZSE.") or sym.startswith("SHSE."):
        return sym
    if sym.endswith(".SZ") and len(sym) >= 9:
        return f"SZSE.{sym[:-3]}"
    if sym.endswith(".SH") and len(sym) >= 9:
        return f"SHSE.{sym[:-3]}"
    if sym.startswith("SZ") and len(sym) == 8 and sym[2:].isdigit():
        return f"SZSE.{sym[2:]}"
    if sym.startswith("SH") and len(sym) == 8 and sym[2:].isdigit():
        return f"SHSE.{sym[2:]}"
    if sym.startswith("0") or sym.startswith("3") or sym.startswith("2"):
        if sym.isdigit() and len(sym) == 6:
            return f"SZSE.{sym}"
    if sym.isdigit() and len(sym) == 6:
        return f"SHSE.{sym}"
    return sym


@dataclass
class ApplyResult:
    positions: list[dict[str, Any]]
    sold: list[dict[str, Any]]
    cash_delta: float
    buy_notional: float
    sell_notional: float
    processed: int
    warnings: list[str]


def _to_dataframe(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = 0.0
    return df[columns]


def apply_orders(
    current_positions: Sequence[dict[str, Any]] | pd.DataFrame,
    orders: Sequence[dict[str, Any]] | pd.DataFrame,
) -> ApplyResult:
    if isinstance(current_positions, pd.DataFrame):
        positions_df = current_positions.copy()
    else:
        positions_df = _to_dataframe(
            current_positions, ["symbol", "qty", "cost_price", "market_value"]
        )
    if isinstance(orders, pd.DataFrame):
        orders_df = orders.copy()
    else:
        orders_df = _to_dataframe(orders, ["symbol", "side", "volume", "price"])

    positions_df["symbol"] = positions_df["symbol"].apply(normalize_symbol)
    positions_df["qty"] = pd.to_numeric(positions_df["qty"], errors="coerce").fillna(0.0)
    positions_df["cost_price"] = pd.to_numeric(positions_df["cost_price"], errors="coerce").fillna(
        0.0
    )
    positions_map: dict[str, dict[str, Any]] = {}
    for row in positions_df.itertuples(index=False):
        symbol = str(row.symbol)
        if not symbol:
            continue
        qty = float(getattr(row, "qty", 0.0) or 0.0)
        cost_price = float(getattr(row, "cost_price", 0.0) or 0.0)
        market_value = float(getattr(row, "market_value", qty * cost_price) or 0.0)
        if symbol in positions_map:
            positions_map[symbol]["qty"] += qty
        else:
            positions_map[symbol] = {
                "symbol": symbol,
                "qty": qty,
                "cost_price": cost_price,
                "market_value": market_value,
            }

    orders_df["symbol"] = orders_df["symbol"].apply(normalize_symbol)
    orders_df["side"] = orders_df["side"].astype(str).str.upper()
    orders_df["volume"] = pd.to_numeric(orders_df["volume"], errors="coerce").fillna(0.0)
    orders_df["price"] = pd.to_numeric(orders_df["price"], errors="coerce").fillna(0.0)

    sold_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    cash_delta = 0.0
    buy_notional = 0.0
    sell_notional = 0.0
    processed = 0

    for row in orders_df.itertuples(index=False):
        symbol = str(row.symbol)
        side = str(row.side).upper()
        volume = float(getattr(row, "volume", 0.0) or 0.0)
        price = float(getattr(row, "price", 0.0) or 0.0)
        if not symbol or volume <= 0:
            continue
        processed += 1
        if side == "BUY":
            pos = positions_map.get(symbol)
            if pos is None:
                pos = {"symbol": symbol, "qty": 0.0, "cost_price": 0.0, "market_value": 0.0}
            existing_qty = float(pos.get("qty", 0.0))
            new_qty = existing_qty + volume
            if price > 0:
                buy_notional += price * volume
                cash_delta -= price * volume
                if new_qty > 0:
                    if existing_qty > 0:
                        weighted = (
                            existing_qty * pos.get("cost_price", 0.0) + price * volume
                        ) / new_qty
                        pos["cost_price"] = weighted
                    else:
                        pos["cost_price"] = price
            pos["qty"] = new_qty
            pos["market_value"] = pos.get("cost_price", 0.0) * new_qty
            positions_map[symbol] = pos
        elif side == "SELL":
            pos = positions_map.get(symbol)
            if pos is None:
                warnings.append(
                    f"Order {symbol} SELL {volume} not found in current positions, ignored."
                )
                continue
            existing_qty = float(pos.get("qty", 0.0))
            if existing_qty <= 0:
                warnings.append(f"Order {symbol} SELL {volume} current quantity is 0, ignored.")
                continue
            sell_qty = min(existing_qty, volume)
            if price > 0:
                sell_notional += price * sell_qty
                cash_delta += price * sell_qty
            remaining = existing_qty - sell_qty
            pos["qty"] = remaining
            pos["market_value"] = pos.get("cost_price", 0.0) * remaining
            if remaining <= 0:
                sold_records.append(
                    {
                        "symbol": symbol,
                        "qty": sell_qty,
                        "cost_price": float(pos.get("cost_price", 0.0) or 0.0),
                    }
                )
                positions_map.pop(symbol, None)
            else:
                positions_map[symbol] = pos
            if sell_qty < volume:
                warnings.append(
                    f"Order {symbol} SELL {volume} exceeds current quantity {existing_qty}, only deducting {sell_qty}."
                )
        else:
            warnings.append(f"Unknown side {side}, Order {symbol} skipped.")

    updated_positions: list[dict[str, Any]] = []
    for symbol in sorted(positions_map):
        pos = positions_map[symbol]
        qty = float(pos.get("qty", 0.0) or 0.0)
        cost_price = float(pos.get("cost_price", 0.0) or 0.0)
        market_value = float(qty * cost_price)
        updated_positions.append(
            {
                "symbol": symbol,
                "qty": qty,
                "cost_price": cost_price,
                "market_value": market_value,
            }
        )

    return ApplyResult(
        positions=updated_positions,
        sold=sold_records,
        cash_delta=cash_delta,
        buy_notional=buy_notional,
        sell_notional=sell_notional,
        processed=processed,
        warnings=warnings,
    )


def cap_buy_orders_by_cash(
    current_positions: Sequence[dict[str, Any]] | pd.DataFrame,
    orders: Sequence[dict[str, Any]] | pd.DataFrame,
    available_cash: float,
    lot_size: int = 1,
) -> tuple[pd.DataFrame, list[str]]:
    """Limit buy order count by available capital, return adjusted orders and trimmed symbols."""

    lot = max(1, int(lot_size or 1))
    cash = max(0.0, float(available_cash or 0.0))

    if isinstance(current_positions, pd.DataFrame):
        positions_df = current_positions.copy()
    else:
        positions_df = _to_dataframe(
            current_positions, ["symbol", "qty", "cost_price", "market_value"]
        )
    positions_df["symbol"] = positions_df["symbol"].apply(normalize_symbol)
    positions_df["qty"] = pd.to_numeric(positions_df["qty"], errors="coerce").fillna(0.0)
    qty_lookup = {
        str(row.symbol): float(getattr(row, "qty", 0.0) or 0.0)
        for row in positions_df.itertuples(index=False)
        if str(row.symbol)
    }

    if isinstance(orders, pd.DataFrame):
        orders_df = orders.copy()
    else:
        orders_df = _to_dataframe(orders, ["symbol", "side", "volume", "price"])

    orders_df["symbol"] = orders_df["symbol"].apply(normalize_symbol)
    orders_df["side"] = orders_df["side"].astype(str).str.upper()
    orders_df["volume"] = pd.to_numeric(orders_df["volume"], errors="coerce").fillna(0.0)
    orders_df["price"] = pd.to_numeric(orders_df["price"], errors="coerce").fillna(0.0)

    sell_cash = 0.0
    for row in orders_df.itertuples(index=False):
        if row.side != "SELL" or row.price <= 0:
            continue
        symbol = str(row.symbol)
        existing_qty = qty_lookup.get(symbol, 0.0)
        if existing_qty <= 0:
            continue
        sell_qty = min(existing_qty, float(getattr(row, "volume", 0.0) or 0.0))
        if sell_qty <= 0:
            continue
        sell_cash += row.price * sell_qty

    remaining_cash = cash + sell_cash
    trimmed_symbols: list[str] = []
    new_volumes: list[float] = []

    for row in orders_df.itertuples(index=False):
        volume = float(getattr(row, "volume", 0.0) or 0.0)
        price = float(getattr(row, "price", 0.0) or 0.0)
        if row.side != "BUY" or volume <= 0 or price <= 0:
            new_volumes.append(volume)
            continue
        affordable = remaining_cash / price
        if lot > 1:
            affordable = math.floor(affordable / lot) * lot
        else:
            affordable = math.floor(affordable)
        affordable = max(0.0, affordable)
        capped = min(volume, affordable)
        if capped + 1e-6 < volume:
            trimmed_symbols.append(str(row.symbol))
        new_volumes.append(capped)
        remaining_cash -= capped * price

    capped_df = orders_df.copy()
    capped_df["volume"] = new_volumes
    return capped_df, trimmed_symbols
