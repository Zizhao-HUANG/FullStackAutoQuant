"""Market data handler: caching, refreshing, and market cap updates."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import pandas as pd
import streamlit as st

from app.ui.positions_data import normalize_symbols


_QUOTE_STATE_KEY = "positions_quotes"
_QUOTE_UPDATED_AT_KEY = "positions_quotes_updated_at"


@dataclass(frozen=True)
class QuoteFetchResult:
    """Return quote dict and update time."""

    quotes: Dict[str, Dict[str, object]]
    updated_at: Optional[dt.datetime]


def format_currency(value: float) -> str:
    """Format monetary display."""
    return f"{value:,.2f}" if pd.notna(value) else "-"


def fetch_quotes(
    symbols: Iterable[str],
    market,
    *,
    force: bool = False,
    show_warning: bool = False,
    recompute_missing: bool = False,
) -> QuoteFetchResult:
    """Fetch real-time prices from market service and write to session cache."""
    normalized = normalize_symbols(symbols)
    if not normalized:
        st.session_state.setdefault(_QUOTE_STATE_KEY, {})
        return QuoteFetchResult(quotes={}, updated_at=None)

    cached = st.session_state.get(_QUOTE_STATE_KEY, {})

    if market is None:
        if show_warning:
            st.warning("TuShare market service not configured, cannot fetch real-time prices.")
        quotes = {symbol: cached.get(symbol, {}) for symbol in normalized}
        return QuoteFetchResult(quotes=quotes, updated_at=None)

    need_refresh: list[str] = []
    for symbol in normalized:
        if force:
            need_refresh.append(symbol)
            continue
        cached_quote = cached.get(symbol)
        price_missing = cached_quote is None or cached_quote.get("price") in (None, "")
        name_missing = cached_quote is None or cached_quote.get("name") in (None, "")
        if symbol not in cached or price_missing or (recompute_missing and name_missing):
            need_refresh.append(symbol)

    quotes: Dict[str, Dict[str, object]] = {}
    if need_refresh or force:
        try:
            target_symbols = normalized if force else need_refresh
            quotes = market.get_realtime_quotes(target_symbols)
        except Exception as exc:  # noqa: BLE001
            if show_warning or not cached:
                st.warning(f"Real-time quote fetch failed:{exc}")
            quotes = {}

    updated = dict(cached)
    updated.update(quotes)
    st.session_state[_QUOTE_STATE_KEY] = updated

    updated_at: Optional[dt.datetime] = None
    if quotes:
        updated_at = dt.datetime.now()
        st.session_state[_QUOTE_UPDATED_AT_KEY] = updated_at
    else:
        session_time = st.session_state.get(_QUOTE_UPDATED_AT_KEY)
        if isinstance(session_time, dt.datetime):
            updated_at = session_time

    return QuoteFetchResult(
        quotes={symbol: updated.get(symbol, {}) for symbol in normalized},
        updated_at=updated_at,
    )


def apply_quotes_to_market_value(
    df: pd.DataFrame,
    quotes: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    """Update market value column based on quotes."""
    if df.empty or not quotes:
        return df

    updated = df.copy()
    qty_series = pd.to_numeric(updated["qty"], errors="coerce").fillna(0.0)
    price_series = updated["symbol"].map(lambda x: quotes.get(x, {}).get("price"))
    price_numeric = pd.to_numeric(price_series, errors="coerce")
    mask = price_numeric.notna()
    updated.loc[mask, "market_value"] = (price_numeric[mask] * qty_series[mask]).round(4)
    updated["market_value"] = pd.to_numeric(updated["market_value"], errors="coerce").fillna(0.0)
    return updated
