"""History & Logs page: view snapshots and edit logs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st
from app.database import Database
from app.quotes import fetch_quotes
from app.ui.positions_data import digits_only_symbol, get_positions_dataframe, rename_columns


@dataclass(frozen=True)
class HistoryDependencies:
    db: Database
    market: Any | None


class HistoryPage:
    """History tab."""

    def __init__(self, deps: HistoryDependencies):
        self._deps = deps

    def render(self) -> None:
        st.header("📚 History & Logs", divider="gray")
        st.caption(
            "Snapshot saves current positions to SQLite snapshots table, can be combined with external scheduling for daily execution."
        )

        positions_df = get_positions_dataframe(self._deps.db)
        quotes_result = fetch_quotes(positions_df.get("symbol", []), self._deps.market)

        if st.button("Record Snapshot Now", key="record_snapshot"):
            self._record_snapshot(positions_df, quotes_result.quotes)

        self._render_snapshots()
        self._render_logs()

    def _record_snapshot(
        self, positions_df: pd.DataFrame, quotes: dict[str, dict[str, Any]]
    ) -> None:
        rows = []
        for row in positions_df.itertuples(index=False):
            price = quotes.get(row.symbol, {}).get("price")
            market_value = float(row.qty) * float(price or 0.0)
            rows.append(
                {
                    "symbol": row.symbol,
                    "qty": float(row.qty),
                    "market_value": market_value,
                }
            )
        self._deps.db.insert_snapshot(rows, date=pd.Timestamp.today().date())
        st.success("Today's snapshot recorded.")

    def _render_snapshots(self) -> None:
        snapshots = self._deps.db.get_snapshots()
        if snapshots:
            snap_df = pd.DataFrame(
                {
                    "date": [s.date for s in snapshots],
                    "symbol": [s.symbol for s in snapshots],
                    "qty": [s.qty for s in snapshots],
                    "market_value": [s.market_value for s in snapshots],
                }
            )
            st.markdown("#### Snapshot Records")
            snap_df["symbol"] = snap_df["symbol"].map(digits_only_symbol)
            st.dataframe(
                rename_columns(
                    snap_df,
                    {
                        "date": "Date",
                        "symbol": "Symbol",
                        "qty": "Qty",
                        "market_value": "Market Value",
                    },
                ),
                use_container_width=True,
            )
        else:
            st.caption("No snapshot records.")

    def _render_logs(self) -> None:
        logs = self._deps.db.get_edit_logs(limit=200)
        if logs:
            log_df = pd.DataFrame(
                {
                    "modify_time": [log.modify_time for log in logs],
                    "symbol": [log.symbol for log in logs],
                    "before_qty": [log.before_qty for log in logs],
                    "after_qty": [log.after_qty for log in logs],
                }
            )
            st.markdown("#### Recent Edit Logs")
            log_df["symbol"] = log_df["symbol"].map(digits_only_symbol)
            st.dataframe(
                rename_columns(
                    log_df,
                    {
                        "modify_time": "Modified At",
                        "symbol": "Symbol",
                        "before_qty": "Before Qty",
                        "after_qty": "After Qty",
                    },
                ),
                use_container_width=True,
            )
        else:
            st.caption("No edit logs.")
