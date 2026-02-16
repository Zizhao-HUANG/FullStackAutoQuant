"""Position management page: import, edit, export, and save logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from app.database import Database
from app.quotes import QuoteFetchResult, apply_quotes_to_market_value, fetch_quotes, format_currency
from app.ui.positions_data import (
    PositionDiffResult,
    compute_position_diff,
    digits_only_symbol,
    fill_missing_numeric,
    get_positions_dataframe,
    load_positions_from_upload,
    rename_columns,
    standardize_positions_dataframe,
)
from app.ui.positions_metrics import PositionMetricsCalculator
from app.ui_apply_plan import render_apply_plan_section
from app.services.data_access import DataAccess


@dataclass(frozen=True)
class PositionPageDependencies:
    db: Database
    market: Optional[Any]
    data_access: DataAccess
    config: Dict[str, object]


class PositionsPage:
    """Position Management tab."""

    def __init__(self, deps: PositionPageDependencies):
        self._deps = deps

    def render(self) -> None:
        st.header("📊 Position Management", divider="gray")
        st.caption("Edit or import broker-exported positions. Saved to SQLite with edit logs.")

        raw_df = get_positions_dataframe(self._deps.db)
        self._initialize_cash_state()
        symbols_in_view = raw_df.get("symbol", pd.Series(dtype=str)).tolist()

        quotes_result = self._render_quote_controls(symbols_in_view)

        metrics = PositionMetricsCalculator(quotes_result.quotes).compute(raw_df)
        display_df = metrics.display.copy()
        realtime_price_numeric = display_df.pop("Live Price")

        available_cash = self._sync_available_cash()
        self._render_summary(metrics.totals, available_cash)

        edited_df = self._render_editor(display_df)

        export_df = self._build_export_dataframe(
            metrics.raw,
            metrics.totals,
            quotes_result.quotes,
            realtime_price_numeric,
        )
        self._render_import_export(export_df)
        self._render_diff(raw_df)
        self._render_save_button(edited_df)
        self._render_apply_plan(raw_df)

    def _render_quote_controls(self, symbols: list[str]) -> QuoteFetchResult:
        action_col, info_col = st.columns([1, 3])
        with action_col:
            refresh_trigger = st.button("Refresh Live Quotes", key="positions_refresh_quotes")
        quotes_result = fetch_quotes(
            symbols,
            self._deps.market,
            force=refresh_trigger,
            show_warning=refresh_trigger,
            recompute_missing=True,
        )
        with info_col:
            if quotes_result.updated_at:
                st.caption(f"Quotes updated at:{quotes_result.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
            elif self._deps.market is None:
                st.caption("TuShare not configured, market values will use current values.")
        if refresh_trigger:
            if quotes_result.quotes:
                st.success("Live quotes refreshed.")
            else:
                st.info("No live quote data received.")
        return quotes_result

    def _initialize_cash_state(self) -> None:
        def _loader() -> float:
            saved_cash_str = self._deps.db.get_config("positions.available_cash", "0")
            try:
                return float(saved_cash_str or 0.0)
            except (TypeError, ValueError):
                return 0.0

        saved_cash = self._deps.data_access.ensure_available_cash(_loader)
        st.session_state.setdefault("available_cash_saved", saved_cash)
        st.session_state.setdefault("available_cash", saved_cash)

    def _sync_available_cash(self) -> float:
        default_value = float(st.session_state.get("available_cash", self._deps.data_access.get_available_cash()))
        available_cash_input = st.number_input(
            "Available Cash",
            min_value=0.0,
            step=1000.0,
            value=default_value,
            key="positions_available_cash_input",
        )
        updated_cash = float(available_cash_input)
        st.session_state["available_cash"] = updated_cash
        self._deps.data_access.set_available_cash(updated_cash)
        saved_cash = float(st.session_state.get("available_cash_saved", updated_cash))
        if abs(updated_cash - saved_cash) > 1e-6:
            self._deps.db.set_config("positions.available_cash", f"{updated_cash:.4f}")
            st.session_state["available_cash_saved"] = updated_cash
        return updated_cash

    def _render_summary(self, totals: Dict[str, float], available_cash: float) -> None:
        total_market_value = totals["market_value"]
        total_pnl = totals["pnl"]
        total_daily = totals["daily_pnl"]
        total_cost = totals["cost"]
        total_assets = total_market_value + available_cash
        position_ratio = total_market_value / total_assets if total_assets else 0.0

        col_metrics = st.columns([1, 1, 1, 1, 1, 1])
        with col_metrics[0]:
            st.metric("Total Market Value", format_currency(total_market_value))
        with col_metrics[1]:
            st.metric("Total P&L", format_currency(total_pnl))
        with col_metrics[2]:
            st.metric("Today's Ref P&L", format_currency(total_daily))
        with col_metrics[3]:
            st.metric(
                "Position Cost", format_currency(total_cost), help="Position Cost = sum of cost_price × qty"
            )
        with col_metrics[4]:
            st.metric("Total Assets", format_currency(total_assets), help="Total Assets = Total MV + Available Cash")
        with col_metrics[5]:
            st.metric("Position %", f"{position_ratio:.2%}")

    def _render_editor(self, display_df: pd.DataFrame) -> pd.DataFrame:
        st.markdown("### Position Details")
        editor_df = st.data_editor(
            display_df,
            num_rows="dynamic",
            use_container_width=True,
            key="positions_editor",
            hide_index=True,
            disabled=[
                "Name",
                "Live Value",
                "Floating P&L",
                "Daily P&L",
                "Weight",
                "Return %",
            ],
            column_config={
                "Qty": st.column_config.NumberColumn("Qty", format="%.0f"),
                "Cost Price": st.column_config.NumberColumn("Cost Price", format="%.4f"),
                "Live Value": st.column_config.NumberColumn("Live Value", format="%.2f", disabled=True),
                "Floating P&L": st.column_config.NumberColumn("Floating P&L", format="%.2f", disabled=True),
                "Daily P&L": st.column_config.NumberColumn("Daily P&L", format="%.2f", disabled=True),
                "Weight": st.column_config.TextColumn("Weight", disabled=True),
                "Return %": st.column_config.TextColumn("Return %", disabled=True),
            },
        )
        return editor_df

    def _build_export_dataframe(
        self,
        raw_df: pd.DataFrame,
        totals: Dict[str, float],
        quotes: Dict[str, Dict[str, object]],
        realtime_price: pd.Series,
    ) -> pd.DataFrame:
        export_df = raw_df.copy()
        export_df = fill_missing_numeric(export_df, ["qty", "cost_price", "market_value"])
        export_df["floating_pnl"] = (
            export_df["market_value"] - export_df["cost_price"] * export_df["qty"]
        )
        export_df["daily_pnl"] = export_df["symbol"].map(
            lambda x: quotes.get(x, {}).get("price", 0.0)
        ) - export_df["symbol"].map(lambda x: quotes.get(x, {}).get("pre_close", 0.0))
        export_df["daily_pnl"] = export_df["daily_pnl"] * export_df["qty"]
        export_df["weight"] = export_df["market_value"] / totals["market_value"] if totals["market_value"] else 0.0
        export_df["realtime_price"] = realtime_price
        export_df["name"] = export_df["symbol"].map(lambda x: quotes.get(x, {}).get("name", ""))
        export_df["symbol"] = export_df["symbol"].map(digits_only_symbol)
        export_df = export_df.rename(
            columns={
                "symbol": "Symbol",
                "qty": "Qty",
                "cost_price": "Cost Price",
                "market_value": "Live Value",
                "floating_pnl": "Floating P&L",
                "daily_pnl": "Daily P&L",
                "weight": "Weight",
                "realtime_price": "Live Price",
                "name": "Name",
            }
        )
        export_df["Live Value"] = pd.to_numeric(export_df["Live Value"], errors="coerce").fillna(0.0).round(2)
        export_df["Floating P&L"] = pd.to_numeric(export_df["Floating P&L"], errors="coerce").fillna(0.0).round(2)
        export_df["Daily P&L"] = pd.to_numeric(export_df["Daily P&L"], errors="coerce").fillna(0.0).round(2)
        export_df["Weight"] = export_df["Weight"].map(
            lambda v: f"{float(v):.4%}" if pd.notna(v) else ""
        )
        export_df["Live Price"] = pd.to_numeric(export_df["Live Price"], errors="coerce").fillna(0.0).round(4)
        return export_df

    def _render_import_export(self, export_df: pd.DataFrame) -> None:
        st.markdown("### Import / Export")
        col_import, col_export = st.columns(2)
        with col_import:
            uploaded = st.file_uploader(
                "Import Broker Position CSV / JSON", type=["csv", "json"], key="positions_upload"
            )
            if uploaded is not None:
                try:
                    new_df = load_positions_from_upload(uploaded)
                    st.session_state["positions_import_df"] = new_df
                    st.success("Import file parsed. Click Save to overwrite database.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Import failed:{exc}")
        with col_export:
            st.download_button(
                "Download Current Positions CSV",
                export_df.to_csv(index=False).encode("utf-8"),
                file_name="positions_export.csv",
                mime="text/csv",
            )

    def _render_diff(self, raw_df: pd.DataFrame) -> None:
        imported_df: Optional[pd.DataFrame] = st.session_state.get(  # type: ignore[assignment]
            "positions_import_df"
        )
        if imported_df is not None:
            diff_result: PositionDiffResult = compute_position_diff(raw_df, imported_df)
            st.caption("Import vs Current Database Comparison:")
            diff_df = diff_result.diff.copy()
            diff_df["symbol"] = diff_df["symbol"].map(digits_only_symbol)
            st.dataframe(
                rename_columns(
                    diff_df,
                    {
                        "symbol": "Symbol",
                        "qty_current": "Current Qty",
                        "qty_new": "New Qty",
                        "delta_qty": "Qty Change",
                        "market_value_current": "Current MV",
                        "market_value_new": "New MV",
                        "delta_value": "MV Change",
                    },
                ),
                use_container_width=True,
            )

    def _render_save_button(self, edited_df: pd.DataFrame) -> None:
        if st.button("Save Positions", type="primary"):
            imported_df = st.session_state.get("positions_import_df")
            if imported_df is not None:
                to_save_df = standardize_positions_dataframe(imported_df)
                st.session_state.pop("positions_import_df", None)
            else:
                renamed = edited_df.rename(
                    columns={
                        "Symbol": "symbol",
                        "Qty": "qty",
                        "Cost Price": "cost_price",
                        "Live Value": "market_value",
                    }
                )
                to_save_df = standardize_positions_dataframe(renamed)

            quotes_result = fetch_quotes(
                to_save_df.get("symbol", pd.Series(dtype=str)).tolist(),
                self._deps.market,
                force=True,
                show_warning=True,
            )
            to_save_df = apply_quotes_to_market_value(to_save_df, quotes_result.quotes)
            to_save_df = fill_missing_numeric(to_save_df, ["qty", "cost_price", "market_value"])

            self._deps.db.replace_positions(to_save_df.to_dict("records"))

            updated_count = sum(
                1
                for data in quotes_result.quotes.values()
                if pd.notna(pd.to_numeric(pd.Series([data.get("price")]), errors="coerce").iloc[0])
            )
            if updated_count > 0:
                st.success(f"Positions saved, updated {updated_count}  live values.")
            else:
                st.success("Positions saved with edit logs.")

    def _render_apply_plan(self, raw_df: pd.DataFrame) -> None:
        workflow_result = self._deps.data_access.get_workflow_result()
        render_apply_plan_section(
            db=self._deps.db,
            current_positions_df=raw_df,
            workflow_result=workflow_result,
            data_access=self._deps.data_access,
            config=self._deps.config,
            key_prefix="positions",
        )

        st.markdown("---")
