from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

import pandas as pd
import streamlit as st

# External dependency references for cross-module imports
from app.database import Database
from app.manual_decisions import log_manual_decisions
from app.positions_apply import apply_orders, cap_buy_orders_by_cash
from app.services.data_access import DataAccess
from app.ui.positions_data import digits_only_symbol


def _as_dataframe(data: Sequence[dict[str, object]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if not data:
        return pd.DataFrame(columns=["symbol", "qty", "cost_price", "market_value"])
    return pd.DataFrame(data)


def render_apply_plan_section(
    *,
    db: Database,
    current_positions_df: Sequence[dict[str, object]] | pd.DataFrame,
    workflow_result: dict[str, Any] | None,
    data_access: DataAccess,
    config: dict[str, Any],
    key_prefix: str,
) -> None:
    """Display and execute"apply daily plan"operation, reusing apply_orders logic."""

    if not workflow_result:
        return

    orders_data = workflow_result.get("orders") if isinstance(workflow_result, dict) else None
    if isinstance(orders_data, dict):
        orders_data = orders_data.get("orders")

    orders_df = pd.DataFrame(orders_data) if orders_data else pd.DataFrame()

    container = st.container()
    with container:
        last_manual = data_access.manual_log_summary()
        if last_manual.get("path"):
            st.info(
                f"Recently recorded {int(last_manual.get('count', 0))} manual decisions:{last_manual.get('path', '')}"
            )

        if orders_df.empty or not {"symbol", "side", "volume", "price"}.issubset(orders_df.columns):
            st.info("Current plan has no applicable buy/sell orders.")
            return

        st.markdown("### Apply Daily Plan to Current Positions")

        if "apply_defaults" not in st.session_state:
            st.session_state["apply_defaults"] = {
                "sync_cash": True,
                "note": dt.date.today().isoformat(),
            }

        defaults = st.session_state.get("apply_defaults", {})
        apply_col1, apply_col2, apply_col3 = st.columns([1.2, 1.2, 1.0])
        auto_refresh_cash = apply_col1.checkbox(
            "Sync available cash",
            value=bool(defaults.get("sync_cash", True)),
            key=f"{key_prefix}_apply_sync_cash",
        )
        note_text = apply_col2.text_input(
            "Historical position note",
            value=str(defaults.get("note", dt.date.today().isoformat())),
            key=f"{key_prefix}_apply_history_note",
        )
        confirm = apply_col3.button(
            "Apply positions",
            type="primary",
            key=f"{key_prefix}_apply_positions_button",
        )

        st.session_state["apply_defaults"] = {"sync_cash": auto_refresh_cash, "note": note_text}

        if not confirm:
            return

        positions_df = _as_dataframe(current_positions_df)
        try:
            available_cash = float(st.session_state.get("available_cash", data_access.get_available_cash()))
        except (TypeError, ValueError):
            available_cash = 0.0
        portfolio_cfg = config.get("portfolio") if isinstance(config, dict) else None
        lot_size = 1
        if isinstance(portfolio_cfg, dict):
            try:
                lot_size = int(portfolio_cfg.get("lot", 1) or 1)
            except (TypeError, ValueError):
                lot_size = 1
        capped_orders_df, trimmed_symbols = cap_buy_orders_by_cash(
            positions_df,
            orders_df,
            available_cash,
            lot_size=lot_size,
        )
        orders_payload = capped_orders_df[["symbol", "side", "volume", "price"]].copy()

        result = apply_orders(
            positions_df,
            orders_payload,
        )

        db.replace_positions(result.positions)

        if result.sold:
            db.append_historical_positions(result.sold, date=dt.date.today(), note=note_text)

        manual_log_path = None
        manual_records = []
        try:
            manual_log_path, manual_records = log_manual_decisions(
                config,
                orders_payload,
                workflow_result.get("signals") if isinstance(workflow_result, dict) else None,
                note=note_text,
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Failed to record manual decision log:{exc}")

        if auto_refresh_cash:
            current_cash = float(st.session_state.get("available_cash", 0.0) or 0.0)
            new_cash = max(0.0, current_cash + result.cash_delta)
            st.session_state["available_cash"] = new_cash
            st.session_state["available_cash_saved"] = new_cash
            db.set_config("positions.available_cash", f"{new_cash:.4f}")
            data_access.set_available_cash(new_cash)

        st.session_state.pop("positions_import_df", None)
        st.session_state["last_apply_result"] = result
        if manual_log_path:
            data_access.record_manual_log(str(manual_log_path), len(manual_records))

        data_access.clear_workflow_result()
        st.session_state.pop("workflow_result", None)

        st.success(
            f"Applied {result.processed}  orders: bought {result.buy_notional:.2f}  CNY, sold {result.sell_notional:.2f}  CNY."
        )
        if trimmed_symbols:
            formatted = ", ".join(sorted({digits_only_symbol(code) for code in trimmed_symbols if code}))
            st.info(f"Due to insufficient capital, the following buy orders were partially executed or skipped:{formatted}")
        for warn in result.warnings:
            st.warning(warn)

        st.rerun()
