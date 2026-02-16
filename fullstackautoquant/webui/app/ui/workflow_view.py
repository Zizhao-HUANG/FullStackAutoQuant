"""Strategy workflow page: execute strategy scripts and display results."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import streamlit as st

from app.database import Database
from app.market_data import MarketDataService
from app.ui.positions_data import digits_only_symbol, to_payload
from app.services.data_access import DataAccess
from app.workflow import WorkflowError, run_full_workflow, run_single_step


@dataclass(frozen=True)
class WorkflowDependencies:
    config: Dict[str, object]
    data_access: DataAccess
    db: Optional[Database] = None
    market: Optional[MarketDataService] = None


class WorkflowPage:
    """Workflow execution tab."""

    def __init__(self, deps: WorkflowDependencies):
        self._deps = deps

    def render(self) -> None:
        st.header("🚀 Strategy Script Workflow", divider="gray")
        st.caption("Runs the full pipeline: signals_from_csv.py → risk_manager.py → strategy_rebalance.py")

        debug_mode = st.checkbox("Debug mode: save and display script output", value=False, key="workflow_debug_mode")
        self._render_controls(debug_mode)
        data_access = self._deps.data_access
        result = data_access.get_workflow_result()

        manual_summary = data_access.manual_log_summary()
        if manual_summary.get("path"):
            st.caption(
                f"Manual decision log recently written:{manual_summary['path']} ( {manual_summary['count']}  entries)"
            )

        if not result:
            st.info("Workflow not yet executed. Click \"Run Full Workflow\" for latest results.")
            return

        self._render_targets_section(result)
        self._render_risk_section(result, debug_mode)
        self._render_history_section()

    def _render_controls(self, debug_mode: bool) -> None:
        data_access = self._deps.data_access
        col_run, col_select, col_single = st.columns([1.6, 1.2, 1.0])
        with col_run:
            if st.button("Run Full Workflow", key="run_full_workflow", type="primary"):
                db_service = self._deps.db
                positions_payload, skipped = to_payload(db_service)
                if skipped:
                    st.warning(f"The following positions could not be converted to GM symbols and were skipped:{', '.join(skipped)}")
                try:
                    if positions_payload:
                        result = run_full_workflow(self._deps.config, current_positions=positions_payload)
                    else:
                        result = run_full_workflow(self._deps.config)
                    data_access.set_workflow_result(result)
                    st.success("Workflow execution complete.")
                except WorkflowError as exc:  # noqa: BLE001
                    st.error(f"Workflow failed:{exc}")
        with col_select:
            selected_step = st.selectbox(
                "Single step", ["signals", "risk", "strategy"], key="workflow_step_select"
            )
        with col_single:
            if st.button("Execute Step", key="run_single_step"):
                try:
                    res = run_single_step(self._deps.config, selected_step)
                    data_access.append_debug_step(self._serialize_step(res))
                    st.success(f"{selected_step}  execution succeeded.")
                except WorkflowError as exc:  # noqa: BLE001
                    st.error(f"Single step execution failed:{exc}")

        if debug_mode:
            debug_steps = data_access.get_debug_steps()
            if debug_steps:
                with st.expander("Debug Output History", expanded=False):
                    for item in debug_steps:
                        name = item.get("name", "step")
                        with st.expander(name, expanded=False):
                            if item.get("stdout"):
                                st.text("STDOUT:\n" + item["stdout"])
                            if item.get("stderr"):
                                st.text("STDERR:\n" + item["stderr"])
                            outputs = item.get("outputs")
                            if outputs:
                                st.write(outputs)

    def _render_targets_section(self, result: Dict[str, object]) -> None:
        st.subheader("Plan Results", divider="blue")

        signals_info = result.get("signals", {})
        if isinstance(signals_info, dict):
            signal_list = signals_info.get("signals", [])
            count_label = signals_info.get("count", len(signal_list))
            st.caption(f"Signal count:{count_label}, Trading date:{signals_info.get('date', 'N/A')}.")

        targets_data = result.get("targets")
        if isinstance(targets_data, dict):
            targets_data = targets_data.get("targets")
        targets_df = pd.DataFrame(targets_data) if targets_data else pd.DataFrame()

        summary_cols = st.columns(3)
        targets_payload = result.get("targets", {}) if isinstance(result.get("targets"), dict) else {}
        invest_capital = float(targets_payload.get("invest_capital", 0.0) or 0.0)
        est_target_value = float(targets_payload.get("total_capital", 0.0) or 0.0)
        remaining_cash = float(targets_payload.get("remaining_cash", 0.0) or 0.0)
        available_cash = self._deps.data_access.get_available_cash()

        with summary_cols[0]:
            st.metric("Estimated Total Target Market Value", f"{est_target_value:,.2f}")
            if invest_capital > 0:
                est_weight = est_target_value / invest_capital if invest_capital else 0.0
                st.caption(f"Estimated position:{est_weight:.2%}")

        if not targets_df.empty:
            symbols_for_name = set(targets_df.get("symbol", []))
            names_map = self._fetch_names(symbols_for_name)
            targets_df = targets_df.copy()
            targets_df["name"] = targets_df["symbol"].map(lambda x: names_map.get(x, ""))
            if "symbol" in targets_df.columns:
                targets_df["symbol"] = targets_df["symbol"].map(digits_only_symbol)
            st.markdown("#### Target Positions")
            st.dataframe(
                targets_df.rename(
                    columns={
                        "symbol": "Symbol",
                        "name": "Name",
                        "instrument": "Instrument",
                        "target_shares": "Target Shares",
                        "weight": "Target Weight",
                        "ref_price": "Ref Price",
                    }
                ),
                use_container_width=True,
            )

        self._render_summary_and_orders(result, summary_cols, remaining_cash, available_cash)

    def _render_summary_and_orders(
        self,
        result: Dict[str, object],
        summary_cols,
        remaining_cash: float,
        available_cash: float,
    ) -> None:
        orders_data = result.get("orders")
        if isinstance(orders_data, dict):
            orders_data = orders_data.get("orders")
        orders_df = pd.DataFrame(orders_data) if orders_data else pd.DataFrame()
        signal_order = self._extract_signal_order(result)
        orders_df = self._cap_buy_orders(orders_df, available_cash, signal_order)

        buy_total = 0.0
        sell_total = 0.0
        if not orders_df.empty and {"price", "volume", "side"}.issubset(orders_df.columns):
            orders_df["price"] = pd.to_numeric(orders_df["price"], errors="coerce").fillna(0.0)
            orders_df["volume"] = pd.to_numeric(orders_df["volume"], errors="coerce").fillna(0.0)
            buy_mask = orders_df["side"].str.upper() == "BUY"
            sell_mask = orders_df["side"].str.upper() == "SELL"
            buy_total = float((orders_df.loc[buy_mask, "price"] * orders_df.loc[buy_mask, "volume"]).sum())
            sell_total = float((orders_df.loc[sell_mask, "price"] * orders_df.loc[sell_mask, "volume"]).sum())

        with summary_cols[1]:
            st.metric("Estimated Buy Amount", f"{buy_total:,.2f}")
            st.caption(f"Estimated sell amount:{sell_total:,.2f}")

        with summary_cols[2]:
            st.metric("Available Cash", f"{available_cash:,.2f}")
            if remaining_cash:
                st.caption(f"Estimated remaining cash from workflow:{remaining_cash:,.2f}")
            gap = buy_total - available_cash
            if gap > 0:
                st.error(f"Buy demand exceeds available cash {gap:,.2f}, please adjust positions or add cash.")

        signal_strengths = self._extract_signal_strengths(result)
        self._render_orders_section(orders_df, signal_strengths, signal_order)

    def _render_orders_section(
        self,
        orders_df: pd.DataFrame,
        signal_strengths: Dict[str, float],
        signal_order: Dict[str, int],
    ) -> None:
        if orders_df.empty:
            st.caption("No buy/sell orders.")
            return

        names_map = self._fetch_names(set(orders_df.get("symbol", [])))
        orders_df = orders_df.copy()
        orders_df["name"] = orders_df["symbol"].map(lambda x: names_map.get(x, ""))

        if signal_strengths:
            orders_df["_signal_strength"] = orders_df["symbol"].map(
                lambda s: self._lookup_signal_value(signal_strengths, s, 0.0)
            )
        if signal_order:
            default_order = len(signal_order) + 1
            orders_df["_signal_order"] = orders_df["symbol"].map(
                lambda s: self._lookup_signal_value(signal_order, s, default_order)
            )
        strength_column = self._pick_strength_column(orders_df.columns)
        if strength_column is None and "_signal_strength" in orders_df:
            strength_column = "_signal_strength"
        buy_mask = orders_df["side"].str.upper() == "BUY"
        if signal_order:
            orders_df.loc[buy_mask] = orders_df.loc[buy_mask].sort_values(
                by=["_signal_order"],
                ascending=True,
            )
        elif strength_column is not None:
            orders_df.loc[buy_mask] = orders_df.loc[buy_mask].sort_values(
                by=strength_column,
                ascending=False,
            )

        buy_df = orders_df[orders_df["side"].str.upper() == "BUY"].copy()
        sell_df = orders_df[orders_df["side"].str.upper() == "SELL"].copy()

        if not buy_df.empty:
            if signal_order:
                buy_df = buy_df.sort_values(by="_signal_order", ascending=True)
            elif strength_column is not None and strength_column in buy_df.columns:
                buy_df = buy_df.sort_values(by=strength_column, ascending=False)
            if "symbol" in buy_df.columns:
                buy_df["symbol"] = buy_df["symbol"].map(digits_only_symbol)
            st.markdown("#### Buy Orders")
            st.dataframe(
                buy_df.rename(
                    columns={
                        "symbol": "Symbol",
                        "name": "Name",
                        "side": "Side",
                        "volume": "Volume",
                        "price": "Price",
                        "type": "Type",
                        "est_cash": "Cash Budget",
                    }
                ),
                use_container_width=True,
            )
        if not sell_df.empty:
            if "symbol" in sell_df.columns:
                sell_df["symbol"] = sell_df["symbol"].map(digits_only_symbol)
            st.markdown("#### Sell Orders")
            st.dataframe(
                sell_df.rename(
                    columns={
                        "symbol": "Symbol",
                        "name": "Name",
                        "side": "Side",
                        "volume": "Volume",
                        "price": "Price",
                        "type": "Type",
                        "est_cash": "Cash Budget",
                    }
                ),
                use_container_width=True,
            )

        st.download_button(
            "Download All Orders CSV",
            orders_df.to_csv(index=False).encode("utf-8"),
            file_name="orders_all.csv",
            mime="text/csv",
        )

    def _cap_buy_orders(
        self,
        orders_df: pd.DataFrame,
        available_cash: float,
        signal_order: Dict[str, int],
    ) -> pd.DataFrame:
        if orders_df.empty or "side" not in orders_df or "price" not in orders_df:
            return orders_df

        capped_df = orders_df.copy()
        buy_mask = capped_df["side"].str.upper() == "BUY"
        if not buy_mask.any():
            return capped_df

        capped_df.loc[:, "price"] = pd.to_numeric(capped_df["price"], errors="coerce").fillna(0.0)
        capped_df.loc[:, "volume"] = pd.to_numeric(capped_df.get("volume", 0.0), errors="coerce").fillna(0.0)

        buy_orders = capped_df[buy_mask]
        strength_column = self._pick_strength_column(buy_orders.columns)
        if signal_order:
            default_order = len(signal_order) + 1
            buy_orders = buy_orders.assign(
                _signal_order=buy_orders["symbol"].map(
                    lambda s: self._lookup_signal_value(signal_order, s, default_order)
                )
            )
            buy_orders = buy_orders.sort_values(by="_signal_order", ascending=True)
        elif strength_column is not None:
            buy_orders = buy_orders.sort_values(by=strength_column, ascending=False)
        else:
            buy_orders = buy_orders.sort_values(by="price", ascending=True)

        remaining_cash = available_cash
        lot_size = self._resolve_lot_size()
        capped_volumes = []
        for _, row in buy_orders.iterrows():
            price = float(row["price"])
            if price <= 0 or remaining_cash <= 0:
                capped_volumes.append(0.0)
                continue
            original_volume = float(row.get("volume", 0.0))
            if original_volume <= 0:
                capped_volumes.append(0.0)
                continue
            affordable_volume = remaining_cash / price
            if lot_size > 1:
                affordable_volume = math.floor(affordable_volume / lot_size) * lot_size
            else:
                affordable_volume = math.floor(affordable_volume)
            capped_volume = min(original_volume, affordable_volume)
            capped_volumes.append(capped_volume)
            remaining_cash -= capped_volume * price

        buy_orders = buy_orders.assign(volume=capped_volumes)
        if "est_cash" in buy_orders.columns:
            buy_orders["est_cash"] = buy_orders["price"] * buy_orders["volume"]
        capped_df.update(buy_orders)
        others = capped_df.loc[~buy_mask]
        return pd.concat([buy_orders, others], ignore_index=True)

    def _pick_strength_column(self, columns: Iterable[str]) -> Optional[str]:
        for candidate in [
            "signal_strength",
            "confidence",
            "score",
            "rank",
            "priority",
            "0",
            "signal",
        ]:
            if candidate in columns:
                return candidate
        return None

    def _resolve_lot_size(self) -> int:
        portfolio_cfg = self._deps.config.get("portfolio") if isinstance(self._deps.config, dict) else None
        if isinstance(portfolio_cfg, dict):
            lot = int(portfolio_cfg.get("lot", 1) or 1)
            return max(lot, 1)
        return 1

    def _render_orders_table(self, orders_df: pd.DataFrame) -> None:
        if orders_df.empty:
            return
        names_map = self._fetch_names(set(orders_df.get("symbol", [])))
        orders_df = orders_df.copy()
        orders_df["name"] = orders_df["symbol"].map(lambda x: names_map.get(x, ""))
        buy_df = orders_df[orders_df["side"].str.upper() == "BUY"].copy()
        sell_df = orders_df[orders_df["side"].str.upper() == "SELL"].copy()

        if not buy_df.empty:
            if "symbol" in buy_df.columns:
                buy_df["symbol"] = buy_df["symbol"].map(digits_only_symbol)
            st.markdown("#### Buy Orders")
            st.dataframe(
                buy_df.rename(
                    columns={
                        "symbol": "Symbol",
                        "name": "Name",
                        "side": "Side",
                        "volume": "Volume",
                        "price": "Price",
                        "type": "Type",
                        "est_cash": "Cash Budget",
                    }
                ),
                use_container_width=True,
            )
        if not sell_df.empty:
            if "symbol" in sell_df.columns:
                sell_df["symbol"] = sell_df["symbol"].map(digits_only_symbol)
            st.markdown("#### Sell Orders")
            st.dataframe(
                sell_df.rename(
                    columns={
                        "symbol": "Symbol",
                        "name": "Name",
                        "side": "Side",
                        "volume": "Volume",
                        "price": "Price",
                        "type": "Type",
                        "est_cash": "Cash Budget",
                    }
                ),
                use_container_width=True,
            )

        st.download_button(
            "Download All Orders CSV",
            orders_df.to_csv(index=False).encode("utf-8"),
            file_name="orders_all.csv",
            mime="text/csv",
        )

    def _render_risk_section(self, result: Dict[str, object], debug_mode: bool) -> None:
        risk_state = result.get("risk_state")
        if isinstance(risk_state, dict):
            allow_buy = bool(risk_state.get("allow_buy", True))
            if allow_buy:
                st.success("Risk status: buying allowed")
            else:
                st.error("Risk status: buying prohibited")

            reasons = risk_state.get("reasons") or []
            if reasons:
                st.markdown("**Risk alerts:**")
                for reason in reasons:
                    st.write(f"- {reason}")

            details = risk_state.get("details")
            if debug_mode and details:
                st.markdown("**Risk details:**")
                st.json(details)
        else:
            st.caption("Risk note: no risk state returned.")

    def _render_history_section(self) -> None:
        st.markdown("### Historical Positions")
        if self._deps.db is None:
            st.caption("Database connection not configured, cannot show history.")
            return
        historical_positions = self._deps.db.get_historical_positions(limit=200)
        if historical_positions:
            hist_df = pd.DataFrame(
                {
                    "Record Date": [hp.record_date for hp in historical_positions],
                    "Symbol": [hp.symbol for hp in historical_positions],
                    "Volume": [hp.qty for hp in historical_positions],
                    "Cost Price": [hp.cost_price for hp in historical_positions],
                    "Market Value": [hp.qty * hp.cost_price for hp in historical_positions],
                    "Note": [hp.note for hp in historical_positions],
                    "Written At": [hp.inserted_at for hp in historical_positions],
                }
            )
            hist_df["Symbol"] = hist_df["Symbol"].map(digits_only_symbol)
            hist_df["Market Value"] = hist_df["Market Value"].map(lambda v: round(float(v), 2))
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.caption("No historical position records.")

    def _fetch_names(self, symbols: set[str]) -> Dict[str, str]:
        market = self._deps.market
        if not symbols or market is None:
            return {}
        try:
            names_quotes = market.get_realtime_quotes(list(symbols))
            return {code: data.get("name", "") for code, data in names_quotes.items()}
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Failed to get stock names:{exc}")
            return {}

    def _serialize_step(self, step: Any) -> Dict[str, Any]:
        outputs = {}
        for key, value in getattr(step, "output_paths", {}).items():
            outputs[key] = str(value)
        return {
            "name": getattr(step, "name", "step"),
            "stdout": getattr(step, "stdout", ""),
            "stderr": getattr(step, "stderr", ""),
            "outputs": outputs,
            "recorded_at": dt.datetime.utcnow().isoformat(),
        }

    def _extract_signal_strengths(self, result: Dict[str, object]) -> Dict[str, float]:
        signals_info = result.get("signals")
        if isinstance(signals_info, dict):
            signals = signals_info.get("signals")
        else:
            signals = None
        if not signals:
            return {}
        df = pd.DataFrame(signals)
        if df.empty:
            return {}
        strength_column = self._pick_strength_column(df.columns)
        if strength_column is None and "confidence" in df.columns:
            strength_column = "confidence"
        if strength_column is None or "instrument" not in df.columns:
            return {}
        mapping: Dict[str, float] = {}
        column = "instrument" if "instrument" in df.columns else "symbol" if "symbol" in df.columns else None
        if column is None:
            return {}
        for instrument, value in df[[column, strength_column]].itertuples(index=False):
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            for alias in self._expand_symbol_aliases(instrument):
                mapping[alias] = val
        return mapping

    def _extract_signal_order(self, result: Dict[str, object]) -> Dict[str, int]:
        signals_info = result.get("signals")
        if isinstance(signals_info, dict):
            signals = signals_info.get("signals")
        else:
            signals = None
        if not signals:
            return {}
        df = pd.DataFrame(signals)
        if df.empty:
            return {}
        column = "instrument" if "instrument" in df.columns else "symbol" if "symbol" in df.columns else None
        if column is None:
            return {}
        order: Dict[str, int] = {}
        for rank, instrument in enumerate(df[column].tolist(), start=1):
            for alias in self._expand_symbol_aliases(instrument):
                order[alias] = rank
        return order

    def _lookup_signal_value(self, signal_map: Dict[str, float], symbol: str, default: float) -> float:
        if not isinstance(symbol, str):
            return default
        for alias in self._expand_symbol_aliases(symbol):
            if alias in signal_map:
                return signal_map[alias]
        return default

    def _expand_symbol_aliases(self, symbol: str | None) -> set[str]:
        if symbol is None:
            return {""}
        text = str(symbol).strip().upper()
        if not text:
            return {""}
        aliases: set[str] = {text}
        code = ""
        prefix = ""
        if "." in text:
            prefix, code = text.split(".", 1)
        else:
            if text.startswith("SH") or text.startswith("SZ"):
                prefix = text[:2]
                code = text[2:]
        if prefix.startswith("SH"):
            aliases.add(f"SH{code}")
            aliases.add(f"SHSE.{code}")
        if prefix.startswith("SZ"):
            aliases.add(f"SZ{code}")
            aliases.add(f"SZSE.{code}")
        if code.isdigit():
            aliases.add(code)
        return aliases
