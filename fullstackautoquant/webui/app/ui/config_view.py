"""Configuration page: edit trading config file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import streamlit as st
import yaml


@dataclass(frozen=True)
class ConfigDependencies:
    config: Dict[str, object]


class ConfigPage:
    """System Config tab."""

    def __init__(self, deps: ConfigDependencies):
        self._deps = deps

    def render(self) -> None:
        st.header("⚙️ System Config", divider="gray")
        st.caption("Edit trading/config.auto.local.yaml parameters. Re-run workflow after saving.")

        trading_config_path = self._resolve_config_path()
        if not trading_config_path.exists():
            st.error(f"Config file does not exist: {trading_config_path}")
            return

        cfg_data = yaml.safe_load(trading_config_path.read_text(encoding="utf-8")) or {}

        self._render_portfolio_section(cfg_data)
        self._render_weights_section(cfg_data)
        self._render_order_section(cfg_data)
        self._render_risk_section(cfg_data)
        self._render_capital_paths_section(cfg_data)

        if st.button("Save Config", key="save_trading_config"):
            trading_config_path.write_text(
                yaml.safe_dump(cfg_data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            st.success(f"Written to {trading_config_path}, please use the latest config for subsequent runs.")

    def _resolve_config_path(self) -> Path:
        raw_value = None
        paths_cfg = self._deps.config.get("paths") if isinstance(self._deps.config, dict) else None
        if isinstance(paths_cfg, dict):
            raw_value = paths_cfg.get("trading_config")
        trading_config_path = Path(raw_value) if raw_value else None

        if trading_config_path is not None and trading_config_path.is_absolute():
            return trading_config_path

        candidate_rel = trading_config_path or Path("../trading/config.auto.local.yaml")
        # Try app, webui, project root directory in order
        base_candidates = [
            Path(__file__).resolve().parents[1],
            Path(__file__).resolve().parents[2],
            Path(__file__).resolve().parents[3],
            Path(__file__).resolve().parents[4],
        ]
        for base in base_candidates:
            candidate = (base / candidate_rel).resolve()
            if candidate.exists():
                return candidate

        # If still not found, default to webui base path
        return (Path(__file__).resolve().parents[2] / candidate_rel).resolve()

    def _render_portfolio_section(self, cfg_data: Dict[str, object]) -> None:
        st.markdown("### Portfolio Parameters")
        portfolio = cfg_data.get("portfolio", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            portfolio["topk"] = st.number_input(
                "TopK Count", value=int(portfolio.get("topk", 30)), step=1
            )
            portfolio["lot"] = st.number_input(
                "Minimum Lot Size", value=int(portfolio.get("lot", 100)), step=1
            )
        with col2:
            portfolio["n_drop"] = st.number_input(
                "Replaceable Non-Target Positions", value=int(portfolio.get("n_drop", 5)), step=1
            )
            portfolio["confidence_floor"] = st.number_input(
                "Confidence Threshold", value=float(portfolio.get("confidence_floor", 0.96)), step=0.01
            )
        with col3:
            portfolio["invest_ratio"] = st.number_input(
                "Capital Usage Ratio", value=float(portfolio.get("invest_ratio", 0.95)), step=0.01
            )
            portfolio["max_weight"] = st.number_input(
                "Max Weight per Name", value=float(portfolio.get("max_weight", 0.05)), step=0.01
            )
        cfg_data["portfolio"] = portfolio

    def _render_weights_section(self, cfg_data: Dict[str, object]) -> None:
        st.markdown("### Weight Mode")
        weights = cfg_data.get("weights", {}) or {}
        current_mode = str(weights.get("mode", "equal"))
        mode_options = ["equal", "ranked"]
        if current_mode not in mode_options:
            current_mode = "equal"

        col1, col2 = st.columns(2)
        with col1:
            selected_mode = st.selectbox(
                "Mode",
                mode_options,
                index=mode_options.index(current_mode),
                help="equal: equal weight; ranked: weight by signal rank/score",
            )
            weights["mode"] = selected_mode
        with col2:
            default_tilt = weights.get("confidence_tilt")
            if default_tilt is None:
                default_tilt = selected_mode != "ranked"
            weights["confidence_tilt"] = st.checkbox(
                "Tilt by Confidence",
                value=bool(default_tilt),
                help="When enabled, slightly tilts base weights by confidence",
            )

        if weights.get("mode") == "ranked":
            col3, col4 = st.columns(2)
            metric_options = ["rank", "score"]
            current_metric = weights.get("rank_metric", "rank")
            if current_metric not in metric_options:
                current_metric = "rank"
            with col3:
                weights["rank_metric"] = st.selectbox(
                    "Sort By",
                    metric_options,
                    index=metric_options.index(current_metric),
                    help="Generate base weights by rank or raw score",
                )
            with col4:
                weights["rank_exponent"] = st.number_input(
                    "Power (ranked)",
                    value=float(weights.get("rank_exponent", 1.0)),
                    min_value=0.1,
                    max_value=8.0,
                    step=0.1,
                    help=">1 concentrates on top, <1 more uniform",
                )

        cfg_data["weights"] = weights

    def _render_order_section(self, cfg_data: Dict[str, object]) -> None:
        st.markdown("### Order Parameters")
        order = cfg_data.get("order", {})
        price_source_options = ["qlib_close", "tushare"]
        current_source = order.get("price_source", "qlib_close")
        if current_source not in price_source_options:
            price_source_options.append(current_source)

        col1, col2, col3 = st.columns(3)
        with col1:
            order["mode"] = st.selectbox(
                "Trading Mode", ["auto", "manual"], index=0 if order.get("mode", "auto") == "auto" else 1
            )
            order["price_source"] = st.selectbox(
                "Reference Price Source", price_source_options, index=price_source_options.index(current_source)
            )
        with col2:
            order["buy_limit_offset"] = st.number_input(
                "Buy Limit Offset", value=float(order.get("buy_limit_offset", 0.02)), step=0.001, format="%.3f"
            )
            order["sell_limit_offset"] = st.number_input(
                "Sell Limit Offset", value=float(order.get("sell_limit_offset", -0.02)), step=0.001, format="%.3f"
            )
        with col3:
            order["limit_threshold"] = st.number_input(
                "Limit Up/Down Threshold", value=float(order.get("limit_threshold", 0.095)), step=0.001, format="%.3f"
            )
            order["clamp_tick"] = st.number_input(
                "Limit Price Min Step", value=float(order.get("clamp_tick", 0.01)), step=0.001, format="%.3f"
            )

        if order.get("price_source") == "tushare":
            src_options = ["sina", "dc"]
            current_src = order.get("tushare_src", "sina")
            if current_src not in src_options:
                src_options.append(current_src)
            order["tushare_src"] = st.selectbox(
                "TuShare Data Source", src_options, index=src_options.index(current_src)
            )
        cfg_data["order"] = order

    def _render_risk_section(self, cfg_data: Dict[str, object]) -> None:
        st.markdown("### Risk Parameters")
        risk = cfg_data.get("risk", {})
        col1, col2 = st.columns(2)
        with col1:
            risk["day_drawdown_limit"] = st.number_input(
                "Max Intraday Drawdown", value=float(risk.get("day_drawdown_limit", 0.03)), step=0.005
            )
            risk["rolling5d_drawdown_limit"] = st.number_input(
                "5-Day Rolling Drawdown", value=float(risk.get("rolling5d_drawdown_limit", 0.08)), step=0.005
            )
        with col2:
            risk["enforce_limit_up_down_filter"] = st.checkbox(
                "Enable Limit-Up/Down Filter", value=bool(risk.get("enforce_limit_up_down_filter", True))
            )
        cfg_data["risk"] = risk

    def _render_capital_paths_section(self, cfg_data: Dict[str, object]) -> None:
        st.markdown("### Capital & Paths")
        capital = cfg_data.get("capital", {})
        capital["initial"] = st.number_input(
            "Initial Capital (CNY)", value=float(capital.get("initial", 300000)), step=10000.0
        )
        cfg_data["capital"] = capital

        paths_cfg = cfg_data.get("paths", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            paths_cfg["daily_pv"] = st.text_input(
                "Daily PV H5 Path", value=str(paths_cfg.get("daily_pv", ""))
            )
        with col2:
            paths_cfg["logs_dir"] = st.text_input("Log Directory", value=str(paths_cfg.get("logs_dir", "")))
        with col3:
            paths_cfg["tushare_token"] = st.text_input(
                "TuShare Token",
                value=str(paths_cfg.get("tushare_token", "")),
                type="password",
                help="Used to access TuShare API. Leave empty to use environment variable.",
            )
        cfg_data["paths"] = paths_cfg
