"""Streamlit helpers for editing strategy config sections on the backtest page."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import streamlit as st

from .backtest_config import infer_backtest_logs_root


@dataclass(frozen=True)
class StrategyDefaults:
    default_daily_pv: Path
    config_path: Path
    logs_base: Path


@dataclass(frozen=True)
class StrategyOverrides:
    config: Dict[str, object]
    daily_pv: Path
    initial_capital: float
    logs_root: Path


def render_strategy_sections(strategy_cfg: Dict[str, object], defaults: StrategyDefaults) -> StrategyOverrides:
    """Render all strategy-related tabs and return collected overrides."""

    working_cfg: Dict[str, Any] = copy.deepcopy(strategy_cfg)

    portfolio_cfg = _ensure_dict(working_cfg.get("portfolio"))
    weights_cfg = _ensure_dict(working_cfg.get("weights"))
    order_cfg = _ensure_dict(working_cfg.get("order"))
    risk_cfg = _ensure_dict(working_cfg.get("risk"))
    capital_cfg = _ensure_dict(working_cfg.get("capital"))
    paths_cfg = _ensure_dict(working_cfg.get("paths"))
    rebalance_cfg = _ensure_dict(working_cfg.get("rebalance_trigger"))

    (
        portfolio_tab,
        weights_tab,
        order_tab,
        risk_tab,
        capital_paths_tab,
        rebalance_tab,
    ) = st.tabs(["Portfolio", "Weights", "Trade Params", "Risk Params", "Capital & Paths", "Rebalance Trigger"])

    working_cfg["portfolio"] = _render_portfolio_tab(portfolio_tab, portfolio_cfg)
    working_cfg["weights"] = _render_weights_tab(weights_tab, weights_cfg)
    working_cfg["order"] = _render_order_tab(order_tab, order_cfg)
    working_cfg["risk"] = _render_risk_tab(risk_tab, risk_cfg)
    capital_cfg = _render_capital_tab(capital_paths_tab, capital_cfg)
    paths_cfg = _render_paths_tab(capital_paths_tab, paths_cfg, defaults)
    working_cfg["capital"] = capital_cfg
    working_cfg["paths"] = paths_cfg
    working_cfg["rebalance_trigger"] = _render_rebalance_tab(rebalance_tab, rebalance_cfg)
    working_cfg.pop("gm", None)

    daily_pv_path = _resolve_daily_pv(paths_cfg, defaults.default_daily_pv)
    logs_root = infer_backtest_logs_root(paths_cfg, defaults.config_path, defaults.logs_base)
    initial_capital = float(capital_cfg.get("initial", 1_000_000.0))

    return StrategyOverrides(
        config=working_cfg,
        daily_pv=daily_pv_path,
        initial_capital=initial_capital,
        logs_root=logs_root,
    )


def _render_portfolio_tab(tab, portfolio: Dict[str, Any]) -> Dict[str, Any]:
    with tab:
        col1, col2, col3 = st.columns(3)
        portfolio["topk"] = int(col1.number_input("TopK Count", value=int(portfolio.get("topk", 20)), step=1))
        portfolio["n_drop"] = int(col2.number_input("Replaceable Non-Target Positions", value=int(portfolio.get("n_drop", 0)), step=1))
        portfolio["invest_ratio"] = float(
            col3.number_input(
                "Capital Usage Ratio",
                value=float(portfolio.get("invest_ratio", 0.95)),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            )
        )
        col4, col5, col6 = st.columns(3)
        portfolio["max_weight"] = float(
            col4.number_input(
                "Max Weight per Name",
                value=float(portfolio.get("max_weight", 0.05)),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            )
        )
        portfolio["lot"] = int(col5.number_input("Minimum Lot Size", value=int(portfolio.get("lot", 100)), step=1))
        portfolio["confidence_floor"] = float(
            col6.number_input(
                "Confidence Threshold",
                value=float(portfolio.get("confidence_floor", 0.9)),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            )
        )
    return portfolio


def _render_weights_tab(tab, weights: Dict[str, Any]) -> Dict[str, Any]:
    with tab:
        options = ["equal", "ranked"]
        current_mode = str(weights.get("mode", "equal"))
        if current_mode not in options:
            options.append(current_mode)
        col1, col2 = st.columns(2)
        weights["mode"] = col1.selectbox(
            "Weight Mode",
            options,
            index=options.index(current_mode) if current_mode in options else 0,
        )
        weights["confidence_tilt"] = col2.checkbox(
            "Tilt by Confidence",
            value=bool(weights.get("confidence_tilt", True)),
        )
        if weights["mode"] == "ranked":
            col3, col4 = st.columns(2)
            metric_options = ["rank", "score"]
            rank_metric = str(weights.get("rank_metric", "rank"))
            if rank_metric not in metric_options:
                metric_options.append(rank_metric)
            weights["rank_metric"] = col3.selectbox(
                "Sort By",
                metric_options,
                index=metric_options.index(rank_metric) if rank_metric in metric_options else 0,
            )
            weights["rank_exponent"] = float(
                col4.number_input(
                    "Rank Power",
                    value=float(weights.get("rank_exponent", 1.0)),
                    min_value=0.1,
                    max_value=8.0,
                    step=0.1,
                )
            )
    return weights


def _render_order_tab(tab, order: Dict[str, Any]) -> Dict[str, Any]:
    with tab:
        modes = ["auto", "manual"]
        current_mode = str(order.get("mode", "auto"))
        if current_mode not in modes:
            modes.append(current_mode)
        order["mode"] = st.selectbox(
            "Order Mode",
            modes,
            index=modes.index(current_mode) if current_mode in modes else 0,
        )
        order["price_source"] = "qlib_close"
        order.pop("tushare_src", None)

        col1, col2, col3 = st.columns(3)
        order["buy_limit_offset"] = float(
            col1.number_input(
                "Buy Limit Offset",
                value=float(order.get("buy_limit_offset", 0.02)),
                step=0.001,
                format="%.3f",
            )
        )
        order["sell_limit_offset"] = float(
            col2.number_input(
                "Sell Limit Offset",
                value=float(order.get("sell_limit_offset", -0.02)),
                step=0.001,
                format="%.3f",
            )
        )
        order["limit_threshold"] = float(
            col3.number_input(
                "Limit Up/Down Threshold",
                value=float(order.get("limit_threshold", 0.095)),
                step=0.001,
                format="%.3f",
            )
        )
        col4, col5 = st.columns(2)
        order["min_buy_notional"] = float(
            col4.number_input(
                "Min Buy Amount",
                value=float(order.get("min_buy_notional", 0.0)),
                step=100.0,
            )
        )
        order["min_sell_notional"] = float(
            col5.number_input(
                "Min Sell Amount",
                value=float(order.get("min_sell_notional", 0.0)),
                step=100.0,
            )
        )
        order["clamp_tick"] = float(
            st.number_input(
                "Limit Price Min Step",
                value=float(order.get("clamp_tick", 0.01)),
                step=0.001,
                format="%.3f",
            )
        )
    return order


def _render_risk_tab(tab, risk: Dict[str, Any]) -> Dict[str, Any]:
    with tab:
        col1, col2 = st.columns(2)
        risk["day_drawdown_limit"] = float(
            col1.number_input(
                "Max Intraday Drawdown",
                value=float(risk.get("day_drawdown_limit", 0.05)),
                step=0.005,
            )
        )
        risk["rolling5d_drawdown_limit"] = float(
            col2.number_input(
                "5-Day Rolling Drawdown",
                value=float(risk.get("rolling5d_drawdown_limit", 0.1)),
                step=0.005,
            )
        )
        risk["enforce_limit_up_down_filter"] = st.checkbox(
            "Enable Limit-Up/Down Filter",
            value=bool(risk.get("enforce_limit_up_down_filter", True)),
        )
    return risk


def _render_capital_tab(tab, capital: Dict[str, Any]) -> Dict[str, Any]:
    with tab:
        capital["initial"] = float(
            st.number_input(
                "Initial Capital (CNY)",
                value=float(capital.get("initial", 1_000_000.0)),
                min_value=10000.0,
                step=50000.0,
            )
        )
    return capital


def _render_paths_tab(tab, paths: Dict[str, Any], defaults: StrategyDefaults) -> Dict[str, Any]:
    with tab:
        st.markdown("#### Path Configuration")
        col1, col2, col3 = st.columns(3)
        paths["daily_pv"] = col1.text_input(
            "Daily PV H5 Path",
            value=str(paths.get("daily_pv", defaults.default_daily_pv)),
        )
        paths["logs_dir"] = col2.text_input(
            "Log Directory",
            value=str(paths.get("logs_dir", defaults.logs_base)),
        )
        paths["tushare_token"] = col3.text_input(
            "TuShare Token",
            value=str(paths.get("tushare_token", "")),
            type="password",
        )
    return paths


def _render_rebalance_tab(tab, rebalance: Dict[str, Any]) -> Dict[str, Any]:
    with tab:
        col1, col2, col3 = st.columns(3)
        rebalance["max_non_target_to_replace"] = int(
            col1.number_input(
                "Replaceable Non-Target",
                value=int(rebalance.get("max_non_target_to_replace", 0)),
                step=1,
                min_value=0,
            )
        )
        rebalance["enable_weight_drift"] = col2.checkbox(
            "Enable Weight Drift Detection",
            value=bool(rebalance.get("enable_weight_drift", False)),
        )
        rebalance["weight_drift_threshold"] = float(
            col3.number_input(
                "Weight Drift Threshold",
                value=float(rebalance.get("weight_drift_threshold", 0.2)),
                min_value=0.0,
                step=0.05,
            )
        )
    return rebalance


def _resolve_daily_pv(paths_cfg: Mapping[str, Any], default_daily_pv: Path) -> Path:
    raw = paths_cfg.get("daily_pv") if isinstance(paths_cfg, Mapping) else None
    if not raw:
        return default_daily_pv
    try:
        return Path(str(raw)).expanduser().resolve()
    except (TypeError, ValueError):
        return default_daily_pv


def _ensure_dict(payload: Any) -> Dict[str, Any]:
    return dict(payload) if isinstance(payload, Mapping) else {}


__all__ = [
    "StrategyDefaults",
    "StrategyOverrides",
    "render_strategy_sections",
]
