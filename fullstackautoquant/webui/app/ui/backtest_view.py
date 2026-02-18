"""Backtest Analysis page: run and view backtest results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .backtest_config import (
    ConfigLoadError,
    infer_backtest_logs_root,
    load_strategy_config,
    resolve_trading_config_path,
)
from .backtest_sections import StrategyDefaults, StrategyOverrides, render_strategy_sections
from .backtest_utils import default_daily_pv, default_logs_base, format_percent, format_ratio

try:
    from fullstackautoquant.backtest.config import BacktestConfig as BTConfig
    from fullstackautoquant.backtest.engine import BacktestEngine
    from fullstackautoquant.backtest.storage import list_backtest_runs, load_backtest_result
except ModuleNotFoundError:
    try:
        from trading.backtest.config import BacktestConfig as BTConfig  # type: ignore
        from trading.backtest.engine import BacktestEngine  # type: ignore
        from trading.backtest.storage import (  # type: ignore
            list_backtest_runs,
            load_backtest_result,
        )
    except ModuleNotFoundError as exc:  # noqa: BLE001
        BTConfig = None  # type: ignore
        BacktestEngine = None  # type: ignore
        list_backtest_runs = None  # type: ignore
        load_backtest_result = None  # type: ignore
        BACKTEST_IMPORT_ERROR: Exception | None = exc
    else:
        BACKTEST_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    BTConfig = None  # type: ignore
    BacktestEngine = None  # type: ignore
    list_backtest_runs = None  # type: ignore
    load_backtest_result = None  # type: ignore
    BACKTEST_IMPORT_ERROR = exc
else:
    BACKTEST_IMPORT_ERROR = None


@dataclass(frozen=True)
class BacktestDependencies:
    config: dict[str, Any]


class BacktestPage:
    """Backtest Analysis tab."""

    def __init__(self, deps: BacktestDependencies):
        self._deps = deps

    def render(self) -> None:
        st.header("📈 Backtest Analysis", divider="gray")
        if BacktestEngine is None or BTConfig is None:
            if BACKTEST_IMPORT_ERROR is not None:
                st.warning(
                    "Backtest dependencies not loaded, reason:"
                    f"{BACKTEST_IMPORT_ERROR}. Install required dependencies and retry."
                )
            else:
                st.info("Backtest dependencies not yet loaded, confirm trading/backtest module is available.")
            return

        inputs = self._collect_inputs()
        if inputs.run_button:
            self._run_backtest(inputs)
        self._render_history()

    def _collect_inputs(self) -> BacktestInputs:
        config_path = resolve_trading_config_path(self._deps.config)
        try:
            raw_strategy_cfg = load_strategy_config(config_path)
        except ConfigLoadError as exc:
            st.warning(str(exc))
            raw_strategy_cfg = {}

        repo_root = Path(__file__).resolve().parents[4]
        daily_pv_default = default_daily_pv(raw_strategy_cfg, repo_root)
        logs_base = default_logs_base(raw_strategy_cfg, config_path)

        st.caption("Select backtest range and params. Max 2000 trading days per run. Current version supports QLib inference only. Provide factor and weight files.")
        col1, col2 = st.columns(2)
        today = pd.Timestamp.today().date()
        default_start = today - pd.Timedelta(days=365)
        start_date = col1.date_input("Start date", value=default_start, max_value=today - pd.Timedelta(days=1))
        end_date = col2.date_input("End Date", value=today, max_value=today)

        st.subheader("Backtest Cost Parameters")
        col_cost_1, col_cost_2 = st.columns(2)
        slippage_bps = col_cost_1.number_input("Slippage (bps)", value=5.0, min_value=0.0, step=0.5)
        commission = col_cost_2.number_input("Commission Rate", value=0.001, min_value=0.0, step=0.0001, format="%.4f")

        st.subheader("Signal Source")
        combined_factors_default = repo_root / "ModelInferenceBundle" / "combined_factors_df.parquet"
        combined_factors = st.text_input(
            "combined_factors_df.parquet",
            value=str(combined_factors_default.resolve()),
        )
        params_path = st.text_input(
            "state_dict_cpu.pt path",
            value=str((repo_root / "ModelInferenceBundle" / "state_dict_cpu.pt").resolve()),
        )
        provider_uri = st.text_input(
            "Qlib provider_uri", value=str(Path.home() / ".qlib/qlib_data/cn_data")
        )

        overrides: StrategyOverrides = render_strategy_sections(
            raw_strategy_cfg,
            StrategyDefaults(
                default_daily_pv=daily_pv_default,
                config_path=config_path,
                logs_base=logs_base,
            ),
        )

        run_button = st.button("Run Backtest", type="primary")

        return BacktestInputs(
            start_date=start_date,
            end_date=end_date,
            initial_capital=overrides.initial_capital,
            slippage_bps=slippage_bps,
            commission=commission,
            daily_pv=overrides.daily_pv,
            combined_factors=combined_factors,
            params_path=params_path,
            provider_uri=provider_uri,
            run_button=run_button,
            strategy_config_path=config_path,
            strategy_config=overrides.config,
            backtest_logs_root=overrides.logs_root,
        )

    def _run_backtest(self, inputs: BacktestInputs) -> None:
        with st.spinner("Running backtest..."):
            try:
                portfolio_cfg = (
                    inputs.strategy_config.get("portfolio", {})
                    if isinstance(inputs.strategy_config, dict)
                    else {}
                )
                weights_cfg = (
                    inputs.strategy_config.get("weights", {})
                    if isinstance(inputs.strategy_config, dict)
                    else {}
                )
                order_metadata = (
                    inputs.strategy_config.get("order", {})
                    if isinstance(inputs.strategy_config, dict)
                    else {}
                )
                invest_ratio = float(portfolio_cfg.get("invest_ratio", 0.95)) if isinstance(portfolio_cfg, dict) else 0.95
                confidence_floor = (
                    float(portfolio_cfg.get("confidence_floor", 0.9)) if isinstance(portfolio_cfg, dict) else 0.9
                )
                portfolio_payload: dict[str, Any] = {
                    "invest_ratio": invest_ratio,
                    "confidence_floor": confidence_floor,
                }
                if isinstance(portfolio_cfg, dict):
                    if "topk" in portfolio_cfg:
                        portfolio_payload["topk"] = int(portfolio_cfg.get("topk", 20))
                    if "max_weight" in portfolio_cfg:
                        portfolio_payload["max_weight"] = float(portfolio_cfg.get("max_weight", 0.05))
                    if "lot" in portfolio_cfg:
                        portfolio_payload["lot"] = int(portfolio_cfg.get("lot", 100))
                if isinstance(weights_cfg, dict) and weights_cfg.get("mode"):
                    portfolio_payload["weight_mode"] = str(weights_cfg.get("mode"))

                signal_cfg: dict[str, Any] = {
                    "combined_factors": inputs.combined_factors,
                    "params_path": inputs.params_path,
                    "provider_uri": inputs.provider_uri,
                }

                cfg = BTConfig.from_dict(  # type: ignore[call-arg]
                    {
                        "start_date": inputs.start_date.strftime("%Y-%m-%d"),
                        "end_date": inputs.end_date.strftime("%Y-%m-%d"),
                        "initial_capital": inputs.initial_capital,
                        "portfolio": portfolio_payload,
                        "costs": {"commission": inputs.commission, "slippage_bps": inputs.slippage_bps},
                        "data": {"daily_pv": str(inputs.daily_pv)},
                        "signal": signal_cfg,
                        "metadata": {
                            "config_path": str(inputs.strategy_config_path),
                            "config_payload": inputs.strategy_config,
                            "logs_dir": str(inputs.backtest_logs_root),
                            "order": order_metadata,
                        },
                    }
                )

                engine = BacktestEngine(cfg)
                result = engine.run()

                with st.expander("Backtest Results", expanded=True):
                    self._render_backtest_result(cfg, result)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Backtest failed:{exc}")

    def _render_backtest_result(self, cfg: BTConfig, result) -> None:  # type: ignore[valid-type]
        equity_df = result.equity_curve
        summary = result.summary.to_dict() if hasattr(result.summary, "to_dict") else result.summary

        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        metrics_col1.metric("Total return", format_percent(summary.get("total_return")))
        metrics_col2.metric("Annualized Return", format_percent(summary.get("annualized_return"), placeholder="Less than 21 trading days"))
        metrics_col3.metric("Max Drawdown", format_percent(summary.get("max_drawdown")))

        metrics_col4, metrics_col5 = st.columns(2)
        metrics_col4.metric("Sharpe Ratio", format_ratio(summary.get("sharpe"), placeholder="Insufficient samples"))
        metrics_col5.metric("Return Volatility", format_percent(summary.get("volatility"), placeholder="Insufficient samples"))

        st.line_chart(equity_df[["equity"]])
        st.area_chart(equity_df[["daily_return"]])

        st.markdown("### Trade Details")
        st.dataframe(result.trades)
        st.markdown("### Position History")
        st.dataframe(result.positions)

        st.download_button(
            "Download NAV Curve CSV",
            equity_df.to_csv().encode("utf-8"),
            file_name=f"backtest_equity_{cfg.start_date}_{cfg.end_date}.csv",
            mime="text/csv",
        )

    def _render_history(self) -> None:
        """Render historical backtest records."""
        config_path = resolve_trading_config_path(self._deps.config)
        try:
            strategy_cfg = load_strategy_config(config_path)
        except ConfigLoadError:
            strategy_cfg = {}
        logs_base = default_logs_base(strategy_cfg, config_path)
        paths_cfg = strategy_cfg.get("paths")
        runs_root = infer_backtest_logs_root(
            paths_cfg if isinstance(paths_cfg, dict) else None,
            config_path,
            logs_base,
        )

        if list_backtest_runs is None:
            st.info("History record API unavailable.")
            return
        runs = list_backtest_runs(Path(runs_root))
        if not runs:
            st.caption("No historical records.")
            return

        for meta in runs:
            with st.expander(f"{meta.run_id} | Total return {meta.summary.get('total_return', 0):.2%}"):
                try:
                    (
                        equity,
                        trades_df,
                        positions_df,
                        summary,
                        config,
                        metadata,
                        signals,
                        risk_records,
                        manual_decisions,
                        nav_logs,
                    ) = load_backtest_result(meta.path)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Load failed:{exc}")
                    continue
                st.write("**Backtest Summary**", summary)
                st.write("**Config**", config)
                st.write("**Risk Records**", risk_records[:5])
                st.write("**Signal Records**", signals[:5])
                if manual_decisions:
                    st.write("**Manual Decision Records**", manual_decisions[:5])
                if not equity.empty:
                    st.line_chart(equity["equity"])
                col_download1, col_download2, col_download3 = st.columns(3)
                with col_download1:
                    st.download_button(
                        "Download NAV",
                        equity.to_csv().encode("utf-8"),
                        file_name=f"{meta.run_id}_equity.csv",
                    )
                with col_download2:
                    st.download_button(
                        "Download Trades",
                        trades_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{meta.run_id}_trades.csv",
                    )
                with col_download3:
                    st.download_button(
                        "Download Positions",
                        positions_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{meta.run_id}_positions.csv",
                    )


@dataclass(frozen=True)
class BacktestInputs:
    start_date: date
    end_date: date
    initial_capital: float
    slippage_bps: float
    commission: float
    daily_pv: Path
    combined_factors: str
    params_path: str
    provider_uri: str
    run_button: bool
    strategy_config_path: Path
    strategy_config: dict[str, Any]
    backtest_logs_root: Path
