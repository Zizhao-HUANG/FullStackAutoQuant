"""Backtesting engine main entry point."""

from __future__ import annotations

import datetime as dt
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from ..manual_workflow import ManualWorkflowSimulator
from .components.execution import ExecutionEngine
from .components.nav_tracker import NavTracker
from .components.records import (
    BacktestIntermediate,
    BacktestResult,
    PositionSnapshot,
)
from .components.risk_evaluator import RiskEvaluator
from .components.signal_provider import SignalProvider
from .components.strategy_runner import StrategyRunner
from .config import BacktestConfig
from .metrics import compute_summary, enrich_equity_curve
from .pipeline import BacktestContext, BacktestPipeline
from .qlib_adapter import build_qlib_adapter
from .storage import persist_backtest_result

try:
    from fullstackautoquant.trading.utils import ensure_logs_dir, load_config
except ImportError:
    from ..utils import ensure_logs_dir, load_config


class BacktestEngine:
    """Controller that orchestrates signals, risk, rebalancing, execution, and persistence."""

    def __init__(self, config: BacktestConfig):
        self._config = config
        self._symbol_cache: dict[str, str] = {}
        self._close_cache: dict[tuple[dt.date, str], float] = {}
        self._daily_data = self._load_daily_data()
        self._calendar = self._build_calendar()
        self._strategy_config = self._load_strategy_config()
        signal_provider = SignalProvider(
            build_qlib_adapter(config.signal),
            config.portfolio.confidence_floor,
            config.signal.fallback_days,
        )
        nav_tracker = NavTracker()
        manual_simulator = None
        if getattr(config, "manual", None) and config.manual.enabled:
            manual_simulator = ManualWorkflowSimulator(config.manual, self._calendar)
        ctx = BacktestContext(
            calendar=self._calendar,
            signal_provider=signal_provider,
            risk_evaluator=RiskEvaluator(
                self._strategy_config,
                Path(self._strategy_config["paths"]["logs_dir"]),
                bool(config.metadata.get("override_buy", False)),
            ),
            strategy_runner=StrategyRunner(self._strategy_config),
            execution=ExecutionEngine(config.costs),
            nav_tracker=nav_tracker,
            logs_dir=Path(self._config.metadata.get("logs_dir") or Path(__file__).resolve().parents[2] / "logs/backtest"),
            manual_simulator=manual_simulator,
        )
        self._pipeline = BacktestPipeline(ctx)
        self._context = ctx
        self._nav_tracker = nav_tracker

    def run(self) -> BacktestResult:
        intermediate = self._pipeline.run(
            initial_cash=float(self._config.initial_capital),
            portfolio_value_getter=self._portfolio_value,
            snapshot_collector=self._collect_snapshots,
        )
        equity_df = enrich_equity_curve(intermediate.equity_curve_df())
        summary = compute_summary(equity_df)
        metadata = {"config": self._config.to_dict()}
        result = intermediate.to_result(summary=summary, metadata=metadata)

        run_dir = persist_backtest_result(
            Path(self._context.logs_dir),
            self._config.to_dict(),
            summary.to_dict(),
            result.equity_curve,
            result.trades,
            result.positions,
        )
        self._write_artifacts(run_dir, result, intermediate)
        return result

    def _load_daily_data(self) -> pd.DataFrame:
        path = Path(self._config.data.daily_pv or "")
        if not path.exists():
            raise FileNotFoundError(f"daily_pv does not exist: {path}")
        df = pd.read_hdf(path, key="data")
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("daily_pv should have MultiIndex(datetime, instrument)")
        return df

    def _build_calendar(self) -> list[dt.date]:
        if self._config.data.calendar_csv and Path(self._config.data.calendar_csv).exists():
            cal = pd.read_csv(self._config.data.calendar_csv)
            date_col = None
            for candidate in ("date", "trade_date", "cal_date"):
                if candidate in cal.columns:
                    date_col = candidate
                    break
            if date_col is None:
                raise KeyError("Trading calendar is missing 'date' column")
            dates = []
            for val in cal[date_col]:
                text = str(val)
                if len(text) == 8 and text.isdigit():
                    parsed = dt.datetime.strptime(text, "%Y%m%d").date()
                else:
                    parsed = dt.datetime.strptime(text, "%Y-%m-%d").date()
                dates.append(parsed)
        else:
            idx = self._daily_data.index.get_level_values("datetime")
            dates = sorted(pd.to_datetime(idx.unique()).date)
        return [d for d in dates if self._config.start_date <= d <= self._config.end_date]

    def _load_strategy_config(self) -> dict:
        cfg_path = self._config.metadata.get("config_path")
        path = Path(cfg_path) if cfg_path else Path(__file__).resolve().parents[1] / "config.auto.local.yaml"
        cfg = load_config(str(path))

        overrides = self._config.metadata.get("config_payload")
        if isinstance(overrides, dict):
            cfg = self._merge_overrides(cfg, overrides)

        cfg = self._apply_runtime_params(cfg)

        cfg_paths = cfg.setdefault("paths", {})
        daily_pv_path = Path(self._config.data.daily_pv).resolve()
        cfg_paths["daily_pv"] = str(daily_pv_path)

        default_logs_root = Path(__file__).resolve().parents[2] / "logs/backtest"
        logs_hint = self._config.metadata.get("logs_dir")
        logs_base = Path(str(logs_hint)).expanduser().resolve() if logs_hint else default_logs_root

        override_logs_raw = cfg_paths.get("logs_dir")
        if isinstance(override_logs_raw, str) and override_logs_raw.strip():
            override_logs_path = Path(override_logs_raw)
            if not override_logs_path.is_absolute():
                override_logs_path = (path.parent / override_logs_path).resolve()
        else:
            override_logs_path = logs_base

        runtime_dir = (override_logs_path / "runtime").resolve()
        cfg_paths["logs_dir"] = str(runtime_dir)
        ensure_logs_dir(cfg)
        return cfg

    @staticmethod
    def _merge_overrides(base: dict, overrides: dict) -> dict:
        merged = deepcopy(base)

        def _apply(target: dict, patch: dict) -> None:
            for key, value in patch.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    _apply(target[key], value)
                else:
                    target[key] = value

        _apply(merged, overrides)
        return merged

    def _apply_runtime_params(self, cfg: dict) -> dict:
        """Inject BacktestConfig overrides into the strategy config used during execution."""

        portfolio_cfg = cfg.setdefault("portfolio", {})
        bt_portfolio = self._config.portfolio
        portfolio_cfg["topk"] = bt_portfolio.topk
        portfolio_cfg["invest_ratio"] = bt_portfolio.invest_ratio
        portfolio_cfg["max_weight"] = bt_portfolio.max_weight
        portfolio_cfg["lot"] = bt_portfolio.lot
        portfolio_cfg["confidence_floor"] = bt_portfolio.confidence_floor

        weight_mode = (bt_portfolio.weight_mode or "").strip().lower()
        if weight_mode in {"equal", "ranked"}:
            portfolio_cfg["weight_mode"] = weight_mode
            cfg.setdefault("weights", {})["mode"] = weight_mode

        return cfg

    def _portfolio_value(self, date: dt.date, positions: dict[str, float]) -> float:
        total = 0.0
        for symbol, volume in positions.items():
            if volume <= 0:
                continue
            close_price = self._close_price(date, symbol)
            if close_price <= 0:
                continue
            total += volume * close_price
        return total

    def _close_price(self, date: dt.date, symbol: str) -> float:
        cache_key = (date, symbol)
        if cache_key in self._close_cache:
            return self._close_cache[cache_key]

        qlib_symbol = self._to_qlib_symbol(symbol)
        price = self._lookup_close_price(date, qlib_symbol)
        self._close_cache[cache_key] = price
        return price

    def _lookup_close_price(self, date: dt.date, qlib_symbol: str | None) -> float:
        if not qlib_symbol:
            return 0.0
        ts = pd.Timestamp(date)
        try:
            value = self._daily_data.loc[(ts, qlib_symbol), "$close"]  # type: ignore[index]
            return float(value)
        except KeyError:
            try:
                symbol_df = self._daily_data.xs(qlib_symbol, level="instrument")
            except KeyError:
                return 0.0
            if symbol_df.empty:
                return 0.0
            try:
                eligible = symbol_df.loc[symbol_df.index <= ts]
            except Exception:
                return 0.0
            if eligible.empty:
                return 0.0
            try:
                return float(eligible.iloc[-1]["$close"])
            except Exception:
                return 0.0

    def _to_qlib_symbol(self, symbol: str) -> str | None:
        sym = str(symbol or "").upper()
        if not sym:
            return None
        cached = self._symbol_cache.get(sym)
        if cached is not None:
            return cached

        normalized: str | None = None
        if sym.startswith(("SHSE.", "SSE.")):
            code = sym.split(".", 1)[1]
            normalized = f"SH{code}"
        elif sym.startswith(("SZSE.", "SZE.")):
            code = sym.split(".", 1)[1]
            normalized = f"SZ{code}"
        elif sym.startswith(("SH", "SZ")) and len(sym) == 8 and sym[2:].isdigit():
            normalized = sym
        else:
            normalized = sym

        self._symbol_cache[sym] = normalized
        return normalized

    def _collect_snapshots(self, date: dt.date, positions: dict[str, float]) -> list[PositionSnapshot]:
        snaps: list[PositionSnapshot] = []
        for sym, qty in positions.items():
            if qty <= 0:
                continue
            close_price = self._close_price(date, sym)
            snaps.append(
                PositionSnapshot(
                    date=date,
                    symbol=sym,
                    shares=qty,
                    market_value=qty * close_price if close_price > 0 else 0.0,
                )
            )
        return snaps

    def _write_artifacts(self, run_dir: Path, result: BacktestResult, intermediate: BacktestIntermediate) -> None:
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        self._nav_tracker.write_csv(logs_dir / "nav_history.csv")
        (run_dir / "risk_records.json").write_text(
            json.dumps(intermediate.risk_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "signals.json").write_text(
            json.dumps(intermediate.signal_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if intermediate.manual_decisions:
            manual_payload = {
                "version": 1,
                "generated_at": dt.datetime.utcnow().isoformat(),
                "count": len(intermediate.manual_decisions),
                "entries": intermediate.manual_decisions,
            }
            (run_dir / "manual_decisions.json").write_text(
                json.dumps(manual_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        metadata_payload = {
            "config": self._config.to_dict(),
            "nav_history": list(self._nav_tracker.to_rows()),
            "risk_records": intermediate.risk_records,
            "signal_records": intermediate.signal_records,
            "manual_decisions": intermediate.manual_decisions,
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8")
