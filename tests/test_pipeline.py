from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from fullstackautoquant.backtest.components.execution import ExecutionEngine
from fullstackautoquant.backtest.components.nav_tracker import NavTracker
from fullstackautoquant.backtest.components.records import BacktestIntermediate
from fullstackautoquant.backtest.components.risk_evaluator import RiskEvaluator
from fullstackautoquant.backtest.components.signal_provider import SignalProvider
from fullstackautoquant.backtest.components.strategy_runner import StrategyRunner
from fullstackautoquant.backtest.metrics import enrich_equity_curve
from fullstackautoquant.backtest.pipeline import BacktestContext, BacktestPipeline
from fullstackautoquant.backtest.storage import ResultSerializer


@pytest.fixture
def minimal_context(tmp_path: Path) -> BacktestContext:
    class DummyAdapter:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(cache_dir=None)

        def iterate(self, calendar):
            for d in calendar:
                yield d, d, []

        def generate_for_date(self, candidate):
            return candidate, []

    strategy_config = {
        "portfolio": {
            "topk": 1,
            "max_weight": 0.1,
            "invest_ratio": 1.0,
            "lot": 100,
            "confidence_floor": 0.0,
            "n_drop": 0,
        },
        "order": {
            "price_source": "qlib_close",
            "limit_threshold": 0.095,
            "buy_limit_offset": 0.07,
            "sell_limit_offset": -0.05,
            "min_buy_notional": 0.0,
            "mode": "auto",
            "clamp_tick": 0.01,
        },
        "paths": {
            "daily_pv": str(tmp_path / "daily_pv.h5"),
        },
    }

    signal_provider = SignalProvider(DummyAdapter(), 0.0, 0)
    risk = RiskEvaluator(strategy_config, tmp_path, False)
    strategy = StrategyRunner(strategy_config)
    execution = ExecutionEngine({})
    nav_tracker = NavTracker()
    calendar = [dt.date(2025, 10, 17)]
    return BacktestContext(calendar, signal_provider, risk, strategy, execution, nav_tracker, tmp_path)


def test_pipeline_produces_intermediate(minimal_context: BacktestContext) -> None:
    pipeline = BacktestPipeline(minimal_context)

    def portfolio_value(date, positions):
        return 0.0

    def collector(date, positions):
        return []

    intermediate = pipeline.run(1_000_000.0, portfolio_value, collector)
    assert isinstance(intermediate, BacktestIntermediate)
    df = enrich_equity_curve(intermediate.equity_curve_df())
    assert "daily_return" in df.columns


def test_result_serializer_writes_files(tmp_path: Path, minimal_context: BacktestContext) -> None:
    serializer = ResultSerializer(tmp_path)
    config = {"foo": "bar"}
    summary = {"total_return": 0.1}
    equity = pd.DataFrame({"equity": [1_000_000, 1_010_000]}, index=["2025-10-17", "2025-10-20"])
    trades = pd.DataFrame()
    positions = pd.DataFrame()

    bundle = serializer.persist(config, summary, equity, trades, positions, artifacts={"signals.json": []})

    assert (bundle.root / "config.json").exists()
    assert (bundle.root / "equity_curve.csv").exists()
    assert (bundle.logs_dir / "signals.json").exists()



