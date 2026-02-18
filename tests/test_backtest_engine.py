"""BacktestEngine basic regression tests."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
TRADING_ROOT = ROOT / "Trading" / "trading"
MODEL_ROOT = ROOT / "ModelInferenceBundle"
for path in (ROOT, TRADING_ROOT, MODEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fullstackautoquant.backtest.config import BacktestConfig


def _formal_config() -> BacktestConfig:
    deploy_root = ROOT
    mib_root = deploy_root / "ModelInferenceBundle"
    cfg = BacktestConfig.from_dict(
        {
            "start_date": "2025-09-01",
            "end_date": "2025-09-05",
            "initial_capital": 1_000_000,
            "data": {"daily_pv": str(mib_root / "daily_pv.h5")},
            "signal": {
                "combined_factors": str(mib_root / "combined_factors_df.parquet"),
                "params_path": str(mib_root / "state_dict_cpu.pt"),
                "provider_uri": str(Path.home() / ".qlib/qlib_data/cn_data"),
            },
        }
    )
    return cfg


@pytest.mark.skipif(
    not (ROOT / "Trading").is_dir(),
    reason="Legacy Trading module not available (CI environment)",
)
def test_engine_with_formal_data() -> None:
    engine_module = importlib.import_module("Trading.trading.backtest.engine")
    BacktestEngine = engine_module.BacktestEngine
    cfg = _formal_config()
    engine = BacktestEngine(cfg)
    result = engine.run()
    assert result.summary.total_return is not None
    assert "risk_records" in result.metadata
    assert not result.equity_curve.empty


@pytest.mark.skipif(
    not (ROOT / "Trading").is_dir(),
    reason="Legacy Trading module not available (CI environment)",
)
def test_strategy_config_reflects_portfolio_overrides() -> None:
    engine_module = importlib.import_module("Trading.trading.backtest.engine")
    BacktestEngine = engine_module.BacktestEngine
    cfg = _formal_config().with_overrides(
        {
            "portfolio.topk": 7,
            "portfolio.invest_ratio": 0.5,
            "portfolio.max_weight": 0.02,
            "portfolio.lot": 200,
            "portfolio.weight_mode": "ranked",
        }
    )
    engine = BacktestEngine(cfg)
    strat_cfg = engine._strategy_config  # noqa: SLF001 - internal attribute for verification
    assert strat_cfg["portfolio"]["topk"] == 7
    assert abs(strat_cfg["portfolio"]["invest_ratio"] - 0.5) < 1e-9
    assert abs(strat_cfg["portfolio"]["max_weight"] - 0.02) < 1e-9
    assert strat_cfg["portfolio"]["lot"] == 200
    assert strat_cfg["portfolio"]["weight_mode"] == "ranked"
    assert strat_cfg["weights"]["mode"] == "ranked"

