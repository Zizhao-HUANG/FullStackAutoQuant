"""Tests for backtest utilities, metrics, and config dataclasses."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from fullstackautoquant.backtest.config import (
    BacktestConfig,
    CostParams,
    DataParams,
    ManualWorkflowParams,
    PortfolioParams,
    SignalParams,
    expand_parameter_grid,
)
from fullstackautoquant.backtest.metrics import (
    compute_summary,
    enrich_equity_curve,
    max_drawdown,
)
from fullstackautoquant.backtest.utils import (
    align_to_trading_days,
    compute_annualized_return,
    compute_max_drawdown,
    compute_volatility,
)

# ═══════════════════════════════ backtest/utils.py ═══════════════════


class TestComputeAnnualizedReturn:
    def test_empty(self):
        assert compute_annualized_return([]) == 0.0

    def test_constant_returns(self):
        result = compute_annualized_return([0.01] * 252)
        assert result > 0

    def test_single_return(self):
        result = compute_annualized_return([0.05])
        assert result > 0


class TestComputeVolatility:
    def test_empty(self):
        assert compute_volatility([]) == 0.0

    def test_single(self):
        assert compute_volatility([0.01]) == 0.0

    def test_normal_returns(self):
        returns = [0.01, -0.02, 0.015, -0.005, 0.02]
        vol = compute_volatility(returns)
        assert vol > 0


class TestComputeMaxDrawdown:
    def test_no_drawdown(self):
        assert compute_max_drawdown([100, 110, 120, 130]) == 0.0

    def test_drawdown(self):
        dd = compute_max_drawdown([100, 120, 90, 110])
        assert dd < 0
        assert dd == pytest.approx(-0.25)  # 90/120 - 1

    def test_empty(self):
        assert compute_max_drawdown([]) == 0.0


class TestAlignToTradingDays:
    def test_basic_alignment(self):
        calendar = [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
        series = pd.Series(
            [100.0, 110.0],
            index=[dt.date(2024, 1, 1), dt.date(2024, 1, 3)],
        )
        aligned = align_to_trading_days(calendar, series)
        assert len(aligned) == 3
        assert aligned.iloc[1] == 100.0  # ffill

    def test_bfill(self):
        calendar = [dt.date(2024, 1, 1), dt.date(2024, 1, 2)]
        series = pd.Series([200.0], index=[dt.date(2024, 1, 2)])
        aligned = align_to_trading_days(calendar, series, fill_method="bfill")
        assert aligned.iloc[0] == 200.0

    def test_no_fill(self):
        calendar = [dt.date(2024, 1, 1), dt.date(2024, 1, 2)]
        series = pd.Series([200.0], index=[dt.date(2024, 1, 2)])
        aligned = align_to_trading_days(calendar, series, fill_method=None)
        assert pd.isna(aligned.iloc[0])


# ═══════════════════════════════ backtest/metrics.py ═════════════════


class TestMaxDrawdownFunc:
    def test_basic(self):
        equity = np.array([100, 120, 90, 110, 80])
        dd = max_drawdown(equity)
        assert dd < 0
        assert dd == pytest.approx(-1 / 3)  # 80/120 - 1

    def test_monotone_up(self):
        assert max_drawdown(np.array([100, 200, 300])) == 0.0

    def test_zero_equity(self):
        dd = max_drawdown(np.array([0, 0, 0]))
        assert dd == 0.0


class TestEnrichEquityCurve:
    def test_empty_df(self):
        result = enrich_equity_curve(pd.DataFrame())
        assert result.empty

    def test_adds_daily_return(self):
        df = pd.DataFrame({"equity": [100, 110, 105]}, index=[0, 1, 2])
        result = enrich_equity_curve(df)
        assert "daily_return" in result.columns
        assert result["daily_return"].iloc[0] == 0.0
        assert result["daily_return"].iloc[1] == pytest.approx(0.1)


class TestComputeSummary:
    def test_empty_df(self):
        summary = compute_summary(pd.DataFrame())
        assert summary.total_return == 0.0

    def test_basic_equity(self):
        eq = pd.DataFrame(
            {
                "equity": [100_000.0] + [100_000 * (1.001) ** i for i in range(1, 253)],
            }
        )
        eq = enrich_equity_curve(eq)
        summary = compute_summary(eq)
        assert summary.total_return > 0
        assert summary.max_drawdown <= 0
        # With enough periods, annualized stats should be finite
        assert not np.isnan(summary.annualized_return)
        assert not np.isnan(summary.volatility)
        assert not np.isnan(summary.sharpe)

    def test_short_equity(self):
        eq = pd.DataFrame({"equity": [100, 110]})
        eq = enrich_equity_curve(eq)
        summary = compute_summary(eq)
        assert summary.total_return == pytest.approx(0.1)


# ═══════════════════════════════ backtest/config.py ══════════════════


class TestDataParams:
    def test_from_dict_empty(self):
        dp = DataParams.from_dict({})
        assert dp.daily_pv is None
        assert dp.qlib_root is None

    def test_from_dict_with_paths(self):
        dp = DataParams.from_dict({"daily_pv": "/tmp/data.h5"})
        assert dp.daily_pv is not None

    def test_roundtrip(self):
        dp = DataParams.from_dict({"daily_pv": "/tmp/data.h5"})
        d = dp.to_dict()
        assert "daily_pv" in d


class TestPortfolioParams:
    def test_defaults(self):
        pp = PortfolioParams()
        assert pp.topk == 20
        assert pp.max_weight == 0.05

    def test_from_dict(self):
        pp = PortfolioParams.from_dict({"topk": 50, "max_weight": 0.03})
        assert pp.topk == 50
        assert pp.max_weight == 0.03

    def test_roundtrip(self):
        pp = PortfolioParams.from_dict({"topk": 50})
        d = pp.to_dict()
        pp2 = PortfolioParams.from_dict(d)
        assert pp2.topk == 50


class TestCostParams:
    def test_defaults(self):
        cp = CostParams()
        assert cp.commission == 0.001
        assert cp.stamp_tax == 0.001

    def test_from_dict(self):
        cp = CostParams.from_dict({"commission": 0.002, "slippage_bps": 10.0})
        assert cp.commission == 0.002
        assert cp.slippage_bps == 10.0

    def test_roundtrip(self):
        cp = CostParams()
        d = cp.to_dict()
        assert d["commission"] == 0.001


class TestManualWorkflowParams:
    def test_defaults(self):
        mwp = ManualWorkflowParams()
        assert mwp.enabled is False
        assert mwp.partial_fill_ratio == 1.0

    def test_from_dict_all_fields(self):
        data = {
            "enabled": True,
            "confirm_delay_days": 3,
            "max_signals_per_day": 10,
            "min_confidence": 0.5,
            "record_rejected": False,
            "strategy_field": "sector",
            "strategy_limits": {"tech": 5, "fin": 3},
            "partial_fill_ratio": 0.8,
        }
        mwp = ManualWorkflowParams.from_dict(data)
        assert mwp.enabled is True
        assert mwp.confirm_delay_days == 3
        assert mwp.max_signals_per_day == 10
        assert mwp.min_confidence == 0.5
        assert mwp.strategy_limits == {"tech": 5, "fin": 3}
        assert mwp.partial_fill_ratio == 0.8

    def test_from_dict_edge_cases(self):
        mwp = ManualWorkflowParams.from_dict(
            {
                "partial_fill_ratio": -0.5,
                "max_signals_per_day": None,
                "min_confidence": None,
            }
        )
        assert mwp.partial_fill_ratio == 0.0
        assert mwp.max_signals_per_day is None
        assert mwp.min_confidence is None

    def test_from_non_mapping(self):
        mwp = ManualWorkflowParams.from_dict("not a dict")
        assert mwp.enabled is False

    def test_partial_fill_clamped_high(self):
        mwp = ManualWorkflowParams.from_dict({"partial_fill_ratio": 2.0})
        assert mwp.partial_fill_ratio == 1.0

    def test_roundtrip(self):
        mwp = ManualWorkflowParams.from_dict({"enabled": True, "strategy_limits": {"a": 1}})
        d = mwp.to_dict()
        assert d["strategy_limits"] == {"a": 1}


class TestSignalParams:
    def test_requires_factors_and_params(self):
        with pytest.raises(ValueError, match="requires combined_factors"):
            SignalParams.from_dict({})

    def test_valid(self):
        sp = SignalParams.from_dict(
            {
                "combined_factors": "/tmp/factors.parquet",
                "params_path": "/tmp/params.yaml",
            }
        )
        assert sp.combined_factors is not None
        assert sp.region == "cn"

    def test_roundtrip(self):
        sp = SignalParams.from_dict(
            {
                "combined_factors": "/tmp/factors.parquet",
                "params_path": "/tmp/params.yaml",
                "fallback_days": 5,
            }
        )
        d = sp.to_dict()
        assert d["fallback_days"] == 5


class TestBacktestConfig:
    @pytest.fixture()
    def minimal_config_data(self):
        return {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "signal": {
                "combined_factors": "/tmp/f.parquet",
                "params_path": "/tmp/p.yaml",
            },
        }

    def test_from_dict(self, minimal_config_data):
        cfg = BacktestConfig.from_dict(minimal_config_data)
        assert cfg.start_date == dt.date(2024, 1, 1)
        assert cfg.end_date == dt.date(2024, 12, 31)
        assert cfg.initial_capital == 1_000_000.0

    def test_to_dict_roundtrip(self, minimal_config_data):
        cfg = BacktestConfig.from_dict(minimal_config_data)
        d = cfg.to_dict()
        cfg2 = BacktestConfig.from_dict(d)
        assert cfg2.start_date == cfg.start_date

    def test_with_overrides(self, minimal_config_data):
        cfg = BacktestConfig.from_dict(minimal_config_data)
        cfg2 = cfg.with_overrides({"portfolio.topk": 50, "initial_capital": 500_000})
        assert cfg2.portfolio.topk == 50
        assert cfg2.initial_capital == 500_000
        assert cfg.portfolio.topk == 20  # original unchanged

    def test_apply_overrides_inplace(self, minimal_config_data):
        cfg = BacktestConfig.from_dict(minimal_config_data)
        cfg.apply_overrides_inplace({"portfolio.topk": 30})
        assert cfg.portfolio.topk == 30

    def test_assign_dot_key_nested(self):
        target: dict = {"a": {"b": 1}}
        BacktestConfig._assign_dot_key(target, "a.b", 2)
        assert target["a"]["b"] == 2

    def test_assign_dot_key_new(self):
        target: dict = {}
        BacktestConfig._assign_dot_key(target, "x.y.z", 42)
        assert target["x"]["y"]["z"] == 42


class TestExpandParameterGrid:
    @pytest.fixture()
    def base_config(self):
        return BacktestConfig.from_dict(
            {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "signal": {
                    "combined_factors": "/tmp/f.parquet",
                    "params_path": "/tmp/p.yaml",
                },
            }
        )

    def test_empty_grid(self, base_config):
        configs = list(expand_parameter_grid(base_config, {}))
        assert len(configs) == 1

    def test_single_dim(self, base_config):
        configs = list(expand_parameter_grid(base_config, {"portfolio.topk": [10, 20, 30]}))
        assert len(configs) == 3
        assert configs[0].portfolio.topk == 10
        assert configs[2].portfolio.topk == 30

    def test_multi_dim(self, base_config):
        grid = {"portfolio.topk": [10, 20], "costs.slippage_bps": [2, 5]}
        configs = list(expand_parameter_grid(base_config, grid))
        assert len(configs) == 4
        # Each should have grid metadata
        assert "grid" in configs[0].metadata
