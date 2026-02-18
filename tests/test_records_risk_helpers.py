"""Tests for backtest records, risk evaluator, config date parsing, run_backtest helpers."""

from __future__ import annotations

import datetime as dt
import json
from math import isfinite, nan
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fullstackautoquant.backtest.components.records import (
    BacktestIntermediate,
    BacktestResult,
    BacktestSummary,
    DailyEquity,
    PositionSnapshot,
    TradeRecord,
)
from fullstackautoquant.backtest.config import _as_path, _to_date


# ═══════════════════════════════ config helpers ══════════════════════


class TestToDate:
    def test_date_obj(self):
        d = dt.date(2024, 1, 1)
        assert _to_date(d) == d

    def test_datetime_obj(self):
        # datetime is subclass of date, so _to_date returns it directly
        d = dt.datetime(2024, 6, 15, 12, 0, 0)
        result = _to_date(d)
        # Result is a datetime (which IS a date subclass)
        assert isinstance(result, dt.date)
        assert result.year == 2024 and result.month == 6 and result.day == 15

    def test_string(self):
        assert _to_date("2024-03-20") == dt.date(2024, 3, 20)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _to_date("")

    def test_whitespace_string(self):
        with pytest.raises(ValueError, match="empty"):
            _to_date("   ")

    def test_invalid_type(self):
        with pytest.raises(TypeError, match="Cannot parse"):
            _to_date(12345)


class TestAsPath:
    def test_none(self):
        assert _as_path(None) is None

    def test_empty_str(self):
        assert _as_path("") is None

    def test_path_obj(self):
        p = Path("/tmp/test")
        assert _as_path(p) == p

    def test_string(self):
        result = _as_path("/tmp/test")
        assert isinstance(result, Path)

    def test_home_expansion(self):
        result = _as_path("~/test")
        assert isinstance(result, Path)
        assert "~" not in str(result)


# ═══════════════════════════════ records.py ══════════════════════════


class TestTradeRecord:
    def test_to_dict(self):
        tr = TradeRecord(
            date=dt.date(2024, 1, 1),
            symbol="SH600000",
            side="BUY",
            volume=100,
            price=10.0,
            fee=5.0,
        )
        d = tr.to_dict()
        assert d["symbol"] == "SH600000"
        assert d["volume"] == 100


class TestPositionSnapshot:
    def test_to_dict(self):
        ps = PositionSnapshot(
            date=dt.date(2024, 1, 1),
            symbol="SH600000",
            shares=100,
            market_value=1000.0,
        )
        d = ps.to_dict()
        assert d["shares"] == 100


class TestBacktestSummary:
    def test_to_dict_finite(self):
        s = BacktestSummary(0.1, 0.15, -0.05, 1.5, 0.2, 3.0)
        d = s.to_dict()
        assert d["total_return"] == 0.1
        assert d["sharpe"] == 1.5

    def test_to_dict_nan(self):
        s = BacktestSummary(0.1, float("nan"), -0.05, float("nan"), float("nan"), float("nan"))
        d = s.to_dict()
        assert d["total_return"] == 0.1
        assert d["annualized_return"] is None
        assert d["sharpe"] is None


class TestDailyEquity:
    def test_to_dict(self):
        de = DailyEquity(
            date=dt.date(2024, 1, 1),
            cash=50_000.0,
            market_value=50_000.0,
            equity=100_000.0,
            daily_return=0.01,
        )
        d = de.to_dict()
        assert d["equity"] == 100_000.0


class TestBacktestResult:
    def test_with_metadata(self):
        result = BacktestResult(
            equity_curve=pd.DataFrame(),
            trades=pd.DataFrame(),
            positions=pd.DataFrame(),
            summary=BacktestSummary(0.1, 0.15, -0.05, 1.5, 0.2, 3.0),
            metadata={"key1": "val1"},
        )
        updated = result.with_metadata({"key2": "val2"})
        assert "key1" in updated.metadata
        assert "key2" in updated.metadata


class TestBacktestIntermediate:
    def test_equity_curve_df_empty(self):
        inter = BacktestIntermediate(
            equities=[],
            trades=[],
            snapshots=[],
            risk_records=[],
            signal_records=[],
            manual_decisions=[],
        )
        df = inter.equity_curve_df()
        assert df.empty

    def test_equity_curve_df_with_data(self):
        equities = [
            DailyEquity(dt.date(2024, 1, 1), 50000, 50000, 100000, 0.0),
            DailyEquity(dt.date(2024, 1, 2), 51000, 50500, 101500, 0.015),
        ]
        inter = BacktestIntermediate(equities, [], [], [], [], [])
        df = inter.equity_curve_df()
        assert len(df) == 2
        assert "equity" in df.columns

    def test_to_result(self):
        equities = [
            DailyEquity(dt.date(2024, 1, 1), 50000, 50000, 100000, 0.0),
        ]
        trades = [
            TradeRecord(dt.date(2024, 1, 1), "SH600000", "BUY", 100, 10.0, 5.0),
        ]
        snapshots = [
            PositionSnapshot(dt.date(2024, 1, 1), "SH600000", 100, 1000.0),
        ]
        inter = BacktestIntermediate(equities, trades, snapshots, [], [], [])
        summary = BacktestSummary(0.1, 0.15, -0.05, 1.5, 0.2, 3.0)
        result = inter.to_result(summary, {"test": True})
        assert not result.equity_curve.empty
        assert not result.trades.empty
        assert not result.positions.empty
        assert result.metadata["test"] is True


# ═══════════════════════════════ risk_evaluator.py ══════════════════


class TestRiskEvaluator:
    def test_decide_buy_normal(self, tmp_path):
        from fullstackautoquant.backtest.components.risk_evaluator import RiskEvaluator

        evaluator = RiskEvaluator(
            config={"risk": {"day_drawdown_limit": 0.03, "rolling5d_drawdown_limit": 0.08}},
            logs_dir=tmp_path,
            allow_override=False,
        )
        assert evaluator._decide_buy(0.01, 0.02) is True
        assert evaluator._decide_buy(0.05, 0.02) is False  # day exceeds
        assert evaluator._decide_buy(0.01, 0.10) is False  # rolling exceeds

    def test_decide_buy_override(self, tmp_path):
        from fullstackautoquant.backtest.components.risk_evaluator import RiskEvaluator

        evaluator = RiskEvaluator(
            config={"risk": {"day_drawdown_limit": 0.03}},
            logs_dir=tmp_path,
            allow_override=True,
        )
        assert evaluator._decide_buy(0.05, 0.10) is True  # forced override

    def test_detect_limits_no_signals(self, tmp_path):
        from fullstackautoquant.backtest.components.risk_evaluator import RiskEvaluator

        evaluator = RiskEvaluator(config={}, logs_dir=tmp_path, allow_override=False)
        lu, ld = evaluator._detect_limits([])
        assert lu == []
        assert ld == []

    def test_detect_limits_no_h5(self, tmp_path):
        from fullstackautoquant.backtest.components.risk_evaluator import RiskEvaluator

        evaluator = RiskEvaluator(
            config={"paths": {}, "order": {}},
            logs_dir=tmp_path,
            allow_override=False,
        )
        lu, ld = evaluator._detect_limits([{"instrument": "SH600000"}])
        assert lu == []

    def test_ensure_logs_dir_with_config(self, tmp_path):
        from fullstackautoquant.backtest.components.risk_evaluator import RiskEvaluator

        logs = tmp_path / "custom_logs"
        evaluator = RiskEvaluator(
            config={"paths": {"logs_dir": str(logs)}},
            logs_dir=tmp_path,
            allow_override=False,
        )
        result = evaluator._ensure_logs_dir()
        assert result == logs
        assert logs.exists()

    def test_ensure_logs_dir_default(self, tmp_path):
        from fullstackautoquant.backtest.components.risk_evaluator import RiskEvaluator

        evaluator = RiskEvaluator(config={}, logs_dir=tmp_path, allow_override=False)
        result = evaluator._ensure_logs_dir()
        assert result == tmp_path

    def test_write_limit_log(self, tmp_path):
        from fullstackautoquant.backtest.components.risk_evaluator import RiskEvaluator

        evaluator = RiskEvaluator(config={}, logs_dir=tmp_path, allow_override=False)
        record = {"trade_date": "2024-01-01", "allow_buy": True}
        evaluator._write_limit_log(record)
        log_path = tmp_path / "limit_states.json"
        assert log_path.exists()
        data = json.loads(log_path.read_text())
        assert "2024-01-01" in data


# ═══════════════════════════════ run_backtest helpers ═════════════════


class TestRunBacktestHelpers:
    def test_parse_value(self):
        from fullstackautoquant.backtest.run_backtest import _parse_value

        assert _parse_value("42") == 42
        assert _parse_value("3.14") == 3.14
        assert _parse_value("true") is True
        assert _parse_value("false") is False
        assert _parse_value("hello") == "hello"

    def test_load_config_yaml(self, tmp_path):
        from fullstackautoquant.backtest.run_backtest import _load_config

        import yaml

        cfg_file = tmp_path / "test.yaml"
        data = {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "signal": {
                "combined_factors": "/tmp/f.parquet",
                "params_path": "/tmp/p.yaml",
            },
        }
        cfg_file.write_text(yaml.dump(data), encoding="utf-8")
        cfg = _load_config(cfg_file)
        assert cfg.start_date == dt.date(2024, 1, 1)

    def test_load_config_json(self, tmp_path):
        from fullstackautoquant.backtest.run_backtest import _load_config

        cfg_file = tmp_path / "test.json"
        data = {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "signal": {
                "combined_factors": "/tmp/f.parquet",
                "params_path": "/tmp/p.yaml",
            },
        }
        cfg_file.write_text(json.dumps(data), encoding="utf-8")
        cfg = _load_config(cfg_file)
        assert cfg.end_date == dt.date(2024, 12, 31)

    def test_apply_overrides(self):
        from fullstackautoquant.backtest.config import BacktestConfig
        from fullstackautoquant.backtest.run_backtest import _apply_overrides

        cfg = BacktestConfig.from_dict({
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "signal": {
                "combined_factors": "/tmp/f.parquet",
                "params_path": "/tmp/p.yaml",
            },
        })
        cfg2 = _apply_overrides(cfg, ["portfolio.topk=50", "initial_capital=500000"])
        assert cfg2.portfolio.topk == 50
        assert cfg2.initial_capital == 500_000

    def test_apply_overrides_invalid_format(self):
        from fullstackautoquant.backtest.config import BacktestConfig
        from fullstackautoquant.backtest.run_backtest import _apply_overrides

        cfg = BacktestConfig.from_dict({
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "signal": {
                "combined_factors": "/tmp/f.parquet",
                "params_path": "/tmp/p.yaml",
            },
        })
        with pytest.raises(ValueError, match="Invalid override format"):
            _apply_overrides(cfg, ["portfolio.topk"])


# ═══════════════════════════════ model scoring ═══════════════════════


class TestModelScoring:
    def test_compute_confidence(self):
        import numpy as np

        from fullstackautoquant.model.scoring import compute_confidence

        preds = [np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 2.9])]
        conf = compute_confidence(preds)
        assert len(conf) == 3
        assert all(0 < c <= 1.0 for c in conf)

    def test_compute_confidence_single_run(self):
        import numpy as np

        from fullstackautoquant.model.scoring import compute_confidence

        preds = [np.array([1.0, 2.0, 3.0])]
        conf = compute_confidence(preds)
        assert len(conf) == 3
        assert all(c == 1.0 for c in conf)  # zero std => confidence = 1

    def test_rank_signals(self):
        import numpy as np

        from fullstackautoquant.model.scoring import rank_signals

        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-01"), "SH600000"),
                (pd.Timestamp("2024-01-01"), "SZ000001"),
            ],
            names=["datetime", "instrument"],
        )
        pred = pd.Series([0.8, 0.5], index=index)
        pred_day, used_date = rank_signals(pred, "2024-01-01")
        assert len(pred_day) == 2
        assert used_date == "2024-01-01"


# ═══════════════════════════════ model archive ═══════════════════════


class TestModelArchive:
    def test_local_archive(self, tmp_path):
        from fullstackautoquant.model.archive.writers import LocalArchive

        # Create a dummy CSV file
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("col1,col2\n1,2", encoding="utf-8")

        archive = LocalArchive(tmp_path / "archive")
        result = archive.persist(csv_path, "2024-01-01")
        assert result.exists()
