"""Tests for backtest components: ExecutionEngine, NavTracker, storage, schema."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from fullstackautoquant.backtest.components.execution import ExecutionEngine
from fullstackautoquant.backtest.components.nav_tracker import NavTracker
from fullstackautoquant.backtest.components.records import (
    BacktestIntermediate,
    BacktestResult,
    BacktestSummary,
    DailyEquity,
    TradeRecord,
)
from fullstackautoquant.backtest.storage import (
    ResultSerializer,
    list_backtest_runs,
    load_backtest_result,
    persist_backtest_result,
)


# ═══════════════════════════════ ExecutionEngine ═════════════════════


@dataclass
class MockCosts:
    commission: float = 0.001
    stamp_tax: float = 0.001
    min_commission: float = 5.0


class TestExecutionEngine:
    @pytest.fixture()
    def engine(self):
        return ExecutionEngine(MockCosts())

    def test_sell_order(self, engine):
        orders = [{"symbol": "SH600000", "side": "SELL", "volume": 100, "price": 10.0}]
        positions = {"SH600000": 200.0}
        cash = 100_000.0

        def market_value(date, pos):
            return sum(v * 10.0 for v in pos.values())

        cash_after, pos_after, trades, equity_rec, mv = engine.run(
            dt.date(2024, 1, 1), orders, positions, cash, 100_000.0, market_value
        )
        assert cash_after > cash  # sold, gained money
        assert pos_after["SH600000"] == 100.0
        assert len(trades) == 1
        assert trades[0].side == "SELL"
        assert trades[0].fee > 0

    def test_buy_order(self, engine):
        orders = [{"symbol": "SH600000", "side": "BUY", "volume": 100, "price": 10.0}]
        positions: dict[str, float] = {}
        cash = 100_000.0

        def market_value(date, pos):
            return sum(v * 10.0 for v in pos.values())

        cash_after, pos_after, trades, equity_rec, mv = engine.run(
            dt.date(2024, 1, 1), orders, positions, cash, 100_000.0, market_value
        )
        assert cash_after < cash
        assert pos_after["SH600000"] == 100.0
        assert len(trades) == 1
        assert trades[0].side == "BUY"

    def test_buy_insufficient_cash(self, engine):
        orders = [{"symbol": "SH600000", "side": "BUY", "volume": 10000, "price": 100.0}]
        positions: dict[str, float] = {}
        cash = 1_000.0  # not enough

        def market_value(date, pos):
            return 0.0

        cash_after, pos_after, trades, equity_rec, mv = engine.run(
            dt.date(2024, 1, 1), orders, positions, cash, 1_000.0, market_value
        )
        assert cash_after == cash  # no trade
        assert len(trades) == 0

    def test_invalid_orders_skipped(self, engine):
        orders = [
            {"symbol": "SH600000", "side": "BUY", "volume": 0, "price": 10.0},  # zero volume
            {"symbol": "SH600000", "side": "BUY", "volume": 100, "price": 0.0},  # zero price
        ]
        positions: dict[str, float] = {}

        def market_value(date, pos):
            return 0.0

        cash_after, pos_after, trades, equity_rec, mv = engine.run(
            dt.date(2024, 1, 1), orders, positions, 100_000.0, 100_000.0, market_value
        )
        assert len(trades) == 0

    def test_equity_record_daily_return(self, engine):
        # With no orders, equity should be unchanged
        def market_value(date, pos):
            return 0.0

        cash_after, pos_after, trades, equity_rec, mv = engine.run(
            dt.date(2024, 1, 1), [], {}, 100_000.0, 100_000.0, market_value
        )
        assert equity_rec.daily_return == pytest.approx(0.0)


# ═══════════════════════════════ NavTracker ══════════════════════════


class TestNavTracker:
    def test_append_and_to_rows(self):
        tracker = NavTracker()
        tracker.append(dt.date(2024, 1, 1), equity=100_000.0, cash=50_000.0, market_value=50_000.0)
        tracker.append(dt.date(2024, 1, 2), equity=102_000.0)
        rows = list(tracker.to_rows())
        assert len(rows) == 2
        assert rows[0]["equity"] == 100_000.0
        assert rows[0]["cash"] == 50_000.0
        assert rows[1]["equity"] == 102_000.0

    def test_to_dataframe(self):
        tracker = NavTracker()
        tracker.append(dt.date(2024, 1, 1), equity=100_000.0)
        df = tracker.to_dataframe()
        assert len(df) == 1
        assert "equity" in df.columns
        assert "nav" in df.columns

    def test_ensure_previous(self):
        tracker = NavTracker()
        tracker.ensure_previous(dt.date(2024, 1, 2), prev_equity=99_000.0)
        rows = list(tracker.to_rows())
        assert len(rows) == 1
        assert rows[0]["date"] == "2024-01-01"  # previous day

    def test_ensure_previous_none(self):
        tracker = NavTracker()
        tracker.ensure_previous(dt.date(2024, 1, 2), prev_equity=None)
        rows = list(tracker.to_rows())
        assert len(rows) == 0

    def test_update_components(self):
        tracker = NavTracker()
        tracker.update_components(dt.date(2024, 1, 1), cash=60_000.0, market_value=40_000.0)
        rows = list(tracker.to_rows())
        assert rows[0]["equity"] == 100_000.0  # auto-computed
        assert rows[0]["cash"] == 60_000.0

    def test_write_csv(self, tmp_path):
        tracker = NavTracker()
        tracker.append(dt.date(2024, 1, 1), equity=100_000.0)
        target = tmp_path / "nav.csv"
        tracker.write_csv(target)
        assert target.exists()
        df = pd.read_csv(target)
        assert len(df) == 1

    def test_write_csv_empty(self, tmp_path):
        tracker = NavTracker()
        target = tmp_path / "empty_nav.csv"
        tracker.write_csv(target)
        assert target.exists()
        assert target.read_text() == ""


# ═══════════════════════════════ Storage ═════════════════════════════


class TestResultSerializer:
    def test_persist_creates_files(self, tmp_path):
        serializer = ResultSerializer(tmp_path)
        equity = pd.DataFrame({"equity": [100_000, 101_000]}, index=[0, 1])
        trades = pd.DataFrame({"symbol": ["SH600000"], "side": ["BUY"], "volume": [100], "price": [10.0]})
        positions = pd.DataFrame({"symbol": ["SH600000"], "shares": [100]})
        config = {"topk": 20}
        summary = {"total_return": 0.01}

        result = serializer.persist(config, summary, equity, trades, positions)
        assert result.root.exists()
        assert (result.root / "config.json").exists()
        assert (result.root / "summary.json").exists()
        assert (result.root / "equity_curve.csv").exists()
        assert (result.root / "trades.csv").exists()
        assert (result.root / "positions.csv").exists()

    def test_persist_with_artifacts(self, tmp_path):
        serializer = ResultSerializer(tmp_path)
        artifacts = {
            "extra.json": {"key": "value"},
            "signals.csv": pd.DataFrame({"sym": ["A", "B"]}),
        }
        result = serializer.persist(
            {}, {}, pd.DataFrame({"equity": [100]}), pd.DataFrame(), pd.DataFrame(), artifacts
        )
        assert (result.logs_dir / "extra.json").exists()
        assert (result.logs_dir / "signals.csv").exists()

    def test_persist_empty_dataframes(self, tmp_path):
        serializer = ResultSerializer(tmp_path)
        result = serializer.persist({}, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        assert result.root.exists()


class TestPersistBacktestResult:
    def test_returns_path(self, tmp_path):
        result_path = persist_backtest_result(
            tmp_path,
            {"a": 1},
            {"total_return": 0.05},
            pd.DataFrame({"equity": [100, 105]}),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        assert result_path.exists()
        assert (result_path / "summary.json").exists()


class TestListBacktestRuns:
    def test_empty_dir(self, tmp_path):
        runs = list_backtest_runs(tmp_path)
        assert runs == []

    def test_nonexistent_dir(self, tmp_path):
        runs = list_backtest_runs(tmp_path / "no_such_dir")
        assert runs == []

    def test_lists_valid_runs(self, tmp_path):
        run_dir = tmp_path / "20240101-120000"
        run_dir.mkdir()
        (run_dir / "config.json").write_text("{}", encoding="utf-8")
        (run_dir / "summary.json").write_text('{"total_return": 0.05}', encoding="utf-8")
        runs = list_backtest_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0].run_id == "20240101-120000"

    def test_skips_invalid_runs(self, tmp_path):
        run_dir = tmp_path / "invalid"
        run_dir.mkdir()
        # no summary.json
        (run_dir / "config.json").write_text("{}", encoding="utf-8")
        runs = list_backtest_runs(tmp_path)
        assert len(runs) == 0


class TestLoadBacktestResult:
    def test_load_full(self, tmp_path):
        # Create a valid run directory — equity_curve has index col
        eq = pd.DataFrame({"equity": [100_000, 101_000]}, index=pd.Index(["2024-01-01", "2024-01-02"]))
        eq.to_csv(tmp_path / "equity_curve.csv")
        (tmp_path / "trades.csv").write_text("symbol,side\nSH600000,BUY", encoding="utf-8")
        (tmp_path / "positions.csv").write_text("symbol,shares\nSH600000,100", encoding="utf-8")
        (tmp_path / "summary.json").write_text('{"total_return": 0.05}', encoding="utf-8")
        (tmp_path / "config.json").write_text('{"topk": 20}', encoding="utf-8")

        result = load_backtest_result(tmp_path)
        equity, trades, positions, summary, config, *rest = result
        assert not equity.empty
        assert summary["total_return"] == 0.05
        assert config["topk"] == 20

    def test_load_missing_files(self, tmp_path):
        result = load_backtest_result(tmp_path)
        equity, trades, positions, summary, config, *rest = result
        assert equity.empty
        assert summary == {}
