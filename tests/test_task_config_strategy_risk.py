"""Tests for task_config, strategy weight helpers, and risk manager supplementary."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from fullstackautoquant.model.task_config import (
    TaskConfigError,
    _locate_handler_cfg,
    _normalize_date,
    get_dataset_segments,
    get_dataset_step_len,
    get_training_time_range,
    load_task_config,
)
from fullstackautoquant.trading.strategy import (
    _ranked_weight_candidates,
    _resolve_weight_mode,
    waterfill_with_cap,
)


# ═══════════════════════════════ model/task_config.py ══════════════════


class TestNormalizeDate:
    def test_str(self):
        assert _normalize_date("2024-01-01") == "2024-01-01"

    def test_datetime(self):
        assert _normalize_date(dt.datetime(2024, 1, 1)) == "2024-01-01T00:00:00"

    def test_date(self):
        assert _normalize_date(dt.date(2024, 6, 15)) == "2024-06-15"

    def test_none(self):
        assert _normalize_date(None) == "None"


class TestLoadTaskConfig:
    def test_valid_yaml(self, tmp_path):
        cfg_file = tmp_path / "task.yaml"
        cfg_file.write_text(yaml.dump({"dataset": {"class": "TSDatasetH"}}), encoding="utf-8")
        result = load_task_config(cfg_file)
        assert result["dataset"]["class"] == "TSDatasetH"

    def test_no_file(self, tmp_path):
        with pytest.raises(TaskConfigError, match="Cannot find"):
            load_task_config(tmp_path / "nonexistent.yaml")

    def test_non_dict(self, tmp_path):
        cfg_file = tmp_path / "task.yaml"
        cfg_file.write_text("just a string", encoding="utf-8")
        with pytest.raises(TaskConfigError, match="dictionary"):
            load_task_config(cfg_file)


class TestLocateHandlerCfg:
    def test_valid(self):
        task_cfg: dict[str, Any] = {
            "dataset": {
                "kwargs": {
                    "handler": {"class": "DataHandlerLP", "kwargs": {}}
                }
            }
        }
        result = _locate_handler_cfg(task_cfg)
        assert result["class"] == "DataHandlerLP"

    def test_missing_dataset(self):
        with pytest.raises(TaskConfigError, match="dataset"):
            _locate_handler_cfg({})

    def test_missing_kwargs(self):
        with pytest.raises(TaskConfigError, match="kwargs"):
            _locate_handler_cfg({"dataset": {}})

    def test_missing_handler(self):
        with pytest.raises(TaskConfigError, match="handler"):
            _locate_handler_cfg({"dataset": {"kwargs": {}}})


class TestGetDatasetStepLen:
    def test_present(self):
        assert get_dataset_step_len({"dataset": {"kwargs": {"step_len": 72}}}) == 72

    def test_missing(self):
        assert get_dataset_step_len({"dataset": {"kwargs": {}}}) is None

    def test_string_numeric(self):
        assert get_dataset_step_len({"dataset": {"kwargs": {"step_len": "72"}}}) == 72

    def test_invalid_type(self):
        with pytest.raises(TaskConfigError, match="step_len"):
            get_dataset_step_len({"dataset": {"kwargs": {"step_len": "abc"}}})

    def test_empty_dataset(self):
        assert get_dataset_step_len({}) is None


class TestGetDatasetSegments:
    def test_valid(self):
        cfg: dict[str, Any] = {
            "dataset": {"kwargs": {"segments": {"train": ["2020-01-01", "2021-12-31"]}}}
        }
        result = get_dataset_segments(cfg)
        assert "train" in result

    def test_missing(self):
        with pytest.raises(TaskConfigError, match="segments"):
            get_dataset_segments({"dataset": {"kwargs": {}}})

    def test_non_dict_segments(self):
        with pytest.raises(TaskConfigError, match="segments"):
            get_dataset_segments({"dataset": {"kwargs": {"segments": "invalid"}}})


class TestGetTrainingTimeRange:
    def test_valid(self):
        cfg: dict[str, Any] = {
            "dataset": {
                "kwargs": {
                    "handler": {
                        "kwargs": {
                            "start_time": "2005-01-04",
                            "end_time": "2021-12-31",
                        }
                    }
                }
            }
        }
        result = get_training_time_range(cfg)
        assert result["start_time"] == "2005-01-04"
        assert result["end_time"] == "2021-12-31"

    def test_missing_times(self):
        cfg: dict[str, Any] = {
            "dataset": {"kwargs": {"handler": {"kwargs": {}}}}
        }
        with pytest.raises(TaskConfigError, match="start_time/end_time"):
            get_training_time_range(cfg)

    def test_datetime_objects(self):
        cfg: dict[str, Any] = {
            "dataset": {
                "kwargs": {
                    "handler": {
                        "kwargs": {
                            "start_time": dt.datetime(2005, 1, 4),
                            "end_time": dt.datetime(2021, 12, 31),
                        }
                    }
                }
            }
        }
        result = get_training_time_range(cfg)
        assert "2005" in result["start_time"]


# ═══════════════════════════════ strategy helpers ═════════════════════


class TestResolveWeightMode:
    def test_default_equal(self):
        mode, cfg = _resolve_weight_mode({})
        assert mode == "equal"

    def test_ranked_from_weights(self):
        mode, cfg = _resolve_weight_mode({"weights": {"mode": "ranked"}})
        assert mode == "ranked"

    def test_ranked_from_portfolio(self):
        mode, cfg = _resolve_weight_mode({"portfolio": {"weight_mode": "ranked"}})
        assert mode == "ranked"

    def test_invalid_falls_back_to_equal(self):
        mode, cfg = _resolve_weight_mode({"weights": {"mode": "invalid"}})
        assert mode == "equal"

    def test_non_dict_weights(self):
        mode, cfg = _resolve_weight_mode({"weights": "not_a_dict"})
        assert mode == "equal"


class TestRankedWeightCandidates:
    def test_empty(self):
        result = _ranked_weight_candidates([], {})
        assert len(result) == 0

    def test_rank_metric_default(self):
        signals = [
            {"score": 0.9, "rank": 1},
            {"score": 0.8, "rank": 2},
            {"score": 0.7, "rank": 3},
        ]
        result = _ranked_weight_candidates(signals, {})
        assert len(result) == 3
        # Default is "rank", descending: 3, 2, 1
        assert result[0] > result[2]

    def test_score_metric(self):
        signals = [
            {"score": 0.9, "rank": 1},
            {"score": 0.5, "rank": 2},
        ]
        result = _ranked_weight_candidates(signals, {"rank_metric": "score"})
        assert len(result) == 2

    def test_score_metric_no_spread(self):
        signals = [
            {"score": 0.5, "rank": 1},
            {"score": 0.5, "rank": 2},
        ]
        # When scores are identical, falls back to rank
        result = _ranked_weight_candidates(signals, {"rank_metric": "score"})
        assert len(result) == 2

    def test_exponent(self):
        signals = [
            {"score": 0.9, "rank": 1},
            {"score": 0.8, "rank": 2},
        ]
        result_1 = _ranked_weight_candidates(signals, {"rank_exponent": 1.0})
        result_2 = _ranked_weight_candidates(signals, {"rank_exponent": 2.0})
        # With exponent 2, the ratio is more extreme
        ratio_1 = result_1[0] / result_1[1]
        ratio_2 = result_2[0] / result_2[1]
        assert ratio_2 > ratio_1

    def test_invalid_exponent(self):
        signals = [{"score": 0.9, "rank": 1}]
        result = _ranked_weight_candidates(signals, {"rank_exponent": "abc"})
        assert len(result) == 1


class TestWaterfillWithCap:
    def test_basic_capping(self):
        weights = [1.0, 1.0, 1.0, 1.0]
        result = waterfill_with_cap(weights, max_w=0.3)
        assert len(result) == 4
        assert all(w <= 0.3 + 1e-9 for w in result)
        assert sum(result) <= 1.0 + 1e-9

    def test_no_capping_needed(self):
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = waterfill_with_cap(weights, max_w=0.5)
        # 5 equal weights, each ≤ 0.2 < 0.5 cap
        assert all(w <= 0.5 + 1e-9 for w in result)

    def test_single_element(self):
        result = waterfill_with_cap([1.0], max_w=0.5)
        assert len(result) == 1
        assert result[0] <= 0.5 + 1e-9

    def test_empty(self):
        assert waterfill_with_cap([], max_w=0.5) == []

    def test_cap_at_1(self):
        result = waterfill_with_cap([1.0, 0.5], max_w=1.0)
        assert sum(result) <= 1.0 + 1e-9

    def test_all_zeros(self):
        result = waterfill_with_cap([0.0, 0.0, 0.0], max_w=0.5)
        # waterfill with all-zero weights may return zeros or small values
        assert all(w >= 0 for w in result)

    def test_large_portfolio(self):
        weights = [1.0] * 50
        result = waterfill_with_cap(weights, max_w=0.05)
        assert len(result) == 50
        assert all(w <= 0.05 + 1e-9 for w in result)

    def test_unequal_weights(self):
        weights = [10.0, 5.0, 1.0]
        result = waterfill_with_cap(weights, max_w=0.5)
        assert all(w <= 0.5 + 1e-9 for w in result)
        assert sum(result) <= 1.0 + 1e-9


# ═══════════════════════════════ strategy read_close_prices ═══════════


class TestReadClosePricesFromH5:
    def test_no_file(self, tmp_path):
        from fullstackautoquant.trading.strategy import read_close_prices_from_h5

        result = read_close_prices_from_h5(str(tmp_path / "nonexistent.h5"), [])
        assert result == {}


# ═══════════════════════════════ risk/manager.py main function ════════


class TestRiskManagerMain:
    def test_compute_drawdowns_corrupted_csv(self, tmp_path):
        from fullstackautoquant.trading.risk.manager import compute_drawdowns

        (tmp_path / "nav_history.csv").write_text("bad,data\n1,2\n3,4", encoding="utf-8")
        day_dd, rolling_dd = compute_drawdowns(str(tmp_path))
        # Should handle gracefully
        assert isinstance(day_dd, float)
        assert isinstance(rolling_dd, float)

    def test_compute_drawdowns_old_nav_column(self, tmp_path):
        from fullstackautoquant.trading.risk.manager import compute_drawdowns

        data = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "nav": [100_000, 99_000, 98_000],
        })
        data.to_csv(tmp_path / "nav_history.csv", index=False)
        day_dd, rolling_dd = compute_drawdowns(str(tmp_path))
        assert day_dd > 0  # should detect drawdown


# ═══════════════════════════════ risk/service.py normalize_nav ════════


class TestRiskServiceNormalizeNav:
    def test_normalize_nav_basic(self, tmp_path):
        from fullstackautoquant.trading.risk.service import RiskEvaluatorService, RiskInputs

        nav_data = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "equity": [100_000, 101_000],
        })
        nav_data.to_csv(tmp_path / "nav_history.csv", index=False)
        inputs = RiskInputs(
            signals=[],
            logs_dir=tmp_path,
            risk_config={},
            order_config={},
            paths_config={},
        )
        service = RiskEvaluatorService(inputs)
        nav = service._normalize_nav(nav_data.copy())
        assert len(nav) == 2
        assert float(nav.iloc[0]) == 100_000

    def test_normalize_nav_with_nav_column(self, tmp_path):
        from fullstackautoquant.trading.risk.service import RiskEvaluatorService, RiskInputs

        nav_data = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "nav": [100_000, 101_000],
        })
        inputs = RiskInputs(
            signals=[],
            logs_dir=tmp_path,
            risk_config={},
            order_config={},
            paths_config={},
        )
        service = RiskEvaluatorService(inputs)
        nav = service._normalize_nav(nav_data.copy())
        assert len(nav) == 2

    def test_normalize_nav_empty(self, tmp_path):
        from fullstackautoquant.trading.risk.service import RiskEvaluatorService, RiskInputs

        inputs = RiskInputs(
            signals=[],
            logs_dir=tmp_path,
            risk_config={},
            order_config={},
            paths_config={},
        )
        service = RiskEvaluatorService(inputs)
        nav = service._normalize_nav(pd.DataFrame())
        assert nav.empty

    def test_rolling_drawdown_exceeds(self, tmp_path):
        from fullstackautoquant.trading.risk.service import RiskEvaluatorService, RiskInputs

        # Create NAV with large rolling drawdown (>10%)
        nav_data = pd.DataFrame({
            "date": [f"2024-01-{i:02d}" for i in range(1, 8)],
            "equity": [100_000, 95_000, 92_000, 90_000, 88_000, 87_000, 85_000],
        })
        nav_data.to_csv(tmp_path / "nav_history.csv", index=False)
        inputs = RiskInputs(
            signals=[],
            logs_dir=tmp_path,
            risk_config={"day_drawdown_limit": 0.5, "rolling5d_drawdown_limit": 0.05},
            order_config={},
            paths_config={},
        )
        state = RiskEvaluatorService(inputs).evaluate()
        assert state.allow_buy is False
        assert any("rolling5d" in r for r in state.reasons)
