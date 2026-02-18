"""Tests for model architecture, risk service, schema, strategy helpers, and risk manager."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from fullstackautoquant.model.architecture import Net
from fullstackautoquant.trading.risk.service import (
    RiskEvaluatorService,
    RiskInputs,
)
from fullstackautoquant.trading.signals.schema import (
    _resolve_schema_path,
    load_schema,
    validate_config,
)

# ═══════════════════════════════ model/architecture.py ════════════════


class TestNetArchitecture:
    def test_forward_default_features(self):
        model = Net(num_features=22, num_timesteps=72)
        model.eval()
        x = torch.randn(4, 72, 22)  # batch=4, timesteps=72, features=22
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 1)

    def test_forward_small(self):
        model = Net(num_features=6, num_timesteps=16)
        model.eval()
        x = torch.randn(2, 16, 6)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1)

    def test_model_has_parameters(self):
        model = Net()
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_training_mode_with_dropout(self):
        model = Net(num_features=6, num_timesteps=16)
        model.train()
        x = torch.randn(2, 16, 6)
        out = model(x)
        assert out.shape == (2, 1)


# ═══════════════════════════════ trading/signals/schema.py ═══════════


class TestSignalSchema:
    def test_resolve_schema_path(self):
        path = _resolve_schema_path()
        assert path.exists()
        assert path.name == "trading.schema.json"

    def test_load_schema(self):
        schema = load_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema

    def test_validate_config_valid(self):
        config = {
            "signal": {
                "source": {"type": "csv"},
                "preprocessors": [],
            },
            "risk": {"max_drawdown": 0.1},
            "strategy": {"name": "topk"},
            "execution": {"broker": "gm"},
        }
        result = validate_config(config)
        assert result == config

    def test_validate_config_invalid(self):
        import jsonschema as _jsonschema

        with pytest.raises(_jsonschema.ValidationError):
            validate_config({"invalid": True})


# ═══════════════════════════════ trading/risk/service.py ══════════════


class TestRiskEvaluatorService:
    def test_no_nav_history(self, tmp_path):
        inputs = RiskInputs(
            signals=[],
            logs_dir=tmp_path,
            risk_config={"day_drawdown_limit": 0.03, "rolling5d_drawdown_limit": 0.08},
            order_config={},
            paths_config={},
        )
        service = RiskEvaluatorService(inputs)
        state = service.evaluate()
        assert state.allow_buy is True
        assert state.day_drawdown == 0.0

    def test_with_nav_history_no_drawdown(self, tmp_path):
        nav_data = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "equity": [100_000, 101_000, 102_000],
            }
        )
        nav_data.to_csv(tmp_path / "nav_history.csv", index=False)
        inputs = RiskInputs(
            signals=[],
            logs_dir=tmp_path,
            risk_config={"day_drawdown_limit": 0.03},
            order_config={},
            paths_config={},
        )
        state = RiskEvaluatorService(inputs).evaluate()
        assert state.allow_buy is True

    def test_with_drawdown_exceeds_limit(self, tmp_path):
        nav_data = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "equity": [100_000, 95_000],  # 5% drawdown > 3%
            }
        )
        nav_data.to_csv(tmp_path / "nav_history.csv", index=False)
        inputs = RiskInputs(
            signals=[],
            logs_dir=tmp_path,
            risk_config={"day_drawdown_limit": 0.03, "rolling5d_drawdown_limit": 0.08},
            order_config={},
            paths_config={},
        )
        state = RiskEvaluatorService(inputs).evaluate()
        assert state.allow_buy is False
        assert any("day_drawdown_exceed" in r for r in state.reasons)

    def test_override_buy_forces_allow(self, tmp_path):
        nav_data = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "equity": [100_000, 95_000],
            }
        )
        nav_data.to_csv(tmp_path / "nav_history.csv", index=False)
        inputs = RiskInputs(
            signals=[],
            logs_dir=tmp_path,
            risk_config={"day_drawdown_limit": 0.03},
            order_config={},
            paths_config={},
            override_buy=True,
        )
        state = RiskEvaluatorService(inputs).evaluate()
        assert state.allow_buy is True
        assert "override_buy=true" in state.reasons

    def test_instrument_list_extraction(self, tmp_path):
        inputs = RiskInputs(
            signals=[
                {"symbol": "SHSE.600000"},
                {"symbol": "SZSE.000001"},
                {"symbol": ""},  # empty
                {},  # no symbol key
            ],
            logs_dir=tmp_path,
            risk_config={},
            order_config={},
            paths_config={},
        )
        service = RiskEvaluatorService(inputs)
        instruments = service._instrument_list()
        assert "SH600000" in instruments
        assert "SZ000001" in instruments
        assert len(instruments) == 2

    def test_limit_states_manual_mode_disabled(self, tmp_path):
        inputs = RiskInputs(
            signals=[{"symbol": "SHSE.600000"}],
            logs_dir=tmp_path,
            risk_config={"enforce_limit_up_down_filter": True},
            order_config={"mode": "manual"},
            paths_config={},
        )
        service = RiskEvaluatorService(inputs)
        lu, ld = service._detect_limit_states()
        assert lu == []
        assert ld == []


# ═══════════════════════════════ trading/risk/manager.py ══════════════


class TestComputeDrawdowns:
    def test_no_file(self, tmp_path):
        from fullstackautoquant.trading.risk.manager import compute_drawdowns

        day_dd, rolling_dd = compute_drawdowns(str(tmp_path))
        assert day_dd == 0.0
        assert rolling_dd == 0.0

    def test_empty_csv(self, tmp_path):
        from fullstackautoquant.trading.risk.manager import compute_drawdowns

        (tmp_path / "nav_history.csv").write_text("date,equity\n", encoding="utf-8")
        day_dd, rolling_dd = compute_drawdowns(str(tmp_path))
        assert day_dd == 0.0

    def test_with_drawdown(self, tmp_path):
        from fullstackautoquant.trading.risk.manager import compute_drawdowns

        data = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "equity": [100_000, 95_000, 90_000],
            }
        )
        data.to_csv(tmp_path / "nav_history.csv", index=False)
        day_dd, rolling_dd = compute_drawdowns(str(tmp_path))
        assert day_dd > 0
        assert rolling_dd > 0

    def test_no_drawdown(self, tmp_path):
        from fullstackautoquant.trading.risk.manager import compute_drawdowns

        data = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "equity": [100_000, 101_000, 102_000],
            }
        )
        data.to_csv(tmp_path / "nav_history.csv", index=False)
        day_dd, rolling_dd = compute_drawdowns(str(tmp_path))
        assert day_dd == 0.0
        # rolling_dd should also be 0 since monotone up
        assert rolling_dd == 0.0

    def test_single_row(self, tmp_path):
        from fullstackautoquant.trading.risk.manager import compute_drawdowns

        data = pd.DataFrame({"date": ["2024-01-01"], "equity": [100_000]})
        data.to_csv(tmp_path / "nav_history.csv", index=False)
        day_dd, rolling_dd = compute_drawdowns(str(tmp_path))
        assert day_dd == 0.0


# ═══════════════════════════════ strategy helpers ════════════════════


class TestStrategyHelpers:
    def test_waterfill_basic(self):
        from fullstackautoquant.trading.strategy import waterfill_with_cap

        weights = [1.0, 1.0, 1.0, 1.0]
        result = waterfill_with_cap(weights, max_w=0.5)
        assert len(result) == 4
        assert all(w <= 0.5 + 1e-9 for w in result)
        assert sum(result) <= 1.0 + 1e-9

    def test_waterfill_empty(self):
        from fullstackautoquant.trading.strategy import waterfill_with_cap

        result = waterfill_with_cap([], max_w=0.5)
        assert result == []

    def test_waterfill_single(self):
        from fullstackautoquant.trading.strategy import waterfill_with_cap

        result = waterfill_with_cap([1.0], max_w=0.5)
        assert len(result) == 1
        assert result[0] <= 0.5 + 1e-9

    def test_waterfill_low_cap(self):
        from fullstackautoquant.trading.strategy import waterfill_with_cap

        weights = [1.0] * 10
        result = waterfill_with_cap(weights, max_w=0.08)
        assert all(w <= 0.08 + 1e-9 for w in result)
        # sum might be < 1 if all capped
        assert sum(result) <= 1.0 + 1e-9

    def test_compute_weight_candidates_equal(self):
        from fullstackautoquant.trading.strategy import compute_weight_candidates

        signals = [
            {"symbol": "A", "score": 0.9, "confidence": 0.95, "rank": 1},
            {"symbol": "B", "score": 0.8, "confidence": 0.90, "rank": 2},
        ]
        cfg = {
            "portfolio": {"max_weight": 0.5},
            "weights": {"mode": "equal", "confidence_tilt": False},
        }
        result = compute_weight_candidates(signals, cfg)
        assert len(result) == 2
        # equal mode returns raw equal weights (not normalized)
        assert all(w > 0 for w in result)

    def test_compute_weight_candidates_ranked(self):
        from fullstackautoquant.trading.strategy import compute_weight_candidates

        signals = [
            {"symbol": "A", "score": 0.9, "confidence": 0.95, "rank": 1},
            {"symbol": "B", "score": 0.8, "confidence": 0.90, "rank": 2},
        ]
        cfg = {
            "portfolio": {"max_weight": 0.5},
            "weights": {
                "mode": "ranked",
                "confidence_tilt": False,
                "rank_metric": "rank",
                "rank_exponent": 1.0,
            },
        }
        result = compute_weight_candidates(signals, cfg)
        assert len(result) == 2
