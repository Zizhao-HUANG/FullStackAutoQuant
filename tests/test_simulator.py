from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fullstackautoquant.backtest.config import ManualWorkflowParams
from fullstackautoquant.trading.manual_workflow.simulator import ManualWorkflowSimulator


def test_manual_workflow_delays_and_limits() -> None:
    calendar = [
        dt.date(2025, 9, 1),
        dt.date(2025, 9, 2),
        dt.date(2025, 9, 3),
    ]
    params = ManualWorkflowParams(
        enabled=True,
        confirm_delay_days=1,
        max_signals_per_day=1,
        min_confidence=0.5,
        record_rejected=True,
    )
    sim = ManualWorkflowSimulator(params, calendar)
    day1_signals = [
        {"instrument": "AAA", "confidence": 0.9},
        {"instrument": "BBB", "confidence": 0.8},
        {"instrument": "CCC", "confidence": 0.4},  # below min confidence
    ]

    active_day1, log_day1, queued_day1 = sim.process(calendar[0], calendar[0], day1_signals)
    assert active_day1 == []
    assert queued_day1 == 1  # one signal queued for next day
    statuses = {entry["status"] for entry in log_day1}
    assert "rejected" in statuses  # BBB or CCC rejected

    active_day2, log_day2, queued_day2 = sim.process(calendar[1], calendar[1], [])
    assert len(active_day2) == 1
    assert all(item["status"] == "executed" for item in log_day2)
    assert queued_day2 == 0

    # Flush to ensure no leftovers
    remaining = sim.flush_pending()
    assert remaining == []


def test_manual_workflow_strategy_limits_and_partial_fill() -> None:
    calendar = [
        dt.date(2025, 9, 1),
        dt.date(2025, 9, 2),
    ]
    params = ManualWorkflowParams(
        enabled=True,
        confirm_delay_days=0,
        max_signals_per_day=3,
        strategy_field="strategy",
        strategy_limits={"alpha": 1, "beta": 2},
        partial_fill_ratio=0.5,
        record_rejected=True,
    )
    sim = ManualWorkflowSimulator(params, calendar)
    signals = [
        {"instrument": "SZ000001", "symbol": "SZSE.000001", "confidence": 0.95, "strategy": "alpha"},
        {"instrument": "SZ000002", "symbol": "SZSE.000002", "confidence": 0.90, "strategy": "alpha"},
        {"instrument": "SZ000003", "symbol": "SZSE.000003", "confidence": 0.85, "strategy": "beta"},
        {"instrument": "SZ000004", "symbol": "SZSE.000004", "confidence": 0.80, "strategy": "beta"},
        {"instrument": "SZ000005", "symbol": "SZSE.000005", "confidence": 0.70, "strategy": "beta"},
    ]

    active, manual_log, queued = sim.process(calendar[0], calendar[0], signals)

    # Confirm same-day execution with partial fill ratio applied
    executed = [entry for entry in manual_log if entry["status"] == "partial"]
    rejected = [entry for entry in manual_log if entry["status"] == "rejected"]
    assert len(active) == 3
    assert len(executed) == 3
    assert all(abs(entry.get("fill_ratio", 0.0) - 0.5) < 1e-9 for entry in executed)
    assert any(entry.get("reason") == "strategy_limit" for entry in rejected)
    assert any(entry.get("reason") == "exceeds_daily_cap" for entry in rejected)
    assert queued == 0
