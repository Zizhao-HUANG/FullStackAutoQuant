from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from .components.execution import ExecutionEngine
from .components.nav_tracker import NavTracker
from .components.records import BacktestIntermediate, PositionSnapshot
from .components.risk_evaluator import RiskEvaluator
from .components.signal_provider import SignalProvider
from .components.strategy_runner import StrategyRunner
from ..manual_workflow.simulator import ManualWorkflowSimulator


@dataclass
class BacktestContext:
    calendar: List[dt.date]
    signal_provider: SignalProvider
    risk_evaluator: RiskEvaluator
    strategy_runner: StrategyRunner
    execution: ExecutionEngine
    nav_tracker: NavTracker
    logs_dir: Path
    manual_simulator: Optional["ManualWorkflowSimulator"] = None


class BacktestPipeline:
    def __init__(self, ctx: BacktestContext) -> None:
        self._ctx = ctx
        self._risk_records: List[Dict[str, object]] = []
        self._signal_records: List[Dict[str, object]] = []
        self._manual_records: List[Dict[str, object]] = []

    def run(
        self,
        initial_cash: float,
        portfolio_value_getter: Callable[[dt.date, Dict[str, float]], float],
        snapshot_collector: Callable[[dt.date, Dict[str, float]], List[PositionSnapshot]],
    ) -> BacktestIntermediate:
        cash = initial_cash
        positions: Dict[str, float] = {}
        equities: List = []
        trades: List = []
        snapshots: List[PositionSnapshot] = []
        prev_equity = initial_cash
        prev_cash: Optional[float] = None
        prev_market_value: Optional[float] = None
        manual_sim = getattr(self._ctx, "manual_simulator", None)

        for trade_date, signal_date, signals in self._ctx.signal_provider.iterate(self._ctx.calendar):
            active_signals = signals
            manual_log: List[Dict[str, object]] = []
            queued_total = 0
            if manual_sim is not None:
                active_signals, manual_log, queued_total = manual_sim.process(
                    trade_date=trade_date,
                    signal_date=signal_date,
                    raw_signals=signals,
                )
                self._manual_records.extend(manual_log)
            market_value = portfolio_value_getter(trade_date, positions)
            equity_before = cash + market_value
            risk_state = self._ctx.risk_evaluator.evaluate(
                trade_date,
                signal_date,
                active_signals,
                positions,
                manual_log,
            )
            orders, _, _ = self._ctx.strategy_runner.generate_orders(active_signals, risk_state, positions, equity_before)
            cash, positions, day_trades, equity_record, portfolio_value = self._ctx.execution.run(
                trade_date,
                orders,
                positions,
                cash,
                prev_equity,
                portfolio_value_getter,
            )
            trades.extend(day_trades)
            equities.append(equity_record)
            snapshots.extend(snapshot_collector(trade_date, positions))
            self._ctx.nav_tracker.append(trade_date, equity_record.equity, cash, portfolio_value)
            self._ctx.nav_tracker.ensure_previous(trade_date, prev_equity, prev_cash, prev_market_value)
            self._risk_records.append(risk_state)
            self._signal_records.append(
                {
                    "trade_date": trade_date.isoformat(),
                    "signal_date": signal_date.isoformat(),
                    "count": len(active_signals),
                    "raw_count": len(signals),
                    "pending_queue": queued_total,
                }
            )
            prev_cash = cash
            prev_market_value = portfolio_value
            prev_equity = equity_record.equity
        if manual_sim is not None:
            remaining = manual_sim.flush_pending()
            if remaining:
                self._manual_records.extend(remaining)

        return BacktestIntermediate(
            equities,
            trades,
            snapshots,
            self._risk_records,
            self._signal_records,
            self._manual_records,
        )

    def flush(self, run_dir: Path, intermediate: BacktestIntermediate, meta: Dict[str, object]) -> None:
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        self._ctx.nav_tracker.write_csv(logs_dir / "nav_history.csv")
