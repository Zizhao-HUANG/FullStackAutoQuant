from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from ...backtest.config import ManualWorkflowParams


@dataclass(slots=True)
class ManualDecision:
    decision_id: str
    queued_at: dt.date
    signal_date: dt.date
    execute_date: dt.date
    signal: dict[str, Any]
    strategy: str | None = None
    fill_ratio: float = 1.0


class ManualWorkflowSimulator:
    """Simulates WebUI manual confirmation, delay, and filtering logic.

    Design principles:
    - All incoming signals are queued first (based on confirmation delay/filter rules).
    - On each trading day, calling `process` releases executable decisions while queueing new signals.
    - For audit convenience, executed/rejected/expired decisions generate structured records.
    """

    def __init__(self, params: ManualWorkflowParams, calendar: list[dt.date]) -> None:
        self._params = params
        self._calendar = list(calendar)
        self._date_index: dict[dt.date, int] = {d: idx for idx, d in enumerate(self._calendar)}
        self._pending: dict[dt.date, list[ManualDecision]] = {}
        self._counter = 0
        self._record_rejected = bool(params.record_rejected)
        self._strategy_field = params.strategy_field or None
        self._strategy_limits = dict(params.strategy_limits)
        self._fill_ratio = max(0.0, min(1.0, float(params.partial_fill_ratio or 1.0)))
        self._queued_per_date_total: dict[dt.date, int] = {}
        self._queued_per_date_strategy: dict[dt.date, dict[str, int]] = {}

    def process(
        self,
        trade_date: dt.date,
        signal_date: dt.date,
        raw_signals: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        """Process the current trading day:

        Returns:
            - List of confirmed tradable signals
            - Decision log (executed + rejected)
            - Number of decisions still queued
        """

        executed = self._release(trade_date)
        active_signals = [self._decorate_signal(decision, trade_date) for decision in executed]
        manual_log = [self._decision_record(decision, "executed") for decision in executed]

        rejection_log = self._schedule(trade_date, signal_date, raw_signals)
        if rejection_log:
            manual_log.extend(rejection_log)

        if self._params.confirm_delay_days == 0:
            executed_same_day = self._release(trade_date)
            if executed_same_day:
                active_signals.extend(
                    self._decorate_signal(item, trade_date) for item in executed_same_day
                )
                manual_log.extend(
                    self._decision_record(item, "executed") for item in executed_same_day
                )

        queued_total = sum(len(items) for items in self._pending.values())
        return active_signals, manual_log, queued_total

    def flush_pending(self) -> list[dict[str, Any]]:
        """Mark remaining unexecuted decisions as expired."""

        leftovers: list[dict[str, Any]] = []
        for items in list(self._pending.values()):
            for decision in items:
                leftovers.append(self._decision_record(decision, "expired"))
        self._pending.clear()
        self._queued_per_date_total.clear()
        self._queued_per_date_strategy.clear()
        return leftovers

    # --- Internal utilities ---------------------------------------------------------

    def _release(self, trade_date: dt.date) -> list[ManualDecision]:
        decisions = self._pending.pop(trade_date, [])
        self._queued_per_date_total.pop(trade_date, None)
        self._queued_per_date_strategy.pop(trade_date, None)
        return decisions

    def _schedule(
        self,
        trade_date: dt.date,
        signal_date: dt.date,
        raw_signals: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Queue based on signals and parameters, return rejection records."""

        if not raw_signals:
            return []

        accepted, rejected = self._filter_signals(raw_signals)
        execute_date = self._resolve_execute_date(trade_date)
        log: list[dict[str, Any]] = []

        if execute_date is None:
            if self._record_rejected:
                for signal in raw_signals:
                    log.append(self._rejection_record(signal, signal_date, "out_of_calendar"))
            return log

        limited, limit_rejections = self._apply_limits(execute_date, accepted)

        for signal in limited:
            fill_ratio = self._fill_ratio if self._fill_ratio is not None else 1.0
            decision = ManualDecision(
                decision_id=self._next_id(signal),
                queued_at=trade_date,
                signal_date=signal_date,
                execute_date=execute_date,
                signal=dict(signal),
                strategy=self._extract_strategy(signal),
                fill_ratio=fill_ratio,
            )
            self._pending.setdefault(execute_date, []).append(decision)

        if self._record_rejected:
            for signal, reason in rejected + limit_rejections:
                log.append(self._rejection_record(signal, signal_date, reason))
        return log

    def _filter_signals(
        self,
        raw_signals: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
        filtered = sorted(
            raw_signals, key=lambda s: float(s.get("confidence", 0.0) or 0.0), reverse=True
        )
        accepted: list[dict[str, Any]] = []
        rejected: list[tuple[dict[str, Any], str]] = []
        min_conf = self._params.min_confidence
        for signal in filtered:
            confidence = float(signal.get("confidence", 0.0) or 0.0)
            if min_conf is not None and confidence < min_conf:
                rejected.append((dict(signal), "below_confidence"))
                continue
            accepted.append(dict(signal))
        return accepted, rejected

    def _apply_limits(
        self,
        execute_date: dt.date,
        signals: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
        if not signals:
            return [], []
        total_limit = self._params.max_signals_per_day
        total_count = self._queued_per_date_total.get(execute_date, 0)
        strategy_counts = dict(self._queued_per_date_strategy.get(execute_date, {}))
        limited: list[dict[str, Any]] = []
        rejected: list[tuple[dict[str, Any], str]] = []
        for signal in signals:
            strategy = self._extract_strategy(signal)
            strategy_limit = self._strategy_limits.get(strategy)
            if total_limit is not None and total_count >= total_limit:
                rejected.append((signal, "exceeds_daily_cap"))
                continue
            if strategy_limit is not None and strategy_counts.get(strategy, 0) >= strategy_limit:
                rejected.append((signal, "strategy_limit"))
                continue
            limited.append(signal)
            total_count += 1
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        self._queued_per_date_total[execute_date] = total_count
        self._queued_per_date_strategy[execute_date] = strategy_counts
        return limited, rejected

    def _resolve_execute_date(self, trade_date: dt.date) -> dt.date | None:
        delay = max(0, int(self._params.confirm_delay_days))
        idx = self._date_index.get(trade_date)
        if idx is None:
            return None
        target_idx = idx + delay
        if target_idx >= len(self._calendar):
            return None
        return self._calendar[target_idx]

    def _decorate_signal(self, decision: ManualDecision, approved_date: dt.date) -> dict[str, Any]:
        signal = dict(decision.signal)
        signal.setdefault("approved_at", approved_date.isoformat())
        signal.setdefault("signal_date", decision.signal_date.isoformat())
        signal.setdefault("decision_id", decision.decision_id)
        strategy = decision.strategy
        if strategy and strategy != "__default__":
            signal.setdefault("strategy", strategy)
        ratio = decision.fill_ratio
        if ratio is not None and ratio < 1.0:
            signal.setdefault("fill_ratio", ratio)
        return signal

    def _decision_record(self, decision: ManualDecision, status: str) -> dict[str, Any]:
        ratio = decision.fill_ratio
        exec_status = status
        if status == "executed" and ratio is not None and ratio < 1.0:
            exec_status = "partial"
        payload = {
            "decision_id": decision.decision_id,
            "signal_date": decision.signal_date.isoformat(),
            "queued_at": decision.queued_at.isoformat(),
            "execute_date": decision.execute_date.isoformat(),
            "symbol": decision.signal.get("instrument") or decision.signal.get("symbol"),
            "action": decision.signal.get("action", "BUY"),
            "confidence": float(decision.signal.get("confidence", 0.0) or 0.0),
            "status": exec_status,
        }
        strategy = decision.strategy
        if strategy and strategy != "__default__":
            payload["strategy"] = strategy
        if ratio is not None and ratio < 1.0:
            payload["fill_ratio"] = ratio
        return payload

    def _rejection_record(
        self, signal: dict[str, Any], signal_date: dt.date, reason: str
    ) -> dict[str, Any]:
        payload = {
            "decision_id": None,
            "signal_date": signal_date.isoformat(),
            "queued_at": signal_date.isoformat(),
            "execute_date": None,
            "symbol": signal.get("instrument") or signal.get("symbol"),
            "action": signal.get("action", "BUY"),
            "confidence": float(signal.get("confidence", 0.0) or 0.0),
            "status": "rejected",
            "reason": reason,
        }
        strategy = self._extract_strategy(signal)
        if strategy and strategy != "__default__":
            payload["strategy"] = strategy
        return payload

    def _next_id(self, signal: dict[str, Any]) -> str:
        self._counter += 1
        symbol = signal.get("instrument") or signal.get("symbol") or "UNKNOWN"
        return f"MW-{self._counter:06d}-{symbol}"

    def _extract_strategy(self, signal: dict[str, Any]) -> str:
        key = self._strategy_field
        if key:
            value = signal.get(key)
            if value:
                return str(value)
        fallback = signal.get("strategy") or signal.get("strategy_id")
        if fallback:
            return str(fallback)
        return "__default__"
