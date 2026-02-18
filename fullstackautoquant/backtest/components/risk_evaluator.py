"""Risk evaluation module."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from fullstackautoquant.trading.risk.manager import compute_drawdowns, detect_limit_states


class RiskEvaluator:
    """Replicate live trading risk control logic."""

    def __init__(self, config: dict, logs_dir: Path, allow_override: bool) -> None:
        self._config = config
        self._logs_dir = logs_dir
        self._allow_override = allow_override

    def evaluate(
        self,
        trade_date: dt.date,
        signal_date: dt.date,
        signals: list[dict],
        positions,
        manual_feedback: list[dict] | None = None,
    ) -> dict[str, Any]:
        logs_dir = self._ensure_logs_dir()
        day_dd, rolling_dd = compute_drawdowns(str(logs_dir))
        allow_buy = self._decide_buy(day_dd, rolling_dd)
        limit_up, limit_down = self._detect_limits(signals)
        manual_feedback = manual_feedback or []
        rejected = [item for item in manual_feedback if item.get("status") == "rejected"]
        partial = [item for item in manual_feedback if item.get("status") == "partial"]
        executed = [
            item for item in manual_feedback if item.get("status") in {"executed", "partial"}
        ]
        record = {
            "trade_date": trade_date.isoformat(),
            "signal_date": signal_date.isoformat(),
            "allow_buy": allow_buy,
            "day_drawdown": day_dd,
            "rolling5d_drawdown": rolling_dd,
            "limit_up_symbols": limit_up,
            "limit_down_symbols": limit_down,
            "manual_executed": len(executed),
            "manual_rejected": len(rejected),
            "manual_partial": len(partial),
            "manual_rejection_reasons": sorted(
                {item.get("reason", "unknown") for item in rejected if item.get("reason")}
            ),
        }
        self._write_limit_log(record)
        return record

    def _ensure_logs_dir(self) -> Path:
        logs_dir = self._config.get("paths", {}).get("logs_dir")
        if logs_dir:
            path = Path(logs_dir)
            path.mkdir(parents=True, exist_ok=True)
            return path
        default = self._logs_dir
        default.mkdir(parents=True, exist_ok=True)
        return default

    def _decide_buy(self, day_dd: float, rolling_dd: float) -> bool:
        risk_cfg = self._config.get("risk", {})
        allow_buy = True
        if day_dd >= float(risk_cfg.get("day_drawdown_limit", 0.03)):
            allow_buy = False
        if rolling_dd >= float(risk_cfg.get("rolling5d_drawdown_limit", 0.08)):
            allow_buy = False
        if self._allow_override:
            allow_buy = True
        return allow_buy

    def _detect_limits(self, signals: list[dict]) -> tuple[list[str], list[str]]:
        if not signals:
            return [], []
        paths = self._config.get("paths", {})
        h5_path = paths.get("daily_pv")
        if not h5_path:
            return [], []
        limit_threshold = float(self._config.get("order", {}).get("limit_threshold", 0.095))
        instruments = [str(sig["instrument"]) for sig in signals if sig.get("instrument") is not None]
        if not instruments:
            return [], []
        limit_up, limit_down = detect_limit_states(h5_path, instruments, limit_threshold)
        return limit_up, limit_down

    def _write_limit_log(self, record: dict[str, Any]) -> None:
        limit_path = self._ensure_logs_dir() / "limit_states.json"
        try:
            payload = {}
            if limit_path.exists():
                payload = json.loads(limit_path.read_text(encoding="utf-8"))
            payload[record["trade_date"]] = record
            limit_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
