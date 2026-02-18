"""Centralized management of WebUI session state and data cache."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowState:
    result: dict[str, Any] | None = None
    last_run_at: dt.datetime | None = None
    debug_steps: list[dict[str, Any]] = field(default_factory=list)
    manual_log_path: str | None = None
    manual_log_count: int = 0


class DataAccess:
    """Maintains workflow results, debug output, and cash state within a session."""

    def __init__(self) -> None:
        self._workflow = WorkflowState()
        self._available_cash: float | None = None

    # ---- Workflow state -------------------------------------------------
    def get_workflow_result(self) -> dict[str, Any] | None:
        return self._workflow.result

    def set_workflow_result(
        self, result: dict[str, Any], *, run_at: dt.datetime | None = None
    ) -> None:
        self._workflow.result = result
        self._workflow.last_run_at = run_at or dt.datetime.utcnow()
        self.clear_debug_steps()

    def clear_workflow_result(self) -> None:
        self._workflow.result = None
        self._workflow.last_run_at = None
        self.clear_debug_steps()

    def get_workflow_timestamp(self) -> dt.datetime | None:
        return self._workflow.last_run_at

    def append_debug_step(self, payload: dict[str, Any]) -> None:
        self._workflow.debug_steps.append(payload)

    def get_debug_steps(self) -> list[dict[str, Any]]:
        return list(self._workflow.debug_steps)

    def clear_debug_steps(self) -> None:
        self._workflow.debug_steps.clear()

    # ---- Manual decision logs ------------------------------------------
    def record_manual_log(self, path: str | None, count: int) -> None:
        self._workflow.manual_log_path = path
        self._workflow.manual_log_count = count

    def manual_log_summary(self) -> dict[str, Any]:
        return {
            "path": self._workflow.manual_log_path,
            "count": self._workflow.manual_log_count,
        }

    # ---- Available cash helpers ----------------------------------------
    def ensure_available_cash(self, loader: Callable[[], float]) -> float:
        if self._available_cash is None:
            self._available_cash = float(loader())
        return self._available_cash

    def get_available_cash(self) -> float:
        return float(self._available_cash or 0.0)

    def set_available_cash(self, value: float) -> None:
        self._available_cash = float(value)

    def reset_available_cash(self) -> None:
        self._available_cash = None


__all__ = ["DataAccess", "WorkflowState"]
