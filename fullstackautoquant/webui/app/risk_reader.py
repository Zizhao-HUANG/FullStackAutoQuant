"""Risk state reader module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class RiskReaderError(Exception):
    pass


def load_risk_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RiskReaderError(f"Not foundRisk state file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise RiskReaderError("Invalid risk state file format")
    return data


def find_risk_state_file(targets_dir: Path) -> Path:
    candidates = sorted(targets_dir.glob("risk_state_*.json"), reverse=True)
    if candidates:
        return candidates[0]
    default = targets_dir / "risk_state_AUTO.json"
    if default.exists():
        return default
    raise RiskReaderError("No risk_state file found in directory")

