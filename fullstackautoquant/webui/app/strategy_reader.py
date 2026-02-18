"""Strategy output reader and parser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class StrategyReaderError(Exception):
    pass


def load_targets_from_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise StrategyReaderError(f"Target positions file does not exist: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "targets" in data:
        return data["targets"]
    if isinstance(data, list):
        return data
    raise StrategyReaderError("Unsupported target positions format")


def load_targets(path: Path) -> pd.DataFrame:
    data = load_targets_from_json(path)
    df = pd.DataFrame(data)
    expected_columns = {"symbol", "target_qty", "weight"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise StrategyReaderError(f"Target positions missing columns: {missing}")
    return df


def find_latest_targets(targets_dir: Path) -> Path:
    files = sorted(targets_dir.glob("targets_*.json"), reverse=True)
    if not files:
        raise StrategyReaderError(f"directory {targets_dir} contains no targets_* files")
    return files[0]

