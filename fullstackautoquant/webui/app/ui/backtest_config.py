"""Helpers for loading and resolving trading strategy configuration."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import yaml


class ConfigLoadError(RuntimeError):
    """Raised when strategy configuration cannot be loaded."""


def resolve_trading_config_path(app_config: Mapping[str, object]) -> Path:
    """Locate the trading config file using WebUI settings fallbacks."""

    raw_value: str | None = None
    paths_cfg = app_config.get("paths") if isinstance(app_config, Mapping) else None
    if isinstance(paths_cfg, Mapping):
        raw_candidate = paths_cfg.get("trading_config")
        if isinstance(raw_candidate, str) and raw_candidate.strip():
            raw_value = raw_candidate

    if raw_value:
        candidate = Path(raw_value).expanduser()
        if candidate.is_absolute():
            return candidate
    else:
        candidate = Path("../trading/config.auto.local.yaml")

    search_roots = [
        Path(__file__).resolve().parents[1],
        Path(__file__).resolve().parents[2],
        Path(__file__).resolve().parents[3],
        Path(__file__).resolve().parents[4],
    ]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (Path(__file__).resolve().parents[2] / candidate).resolve()


def load_strategy_config(path: Path) -> Dict[str, object]:
    """Load the trading strategy YAML config."""

    if not path.exists():
        raise ConfigLoadError(f"Strategy config file does not exist: {path}")
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem errors
        raise ConfigLoadError(f"Reading strategy configfailed: {exc}") from exc
    data = yaml.safe_load(payload) or {}
    if not isinstance(data, MutableMapping):
        raise ConfigLoadError("Invalid strategy config format, must be a dict.")
    return copy.deepcopy(dict(data))


def infer_backtest_logs_root(
    strategy_paths: Mapping[str, object] | None,
    config_path: Path,
    default_base: Path,
) -> Path:
    """Resolve backtest log root based on strategy config."""

    logs_raw = None
    if isinstance(strategy_paths, Mapping):
        raw = strategy_paths.get("logs_dir")
        if isinstance(raw, str) and raw.strip():
            logs_raw = raw

    if logs_raw:
        base_candidate = Path(logs_raw).expanduser()
        if not base_candidate.is_absolute():
            base = (config_path.parent / base_candidate).resolve()
        else:
            base = base_candidate.resolve()
    else:
        base = default_base.resolve()
    return (base / "backtest").resolve()


__all__ = [
    "ConfigLoadError",
    "infer_backtest_logs_root",
    "load_strategy_config",
    "resolve_trading_config_path",
]
