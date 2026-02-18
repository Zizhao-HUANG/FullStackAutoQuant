"""Utility helpers shared by backtest UI components."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path


def format_percent(value: object, placeholder: str | None = None) -> str:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value:.2%}"
    return placeholder or "--"


def format_ratio(value: object, placeholder: str | None = None) -> str:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value:.2f}"
    return placeholder or "--"


def default_daily_pv(strategy_cfg: Mapping[str, object], repo_root: Path) -> Path:
    paths_cfg = strategy_cfg.get("paths")
    if isinstance(paths_cfg, Mapping):
        raw = paths_cfg.get("daily_pv")
        if raw:
            try:
                return Path(str(raw)).expanduser().resolve()
            except (TypeError, ValueError):
                pass
    return (repo_root / "ModelInferenceBundle" / "daily_pv.h5").resolve()


def default_logs_base(strategy_cfg: Mapping[str, object], config_path: Path) -> Path:
    paths_cfg = strategy_cfg.get("paths")
    if isinstance(paths_cfg, Mapping):
        raw = paths_cfg.get("logs_dir")
        if isinstance(raw, str) and raw.strip():
            candidate = Path(raw).expanduser()
            if candidate.is_absolute():
                return candidate.resolve()
            return (config_path.parent / candidate).resolve()
    return (Path(__file__).resolve().parents[3] / "logs").resolve()


__all__ = [
    "default_daily_pv",
    "default_logs_base",
    "format_percent",
    "format_ratio",
]
