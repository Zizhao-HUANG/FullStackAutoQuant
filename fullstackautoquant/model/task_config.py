from __future__ import annotations

import sys
import warnings
from collections.abc import Iterable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    from qlib.utils import init_instance_by_config
except ModuleNotFoundError:
    _QLIB_FALLBACK = Path(__file__).resolve().parents[1] / "qlib_official"
    if _QLIB_FALLBACK.exists() and str(_QLIB_FALLBACK) not in sys.path:
        sys.path.insert(0, str(_QLIB_FALLBACK))
    from qlib.utils import init_instance_by_config

from fullstackautoquant.model.io.data_loader import load_combined_factors

TaskCfg = dict[str, Any]

__all__ = [
    "TaskConfigError",
    "load_task_config",
    "build_handler_from_task",
    "get_dataset_step_len",
    "get_dataset_segments",
    "get_training_time_range",
    "TASK_DEFAULT_PATH",
]


class TaskConfigError(RuntimeError):
    """Raised when the rendered task configuration is missing or malformed."""


_REPO_ROOT = Path(__file__).resolve().parents[2]  # fullstackautoquant project root
TASK_DEFAULT_PATH = _REPO_ROOT / "configs" / "task_rendered.yaml"


def _normalize_date(value: str | Any) -> str:
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def load_task_config(path: Path) -> TaskCfg:
    if not path.exists():
        raise TaskConfigError(f"Cannot find task_rendered.yaml: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:  # pragma: no cover - defensive
        raise TaskConfigError(f"Failed to read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TaskConfigError("task_rendered.yaml content should be a dictionary")
    return data


def _locate_handler_cfg(task_cfg: TaskCfg) -> dict[str, Any]:
    dataset_cfg = task_cfg.get("dataset")
    if not isinstance(dataset_cfg, dict):
        raise TaskConfigError("task_rendered.yaml is missing dataset config")
    dataset_kwargs = dataset_cfg.get("kwargs")
    if not isinstance(dataset_kwargs, dict):
        raise TaskConfigError("task_rendered.yaml dataset.kwargs is missing")
    handler_cfg = dataset_kwargs.get("handler")
    if not isinstance(handler_cfg, dict):
        raise TaskConfigError("task_rendered.yaml is missing handler config")
    return handler_cfg


def _resolve_instrument_universe(spec: str | Sequence[str] | None) -> set[str] | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        try:
            from qlib.data import D  # lazy import; requires qlib.init beforehand
        except Exception as exc:  # pragma: no cover - defensive
            raise TaskConfigError(f"Cannot resolve instruments={spec}: {exc}") from exc
        inst_conf = D.instruments(market=spec)
        inst_list = D.list_instruments(inst_conf, as_list=True)
    elif isinstance(spec, (list, tuple, set)):
        inst_list = [str(i) for i in spec]
    else:
        return None
    inst_set = {inst for inst in inst_list if inst}
    return inst_set or None


def _filter_factor_df_by_instruments(df, instrument_spec, strict: bool = False) -> Any:
    instrument_set = _resolve_instrument_universe(instrument_spec)
    if not instrument_set:
        return df
    idx = getattr(df, "index", None)
    if not isinstance(idx, pd.MultiIndex) or "instrument" not in idx.names:
        raise TaskConfigError(
            "combined_factors_df.parquet is missing ('datetime','instrument') MultiIndex"
        )
    inst_level = idx.get_level_values("instrument")
    actual_set = set(inst_level.unique())
    extra = actual_set - instrument_set
    if extra:
        sample = sorted(extra)[:5]
        message = (
            f"combined_factors_df.parquet contains {len(extra)} instruments not in {instrument_spec}, e.g. {sample}."
            " These will be automatically dropped; limit the universe when generating combined_factors_df.parquet for best performance."
        )
        if strict:
            raise TaskConfigError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    mask = inst_level.isin(instrument_set)
    filtered = df[mask]
    if filtered.empty:
        raise TaskConfigError(
            f"combined_factors_df.parquet has no available data in {instrument_spec}. Check the factor synthesis pipeline."
        )
    return filtered


def _inject_combined_factors(
    dl_sequence: Iterable[dict[str, Any]],
    combined_factors_path: Path,
) -> None:
    cf_df = load_combined_factors(combined_factors_path)
    # Infer the target instrument universe (assuming consistent instruments across the handler)
    for loader_cfg in dl_sequence:
        if not isinstance(loader_cfg, dict):
            continue
        if loader_cfg.get("class") in {"qlib.contrib.data.loader.Alpha158DL", "Alpha158DL"}:
            loader_kwargs = loader_cfg.get("kwargs", {})
            loader_kwargs.get("instruments")
    # If spec was not captured above, leave it to handler_kwargs for further filtering
    for loader_cfg in dl_sequence:
        if not isinstance(loader_cfg, dict):
            continue
        loader_class = loader_cfg.get("class", "")
        loader_kwargs = loader_cfg.setdefault("kwargs", {})
        # StaticDataLoader is responsible for loading the concatenated custom factors
        if loader_class in {"qlib.data.dataset.loader.StaticDataLoader", "StaticDataLoader"}:
            loader_kwargs["config"] = cf_df


def build_handler_from_task(
    task_cfg: TaskCfg,
    combined_factors_path: Path,
    start_time: str | None = None,
    end_time: str | None = None,
) -> Any:
    handler_cfg = deepcopy(_locate_handler_cfg(task_cfg))
    handler_kwargs = handler_cfg.setdefault("kwargs", {})
    data_loader_cfg = handler_kwargs.get("data_loader", {})
    dl_kwargs = data_loader_cfg.get("kwargs", {})
    dataloader_l = dl_kwargs.get("dataloader_l", [])
    if isinstance(dataloader_l, list):
        _inject_combined_factors(dataloader_l, combined_factors_path)

    instrument_spec = handler_kwargs.get("instruments")
    if instrument_spec:
        # If a separate StaticDataLoader config exists, filter again at handler level (in case loader config lacks instruments)
        for loader_cfg in dataloader_l:
            if isinstance(loader_cfg, dict) and loader_cfg.get("class") in {
                "qlib.data.dataset.loader.StaticDataLoader",
                "StaticDataLoader",
            }:
                config = loader_cfg.get("kwargs", {}).get("config")
                if config is not None:
                    loader_cfg["kwargs"]["config"] = _filter_factor_df_by_instruments(
                        config, instrument_spec
                    )

    if start_time is not None:
        handler_kwargs["start_time"] = start_time
    if end_time is not None:
        handler_kwargs["end_time"] = end_time

    return init_instance_by_config(handler_cfg)


def get_dataset_step_len(task_cfg: TaskCfg) -> int | None:
    dataset_cfg = task_cfg.get("dataset", {})
    dataset_kwargs = dataset_cfg.get("kwargs", {})
    step_len = dataset_kwargs.get("step_len")
    if step_len is not None:
        try:
            return int(step_len)
        except (TypeError, ValueError):
            raise TaskConfigError(f"step_len cannot be converted to integer: {step_len}")
    return None


def get_dataset_segments(task_cfg: TaskCfg) -> dict[str, Any]:
    dataset_cfg = task_cfg.get("dataset", {})
    dataset_kwargs = dataset_cfg.get("kwargs", {})
    segments = dataset_kwargs.get("segments")
    if not isinstance(segments, dict):
        raise TaskConfigError("task_rendered.yaml is missing dataset.kwargs.segments")
    return segments


def get_training_time_range(task_cfg: TaskCfg) -> dict[str, str]:
    handler_cfg = _locate_handler_cfg(task_cfg)
    handler_kwargs = handler_cfg.get("kwargs", {})
    start = handler_kwargs.get("start_time")
    end = handler_kwargs.get("end_time")
    if start is None or end is None:
        raise TaskConfigError("task_rendered.yaml handler.kwargs is missing start_time/end_time")
    return {"start_time": _normalize_date(start), "end_time": _normalize_date(end)}
