#!/usr/bin/env python3

"""
Production inference script that strictly replicates the training data pipeline.

Pipeline overview:
- DataHandlerLP:
    * NestedDataLoader(Alpha158DL: 20 features + label)
    * StaticDataLoader(combined_factors_df.parquet: 2 custom features)
    * infer_processors: RobustZScoreNorm(fit=[2005-01-04, 2021-12-31], clip_outlier=True) + Fillna
- TSDatasetH(step_len=72, segments={'test':[T,T]}) auto lookback 72 steps -> (N, 72, 22)
- GeneralPTNN model deserialized from params.pkl (Qlib object), fallback to torch.load(state_dict)
- Output: per-stock ranked scores CSV for day T across CSI300 universe

Usage:
  python -m fullstackautoquant.model.inference \\
    --date auto \\
    --combined_factors ./data/combined_factors_df.parquet \\
    --params ./weights/params.pkl \\
    --out ./output/ranked_scores.csv
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from typing import Any


def _ensure_conda_env() -> None:
    target_env = "rdagent4qlib"
    if os.environ.get("CONDA_DEFAULT_ENV") == target_env:
        return
    conda_exe = shutil.which("conda")
    if not conda_exe:
        return
    # Only attempt switch if target env actually exists
    import subprocess

    result = subprocess.run(
        [conda_exe, "env", "list"], capture_output=True, text=True
    )
    if target_env not in result.stdout:
        return
    script = str(Path(__file__).resolve())
    cmd = [conda_exe, "run", "-n", target_env, "python", script, *sys.argv[1:]]
    os.execv(conda_exe, cmd)


_ensure_conda_env()


import jsonschema
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Allow running without installing the package.
PACKAGE_ROOT = Path(__file__).resolve().parent
PARENT_ROOT = PACKAGE_ROOT.parent
for candidate in (str(PARENT_ROOT), str(PACKAGE_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from fullstackautoquant.model.archive import LocalArchive
from fullstackautoquant.model.scoring import compute_confidence, rank_signals
from fullstackautoquant.model.task_config import (
    TASK_DEFAULT_PATH,
    TaskConfigError,
    build_handler_from_task,
    get_dataset_step_len,
    get_training_time_range,
    load_task_config,
)

CONFIG_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "schema" / "model_inference.schema.json"
)


@dataclass(frozen=True)
class ModelPaths:
    combined_factors: Path
    params: Path
    output_csv: Path
    archive_dir: Path | None


@dataclass(frozen=True)
class CalendarRange:
    start_date: str
    end_date: str
    skip_non_trading: bool = True


@dataclass(frozen=True)
class ModelSettings:
    weights: Path
    device: str = "cpu"
    dropout_samples: int = 16


@dataclass(frozen=True)
class OutputSettings:
    latest_csv: Path
    archive_dir: Path
    copy_latest_to_archive: bool
    archive_strategy: str


@dataclass(frozen=True)
class InferenceConfig:
    calendar: CalendarRange
    data: dict[str, Any]
    model: ModelSettings
    output: OutputSettings
    logging: dict[str, Any]
    task_config: Path


def _ensure_cuda_stub() -> None:
    """Patch torch.load to force map_location='cpu' on CPU-only machines.

    The previous approach (replacing torch.cuda with a SimpleNamespace) causes
    libc++abi SIGABRT crashes in PyTorch >= 2.9 because PyTorch validates CUDA
    at the C++ layer.  Instead, we simply intercept torch.load to inject
    map_location='cpu', which is the only safe way to load CUDA-pickled models
    on CPU-only machines.
    """
    import torch

    if torch.cuda.is_available():
        return

    if getattr(torch, "_cpu_load_patched", False):
        return

    _original_torch_load = torch.load

    def _cpu_torch_load(*args, **kwargs):
        if "map_location" not in kwargs:
            kwargs["map_location"] = "cpu"
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _cpu_torch_load  # type: ignore[assignment]
    torch._cpu_load_patched = True  # type: ignore[attr-defined]


class ConfigError(Exception):
    """Configuration validation failed."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Cannot read config file {path}: {exc}") from exc


def _load_schema() -> dict[str, Any]:
    try:
        return json.loads(CONFIG_SCHEMA_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Missing config schema: {CONFIG_SCHEMA_PATH}") from exc


def _validate_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    schema = _load_schema()
    try:
        jsonschema.validate(instance=raw_cfg, schema=schema)
    except jsonschema.ValidationError as exc:
        raise ConfigError(f"Config does not match schema: {exc.message}") from exc
    return raw_cfg


def _dataclass_from_config(raw_cfg: dict[str, Any]) -> InferenceConfig:
    calendar = raw_cfg["calendar"]
    data = raw_cfg["data"]
    model = raw_cfg["model"]
    output = raw_cfg["output"]
    logging_cfg = raw_cfg.get("logging", {})
    task_cfg_path = raw_cfg.get("task_config", str(TASK_DEFAULT_PATH))
    return InferenceConfig(
        calendar=CalendarRange(
            start_date=calendar["start_date"],
            end_date=calendar["end_date"],
            skip_non_trading=calendar.get("skip_non_trading", True),
        ),
        data=data,
        model=ModelSettings(
            weights=Path(model["weights"]),
            device=model.get("device", "cpu"),
            dropout_samples=model.get("dropout_samples", 16),
        ),
        output=OutputSettings(
            latest_csv=Path(output["latest_csv"]),
            archive_dir=Path(output["archive_dir"]),
            copy_latest_to_archive=output.get("copy_latest_to_archive", True),
            archive_strategy=output.get("archive_strategy", "local"),
        ),
        logging=logging_cfg,
        task_config=Path(task_cfg_path),
    )


def _build_paths(cfg: InferenceConfig, args: argparse.Namespace) -> ModelPaths:
    combined_source: str | None = args.combined_factors or cfg.data.get("combined_factors_out")
    if not combined_source:
        raise ConfigError(
            "Missing combined_factors path. Provide via --combined_factors or config file."
        )
    params_source: str | None = args.params or str(cfg.model.weights)
    if not params_source:
        raise ConfigError("Missing model weights path. Provide via --params or config file.")
    output_target: str | None = args.out or str(cfg.output.latest_csv)
    if not output_target:
        raise ConfigError("Missing output CSV path. Provide via --out or config file.")

    archive_dir = cfg.output.archive_dir if cfg.output.copy_latest_to_archive else None
    return ModelPaths(
        combined_factors=Path(combined_source),
        params=Path(params_source),
        output_csv=Path(output_target),
        archive_dir=archive_dir,
    )


def _ensure_paths(paths: ModelPaths) -> None:
    if not paths.combined_factors.exists():
        raise FileNotFoundError(f"Cannot find factor file: {paths.combined_factors}")
    if not paths.params.exists():
        raise FileNotFoundError(f"Cannot find model weights: {paths.params}")


def _auto_last_trading_day() -> str:
    import qlib  # noqa: F401
    from qlib.data import D

    cal = D.calendar(start_time="2005-01-04", end_time=None, future=True)
    return str(pd.Timestamp(cal[-1]).date())


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="auto", help="Target trading day YYYY-MM-DD | auto (default)")
    ap.add_argument("--combined_factors", help="Absolute path to combined_factors_df.parquet")
    ap.add_argument("--params", help="Absolute path to trained params.pkl")
    ap.add_argument("--out", help="Output CSV path")
    ap.add_argument("--config", help="Inference config JSON file")
    ap.add_argument("--task-config", help="Training task_rendered.yaml path")
    ap.add_argument("--provider_uri", help="Qlib provider URI")
    ap.add_argument("--region", help="Qlib region")
    ap.add_argument("--start", help="Factor start date, default 2005-01-04")
    ap.add_argument("--instruments", help="Instrument universe, default csi300")
    ap.add_argument("--norm-cache", help="Path to cached normalizer params (weights/norm_params.pkl)")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    base_cfg = InferenceConfig(
        calendar=CalendarRange(start_date="2005-01-04", end_date="2099-12-31"),
        data={
            "qlib_data_dir": os.path.expanduser("~/.qlib/qlib_data/cn_data"),
            "daily_pv": str(PACKAGE_ROOT / "daily_pv.h5"),
            "factors_workspace": [],
            "combined_factors_out": str(PACKAGE_ROOT / "combined_factors_df.parquet"),
        },
        model=ModelSettings(weights=PACKAGE_ROOT / "state_dict_cpu.pt"),
        output=OutputSettings(
            latest_csv=PACKAGE_ROOT / "ranked_scores_AUTO_via_qlib.csv",
            archive_dir=PACKAGE_ROOT / "ranked_scores_archive",
            copy_latest_to_archive=True,
            archive_strategy="local",
        ),
        logging={},
        task_config=TASK_DEFAULT_PATH,
    )

    cfg = base_cfg
    if args.config:
        user_cfg_raw = _load_json(Path(args.config))
        user_cfg_validated = _validate_config(user_cfg_raw)
        cfg = _dataclass_from_config(user_cfg_validated)

    # 0) Environment setup
    ws = str(Path(__file__).resolve().parent)
    if ws not in sys.path:
        sys.path.append(ws)  # Ensure pickle deserialization can resolve model.model_cls

    import qlib

    provider_uri = args.provider_uri or cfg.data.get("qlib_data_dir")
    region = args.region or "cn"
    qlib.init(provider_uri=provider_uri, region=region)
    ideal_workers = 0

    # 1) Determine target trading day
    target_date = (
        _auto_last_trading_day()
        if (args.date is None or str(args.date).lower() == "auto")
        else args.date
    )

    # 2) Build training-equivalent Handler + Dataset (TSDatasetH)
    paths = _build_paths(cfg, args)
    _ensure_paths(paths)
    args.instruments or cfg.data.get("instruments", "csi300")

    task_cfg_path = Path(args.task_config) if args.task_config else cfg.task_config
    try:
        task_cfg = load_task_config(task_cfg_path)
    except TaskConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    training_range = get_training_time_range(task_cfg)
    training_start = training_range["start_time"]
    training_end_ts = pd.Timestamp(training_range["end_time"])
    dataset_step_len = get_dataset_step_len(task_cfg)
    if dataset_step_len is None:
        raise TaskConfigError("task_rendered.yaml is missing step_len")

    start_date = args.start or training_start
    handler_end_ts = pd.Timestamp(target_date)
    if handler_end_ts > training_end_ts:
        print(
            f"[WARN] Requested inference date {handler_end_ts.date()} exceeds training cutoff "
            f"{training_end_ts.date()}; proceeding with extended horizon."
        )
    handler_end = str(handler_end_ts.date())
    if pd.Timestamp(start_date) > handler_end_ts:
        raise TaskConfigError(f"Start date {start_date} is after inference cutoff {handler_end}")

    handler = build_handler_from_task(
        task_cfg=task_cfg,
        combined_factors_path=paths.combined_factors,
        start_time=start_date,
        end_time=handler_end,
        norm_cache_path=Path(args.norm_cache) if args.norm_cache else None,
    )
    # Determine the trading day for inference: prefer the nearest date <= target (no strict N=300 requirement)
    from qlib.data.dataset import DataHandlerLP as DH
    from qlib.data.dataset import TSDatasetH

    try:
        full_feat = handler.fetch(
            selector=slice(None, None), level="datetime", col_set="feature", data_key=DH.DK_I
        )
        if full_feat.empty:
            raise KeyError("Processed features empty; please check data coverage")
        all_dates = full_feat.index.get_level_values("datetime").unique().sort_values()
        req = pd.Timestamp(target_date)
        cand = list(all_dates[all_dates <= req]) if len(all_dates) else []
        # Default to the most recent available date (no N=300 check)
        used_dt = (cand[-1] if len(cand) else all_dates.max()) if len(all_dates) else None
        used_date_str = str(pd.Timestamp(used_dt).date()) if used_dt is not None else target_date
    except Exception:
        # Fallback: use target_date as-is
        used_date_str = target_date

    dataset = TSDatasetH(
        handler=handler,
        segments={"test": [used_date_str, used_date_str]},
        step_len=dataset_step_len,
    )

    # 3) Load model weights (prefer state_dict, fallback to pickle)
    params_path = paths.params
    if not params_path.exists():
        raise FileNotFoundError(f"Model weights not found: {params_path}")

    _ensure_cuda_stub()

    from fullstackautoquant.model.model import load_model

    try:
        model, load_strategy = load_model(params_path, device="cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load trained model: {params_path}") from exc

    # For state_dict loading, wrap in Qlib's GeneralPTNN for predict() compat
    if load_strategy == "state_dict":
        from qlib.contrib.model.pytorch_general_nn import GeneralPTNN

        wrapper = object.__new__(GeneralPTNN)  # skip __init__
        wrapper.dnn_model = model
        wrapper.device = torch.device("cpu")
        wrapper.fitted = True
        wrapper.batch_size = 8096
        wrapper.n_jobs = 0
        wrapper.logger = __import__("logging").getLogger(__name__)
        model = wrapper

    if hasattr(model, "n_jobs"):
        model.n_jobs = ideal_workers
    if not getattr(model, "fitted", True):
        model.fitted = True

    # 4) Predict
    pred_series = model.predict(dataset)  # MultiIndex[(datetime,instrument)] -> score
    pred_day, used_date = rank_signals(pred_series, used_date_str)

    # Sanity check: CSI300 count (warn only, do not abort)
    if len(pred_day) != 300:
        print(f"[WARN] Available CSI300 instruments={len(pred_day)} (expected 300)")

    # 5) Compute confidence (based on MC Dropout prediction std; near 0 without Dropout) and persist
    try:
        from qlib.data.dataset import (
            DataHandlerLP as DH,  # ensure DH available even if previous try failed
        )

        # Data preparation consistent with GeneralPTNN.predict
        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DH.DK_I)
        if hasattr(dl_test, "config"):
            dl_test.config(fillna_type="ffill+bfill")
        index_all = dl_test.get_index() if hasattr(dl_test, "get_index") else dl_test.index

        test_loader = DataLoader(
            dl_test,
            batch_size=getattr(model, "batch_size", 256),
            num_workers=ideal_workers,
        )

        dropout_passes = int(getattr(cfg.model, "dropout_samples", 0) or 0)
        preds_runs: list[np.ndarray] = []

        if dropout_passes > 0:
            dropout_model = copy.deepcopy(model)
            dropout_net = getattr(dropout_model, "dnn_model", None)
            if dropout_net is None:
                raise AttributeError("Model is missing dnn_model attribute")
            dropout_net.train()
            with torch.no_grad():
                for _ in range(dropout_passes):
                    run_preds = []
                    for data in test_loader:
                        feature, _ = dropout_model._get_fl(data)
                        pred = dropout_net(feature.float()).detach().cpu().numpy()
                        run_preds.append(pred)
                    run_concat = np.concatenate(run_preds)
                    if run_concat.ndim != 1:
                        run_concat = run_concat.ravel()
                    preds_runs.append(run_concat)
            dropout_net.eval()

        if not preds_runs:
            preds_runs = [np.zeros(len(index_all))]

        # Compute std -> confidence (lower variance => higher confidence). Using 1/(1+std) to map to (0,1].
        conf_vec = compute_confidence(preds_runs if len(preds_runs) else [np.zeros(len(index_all))])
        conf_series_all = pd.Series(conf_vec, index=index_all)

        # Extract target day only, aligned with the sorted index
        conf_day = conf_series_all.loc[pred_day.index]

        # Assemble DataFrame: keep original score column name (typically '0'), add 'confidence' column
        pred_df = pred_day.to_frame(name=pred_day.name if pred_day.name is not None else 0)
        pred_df["confidence"] = conf_day.values
    except Exception as e:
        # Safety fallback: if confidence computation fails, do not block main output
        print(f"[WARN] Confidence computation failed: {e}")
        pred_df = pred_day.to_frame(name=pred_day.name if pred_day.name is not None else 0)

    out_path = paths.output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path)

    archive_file = None
    if paths.archive_dir and cfg.output.copy_latest_to_archive:
        archive_strategy = LocalArchive(paths.archive_dir)
        archive_file = archive_strategy.persist(out_path, used_date)

    # 6) Summary
    mean_signal = float(np.nanmean(pred_day.values)) if len(pred_day) else np.nan
    direction = "UP" if mean_signal > 0 else ("DOWN" if mean_signal < 0 else "FLAT")
    print("==== Inference Summary (Training-Equivalent Pipeline) ====")
    print(f"Target trading day requested: {target_date}")
    print(f"Target trading day used: {used_date}")
    print(f"#Instruments(CSI300): {len(pred_day)}")
    print(f"Mean signal: {mean_signal:.6f}")
    print(f"CSI300 direction (EW): {direction}")
    print(f"Saved per-stock predictions: {out_path}")
    if archive_file:
        print(f"Archived copy: {archive_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
