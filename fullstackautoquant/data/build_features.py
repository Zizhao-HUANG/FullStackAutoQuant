#!/usr/bin/env python3

"""
Replicate feature construction pipeline from training snapshot task_rendered.yaml:
1) Read DataHandler config directly from training task(NestedDataLoader + processors).
2) Auto-inject combined_factors_df.parquet, maintaining same join strategy and label definitions as training.
3) Process target trading day (fallback to nearest available if missing), output 22 features with no missing values and consistent ordering.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fullstackautoquant.model.io.data_loader import load_combined_factors
from fullstackautoquant.model.task_config import (
    TASK_DEFAULT_PATH,
    TaskConfigError,
    build_handler_from_task,
    get_training_time_range,
    load_task_config,
)


def _auto_last_trading_day() -> str:
    import qlib  # noqa: F401
    from qlib.data import D

    cal = D.calendar(start_time="2005-01-04", end_time=None, future=True)
    return str(pd.Timestamp(cal[-1]).date())


def _extract_alpha_feature_names(task_cfg: dict) -> list[str]:
    dataset_cfg = task_cfg.get("dataset", {})
    dataset_kwargs = dataset_cfg.get("kwargs", {})
    handler_cfg = dataset_kwargs.get("handler", {})
    handler_kwargs = handler_cfg.get("kwargs", {})
    data_loader = handler_kwargs.get("data_loader", {})
    dl_kwargs = data_loader.get("kwargs", {})
    dl_list = dl_kwargs.get("dataloader_l", [])

    for loader in dl_list:
        loader_class = loader.get("class", "")
        if "Alpha158" not in loader_class:
            continue
        config = loader.get("kwargs", {}).get("config", {})
        feature_cfg = config.get("feature")
        if isinstance(feature_cfg, list) and len(feature_cfg) >= 2:
            return list(feature_cfg[1])
    raise TaskConfigError("task_rendered.yaml is missing Alpha158 feature name definitions")


def _prepare_feature_df_for_day(handler: Any, target_date: str) -> pd.DataFrame:
    from qlib.data.dataset import DataHandlerLP as DH

    t = pd.Timestamp(target_date)
    try:
        return handler.fetch(selector=t, level="datetime", col_set="feature", data_key=DH.DK_I)
    except Exception:
        full = handler.fetch(
            selector=slice(None, None), level="datetime", col_set="feature", data_key=DH.DK_I
        )
        if full.empty:
            raise KeyError("Processed features empty")
        all_dates = full.index.get_level_values("datetime").unique().sort_values()
        cand = all_dates[all_dates <= t]
        if len(cand) == 0:
            cand = all_dates
        last_dt = cand.max()
        return full.loc[(last_dt, slice(None)), :]


def _assert_value_and_missing(df_feat: pd.DataFrame) -> None:
    if df_feat.isnull().sum().sum() != 0:
        raise AssertionError("Features have missing values, fill before inference.")
    vmin, vmax = float(np.nanmin(df_feat.values)), float(np.nanmax(df_feat.values))
    if not (vmin >= -3 - 1e-8 and vmax <= 3 + 1e-8):
        raise AssertionError(
            f"Feature values exceed normalization range [-3, 3], actual [{vmin:.6f}, {vmax:.6f}]"
        )


def _reorder_columns(df_feat: pd.DataFrame, expected_order: list[str]) -> pd.DataFrame:
    if not isinstance(df_feat.columns, pd.MultiIndex):
        df_feat.columns = pd.MultiIndex.from_tuples([("feature", str(c)) for c in df_feat.columns])

    multi_cols = list(df_feat.columns)
    actual_names = [c[1] for c in multi_cols if c[0] == "feature"]
    if len(actual_names) != len(expected_order):
        raise AssertionError(
            f"Feature count mismatch: expected {len(expected_order)}, actual {len(actual_names)}"
        )

    missing = [name for name in expected_order if name not in actual_names]
    if missing:
        raise AssertionError(f"Missing feature columns:{missing}")

    ordered_cols = pd.MultiIndex.from_product([["feature"], expected_order])
    return df_feat.loc[:, ordered_cols]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        default=None,
        help="Target trading day YYYY-MM-DD;defaults to Qlib Latest trading day",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output parquet path;defaults to features_ready_infer_YYYY-MM-DD.parquet",
    )
    parser.add_argument(
        "--combined_factors",
        default=str(Path(__file__).resolve().parent / "combined_factors_df.parquet"),
        help="Path to training output combined_factors_df.parquet",
    )
    parser.add_argument(
        "--task-config",
        default=str(TASK_DEFAULT_PATH),
        help="Path to training task snapshot task_rendered.yaml",
    )
    parser.add_argument(
        "--provider_uri", default="~/.qlib/qlib_data/cn_data", help="Qlib provider_uri"
    )
    parser.add_argument("--region", default="cn", help="Qlib region")
    parser.add_argument(
        "--start", default=None, help="Data start date(defaults to training config)"
    )
    parser.add_argument(
        "--instruments", default="csi300", help="Instrument universe (default csi300)"
    )
    args = parser.parse_args()

    import qlib

    qlib.init(provider_uri=args.provider_uri, region=args.region)

    target_date = (
        _auto_last_trading_day()
        if (args.date is None or str(args.date).lower() == "auto")
        else args.date
    )

    task_cfg_path = Path(args.task_config)
    try:
        task_cfg = load_task_config(task_cfg_path)
    except TaskConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    training_range = get_training_time_range(task_cfg)
    training_start = training_range["start_time"]
    training_end_ts = pd.Timestamp(training_range["end_time"])

    start_date = args.start or training_start
    handler_end_ts = min(pd.Timestamp(target_date), training_end_ts)
    handler_end = str(handler_end_ts.date())
    if pd.Timestamp(start_date) > handler_end_ts:
        raise TaskConfigError(f"Start date {start_date} is after training cutoff {handler_end}")

    combined_path = Path(args.combined_factors)
    combined_df = load_combined_factors(combined_path)
    alpha_names = _extract_alpha_feature_names(task_cfg)
    new_factor_names = [c[1] for c in combined_df.columns if c[0] == "feature"]
    expected_order = alpha_names + [name for name in new_factor_names if name not in alpha_names]

    handler = build_handler_from_task(
        task_cfg=task_cfg,
        combined_factors_path=combined_path,
        start_time=start_date,
        end_time=handler_end,
    )

    df_feat = _prepare_feature_df_for_day(handler, target_date)
    df_feat = _reorder_columns(df_feat, expected_order)
    _assert_value_and_missing(df_feat)

    used_dates = df_feat.index.get_level_values("datetime").unique()
    used_date = str(pd.Timestamp(used_dates[0]).date()) if len(used_dates) else handler_end

    out_path = args.out or str(
        Path(__file__).resolve().parent / f"features_ready_infer_{used_date}.parquet"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(out_path, engine="pyarrow", compression="snappy")

    n_inst = df_feat.index.get_level_values("instrument").nunique()
    first_cols = [c[1] for c in df_feat.columns[:5]]
    print("==== Build Infer Features (Qlib) ====")
    print(f"Target trading day requested: {target_date}")
    print(f"Target trading day used: {used_date}")
    print(f"#Instruments: {n_inst}")
    print(f"#Features: {len(df_feat.columns)}")
    print(f"Columns preview: {first_cols} ...")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
