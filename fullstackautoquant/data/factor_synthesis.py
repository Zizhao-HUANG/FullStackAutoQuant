#!/usr/bin/env python3

"""
Daily update of combined_factors_df.parquet(native Experiment/Runner alternative):
- Uses RD-Agent QlibFactorExperiment + process_factor_data execution and merge
- Uses existing factor directories (containing factor.py) as implementation source, links to daily_pv.h5 in the data directory
- Save/overwrite new synthesized factor DataFrame to target workspace:combined_factors_df.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
from pathlib import Path

import pandas as pd
import qlib
from qlib.data import D

# Note: avoid importing RD-Agent Config module before setting env vars (Settings freeze env vars on import).


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workspace",
        required=True,
        help="Target workspace directory, output will be workspace/combined_factors_df.parquet",
    )
    ap.add_argument(
        "--log-dir", default=None, help="log outputdirectory, defaults to workspace/logs"
    )
    ap.add_argument(
        "--provider_uri",
        default="~/.qlib/qlib_data/cn_data",
        help="QLib provider uri(default ~/.qlib/qlib_data/cn_data)",
    )
    ap.add_argument("--region", default="cn", help="QLib region (default cn)")
    ap.add_argument(
        "--instruments", default="csi300", help="Instrument universe to keep (default csi300)"
    )
    args = ap.parse_args()

    ws = Path(args.workspace).resolve()
    assert ws.exists(), f"workspace does not exist: {ws}"

    log_dir = Path(args.log_dir).resolve() if args.log_dir else ws / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"update_combined_factors_{timestamp}.log"

    logger = logging.getLogger("factor_update")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info("==== Combined Factors Daily Update (Native Experiment) ====")
    logger.info("workspace=%s", ws)
    logger.info("log_file=%s", log_file)

    # Runtime dependency check: ensure polars exists for factor implementations (avoid ModuleNotFoundError)
    # Minimum intrusion: install fixed version only if missing, for stable behavior.
    try:
        import importlib
        import subprocess
        import sys

        importlib.import_module("polars")
    except Exception:
        logger.info(
            "polars not installed, installing polars==1.32.0 to support factor computation……"
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "polars==1.32.0"]
        )  # noqa: S603,S607
        importlib.invalidate_caches()

    # Specify factor implementation directory (containing factor.py)
    base_dir = Path(__file__).resolve().parent
    factor_dirs = [
        base_dir / "8fef89cc9aca41c0bc843a1c14229259",
        base_dir / "1df46d60d6134d1a9801dfca16491986",
    ]
    for d in factor_dirs:
        if not (d.exists() and (d / "factor.py").exists()):
            raise FileNotFoundError(f"Missing factor implementation:{d}/factor.py")

    # Ensure data directory contains daily_pv.h5(for factor implementations to read)
    data_dir = Path(__file__).resolve().parent  # defaults to this directory, requires daily_pv.h5
    data_file = data_dir / "daily_pv.h5"
    if not data_file.exists():
        logger.error("Not found daily_pv.h5:%s", data_file)
        raise FileNotFoundError(
            f"Not found daily_pv.h5:{data_file}, please run export script first export_daily_pv_from_qlib.py"
        )

    daily_stats = _inspect_h5_last_rows(data_file)
    logger.info(
        "daily_pv.h5 Latest date=%s, samples=%d",
        daily_stats.get("max_date"),
        daily_stats.get("rows"),
    )

    # Initialize Qlib (for subsequent instrument universe filtering)
    provider_uri = str(Path(args.provider_uri).expanduser())
    qlib.init(provider_uri=provider_uri, region=args.region)

    # Set RD-Agent factor execution data directory (overrides both debug and full). Must be set before importing RD-Agent components!
    os.environ["FACTOR_COSTEER_DATA_FOLDER"] = str(base_dir)
    os.environ["FACTOR_COSTEER_DATA_FOLDER_DEBUG"] = str(base_dir)

    # Strict strategy: disable native execution path, always locally execute both factor.py and merge results
    def _save_dataframe_to_parquet(df: pd.DataFrame, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, engine="pyarrow")

    import subprocess
    import sys

    # For each factor directory, ensure data access: create symlink in factor directory pointing to base_dir/daily_pv.h5 (if not exists)
    existing_snapshot = _inspect_parquet(ws / "combined_factors_df.parquet")
    if existing_snapshot:
        logger.info(
            "Current combined_factors_df.parquet coverage: %s -> %s (#rows=%d)",
            existing_snapshot["min_date"],
            existing_snapshot["max_date"],
            existing_snapshot["rows"],
        )

    for d in factor_dirs:
        link_path = d / "daily_pv.h5"
        if link_path.exists() or link_path.is_symlink():
            try:
                link_path.unlink()
            except OSError:
                logger.warning("Cannot delete old daily_pv.h5:%s, attempting overwrite", link_path)
        if not link_path.exists():
            try:
                link_path.symlink_to(data_file)
            except OSError as exc:
                logger.warning(
                    "Symlink creation failed, will copy daily_pv.h5 to factor directory: %s", exc
                )
                import shutil

                shutil.copy2(data_file, link_path)

    # Execute each factor.py locally (using current interpreter), working directory set to each factor directory, expected output result.h5
    result_paths = []
    for d in factor_dirs:
        fp = d / "factor.py"
        assert fp.exists(), f"Missing factor script: {fp}"
        logger.info("Running factor script: %s", fp)
        proc = subprocess.run([sys.executable, str(fp)], cwd=str(d), capture_output=True, text=True)
        if proc.stdout:
            logger.info("%s stdout:\n%s", fp.name, proc.stdout[-2000:])
        if proc.stderr:
            logger.warning("%s stderr:\n%s", fp.name, proc.stderr[-2000:])
        if proc.returncode != 0:
            logger.error("Factor script failed: %s, exit code=%s", fp, proc.returncode)
            raise RuntimeError(f"Factor script execution failed: {fp}")
        rp = d / "result.h5"
        if not rp.exists():
            raise FileNotFoundError(f"Factor did not generate result file: {rp}")
        stats = _inspect_h5_last_rows(rp)
        logger.info(
            "%s Latest date=%s, samples=%d", rp.name, stats.get("max_date"), stats.get("rows")
        )
        result_paths.append(rp)

    # Merge two HDF5 results into unified MultiIndex('feature', name) DataFrame
    dfs = []
    for rp in result_paths:
        df_i = pd.read_hdf(rp, key="data")
        # Promote single column name to feature level
        if not isinstance(df_i.columns, pd.MultiIndex):
            df_i.columns = pd.MultiIndex.from_product([["feature"], df_i.columns])
        dfs.append(df_i)
    combined = pd.concat(dfs, axis=1)
    # Deduplicate (if column names conflict)
    combined = combined.loc[:, ~combined.columns.duplicated(keep="last")]
    combined = _filter_to_universe(combined, args.instruments, logger)

    out_path = ws / "combined_factors_df.parquet"
    _save_dataframe_to_parquet(combined, out_path)

    # Output
    # Verify output file exists
    out_path = ws / "combined_factors_df.parquet"
    if not out_path.exists():
        raise FileNotFoundError(f"Not generated: {out_path}")

    # Read and summarize
    df = pd.read_parquet(out_path)
    if not isinstance(df.columns, pd.MultiIndex):
        raise AssertionError(
            "combined_factors_df.parquet columns must be MultiIndex('feature', name)"
        )
    n_dates = df.index.get_level_values("datetime").nunique()
    n_insts = df.index.get_level_values("instrument").nunique()

    logger.info("Saved: %s", out_path)
    logger.info(
        "Shape: %s, #dates=%d, #insts=%d, #features=%d", df.shape, n_dates, n_insts, len(df.columns)
    )

    combined_max = df.index.get_level_values("datetime").max()
    expected_max = daily_stats.get("max_date")
    if expected_max and combined_max < expected_max:
        logger.error(
            "combined_factors_df.parquet Latest date(%s) is earlier than daily_pv.h5(%s)",
            combined_max,
            expected_max,
        )
        raise RuntimeError("Factor data does not cover latest trading day, see logs")

    logger.info("Done.Log: %s", log_file)

    return 0


def _filter_to_universe(df: pd.DataFrame, universe: str, logger: logging.Logger) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex) or "instrument" not in df.index.names:
        raise ValueError("combined factor index must be MultiIndex('datetime','instrument')")
    target = universe or "csi300"
    inst_conf = D.instruments(market=target)
    instruments = D.list_instruments(inst_conf, as_list=True)
    if not instruments:
        raise ValueError(f"Cannot get instrument universe {target}")
    inst_set = set(instruments)
    inst_level = df.index.get_level_values("instrument")
    extra = set(inst_level.unique()) - inst_set
    if extra:
        logger.warning(
            "Factor result contains %d  non- %s stocks, will be automatically removed. e.g.: %s",
            len(extra),
            target,
            sorted(extra)[:5],
        )
    filtered = df[inst_level.isin(inst_set)]
    if filtered.empty:
        raise ValueError(
            f"Filter instrument universe {target} resulted in empty data, check factor output"
        )
    return filtered


def _inspect_h5_last_rows(path: Path, tail_rows: int = 5) -> dict:
    if not path.exists():
        return {"exists": False}
    with pd.HDFStore(path, "r") as store:
        storer = store.get_storer("data")
        rows = getattr(storer, "nrows", None)
        if rows is None:
            # fixed-format HDF5: fall back to loading entire dataset then truncating
            logger = logging.getLogger("factor_update")
            try:
                df_full = store.get("data")
            except (KeyError, ValueError):
                return {"exists": False}
            if df_full.empty:
                return {"exists": True, "rows": 0, "max_date": None, "tail": df_full}
            rows = len(df_full)
            tail = df_full.tail(tail_rows)
            if logger.handlers:
                logger.warning(
                    "daily_pv.h5 is fixed format, degraded to full read then truncate. Recommend using export_daily_pv_from_qlib.py re-export (format='table') for faster data inspection."
                )
        else:
            start = max(0, rows - tail_rows)
            tail = store.select("data", start=start)
    dates = (
        tail.index.get_level_values("datetime")
        if isinstance(tail.index, pd.MultiIndex)
        else tail.index
    )
    return {
        "exists": True,
        "rows": rows,
        "max_date": dates.max() if len(dates) else None,
        "tail": tail,
    }


def _inspect_parquet(path: Path) -> dict | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    idx = df.index
    dates = idx.get_level_values("datetime") if isinstance(idx, pd.MultiIndex) else idx
    return {
        "rows": len(df),
        "min_date": dates.min(),
        "max_date": dates.max(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
