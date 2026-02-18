"""Research/inference pipeline subprocess runner wrapper.

Provides StepResult structure and error handling consistent with `workflow._run`,
executes scripts via `conda run -n` in an external environment.
"""

from __future__ import annotations

import datetime as dt
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class InferenceError(RuntimeError):
    pass


@dataclass
class StepResult:
    name: str
    stdout: str
    stderr: str
    output_paths: dict[str, Path]
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    duration_sec: float | None = None


def _run_cmd(cmd: list[str], cwd: Path) -> StepResult:
    try:
        started_at = dt.datetime.now()
        start_perf = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
        finished_at = dt.datetime.now()
        duration_sec = time.perf_counter() - start_perf
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise InferenceError(
            f"Execution failed: {' '.join(shlex.quote(x) for x in cmd)}\nstdout: {exc.stdout}\nstderr: {exc.stderr}"
        ) from exc
    except Exception as exc:  # pragma: no cover
        raise InferenceError(
            f"Execution error: {' '.join(shlex.quote(x) for x in cmd)} -> {exc}"
        ) from exc
    name = cmd[2] if len(cmd) > 2 and cmd[0] == "conda" else (cmd[0] if cmd else "")
    return StepResult(
        name=name,
        stdout=proc.stdout,
        stderr=proc.stderr,
        output_paths={},
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration_sec,
    )


def _resolve(base: Path, rel_or_abs: str) -> Path:
    p = Path(os.path.expanduser(rel_or_abs))
    if p.is_absolute():
        return p
    return (base / p).resolve()


def run_update_qlib(config: dict[str, Any]) -> StepResult:
    base_dir = Path(__file__).resolve().parents[2]
    paths = config.get("paths", {})  # type: ignore[assignment]
    infer_dir = _resolve(base_dir, str(paths.get("model_infer_dir", "../../ModelInferenceBundle")))

    infer_cfg = config.get("inference", {})  # type: ignore[assignment]
    conda_env = str(infer_cfg.get("conda_env", "rdagent4qlib"))

    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "bash",
        str(infer_dir / "qlib_update_cn_latest.sh"),
    ]
    return _run_cmd(cmd, cwd=infer_dir)


def run_export_daily_pv(config: dict[str, Any]) -> StepResult:
    base_dir = Path(__file__).resolve().parents[2]
    paths = config.get("paths", {})  # type: ignore[assignment]
    infer_dir = _resolve(base_dir, str(paths.get("model_infer_dir", "../../ModelInferenceBundle")))

    infer_cfg = config.get("inference", {})  # type: ignore[assignment]
    conda_env = str(infer_cfg.get("conda_env", "rdagent4qlib"))
    python_exec = str(infer_cfg.get("python_exec", "python"))
    qlib_data = os.path.expanduser(str(infer_cfg.get("qlib_data", "~/.qlib/qlib_data/cn_data")))

    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        python_exec,
        "./export_daily_pv_from_qlib.py",
        "--end",
        "auto",
        "--instruments",
        "all",
        "--provider_uri",
        qlib_data,
        "--region",
        "cn",
        "--out",
        "./daily_pv.h5",
    ]
    return _run_cmd(cmd, cwd=infer_dir)


def run_verify_parquet(config: dict[str, Any]) -> StepResult:
    base_dir = Path(__file__).resolve().parents[2]
    paths = config.get("paths", {})  # type: ignore[assignment]
    infer_dir = _resolve(base_dir, str(paths.get("model_infer_dir", "../../ModelInferenceBundle")))

    infer_cfg = config.get("inference", {})  # type: ignore[assignment]
    conda_env = str(infer_cfg.get("conda_env", "rdagent4qlib"))
    python_exec = str(infer_cfg.get("python_exec", "python"))

    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        python_exec,
        "./verify_parquet_ready_for_infer.py",
        "--path",
        "./features_ready_infer_AUTO.parquet",
        "--combined_factors",
        "./combined_factors_df.parquet",
    ]
    return _run_cmd(cmd, cwd=infer_dir)


def run_build_factors(config: dict[str, Any]) -> StepResult:
    base_dir = Path(__file__).resolve().parents[2]
    paths = config.get("paths", {})  # type: ignore[assignment]
    infer_dir = _resolve(base_dir, str(paths.get("model_infer_dir", "../../ModelInferenceBundle")))

    infer_cfg = config.get("inference", {})  # type: ignore[assignment]
    conda_env = str(infer_cfg.get("conda_env", "rdagent4qlib"))
    python_exec = str(infer_cfg.get("python_exec", "python"))

    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        python_exec,
        "./update_combined_factors_daily.py",
        "--workspace",
        str(infer_dir),
    ]
    return _run_cmd(cmd, cwd=infer_dir)


def run_inference(config: dict[str, Any]) -> StepResult:
    base_dir = Path(__file__).resolve().parents[2]
    paths = config.get("paths", {})  # type: ignore[assignment]
    infer_dir = _resolve(base_dir, str(paths.get("model_infer_dir", "../../ModelInferenceBundle")))

    infer_cfg = config.get("inference", {})  # type: ignore[assignment]
    conda_env = str(infer_cfg.get("conda_env", "rdagent4qlib"))
    python_exec = str(infer_cfg.get("python_exec", "python"))

    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        python_exec,
        "./run_inference_via_qlib.py",
        "--date",
        "auto",
        "--combined_factors",
        "./combined_factors_df.parquet",
        "--params",
        "./params.pkl",
        "--out",
        "./ranked_scores_AUTO_via_qlib.csv",
    ]
    return _run_cmd(cmd, cwd=infer_dir)


def run_full_pipeline(config: dict[str, Any]) -> list[StepResult]:
    steps: list[StepResult] = []
    # Phase 1: Update Qlib data (optional, user may skip)
    try:
        steps.append(run_update_qlib(config))
    except Exception as exc:  # pragma: no cover
        now = dt.datetime.now()
        steps.append(
            StepResult(
                name="qlib_update",
                stdout="",
                stderr=str(exc),
                output_paths={},
                started_at=now,
                finished_at=now,
                duration_sec=0.0,
            )
        )

    # Phase 2: Export daily_pv.h5
    steps.append(run_export_daily_pv(config))

    # Phase 3: Factor Synthesis
    steps.append(run_build_factors(config))

    # Phase 4: Inference
    steps.append(run_inference(config))

    return steps
