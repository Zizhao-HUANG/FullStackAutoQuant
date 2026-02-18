"""Workflow execution and debug module."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def _ensure_repo_paths() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    trading_dir = repo_root / "Trading"
    for candidate in (str(repo_root), str(trading_dir)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


_ensure_repo_paths()

from fullstackautoquant.trading.utils import compute_limit_price_from_rt_preclose, load_config

REPO_ROOT = Path(__file__).resolve().parents[3]
TRADING_DIR = REPO_ROOT / "Trading"
for candidate in (str(REPO_ROOT), str(TRADING_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


class WorkflowError(RuntimeError):
    pass


@dataclass
class StepResult:
    name: str
    stdout: str
    stderr: str
    output_paths: dict[str, Path]


def _resolve_path(base: Path, value: str | None, default: str) -> Path:
    ref = Path(value or default)
    return (base / ref).resolve()


def _filter_science_board_symbols(csv_path: Path) -> Path:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return csv_path

    if df.empty:
        return csv_path

    candidates = [
        col for col in df.columns if col and str(col).lower() in {"instrument", "symbol", "code"}
    ]
    if not candidates:
        return csv_path

    column = candidates[0]

    def _strip_digits(value: object) -> str:
        text = str(value)
        digits = re.sub(r"\D", "", text)
        return digits

    mask = df[column].apply(_strip_digits).str.startswith("68", na=False)
    filtered = df.loc[~mask].copy()

    if len(filtered) == len(df):
        return csv_path

    tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    try:
        filtered.to_csv(tmp_file, index=False)
    finally:
        tmp_file.close()

    return Path(tmp_file.name)


def _run(cmd: list[str], cwd: Path) -> StepResult:
    try:
        env = dict(os.environ)
        pythonpath = env.get("PYTHONPATH", "")
        repo_root_str = str(REPO_ROOT)
        if pythonpath:
            env["PYTHONPATH"] = repo_root_str + os.pathsep + pythonpath
        else:
            env["PYTHONPATH"] = repo_root_str
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise WorkflowError(
            f"Execution failed: {' '.join(cmd)}\nstdout: {exc.stdout}\nstderr: {exc.stderr}"
        ) from exc
    except Exception as exc:  # pragma: no cover
        raise WorkflowError(f"Execution error: {' '.join(cmd)} -> {exc}") from exc
    return StepResult(
        name=cmd[1] if len(cmd) > 1 else cmd[0],
        stdout=proc.stdout,
        stderr=proc.stderr,
        output_paths={},
    )


def run_full_workflow(
    config: dict[str, Any],
    *,
    current_positions: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    base_dir = Path(__file__).resolve().parents[1]
    paths_cfg = config.get("paths", {})  # type: ignore[assignment]
    trading_dir = _resolve_path(
        base_dir, str(paths_cfg.get("trading_dir", "../trading")), "../trading"
    )
    ranked_csv = _resolve_path(
        base_dir,
        str(
            paths_cfg.get(
                "ranked_csv", "../../ModelInferenceBundle/ranked_scores_AUTO_via_qlib.csv"
            )
        ),
        "../../ModelInferenceBundle/ranked_scores_AUTO_via_qlib.csv",
    )
    ranked_csv = _filter_science_board_symbols(ranked_csv)
    config_path = _resolve_path(
        base_dir,
        str(paths_cfg.get("trading_config", "../trading/config.auto.local.yaml")),
        "../trading/config.auto.local.yaml",
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="webui_workflow_"))
    python_exec = config.get("trading", {}).get("python", "python")  # type: ignore[assignment]

    signals_path = temp_dir / "signals.json"
    risk_path = temp_dir / "risk_state.json"
    targets_path = temp_dir / "targets.json"
    orders_path = temp_dir / "orders.json"
    current_positions_path: Path | None = None
    if current_positions is not None:
        payload = {"positions": list(current_positions)}
        current_positions_path = temp_dir / "current_positions.json"
        current_positions_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    steps: list[StepResult] = []
    step = _run(
        [
            python_exec,
            "signals_from_csv.py",
            "--csv",
            str(ranked_csv),
            "--config",
            str(config_path),
            "--out",
            str(signals_path),
        ],
        cwd=trading_dir,
    )
    step.output_paths["signals"] = signals_path
    steps.append(step)

    step = _run(
        [
            python_exec,
            "risk_manager.py",
            "--signals",
            str(signals_path),
            "--config",
            str(config_path),
            "--out",
            str(risk_path),
        ],
        cwd=trading_dir,
    )
    step.output_paths["risk_state"] = risk_path
    steps.append(step)

    step_cmd = [
        python_exec,
        "strategy_rebalance.py",
        "--signals",
        str(signals_path),
        "--risk_state",
        str(risk_path),
        "--config",
        str(config_path),
        "--targets",
        str(targets_path),
        "--orders",
        str(orders_path),
    ]

    if current_positions_path is not None and current_positions_path.exists():
        step_cmd += ["--current_positions", str(current_positions_path)]

    step = _run(step_cmd, cwd=trading_dir)
    step.output_paths.update({"targets": targets_path, "orders": orders_path})
    steps.append(step)

    result: dict[str, Any] = {
        "steps": [
            {
                "name": s.name,
                "stdout": s.stdout,
                "stderr": s.stderr,
                "outputs": {k: str(v) for k, v in s.output_paths.items()},
            }
            for s in steps
        ]
    }

    if signals_path.exists():
        result["signals"] = json.loads(signals_path.read_text(encoding="utf-8"))
    if risk_path.exists():
        result["risk_state"] = json.loads(risk_path.read_text(encoding="utf-8"))
    if targets_path.exists():
        result["targets"] = json.loads(targets_path.read_text(encoding="utf-8"))
    if orders_path.exists():
        raw_orders = json.loads(orders_path.read_text(encoding="utf-8"))
        corrected_orders = _recalculate_buy_limits(
            raw_orders,
            cfg_path=config_path,
            signals_payload=result.get("signals"),
        )
        result["orders"] = corrected_orders
        orders_path.write_text(json.dumps(corrected_orders, ensure_ascii=False), encoding="utf-8")

    return result


def _recalculate_buy_limits(
    orders_payload: dict, cfg_path: Path, signals_payload: dict | None
) -> dict:
    if not isinstance(orders_payload, dict) or "orders" not in orders_payload:
        return orders_payload
    orders = orders_payload.get("orders", [])
    if not orders:
        return orders_payload

    cfg = load_config(str(cfg_path))
    buy_offset = float(cfg.get("order", {}).get("buy_limit_offset", 0.02))
    sell_offset = float(cfg.get("order", {}).get("sell_limit_offset", -0.05))
    limit_threshold = float(cfg.get("order", {}).get("limit_threshold", 0.095))
    tick = float(cfg.get("order", {}).get("clamp_tick", 0.01))

    signals = (
        (signals_payload or {}).get("signals", []) if isinstance(signals_payload, dict) else []
    )
    quote_map = {item.get("symbol"): item for item in signals if isinstance(item, dict)}

    adjusted = []
    for order in orders:
        if not isinstance(order, dict) or order.get("side", "").upper() != "BUY":
            adjusted.append(order)
            continue
        symbol = order.get("symbol")
        quote_info = quote_map.get(symbol, {})
        ref_price = float(
            order.get("ref_price") or quote_info.get("price") or order.get("price") or 0.0
        )
        quote_info = quote_map.get(symbol, {})
        rt_price = float(
            quote_info.get("price") or quote_info.get("ask") or quote_info.get("bid") or ref_price
        )
        pre_close = float(quote_info.get("pre_close") or ref_price)
        if ref_price <= 0 and rt_price <= 0:
            adjusted.append(order)
            continue
        buy_price = compute_limit_price_from_rt_preclose(
            "BUY",
            rt_price if rt_price > 0 else ref_price,
            pre_close if pre_close > 0 else ref_price,
            buy_offset,
            sell_offset,
            limit_threshold,
            tick,
        )
        updated = dict(order)
        updated["price"] = buy_price
        adjusted.append(updated)

    return {**orders_payload, "orders": adjusted}


def run_single_step(
    config: dict[str, Any],
    step: str,
    *,
    current_positions: Iterable[dict[str, Any]] | None = None,
) -> StepResult:
    base_dir = Path(__file__).resolve().parents[1]
    paths_cfg = config.get("paths", {})  # type: ignore[assignment]
    trading_dir = _resolve_path(
        base_dir, str(paths_cfg.get("trading_dir", "../trading")), "../trading"
    )
    ranked_csv = _resolve_path(
        base_dir,
        str(
            paths_cfg.get(
                "ranked_csv", "../../ModelInferenceBundle/ranked_scores_AUTO_via_qlib.csv"
            )
        ),
        "../../ModelInferenceBundle/ranked_scores_AUTO_via_qlib.csv",
    )
    ranked_csv = _filter_science_board_symbols(ranked_csv)
    config_path = _resolve_path(
        base_dir,
        str(paths_cfg.get("trading_config", "../trading/config.auto.local.yaml")),
        "../trading/config.auto.local.yaml",
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="webui_single_"))
    python_exec = config.get("trading", {}).get("python", "python")  # type: ignore[assignment]

    if step == "signals":
        out_path = temp_dir / "signals.json"
        res = _run(
            [
                python_exec,
                "signals_from_csv.py",
                "--csv",
                str(ranked_csv),
                "--config",
                str(config_path),
                "--out",
                str(out_path),
            ],
            cwd=trading_dir,
        )
        res.output_paths["signals"] = out_path
        return res

    if step == "risk":
        signals_path = temp_dir / "signals.json"
        _run(
            [
                python_exec,
                "signals_from_csv.py",
                "--csv",
                str(ranked_csv),
                "--config",
                str(config_path),
                "--out",
                str(signals_path),
            ],
            cwd=trading_dir,
        )
        risk_path = temp_dir / "risk_state.json"
        res = _run(
            [
                python_exec,
                "risk_manager.py",
                "--signals",
                str(signals_path),
                "--config",
                str(config_path),
                "--out",
                str(risk_path),
            ],
            cwd=trading_dir,
        )
        res.output_paths.update({"signals": signals_path, "risk_state": risk_path})
        return res

    if step == "strategy":
        signals_path = temp_dir / "signals.json"
        risk_path = temp_dir / "risk_state.json"
        _run(
            [
                python_exec,
                "signals_from_csv.py",
                "--csv",
                str(ranked_csv),
                "--config",
                str(config_path),
                "--out",
                str(signals_path),
            ],
            cwd=trading_dir,
        )
        _run(
            [
                python_exec,
                "risk_manager.py",
                "--signals",
                str(signals_path),
                "--config",
                str(config_path),
                "--out",
                str(risk_path),
            ],
            cwd=trading_dir,
        )
        targets_path = temp_dir / "targets.json"
        orders_path = temp_dir / "orders.json"
        current_positions_path = None
        if current_positions is not None:
            current_positions_path = temp_dir / "current_positions.json"
            payload = {"positions": list(current_positions)}
            current_positions_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )

        cmd = [
            python_exec,
            "strategy_rebalance.py",
            "--signals",
            str(signals_path),
            "--risk_state",
            str(risk_path),
            "--config",
            str(config_path),
            "--targets",
            str(targets_path),
            "--orders",
            str(orders_path),
        ]
        if current_positions_path is not None:
            cmd += ["--current_positions", str(current_positions_path)]
        res = _run(cmd, cwd=trading_dir)
        res.output_paths.update(
            {
                "signals": signals_path,
                "risk_state": risk_path,
                "targets": targets_path,
                "orders": orders_path,
            }
        )
        return res

    raise WorkflowError(f"Unknown step: {step}")
