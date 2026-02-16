"""Backtest result persistence and reader utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass(slots=True)
class BacktestRunMeta:
    """Persist backtest record metadata."""

    run_id: str
    created_at: datetime
    path: Path
    summary: Dict[str, float]
    config_path: Path


@dataclass(slots=True)
class SerializedArtifacts:
    root: Path
    logs_dir: Path


class ResultSerializer:
    def __init__(self, root: Path) -> None:
        self._root = root

    def persist(
        self,
        config: Dict[str, object],
        summary: Dict[str, float],
        equity: pd.DataFrame,
        trades: pd.DataFrame,
        positions: pd.DataFrame,
        artifacts: Dict[str, object] | None = None,
    ) -> SerializedArtifacts:
        run_dir = _timestamp_dir(self._root)
        run_dir.mkdir(parents=True, exist_ok=False)
        self._write_json(run_dir / "config.json", config)
        self._write_json(run_dir / "summary.json", summary)
        self._write_dataframe(run_dir / "equity_curve.csv", equity)
        self._write_dataframe(run_dir / "trades.csv", trades, index=False)
        self._write_dataframe(run_dir / "positions.csv", positions, index=False)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        for name, payload in (artifacts or {}).items():
            self._write_artifact(logs_dir / name, payload)
        return SerializedArtifacts(root=run_dir, logs_dir=logs_dir)

    def _write_json(self, path: Path, payload: Dict[str, object]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_dataframe(self, path: Path, df: pd.DataFrame, *, index: bool = True) -> None:
        if df.empty:
            path.write_text("", encoding="utf-8")
        else:
            df.to_csv(path, index=index)

    def _write_artifact(self, path: Path, payload: object) -> None:
        if isinstance(payload, pd.DataFrame):
            payload.to_csv(path, index=False)
        elif isinstance(payload, (list, dict)):
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        elif isinstance(payload, Path) and payload.exists():
            path.write_bytes(payload.read_bytes())


def _timestamp_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / ts
    counter = 1
    while run_dir.exists():
        run_dir = root / f"{ts}-{counter}"
        counter += 1
    return run_dir


def persist_backtest_result(
    root: Path,
    config: Dict[str, object],
    summary: Dict[str, float],
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    artifacts: Dict[str, object] | None = None,
) -> Path:
    serializer = ResultSerializer(root)
    bundle = serializer.persist(config, summary, equity, trades, positions, artifacts)
    return bundle.root


def list_backtest_runs(root: Path) -> List[BacktestRunMeta]:
    if not root.exists():
        return []
    runs: List[BacktestRunMeta] = []
    for child in sorted(root.iterdir() if root.is_dir() else [], reverse=True):
        if not child.is_dir():
            continue
        summary_file = child / "summary.json"
        config_file = child / "config.json"
        if not summary_file.exists() or not config_file.exists():
            continue
        try:
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        try:
            created_at = datetime.strptime(child.name[:19], "%Y%m%d-%H%M%S")
        except ValueError:
            created_at = datetime.fromtimestamp(child.stat().st_mtime)
        runs.append(
            BacktestRunMeta(
                run_id=child.name,
                created_at=created_at,
                path=child,
                summary=summary,
                config_path=config_file,
            )
        )
    runs.sort(key=lambda item: item.created_at, reverse=True)
    return runs


def load_backtest_result(run_dir: Path):
    equity_path = run_dir / "equity_curve.csv"
    trades_path = run_dir / "trades.csv"
    positions_path = run_dir / "positions.csv"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    metadata_path = run_dir / "metadata.json"
    signals_path = run_dir / "signals.json"
    risk_path = run_dir / "risk_records.json"
    manual_path = run_dir / "manual_decisions.json"
    nav_logs_path = run_dir / "logs" / "nav_history.csv"

    equity = pd.read_csv(equity_path, index_col=0, parse_dates=True) if equity_path.exists() and equity_path.stat().st_size > 0 else pd.DataFrame()
    trades = pd.read_csv(trades_path) if trades_path.exists() and trades_path.stat().st_size > 0 else pd.DataFrame()
    positions = pd.read_csv(positions_path) if positions_path.exists() and positions_path.stat().st_size > 0 else pd.DataFrame()
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    signals = json.loads(signals_path.read_text(encoding="utf-8")) if signals_path.exists() else []
    risk_records = json.loads(risk_path.read_text(encoding="utf-8")) if risk_path.exists() else []
    manual_decisions = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.exists() else []
    nav_logs = pd.read_csv(nav_logs_path) if nav_logs_path.exists() and nav_logs_path.stat().st_size > 0 else pd.DataFrame()
    return equity, trades, positions, summary, config, metadata, signals, risk_records, manual_decisions, nav_logs


__all__ = [
    "BacktestRunMeta",
    "persist_backtest_result",
    "list_backtest_runs",
    "load_backtest_result",
]

