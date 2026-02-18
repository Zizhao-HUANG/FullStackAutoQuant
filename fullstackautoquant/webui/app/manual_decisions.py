from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

MANUAL_DECISIONS_VERSION = 1


def _resolve_manual_dir(config: dict[str, Any]) -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    paths_cfg = config.get("paths", {}) if isinstance(config.get("paths"), dict) else {}
    manual_dir_ref = paths_cfg.get("manual_logs_dir", "../trading/logs/manual")
    manual_dir = (base_dir / str(manual_dir_ref)).resolve()
    manual_dir.mkdir(parents=True, exist_ok=True)
    return manual_dir


def _signal_lookup(signals_payload: object | None) -> tuple[str | None, dict[str, dict[str, Any]]]:
    if not isinstance(signals_payload, dict):
        return None, {}
    signal_date = str(signals_payload.get("date")) if signals_payload.get("date") else None
    entries = signals_payload.get("signals", [])
    lookup: dict[str, dict[str, Any]] = {}
    if isinstance(entries, Iterable):
        for item in entries:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or item.get("instrument") or "").strip()
            if not symbol:
                continue
            lookup[symbol] = dict(item)
    return signal_date, lookup


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_manual_decisions(
    orders: pd.DataFrame,
    signals_payload: object | None = None,
    *,
    applied_at: dt.datetime | None = None,
    note: str | None = None,
    approved_by: str | None = None,
) -> tuple[list[dict[str, Any]], dt.datetime]:
    if orders.empty:
        return [], applied_at or dt.datetime.now(dt.timezone.utc)

    applied_ts = applied_at or dt.datetime.now(dt.timezone.utc)
    signal_date, signal_lookup = _signal_lookup(signals_payload)
    executed_date = applied_ts.date().isoformat()

    decisions: list[dict[str, Any]] = []
    base_prefix = applied_ts.strftime("%Y%m%d%H%M%S")

    for idx, row in orders.iterrows():
        symbol = str(row.get("symbol", "")).strip()
        if not symbol:
            continue
        side = str(row.get("side", "")).upper()
        if side not in {"BUY", "SELL"}:
            continue
        volume = _coerce_float(row.get("volume"))
        if volume is None or volume <= 0:
            continue
        price = _coerce_float(row.get("price"))
        meta = signal_lookup.get(symbol) or {}
        confidence = _coerce_float(meta.get("confidence"))
        record = {
            "decision_id": f"WEBUI-{base_prefix}-{idx:03d}-{side}-{symbol.replace('.', '')}",
            "signal_date": str(meta.get("date") or signal_date or executed_date),
            "queued_at": applied_ts.isoformat(),
            "execute_date": executed_date,
            "symbol": symbol,
            "action": side,
            "confidence": confidence,
            "status": "executed",
            "volume": float(volume),
            "price": float(price) if price is not None else None,
            "source": "webui",
        }
        if note:
            record["note"] = note
        if approved_by:
            record["approved_by"] = approved_by
        if "score" in meta:
            record["score"] = _coerce_float(meta.get("score"))
        if "strategy" in meta and meta.get("strategy"):
            record["strategy"] = meta.get("strategy")
        elif "strategy_id" in meta and meta.get("strategy_id"):
            record["strategy"] = meta.get("strategy_id")
        decisions.append(record)

    return decisions, applied_ts


def append_manual_decisions(
    config: dict[str, Any],
    decisions: list[dict[str, Any]],
    *,
    applied_at: dt.datetime,
) -> Path:
    manual_dir = _resolve_manual_dir(config)
    aggregate_path = manual_dir / "manual_decisions.json"

    existing: list[dict[str, Any]] = []
    existing_version: int | None = None
    if aggregate_path.exists():
        try:
            loaded = json.loads(aggregate_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = list(loaded)
            elif isinstance(loaded, dict):
                existing_version = int(loaded.get("version") or MANUAL_DECISIONS_VERSION)
                entries = loaded.get("entries")
                if isinstance(entries, list):
                    existing = list(entries)
        except Exception:
            backup_path = (
                manual_dir / f"manual_decisions_corrupt_{applied_at.strftime('%Y%m%d%H%M%S')}.json"
            )
            aggregate_path.replace(backup_path)

    aggregate_entries = existing + decisions
    aggregate_payload = {
        "version": MANUAL_DECISIONS_VERSION,
        "generated_at": applied_at.isoformat(),
        "count": len(aggregate_entries),
        "previous_version": existing_version,
        "entries": aggregate_entries,
    }
    aggregate_path.write_text(
        json.dumps(aggregate_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    daily_dir = manual_dir / applied_at.strftime("%Y%m%d")
    daily_dir.mkdir(parents=True, exist_ok=True)
    daily_file = daily_dir / f"manual_decisions_{applied_at.strftime('%H%M%S')}.json"
    daily_payload = {
        "version": MANUAL_DECISIONS_VERSION,
        "date": applied_at.date().isoformat(),
        "count": len(decisions),
        "entries": decisions,
    }
    daily_file.write_text(json.dumps(daily_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return daily_file


def log_manual_decisions(
    config: dict[str, Any],
    orders: pd.DataFrame,
    signals_payload: object | None = None,
    *,
    note: str | None = None,
    approved_by: str | None = None,
) -> tuple[Path | None, list[dict[str, Any]]]:
    decisions, applied_at = build_manual_decisions(
        orders,
        signals_payload,
        note=note,
        approved_by=approved_by,
    )
    if not decisions:
        return None, []
    path = append_manual_decisions(config, decisions, applied_at=applied_at)
    return path, decisions
