"""Signal acquisition and filtering."""

from __future__ import annotations

import datetime as dt
from typing import Dict, Iterator, List, Tuple

import json
from pathlib import Path

from ..qlib_adapter import QlibInferenceAdapter


class SignalProvider:
    def __init__(self, adapter: QlibInferenceAdapter, confidence_floor: float, fallback_days: int) -> None:
        self._adapter = adapter
        self._confidence_floor = float(confidence_floor or 0.0)
        self._fallback_days = max(0, int(fallback_days))
        cache_dir = getattr(self._adapter.cfg, "cache_dir", None) or Path(__file__).resolve().parents[2] / "cache/signals"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[dt.date, List[dict]] = {}

    def iterate(self, calendar: List[dt.date]) -> Iterator[Tuple[dt.date, dt.date, List[dict]]]:
        for trade_date in calendar:
            source_date, signals = self._fetch_signals(trade_date)
            yield trade_date, source_date, self._normalize(signals, trade_date)

    def _fetch_signals(self, target_date: dt.date) -> Tuple[dt.date, List[dict]]:
        for offset in range(self._fallback_days + 1):
            candidate = target_date - dt.timedelta(days=offset)
            cached = self._load_cached(candidate)
            if cached is not None:
                return candidate, cached
            used_date, signals = self._adapter.generate_for_date(candidate)
            if signals:
                signals_copy = [dict(item) for item in signals]
                self._store_cache(used_date, signals_copy)
                return used_date, signals_copy
        return target_date, []

    def _normalize(self, signals: List[dict], trade_date: dt.date) -> List[dict]:
        result: List[dict] = []
        for item in signals:
            confidence = float(item.get("confidence", 0.0) or 0.0)
            if confidence < self._confidence_floor:
                continue
            data = dict(item)
            data["date"] = trade_date.isoformat()
            result.append(data)
        return result

    def _cache_path(self, date: dt.date) -> Path:
        return self._cache_dir / f"signals_{date.isoformat()}.json"

    def _load_cached(self, date: dt.date) -> List[dict] | None:
        if date in self._memory_cache:
            return [dict(item) for item in self._memory_cache[date]]
        path = self._cache_path(date)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            signals = payload.get("signals")
            if not isinstance(signals, list):
                return None
            self._memory_cache[date] = [dict(item) for item in signals]
            return [dict(item) for item in signals]
        except Exception:
            return None

    def _store_cache(self, date: dt.date, signals: List[dict]) -> None:
        self._memory_cache[date] = [dict(item) for item in signals]
        payload = {"date": date.isoformat(), "signals": signals}
        try:
            self._cache_path(date).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


