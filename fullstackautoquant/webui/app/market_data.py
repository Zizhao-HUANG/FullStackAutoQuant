"""TuShare market data wrapper."""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

try:
    import tushare as ts
except ImportError:  # pragma: no cover - warn if not installed
    ts = None  # type: ignore


@dataclass
class MarketDataConfig:
    token: str
    max_calls_per_minute: int = 80


class MarketDataService:
    def __init__(self, config: MarketDataConfig) -> None:
        if ts is None:
            raise RuntimeError("tushare library not installed")
        ts.set_token(config.token)
        self.pro = ts.pro_api()
        self.cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _to_ts_code(symbol: str) -> str | None:
        if not symbol:
            return None
        sym = symbol.strip().upper()
        if not sym:
            return None
        if sym.endswith((".SH", ".SZ")) and len(sym) >= 9:
            return sym
        if sym.startswith("SHSE.") and len(sym) >= 11:
            body = sym.split(".", 1)[1]
            if body.isdigit():
                return f"{body}.SH"
        if sym.startswith("SZSE.") and len(sym) >= 11:
            body = sym.split(".", 1)[1]
            if body.isdigit():
                return f"{body}.SZ"
        if sym.startswith("SH") and len(sym) == 8 and sym[2:].isdigit():
            return f"{sym[2:]}.SH"
        if sym.startswith("SZ") and len(sym) == 8 and sym[2:].isdigit():
            return f"{sym[2:]}.SZ"
        if sym.isdigit() and len(sym) == 6:
            if sym.startswith(("0", "2", "3")):
                return f"{sym}.SZ"
            return f"{sym}.SH"
        return None

    def get_last_price(self, symbols: Iterable[str]) -> dict[str, float]:
        quotes = self.get_realtime_quotes(symbols)
        return {code: data["price"] for code, data in quotes.items() if data["price"] is not None}

    def get_realtime_quotes(
        self, symbols: Iterable[str], src: str = "sina"
    ) -> dict[str, dict[str, Any]]:
        seen: set[str] = set()
        original_symbols: list[str] = []
        for s in symbols:
            if not s:
                continue
            key = str(s)
            if key in seen:
                continue
            seen.add(key)
            original_symbols.append(key)

        if not original_symbols:
            return {}

        ts_codes: list[str] = []
        alias_map: dict[str, str] = {}
        for symbol in original_symbols:
            ts_code = self._to_ts_code(symbol)
            if ts_code:
                alias_map[ts_code] = symbol
                ts_codes.append(ts_code)

        if not ts_codes:
            return {}

        result_by_ts: dict[str, dict[str, Any]] = {}
        dt.datetime.now().strftime("%Y%m%d%H%M%S")

        batches: list[list[str]] = []
        step = 50 if src == "sina" else 1
        for i in range(0, len(ts_codes), step):
            batches.append(ts_codes[i : i + step])

        for batch in batches:
            try:
                df = ts.realtime_quote(ts_code=",".join(batch), src=src)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                row_dict = {str(col).lower(): row[col] for col in df.columns}
                code = str(row_dict.get("ts_code") or row_dict.get("symbol") or "").strip().upper()
                if not code:
                    continue
                price_val = row_dict.get("price")
                price = float(price_val) if price_val not in (None, "") else None
                name = str(row_dict.get("name") or row_dict.get("security_name") or "").strip()
                trade_time = str(row_dict.get("time") or row_dict.get("trade_time") or "")
                pre_close_val = (
                    row_dict.get("pre_close")
                    or row_dict.get("preclose")
                    or row_dict.get("prev_close")
                )
                pre_close = float(pre_close_val) if pre_close_val not in (None, "") else None
                payload = {
                    "price": price,
                    "name": name,
                    "time": trade_time,
                    "pre_close": pre_close,
                }
                self.cache[code] = payload
                result_by_ts[code] = payload

        final_result: dict[str, dict[str, Any]] = {}
        for ts_code, original in alias_map.items():
            payload = result_by_ts.get(ts_code)
            if payload:
                final_result[original] = payload
                continue
            if ts_code in self.cache:
                final_result[original] = dict(self.cache[ts_code])

        return final_result


def build_market_service(config: dict[str, Any]) -> MarketDataService:
    tushare_cfg = config.get("tushare", {})  # type: ignore[assignment]
    token = ""
    if isinstance(tushare_cfg, dict):
        token = str(tushare_cfg.get("token", ""))
        if not token:
            env_name = str(tushare_cfg.get("token_env", ""))
            if env_name:
                token = os.getenv(env_name, "")
    if not token:
        raise RuntimeError("TuShare Token not configured")
    return MarketDataService(MarketDataConfig(token=token))
