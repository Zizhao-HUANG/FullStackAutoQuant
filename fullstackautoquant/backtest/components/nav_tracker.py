"""Net value tracker."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


class NavTracker:
    """Records and persists NAV time series."""

    def __init__(self) -> None:
        self._records: Dict[dt.date, Dict[str, float]] = {}

    def append(self, date: dt.date, equity: float, cash: Optional[float] = None, market_value: Optional[float] = None) -> None:
        record = self._records.setdefault(date, {"equity": 0.0, "cash": 0.0, "market_value": 0.0})
        record["equity"] = float(equity)
        if cash is not None:
            record["cash"] = float(cash)
        if market_value is not None:
            record["market_value"] = float(market_value)

    def ensure_previous(
        self,
        trade_date: dt.date,
        prev_equity: Optional[float],
        prev_cash: Optional[float] = None,
        prev_market_value: Optional[float] = None,
    ) -> None:
        if prev_equity is None:
            return
        prev = trade_date - dt.timedelta(days=1)
        record = self._records.setdefault(prev, {"equity": 0.0, "cash": 0.0, "market_value": 0.0})
        record["equity"] = float(prev_equity)
        if prev_cash is not None:
            record["cash"] = float(prev_cash)
        if prev_market_value is not None:
            record["market_value"] = float(prev_market_value)

    def update_components(self, date: dt.date, cash: float, market_value: float) -> None:
        record = self._records.setdefault(date, {"equity": 0.0, "cash": 0.0, "market_value": 0.0})
        record["cash"] = float(cash)
        record["market_value"] = float(market_value)
        if record["equity"] == 0.0:
            record["equity"] = float(cash + market_value)

    def to_rows(self) -> Iterable[Dict[str, object]]:
        for day, data in sorted(self._records.items()):
            row = {
                "date": day.isoformat(),
                "equity": data.get("equity", 0.0),
                "cash": data.get("cash", 0.0),
                "market_value": data.get("market_value", 0.0),
            }
            row["nav"] = row["equity"]
            yield row

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.to_rows()))

    def write_csv(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        if df.empty:
            target.write_text("", encoding="utf-8")
            return
        df.to_csv(target, index=False)


