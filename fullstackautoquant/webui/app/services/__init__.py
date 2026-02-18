"""Service bootstrap: database, market data, and app state access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from app.config_loader import ConfigLoader, ensure_default_config
from app.database import Database, HistoricalPosition, build_database_from_config
from app.market_data import MarketDataService, build_market_service
from app.strategy_reader import find_latest_targets, load_targets

from .data_access import DataAccess


@dataclass
class Services:
    config: dict[str, Any]
    db: Database
    market: MarketDataService | None
    data_access: DataAccess

    def get_historical_positions(self, limit: int = 200) -> list[HistoricalPosition]:
        return self.db.get_historical_positions(limit=limit)


def bootstrap_services(base_dir: Path | None = None) -> Services:
    ensure_default_config(base_dir)
    loader = ConfigLoader(base_dir)
    config = loader.load()
    db = build_database_from_config(config)
    data_access = DataAccess()
    try:
        market = build_market_service(config)
    except RuntimeError:
        market = None
    return Services(config=config, db=db, market=market, data_access=data_access)


def load_latest_targets_dataframe(config: dict[str, Any]) -> pd.DataFrame:
    targets_dir = config.get("paths", {}).get("targets_dir", "../../trading/logs")  # type: ignore[assignment]
    base_dir = Path(__file__).resolve().parents[1]
    path = base_dir / targets_dir
    latest = find_latest_targets(path)
    return load_targets(latest)


def fetch_last_prices(market: MarketDataService | None, symbols: list[str]) -> dict[str, float]:
    if not symbols or market is None:
        return {}
    try:
        return market.get_last_price(symbols)
    except Exception:
        return {}


__all__ = [
    "bootstrap_services",
    "fetch_last_prices",
    "load_latest_targets_dataframe",
    "Services",
    "DataAccess",
]
