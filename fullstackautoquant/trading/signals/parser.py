from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd

from fullstackautoquant.trading.utils import instrument_to_gm


@dataclass(frozen=True)
class SignalRecord:
    date: str
    instrument: str
    symbol: str
    score: float
    confidence: float


def parse_ranked_scores(
    dataframe: pd.DataFrame,
    confidence_floor: float,
    topk: int,
) -> List[SignalRecord]:
    _ensure_required_columns(dataframe)
    date_str = _extract_single_date(dataframe)
    filtered = dataframe[dataframe["confidence"] >= confidence_floor].copy()
    filtered = _normalize_instruments(filtered)
    filtered = filtered.sort_values(by=["0", "confidence"], ascending=[False, False])
    if topk > 0 and len(filtered) > topk:
        filtered = filtered.head(topk)
    return [
        SignalRecord(
            date=date_str,
            instrument=row["instrument"],
            symbol=row["symbol"],
            score=float(row["0"]),
            confidence=float(row["confidence"]),
        )
        for _, row in filtered.iterrows()
    ]


def _ensure_required_columns(df: pd.DataFrame) -> None:
    required = {"datetime", "instrument", "0", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")


def _extract_single_date(df: pd.DataFrame) -> str:
    values = {str(val) for val in df["datetime"].unique()}
    if len(values) != 1:
        raise ValueError(f"Expected a single trading day, found: {sorted(values)}")
    return next(iter(values))


def _normalize_instruments(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["instrument"].str.match(r"^(SH|SZ)\d{6}$", na=False)].copy()
    df["symbol"] = df["instrument"].apply(instrument_to_gm)
    return df[df["symbol"].notna()].copy()




