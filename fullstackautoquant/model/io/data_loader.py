from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["load_combined_factors"]


def load_combined_factors(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples([("feature", str(col)) for col in df.columns])
    if df.index.names != ["datetime", "instrument"]:
        df = df.copy()
        df.index.set_names(["datetime", "instrument"], inplace=True)
    return df.sort_index()

