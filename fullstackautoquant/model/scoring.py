from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

__all__ = ["rank_signals", "compute_confidence"]


def rank_signals(predictions: pd.Series, target_date: str) -> tuple[pd.Series, str]:
    mi_dt = predictions.index.get_level_values("datetime")
    dt_index = pd.DatetimeIndex(mi_dt)
    available_dates = {str(d.date()) for d in dt_index}
    used_date = target_date if target_date in available_dates else str(dt_index.max().date())
    # Try native comparison first (works for strings), fall back to Timestamp (works for datetimes)
    mask = mi_dt == used_date
    if not mask.any():
        mask = mi_dt == pd.Timestamp(used_date)
    subset = predictions[mask].sort_values(ascending=False)
    return subset, used_date


def compute_confidence(matrix: Iterable[np.ndarray]) -> np.ndarray:
    mat = np.vstack([np.atleast_1d(arr) for arr in matrix])
    std_vec = np.std(mat, axis=0)
    return 1.0 / (1.0 + std_vec)
