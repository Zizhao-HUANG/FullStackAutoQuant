from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

__all__ = ["rank_signals", "compute_confidence"]


def rank_signals(predictions: pd.Series, target_date: str) -> Tuple[pd.Series, str]:
    mi_dt = predictions.index.get_level_values("datetime")
    available_dates = {str(dt.date()) for dt in mi_dt}
    used_date = target_date if target_date in available_dates else str(mi_dt.max().date())
    subset = predictions[mi_dt == pd.Timestamp(used_date)].sort_values(ascending=False)
    return subset, used_date


def compute_confidence(matrix: Iterable[np.ndarray]) -> np.ndarray:
    mat = np.vstack([np.atleast_1d(arr) for arr in matrix])
    std_vec = np.std(mat, axis=0)
    return 1.0 / (1.0 + std_vec)

