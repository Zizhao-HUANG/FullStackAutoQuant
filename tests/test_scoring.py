from __future__ import annotations

import numpy as np
import pandas as pd
from fullstackautoquant.model.scoring import compute_confidence, rank_signals


def test_rank_signals_fallback_to_latest() -> None:
    idx = pd.MultiIndex.from_tuples(
        [
            ("2025-10-16", "SH600000"),
            ("2025-10-17", "SH600001"),
        ],
        names=["datetime", "instrument"],
    )
    series = pd.Series([0.1, 0.9], index=idx)

    ranked, used_date = rank_signals(series, "2025-10-18")
    assert used_date == "2025-10-17"
    assert ranked.iloc[0] == 0.9


def test_compute_confidence() -> None:
    runs = [np.array([0.5, 0.5]), np.array([1.0, 0.5])]
    out = compute_confidence(runs)
    assert np.allclose(out, [1 / (1 + 0.25), 1.0])




