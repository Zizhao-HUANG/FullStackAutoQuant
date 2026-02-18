from __future__ import annotations

from pathlib import Path

import pandas as pd

from fullstackautoquant.model.io.data_loader import load_combined_factors


def test_load_combined_factors(tmp_path: Path) -> None:
    data = {
        ("feature", "alpha1"): [0.1, 0.2],
        ("feature", "alpha2"): [0.3, 0.4],
    }
    index = pd.MultiIndex.from_tuples(
        [("2025-10-17", "SH600000"), ("2025-10-17", "SH600001")],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(data, index=index)
    path = tmp_path / "combined.parquet"
    df.to_parquet(path)

    loaded = load_combined_factors(path)
    assert list(loaded.columns) == [
        ("feature", "alpha1"),
        ("feature", "alpha2"),
    ]
    assert loaded.index.names == ["datetime", "instrument"]
