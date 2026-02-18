from __future__ import annotations

import pandas as pd
import pytest
from fullstackautoquant.trading.signals.parser import SignalRecord, parse_ranked_scores


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": ["2025-10-17", "2025-10-17", "2025-10-17"],
            "instrument": ["SH600000", "SZ000001", "INVALID"],
            "0": [0.9, 0.8, 0.7],
            "confidence": [0.98, 0.5, 0.99],
        }
    )


def test_parse_ranked_scores_filters_and_maps():
    df = _make_df()
    records = parse_ranked_scores(df, confidence_floor=0.9, topk=10)
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, SignalRecord)
    assert record.instrument == "SH600000"
    assert record.symbol == "SHSE.600000"
    assert record.score == 0.9


def test_parse_ranked_scores_topk():
    df = _make_df()
    records = parse_ranked_scores(df, confidence_floor=0.0, topk=1)
    assert len(records) == 1
    assert records[0].score == 0.9


def test_parse_ranked_scores_requires_single_date():
    df = _make_df()
    df.loc[1, "datetime"] = "2025-10-20"
    with pytest.raises(ValueError):
        parse_ranked_scores(df, confidence_floor=0.0, topk=10)




