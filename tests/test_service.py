from __future__ import annotations

from pathlib import Path

import pandas as pd
from fullstackautoquant.trading.risk.service import RiskEvaluatorService, RiskInputs


def test_risk_service_returns_state(tmp_path: Path) -> None:
    nav_path = tmp_path / "nav_history.csv"
    nav_df = pd.DataFrame({"date": ["2025-10-16", "2025-10-17"], "nav": [1.0, 0.95]})
    nav_df.to_csv(nav_path, index=False)
    inputs = RiskInputs(
        signals=[],
        logs_dir=tmp_path,
        risk_config={"day_drawdown_limit": 0.02},
        order_config={"limit_threshold": 0.095},
        paths_config={},
        override_buy=False,
    )
    service = RiskEvaluatorService(inputs)
    state = service.evaluate()
    assert not state.allow_buy
    assert any("day_drawdown_exceed" in reason for reason in state.reasons)




