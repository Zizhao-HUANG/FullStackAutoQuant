"""Strategy rebalancing execution module."""

from __future__ import annotations

from typing import Dict, List, Tuple

from fullstackautoquant.trading.strategy import (
    build_targets,
    dynamic_adjust_with_positions,
)


class StrategyRunner:
    """Encapsulates target construction and order generation."""

    def __init__(self, config: dict) -> None:
        self._config = config

    def generate_orders(
        self,
        signals: List[dict],
        risk_state: Dict[str, object],
        positions: Dict[str, float],
        total_capital: float,
    ) -> Tuple[List[dict], Dict[str, float], Dict[str, dict]]:
        allow_buy = bool(risk_state.get("allow_buy", True))
        limit_up = list(risk_state.get("limit_up_symbols", []))
        limit_down = list(risk_state.get("limit_down_symbols", []))

        targets, _, ref_prices, quote_details, _, _ = build_targets(
            signals,
            self._config,
            allow_buy,
            limit_up,
            limit_down,
            total_capital,
        )

        orders = dynamic_adjust_with_positions(
            targets,
            {sym: int(qty) for sym, qty in positions.items()},
            self._config,
            ref_prices,
            quote_details,
            self._config["order"].get("price_source", "qlib_close"),
            allow_buy,
            limit_up,
            limit_down,
        )
        return orders, ref_prices, quote_details
