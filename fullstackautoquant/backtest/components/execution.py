"""Execution and bookkeeping logic."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from .records import DailyEquity, TradeRecord


class ExecutionEngine:
    """Update cash and positions based on generated orders, record fills."""

    def __init__(self, costs) -> None:
        self._costs = costs

    def run(
        self,
        trade_date: dt.date,
        orders: Sequence[dict],
        positions: dict[str, float],
        cash: float,
        prev_equity: float,
        market_value_func,
    ) -> tuple[float, dict[str, float], list[TradeRecord], DailyEquity, float]:
        trades: list[TradeRecord] = []
        cash_after = cash
        sell_orders = [
            od for od in orders if self._is_valid_order(od) and str(od["side"]).upper() == "SELL"
        ]
        buy_orders = [
            od for od in orders if self._is_valid_order(od) and str(od["side"]).upper() == "BUY"
        ]

        for order in sell_orders:
            cash_after, positions = self._execute_sell(
                trade_date, order, cash_after, positions, trades
            )
        for order in buy_orders:
            cash_after, positions = self._execute_buy(
                trade_date, order, cash_after, positions, trades
            )

        portfolio_value = market_value_func(trade_date, positions)
        equity_value = cash_after + portfolio_value
        prev_equity_val = prev_equity if prev_equity > 0 else cash
        daily_return = 0.0 if prev_equity_val <= 0 else (equity_value / prev_equity_val) - 1.0
        equity_record = DailyEquity(
            date=trade_date,
            cash=cash_after,
            market_value=portfolio_value,
            equity=equity_value,
            daily_return=daily_return,
        )
        return cash_after, positions, trades, equity_record, portfolio_value

    def _is_valid_order(self, order: dict) -> bool:
        return int(order.get("volume", 0)) > 0 and float(order.get("price", 0.0)) > 0

    def _execute_sell(
        self,
        trade_date: dt.date,
        order: dict,
        cash: float,
        positions: dict[str, float],
        trades: list[TradeRecord],
    ) -> tuple[float, dict[str, float]]:
        if str(order["side"]).upper() != "SELL":
            return cash, positions
        symbol = order["symbol"]
        price = float(order["price"])
        volume = int(order["volume"])
        proceeds = price * volume
        commission = max(self._costs.min_commission, proceeds * self._costs.commission)
        tax = proceeds * self._costs.stamp_tax
        cash += proceeds - commission - tax
        positions[symbol] = max(0.0, positions.get(symbol, 0.0) - volume)
        trades.append(
            TradeRecord(
                date=trade_date,
                symbol=symbol,
                side="SELL",
                volume=volume,
                price=price,
                fee=commission + tax,
            )
        )
        return cash, positions

    def _execute_buy(
        self,
        trade_date: dt.date,
        order: dict,
        cash: float,
        positions: dict[str, float],
        trades: list[TradeRecord],
    ) -> tuple[float, dict[str, float]]:
        if str(order["side"]).upper() != "BUY":
            return cash, positions
        symbol = order["symbol"]
        price = float(order["price"])
        volume = int(order["volume"])
        cost = price * volume
        commission = max(self._costs.min_commission, cost * self._costs.commission)
        total_cost = cost + commission
        if cash + 1e-6 < total_cost:
            return cash, positions
        cash -= total_cost
        positions[symbol] = positions.get(symbol, 0.0) + volume
        trades.append(
            TradeRecord(
                date=trade_date,
                symbol=symbol,
                side="BUY",
                volume=volume,
                price=price,
                fee=commission,
            )
        )
        return cash, positions
