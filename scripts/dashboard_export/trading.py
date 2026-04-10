"""Trading log artifacts extraction.

Reads signals, targets, orders, receipts, and risk state JSON files
from the logs/ directory and assembles sanitized trading data for export.
All sensitive fields (account IDs, capital amounts) are stripped.
"""

from __future__ import annotations

import json
from typing import Any

from scripts.dashboard_export.constants import REPO_ROOT


def extract_trading_logs() -> dict[str, Any]:
    """Extract all trading log artifacts (signals, targets, orders, receipts)."""
    logs_dir = REPO_ROOT / "logs"
    result: dict[str, Any] = {
        "latest_signals": None,
        "latest_targets": None,
        "latest_orders": None,
        "execution_history": [],
    }

    if not logs_dir.is_dir():
        return result

    _extract_latest_signals(logs_dir, result)
    _extract_latest_targets(logs_dir, result)
    _extract_latest_orders(logs_dir, result)
    _extract_execution_history(logs_dir, result)
    _extract_risk_state(logs_dir, result)

    return result


def _extract_latest_signals(logs_dir, result: dict[str, Any]) -> None:
    """Load the most recent signals JSON."""
    signals_files = sorted(logs_dir.glob("signals_*.json"))
    if signals_files:
        try:
            with open(signals_files[-1], encoding="utf-8") as f:
                result["latest_signals"] = json.load(f)
        except Exception:
            pass


def _extract_latest_targets(logs_dir, result: dict[str, Any]) -> None:
    """Load and sanitize the most recent targets JSON."""
    targets_files = sorted(logs_dir.glob("targets_*.json"))
    if not targets_files:
        return
    try:
        with open(targets_files[-1], encoding="utf-8") as f:
            data = json.load(f)
        targets = data.get("targets", [])
        sanitized = [
            {
                "instrument": t.get("instrument"),
                "weight": round(float(t.get("weight", 0)), 6),
                "target_shares": t.get("target_shares"),
                "ref_price": round(float(t.get("ref_price", 0)), 2),
            }
            for t in targets
        ]
        result["latest_targets"] = {
            "date": data.get("date"),
            "target_count": len(sanitized),
            "targets": sanitized,
        }
    except Exception:
        pass


def _extract_latest_orders(logs_dir, result: dict[str, Any]) -> None:
    """Load and sanitize the most recent orders JSON."""
    orders_files = sorted(logs_dir.glob("orders_*.json"))
    if not orders_files:
        return
    try:
        with open(orders_files[-1], encoding="utf-8") as f:
            data = json.load(f)
        orders = data.get("orders", [])
        sanitized = [
            {
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "volume": o.get("volume"),
                "price": round(float(o.get("price", 0)), 2),
                "type": o.get("type"),
            }
            for o in orders
        ]
        result["latest_orders"] = {
            "date": data.get("date"),
            "order_count": len(sanitized),
            "buy_count": sum(1 for o in sanitized if o["side"] == "BUY"),
            "sell_count": sum(1 for o in sanitized if o["side"] == "SELL"),
            "orders": sanitized,
        }
    except Exception:
        pass


def _extract_execution_history(logs_dir, result: dict[str, Any]) -> None:
    """Load all execution receipts, stripping account IDs."""
    receipt_files = sorted(logs_dir.glob("receipts_*.json"))
    for rf in receipt_files:
        try:
            with open(rf, encoding="utf-8") as f:
                data = json.load(f)
            receipts = data.get("receipts", [])
            result["execution_history"].append({
                "date": data.get("date"),
                "order_count": len(receipts),
                "placed": data.get("placed", False),
                "orders": [
                    {
                        "symbol": r.get("symbol"),
                        "side": r.get("side"),
                        "volume": r.get("volume"),
                        "price": round(float(r.get("price", 0)), 2),
                        "status": r.get("status"),
                    }
                    for r in receipts
                ],
            })
        except Exception:
            pass


def _extract_risk_state(logs_dir, result: dict[str, Any]) -> None:
    """Load the latest risk state JSON."""
    risk_file = logs_dir / "risk_state_AUTO.json"
    if risk_file.exists():
        try:
            with open(risk_file, encoding="utf-8") as f:
                result["latest_risk_state"] = json.load(f)
        except Exception:
            pass
