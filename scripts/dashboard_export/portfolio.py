"""Real portfolio equity curve from GM Trade API position snapshots.

Workflow:
    1. ``snapshot_positions()`` — called during the daily pipeline to fetch
       current positions from GM (via ``get_positions``), persist a daily
       snapshot to ``output/history/portfolio_snapshots.json``.
    2. ``build_real_equity_curve()`` — reads all snapshots and computes a
       proper equity curve using actual market values and cost basis.

The GM ``get_positions()`` response contains per-position fields:
    symbol, volume, vwap (cost price), cost, market_value, fpnl, price

We track total_market_value + available_cash across days and compute
daily returns as:  r_t = (NAV_t / NAV_{t-1}) - 1
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from scripts.dashboard_export.constants import REPO_ROOT, log

SNAPSHOT_FILE = REPO_ROOT / "output" / "history" / "portfolio_snapshots.json"


def _load_snapshots() -> list[dict[str, Any]]:
    """Load existing snapshots from disk."""
    if SNAPSHOT_FILE.exists():
        try:
            with open(SNAPSHOT_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def _save_snapshots(snapshots: list[dict[str, Any]]) -> None:
    """Persist snapshots to disk."""
    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshots, f, ensure_ascii=False, indent=2)


def snapshot_positions(
    account_id: str | None = None,
    config_path: str | None = None,
    date_override: str | None = None,
) -> dict[str, Any] | None:
    """Fetch current positions from GM Trade API and save a daily snapshot.

    This should be called once per trading day, ideally after the market
    close (15:00+ Beijing time) so that ``market_value`` reflects the
    closing price.

    Returns the snapshot dict on success, None on failure.
    """
    # Guard: don't crash the pipeline if GM SDK is unavailable
    try:
        _gm_available = True
        # Set protobuf implementation before any GM imports
        if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" not in os.environ:
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

        from fullstackautoquant.trading.execution import gm_login
        from fullstackautoquant.trading.utils import load_config
    except ImportError as exc:
        log("WARN", f"GM SDK not available, cannot snapshot positions: {exc}")
        return None

    try:
        from gmtrade.api import get_cash, get_positions
    except ImportError:
        try:
            from gm.api import get_cash, get_positions
        except ImportError as exc:
            log("WARN", f"No GM SDK found: {exc}")
            return None

    # Load config
    cfg_path = config_path or str(REPO_ROOT / "configs" / "trading.yaml")
    try:
        cfg = load_config(cfg_path)
    except Exception as exc:
        log("WARN", f"Cannot load trading config: {exc}")
        return None

    # Login
    try:
        gm_login(cfg, account_id_override=account_id)
    except Exception as exc:
        log("WARN", f"GM login failed: {exc}")
        return None

    # Fetch positions
    try:
        raw_positions = get_positions()
    except Exception as exc:
        log("WARN", f"get_positions() failed: {exc}")
        return None

    # Parse positions
    positions: list[dict[str, Any]] = []
    total_market_value = 0.0
    total_cost = 0.0
    total_fpnl = 0.0

    for p in raw_positions or []:
        entry: dict[str, Any] = {}
        for field in ["symbol", "volume", "vwap", "cost", "market_value", "fpnl", "price",
                       "side", "available", "amount", "qty", "current_qty"]:
            val = None
            if hasattr(p, field):
                val = getattr(p, field, None)
            elif isinstance(p, dict) and field in p:
                val = p[field]
            if val is not None:
                entry[field] = val

        # Normalise symbol
        sym = entry.get("symbol", "")
        if not sym:
            continue

        # Get volume (try multiple field names)
        vol = 0
        for vf in ["volume", "qty", "current_qty", "amount"]:
            v = entry.get(vf)
            if v is not None and float(v) > 0:
                vol = int(float(v))
                break
        if vol <= 0:
            continue

        # Get market value
        mv = float(entry.get("market_value", 0) or 0)
        if mv <= 0:
            # Fallback: compute from price × volume
            px = float(entry.get("price", 0) or 0)
            if px > 0:
                mv = px * vol

        # Get cost
        cost_val = float(entry.get("cost", 0) or 0)
        if cost_val <= 0:
            vwap = float(entry.get("vwap", 0) or 0)
            if vwap > 0:
                cost_val = vwap * vol

        fpnl_val = float(entry.get("fpnl", 0) or 0)

        positions.append({
            "symbol": str(sym),
            "volume": vol,
            "vwap": round(float(entry.get("vwap", 0) or 0), 4),
            "cost": round(cost_val, 2),
            "market_value": round(mv, 2),
            "fpnl": round(fpnl_val, 2),
            "price": round(float(entry.get("price", 0) or 0), 4),
        })
        total_market_value += mv
        total_cost += cost_val
        total_fpnl += fpnl_val

    # Fetch available cash
    available_cash = 0.0
    try:
        cash_obj = get_cash()
        if hasattr(cash_obj, "available"):
            available_cash = float(cash_obj.available)
        elif isinstance(cash_obj, dict):
            available_cash = float(cash_obj.get("available", 0))
        # Also get nav if available
    except Exception:
        pass

    today = date_override or time.strftime("%Y-%m-%d")
    nav = total_market_value + available_cash

    snapshot: dict[str, Any] = {
        "date": today,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nav": round(nav, 2),
        "total_market_value": round(total_market_value, 2),
        "total_cost": round(total_cost, 2),
        "total_fpnl": round(total_fpnl, 2),
        "available_cash": round(available_cash, 2),
        "position_count": len(positions),
        "positions": positions,
    }

    # Merge into existing snapshots (replace same date, append otherwise)
    snapshots = _load_snapshots()
    existing_dates = {s["date"] for s in snapshots}
    if today in existing_dates:
        snapshots = [s for s in snapshots if s["date"] != today]
    snapshots.append(snapshot)
    snapshots.sort(key=lambda s: s["date"])
    _save_snapshots(snapshots)

    log("OK", f"Portfolio snapshot saved: date={today}, NAV={nav:,.2f}, "
        f"positions={len(positions)}, cash={available_cash:,.2f}")
    return snapshot


def build_real_equity_curve(
    initial_capital: float | None = None,
) -> list[dict[str, Any]]:
    """Build an equity curve from saved portfolio snapshots.

    Uses daily NAV (total_market_value + available_cash) to compute
    true daily returns and cumulative performance.

    Args:
        initial_capital: If provided, used as the baseline NAV for day-1
            return calculation. Otherwise the first snapshot's NAV is
            used as the starting point (day-1 return = 0).

    Returns:
        List of equity curve points compatible with the dashboard format:
        [{date, equity, daily_return, cumulative_return, drawdown}, ...]
        Empty list if no snapshots are available.
    """
    snapshots = _load_snapshots()
    if not snapshots:
        return []

    curve: list[dict[str, Any]] = []
    prev_nav: float | None = initial_capital
    peak = 0.0

    for snap in snapshots:
        nav = float(snap.get("nav", 0))
        if nav <= 0:
            continue

        if prev_nav is not None and prev_nav > 0:
            daily_return = (nav / prev_nav) - 1.0
        else:
            daily_return = 0.0

        base_nav = initial_capital if initial_capital and initial_capital > 0 else snapshots[0].get("nav", nav)
        if base_nav <= 0:
            base_nav = nav
        equity = nav / base_nav
        cumulative_return = equity - 1.0

        peak = max(peak, equity)
        dd = (equity / peak) - 1.0 if peak > 0 else 0.0

        curve.append({
            "date": snap["date"],
            "equity": round(equity, 6),
            "daily_return": round(daily_return, 8),
            "cumulative_return": round(cumulative_return, 6),
            "drawdown": round(dd, 6),
            "nav": round(nav, 2),
            "total_market_value": round(float(snap.get("total_market_value", 0)), 2),
            "available_cash": round(float(snap.get("available_cash", 0)), 2),
            "position_count": snap.get("position_count", 0),
        })

        prev_nav = nav

    return curve


def compute_real_performance(
    initial_capital: float | None = None,
) -> dict[str, Any] | None:
    """Compute performance metrics from real portfolio snapshots.

    Returns a dict with keys matching PerformanceSummary fields,
    or None if no snapshot data is available.
    """
    import numpy as np

    from scripts.dashboard_export.constants import ANNUALIZATION_DAYS, RISK_FREE_ANNUAL

    curve = build_real_equity_curve(initial_capital)
    if len(curve) < 2:
        return None

    daily_returns = np.array([pt["daily_return"] for pt in curve])
    # Skip the first point if daily_return is 0 (no prior reference)
    if daily_returns[0] == 0.0 and len(daily_returns) > 2:
        daily_returns = daily_returns[1:]

    n_days = len(daily_returns)
    rf_daily = RISK_FREE_ANNUAL / ANNUALIZATION_DAYS

    mean_ret = float(np.mean(daily_returns))
    std_ret = float(np.std(daily_returns, ddof=1)) if n_days > 1 else 1e-9

    # Sharpe
    sharpe = (mean_ret - rf_daily) * np.sqrt(ANNUALIZATION_DAYS) / max(std_ret, 1e-9)

    # Win rate
    win_rate = float(np.mean(daily_returns > 0))

    # Cumulative return
    cumulative_return = curve[-1]["cumulative_return"]

    # Max drawdown
    equities = np.array([pt["equity"] for pt in curve])
    peaks = np.maximum.accumulate(equities)
    drawdowns = equities / peaks - 1.0
    max_dd = float(np.min(drawdowns))

    # Calmar
    ann_ret = cumulative_return * (ANNUALIZATION_DAYS / max(n_days, 1))
    calmar = ann_ret / abs(max_dd) if max_dd < -1e-9 else float("inf")

    return {
        "source": "real_portfolio",
        "total_trading_days": len(curve),
        "first_date": curve[0]["date"],
        "last_date": curve[-1]["date"],
        "cumulative_return": round(cumulative_return, 6),
        "annualized_return": round(ann_ret, 6),
        "sharpe_ratio": round(float(sharpe), 4),
        "win_rate": round(win_rate, 4),
        "max_drawdown": round(max_dd, 6),
        "calmar_ratio": round(float(calmar), 4) if np.isfinite(calmar) else None,
        "daily_return_mean": round(mean_ret, 8),
        "daily_return_std": round(std_ret, 8),
        "latest_daily_return": round(float(daily_returns[-1]), 8),
        "latest_nav": curve[-1].get("nav", 0),
    }


def has_real_portfolio_data() -> bool:
    """Check if we have any real portfolio snapshots available."""
    snapshots = _load_snapshots()
    return len(snapshots) >= 1
