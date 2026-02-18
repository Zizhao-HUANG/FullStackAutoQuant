from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fullstackautoquant.logging_config import get_logger
from fullstackautoquant.trading.risk.service import (
    RiskEvaluatorService,
    RiskInputs,
    detect_limit_states,
)
from fullstackautoquant.trading.utils import (
    ensure_logs_dir,
    gm_to_instrument,
    load_config,
    save_json,
)

logger = get_logger(__name__)


def compute_drawdowns(logs_dir: str) -> tuple[float, float]:
    nav_path = Path(logs_dir) / "nav_history.csv"
    if not nav_path.exists():
        return 0.0, 0.0
    try:
        df = pd.read_csv(nav_path)
    except Exception:
        return 0.0, 0.0
    if df.empty:
        return 0.0, 0.0
    if "equity" in df.columns:
        df["nav"] = pd.to_numeric(df["equity"], errors="coerce")
    else:
        df["nav"] = pd.to_numeric(df.get("nav"), errors="coerce")
    df = df.dropna(subset=["nav"])
    df = df[df["nav"] > 0]
    if df.empty:
        return 0.0, 0.0
    date_col = next((col for col in ("date", "trade_date", "datetime") if col in df.columns), None)
    if date_col is None:
        return 0.0, 0.0
    try:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    except Exception:
        df["date"] = df[date_col].astype(str)
    df = df.dropna(subset=["date"])
    if df.empty:
        return 0.0, 0.0
    grouped = df.groupby("date", as_index=False)["nav"].max().sort_values("date")
    if len(grouped) < 2:
        return 0.0, 0.0
    today_nav = float(grouped.iloc[-1]["nav"])
    prev_nav = float(grouped.iloc[-2]["nav"])
    day_dd = 0.0 if prev_nav <= 0 else max(0.0, min(1.0, 1.0 - (today_nav / prev_nav)))
    window = grouped.tail(5)["nav"].astype(float)
    peak = float(window.max()) if len(window) > 0 else today_nav
    rolling_dd = 0.0 if peak <= 0 else max(0.0, min(1.0, 1.0 - (today_nav / peak)))
    return day_dd, rolling_dd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Risk manager service")
    parser.add_argument("--signals", required=True, help="Signal JSON path")
    parser.add_argument("--config", help="Trading config YAML")
    parser.add_argument("--out", help="Output risk_state.json")
    parser.add_argument(
        "--override_buy", action="store_true", help="Force allow buying for this run"
    )
    return parser.parse_args()


def _load_signals(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    logs_dir = Path(ensure_logs_dir(cfg))
    signals_payload = _load_signals(Path(args.signals))
    signals = signals_payload.get("signals", [])
    inputs = RiskInputs(
        signals=signals,
        logs_dir=logs_dir,
        risk_config=cfg.get("risk", {}),
        order_config=cfg.get("order", {}),
        paths_config=cfg.get("paths", {}),
        override_buy=args.override_buy,
    )
    service = RiskEvaluatorService(inputs)
    state = service.evaluate()
    # Keep limit_up/down for backward compatibility with older field names
    if not state.limit_up_symbols:
        _, state_limit_down = detect_limit_states(
            Path(cfg.get("paths", {}).get("daily_pv", "")),
            [
                inst
                for sig in signals
                if sig.get("symbol")
                for inst in [gm_to_instrument(sig.get("symbol", ""))]
                if inst is not None
            ],
            float(cfg.get("order", {}).get("limit_threshold", 0.095)),
        )
    out_path = Path(args.out) if args.out else logs_dir / "risk_state.json"
    save_json(
        {
            "allow_buy": state.allow_buy,
            "day_drawdown": state.day_drawdown,
            "rolling5d_drawdown": state.rolling5d_drawdown,
            "limit_up_symbols": state.limit_up_symbols,
            "limit_down_symbols": state.limit_down_symbols,
            "reasons": state.reasons,
        },
        str(out_path),
    )
    logger.info(
        "Risk evaluation complete: allow_buy=%s, out=%s",
        state.allow_buy,
        out_path,
    )


if __name__ == "__main__":
    main()
