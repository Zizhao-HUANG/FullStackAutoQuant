from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fullstackautoquant.logging_config import get_logger
from fullstackautoquant.trading.signals.parser import SignalRecord, parse_ranked_scores
from fullstackautoquant.trading.utils import ensure_logs_dir, load_config, save_json

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse ranked_scores CSV into signal JSON")
    parser.add_argument("--csv", required=True, help="ranked_scores CSV path")
    parser.add_argument("--config", help="Trading config YAML")
    parser.add_argument("--out", help="Output JSON path")
    return parser.parse_args()


def _load_ranked_scores(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _config_parameters(cfg: dict) -> tuple[float, int]:
    portfolio = cfg.get("portfolio", {})
    confidence_floor = float(portfolio.get("confidence_floor", 0.0))
    topk = int(portfolio.get("topk", 0))
    return confidence_floor, topk


def _to_payload(date: str, records: list[SignalRecord]) -> dict:
    return {
        "date": date,
        "count": len(records),
        "signals": [record.__dict__ for record in records],
    }


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    confidence_floor, topk = _config_parameters(cfg)

    df = _load_ranked_scores(Path(args.csv))
    records = parse_ranked_scores(df, confidence_floor, topk)
    date = records[0].date if records else str(df["datetime"].iloc[0])

    payload = _to_payload(date, records)
    logs_dir = ensure_logs_dir(cfg)
    out_path = Path(args.out) if args.out else Path(logs_dir) / f"signals_{date}.json"
    save_json(payload, str(out_path))
    logger.info(
        "Parsed %d signals from %s, output: %s",
        len(records),
        args.csv,
        out_path,
    )


if __name__ == "__main__":
    main()
