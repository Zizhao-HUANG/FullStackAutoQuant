"""Formal backtest script with progress output."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

BACKTEST_ROOT = Path(__file__).resolve().parent
TRADING_DIR = BACKTEST_ROOT.parent
PACKAGE_ROOT = TRADING_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent

for path in (str(REPO_ROOT), str(PACKAGE_ROOT), str(TRADING_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fullstackautoquant.backtest.config import BacktestConfig
from fullstackautoquant.backtest.engine import BacktestEngine

logger = logging.getLogger("backtest")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal backtest")
    parser.add_argument("--config", type=Path, required=True, help="Backtest config JSON/YAML")
    parser.add_argument("--overrides", type=str, nargs="*", help="Override params, key=value")
    parser.add_argument("--out", type=Path, help="Performance log output path")
    parser.add_argument("--log-file", type=Path, help="Run log output path")
    parser.add_argument("--verbose", action="store_true", help="Print more progress info")
    return parser.parse_args()


def _load_config(path: Path) -> BacktestConfig:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return BacktestConfig.from_dict(data)


def _parse_value(raw: str) -> object:
    for cast in (int, float):
        try:
            return cast(raw)
        except Exception:
            continue
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return raw


def _apply_overrides(cfg: BacktestConfig, overrides: list[str]) -> BacktestConfig:
    parsed: dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item}")
        key, value = item.split("=", 1)
        parsed[key] = _parse_value(value)
    return cfg.with_overrides(parsed)


def _configure_logging(args: argparse.Namespace) -> None:
    handlers: list[logging.Handler] = []
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    handlers.append(console)
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(args.log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        handlers.append(file_handler)
    logging.basicConfig(level=logging.INFO, handlers=handlers)


def _inject_progress(engine: BacktestEngine) -> None:
    """Add progress output to SignalProvider."""

    import types

    original_iterate = engine._signal_provider.iterate  # type: ignore[attr-defined]

    def iterate_with_progress(self, calendar):  # type: ignore[override]
        total = len(calendar)
        for idx, item in enumerate(original_iterate(calendar), start=1):
            trade_date = item[0]
            logger.info("PROGRESS %d/%d %s", idx, total, trade_date)
            yield item

    engine._signal_provider.iterate = types.MethodType(iterate_with_progress, engine._signal_provider)  # type: ignore[attr-defined]


def main() -> int:
    args = _parse_args()
    _configure_logging(args)
    cfg = _apply_overrides(_load_config(args.config), args.overrides or [])
    logger.info("Backtest range: %s → %s", cfg.start_date, cfg.end_date)
    logger.info("Initial capital: %.2f", cfg.initial_capital)
    logger.info("Preparing to load engine...")
    engine = BacktestEngine(cfg)
    logger.info("Engine initialized, starting run...")
    start_time = time.perf_counter()
    result = engine.run()
    duration = time.perf_counter() - start_time
    logger.info(
        "Backtest Done, total duration %.2f s, trading day count %d",
        duration,
        len(result.equity_curve),
    )
    logger.info("Total return %.2f%%", result.summary.total_return * 100)
    if args.out:
        payload = {
            "interval": {
                "start": cfg.start_date.isoformat(),
                "end": cfg.end_date.isoformat(),
            },
            "duration_seconds": duration,
            "total_return": result.summary.total_return,
            "num_days": len(result.equity_curve),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
