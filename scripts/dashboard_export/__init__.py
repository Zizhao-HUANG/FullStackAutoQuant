"""Dashboard data export package.

Modular pipeline for exporting FullStackAutoQuant inference/trading
artifacts into structured JSON for the Live Dashboard frontend.

Submodules:
    constants   — Shared constants and project path setup
    scores      — CSV collection and parsing (DailyScores)
    performance — Performance metrics and equity curve computation
    confidence  — MC Dropout confidence distribution analysis
    metadata    — System info, model architecture, trading config extraction
    trading     — Trading log artifacts (signals, targets, orders, receipts)
    health      — System health and operational status
    portfolio   — Real portfolio snapshots from GM Trade API
    assembler   — Master JSON assembly and per-subdirectory file generation
    writer      — File I/O helpers
"""

from scripts.dashboard_export.assembler import build_master_json, build_subdirectory_files
from scripts.dashboard_export.confidence import build_confidence_distribution
from scripts.dashboard_export.constants import REPO_ROOT, log
from scripts.dashboard_export.health import build_system_health
from scripts.dashboard_export.metadata import extract_system_info, extract_trading_config
from scripts.dashboard_export.performance import build_equity_curve, compute_performance
from scripts.dashboard_export.portfolio import (
    build_real_equity_curve,
    has_real_portfolio_data,
    snapshot_positions,
)
from scripts.dashboard_export.scores import DailyScores, collect_and_parse_scores
from scripts.dashboard_export.trading import extract_trading_logs
from scripts.dashboard_export.writer import write_json

__all__ = [
    "REPO_ROOT",
    "log",
    "DailyScores",
    "collect_and_parse_scores",
    "compute_performance",
    "build_equity_curve",
    "build_confidence_distribution",
    "extract_system_info",
    "extract_trading_config",
    "extract_trading_logs",
    "build_system_health",
    "build_master_json",
    "build_subdirectory_files",
    "write_json",
    "snapshot_positions",
    "build_real_equity_curve",
    "has_real_portfolio_data",
]

