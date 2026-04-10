"""Ranked scores CSV collection and parsing.

Scans known artifact directories for ranked_scores_*.csv files,
deduplicates by content date, and produces structured DailyScores objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.dashboard_export.constants import REPO_ROOT, log


@dataclass
class DailyScores:
    """Parsed daily inference scores for one trading day."""

    date: str
    instruments: list[dict[str, Any]]
    total_count: int
    positive_count: int
    negative_count: int
    mean_score: float
    median_score: float
    std_score: float
    min_score: float
    max_score: float
    mean_confidence: float
    min_confidence: float
    max_confidence: float
    score_quantiles: dict[str, float]


def _collect_score_csvs() -> list[Path]:
    """Find all ranked_scores CSV files across known locations."""
    locations = [
        REPO_ROOT / "output" / "history",
        REPO_ROOT / "fullstackautoquant" / "model" / "ranked_scores_archive",
    ]
    csvs: list[Path] = []
    for loc in locations:
        if loc.is_dir():
            csvs.extend(sorted(loc.glob("ranked_scores_*.csv")))
    # Deduplicate by filename date (prefer output/history over archive)
    seen_dates: set[str] = set()
    unique: list[Path] = []
    for csv_path in csvs:
        stem = csv_path.stem
        date_part = stem.replace("ranked_scores_", "")
        if date_part not in seen_dates:
            seen_dates.add(date_part)
            unique.append(csv_path)
    return sorted(unique, key=lambda p: p.stem)


def _parse_scores_csv(csv_path: Path) -> DailyScores | None:
    """Parse a single ranked_scores CSV into a DailyScores object."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        log("WARN", f"Cannot read {csv_path}: {exc}")
        return None

    if df.empty:
        return None

    # Column names: datetime, instrument, 0 (score), confidence
    score_col = "0" if "0" in df.columns else df.columns[2]
    conf_col = "confidence" if "confidence" in df.columns else None

    date_str = str(df["datetime"].iloc[0])
    scores = df[score_col].astype(float)
    instruments_data: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        entry: dict[str, Any] = {
            "rank": len(instruments_data) + 1,
            "instrument": str(row["instrument"]),
            "score": round(float(row[score_col]), 8),
        }
        if conf_col and conf_col in df.columns:
            entry["confidence"] = round(float(row[conf_col]), 8)
        instruments_data.append(entry)

    confidences = df[conf_col].astype(float) if conf_col else pd.Series(dtype=float)

    quantiles = scores.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
    score_quantiles = {f"p{int(k*100)}": round(float(v), 8) for k, v in quantiles.items()}

    return DailyScores(
        date=date_str,
        instruments=instruments_data,
        total_count=len(df),
        positive_count=int((scores > 0).sum()),
        negative_count=int((scores < 0).sum()),
        mean_score=round(float(scores.mean()), 8),
        median_score=round(float(scores.median()), 8),
        std_score=round(float(scores.std()), 8),
        min_score=round(float(scores.min()), 8),
        max_score=round(float(scores.max()), 8),
        mean_confidence=round(float(confidences.mean()), 8) if not confidences.empty else 0.0,
        min_confidence=round(float(confidences.min()), 8) if not confidences.empty else 0.0,
        max_confidence=round(float(confidences.max()), 8) if not confidences.empty else 0.0,
        score_quantiles=score_quantiles,
    )


def collect_and_parse_scores() -> list[DailyScores]:
    """Collect all CSVs, parse, deduplicate by content date, return sorted list.

    Returns:
        Sorted list of DailyScores (ascending by date), empty if no data found.
    """
    csv_files = _collect_score_csvs()
    if not csv_files:
        log("ERROR", "No ranked_scores CSV files found. Run the inference pipeline first.")
        return []

    log("1/7", f"Found {len(csv_files)} CSV files")
    all_scores: list[DailyScores] = []
    seen_content_dates: set[str] = set()
    for csv_path in csv_files:
        parsed = _parse_scores_csv(csv_path)
        if parsed and parsed.date not in seen_content_dates:
            seen_content_dates.add(parsed.date)
            all_scores.append(parsed)
    # Sort by content date (not filename)
    all_scores.sort(key=lambda s: s.date)
    log("1/7", f"Parsed {len(all_scores)} unique trading days")
    return all_scores
