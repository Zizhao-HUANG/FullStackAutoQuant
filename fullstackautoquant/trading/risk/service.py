from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fullstackautoquant.trading.utils import gm_to_instrument, instrument_to_gm


@dataclass(frozen=True)
class RiskInputs:
    signals: Sequence[dict[str, Any]]
    logs_dir: Path
    risk_config: dict[str, Any]
    order_config: dict[str, Any]
    paths_config: dict[str, Any]
    override_buy: bool = False


@dataclass(frozen=True)
class RiskState:
    allow_buy: bool
    day_drawdown: float
    rolling5d_drawdown: float
    limit_up_symbols: list[str]
    limit_down_symbols: list[str]
    reasons: list[str]


class RiskEvaluatorService:
    def __init__(self, inputs: RiskInputs) -> None:
        self._inputs = inputs

    def evaluate(self) -> RiskState:
        day_dd, rolling_dd = self._compute_drawdowns()
        limit_up, limit_down = self._detect_limit_states()
        allow_buy = True
        reasons: list[str] = []

        risk_cfg = self._inputs.risk_config
        if day_dd >= float(risk_cfg.get("day_drawdown_limit", 0.03)):
            allow_buy = False
            reasons.append(f"day_drawdown_exceed:{day_dd:.4f}")
        if rolling_dd >= float(risk_cfg.get("rolling5d_drawdown_limit", 0.08)):
            allow_buy = False
            reasons.append(f"rolling5d_drawdown_exceed:{rolling_dd:.4f}")
        if self._inputs.override_buy:
            allow_buy = True
            reasons.append("override_buy=true")

        return RiskState(
            allow_buy=allow_buy,
            day_drawdown=day_dd,
            rolling5d_drawdown=rolling_dd,
            limit_up_symbols=sorted(set(limit_up)),
            limit_down_symbols=sorted(set(limit_down)),
            reasons=reasons,
        )

    def _compute_drawdowns(self) -> tuple[float, float]:
        nav_path = Path(self._inputs.logs_dir) / "nav_history.csv"
        if not nav_path.exists():
            return 0.0, 0.0
        try:
            df = pd.read_csv(nav_path)
        except Exception:
            return 0.0, 0.0
        if df.empty:
            return 0.0, 0.0
        nav_series = self._normalize_nav(df)
        if nav_series.empty:
            return 0.0, 0.0
        today = float(nav_series.iloc[-1])
        prev = float(nav_series.iloc[-2]) if len(nav_series) > 1 else today
        day_dd = 0.0 if prev <= 0 else max(0.0, min(1.0, 1.0 - (today / prev)))
        rolling_window = nav_series.tail(5)
        peak = float(rolling_window.max()) if len(rolling_window) else today
        rolling_dd = 0.0 if peak <= 0 else max(0.0, min(1.0, 1.0 - (today / peak)))
        return day_dd, rolling_dd

    def _normalize_nav(self, df: pd.DataFrame) -> pd.Series:
        if "equity" in df.columns:
            df["nav"] = pd.to_numeric(df["equity"], errors="coerce")
        else:
            df["nav"] = pd.to_numeric(df.get("nav"), errors="coerce")
        df = df.dropna(subset=["nav"])
        df = df[df["nav"] > 0]
        if df.empty:
            return pd.Series(dtype=float)
        for col in ("date", "trade_date", "datetime"):
            if col in df.columns:
                try:
                    df["date"] = pd.to_datetime(df[col], errors="coerce").dt.date
                except Exception:
                    df["date"] = df[col].astype(str)
                break
        if "date" not in df.columns:
            return pd.Series(dtype=float)
        grouped = (
            df.dropna(subset=["date"])
            .groupby("date", as_index=False)["nav"]
            .max()
            .sort_values("date")
        )
        return grouped["nav"]

    def _detect_limit_states(self) -> tuple[list[str], list[str]]:
        order_cfg = self._inputs.order_config
        enforce = bool(self._inputs.risk_config.get("enforce_limit_up_down_filter", True))
        if order_cfg.get("mode", "auto") == "manual":
            enforce = False
        if not enforce:
            return [], []
        daily_pv = self._inputs.paths_config.get("daily_pv")
        if not daily_pv:
            return [], []
        return detect_limit_states(
            Path(daily_pv), self._instrument_list(), float(order_cfg.get("limit_threshold", 0.095))
        )

    def _instrument_list(self) -> list[str]:
        instruments: list[str] = []
        for item in self._inputs.signals:
            gm_symbol = item.get("symbol")
            if not gm_symbol:
                continue
            ins = gm_to_instrument(str(gm_symbol))
            if ins:
                instruments.append(ins)
        return instruments


def detect_limit_states(
    h5_path: Path | str, instruments: Iterable[str], threshold: float
) -> tuple[list[str], list[str]]:
    h5_path = Path(h5_path)
    if not h5_path.exists():
        return [], []
    df = pd.read_hdf(str(h5_path), key="data")
    dates = sorted(df.index.get_level_values("datetime").unique())
    if len(dates) < 2:
        return [], []
    last_dt, prev_dt = dates[-1], dates[-2]
    df_last = df.xs(last_dt, level="datetime")
    df_prev = df.xs(prev_dt, level="datetime")
    limit_up: list[str] = []
    limit_down: list[str] = []
    for ins in instruments:
        if ins not in df_last.index or ins not in df_prev.index:
            continue
        if "$close" not in df_last.columns or "$close" not in df_prev.columns:
            continue
        last_close = float(df_last.loc[ins, "$close"])
        prev_close = float(df_prev.loc[ins, "$close"])
        if prev_close <= 0:
            continue
        change = (last_close / prev_close) - 1.0
        gm = instrument_to_gm(ins)
        if gm is None:
            continue
        if change >= threshold - 1e-6:
            limit_up.append(gm)
        if change <= -threshold + 1e-6:
            limit_down.append(gm)
    return limit_up, limit_down


__all__ = [
    "RiskInputs",
    "RiskState",
    "RiskEvaluatorService",
    "detect_limit_states",
]
