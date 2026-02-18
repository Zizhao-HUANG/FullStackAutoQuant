"""Backtest configuration and parameter grid utilities.

Goals:
- Structure backtest params into groups for easy manual tuning;
- Provide override function supporting "dot-notation" quick override;
- Provide parameter grid expansion for brute-force search.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _to_date(value: Any) -> dt.date:
    if isinstance(value, dt.date):
        return value
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("Date field cannot be empty")
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    raise TypeError(f"Cannot parse date field type: {type(value)!r}")


def _as_path(value: Any | None) -> Path | None:
    if value is None or value == "":
        return None
    if isinstance(value, Path):
        return value
    return Path(str(value)).expanduser().resolve()


@dataclass(slots=True)
class DataParams:
    daily_pv: Path | None = None
    qlib_root: Path | None = None
    calendar_csv: Path | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DataParams:
        return cls(
            daily_pv=_as_path(data.get("daily_pv")),
            qlib_root=_as_path(data.get("qlib_root")),
            calendar_csv=_as_path(data.get("calendar_csv")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "daily_pv": str(self.daily_pv) if self.daily_pv else None,
            "qlib_root": str(self.qlib_root) if self.qlib_root else None,
            "calendar_csv": str(self.calendar_csv) if self.calendar_csv else None,
        }


@dataclass(slots=True)
class PortfolioParams:
    topk: int = 20
    invest_ratio: float = 0.95
    max_weight: float = 0.05
    lot: int = 100
    confidence_floor: float = 0.9
    weight_mode: str = "score"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PortfolioParams:
        return cls(
            topk=int(data.get("topk", 20)),
            invest_ratio=float(data.get("invest_ratio", 0.95)),
            max_weight=float(data.get("max_weight", 0.05)),
            lot=int(data.get("lot", 100)),
            confidence_floor=float(data.get("confidence_floor", 0.9)),
            weight_mode=str(data.get("weight_mode", "score")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "topk": self.topk,
            "invest_ratio": self.invest_ratio,
            "max_weight": self.max_weight,
            "lot": self.lot,
            "confidence_floor": self.confidence_floor,
            "weight_mode": self.weight_mode,
        }


@dataclass(slots=True)
class CostParams:
    commission: float = 0.001
    stamp_tax: float = 0.001
    slippage_bps: float = 5.0
    borrow_cost: float = 0.0
    min_commission: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CostParams:
        return cls(
            commission=float(data.get("commission", 0.001)),
            stamp_tax=float(data.get("stamp_tax", 0.001)),
            slippage_bps=float(data.get("slippage_bps", 5.0)),
            borrow_cost=float(data.get("borrow_cost", 0.0)),
            min_commission=float(data.get("min_commission", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "commission": self.commission,
            "stamp_tax": self.stamp_tax,
            "slippage_bps": self.slippage_bps,
            "borrow_cost": self.borrow_cost,
            "min_commission": self.min_commission,
        }


@dataclass(slots=True)
class SignalParams:
    combined_factors: Path
    params_path: Path
    provider_uri: Path
    region: str = "cn"
    qlib_start: str = "2005-01-04"
    instruments: str = "csi300"
    fallback_days: int = 0
    cache_dir: Path | None = None
    archive_dir: Path | None = None
    latest_csv: Path | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SignalParams:
        factors = _as_path(data.get("combined_factors"))
        params = _as_path(data.get("params_path"))
        provider = _as_path(data.get("provider_uri")) or Path.home() / ".qlib/qlib_data/cn_data"
        if factors is None or params is None:
            raise ValueError("SignalParams requires combined_factors and params_path")
        return cls(
            combined_factors=factors,
            params_path=params,
            provider_uri=provider,
            region=str(data.get("region", "cn")),
            qlib_start=str(data.get("qlib_start", "2005-01-04")),
            instruments=str(data.get("instruments", "csi300")),
            fallback_days=int(data.get("fallback_days", 0)),
            cache_dir=_as_path(data.get("cache_dir")),
            archive_dir=_as_path(data.get("archive_dir")),
            latest_csv=_as_path(data.get("latest_csv")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "combined_factors": str(self.combined_factors),
            "params_path": str(self.params_path),
            "provider_uri": str(self.provider_uri),
            "region": self.region,
            "qlib_start": self.qlib_start,
            "instruments": self.instruments,
            "fallback_days": self.fallback_days,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "archive_dir": str(self.archive_dir) if self.archive_dir else None,
            "latest_csv": str(self.latest_csv) if self.latest_csv else None,
        }


@dataclass(slots=True)
class ManualWorkflowParams:
    enabled: bool = False
    confirm_delay_days: int = 0
    max_signals_per_day: int | None = None
    min_confidence: float | None = None
    record_rejected: bool = True
    strategy_field: str | None = None
    strategy_limits: dict[str, int] = field(default_factory=dict)
    partial_fill_ratio: float = 1.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ManualWorkflowParams:
        if not isinstance(data, Mapping):
            return cls()
        strategy_field = data.get("strategy_field")
        strategy_field = str(strategy_field).strip() if strategy_field else None
        limits_raw = (
            data.get("strategy_limits") if isinstance(data.get("strategy_limits"), Mapping) else {}
        )
        strategy_limits: dict[str, int] = {}
        if isinstance(limits_raw, Mapping):
            for key, value in limits_raw.items():
                try:
                    strategy_limits[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue
        partial_ratio_raw = data.get("partial_fill_ratio", data.get("default_fill_ratio", 1.0))
        try:
            partial_ratio = float(partial_ratio_raw)
        except (TypeError, ValueError):
            partial_ratio = 1.0
        if partial_ratio < 0.0:
            partial_ratio = 0.0
        if partial_ratio > 1.0:
            partial_ratio = 1.0
        return cls(
            enabled=bool(data.get("enabled", False)),
            confirm_delay_days=max(0, int(data.get("confirm_delay_days", 0) or 0)),
            max_signals_per_day=(
                int(data["max_signals_per_day"])
                if data.get("max_signals_per_day") not in (None, "", "None")
                else None
            ),
            min_confidence=(
                float(data["min_confidence"])
                if data.get("min_confidence") not in (None, "", "None")
                else None
            ),
            record_rejected=bool(data.get("record_rejected", True)),
            strategy_field=strategy_field if strategy_field else None,
            strategy_limits=strategy_limits,
            partial_fill_ratio=partial_ratio,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "confirm_delay_days": self.confirm_delay_days,
            "max_signals_per_day": self.max_signals_per_day,
            "min_confidence": self.min_confidence,
            "record_rejected": self.record_rejected,
            "strategy_field": self.strategy_field,
            "strategy_limits": dict(self.strategy_limits),
            "partial_fill_ratio": self.partial_fill_ratio,
        }


@dataclass(slots=True)
class BacktestConfig:
    start_date: dt.date
    end_date: dt.date
    initial_capital: float = 1_000_000.0
    rebalance_frequency: str = "daily"
    allow_short: bool = False
    max_backtest_days: int = 2_000
    data: DataParams = field(default_factory=DataParams)
    portfolio: PortfolioParams = field(default_factory=PortfolioParams)
    costs: CostParams = field(default_factory=CostParams)
    signal: SignalParams = field(default_factory=SignalParams)
    manual: ManualWorkflowParams = field(default_factory=ManualWorkflowParams)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BacktestConfig:
        start = _to_date(data.get("start_date"))
        end = _to_date(data.get("end_date"))
        return cls(
            start_date=start,
            end_date=end,
            initial_capital=float(data.get("initial_capital", 1_000_000.0)),
            rebalance_frequency=str(data.get("rebalance_frequency", "daily")),
            allow_short=bool(data.get("allow_short", False)),
            max_backtest_days=int(data.get("max_backtest_days", 2_000)),
            data=DataParams.from_dict(data.get("data", {})),
            portfolio=PortfolioParams.from_dict(data.get("portfolio", {})),
            costs=CostParams.from_dict(data.get("costs", {})),
            signal=SignalParams.from_dict(data.get("signal", {})),
            manual=ManualWorkflowParams.from_dict(data.get("manual", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "rebalance_frequency": self.rebalance_frequency,
            "allow_short": self.allow_short,
            "max_backtest_days": self.max_backtest_days,
            "data": self.data.to_dict(),
            "portfolio": self.portfolio.to_dict(),
            "costs": self.costs.to_dict(),
            "signal": self.signal.to_dict(),
            "manual": self.manual.to_dict(),
            "metadata": dict(self.metadata),
        }

    def with_overrides(self, overrides: Mapping[str, Any]) -> BacktestConfig:
        """Return new config with overrides, supports dot-notation."""

        nested = self.to_dict()
        for key, value in overrides.items():
            self._assign_dot_key(nested, key, value)
        nested.setdefault("metadata", {}).setdefault("overrides", {}).update(dict(overrides))
        return BacktestConfig.from_dict(nested)

    def apply_overrides_inplace(self, overrides: Mapping[str, Any]) -> None:
        updated = self.with_overrides(overrides)
        for field_name in dataclasses.asdict(self):
            setattr(self, field_name, getattr(updated, field_name))

    @staticmethod
    def _assign_dot_key(target: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        cursor: MutableMapping[str, Any] = target
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], MutableMapping):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value


def expand_parameter_grid(
    base: BacktestConfig, grid: Mapping[str, Sequence[Any]]
) -> Iterator[BacktestConfig]:
    """Generate parameter combinations.

    Args:
        base: Base configuration
        grid: {"portfolio.topk": [10, 20], "costs.slippage_bps": [2, 5]}
    Returns:
        BacktestConfig sequence generated by combinations, metadata will include `grid_keys` and `grid_index`.
    """

    if not grid:
        yield base
        return

    items = [(key, list(values)) for key, values in grid.items()]
    keys = [item[0] for item in items]
    for idx, combo in enumerate(itertools.product(*[item[1] for item in items])):
        overrides = dict(zip(keys, combo, strict=False))
        cfg = base.with_overrides(overrides)
        metadata = dict(cfg.metadata)
        metadata.setdefault("grid", {})
        metadata["grid"].update({"index": idx, "overrides": overrides})
        cfg.metadata = metadata
        yield cfg


__all__ = [
    "BacktestConfig",
    "CostParams",
    "PortfolioParams",
    "SignalParams",
    "ManualWorkflowParams",
    "DataParams",
    "expand_parameter_grid",
]
