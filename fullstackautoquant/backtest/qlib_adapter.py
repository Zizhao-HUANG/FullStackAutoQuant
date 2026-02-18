"""Qlib inference adapter: replicates training pipeline for backtest signal generation."""

from __future__ import annotations

import datetime as dt
import importlib
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

try:
    from fullstackautoquant.trading.utils import instrument_to_gm
except ImportError:
    from ..utils import instrument_to_gm

import sys

MODEL_PATH = Path(__file__).resolve().parents[3] / "ModelInferenceBundle" / "model.py"
if MODEL_PATH.exists():
    spec = importlib.util.spec_from_file_location("model", MODEL_PATH)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("model", module)
        spec.loader.exec_module(module)


class QlibInferenceError(RuntimeError):
    """Error during Qlib inference."""


@dataclass(slots=True)
class QlibInferenceConfig:
    combined_factors: Path
    params_path: Path
    provider_uri: Path
    region: str
    start: str
    instruments: str
    cache_dir: Path | None = None
    archive_dir: Path | None = None
    latest_csv: Path | None = None


class QlibInferenceAdapter:
    """Encapsulates training-equivalent pipeline, generates signals by date."""

    def __init__(self, cfg: QlibInferenceConfig):
        try:
            import qlib  # type: ignore
            from qlib.contrib.data.handler import DataHandlerLP  # type: ignore
            from qlib.contrib.model.pytorch_general_nn import GeneralPTNN  # type: ignore
            from qlib.data.dataset import DataHandlerLP as DH  # type: ignore
            from qlib.data.dataset import TSDatasetH  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise QlibInferenceError("qlib not installed, cannot run training-equivalent inference") from exc

        self._qlib = qlib
        self._DataHandlerLP = DataHandlerLP
        self._DH = DH
        self._TSDatasetH = TSDatasetH
        self._GeneralPTNN = GeneralPTNN
        self.cfg = cfg
        bundle_root = Path(__file__).resolve().parents[3] / "ModelInferenceBundle"
        archive_dir = cfg.archive_dir or (bundle_root / "ranked_scores_archive")
        latest_csv = cfg.latest_csv or (bundle_root / "ranked_scores_AUTO_via_qlib.csv")
        self.archive_dir = Path(archive_dir).resolve()
        self.latest_csv = Path(latest_csv).resolve()
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        self._init_env()
        self.handler = self._build_handler()
        self.available_dates = self._collect_available_dates()
        self.model, self._ideal_workers = self._load_model()
        self._prediction_cache: dict[dt.date, tuple[dt.date, list[dict]]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def generate_for_date(self, target_date: dt.date) -> tuple[dt.date, list[dict]]:
        if target_date in self._prediction_cache:
            return self._prediction_cache[target_date]
        stored = self._load_from_storage(target_date)
        if stored is not None:
            used_date, signals = stored
            result = (used_date, signals)
            self._prediction_cache[target_date] = result
            if used_date != target_date:
                self._prediction_cache[used_date] = result
            return result
        used_date = self._resolve_date(target_date)
        if used_date is None:
            result: tuple[dt.date, list[dict]] = (target_date, [])
            self._prediction_cache[target_date] = result
            return result

        dataset = self._TSDatasetH(
            handler=self.handler,
            segments={"test": [used_date.isoformat(), used_date.isoformat()]},
            step_len=72,
        )
        try:
            pred_series = self.model.predict(dataset)
        except Exception as exc:  # noqa: BLE001
            raise QlibInferenceError(f"Model prediction failed: {exc}") from exc

        if pred_series.empty:
            result = (target_date, [])
            self._prediction_cache[target_date] = result
            return result

        mi_dt = pred_series.index.get_level_values("datetime")
        used_date_pd = pd.Timestamp(used_date)
        mask = mi_dt == used_date_pd
        pred_day = pred_series[mask].sort_values(ascending=False)
        if pred_day.empty:
            result = (target_date, [])
            self._prediction_cache[target_date] = result
            return result

        score_col = pred_day.name if pred_day.name is not None else "score"
        pred_df = pred_day.to_frame(name=str(score_col)).reset_index()
        pred_df.rename(columns={"datetime": "date", str(score_col): "score"}, inplace=True)

        pred_df["confidence"] = 1.0

        signals: list[dict] = []
        for _, row in pred_df.iterrows():
            instrument = str(row["instrument"])
            gm_symbol = instrument_to_gm(instrument)
            if not gm_symbol:
                continue
            score = float(row.get("score", 0.0))
            confidence = float(row.get("confidence", 0.0) or 0.0)
            signals.append(
                {
                    "date": used_date.isoformat(),
                    "instrument": instrument,
                    "symbol": gm_symbol,
                    "score": score,
                    "confidence": confidence,
                }
            )

        result = (used_date, signals)
        self._prediction_cache[target_date] = result
        self._persist_signals(used_date, signals)
        return result

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _load_from_storage(self, target_date: dt.date) -> tuple[dt.date, list[dict]] | None:
        candidates: list[Path] = []
        archive_path = self.archive_dir / f"ranked_scores_{target_date:%Y-%m-%d}.csv"
        if archive_path.exists():
            candidates.append(archive_path)
        if self.latest_csv.exists() and self.latest_csv not in candidates:
            candidates.append(self.latest_csv)
        for path in candidates:
            loaded = self._load_signals_from_csv(path, target_date)
            if loaded is not None:
                return loaded
        return None

    def _load_signals_from_csv(self, path: Path, target_date: dt.date) -> tuple[dt.date, list[dict]] | None:
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        if df.empty:
            return (target_date, [])
        date_col = None
        for candidate in ("datetime", "date"):
            if candidate in df.columns:
                date_col = candidate
                break
        used_date = target_date
        if date_col:
            try:
                used_date = pd.to_datetime(df[date_col].iloc[0]).date()
            except Exception:
                used_date = target_date
        elif path.name.startswith("ranked_scores_") and path.suffix == ".csv":
            try:
                used_date = dt.datetime.strptime(path.stem.split("_")[-1], "%Y-%m-%d").date()
            except Exception:
                used_date = target_date
        if used_date != target_date and path != self.latest_csv:
            return None
        instrument_col = None
        for candidate in ("instrument", "code", "symbol"):
            if candidate in df.columns:
                instrument_col = candidate
                break
        if instrument_col is None:
            return None
        score_col = None
        for candidate in ("score", "0", "rank", "value"):
            if candidate in df.columns:
                score_col = candidate
                break
        if score_col is None:
            return None
        conf_col = None
        for candidate in df.columns:
            if str(candidate).lower() == "confidence":
                conf_col = candidate
                break
        signals: list[dict] = []
        for row in df.itertuples(index=False):
            instrument = str(getattr(row, instrument_col, "")).strip()
            if not instrument:
                continue
            gm_symbol = instrument_to_gm(instrument)
            if not gm_symbol:
                continue
            try:
                score = float(getattr(row, score_col))
            except Exception:
                continue
            try:
                confidence = float(getattr(row, conf_col)) if conf_col else 1.0
            except Exception:
                confidence = 1.0
            signals.append(
                {
                    "date": used_date.isoformat(),
                    "instrument": instrument,
                    "symbol": gm_symbol,
                    "score": score,
                    "confidence": confidence,
                }
            )
        return (used_date, signals)

    def _persist_signals(self, used_date: dt.date, signals: list[dict]) -> None:
        if not signals:
            return
        try:
            df = self._signals_to_dataframe(used_date, signals)
        except Exception:
            return
        archive_path = self.archive_dir / f"ranked_scores_{used_date:%Y-%m-%d}.csv"
        try:
            if not archive_path.exists():
                df.to_csv(archive_path, index=False)
        except Exception:
            pass
        try:
            self.latest_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.latest_csv, index=False)
        except Exception:
            pass

    def _signals_to_dataframe(self, used_date: dt.date, signals: list[dict]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for item in signals:
            instrument = item.get("instrument")
            score = item.get("score")
            confidence = item.get("confidence")
            if instrument is None or score is None:
                continue
            rows.append(
                {
                    "datetime": used_date.isoformat(),
                    "instrument": instrument,
                    "0": float(score),
                    "confidence": float(confidence) if confidence is not None else 1.0,
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.sort_values(by="0", ascending=False)
        return df

    def _init_env(self) -> None:
        provider_uri = str(self.cfg.provider_uri)
        self._qlib.init(provider_uri=provider_uri, region=self.cfg.region)

    def _build_handler(self):  # noqa: D401
        cf_df = pd.read_parquet(self.cfg.combined_factors)
        if not isinstance(cf_df.columns, pd.MultiIndex):
            cf_df.columns = pd.MultiIndex.from_tuples([("feature", str(col)) for col in cf_df.columns])
        if list(cf_df.index.names or []) != ["datetime", "instrument"]:
            cf_df = cf_df.copy()
            cf_df.index.set_names(["datetime", "instrument"], inplace=True)
        cf_df = cf_df.sort_index()

        data_loader = {
            "class": "NestedDataLoader",
            "kwargs": {
                "join": "left",
                "dataloader_l": [
                    {
                        "class": "qlib.contrib.data.loader.Alpha158DL",
                        "kwargs": {
                            "config": {
                                "label": [["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]],
                                "feature": [
                                    [
                                        "Resi($close, 5)/$close",
                                        "Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)",
                                        "Rsquare($close, 5)",
                                        "($high-$low)/$open",
                                        "Rsquare($close, 10)",
                                        "Corr($close, Log($volume+1), 5)",
                                        "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)",
                                        "Corr($close, Log($volume+1), 10)",
                                        "Ref($close, 60)/$close",
                                        "Resi($close, 10)/$close",
                                        "Std($volume, 5)/($volume+1e-12)",
                                        "Rsquare($close, 60)",
                                        "Corr($close, Log($volume+1), 60)",
                                        "Std(Abs($close/Ref($close, 1)-1)*$volume, 60)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 60)+1e-12)",
                                        "Std($close, 5)/$close",
                                        "Rsquare($close, 20)",
                                        "Corr($close/Ref($close,1), Log($volume/Ref($volume,  1)+1), 60)",
                                        "Corr($close/Ref($close,1), Log($volume/Ref($volume,  1)+1), 10)",
                                        "Corr($close, Log($volume+1), 20)",
                                        "(Less($open, $close)-$low)/$open",
                                    ],
                                    [
                                        "RESI5",
                                        "WVMA5",
                                        "RSQR5",
                                        "KLEN",
                                        "RSQR10",
                                        "CORR5",
                                        "CORD5",
                                        "CORR10",
                                        "ROC60",
                                        "RESI10",
                                        "VSTD5",
                                        "RSQR60",
                                        "CORR60",
                                        "WVMA60",
                                        "STD5",
                                        "RSQR20",
                                        "CORD60",
                                        "CORD10",
                                        "CORR20",
                                        "KLOW",
                                    ],
                                ],
                            }
                        },
                    },
                    {
                        "class": "qlib.data.dataset.loader.StaticDataLoader",
                        "kwargs": {"config": cf_df},
                    },
                ],
            },
        }

        handler_kwargs = {
            "instruments": self.cfg.instruments,
            "start_time": self.cfg.start,
            "end_time": None,
            "data_loader": data_loader,
            "infer_processors": [
                {
                    "class": "RobustZScoreNorm",
                    "kwargs": {
                        "fields_group": "feature",
                        "clip_outlier": True,
                        "fit_start_time": "2005-01-04",
                        "fit_end_time": "2021-12-31",
                    },
                },
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            "learn_processors": [],
        }

        return self._DataHandlerLP(**handler_kwargs)

    def _collect_available_dates(self) -> list[dt.date]:
        feat = self.handler.fetch(
            selector=slice(None, None),
            level="datetime",
            col_set="feature",
            data_key=self._DH.DK_I,
        )
        if feat.empty:
            return []
        dates = sorted(pd.to_datetime(feat.index.get_level_values("datetime").unique()).date)
        return dates

    def _resolve_date(self, target_date: dt.date) -> dt.date | None:
        if not self.available_dates:
            return None
        candidates = [d for d in self.available_dates if d <= target_date]
        if candidates:
            return candidates[-1]
        return self.available_dates[0] if self.available_dates else None

    def _load_model(self):
        params_path = self.cfg.params_path
        ideal_workers = min(8, max(1, (os.cpu_count() or 2) - 1))
        try:
            model = pd.read_pickle(params_path)
            try:
                model.dnn_model.eval()
            except Exception:  # noqa: BLE001
                pass
            if not getattr(model, "fitted", True):
                model.fitted = True
            return model, ideal_workers
        except Exception:
            pass

        model = self._GeneralPTNN(
            n_epochs=1,
            lr=2e-4,
            early_stop=1,
            batch_size=256,
            weight_decay=1e-4,
            metric="loss",
            loss="mse",
            n_jobs=ideal_workers,
            GPU=0,
            pt_model_uri="model.model_cls",
            pt_model_kwargs={"num_features": 22, "num_timesteps": 72},
        )

        state_obj = None
        try:
            state_obj = torch.load(params_path, map_location=torch.device("cpu"), weights_only=True)
        except Exception:
            try:
                state_obj = torch.load(params_path, map_location=lambda storage, loc: storage.cpu(), weights_only=False)
            except Exception:
                state_obj = None
        if state_obj is not None:
            state_dict = state_obj.get("state_dict", state_obj) if isinstance(state_obj, dict) else state_obj
            try:
                model.dnn_model.load_state_dict(state_dict, strict=False)
            except Exception:
                pass
        model.fitted = True
        return model, ideal_workers

def build_qlib_adapter(signal_params) -> QlibInferenceAdapter:
    if signal_params.combined_factors is None or signal_params.params_path is None:
        raise QlibInferenceError("Qlib inference requires combined_factors and params_path config")
    provider = signal_params.provider_uri or Path.home() / ".qlib/qlib_data/cn_data"
    cfg_kwargs = {
        "combined_factors": Path(signal_params.combined_factors),
        "params_path": Path(signal_params.params_path),
        "provider_uri": Path(provider),
        "region": signal_params.region,
        "start": signal_params.qlib_start,
        "instruments": signal_params.instruments,
    }
    cache_dir = getattr(signal_params, "cache_dir", None)
    if cache_dir:
        cfg_kwargs["cache_dir"] = Path(cache_dir)
    archive_dir = getattr(signal_params, "archive_dir", None)
    if archive_dir:
        cfg_kwargs["archive_dir"] = Path(archive_dir)
    latest_csv = getattr(signal_params, "latest_csv", None)
    if latest_csv:
        cfg_kwargs["latest_csv"] = Path(latest_csv)
    cfg = QlibInferenceConfig(**cfg_kwargs)
    return QlibInferenceAdapter(cfg)


__all__ = [
    "QlibInferenceAdapter",
    "QlibInferenceConfig",
    "QlibInferenceError",
    "build_qlib_adapter",
]
