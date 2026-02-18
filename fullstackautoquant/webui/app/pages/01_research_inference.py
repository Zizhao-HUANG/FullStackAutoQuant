from __future__ import annotations

import datetime as dt
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app.config_loader import ConfigLoader
from app.inference_runner import (
    InferenceError,
    StepResult,
    run_build_factors,
    run_export_daily_pv,
    run_full_pipeline,
    run_inference,
    run_update_qlib,
)


def _page_header() -> None:
    st.set_page_config(page_title="Research & Inference", layout="wide")
    st.title("Research & Inference (Training-Equivalent Pipeline)")
    st.caption("Update Qlib → export daily → Factor Synthesis → Inference Ranking, one-stop execution with visual assertions and logs.")


def _collect_paths(cfg: dict) -> dict[str, Path]:
    base_dir = Path(__file__).resolve().parents[2]
    paths_cfg = cfg.get("paths", {})  # type: ignore[index]
    infer_dir = (base_dir / paths_cfg.get("model_infer_dir", "../../ModelInferenceBundle")).resolve()
    ranked_csv = (base_dir / paths_cfg.get("ranked_csv", "../../ModelInferenceBundle/ranked_scores_AUTO_via_qlib.csv")).resolve()
    pv_h5 = infer_dir / "daily_pv.h5"
    factors_pq = infer_dir / "combined_factors_df.parquet"
    feature_pq = infer_dir / "features_ready_infer_AUTO.parquet"
    return {
        "infer_dir": infer_dir,
        "ranked_csv": ranked_csv,
        "daily_pv": pv_h5,
        "combined_factors": factors_pq,
        "feature_parquet": feature_pq,
    }


def _append_history(step: StepResult, run_label: str) -> None:
    history: list[dict[str, Any]] = st.session_state.setdefault("infer_history", [])
    history.append(
        {
            "run": run_label,
            "name": step.name,
            "stdout": step.stdout,
            "stderr": step.stderr,
            "output_paths": {k: str(v) for k, v in (step.output_paths or {}).items()},
            "started_at": step.started_at,
            "finished_at": step.finished_at,
            "duration_sec": step.duration_sec,
        }
    )


def _append_error(name: str, message: str, run_label: str) -> None:
    now = dt.datetime.now()
    history: list[dict[str, Any]] = st.session_state.setdefault("infer_history", [])
    history.append(
        {
            "run": run_label,
            "name": name,
            "stdout": "",
            "stderr": message,
            "output_paths": {},
            "started_at": now,
            "finished_at": now,
            "duration_sec": 0.0,
        }
    )


def _load_config() -> dict:
    loader = ConfigLoader()
    return loader.load()


def _read_tail(path: Path, n: int = 20) -> str:
    if not path.exists():
        return f"File does not exist: {path}"
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            block = min(size, 8192)
            fh.seek(max(0, size - block))
            data = fh.read().decode("utf-8", errors="ignore")
        lines = data.strip().splitlines()
        return "\n".join(lines[-n:])
    except Exception as exc:
        return f"Read failed: {exc}"


def _file_metadata(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        return {
            "exists": True,
            "size": stat.st_size,
            "mtime": dt.datetime.fromtimestamp(stat.st_mtime),
        }
    except FileNotFoundError:
        return {"exists": False, "size": 0, "mtime": None}
    except Exception as exc:  # pragma: no cover
        return {"exists": False, "size": 0, "error": str(exc), "mtime": None}


def _format_filesize(size: int) -> str:
    if size <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = min(int(np.log(size) / np.log(1024)), len(units) - 1)
    value = size / (1024 ** idx)
    return f"{value:.1f} {units[idx]}"


def _human_timedelta(ts: dt.datetime | None) -> str:
    if ts is None:
        return "Unknown"
    delta = dt.datetime.now() - ts
    days = delta.days
    seconds = delta.seconds
    if days > 0:
        return f"{days} days ago"
    hours = seconds // 3600
    if hours > 0:
        return f"{hours} hours ago"
    minutes = (seconds % 3600) // 60
    if minutes > 0:
        return f"{minutes} minutes ago"
    return "Just now"


def _disk_usage_summary(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
        total_gb = usage.total / 1024 ** 3
        free_gb = usage.free / 1024 ** 3
        used_pct = (usage.used / usage.total) * 100 if usage.total > 0 else None
        return {
            "total_gb": total_gb,
            "free_gb": free_gb,
            "used_pct": used_pct,
        }
    except FileNotFoundError:
        return {"total_gb": None, "free_gb": None, "used_pct": None}
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc), "total_gb": None, "free_gb": None, "used_pct": None}


def _latest_trading_day(df: pd.DataFrame) -> dt.date | None:
    candidates = []
    for col in ["datetime", "date", "trade_date"]:
        if col in df.columns:
            series = pd.to_datetime(df[col], errors="coerce")
            series = series.dropna()
            if not series.empty:
                candidates.append(series.max().date())
    if isinstance(df.index, pd.MultiIndex):
        try:
            level0 = pd.to_datetime(df.index.get_level_values(0), errors="coerce")
            level0 = level0.dropna()
            if not level0.empty:
                candidates.append(level0.max().date())
        except Exception:
            pass
    elif isinstance(df.index, pd.Index) and df.index.dtype != object:
        try:
            level = pd.to_datetime(df.index, errors="coerce")
            level = level.dropna()
            if not level.empty:
                candidates.append(level.max().date())
        except Exception:
            pass
    return max(candidates) if candidates else None


def _missing_summary(df: pd.DataFrame, limit: int = 10) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    sample = df.head(5000)
    missing = sample.isna().mean()
    missing = missing[missing > 0].sort_values(ascending=False)
    return missing.head(limit)


def _zero_variance_columns(df: pd.DataFrame, limit: int = 10) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return pd.Series(dtype=float)
    variances = numeric_df.var()
    zeros = variances[variances == 0]
    return zeros.head(limit)


def _build_missing_chart(missing: pd.Series, title: str) -> go.Figure | None:
    if missing.empty:
        return None
    fig = go.Figure(
        data=[go.Bar(x=missing.index.tolist(), y=(missing * 100).tolist(), marker_color="#FF7F0E")],
    )
    fig.update_layout(
        title=title,
        xaxis_title="Field",
        yaxis_title="Missing Rate (%)",
        hovermode="x",
        bargap=0.2,
    )
    return fig


def _format_duration(duration: float | None) -> str:
    if not duration:
        return "-"
    minutes, seconds = divmod(duration, 60)
    if minutes >= 1:
        return f"{int(minutes)}  min  {seconds:.1f} s"
    return f"{seconds:.1f} s"


def _extract_datetime_series(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    candidates = []
    for col in ["datetime", "date", "trade_date", "time", "day"]:
        if col in df.columns:
            series = pd.to_datetime(df[col], errors="coerce")
            if series.notna().any():
                candidates.append(series)
    if candidates:
        series = candidates[0]
        return series
    if isinstance(df.index, (pd.DatetimeIndex, pd.MultiIndex, pd.Index)):
        try:
            if isinstance(df.index, pd.MultiIndex):
                series = pd.to_datetime(df.index.get_level_values(0), errors="coerce")
            else:
                series = pd.to_datetime(df.index, errors="coerce")
            if series.notna().any():
                return series
        except Exception:
            return None
    return None


def _build_time_series(series: pd.Series, title: str, value_name: str) -> go.Figure | None:
    if isinstance(series, (pd.Index, np.ndarray)):
        series = pd.Series(series)
    series = pd.to_datetime(series, errors="coerce")
    if series.empty:
        return None
    counts = series.dropna()
    if counts.empty:
        return None
    counts = counts.dt.date.value_counts().sort_index()
    if counts.empty:
        return None
    counts = counts.tail(30)
    fig = go.Figure(
        data=[go.Scatter(x=counts.index.tolist(), y=counts.values.tolist(), mode="lines+markers", line_color="#1f77b4")]
    )
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=value_name, hovermode="x unified")
    return fig


def _build_histogram(values: pd.Series, title: str, nbins: int = 30) -> go.Figure | None:
    if values.empty:
        return None
    fig = go.Figure(data=[go.Histogram(x=values.dropna(), nbinsx=nbins, marker_color="#2ca02c")])
    fig.update_layout(title=title, xaxis_title="Value", yaxis_title="Frequency")
    return fig


def _build_score_heatmap(df: pd.DataFrame) -> go.Figure | None:
    if df.empty:
        return None
    subset = df.head(100)
    required_cols = [col for col in subset.columns if str(col).isdigit()]
    if not required_cols:
        return None
    matrix = subset[required_cols].to_numpy()
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=required_cols,
            y=subset.get("instrument", subset.index).astype(str),
            colorscale="Viridis",
            colorbar=dict(title="Score"),
        )
    )
    fig.update_layout(title="Top100 Score Distribution Heatmap", xaxis_title="Model Column", yaxis_title="Instrument")
    return fig


def _preview_parquet(path: Path) -> dict[str, Any]:
    try:
        df = pd.read_parquet(path)
        return {"exists": True, "df": df}
    except Exception as exc:
        return {"exists": path.exists(), "error": str(exc)}


def _preview_h5(path: Path) -> dict[str, Any]:
    try:
        df = pd.read_hdf(path, key="data")
        return {"exists": True, "df": df}
    except Exception as exc:
        return {"exists": path.exists(), "error": str(exc)}


def _preview_csv(path: Path) -> dict[str, Any]:
    try:
        df = pd.read_csv(path)
        count = len(df)
        csi300_count = None
        last_day = None
        if {"instrument"}.issubset(df.columns) and "datetime" in df.columns and not df.empty:
            last_day = df["datetime"].iloc[0]
            csi300_count = int((df["datetime"] == last_day).sum())
        return {
            "exists": True,
            "df": df,
            "rows": count,
            "csi300": csi300_count,
            "last_day": last_day,
        }
    except Exception as exc:
        return {"exists": path.exists(), "error": str(exc)}


def _env_overview(cfg: dict) -> dict[str, Path]:
    st.subheader("Environment Overview", divider="gray")
    paths = _collect_paths(cfg)

    cols = st.columns([1, 1, 1, 1])
    with cols[0]:
        st.caption("ModelInferenceBundle directory")
        st.code(str(paths["infer_dir"]))
        usage = _disk_usage_summary(paths["infer_dir"])
        if usage.get("used_pct") is not None:
            st.metric("Disk Usage", f"{usage['used_pct']:.1f}%", help=f"Available {usage['free_gb']:.2f} GB / Total {usage['total_gb']:.2f} GB")
        elif usage.get("error"):
            st.error(f"Disk query failed: {usage['error']}")

    infer_cfg = cfg.get("inference", {})  # type: ignore[index]
    with cols[1]:
        st.caption("Conda Environment")
        st.code(str(infer_cfg.get("conda_env", "rdagent4qlib")))
        python_exec = infer_cfg.get("python_exec", "python")
        st.caption("Python Executable Path")
        st.code(str(python_exec))

    day_txt = Path(os.path.expanduser(infer_cfg.get("qlib_data", "~/.qlib/qlib_data/cn_data"))) / "calendars" / "day.txt"
    day_meta = _file_metadata(day_txt)
    with cols[2]:
        st.caption("Qlib day.txt tail")
        st.text(_read_tail(day_txt, n=5))
        st.metric("Updated", _human_timedelta(day_meta.get("mtime")))

    ranked_meta = _file_metadata(paths["ranked_csv"])
    with cols[3]:
        st.caption("Latest Ranking Output")
        status = "✅ exists" if ranked_meta.get("exists") else "⚠️ missing"
        st.metric("ranked_scores.csv", status, help=_human_timedelta(ranked_meta.get("mtime")))
        if ranked_meta.get("exists"):
            st.caption("File Size")
            st.code(_format_filesize(int(ranked_meta.get("size", 0))))

    with st.expander("Daily Inspection Checklist", expanded=True):
        st.markdown(
            "- [ ] Confirm Qlib day.txt updated today; if lagging, check data ingestion\n"
            "- [ ] Check daily_pv.h5 / combined_factors / ranked CSV exist and timestamps match\n"
            "- [ ] Check the charts below:for anomalies in sample size, missing values, distribution\n"
            "- [ ] If any script failed or duration is abnormal, review logs and re-execute"
        )

    return paths


def _actions(cfg: dict) -> list[StepResult]:
    st.subheader("Execute Operations", divider="gray")
    col_full, col_upd, col_export, col_factor, col_infer = st.columns([1.2, 1, 1, 1, 1])
    run_label = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    executions: list[StepResult] = []

    with col_full:
        if st.button("Run Full Pipeline", type="primary"):
            try:
                steps = run_full_pipeline(cfg)
                st.session_state["infer_steps"] = steps
                for step in steps:
                    _append_history(step, run_label)
                executions.extend(steps)
                st.success("Full pipeline completed.")
            except InferenceError as exc:
                _append_error("full_pipeline", str(exc), run_label)
                st.error(str(exc))

    with col_upd:
        if st.button("Update Qlib"):
            try:
                res = run_update_qlib(cfg)
                st.session_state.setdefault("infer_steps", []).append(res)
                _append_history(res, run_label)
                executions.append(res)
                st.success("Qlib update completed")
            except InferenceError as exc:
                _append_error("run_update_qlib", str(exc), run_label)
                st.error(str(exc))

    with col_export:
        if st.button("Export daily_pv.h5"):
            try:
                res = run_export_daily_pv(cfg)
                st.session_state.setdefault("infer_steps", []).append(res)
                _append_history(res, run_label)
                executions.append(res)
                st.success("Export completed")
            except InferenceError as exc:
                _append_error("run_export_daily_pv", str(exc), run_label)
                st.error(str(exc))

    with col_factor:
        if st.button("Factor Synthesis"):
            try:
                res = run_build_factors(cfg)
                st.session_state.setdefault("infer_steps", []).append(res)
                _append_history(res, run_label)
                executions.append(res)
                st.success("Factors generated/updated")
            except InferenceError as exc:
                _append_error("run_build_factors", str(exc), run_label)
                st.error(str(exc))

    with col_infer:
        if st.button("Inference Ranking"):
            try:
                res = run_inference(cfg)
                st.session_state.setdefault("infer_steps", []).append(res)
                _append_history(res, run_label)
                executions.append(res)
                st.success("Inference complete")
            except InferenceError as exc:
                _append_error("run_inference", str(exc), run_label)
                st.error(str(exc))

    return executions


def _artifacts(cfg: dict, paths: dict[str, Path]) -> None:
    st.subheader("Output Preview & Validation", divider="gray")
    result_h5 = _preview_h5(paths["daily_pv"])
    result_pq = _preview_parquet(paths["combined_factors"])
    result_csv = _preview_csv(paths["ranked_csv"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**daily_pv.h5**")
        if result_h5.get("exists") and "df" in result_h5:
            st.metric("Row Count", len(result_h5["df"]))
            st.dataframe(result_h5["df"].head(10))
            latest_day = _latest_trading_day(result_h5["df"]) or "-"
            meta = _file_metadata(paths["daily_pv"])
            st.caption(f"Latest date:{latest_day}｜Updated:{_human_timedelta(meta.get('mtime'))}｜Size:{_format_filesize(int(meta.get('size', 0)))}")
            dt_series = _extract_datetime_series(result_h5["df"])
            if dt_series is not None:
                fig_ts = _build_time_series(dt_series, "Last 30 Days daily_pv Entry Count", "Entry Count")
                if fig_ts is not None:
                    st.plotly_chart(fig_ts, use_container_width=True)
            missing = _missing_summary(result_h5["df"])
            fig_missing = _build_missing_chart(missing, "Missing Rate Top10")
            if fig_missing is not None:
                st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.error(result_h5.get("error", "File does not exist"))

    with col2:
        st.markdown("**combined_factors_df.parquet**")
        if result_pq.get("exists") and "df" in result_pq:
            df = result_pq["df"]
            st.metric("Latest date", str(df.index.get_level_values(0).max()) if isinstance(df.index, pd.MultiIndex) else "-")
            if isinstance(df.columns, pd.MultiIndex):
                st.write("Column groups:", df.columns.names)
            st.dataframe(df.tail(20))
            latest_day = _latest_trading_day(df) or "-"
            meta = _file_metadata(paths["combined_factors"])
            st.caption(f"Latest date:{latest_day}｜Updated:{_human_timedelta(meta.get('mtime'))}｜Size:{_format_filesize(int(meta.get('size', 0)))}")
            missing = _missing_summary(df)
            fig_missing = _build_missing_chart(missing, "Missing Rate Top10 (factors)")
            if fig_missing is not None:
                st.plotly_chart(fig_missing, use_container_width=True)
            zeros = _zero_variance_columns(df)
            if not zeros.empty:
                st.warning("The following fields have zero variance and may need cleaning:" + ", ".join([str(col) for col in zeros.index]))
            dt_series = _extract_datetime_series(df)
            if dt_series is not None:
                fig_ts = _build_time_series(dt_series, "Last 30 Days Factor Record Count", "Row Count")
                if fig_ts is not None:
                    st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.error(result_pq.get("error", "File does not exist"))

    with col3:
        st.markdown("**ranked_scores_AUTO_via_qlib.csv**")
        if result_csv.get("exists") and "df" in result_csv:
            df = result_csv["df"]
            st.metric("Total Rows", result_csv.get("rows", 0))
            if result_csv.get("csi300") is not None:
                st.metric("CSI300 instruments on this day", result_csv["csi300"])
            st.dataframe(df.head(20))
            if {
                "datetime",
                "0",
            }.issubset(df.columns):
                top = df.head(50)
                fig = px.bar(top, x="instrument", y="0", title="Top50 Scores", height=400)
                st.plotly_chart(fig, use_container_width=True)
                hist_fig = _build_histogram(df["0"], "Score Distribution Histogram")
                if hist_fig is not None:
                    st.plotly_chart(hist_fig, use_container_width=True)
                heatmap_fig = _build_score_heatmap(df)
                if heatmap_fig is not None:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
            dt_series = _extract_datetime_series(df)
            if dt_series is not None:
                fig_ts = _build_time_series(dt_series, "Last 30 Days Ranking Record Count", "Row Count")
                if fig_ts is not None:
                    st.plotly_chart(fig_ts, use_container_width=True)
            meta = _file_metadata(paths["ranked_csv"])
            st.caption(
                f"Latest trading day:{result_csv.get('last_day', '-')}")
            st.caption(f"Updated:{_human_timedelta(meta.get('mtime'))}｜Size:{_format_filesize(int(meta.get('size', 0)))}")
            st.download_button(
                "Download Full Ranking CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="ranked_scores_latest.csv",
                mime="text/csv",
            )
        else:
            st.error(result_csv.get("error", "File does not exist"))

    st.caption("If CSI300 instruments on the last day are fewer than 300, it will be highlighted here; outputs are preserved for manual review.")
    st.caption(
        "If missing rate >5%、distribution cliff or execution time spike, pause automated dispatch and contact quant researcher to verify upstream data."
    )

    if st.button("Use this CSV as trading page input"):
        st.success("Now pointing to paths.ranked_csv. Go to Manual Trading Console to run the workflow.")


def _build_history_timeline(history: list[dict[str, Any]]) -> go.Figure | None:
    if not history:
        return None
    df = pd.DataFrame(history)
    df = df.sort_values(["run", "started_at"])  # type: ignore[arg-type]
    df["start"] = pd.to_datetime(df["started_at"], errors="coerce")
    df["finish"] = pd.to_datetime(df["finished_at"], errors="coerce")
    df["duration"] = df["duration_sec"].fillna(0)
    df = df.dropna(subset=["start"])
    if df.empty:
        return None
    df["duration_display"] = df["duration"].map(_format_duration)
    data = []
    for run_label, group in df.groupby("run"):
        data.append(
            go.Bar(
                base=group["start"],
                x=group["duration"],
                y=group["name"],
                orientation="h",
                width=0.4,
                name=str(run_label),
                hovertext=[
                    f"Start: {row['start']}<br>Duration: {row['duration_display']}<br>stderr: {row['stderr']}"
                    for _, row in group.iterrows()
                ],
                hoverinfo="text",
            )
        )
    fig = go.Figure(data=data)
    fig.update_layout(
        title="Execution Timeline",
        xaxis_title="Start Time",
        yaxis_title="Step",
        legend_title="Execution Batch",
        hovermode="closest",
    )
    return fig


def _render_execution_history(executions: list[StepResult]) -> None:
    st.subheader("Execution History / Diagnostics", divider="gray")
    history: list[dict[str, Any]] = st.session_state.get("infer_history", [])  # type: ignore[assignment]
    if executions:
        last_run_label = executions[-1].started_at.strftime("%Y-%m-%d %H:%M:%S") if executions[-1].started_at else "Recent execution"
        st.success(f"Latest execution batch:{last_run_label}, {len(executions)}  steps.Check duration and stderr for anomalies.")
        for step in executions:
            with st.expander(f"Step {step.name}(Duration { _format_duration(step.duration_sec) })", expanded=False):
                st.text("STDOUT:\n" + (step.stdout or "<empty>"))
                if step.stderr:
                    st.error(step.stderr)
                if step.output_paths:
                    st.write({k: str(v) for k, v in step.output_paths.items()})
    else:
        st.info("No new execution records.Click buttons above to start the pipeline.")

    if not history:
        st.warning("No history records yet, Run the full pipeline once to establish a baseline.")
        return

    timeline_fig = _build_history_timeline(history)
    if timeline_fig is not None:
        st.plotly_chart(timeline_fig, use_container_width=True)
    else:
        st.write("History records missing timestamps or duration, cannot plot.")

    hist_df = pd.DataFrame(history)
    hist_df["started_at"] = pd.to_datetime(hist_df["started_at"], errors="coerce")
    hist_df["finished_at"] = pd.to_datetime(hist_df["finished_at"], errors="coerce")
    hist_df["duration"] = hist_df["duration_sec"].map(_format_duration)
    st.dataframe(
        hist_df[["run", "name", "started_at", "finished_at", "duration", "stderr"]].tail(20),
        use_container_width=True,
    )


def main() -> None:
    _page_header()
    cfg = _load_config()
    paths = _env_overview(cfg)
    latest_execs = _actions(cfg)
    _artifacts(cfg, paths)
    _render_execution_history(latest_execs)


if __name__ == "__main__":
    main()


