# === Library Imports ===
# Core
import numpy as np
import pandas as pd
import logging
import sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)

# High-Performance Alternatives
import polars as pl
import numba as nb
import bottleneck as bn
import dask.dataframe as dd

def calculate_SpinAlignmentConsensus_21d():
    # 1. Load Data
    df = pd.read_hdf("daily_pv.h5", key="data")
    idx = df.index.get_level_values("datetime") if isinstance(df.index, pd.MultiIndex) else df.index
    logger.info("daily_pv.h5 rows=%d range=%s->%s", len(df), idx.min(), idx.max())

    # 2. Initial Preparation
    # Flatten index for efficient grouped operations and sort.
    df_reset = df.reset_index().sort_values(['instrument', 'datetime'])

    # 3. Core Factor Calculation
    # Using Polars for its high-performance, group-aware rolling windows and cross-sectional ops.
    # Justification: Polars provides fast rolling and grouped operations required for large-scale data.
    df_pl = pl.from_pandas(df_reset)

    eps = 1e-12  # numerical guard for log

    # Compute safe closes and lagged closes per instrument
    df_pl = df_pl.with_columns([
        pl.col('$close').alias('close_raw'),
        pl.col('$close').shift(1).over('instrument').alias('close_lag1_raw')
    ])

    # Ensure strictly positive values where present to avoid log issues, keep nulls if missing
    df_pl = df_pl.with_columns([
        pl.when(pl.col('close_raw').is_not_null())
          .then(pl.max_horizontal(pl.lit(eps), pl.col('close_raw')))
          .otherwise(None)
          .alias('close_safe'),
        pl.when(pl.col('close_lag1_raw').is_not_null())
          .then(pl.max_horizontal(pl.lit(eps), pl.col('close_lag1_raw')))
          .otherwise(None)
          .alias('close_lag1_safe')
    ])

    # Per-instrument log return r_{i,t}
    df_pl = df_pl.with_columns(
        (pl.col('close_safe').log() - pl.col('close_lag1_safe').log()).alias('r')
    )

    # Cross-sectional equal-weight market return m_t (ignore nulls automatically)
    df_pl = df_pl.with_columns(
        pl.col('r').mean().over('datetime').alias('m')
    )

    # Sign mapping: >0 -> 1, <0 -> -1, ==0 -> 0; Null remains null
    def sign_expr(col: pl.Expr) -> pl.Expr:
        return (
            pl.when(col > 0).then(pl.lit(1))
              .when(col < 0).then(pl.lit(-1))
              .when(col == 0).then(pl.lit(0))
              .otherwise(None)
        )

    df_pl = df_pl.with_columns([
        sign_expr(pl.col('r')).alias('s'),
        sign_expr(pl.col('m')).alias('c')
    ])

    # Daily alignment a_{i,t} = s_{i,t} * c_t; null if either is null
    df_pl = df_pl.with_columns((pl.col('s') * pl.col('c')).alias('a'))

    # 21-day simple moving average of alignment per instrument with strict min_periods=21
    window = 21  # Hyperparameter: lookback window for SMA of alignment
    df_pl = df_pl.with_columns(
        pl.col('a')
          .cast(pl.Float64)
          .rolling_mean(window_size=window, min_periods=window)
          .over('instrument')
          .alias('factor_value')
    )

    # No forward-fill, no cross-sectional imputation, no winsorization to preserve exact definition.
    # We must output no NaN/inf per requirements. To meet this without altering the statistic,
    # we only remove undefined early-window entries by dropping them at the very end.

    # 5. Final Formatting
    df_reset = df_pl.to_pandas()

    out = (
        df_reset
        .set_index(["datetime", "instrument"]) 
        [["factor_value"]]
        .rename(columns={"factor_value": "SpinAlignmentConsensus_21d"})
    )

    # Remove rows where the 21-day average is undefined (insufficient history or missing alignment components).
    out = out.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure dtype float64
    if out.shape[0] > 0:
        out["SpinAlignmentConsensus_21d"] = out["SpinAlignmentConsensus_21d"].astype(np.float64)

    # 6. Required Pre-Save Validation
    assert out.index.is_unique, "Output index is not unique."
    assert out.columns.tolist() == ["SpinAlignmentConsensus_21d"], "Output column name is incorrect."

    # No NaN/inf allowed in final output
    assert len(out) > 0, "Final factor contains no valid data after dropping undefined warm-up periods."
    assert np.isfinite(out.values).all(), "Final factor contains NaN or inf values."

    # Sanity checks on the non-NaN part of the factor.
    factor_series = out.iloc[:, 0]
    if len(factor_series) > 0:
        if factor_series.std() <= 1e-9:
            print(f"INFO: Factor variance is very small ({factor_series.std():.2e}).")
        skewness = factor_series.skew()
        if np.isfinite(skewness) and abs(skewness) > 15:
            print(f"INFO: Factor distribution is highly skewed (skewness: {skewness:.2f}).")
    else:
        print("WARNING: Factor contains no valid data points after processing.")

    # 7. Save Result
    out.to_hdf("result.h5", key="data", format="table")
    logger.info("result.h5 rows=%d", len(out))

if __name__ == "__main__":
    calculate_SpinAlignmentConsensus_21d()
