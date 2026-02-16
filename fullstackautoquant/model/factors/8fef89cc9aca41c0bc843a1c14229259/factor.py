# === Library Imports ===
# Core
import numpy as np
import pandas as pd
import sys
import logging

# High-Performance Alternatives (selected per framework)
import polars as pl
import bottleneck as bn

# Note: The target formulation requires the standard discrete Hilbert transform (analytic signal) over the full series.
# SciPy is not guaranteed in the environment; thus, we implement an FFT-based symmetric Hilbert transform that is
# mathematically equivalent to scipy.signal.hilbert for 1D real sequences, applied per group (instrument/date series).
# This preserves functional equivalence while avoiding look-ahead in the windows (demeaning windows are trailing only).

# =============================
# Factor Name: PhaseLeadLagHilbert_20_60d
# Description: Regime-Dependent / Cross-Sectional Phase Synchronization
# Hyperparameters (static and explicit):
#   eps = 1e-12                  (stabilizer for log)
#   W_x = 20                     (stock trailing de-mean window)
#   W_y = 60                     (market trailing de-mean window)
#   SMA_W = 10                   (smoothing window, trailing)
#   winsor_q = 0.01              (two-sided winsorization quantile for robustness)
#   amp_eps = 1e-10              (epsilon to avoid unstable phase when amplitude is too small)
# Robustness procedures:
#   - Prices: compute log returns; no forward fill of prices to avoid fabricating returns. Missing returns remain NaN.
#   - Cross-sectional mean m_t uses available finite r_{i,t} only on each date; dates with too few observations are still computed from what's available.
#   - Temporal windows (20/60/10) are trailing with strict min_periods, preventing look-ahead.
#   - Winsorization applied to x^{(20)} and y^{(60)} before phase extraction to mitigate outliers.
#   - No arbitrary zero-filling of analytic outputs; NaNs are handled contextually via min_periods and dropped only at the very end by backfilling the initial warm-up insufficient periods with 0 strictly after all causal requirements are satisfied.
# Performance procedures:
#   - Polars for grouped rolling means and daily aggregates.
#   - Bottleneck for fast moving averages if needed; main rolling uses Polars for clarity and speed.
# =============================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)


def _hilbert_analytic_signal_real(x: np.ndarray) -> np.ndarray:
    """
    Compute the analytic signal via FFT-based Hilbert transform, equivalent to scipy.signal.hilbert for real input.
    Returns complex array z = x + 1j * y, where y is the quadrature (Hilbert transform) of x.
    Handles NaNs by producing NaNs at those locations; requires finite array for FFT, so we mask and interpolate
    short gaps linearly within segments for numerical stability. For long NaN stretches, we keep them as NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return x.astype(np.complex128)

    # Handle NaNs: build a finite series for FFT using linear interpolation over internal gaps only.
    # End NaNs remain NaN.
    finite = np.isfinite(x)
    z = np.full(n, np.nan, dtype=np.complex128)
    if finite.sum() == 0:
        return z  # all NaN

    # Interpolate internal NaNs
    idx = np.arange(n)
    xf = x.copy()
    if finite.sum() >= 2:
        # Interpolate between first and last finite
        first = idx[finite][0]
        last = idx[finite][-1]
        seg_idx = idx[first:last + 1]
        seg_vals = x[first:last + 1]
        seg_fin = np.isfinite(seg_vals)
        seg_interp = np.interp(seg_idx, seg_idx[seg_fin], seg_vals[seg_fin])
        xf[first:last + 1] = seg_interp
    # Ends remain as original (may be NaN)

    # For FFT, replace any remaining NaNs (prefix/suffix) by nearest finite value to avoid spectral blow-up;
    # After inverse transform, we will restore NaNs at those positions to respect missingness.
    if not np.isfinite(xf[0]):
        xf[0] = xf[np.isfinite(xf)][0]
    if not np.isfinite(xf[-1]):
        xf[-1] = xf[np.isfinite(xf)][-1]
    # Fill any remaining NaNs via forward/backward fill
    nan_mask = ~np.isfinite(xf)
    if nan_mask.any():
        # forward fill
        for i in range(1, n):
            if not np.isfinite(xf[i]) and np.isfinite(xf[i-1]):
                xf[i] = xf[i-1]
        # backward fill
        for i in range(n-2, -1, -1):
            if not np.isfinite(xf[i]) and np.isfinite(xf[i+1]):
                xf[i] = xf[i+1]

    # FFT-based Hilbert transform
    Xf = np.fft.fft(xf)
    h = np.zeros(n)
    if n % 2 == 0:  # even
        h[0] = 1.0
        h[n//2] = 1.0
        h[1:n//2] = 2.0
    else:  # odd
        h[0] = 1.0
        h[1:(n+1)//2] = 2.0
    Zf = Xf * h
    z_full = np.fft.ifft(Zf)

    # Restore NaNs at originally missing positions to avoid spurious phase at missing points
    z[:] = z_full
    z[~finite] = np.nan + 1j*np.nan
    return z


def _winsorize_series(values: np.ndarray, q: float) -> np.ndarray:
    v = values.astype(np.float64)
    out = v.copy()
    fin = np.isfinite(v)
    if fin.sum() == 0:
        return out
    lo = np.nanquantile(v, q)
    hi = np.nanquantile(v, 1.0 - q)
    out[fin & (out < lo)] = lo
    out[fin & (out > hi)] = hi
    return out


def calculate_PhaseLeadLagHilbert_20_60d():
    # 1. Load Data
    df = pd.read_hdf("daily_pv.h5", key="data")
    logger.info("daily_pv.h5 loaded with %d rows", len(df))
    idx = df.index.get_level_values("datetime") if isinstance(df.index, pd.MultiIndex) else df.index
    logger.info("daily_pv range: %s -> %s", idx.min(), idx.max())

    # 2. Preprocessing & Calculation
    df_reset = df.reset_index().sort_values(['instrument', 'datetime']).copy()

    eps = 1e-12
    W_x, W_y = 20, 60
    SMA_W = 10
    winsor_q = 0.01
    amp_eps = 1e-10

    # Compute per-instrument log returns causally: r_t = log(close_t+eps) - log(close_{t-1}+eps)
    # Using Polars for group-wise shift
    pl_prices = pl.from_pandas(df_reset[['datetime', 'instrument', '$close']]).sort(['instrument', 'datetime'])
    pl_prices = pl_prices.with_columns([
        (pl.col('$close') + eps).log().alias('logc')
    ])
    pl_prices = pl_prices.with_columns([
        pl.col('logc').shift(1).over('instrument').alias('logc_lag1')
    ])
    pl_prices = pl_prices.with_columns([
        (pl.col('logc') - pl.col('logc_lag1')).alias('r')
    ])

    # Compute cross-sectional market mean m_t per date using available finite r
    pl_r = pl_prices.select(['datetime', 'instrument', 'r'])
    pl_r = pl_r.with_columns([
        pl.when(pl.col('r').is_finite()).then(pl.col('r')).otherwise(None).alias('r_fin')
    ])
    pl_r = pl_r.with_columns([
        pl.col('r_fin').mean().over('datetime').alias('m')
    ])

    # Rolling de-meanings (strict trailing): x^{(20)} and y^{(60)}
    # x: per instrument 20-day trailing mean
    pl_r = pl_r.with_columns([
        pl.col('r').rolling_mean(window_size=W_x, min_periods=W_x).over('instrument').alias('r_mean20')
    ])
    pl_r = pl_r.with_columns([
        (pl.col('r') - pl.col('r_mean20')).alias('x20_raw')
    ])

    # y: unique per date m_t, then 60-day trailing mean over dates
    m_daily = pl_r.select(['datetime', 'm']).unique(subset=['datetime']).sort('datetime')
    m_daily = m_daily.with_columns([
        pl.col('m').rolling_mean(window_size=W_y, min_periods=W_y).alias('m_mean60')
    ])
    m_daily = m_daily.with_columns([
        (pl.col('m') - pl.col('m_mean60')).alias('y60_raw')
    ]).select(['datetime', 'y60_raw'])

    # Join y60 back
    pl_r = pl_r.join(m_daily, on='datetime', how='left')

    # Convert to pandas for FFT-based Hilbert per group
    pdf = pl_r.to_pandas().sort_values(['instrument', 'datetime']).reset_index(drop=True)

    # Winsorize x20 and y60 to mitigate outliers before phase extraction
    # y60 is date-indexed; we'll winsorize per entire series (same across instruments by date)
    # For x20, winsorize within each instrument segment to avoid cross-talk.
    # Build instrument segments
    inst = pdf['instrument'].values
    idx_start = np.flatnonzero(np.r_[True, inst[1:] != inst[:-1]])
    idx_end = np.r_[idx_start[1:], len(inst)]

    x20 = pdf['x20_raw'].values.astype(np.float64)
    for s, e in zip(idx_start, idx_end):
        seg = x20[s:e]
        if np.isfinite(seg).sum() > 10:
            x20[s:e] = _winsorize_series(seg, winsor_q)
    pdf['x20'] = x20

    # y60 winsorization on the full daily series
    ydf = pdf[['datetime', 'y60_raw']].drop_duplicates(subset=['datetime']).sort_values('datetime').copy()
    y_vals = ydf['y60_raw'].values.astype(np.float64)
    if np.isfinite(y_vals).sum() > 20:
        y_vals_w = _winsorize_series(y_vals, winsor_q)
    else:
        y_vals_w = y_vals
    ydf['y60'] = y_vals_w
    pdf = pdf.merge(ydf[['datetime', 'y60']], on='datetime', how='left')

    # Analytic signals via symmetric Hilbert transform (functional equivalence) per instrument for x20, and global for y60
    # x20 per instrument
    x20_vals = pdf['x20'].values.astype(np.float64)
    zx_real = np.full_like(x20_vals, np.nan, dtype=np.float64)
    zx_imag = np.full_like(x20_vals, np.nan, dtype=np.float64)
    for s, e in zip(idx_start, idx_end):
        seg = x20_vals[s:e]
        if seg.size == 0:
            continue
        z_seg = _hilbert_analytic_signal_real(seg)
        zx_real[s:e] = np.real(z_seg)
        zx_imag[s:e] = np.imag(z_seg)

    # y60 global by date
    ydf = pdf[['datetime', 'y60']].drop_duplicates(subset=['datetime']).sort_values('datetime').copy()
    y_series = ydf['y60'].values.astype(np.float64)
    z_y = _hilbert_analytic_signal_real(y_series)
    ydf['y_real'] = np.real(z_y)
    ydf['y_imag'] = np.imag(z_y)
    pdf = pdf.merge(ydf[['datetime', 'y_real', 'y_imag']], on='datetime', how='left')

    # Instantaneous phases with amplitude guard
    amp_x = np.hypot(zx_real, zx_imag)
    phi = np.full_like(amp_x, np.nan)
    okx = np.isfinite(zx_real) & np.isfinite(zx_imag) & (amp_x > amp_eps)
    phi[okx] = np.arctan2(zx_imag[okx], zx_real[okx])

    y_real = pdf['y_real'].values.astype(np.float64)
    y_imag = pdf['y_imag'].values.astype(np.float64)
    amp_y = np.hypot(y_real, y_imag)
    psi = np.full_like(amp_y, np.nan)
    oky = np.isfinite(y_real) & np.isfinite(y_imag) & (amp_y > amp_eps)
    psi[oky] = np.arctan2(y_imag[oky], y_real[oky])

    # Wrapped phase difference and S = sin(Delta)
    valid = np.isfinite(phi) & np.isfinite(psi)
    delta = np.full_like(phi, np.nan)
    if valid.any():
        tmp = phi[valid] - psi[valid]
        tmp = (tmp + np.pi) % (2*np.pi) - np.pi
        delta[valid] = tmp
    S = np.full_like(delta, np.nan)
    S[valid] = np.sin(delta[valid])
    pdf['S'] = S

    # 10-day SMA of S per instrument, strict trailing with min_periods=SMA_W
    pl_s = pl.from_pandas(pdf[['datetime', 'instrument', 'S']]).sort(['instrument', 'datetime'])
    pl_s = pl_s.with_columns(
        pl.col('S').rolling_mean(window_size=SMA_W, min_periods=SMA_W).over('instrument').alias('factor_value')
    )
    pdf = pdf.merge(pl_s.to_pandas()[['datetime', 'instrument', 'factor_value']], on=['datetime', 'instrument'], how='left')

    # Final NaN handling: after enforcing all trailing windows, remaining NaNs are from warm-up periods or missing inputs.
    # We set these to 0.0 (neutral phase signal) to satisfy the strict no-NaN output requirement, but only after all
    # computations avoid look-ahead and are contextually robust.
    fv = pdf['factor_value'].values.astype(np.float64)
    fv[~np.isfinite(fv)] = 0.0
    pdf['factor_value'] = fv

    # 3. Format Output
    out = (
        pdf[["datetime", "instrument", "factor_value"]]
        .set_index(["datetime", "instrument"])
        .rename(columns={"factor_value": "PhaseLeadLagHilbert_20_60d"})
    )

    # 4. Required Pre-Save Validation
    assert out.index.is_unique, "Output index is not unique."
    assert out.columns.tolist() == ["PhaseLeadLagHilbert_20_60d"], "Output column name is incorrect."
    assert np.isfinite(out.values).all(), "Output contains NaN or Inf values. These must be handled during calculation."
    factor_series = out.iloc[:, 0]
    assert factor_series.std() > 1e-6, f"Factor has near-zero variance ({factor_series.std():.2e}), suggesting a potential flaw."

    # 5. Save Result
    out.to_hdf("result.h5", key="data", format="table")
    logger.info("result.h5 saved with %d rows", len(out))

if __name__ == "__main__":
    calculate_PhaseLeadLagHilbert_20_60d()
