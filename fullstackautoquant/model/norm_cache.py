"""Normalizer state caching: extract, persist, and inject RobustZScoreNorm parameters.

The transform z = clip((x - median) / (MAD*1.4826), -3, 3) is fully determined by 44
scalars (22 medians + 22 MAD-based scales). This module decouples the *fit* phase
(requires 17 years of historical data) from the *transform* phase (requires only
today's features), enabling a lightweight inference pipeline.

Qlib internals (verified against qlib 0.9.7, processor.py:262-297):
    - RobustZScoreNorm.fit() sets:  mean_train, std_train, cols
    - RobustZScoreNorm.__call__() uses:  self.cols, self.mean_train, self.std_train
    - std_train = (MAD + EPS) * 1.4826,  where MAD = median(|x - median(x)|), EPS = 1e-12

See docs/normalizer_caching.md for the statistical justification.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from fullstackautoquant.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Extract fitted state from a live handler
# ---------------------------------------------------------------------------


def extract_norm_params(handler: Any) -> dict[str, Any]:
    """Extract (median, MAD-scale) vectors from a handler's fitted RobustZScoreNorm.

    Reads the processor's ``mean_train`` (median) and ``std_train`` (MAD*1.4826+EPS).

    Returns:
        dict with keys: median, std, cols, feature_names, fit_start, fit_end
    """
    proc = _find_robust_norm(handler)
    median = np.asarray(proc.mean_train, dtype=np.float64).ravel()
    std = np.asarray(proc.std_train, dtype=np.float64).ravel()

    # Cache cols (the column index object that __call__ needs)
    cols = proc.cols

    # Capture metadata for provenance
    fit_start = getattr(proc, "fit_start_time", None)
    fit_end = getattr(proc, "fit_end_time", None)

    feature_names: list[str] = []
    try:
        feature_names = [str(c) for c in cols]
    except Exception:
        feature_names = [f"f{i}" for i in range(len(median))]

    logger.info(
        "Extracted normalizer params: %d features, fit=[%s, %s]",
        len(median),
        fit_start,
        fit_end,
    )
    return {
        "median": median,
        "std": std,
        "cols": cols,
        "feature_names": feature_names,
        "fit_start": str(fit_start) if fit_start else None,
        "fit_end": str(fit_end) if fit_end else None,
    }


# ---------------------------------------------------------------------------
# Serialize / deserialize
# ---------------------------------------------------------------------------


def save_norm_cache(params: dict[str, Any], path: Path) -> None:
    """Persist normalizer parameters to a pickle file (~1 KB)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved normalizer cache -> %s (%d bytes)", path, path.stat().st_size)


def load_norm_cache(path: Path) -> dict[str, Any]:
    """Load cached normalizer parameters from pickle."""
    path = Path(path)
    with open(path, "rb") as f:
        params = pickle.load(f)  # noqa: S301
    n = len(params.get("median", []))
    logger.info(
        "Loaded normalizer cache: %d features, fit=[%s, %s]",
        n,
        params.get("fit_start"),
        params.get("fit_end"),
    )
    return params


# ---------------------------------------------------------------------------
# Inject cached params into a handler (the key trick)
# ---------------------------------------------------------------------------


def inject_norm_params(handler: Any, cached: dict[str, Any]) -> None:
    """Replace the handler's RobustZScoreNorm fitted state with cached values,
    then re-process ALL data through the corrected pipeline.

    The handler has already been constructed by ``init_instance_by_config``, which
    calls ``setup_data() -> fit_process_data() -> process_data(with_fit=True)``.
    By this point, ``handler._infer`` is computed with WRONG statistics (fitted on
    whatever limited data was available).

    This function:
    1. Overwrites ``mean_train`` and ``std_train`` with the cached 17-year values.
    2. Calls ``handler.process_data(with_fit=False)`` to re-run all processors'
       ``__call__`` using the corrected parameters, WITHOUT re-fitting.

    The result is bit-for-bit identical to the full-data pipeline.
    """
    proc = _find_robust_norm(handler)
    proc.mean_train = np.asarray(cached["median"], dtype=np.float64)
    proc.std_train = np.asarray(cached["std"], dtype=np.float64)
    # Restore cols if cached (for robustness); otherwise keep the cols
    # that were set during the initial (wrong) fit, which are structurally correct.
    if "cols" in cached and cached["cols"] is not None:
        proc.cols = cached["cols"]

    # Re-process: applies all processors' __call__ with corrected params.
    # with_fit=False skips proc.fit() so our injected values are preserved.
    handler.process_data(with_fit=False)

    logger.info(
        "Injected cached normalizer params (%d features) and re-processed handler data",
        len(cached["median"]),
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _find_robust_norm(handler: Any) -> Any:
    """Locate the RobustZScoreNorm processor inside a DataHandlerLP."""
    # DataHandlerLP stores processors directly as handler.infer_processors (a list)
    processors = getattr(handler, "infer_processors", [])
    for proc in processors:
        cls_name = type(proc).__name__
        if "RobustZScoreNorm" in cls_name:
            return proc
    raise ValueError(
        "Cannot find RobustZScoreNorm in handler.infer_processors. "
        f"Found: {[type(p).__name__ for p in processors]}"
    )
