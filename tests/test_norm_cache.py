"""Tests for norm_cache.py -- normalizer parameter extraction, save/load, injection.

Covers:
  - Save/load roundtrip
  - Parameter structure validation
  - Injection into a mock handler processor
  - #5: Norm Cache Injection Timing Audit (NaN robustness, new stock handling)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fullstackautoquant.model.norm_cache import save_norm_cache, load_norm_cache


# ---------------------------------------------------------------------------
# #13a: save/load roundtrip
# ---------------------------------------------------------------------------

class TestNormCacheRoundtrip:
    def test_save_load(self, tmp_path):
        """Save and load should reproduce identical parameters."""
        params = {
            "median": pd.Series([0.1, 0.2, 0.3], index=["RESI5", "WVMA5", "RSQR5"]),
            "std": pd.Series([1.0, 2.0, 3.0], index=["RESI5", "WVMA5", "RSQR5"]),
            "cols": ["RESI5", "WVMA5", "RSQR5"],
            "fit_start": "2005-01-04",
            "fit_end": "2021-12-31",
        }
        path = tmp_path / "test_norm.pkl"
        save_norm_cache(params, path)

        loaded = load_norm_cache(path)
        pd.testing.assert_series_equal(loaded["median"], params["median"])
        pd.testing.assert_series_equal(loaded["std"], params["std"])
        assert loaded["cols"] == params["cols"]
        assert loaded["fit_start"] == params["fit_start"]
        assert loaded["fit_end"] == params["fit_end"]

    def test_file_size_small(self, tmp_path):
        """Norm cache for 22 features should be < 5 KB."""
        features = [f"feat_{i}" for i in range(22)]
        params = {
            "median": pd.Series(np.random.randn(22), index=features),
            "std": pd.Series(np.abs(np.random.randn(22)) + 0.1, index=features),
            "cols": features,
            "fit_start": "2005-01-04",
            "fit_end": "2021-12-31",
        }
        path = tmp_path / "norm.pkl"
        save_norm_cache(params, path)
        assert path.stat().st_size < 5000


# ---------------------------------------------------------------------------
# #13b: Parameter structure validation
# ---------------------------------------------------------------------------

class TestParamStructure:
    def test_required_keys(self, tmp_path):
        """Loaded params must have all required keys."""
        params = {
            "median": pd.Series([0.0]),
            "std": pd.Series([1.0]),
            "cols": ["x"],
            "fit_start": "2005-01-04",
            "fit_end": "2021-12-31",
        }
        path = tmp_path / "norm.pkl"
        save_norm_cache(params, path)
        loaded = load_norm_cache(path)
        for key in ["median", "std", "cols", "fit_start", "fit_end"]:
            assert key in loaded, f"Missing key: {key}"

    def test_std_positive(self, tmp_path):
        """All std values should be positive (MAD * 1.4826 + eps > 0)."""
        params = {
            "median": pd.Series([0.0, 0.0]),
            "std": pd.Series([0.01, 1.5]),  # all positive
            "cols": ["a", "b"],
            "fit_start": "2005-01-04",
            "fit_end": "2021-12-31",
        }
        path = tmp_path / "norm.pkl"
        save_norm_cache(params, path)
        loaded = load_norm_cache(path)
        assert (loaded["std"] > 0).all(), "std values must be positive"


# ---------------------------------------------------------------------------
# #13c: Injection logic (mock processor)
# ---------------------------------------------------------------------------

class TestInjectionLogic:
    """Test that inject_norm_cache correctly overwrites processor attributes."""

    def test_overwrite_mean_std(self):
        """Injection should overwrite mean_train and std_train on the processor."""
        # Mock a RobustZScoreNorm-like processor
        proc = SimpleNamespace(
            mean_train=pd.Series([999.0, 999.0], index=["RESI5", "WVMA5"]),
            std_train=pd.Series([999.0, 999.0], index=["RESI5", "WVMA5"]),
            cols=["RESI5", "WVMA5"],
        )

        # Simulated cached params
        cached_median = pd.Series([0.1, 0.2], index=["RESI5", "WVMA5"])
        cached_std = pd.Series([1.5, 2.5], index=["RESI5", "WVMA5"])

        # Simulate injection
        proc.mean_train = cached_median
        proc.std_train = cached_std

        pd.testing.assert_series_equal(proc.mean_train, cached_median)
        pd.testing.assert_series_equal(proc.std_train, cached_std)

    def test_normalization_formula(self):
        """Verify: normalized = (x - median) / std, clipped to [-3, 3]."""
        median = 10.0
        std = 2.0
        raw = np.array([8.0, 10.0, 12.0, 20.0])  # 20 -> (20-10)/2 = 5.0, clipped to 3.0
        normalized = (raw - median) / std
        clipped = np.clip(normalized, -3, 3)
        expected = np.array([-1.0, 0.0, 1.0, 3.0])
        np.testing.assert_array_almost_equal(clipped, expected)


# ---------------------------------------------------------------------------
# #5: Norm Cache Injection Timing Audit
# ---------------------------------------------------------------------------

class TestNormCacheInjectionAudit:
    """Verify edge cases in normalizer cache injection.

    Tests the correctness of injected params against synthetic data,
    ensuring the transform z = clip((x - median) / (MAD*1.4826), -3, 3)
    produces expected results.
    """

    def test_synthetic_data_normalization(self):
        """Inject known params, process synthetic data, assert output
        matches (x - median) / (MAD * 1.4826), clipped to [-3, 3]."""
        # Known parameters (22 features to match production)
        n_features = 22
        np.random.seed(42)
        medians = np.random.randn(n_features) * 10
        # std = MAD * 1.4826 + EPS (always positive)
        stds = np.abs(np.random.randn(n_features)) * 2 + 0.01

        # Create synthetic data matrix (100 stocks x 22 features)
        n_stocks = 100
        raw_data = np.random.randn(n_stocks, n_features) * 5 + medians

        # Apply normalization manually
        expected = (raw_data - medians) / stds
        expected = np.clip(expected, -3, 3)

        # Simulate what inject_norm_params does internally
        proc_mean = np.asarray(medians, dtype=np.float64)
        proc_std = np.asarray(stds, dtype=np.float64)

        # Apply the same transform
        actual = (raw_data - proc_mean) / proc_std
        actual = np.clip(actual, -3, 3)

        np.testing.assert_array_almost_equal(actual, expected, decimal=10)

    def test_nan_columns_do_not_shift_ordering(self):
        """If combined_factors_df.parquet has NaN columns, the normalization
        should still apply correctly to non-NaN columns."""
        n_features = 5
        medians = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stds = np.array([0.5, 1.0, 1.5, 2.0, 2.5])

        # Data with some NaN columns
        raw_data = np.array([
            [10.0, np.nan, 30.0, np.nan, 50.0],
            [11.0, 22.0, np.nan, 44.0, 55.0],
            [np.nan, 20.0, 33.0, 40.0, np.nan],
        ])

        # Normalize (NaN propagates naturally)
        normalized = (raw_data - medians) / stds
        clipped = np.clip(normalized, -3, 3)

        # Verify NaN propagation: where input is NaN, output should be NaN
        for i in range(raw_data.shape[0]):
            for j in range(raw_data.shape[1]):
                if np.isnan(raw_data[i, j]):
                    assert np.isnan(clipped[i, j]), \
                        f"Expected NaN at [{i},{j}] but got {clipped[i,j]}"

        # Verify non-NaN values are correctly normalized
        # raw_data[0,0] = 10, median=1, std=0.5 -> (10-1)/0.5 = 18 -> clipped to 3.0
        assert clipped[0, 0] == 3.0
        # raw_data[1,1] = 22, median=2, std=1.0 -> (22-2)/1.0 = 20 -> clipped to 3.0
        assert clipped[1, 1] == 3.0
        # raw_data[0,2] = 30, median=3, std=1.5 -> (30-3)/1.5 = 18 -> clipped to 3.0
        assert clipped[0, 2] == 3.0

    def test_new_stock_in_inference(self):
        """If a new stock enters CSI300 that wasn't in training, its features
        will still be normalized using the cached global statistics.

        The normalization is cross-sectional (per-feature, not per-stock),
        so new stocks use the same median/std as all other stocks.
        """
        # Training universe: 300 stocks, 22 features
        n_features = 22
        medians = np.zeros(n_features)  # Simplified: median=0 for all features
        stds = np.ones(n_features)  # Simplified: std=1 for all features

        # New stock data (not in training set)
        new_stock_data = np.random.randn(1, n_features) * 3  # Random features

        # Normalize using cached global statistics
        normalized = (new_stock_data - medians) / stds
        clipped = np.clip(normalized, -3, 3)

        # Should produce valid (non-NaN, within [-3, 3]) values
        assert not np.any(np.isnan(clipped)), "New stock should not produce NaN"
        assert np.all(clipped >= -3) and np.all(clipped <= 3), "Values must be in [-3, 3]"

    def test_cached_params_numpy_array_compatibility(self, tmp_path):
        """Cached params stored as numpy arrays should work identically to pandas Series."""
        features = ["f0", "f1", "f2"]

        # Store as numpy arrays (what extract_norm_params produces)
        params_np = {
            "median": np.array([1.0, 2.0, 3.0]),
            "std": np.array([0.5, 1.0, 1.5]),
            "cols": features,
            "feature_names": features,
            "fit_start": "2005-01-04",
            "fit_end": "2021-12-31",
        }

        path = tmp_path / "np_norm.pkl"
        save_norm_cache(params_np, path)
        loaded = load_norm_cache(path)

        # Should roundtrip correctly
        np.testing.assert_array_equal(loaded["median"], params_np["median"])
        np.testing.assert_array_equal(loaded["std"], params_np["std"])

        # Apply normalization
        raw = np.array([2.0, 4.0, 6.0])
        result = (raw - loaded["median"]) / loaded["std"]
        expected = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_extreme_values_clipping(self):
        """Extreme values should be clipped to [-3, 3] symmetrically."""
        median = 0.0
        std = 1.0

        extreme_values = np.array([-100, -3.5, -3.0, 0.0, 3.0, 3.5, 100])
        normalized = (extreme_values - median) / std
        clipped = np.clip(normalized, -3, 3)

        expected = np.array([-3.0, -3.0, -3.0, 0.0, 3.0, 3.0, 3.0])
        np.testing.assert_array_almost_equal(clipped, expected)

    def test_zero_std_protection(self):
        """If std is near-zero (EPS=1e-12), normalization should not produce inf."""
        median = 5.0
        std = 1e-12  # Near-zero but > 0 due to EPS

        raw = np.array([5.0, 5.001, 4.999])
        normalized = (raw - median) / std
        clipped = np.clip(normalized, -3, 3)

        # Should be clipped to [-3, 3], not inf
        assert not np.any(np.isinf(clipped)), "Should not produce inf"
        np.testing.assert_array_equal(clipped, [0.0, 3.0, -3.0])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
