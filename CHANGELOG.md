# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] (targeting v0.2.0)

### Added
* Lite Pipeline: single command daily inference via Tushare, no Docker or Dolt required
* Tushare data provider (`tushare_provider.py`) with incremental fetch, parallel adj_factor, and local constituent cache
* Normalizer parameter caching: 44 cached scalars replace 5 GB of historical data for production inference
* `load_model()` function with automatic format detection (state_dict, pickle, checkpoint) and CUDA to CPU remapping
* Lite pipeline scripts: `run_daily_lite.py`, `run_lite_workflow.sh`, `build_full_history.py`
* Comprehensive test suite: 300+ tests covering Tushare provider, norm cache, pipeline integration, output stability, and backward adjustment

### Fixed
* Normalizer cache was all NaN due to empty instrument coverage in fit window
* Factor synthesis path fragility: now searches both `data/` and `model/factors/` automatically
* `qlib_update.sh` error handling: removed silent error suppression, added Dolt auto upgrade and macOS Docker PATH
* polars deprecation: `min_periods` replaced with `min_samples` in factor definitions

### Changed
* README Quick Start now uses Lite Pipeline as primary path; Docker/Dolt moved to Legacy section
* `inference.py` refactored to use `load_model()` and support norm cache injection
* `RateLimiter` made thread safe with `threading.Lock`

## [0.1.0] - 2026-02-16

### Added
- Initial open-source release
- TCN-LocalAttention-GRU hybrid model architecture for stock ranking
- End-to-end inference pipeline with training-equivalent data handling
- MC Dropout uncertainty estimation for prediction confidence
- Automated trading execution via JoinQuant/GM Trade API
- Multi-layer risk management (drawdown monitoring, limit-up/down filtering)
- TopK rebalancing strategy with water-filling weight allocation
- Full backtesting engine with NAV tracking and performance metrics
- Streamlit WebUI for manual portfolio management
- Qlib data update pipeline (Docker-based)
- Custom factor synthesis framework
