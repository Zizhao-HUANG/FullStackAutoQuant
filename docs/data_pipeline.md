# Data Pipeline

> How FullStackAutoQuant ingests raw market data, synthesizes alpha factors, and constructs the feature matrix for model inference.

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Data Flow Diagram](#data-flow-diagram)
- [Stage 1: Qlib Data Update](#stage-1-qlib-data-update)
- [Stage 2: Daily Price Volume Export](#stage-2-daily-price-volume-export)
- [Stage 3: Factor Synthesis](#stage-3-factor-synthesis)
- [Stage 4: Feature Matrix Construction](#stage-4-feature-matrix-construction)
- [Data Verification](#data-verification)
- [Daily Update Schedule](#daily-update-schedule)
- [Data Formats Reference](#data-formats-reference)
- [Custom Factor Development](#custom-factor-development)

---

## Pipeline Overview

The data pipeline transforms raw A-share market data into a clean, normalized feature matrix suitable for the TCN-LocalAttn-GRU model. It is designed around three principles:

1. **Training-inference equivalence** — The inference pipeline uses the exact same `DataHandlerLP` configuration, processor chain, and feature definitions as the training snapshot. This eliminates train-serve skew.

2. **Auditability** — Every stage is a standalone script with clear inputs/outputs. Intermediate artifacts (`.h5`, `.parquet`) are persisted for debugging and reproducibility.

3. **Idempotency** — Running any stage multiple times produces the same result. Data is overwritten atomically, not appended.

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                      │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   External    │    │  Qlib Binary │    │   Factor     │    │  Feature   │ │
│  │   Sources     │───▶│    Store     │───▶│  Synthesis   │───▶│   Matrix   │ │
│  │              │    │              │    │              │    │            │ │
│  │ • A-share    │    │ ~/.qlib/     │    │ combined_    │    │ N × 72 ×  │ │
│  │   OHLCV      │    │ qlib_data/   │    │ factors_df   │    │ 22 tensor │ │
│  │ • Dividends  │    │ cn_data/     │    │ .parquet     │    │            │ │
│  │ • Splits     │    │              │    │              │    │            │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│        │                    │                   │                  │         │
│   Docker pull          D.features()      RD-Agent factors    DataHandlerLP │
│   chenditc/            Alpha158 DL       + custom alpha      + processors  │
│   investment_data                                                           │
│                                                                              │
│  Scripts:                                                                    │
│  qlib_update.sh     export_daily_pv.py   factor_synthesis.py  build_features│
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Qlib Data Update

**Script:** `fullstackautoquant/data/qlib_update.sh`  
**Input:** Internet (A-share market data)  
**Output:** `~/.qlib/qlib_data/cn_data/` (Qlib binary format)

### What It Does

1. Pulls the [`chenditc/investment_data`](https://github.com/chenditc/investment_data) Docker image — a crowdsourced A-share data pipeline maintained by the open-source community.

2. Runs the data fetch and conversion inside Docker, producing Qlib-compatible binary files (`.bin` + `.today` format).

3. Copies the converted data to the local Qlib data directory.

### Usage

```bash
# Full update (first run or periodic refresh)
bash fullstackautoquant/data/qlib_update.sh

# With custom output directory
QLIB_DATA_DIR=/custom/path bash fullstackautoquant/data/qlib_update.sh
```

### Requirements

- **Docker** must be installed and running
- Internet access for data download
- ~10 GB disk space for full A-share history

### Data Coverage

| Field | Coverage |
|-------|----------|
| Instruments | All A-share listed stocks (SSE + SZSE) |
| Date range | 2005-01-04 to present |
| Frequency | Daily |
| Fields | Open, High, Low, Close, Volume, Amount, Factor (adj) |

### Error Handling

The script includes retry logic for network failures and validates the output directory structure before reporting success.

---

## Stage 2: Daily Price Volume Export

**Script:** `fullstackautoquant/data/export_daily_pv.py`  
**Input:** Qlib binary data store  
**Output:** `daily_pv.h5` (HDF5, key=`data`)

### What It Does

Exports daily OHLCV data (plus optional `$factor` for adjustment) from Qlib's binary format to a portable HDF5 file. This is used by downstream factor scripts that compute custom alpha factors.

### Design Principles

- Uses the official `Qlib Data API`: `D.features(..., freq='day')`
- Index: `MultiIndex['datetime', 'instrument']`, ascending order
- No forward/backward fill — raw data is exported as-is
- Persistence: `pandas.to_hdf(key='data')` with no extra processing

### Usage

```bash
python -m fullstackautoquant.data.export_daily_pv \
  --provider_uri ~/.qlib/qlib_data/cn_data \
  --fields '$open,$high,$low,$close,$volume,$factor' \
  --out fullstackautoquant/data/daily_pv.h5
```

### Output Schema

```
daily_pv.h5 (key='data')
├── Index: MultiIndex['datetime', 'instrument']
│   datetime:   Timestamp (YYYY-MM-DD)
│   instrument: str (e.g., 'SH600519', 'SZ000858')
└── Columns: $open, $high, $low, $close, $volume, $factor
```

---

## Stage 3: Factor Synthesis

**Script:** `fullstackautoquant/data/factor_synthesis.py`  
**Input:** Qlib data store + custom factor definitions  
**Output:** `combined_factors_df.parquet`

### What It Does

This is the most complex stage. It:

1. **Loads individual factor scripts** from `fullstackautoquant/model/factors/*/factor.py` — each script defines a custom alpha factor.

2. **Executes each factor** using the RD-Agent `QlibFactorExperiment` framework, which handles data loading, computation, and validation.

3. **Merges** all computed factors into a single DataFrame alongside the standard Alpha158 features.

4. **Filters** to the target instrument universe (default: CSI300).

5. **Saves** the merged result as `combined_factors_df.parquet`.

### Usage

```bash
python -m fullstackautoquant.data.factor_synthesis \
  --provider_uri ~/.qlib/qlib_data/cn_data \
  --universe csi300 \
  --out fullstackautoquant/data/combined_factors_df.parquet
```

### Custom Factor Structure

Each factor lives in its own directory under `model/factors/`:

```
model/factors/
├── 1df46d60d6134d1a9801dfca16491986/
│   └── factor.py    # Computes "volume_price_divergence" factor
└── 8fef89cc9aca41c0bc843a1c14229259/
    └── factor.py    # Computes "momentum_acceleration" factor
```

Each `factor.py` follows a standard interface:
- Reads `daily_pv.h5` via `pd.read_hdf()`
- Computes a single-column factor value per (datetime, instrument) pair
- Outputs a DataFrame with the same MultiIndex structure as `daily_pv.h5`

### Output Schema

```
combined_factors_df.parquet
├── Index: MultiIndex['datetime', 'instrument']
└── Columns: MultiIndex[('feature', feature_name)]
    ├── ('feature', 'KMID')          # Alpha158 feature 1
    ├── ('feature', 'KLEN')          # Alpha158 feature 2
    ├── ... (20 Alpha158 features)
    ├── ('feature', 'custom_factor_1')  # Custom factor
    └── ('feature', 'custom_factor_2')  # Custom factor
```

---

## Stage 4: Feature Matrix Construction

**Script:** `fullstackautoquant/data/build_features.py`  
**Input:** `combined_factors_df.parquet` + `task_rendered.yaml`  
**Output:** `features_ready_infer_YYYY-MM-DD.parquet`

### What It Does

Replicates the exact feature construction pipeline from the training snapshot:

1. **Reads DataHandler config** directly from `task_rendered.yaml` — the same YAML used during model training. This ensures the processor chain (fillna, CSZScoreNorm) and feature set are identical.

2. **Injects** `combined_factors_df.parquet` into the DataHandler, maintaining the same join strategy and label definitions as training.

3. **Extracts features** for the target trading day with appropriate fallback logic (uses nearest available date if market was closed).

4. **Validates** the output: no missing values, all features within normalized range [-3, 3], correct column ordering.

### Usage

```bash
python -m fullstackautoquant.data.build_features \
  --date auto \
  --combined_factors fullstackautoquant/data/combined_factors_df.parquet \
  --task-config configs/task_rendered.yaml \
  --provider_uri ~/.qlib/qlib_data/cn_data
```

### Processor Chain

The DataHandler applies these processors in order (defined in `task_rendered.yaml`):

| Step | Processor | Purpose |
|------|-----------|---------|
| 1 | `DropnaProcessor` | Remove rows with any NaN values |
| 2 | `CSZScoreNorm` | Cross-sectional z-score normalization per feature |

**CSZScoreNorm:** For each feature at each timestamp, compute:
```
z = (x - mean_cross_section) / std_cross_section
```
Then clip to [-3, 3]. This ensures all features are on a comparable scale regardless of their original units.

### Output Validation

The script performs three assertions before saving:

1. **No missing values** — Every cell must be filled
2. **Value range** — All values must be within [-3, 3] (post-normalization)
3. **Column count** — Must match the expected 22 features in the correct order

---

## Data Verification

Two verification scripts are provided for data integrity checks:

### verify_daily_pv_h5.py

```bash
python -m fullstackautoquant.data.verify.verify_daily_pv_h5 \
  --path fullstackautoquant/data/daily_pv.h5 \
  --require '$close'
```

Checks:
- HDF5 file structure (key='data' exists)
- MultiIndex levels are correct (`datetime`, `instrument`)
- Required columns are present (default: `$close`)
- No completely empty date slices
- Date range continuity

### verify_parquet_ready_for_infer.py

```bash
python -m fullstackautoquant.data.verify.verify_parquet_ready_for_infer \
  --path fullstackautoquant/data/combined_factors_df.parquet
```

Checks:
- Parquet file can be loaded
- MultiIndex structure matches expected format
- Column names include expected Alpha158 features
- Data coverage spans required date range
- No catastrophic data gaps

---

## Daily Update Schedule

The recommended daily pipeline runs sequentially:

```
09:00 CST  ──▶  qlib_update.sh           (~2 min)
              │
09:03 CST  ──▶  factor_synthesis.py       (~3 min)
              │
09:07 CST  ──▶  build_features.py         (~1 min)
              │
09:09 CST  ──▶  inference.py              (~1 min)
              │
09:10 CST  ──▶  Ready for trading
```

**Important timing notes:**
- A-share market data becomes available after 16:00 CST on each trading day
- If running inference for **today's** trading, the data pipeline should run **before market open** at 09:15 CST
- The pipeline uses the **previous day's** closing data to generate today's signals

---

## Data Formats Reference

| Artifact | Format | Typical Size | Key |
|----------|--------|-------------|-----|
| Qlib binary | `.bin` + `.today` | ~5 GB total | N/A |
| `daily_pv.h5` | HDF5 | ~500 MB | `data` |
| `combined_factors_df.parquet` | Parquet (Snappy) | ~200 MB | N/A |
| `features_ready_infer_*.parquet` | Parquet (Snappy) | ~1 MB | N/A |
| `ranked_scores.csv` | CSV | ~15 KB | N/A |

---

## Custom Factor Development

### Adding a New Factor

1. **Create a factor directory:**

```bash
mkdir -p fullstackautoquant/model/factors/my_custom_factor/
```

2. **Write `factor.py`:**

```python
"""My custom alpha factor."""
import pandas as pd

def compute_factor(daily_pv_path: str) -> pd.DataFrame:
    df = pd.read_hdf(daily_pv_path, key='data')
    # Compute your factor...
    result = df['$close'].pct_change(5)  # Example: 5-day momentum
    return result.to_frame('my_factor')
```

3. **Register** the factor in the synthesis pipeline (factor_synthesis.py will auto-discover it).

4. **Rebuild features:**

```bash
python -m fullstackautoquant.data.factor_synthesis
python -m fullstackautoquant.data.build_features --date auto
```

### Factor Conventions

- Each factor must output a DataFrame with `MultiIndex['datetime', 'instrument']`
- Factor values should be finite (no inf, minimal NaN)
- The normalization step (CSZScoreNorm) handles scaling automatically
- Use descriptive names — they appear in feature importance reports

---

## Source Files

| File | Description |
|------|-------------|
| [`data/qlib_update.sh`](../fullstackautoquant/data/qlib_update.sh) | Qlib data update script (Docker) |
| [`data/export_daily_pv.py`](../fullstackautoquant/data/export_daily_pv.py) | HDF5 export utility |
| [`data/factor_synthesis.py`](../fullstackautoquant/data/factor_synthesis.py) | Factor computation and merging |
| [`data/build_features.py`](../fullstackautoquant/data/build_features.py) | Feature matrix construction |
| [`data/verify/`](../fullstackautoquant/data/verify/) | Data verification scripts |
| [`model/factors/`](../fullstackautoquant/model/factors/) | Custom factor definitions |
| [`configs/task_rendered.yaml`](../configs/task_rendered.yaml) | Training config snapshot (DataHandler spec) |
