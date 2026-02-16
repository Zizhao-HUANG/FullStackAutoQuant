# Deployment Guide

> Step-by-step instructions for deploying FullStackAutoQuant in development and production environments.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Configuration](#environment-configuration)
- [Data Initialization](#data-initialization)
- [Model Weights](#model-weights)
- [First Inference Run](#first-inference-run)
- [WebUI Launch](#webui-launch)
- [Automated Daily Workflow](#automated-daily-workflow)
- [Production Deployment Checklist](#production-deployment-checklist)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | macOS 12+ / Ubuntu 20.04+ | macOS 14+ / Ubuntu 22.04+ |
| Python | 3.10 | 3.10–3.12 |
| RAM | 8 GB | 16 GB |
| Disk | 10 GB free | 30 GB free (Qlib data) |
| GPU | Not required | CUDA-capable GPU (optional) |

### External Services

| Service | Required? | Purpose | How to Obtain |
|---------|-----------|---------|---------------|
| **Tushare API** | ✅ Yes | Market data, real-time quotes | [tushare.pro](https://tushare.pro) — free tier available |
| **Docker** | ✅ Yes (for data update) | Qlib data pipeline | [docker.com](https://docker.com) |
| **JoinQuant GM Trade** | ⚠️ Live trading only | Order execution API | [joinquant.com](https://www.joinquant.com) |

---

## Installation

### Option 1: Editable Install (Development)

```bash
# Clone the repository
git clone https://github.com/Zizhao-HUANG/FullStackAutoQuant.git
cd FullStackAutoQuant

# Create and activate a virtual environment (recommended)
python3.10 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -e .

# Or install everything (including trading, WebUI, and dev tools)
pip install -e ".[all]"
```

### Option 2: Minimal Install (Inference Only)

```bash
pip install -e .
# This installs: torch, numpy, pandas, pyqlib, pyyaml, jsonschema, h5py, tushare
```

### Optional Dependency Groups

| Group | Command | What it Adds |
|-------|---------|--------------|
| `trading` | `pip install -e ".[trading]"` | GM Trade API (`gmtrade`) |
| `webui` | `pip install -e ".[webui]"` | Streamlit dashboard |
| `dev` | `pip install -e ".[dev]"` | pytest, ruff, mypy, black, isort |
| `all` | `pip install -e ".[all]"` | Everything above |

### Verify Installation

```bash
python -c "import fullstackautoquant; print('OK')"
make test  # Run test suite
```

---

## Environment Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# REQUIRED: Tushare API token
# Get yours at: https://tushare.pro/register
TUSHARE=your_tushare_token_here

# OPTIONAL: JoinQuant GM Trade (live trading only)
GM_ENDPOINT=              # e.g., https://api.myquant.cn
GM_TOKEN=                 # Your GM API token
GM_ACCOUNT_ID=            # Your trading account ID

# OPTIONAL: Custom Qlib data directory
# QLIB_DATA_DIR=~/.qlib/qlib_data/cn_data
```

> **⚠️ Security:** Never commit `.env` to version control. It is already in `.gitignore`.

### Trading Configuration

For trading-specific settings, edit the YAML config:

```bash
cp configs/trading.yaml.example configs/trading.auto.local.yaml
```

This file controls:
- Portfolio parameters (top-K, weight limits, lot sizes)
- Risk thresholds (drawdown limits, confidence floors)
- Order execution settings (slippage, commission rates)
- File paths (data locations, log directories)

See [Trading System](trading_system.md) for detailed configuration reference.

---

## Data Initialization

### Step 1: Update Qlib Market Data

The data pipeline uses Docker to fetch and process A-share market data:

```bash
bash fullstackautoquant/data/qlib_update.sh
```

This script:
1. Pulls the `chenditc/investment_data` Docker image
2. Downloads and converts A-share daily OHLCV data to Qlib binary format
3. Stores data in `~/.qlib/qlib_data/cn_data/` (default)

Expected runtime: **5–15 minutes** (first run), **1–3 minutes** (daily updates).

### Step 2: Export Daily Price Volume (Optional)

If your setup requires HDF5 format for local factor computation:

```bash
python -m fullstackautoquant.data.export_daily_pv \
  --provider_uri ~/.qlib/qlib_data/cn_data \
  --out fullstackautoquant/data/daily_pv.h5
```

### Step 3: Synthesize Combined Factors

Build the factor matrix that combines Alpha158 features with custom factors:

```bash
python -m fullstackautoquant.data.factor_synthesis \
  --provider_uri ~/.qlib/qlib_data/cn_data \
  --out fullstackautoquant/data/combined_factors_df.parquet
```

### Step 4: Verify Data Integrity

```bash
# Verify HDF5 structure
python -m fullstackautoquant.data.verify.verify_daily_pv_h5 \
  --path fullstackautoquant/data/daily_pv.h5

# Verify parquet file is inference-ready
python -m fullstackautoquant.data.verify.verify_parquet_ready_for_infer \
  --path fullstackautoquant/data/combined_factors_df.parquet
```

---

## Model Weights

Pre-trained model weights are distributed via GitHub Releases (not tracked in the git repository).

### Download

```bash
# Download from GitHub Releases
wget https://github.com/Zizhao-HUANG/FullStackAutoQuant/releases/download/v0.1.0/params.pkl \
  -O weights/params.pkl

wget https://github.com/Zizhao-HUANG/FullStackAutoQuant/releases/download/v0.1.0/state_dict_cpu.pt \
  -O weights/state_dict_cpu.pt
```

### Verify

```bash
ls -la weights/
# Expected:
#   params.pkl          ~3.1 MB  (full Qlib model object)
#   state_dict_cpu.pt   ~3.1 MB  (PyTorch state dict, CPU)
#   README.md                    (download instructions)
#   .gitkeep
```

---

## First Inference Run

Run inference on the latest trading day:

```bash
python -m fullstackautoquant.model.inference \
  --date auto \
  --combined_factors fullstackautoquant/data/combined_factors_df.parquet \
  --params weights/params.pkl \
  --out output/ranked_scores.csv
```

Or use the Makefile shortcut:

```bash
make inference
```

### Expected Output

```
==== Model Inference (Qlib) ====
Target trading day: 2024-12-31
#Instruments: 300
#Features: 22
MC Dropout passes: 16
Output: output/ranked_scores.csv
```

The output CSV contains:
```csv
instrument,score,confidence,rank
SH600519,0.0342,0.97,1
SZ000858,0.0298,0.95,2
...
```

---

## WebUI Launch

Start the Streamlit dashboard for portfolio management:

```bash
make webui
# Or directly:
streamlit run fullstackautoquant/webui/app/streamlit_app.py
```

The WebUI provides:
- **Research & Inference** — Run model inference and inspect signal history
- **Manual Trading Console** — Execute daily plans with order review
- **Position Manager** — Import, edit, and track portfolio positions
- **Backtest** — Configure and run historical backtests with performance charts
- **Risk Dashboard** — Monitor drawdowns, limit states, and risk metrics
- **History & Logs** — View snapshots and audit trails
- **System Config** — Edit trading parameters in-browser

---

## Automated Daily Workflow

### Using cron (Linux/macOS)

Create a daily workflow that runs before market open (e.g., 9:00 AM CST):

```bash
crontab -e
```

Add:

```cron
# FullStackAutoQuant daily pipeline — runs at 09:00 Beijing time (01:00 UTC)
0 1 * * 1-5 cd /path/to/FullStackAutoQuant && bash scripts/daily_pipeline.sh >> logs/cron.log 2>&1
```

### Using launchd (macOS)

Create `~/Library/LaunchAgents/com.fullstackautoquant.daily.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.fullstackautoquant.daily</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/path/to/FullStackAutoQuant/scripts/daily_pipeline.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>WorkingDirectory</key>
    <string>/path/to/FullStackAutoQuant</string>
    <key>StandardOutPath</key>
    <string>/path/to/FullStackAutoQuant/logs/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>/path/to/FullStackAutoQuant/logs/launchd.err</string>
</dict>
</plist>
```

Load the schedule:

```bash
launchctl load ~/Library/LaunchAgents/com.fullstackautoquant.daily.plist
```

### Daily Pipeline Steps

The automated workflow executes these steps in sequence:

```
1. Data Update     → qlib_update.sh          (fetch latest market data)
2. Factor Synthesis → factor_synthesis.py      (build combined factors)
3. Feature Build    → build_features.py        (construct feature matrix)
4. Inference        → inference.py             (generate ranked scores)
5. Risk Evaluation  → risk/manager.py          (compute drawdowns, limits)
6. Strategy         → strategy.py              (generate buy/sell orders)
7. Execution        → execution.py             (place orders via GM Trade)
```

---

## Production Deployment Checklist

Before running live trading, verify each item:

- [ ] **Credentials:** All API tokens in `.env` are valid and tested
- [ ] **Data pipeline:** `qlib_update.sh` completes without errors
- [ ] **Factor synthesis:** `combined_factors_df.parquet` is up to date
- [ ] **Model weights:** `weights/params.pkl` exists and loads correctly
- [ ] **Inference test:** `make inference` produces valid `ranked_scores.csv`
- [ ] **Risk thresholds:** Reviewed and configured in `trading.auto.local.yaml`
- [ ] **GM Trade:** Test order placement with `--dry-run` flag first
- [ ] **Monitoring:** Cron/launchd job is installed and logging to file
- [ ] **Backup:** NAV history and trade logs are persisted to stable storage

---

## Troubleshooting

### Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError: No module named 'qlib'` | Qlib not installed | `pip install pyqlib>=0.9.0` |
| `FileNotFoundError: Cannot find model weights` | Missing params.pkl | Download from GitHub Releases |
| `RuntimeError: CUDA unavailable` | No GPU / CUDA not installed | System auto-falls back to CPU (no action needed) |
| `Docker: permission denied` | Docker not running or user not in docker group | Start Docker Desktop or `sudo usermod -aG docker $USER` |
| `qlib_update.sh: timeout` | Slow network / Docker pull | Retry; consider using a mirror |
| `Features have missing values` | Qlib data not up to date | Run `qlib_update.sh` first |
| `ConfigError: Missing schema` | Incomplete installation | Ensure `configs/schema/` directory exists |
| `GM Trade: authentication failed` | Invalid credentials | Check `GM_TOKEN` and `GM_ACCOUNT_ID` in `.env` |

### Checking Logs

```bash
# Inference logs
ls logs/

# CRON output
tail -f logs/cron.log

# WebUI logs
streamlit run ... 2>&1 | tee logs/webui.log
```

### Getting Help

- **Issues:** [GitHub Issues](https://github.com/Zizhao-HUANG/FullStackAutoQuant/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Zizhao-HUANG/FullStackAutoQuant/discussions)
