#!/usr/bin/env bash
# FullStackAutoQuant — End-to-End Inference Workflow
# Runs: Data Update → Export → Verify → Factor Synthesis → Inference
set -euo pipefail

log(){ printf "[%s] %s\n" "$(date +'%F %T')" "$*"; }

# Activate conda environment
if command -v conda >/dev/null 2>&1; then
  set +u
  eval "$(conda shell.bash hook)" >/dev/null 2>&1 || true
  conda activate rdagent4qlib >/dev/null 2>&1 || true
  set -u 2>/dev/null || true
fi

# Resolve project root (one level up from scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG_DIR="$PROJECT_ROOT/fullstackautoquant"

# Environment (must be set by user via .env or shell)
: "${TUSHARE:?ERROR: TUSHARE env var is required. Copy .env.example to .env and set your token.}"
PROVIDER_URI="$HOME/.qlib/qlib_data/cn_data"
DAILY_PV_PATH="$PKG_DIR/data/daily_pv.h5"

log "[1/5] Update Qlib data (Docker-based daily update + dump)"
bash "$PKG_DIR/data/qlib_update.sh"

log "[2/5] Export Qlib → daily_pv.h5 (all instruments)"
python "$PKG_DIR/data/export_daily_pv.py" \
  --end auto \
  --instruments all \
  --provider_uri "$PROVIDER_URI" \
  --region cn \
  --out "$DAILY_PV_PATH"

log "[3/5] Verify daily_pv.h5 structure and health"
python "$PKG_DIR/data/verify/verify_daily_pv_h5.py" --path "$DAILY_PV_PATH"

log "[4/5] Synthesize custom factors → combined_factors_df.parquet"
python "$PKG_DIR/data/factor_synthesis.py" --workspace "$PKG_DIR/model"

log "[5/5] Run inference (training-equivalent pipeline)"
PARAMS_FILE="$PROJECT_ROOT/weights/params.pkl"
if [[ -f "$PROJECT_ROOT/weights/state_dict_cpu.pt" ]]; then
  PARAMS_FILE="$PROJECT_ROOT/weights/state_dict_cpu.pt"
fi
python -m fullstackautoquant.model.inference \
  --date auto \
  --combined_factors "$PKG_DIR/model/combined_factors_df.parquet" \
  --params "$PARAMS_FILE" \
  --out "$PROJECT_ROOT/output/ranked_scores.csv"

log "Pipeline complete"
