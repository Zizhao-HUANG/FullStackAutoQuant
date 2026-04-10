#!/usr/bin/env bash
# FullStackAutoQuant — Lite Inference Workflow (Tushare-based)
# Lightweight alternative to run_full_workflow.sh: no Docker/Dolt needed.
# Runs: Tushare Fetch → Export → Verify → Factor Synthesis → Inference
set -euo pipefail

log(){ printf "[%s] %s\n" "$(date +'%F %T')" "$*"; }

# Step wrapper: prints label, runs command, reports elapsed time.
# If the command produced no stdout, appends "(cached)" hint.
step(){
  local label="$1"; shift
  log "$label"
  local t0; t0=$(date +%s)
  local tmp; tmp=$(mktemp)
  "$@" > >(tee "$tmp") 2>&1
  local elapsed=$(( $(date +%s) - t0 ))
  local note=""
  [[ ! -s "$tmp" ]] && note=" (cached)"
  rm -f "$tmp"
  log "  └─ ${elapsed}s${note}"
}

# Activate conda (gmtrade requires Python <=3.10; .venv may have incompatible Python)
if command -v conda >/dev/null 2>&1; then
  set +u
  eval "$(conda shell.bash hook)" >/dev/null 2>&1 || true
  conda activate cloudspace >/dev/null 2>&1 || true
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
NORM_CACHE="$PROJECT_ROOT/weights/norm_params.pkl"

# Validate norm cache exists (required since Tushare only fetches ~200 days)
if [[ ! -f "$NORM_CACHE" ]]; then
  log "ERROR: Normalizer cache not found: $NORM_CACHE"
  log "Run once: python scripts/extract_norm_cache.py"
  exit 1
fi

PARAMS_FILE="$PROJECT_ROOT/weights/params.pkl"
[[ -f "$PROJECT_ROOT/weights/state_dict_cpu.pt" ]] && PARAMS_FILE="$PROJECT_ROOT/weights/state_dict_cpu.pt"

step "[1/5] Fetch CSI300 data from Tushare (incremental)" \
  python "$PKG_DIR/data/tushare_provider.py" \
    --token "$TUSHARE" --n-days 200 --qlib-dir "$PROVIDER_URI"

step "[2/5] Export Qlib → daily_pv.h5 (CSI300)" \
  python "$PKG_DIR/data/export_daily_pv.py" \
    --end auto --instruments csi300 \
    --provider_uri "$PROVIDER_URI" --region cn --out "$DAILY_PV_PATH"

step "[3/5] Verify daily_pv.h5" \
  python "$PKG_DIR/data/verify/verify_daily_pv_h5.py" --path "$DAILY_PV_PATH"

step "[4/5] Synthesize factors → combined_factors_df.parquet" \
  python "$PKG_DIR/data/factor_synthesis.py" --workspace "$PKG_DIR/model"

step "[5/7] Run inference (cached normalizer)" \
  python -m fullstackautoquant.model.inference \
    --date auto \
    --combined_factors "$PKG_DIR/model/combined_factors_df.parquet" \
    --params "$PARAMS_FILE" --norm-cache "$NORM_CACHE" \
    --out "$PROJECT_ROOT/output/ranked_scores.csv"

step "[6/7] Push CSV to private repo" \
  bash "$SCRIPT_DIR/push_csv_to_repo.sh"

step "[7/7] Deploy dashboard to GitHub Pages" \
  bash "$SCRIPT_DIR/deploy_dashboard.sh"

log "Pipeline complete"
