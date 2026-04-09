#!/usr/bin/env bash
# Universal trading component — works with any number of GM accounts.
# Calls existing modules: signals.py → risk/manager.py → strategy.py → execution.py
#
# Usage:
#   bash scripts/run_trading.sh --csv output/ranked_scores.csv --accounts "id1,id2" [--place]
#   bash scripts/run_trading.sh --csv output/ranked_scores.csv --config configs/trading.yaml \
#       --accounts "id1,id2,id3" --place --auction-mode
set -euo pipefail

log(){ printf "[%s] %s\n" "$(date +'%F %T')" "$*"; }

# ── Parse args ───────────────────────────────────────────────────
CSV="" CONFIG="" ACCOUNTS="" SRC="sina" MAX_SLICES="1" OPEN_EPS="0.0025"
PLACE="" AUCTION="" OVERRIDE_BUY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv)           CSV="$2";                      shift 2 ;;
    --config)        CONFIG="$2";                    shift 2 ;;
    --accounts)      ACCOUNTS="$2";                  shift 2 ;;
    --place)         PLACE="--place";                shift ;;
    --src)           SRC="$2";                       shift 2 ;;
    --auction-mode)  AUCTION="--auction-mode";       shift ;;
    --override-buy)  OVERRIDE_BUY="--override_buy";  shift ;;
    --max-slices)    MAX_SLICES="$2";                shift 2 ;;
    --open-eps)      OPEN_EPS="$2";                  shift 2 ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

[[ -z "$CSV" ]]      && { echo "ERROR: --csv required" >&2; exit 1; }
[[ -z "$ACCOUNTS" ]] && { echo "ERROR: --accounts required" >&2; exit 1; }
[[ ! -f "$CSV" ]]    && { echo "ERROR: $CSV not found" >&2; exit 1; }

# ── Setup ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRADING="$PROJECT_ROOT/fullstackautoquant/trading"
LOGS="logs"; mkdir -p "$LOGS"
CFG_ARGS=""; [[ -n "$CONFIG" ]] && CFG_ARGS="--config $CONFIG"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
# Force unbuffered Python output so logs appear in real-time
export PYTHONUNBUFFERED=1

# Activate conda (gmtrade requires Python <=3.10; .venv may have incompatible Python)
if command -v conda >/dev/null 2>&1; then
  set +u; eval "$(conda shell.bash hook)" 2>/dev/null; conda activate cloudspace 2>/dev/null; set -u 2>/dev/null || true
fi

IFS=',' read -ra ACCTS <<< "$ACCOUNTS"
log "Trading: ${#ACCTS[@]} accounts, mode=${PLACE:-DRY-RUN}, csv=$CSV"

# ── Shared: signals + risk (once) ────────────────────────────────
SIG="$LOGS/signals_AUTO.json"
RISK="$LOGS/risk_state_AUTO.json"

log "[shared] Parsing signals"
python "$TRADING/signals.py" --csv "$CSV" --out "$SIG" $CFG_ARGS

log "[shared] Evaluating risk"
python "$TRADING/risk/manager.py" --signals "$SIG" --out "$RISK" $CFG_ARGS $OVERRIDE_BUY

# ── Per-account: strategy + execution ────────────────────────────
FAILS=0
for i in "${!ACCTS[@]}"; do
  ID="${ACCTS[$i]}"
  N=$((i+1))
  TAG="${ID:0:8}"
  log "[${N}/${#ACCTS[@]}] Account $TAG…"

  python "$TRADING/strategy.py" \
    --signals "$SIG" --risk_state "$RISK" \
    --targets "$LOGS/targets_${TAG}.json" --orders "$LOGS/orders_${TAG}.json" \
    --account-id "$ID" $CFG_ARGS \
  && python "$TRADING/execution.py" \
    --orders "$LOGS/orders_${TAG}.json" --src "$SRC" --account-id "$ID" \
    --max_slices_open "$MAX_SLICES" --open_eps "$OPEN_EPS" \
    $CFG_ARGS $AUCTION $PLACE \
  && log "  ✓ $TAG done" \
  || { log "  ✗ $TAG FAILED"; FAILS=$((FAILS+1)); }
done

log "Result: $((${#ACCTS[@]}-FAILS))/${#ACCTS[@]} passed"
if [[ $FAILS -gt 0 ]]; then exit 1; fi
