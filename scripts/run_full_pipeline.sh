#!/usr/bin/env bash
# Full Pipeline: Inference → Trading (multi-account)
# Chains run_lite_workflow.sh + run_trading.sh
#
# Env vars (set in .env, which is gitignored):
#   TUSHARE          — Tushare API token (required)
#   GM_TOKEN         — GM API token (required for live trading)
#   TRADE_ACCOUNTS   — comma-separated GM account IDs (required for trading)
#   TRADE_DRY_RUN    — "1" for dry-run (no orders placed)
#   SKIP_INFERENCE   — "1" to skip inference, use existing CSV
#   SKIP_TRADING     — "1" to skip trading
#   SKIP_DASHBOARD   — "1" to skip dashboard export & deploy
set -euo pipefail

log(){ printf "[%s] %s\n" "$(date +'%F %T')" "$*"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CSV="$PROJECT_ROOT/output/ranked_scores.csv"
CONFIG="$PROJECT_ROOT/configs/trading.yaml"

# Load .env (gitignored — contains secrets)
[[ -f "$PROJECT_ROOT/.env" ]] && { set -a; source "$PROJECT_ROOT/.env"; set +a; }

# ── Phase 1: Inference ──────────────────────────────────────────
if [[ "${SKIP_INFERENCE:-0}" == "1" ]]; then
  log "[Phase 1] Inference — skipped"
  [[ ! -f "$CSV" ]] && { log "ERROR: $CSV not found"; exit 1; }
else
  log "[Phase 1] Inference"
  bash "$SCRIPT_DIR/run_lite_workflow.sh"
fi

# ── Phase 2: Trading ────────────────────────────────────────────
if [[ "${SKIP_TRADING:-0}" == "1" ]]; then
  log "[Phase 2] Trading — skipped"
elif [[ -z "${TRADE_ACCOUNTS:-}" ]]; then
  log "[Phase 2] Trading — skipped (TRADE_ACCOUNTS not set in .env)"
else
  ARGS=(--csv "$CSV" --accounts "$TRADE_ACCOUNTS" --src sina)
  [[ -f "$CONFIG" ]] && ARGS+=(--config "$CONFIG")
  [[ "${TRADE_DRY_RUN:-0}" != "1" ]] && ARGS+=(--place)

  log "[Phase 2] Trading (${TRADE_DRY_RUN:+dry-run}${TRADE_DRY_RUN:-live})"
  bash "$SCRIPT_DIR/run_trading.sh" "${ARGS[@]}"
fi

# ── Phase 3: Dashboard ──────────────────────────────────────────
if [[ "${SKIP_DASHBOARD:-0}" == "1" ]]; then
  log "[Phase 3] Dashboard — skipped"
else
  log "[Phase 3] Dashboard export & deploy to GitHub Pages"
  bash "$SCRIPT_DIR/deploy_dashboard.sh"
fi

log "Pipeline complete"
