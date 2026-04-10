#!/usr/bin/env bash
# deploy_dashboard.sh — Export dashboard JSON & push to GitHub Pages
#
# Workflow: export_dashboard_data.py → clone public gh-pages → sync → push
#
# Env vars (set in .env):
#   PUBLIC_REPO_URL  — Public repo HTTPS URL (default: auto-detected)
#   SKIP_EXPORT      — "1" to skip JSON export, use existing data
#
# Usage:
#   bash scripts/deploy_dashboard.sh               # full: export + deploy
#   SKIP_EXPORT=1 bash scripts/deploy_dashboard.sh  # deploy only (reuse existing JSON)
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { printf "${CYAN}[dashboard]${NC} %s\n" "$*"; }
ok()   { printf "${GREEN}[   OK    ]${NC} %s\n" "$*"; }
die()  { printf "${RED}[  FATAL  ]${NC} %s\n" "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DASHBOARD_SRC="$PROJECT_ROOT/dashboard"

# Default public repo URL (can be overridden via .env / env var)
PUBLIC_REPO_URL="${PUBLIC_REPO_URL:-https://github.com/Zizhao-HUANG/FullStackAutoQuant.git}"
DEPLOY_BRANCH="gh-pages"

# Load .env
[[ -f "$PROJECT_ROOT/.env" ]] && { set -a; source "$PROJECT_ROOT/.env"; set +a; }

# Ensure gh credential helper is active for HTTPS push
gh auth setup-git 2>/dev/null || true

# ── Phase A: Export dashboard JSON ──────────────────────────────
if [[ "${SKIP_EXPORT:-0}" == "1" ]]; then
  log "Export — skipped (SKIP_EXPORT=1)"
else
  log "Exporting dashboard JSON data..."

  # Activate conda if available (same as run_lite_workflow.sh)
  if command -v conda >/dev/null 2>&1; then
    set +u
    eval "$(conda shell.bash hook)" >/dev/null 2>&1 || true
    conda activate cloudspace >/dev/null 2>&1 || true
    set -u 2>/dev/null || true
  fi

  python "$SCRIPT_DIR/export_dashboard_data.py" \
    --out-dir "$DASHBOARD_SRC/data" \
    --pretty

  ok "Dashboard JSON exported → $DASHBOARD_SRC/data/"
fi

# Validate dashboard directory
[[ -d "$DASHBOARD_SRC" ]]                     || die "Dashboard directory not found: $DASHBOARD_SRC"
[[ -f "$DASHBOARD_SRC/index.html" ]]          || die "index.html missing in $DASHBOARD_SRC"
[[ -f "$DASHBOARD_SRC/data/dashboard_data.json" ]] || die "dashboard_data.json missing — run export first"

# ── Phase B: Deploy to GitHub Pages ─────────────────────────────
log "Deploying to $PUBLIC_REPO_URL ($DEPLOY_BRANCH branch)..."

WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

# Shallow clone just the gh-pages branch
git clone --depth 1 --branch "$DEPLOY_BRANCH" "$PUBLIC_REPO_URL" "$WORK_DIR/repo" 2>&1

cd "$WORK_DIR/repo"

# Configure git identity
git config user.name  "FullStackAutoQuant Bot"
git config user.email "bot@fullstackautoquant.local"

# Remove old site files (keep .git)
find . -maxdepth 1 ! -name '.git' ! -name '.' -exec rm -rf {} +

# Copy fresh dashboard files
cp -r "$DASHBOARD_SRC"/* .

# Check for actual changes
if git diff --quiet && git diff --cached --quiet && [[ -z "$(git ls-files --others --exclude-standard)" ]]; then
  log "No changes detected — dashboard already up to date"
  exit 0
fi

# Stage, commit, push
TODAY=$(date +'%Y-%m-%d')
DATA_DATE="unknown"
if command -v python3 >/dev/null 2>&1; then
  DATA_DATE=$(python3 -c "
import json, sys
try:
    d = json.load(open('data/dashboard_data.json'))
    print(d.get('signals',{}).get('date','$TODAY'))
except: print('$TODAY')
" 2>/dev/null || echo "$TODAY")
fi

git add -A
git commit -m "Dashboard update — signals ${DATA_DATE}

Auto-deployed by deploy_dashboard.sh
Export time: $(date +'%Y-%m-%d %H:%M:%S %Z')
Source: FullStackAutoQuant-private/dashboard/"

git push origin "$DEPLOY_BRANCH" --force-with-lease

ok "Dashboard deployed! Live at: https://zizhao-huang.github.io/FullStackAutoQuant/"
ok "Signal date: $DATA_DATE"
