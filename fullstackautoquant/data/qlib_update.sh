#!/usr/bin/env bash
set -euo pipefail

log(){ printf "[%s] %s\n" "$(date +'%F %T')" "$*"; }

# Validation-only mode:no download, only print local trading days
if [[ "${QLIB_UPDATE_NO_DOWNLOAD:-0}" == "1" ]]; then
  DEST_DIR="$HOME/.qlib/qlib_data/cn_data"
  if [[ -f "$DEST_DIR/calendars/day.txt" ]]; then
    log "Skipping download(QLIB_UPDATE_NO_DOWNLOAD=1)"
    last_day="$(tail -n 1 "$DEST_DIR/calendars/day.txt" | tr -d '\r\n')"
    log "Latest trading day: $last_day"
    log "Latest 5 trading days:"
    tail -n 5 "$DEST_DIR/calendars/day.txt"
    exit 0
  else
    log "Not found $DEST_DIR/calendars/day.txt, continuing download..."
  fi
fi

# Activate conda environment(avoid set -u interference)
if command -v conda >/dev/null 2>&1; then
  set +u
  eval "$(conda shell.bash hook)" >/dev/null 2>&1 || true
  conda activate rdagent4qlib >/dev/null 2>&1 || true
  set -u 2>/dev/null || true
fi

#============================
# Self-hosted data pipeline(automated wrapper for recommended installation steps)
# 1) One-time initialization:clone dolt repo inside container if no persistent copy exists
# 2) Daily update and export:run daily_update.sh + dump_qlib_bin.sh inside container
#============================

DEST_DIR="$HOME/.qlib/qlib_data/cn_data"
SELFHOST_BASE="$HOME/qlib_selfhost"
DOLT_DIR="$SELFHOST_BASE/dolt"
OUTPUT_DIR="$SELFHOST_BASE/output"
IMAGE_NAME="chenditc/investment_data"
ARCH="$(uname -m)"

# Tushare token:prefer env var, otherwise use preset token for reproducibility
: "${TUSHARE:?ERROR: TUSHARE env var is required. Get your token at https://tushare.pro}"

mkdir -p "$DEST_DIR" "$DOLT_DIR" "$OUTPUT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  log "ERROR: docker not found, please install Docker first." >&2
  exit 1
fi

log "Preparing Docker image:$IMAGE_NAME"
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    log "ARM architecture detected:enable amd64 emulation and pull image (may be slow first time)"
    docker run --privileged --rm tonistiigi/binfmt --install all >/dev/null 2>&1 || true
    docker pull --platform linux/amd64 "$IMAGE_NAME"
  else
    docker pull "$IMAGE_NAME"
  fi
fi

log "Container script orchestration (write and mount):daily_update + new Qlib export"
CONTAINER_SCRIPT="$SELFHOST_BASE/container_update_export.sh"
cat > "$CONTAINER_SCRIPT" <<'IN_CONTAINER'
#!/usr/bin/env bash
set -euo pipefail
echo "[container] Using TUSHARE: ${TUSHARE:+SET}"
# Globally suppress Python FutureWarning, keep progress bars
export PYTHONWARNINGS="ignore::FutureWarning"
# Limit parallel threads to reduce host memory pressure
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=2
mkdir -p /dolt /output /work

# 1) Dolt repo preparation(One-time initialization)
if [[ ! -d /dolt/investment_data/.dolt ]]; then
  echo "[container] Initializing dolt repo at /dolt/investment_data ..."
  dolt clone chenditc/investment_data /dolt/investment_data
else
  echo "[container] Found existing /dolt/investment_data"
fi

# 2) Incremental update (upstream scripts)
cd /investment_data || cd "$PWD"
echo "[container] RUN daily_update.sh"
set +e
bash daily_update.sh || echo "[container] WARN: daily_update failed, continue with existing dolt repo"
set -e

# 3) Export (new Qlib equivalent, with fallback plan B)

# 3.1 Start dolt sql-server with readiness probe
echo "[container] START dolt sql-server for export"
cd /dolt/investment_data
set +e
dolt sql-server > /tmp/dolt_server.log 2>&1 &
server_pid=$!
set -e

echo "[container] WAIT sql-server ready"
for i in $(seq 1 60); do
  if dolt sql -q "select 1" >/dev/null 2>&1; then
    echo "[container] sql-server ready"
    break
  fi
  sleep 1
done

# 3.2 Clone Qlib (scripts only)
if [[ ! -d /qlib/scripts ]]; then
  echo "[container] Cloning microsoft/qlib"
  git clone https://github.com/microsoft/qlib.git /qlib
fi
export PYTHONPATH="${PYTHONPATH:-}:/qlib/scripts"

# 3.3 Generate qlib_source (dump CSV from dolt)
cd /investment_data
mkdir -p ./qlib/qlib_source ./qlib/qlib_normalize
echo "[container] dump_all_to_qlib_source.py (log trimmed to /tmp/dump_source.log)"
try=0; until (python3 ./qlib/dump_all_to_qlib_source.py --max_workers=2 > /tmp/dump_source.log 2>&1 || \
               python3 ./qlib/dump_all_to_qlib_source.py > /tmp/dump_source.log 2>&1); do
  try=$((try+1)); if [[ $try -ge 3 ]]; then echo "[container] dump_all_to_qlib_source failed" >&2; kill $server_pid || true; exit 1; fi; sleep 3; done
files_dumped=$(grep -c '^Dumping to file:' /tmp/dump_source.log 2>/dev/null || true)
echo "[container] dump_source files: ${files_dumped:-0}"
tail -n 5 /tmp/dump_source.log 2>/dev/null || true

# 3.4 Normalize
echo "[container] normalize.py"
try=0; until python3 ./qlib/normalize.py normalize_data \
  --source_dir ./qlib/qlib_source/ \
  --normalize_dir ./qlib/qlib_normalize \
  --max_workers=4 \
  --date_field_name="tradedate"; do
  try=$((try+1)); if [[ $try -ge 3 ]]; then echo "[container] normalize failed" >&2; kill $server_pid || true; exit 1; fi; sleep 3; done

# 3.5 dump_bin(New CLI: --data_path)
echo "[container] dump_bin.py (--data_path)"
try=0; until python3 /qlib/scripts/dump_bin.py dump_all \
  --data_path ./qlib/qlib_normalize/ \
  --qlib_dir /work/qlib_bin \
  --date_field_name=tradedate \
  --exclude_fields=tradedate,symbol 2> >(grep -v -E "FutureWarning|deprecated" >&2); do
  try=$((try+1)); if [[ $try -ge 3 ]]; then echo "[container] dump_bin failed" >&2; kill $server_pid || true; exit 1; fi; sleep 3; done

# 3.6 Append index and trading calendar
mkdir -p /work/qlib_bin/instruments /investment_data/qlib/qlib_index
python3 ./qlib/dump_index_weight.py || true
python3 ./tushare/dump_day_calendar.py /work/qlib_bin/ || true
cp -f ./qlib/qlib_index/csi* /work/qlib_bin/instruments/ 2>/dev/null || true
if [[ ! -f /work/qlib_bin/instruments/csi300.txt && -f ./qlib/qlib_index/csi300.txt ]]; then
  cp -f ./qlib/qlib_index/csi300.txt /work/qlib_bin/instruments/
fi

# 3.7 Stop sql-server
kill $server_pid || true

# 3.8 Package and assert
cd /work
tar -czf /investment_data/qlib_bin.tar.gz qlib_bin
if [[ ! -s /investment_data/qlib_bin.tar.gz ]]; then
  echo "[container] ERROR: qlib_bin.tar.gz not created" >&2
  exit 1
fi
cp /investment_data/qlib_bin.tar.gz /output/qlib_bin.tar.gz
if [[ ! -s /output/qlib_bin.tar.gz ]]; then
  echo "[container] ERROR: /output/qlib_bin.tar.gz missing or empty" >&2
  exit 1
fi
echo "[container] EXPORT DONE: /output/qlib_bin.tar.gz"
IN_CONTAINER

chmod +x "$CONTAINER_SCRIPT"

if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
  docker run --platform linux/amd64 --rm \
    -e TUSHARE="$TUSHARE" \
    -v "$DOLT_DIR":/dolt \
    -v "$OUTPUT_DIR":/output \
    -v "$SELFHOST_BASE":/host \
    "$IMAGE_NAME" \
    bash -lc 'set -euo pipefail; sed -i "s/\r$//" /host/container_update_export.sh 2>/dev/null || true; bash /host/container_update_export.sh' || true
else
  docker run --rm \
  -e TUSHARE="$TUSHARE" \
  -v "$DOLT_DIR":/dolt \
  -v "$OUTPUT_DIR":/output \
  -v "$SELFHOST_BASE":/host \
  "$IMAGE_NAME" \
  bash -lc 'set -euo pipefail; sed -i "s/\r$//" /host/container_update_export.sh 2>/dev/null || true; bash /host/container_update_export.sh' || true
fi

# If container export failed, fallback: repackage existing local Qlib data as qlib_bin.tar.gz
if [[ ! -s "$OUTPUT_DIR/qlib_bin.tar.gz" ]]; then
  if [[ -d "$DEST_DIR" ]]; then
    log "WARNING: Container export failed, using existing local Qlib data to continue"
    tmp_pack_dir="$(mktemp -d)"
    mkdir -p "$tmp_pack_dir/qlib_bin"
    # Prefer rsync; fallback to cp -a
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "$DEST_DIR"/ "$tmp_pack_dir/qlib_bin"/
    else
      cp -a "$DEST_DIR"/. "$tmp_pack_dir/qlib_bin"/
    fi
    tar -czf "$OUTPUT_DIR/qlib_bin.tar.gz" -C "$tmp_pack_dir" qlib_bin || true
    rm -rf "$tmp_pack_dir"
  fi
fi

if [[ ! -s "$OUTPUT_DIR/qlib_bin.tar.gz" ]]; then
  log "ERROR: Export failed, not found in $OUTPUT_DIR found qlib_bin.tar.gz" >&2
  exit 1
fi

log "Extracting qlib_bin.tar.gz to $DEST_DIR ..."
tar -zxf "$OUTPUT_DIR/qlib_bin.tar.gz" -C "$DEST_DIR" --strip-components=1

if [[ -f "$DEST_DIR/calendars/day.txt" ]]; then
  last_day="$(tail -n 1 "$DEST_DIR/calendars/day.txt" | tr -d '\r\n')"
  log "Latest trading day: $last_day"
  log "Latest 5 trading days:"
  tail -n 5 "$DEST_DIR/calendars/day.txt"
else
  log "WARNING: After extraction, not found calendars/day.txt" >&2
fi

log "Done"
