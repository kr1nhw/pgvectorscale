#!/bin/bash
# One clean experiment cycle for hnswsq (or pgvector) on a benchmark database.
#
# Usage: cycle.sh <engine:hnsw|hnswsq> <db> <label> [n]
#
#   engine  hnswsq -> build items_<n>_hnswsq (storage_layout driven by
#                     HNSWSQ_LAYOUT, default plain) and sweep/serve it
#           hnsw    -> pgvector baseline on the same table
#   n       dataset tag used in table names (default 100k => items_100k),
#           with the table looked up as items_<n> and the ground truth as
#           gt_<n> (falling back to gt_10m names for the older dbs)
#
# What it guarantees (each of these bit us before):
#   * stale CREATE INDEX backends are killed BY PID before the run — a leftover
#     build silently doubles the wall clock and invalidates the numbers;
#   * the index is dropped and recreated, so page reuse cannot skew the build;
#   * the build is timed wall-clock; the optional "hnswsq build stats: ..."
#     phase split only logs when the extension is built with pg_test;
#   * build + sweep + insert all land in one CSV row per run, plus a full log.
#
# Environment: PGPORT (54329), PGUSER (pgtest), PGDATABASE, PSQL_BIN,
#              OUT_CSV (/tmp/hnswsq_cycle.csv), HNSWSQ_LAYOUT (plain),
#              INSERT_N (100000), STAGING_FILE, BATCH (1000)
set -uo pipefail

ENGINE="${1:?engine}"
DB="${2:?db}"
LABEL="${3:?label}"
N="${4:-100k}"

PORT="${PGPORT:-54329}"
USER="${PGUSER:-pgtest}"
PSQL="${PSQL_BIN:-/root/.pgrx-hnswsq/17.11/pgrx-install/bin/psql}"
STAGING_FILE="${STAGING_FILE:-/data1/bigann_100m_vectors.txt}"
LAYOUT="${HNSWSQ_LAYOUT:-plain}"
OUT_CSV="${OUT_CSV:-/tmp/hnswsq_cycle.csv}"
LOG="${LOG:-/tmp/hnswsq_cycle_$(date +%Y%m%d_%H%M%S).log}"
INSERT_N="${INSERT_N:-100000}"

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
TABLE="items_${N}"
GT="gt_${N}"

run_psql() { sudo -u "$USER" "$PSQL" -h 127.0.0.1 -p "$PORT" -U "$USER" -d "$DB" -X -q -v ON_ERROR_STOP=1 "$@"; }

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "cycle start engine=$ENGINE db=$DB label=$LABEL n=$N layout=$LAYOUT"

# ---- 0. the installed extension must be a RELEASE build --------------------
# `cargo pgrx test` installs a debug build over the release one; that made every
# build ~25x slower and silently invalidated a whole analysis pass.
SO=$(ls "$(dirname "$PSQL")/../lib/postgresql"*/vectorscale-*.so 2>/dev/null | head -1)
if [ -n "$SO" ]; then
  SO_KB=$(( $(stat -c %s "$SO" 2>/dev/null || stat -f %z "$SO") / 1024 ))
  log "installed extension: $SO (${SO_KB}KB)"
  if [ "$SO_KB" -gt 8000 ]; then
    log "FATAL: that looks like a debug build; reinstall with:"
    log "  cargo pgrx install --release --pg-config \$(dirname $PSQL)/pg_config --no-default-features --features pg17"
    exit 1
  fi
fi

# ---- 0b. dataset sanity ----------------------------------------------------
ROWS=$(run_psql -At -c "SELECT count(*) FROM $TABLE;" 2>/dev/null)
GT_ROWS=$(run_psql -At -c "SELECT count(*) FROM $GT;" 2>/dev/null)
QROWS=$(run_psql -At -c "SELECT count(*) FROM bench_queries;" 2>/dev/null)
log "dataset rows=$ROWS gt=$GT_ROWS queries=$QROWS"
if [ -z "$ROWS" ] || [ "$ROWS" = "0" ]; then
  log "FATAL: table $TABLE is missing or empty"
  exit 1
fi

# ---- 1. kill stale index builds (by PID, the only reliable way) ------------
STALE=$(run_psql -At -c "SELECT string_agg(pid::text, ',') FROM pg_stat_activity
                          WHERE pid <> pg_backend_pid()
                            AND (query ILIKE '%CREATE INDEX%' OR query ILIKE '%VACUUM%');" 2>/dev/null)
if [ -n "$STALE" ] && [ "$STALE" != "" ]; then
  log "killing stale backends: $STALE"
  run_psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity
               WHERE pid IN ($STALE);" >/dev/null 2>&1
  sleep 2
fi
ACTIVE=$(run_psql -At -c "SELECT count(*) FROM pg_stat_activity
                          WHERE pid <> pg_backend_pid() AND state <> 'idle';" 2>/dev/null)
log "active backends after cleanup: $ACTIVE"
if [ "$ACTIVE" != "0" ]; then
  log "WARNING: $ACTIVE background session(s) still active (vacuum?) — numbers may be noisy"
fi

# ---- 2. build (timed, with stats) -----------------------------------------
IDX="${TABLE}_${ENGINE}"
run_psql -c "DROP INDEX IF EXISTS $IDX;" >/dev/null 2>&1

if [ "$ENGINE" = hnswsq ]; then
  DDL="CREATE INDEX $IDX ON $TABLE USING hnswsq (embedding vector_l2_ops)
       WITH (m=16, ef_construction=64, storage_layout=$LAYOUT);"
  PRE="SET maintenance_work_mem = '8GB';"
else
  DDL="CREATE INDEX $IDX ON $TABLE USING hnsw (embedding vector_l2_ops)
       WITH (m=16, ef_construction=64);"
  PRE="SET maintenance_work_mem = '8GB';"
fi

START=$(date +%s)
run_psql -c "$PRE" -c "$DDL" >>"$LOG" 2>&1
RC=$?
END=$(date +%s)
BUILD_S=$((END - START))
if [ $RC -ne 0 ]; then
  log "FATAL: build failed (see $LOG)"
  exit 1
fi
STATS=$(grep -o "hnswsq build stats.*" "$LOG" | tail -1 | tr -d '\r' || true)
SIZE=$(run_psql -At -c "SELECT pg_relation_size('$IDX');" 2>/dev/null)
log "build_s=$BUILD_S size_bytes=$SIZE"
[ -n "$STATS" ] && log "$STATS"

# ---- 3. recall + latency sweep --------------------------------------------
SWEEP_CSV="$(mktemp)"
TABLE="$TABLE" GT="$GT" PGPORT="$PORT" PGUSER="$USER" PGHOST=127.0.0.1 \
PGDATABASE="$DB" \
  bash "$BENCH_DIR/run_sweep_hnswsq.sh" "$PSQL" "$ENGINE" "$LABEL" "$SWEEP_CSV" >>"$LOG" 2>&1
SWEEP_RC=$?
log "sweep rc=$SWEEP_RC"
cat "$SWEEP_CSV" | tee -a "$LOG"

# ---- 4. incremental insert -------------------------------------------------
INSERT_CSV="$(mktemp)"
PGPORT="$PORT" PGUSER="$USER" PSQL_BIN="$PSQL" TABLE="$TABLE" \
STAGING_FILE="$STAGING_FILE" N="$INSERT_N" BATCH="${BATCH:-1000}" \
  bash "$BENCH_DIR/bench_insert.sh" "$PSQL" "$ENGINE" "$DB" "$LABEL" "$INSERT_CSV" >>"$LOG" 2>&1
INSERT_RC=$?
log "insert rc=$INSERT_RC"
cat "$INSERT_CSV" | tee -a "$LOG"

# ---- 5. one CSV row per run ------------------------------------------------
RUN_TS=$(date '+%Y-%m-%d %H:%M:%S')
if [ ! -f "$OUT_CSV" ]; then
  echo "ts,label,engine,layout,dataset,rows,build_s,size_bytes,insert_rows_per_s,insert_row_ms_mean,insert_row_ms_p50,insert_row_ms_p99,recall_40,p50_40,sweep_rows" > "$OUT_CSV"
fi
REC40=$(awk -F, '$4==40 {print $5}' "$SWEEP_CSV" 2>/dev/null | tail -1)
P5040=$(awk -F, '$4==40 {print $6}' "$SWEEP_CSV" 2>/dev/null | tail -1)
INS_RPS=$(awk -F, '{print $6}' "$INSERT_CSV" 2>/dev/null | tail -1)
INS_MEAN=$(awk -F, '{print $7}' "$INSERT_CSV" 2>/dev/null | tail -1)
INS_P50=$(awk -F, '{print $8}' "$INSERT_CSV" 2>/dev/null | tail -1)
INS_P99=$(awk -F, '{print $9}' "$INSERT_CSV" 2>/dev/null | tail -1)
SWEEP_ROWS=$(wc -l < "$SWEEP_CSV" | tr -d ' ')
echo "$RUN_TS,$LABEL,$ENGINE,$LAYOUT,$N,$ROWS,$BUILD_S,$SIZE,$INS_RPS,$INS_MEAN,$INS_P50,$INS_P99,$REC40,$P5040,$SWEEP_ROWS" >> "$OUT_CSV"

rm -f "$SWEEP_CSV" "$INSERT_CSV"
log "cycle done -> $OUT_CSV"
echo "--- cycle summary ---"
tail -1 "$OUT_CSV"
[ -n "$STATS" ] && echo "$STATS"
