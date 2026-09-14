#!/bin/bash
# Local (single-host) hnswsq build + recall cycle on the development scratch
# cluster, for the per-phase perf gates in .design/hnswsq_perf_analysis.md.
#
# Usage: local_cycle.sh <label> [layout] [db] [tag] [m] [ef_construction] [dim]
#
#   label            free-text tag recorded in the CSV row
#   layout           storage_layout (default plain)
#   db               database (default t100kdb, see local_dataset.sql)
#   tag              dataset tag: table t<tag>, ground truth gt_<tag>,
#                    queries bench_queries_<dim> (default 100k; use e.g.
#                    100kd128 for the dim 128 kernel datasets)
#   m / efc          index parameters (default 16 / 64)
#   dim              query-table dimension (default 16)
#
# What it guarantees (each of these bit us before):
#   * the installed extension is a RELEASE build — `cargo pgrx test` overwrites
#     the installed .so with a debug one, which silently makes every build ~25x
#     slower (this invalidated a whole analysis pass once);
#   * stale CREATE INDEX backends are killed by PID before the run, so a
#     leftover build cannot silently double the wall clock;
#   * the index is dropped and recreated, so page reuse cannot skew the build;
#   * the build runs with hnswsq.build_stats = on, so the phase split lands next
#     to the timing;
#   * build + recall sweep both go into one CSV row, plus a per-run log.
#
# Environment: PGPORT (54331), PGUSER ($USER), PSQL_BIN, PGBIN, OUT_CSV
#              (/tmp/hnswsq_local_cycle.csv), HNSWSQ_EF_SWEEP, MAINT_MEM
set -uo pipefail

LABEL="${1:?label}"
LAYOUT="${2:-plain}"
DB="${3:-t100kdb}"
TAG="${4:-100k}"
M="${5:-16}"
EFC="${6:-64}"
DIM="${7:-16}"

PORT="${PGPORT:-54331}"
USER="${PGUSER:-$USER}"
PGBIN="${PGBIN:-/opt/homebrew/opt/postgresql@18/bin}"
PSQL="${PSQL_BIN:-$PGBIN/psql}"
MAINT_MEM="${MAINT_MEM:-2GB}"
# Pin the level RNG: -1 (the default) seeds from entropy, so two builds differ
# in graph quality and recall@10 moves by several points between runs.  Every
# A/B in the perf notes pins the seed so build time *and* recall are comparable.
BUILD_SEED="${HNSWSQ_BUILD_SEED_BENCH:-20240912}"
# Backlink admission policy: 1 = exact incremental re-prune (default),
# 0 = Lance-style ranked/cutoff list.
BACKLINK_MODE="${HNSWSQ_BACKLINK_MODE_BENCH:-1}"
# Build engine: 0 = legacy MemGraph (default), 1 = new flat engine.
ENGINE="${HNSWSQ_ENGINE_BENCH:-0}"
# Own-list backfill knob (flat engine only): 0 = heuristic only (decided policy),
# 1 = closest-pruned backfill.  See .design/hnswsq_parallel_build_todos.md §3d/§0.
BACKFILL="${HNSWSQ_BACKFILL_BENCH:-0}"
EF_SWEEP="${HNSWSQ_EF_SWEEP:-10 40 160 640}"
OUT_CSV="${OUT_CSV:-/tmp/hnswsq_local_cycle.csv}"
LOG="${LOG:-/tmp/hnswsq_local_$(date +%Y%m%d_%H%M%S)_${LABEL}.log}"

TABLE="t${TAG}"
GT="gt_${TAG}"
QT="bench_queries_${DIM}"
IDX="${TABLE}_idx"
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"

q() { "$PSQL" -h 127.0.0.1 -p "$PORT" -U "$USER" -d "$DB" -X -q -v ON_ERROR_STOP=1 "$@"; }
qn() { "$PSQL" -h 127.0.0.1 -p "$PORT" -U "$USER" -d "$DB" -X -q -At -v ON_ERROR_STOP=1 "$@"; }
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "local cycle start label=$LABEL db=$DB table=$TABLE dim=$DIM layout=$LAYOUT m=$M efc=$EFC"

# ---- 0. the extension must be a release build ------------------------------
SO=$(ls "$(dirname "$PSQL")/../lib/postgresql"*/vectorscale-*.so 2>/dev/null \
     || ls "$(dirname "$PSQL")/../lib/postgresql"*/vectorscale-*.dylib 2>/dev/null \
     || true)
if [ -n "$SO" ]; then
    SO_MB=$(( $(stat -f %z "$SO" 2>/dev/null || stat -c %s "$SO") / 1048576 ))
    log "installed extension: $SO (${SO_MB}MB)"
    if [ "$SO_MB" -gt 3 ]; then
        log "FATAL: that looks like a debug build; reinstall with:"
        log "  cargo pgrx install --release --pg-config $PGBIN/pg_config --no-default-features --features pg18"
        exit 1
    fi
fi

ROWS=$(qn -c "SELECT count(*) FROM $TABLE;" 2>/dev/null)
GT_ROWS=$(qn -c "SELECT count(*) FROM $GT;" 2>/dev/null)
QROWS=$(qn -c "SELECT count(*) FROM $QT;" 2>/dev/null)
log "dataset rows=$ROWS gt=$GT_ROWS queries=$QROWS"
if [ -z "${ROWS:-}" ] || [ "$ROWS" = "0" ]; then
    log "FATAL: table $TABLE is missing or empty (run local_dataset.sh $TAG $DB <queries> <clusters> $DIM)"
    exit 1
fi

# ---- 1. kill stale index builds (by PID: killing the psql wrapper is not enough)
STALE=$(ps aux | grep -E "CREATE INDEX|CREATE UNIQUE INDEX" | grep -v grep | awk '{print $2}')
if [ -n "$STALE" ]; then
    log "killing stale CREATE INDEX backends: $STALE"
    kill -9 $STALE 2>/dev/null
    sleep 2
fi

# ---- 2. drop + rebuild, timed, with stats on -------------------------------
q -c "DROP INDEX IF EXISTS $IDX;" >>"$LOG" 2>&1
log "building $IDX (layout=$LAYOUT m=$M efc=$EFC maintenance_work_mem=$MAINT_MEM build_seed=$BUILD_SEED backlink_mode=$BACKLINK_MODE engine=$ENGINE backfill=$BACKFILL)"
BUILD_START=$(date +%s.%N)
q -c "SET maintenance_work_mem = '$MAINT_MEM'; SET hnswsq.build_stats = on;
      SET hnswsq.build_seed = $BUILD_SEED;
      SET hnswsq.build_backlink_mode = $BACKLINK_MODE;
      SET hnswsq.build_engine = $ENGINE;
      SET hnswsq.build_backfill = $BACKFILL;
      CREATE INDEX $IDX ON $TABLE USING hnswsq (embedding vector_l2_ops)
      WITH (storage_layout = '$LAYOUT', m = $M, ef_construction = $EFC);" >>"$LOG" 2>&1
BUILD_RC=$?
BUILD_END=$(date +%s.%N)
BUILD_S=$(echo "$BUILD_END - $BUILD_START" | bc)
if [ $BUILD_RC -ne 0 ]; then
    log "FATAL: CREATE INDEX failed (rc=$BUILD_RC), see $LOG"
    exit 1
fi
STATS=$(grep -o "hnswsq build stats:.*" "$LOG" | tail -1)
log "build ${BUILD_S}s ${STATS}"

SIZE=$(qn -c "SELECT pg_relation_size('$IDX');")

# ---- 3. recall/latency sweep ----------------------------------------------
q -f "$BENCH_DIR/local_sweep.sql" >>"$LOG" 2>&1
EF_ARRAY=$(echo "$EF_SWEEP" | tr ' ' ',')
SWEEP=$(qn -c "SELECT string_agg(format('ef=%s recall=%s p50=%sms p99=%sms', ef, recall, p50_ms, p99_ms), ' | ')
                FROM bench_sweep('$TABLE', '$GT', '$QT', ARRAY[$EF_ARRAY]);" 2>/dev/null)
log "sweep: $SWEEP"

echo "$LABEL,$LAYOUT,$TAG,dim$DIM,$M,$EFC,$BUILD_S,$SIZE,${STATS:-},${SWEEP:-}" >>"$OUT_CSV"
log "csv row appended to $OUT_CSV; log $LOG"
