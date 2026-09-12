#!/bin/bash
# Full query/build gap study against pgvector on the 121 scratch cluster.
#
#   ./gap_study.sh
#
# Measures, with both engines built from source in **release** on the same box
# and the same 1M BIGANN table (`items_1m`, dim 128, m=16, efc=64):
#
#   1. hnswsq 1M build (timed, with hnswsq.build_stats) + query sweep + perf profile
#   2. pgvector 1M build at its default parallelism + query sweep + perf profile
#   3. pgvector 1M build single-backend            (per-core build comparison)
#   4. pgvector 1M build with 32 workers           (parallel scaling)
#
# The query sweep runs each configuration in one backend and records, per ef
# value, the mean execution time for LIMIT 10 (what a user waits for) and for a
# full candidate drain (LIMIT 100000), plus the shared blocks touched and the
# number of rows the index scan actually produced.
#
# Results: /tmp/gap_study.log, /tmp/gap_sweep_<engine>.csv
set -uo pipefail

PSQL=/root/.pgrx-hnswsq/17.11/pgrx-install/bin/psql
SO=/root/.pgrx-hnswsq/17.11/pgrx-install/lib/postgresql/vectorscale-0.9.0.so
BENCH=/data1/pgvectorscale-hnswsq/.design/neon/bench
TABLE=items_1m
LOG=/tmp/gap_study.log

run() { sudo -u pgtest "$PSQL" -h 127.0.0.1 -p 54330 -U pgtest -d postgres -X -q -v ON_ERROR_STOP=1 "$@"; }
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# ---- 0. guards ------------------------------------------------------------
# `cargo pgrx test` installs a DEBUG extension over the release one; every
# number in this study is meaningless if that happened (debug is ~25x slower).
SO_KB=$(stat -c %s "$SO" 2>/dev/null | awk '{print int($1/1024)}')
if [ "${SO_KB:-0}" -gt 8000 ]; then
    echo "FATAL: $SO is ${SO_KB}KB — that is a debug build (release is ~2.3MB)." >&2
    echo "Reinstall with: cargo pgrx install --release ..." >&2
    exit 1
fi

: > "$LOG"
log "study start: extension .so=${SO_KB}KB, server max_parallel_maintenance_workers=$(run -Atc 'SHOW max_parallel_maintenance_workers'), max_worker_processes=$(run -Atc 'SHOW max_worker_processes')"
log "indexes before: $(run -Atc "SELECT coalesce(string_agg(indexname, ','), '-') FROM pg_indexes WHERE tablename='$TABLE'")"

# ---- 1. hnswsq: build + query path ---------------------------------------
run -c "DROP INDEX IF EXISTS items_1m_hnsw; DROP INDEX IF EXISTS items_1m_hnswsq;" >>"$LOG" 2>&1
log "--- hnswsq 1M build (single backend)"
T0=$(date +%s)
run -c "SET maintenance_work_mem = '8GB'; SET hnswsq.build_stats = on;
        CREATE INDEX items_1m_hnswsq ON $TABLE USING hnswsq (embedding vector_l2_ops)
        WITH (m = 16, ef_construction = 64, storage_layout = plain);" >>"$LOG" 2>&1
T1=$(date +%s)
log "hnswsq build seconds=$((T1 - T0)) size_bytes=$(run -Atc "SELECT pg_relation_size('items_1m_hnswsq')")"
grep -o "hnswsq build stats:.*" "$LOG" | tail -1 | tee -a "$LOG"

log "--- hnswsq query sweep"
python3 "$BENCH/gap_sweep.py" hnswsq 10,40,160,640 "$TABLE" 100 /tmp/gap_sweep_hnswsq.csv 2>&1 | tee -a "$LOG"
log "--- hnswsq perf profile (ef=160)"
bash "$BENCH/gap_perf.sh" hnswsq 160 20 "$TABLE" 2>&1 | tee -a "$LOG"

# ---- 2. pgvector: default parallelism ------------------------------------
run -c "DROP INDEX IF EXISTS items_1m_hnswsq;" >>"$LOG" 2>&1
build_hnsw() { # $1 = max_parallel_maintenance_workers
    local w="$1" t0 t1
    run -c "DROP INDEX IF EXISTS items_1m_hnsw;" >>"$LOG" 2>&1
    t0=$(date +%s)
    run -c "SET maintenance_work_mem = '8GB'; SET max_parallel_maintenance_workers = $w;
            CREATE INDEX items_1m_hnsw ON $TABLE USING hnsw (embedding vector_l2_ops)
            WITH (m = 16, ef_construction = 64);" >>"$LOG" 2>&1
    t1=$(date +%s)
    log "pgvector build workers=$w seconds=$((t1 - t0)) size_bytes=$(run -Atc "SELECT pg_relation_size('items_1m_hnsw')")"
}

log "--- pgvector build (server default parallelism)"
build_hnsw "$(run -Atc 'SHOW max_parallel_maintenance_workers')"
log "--- pgvector query sweep"
python3 "$BENCH/gap_sweep.py" hnsw 10,40,160,640 "$TABLE" 100 /tmp/gap_sweep_hnsw.csv 2>&1 | tee -a "$LOG"
log "--- pgvector perf profile (ef=160)"
bash "$BENCH/gap_perf.sh" hnsw 160 20 "$TABLE" 2>&1 | tee -a "$LOG"

# ---- 3. pgvector build scaling -------------------------------------------
log "--- pgvector build, single backend"
build_hnsw 0
log "--- pgvector build, 32 workers"
build_hnsw 32

log "study done"
