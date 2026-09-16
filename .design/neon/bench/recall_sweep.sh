#!/bin/bash
# Recall@10 sweep across ef_search for pgvector and every hnswsq storage
# layout, against the exact top-10 ground truth, on the 1M BIGANN dataset.
#
# Usage: recall_sweep.sh [table] [queries] [gt] ["ef list"]
#
# For each configuration (pgvector + hnswsq plain/ieeefp16/ieeefp8/f8) this
# builds the index fresh (m=16, efc=64, mwm 8GB, pinned build_seed for
# hnswsq), then measures recall@10 at each ef. One index at a time — the
# planner picks the surviving index, so every other hnsw-family index on the
# table is dropped first.
#
# Results: CSV on stdout and /tmp/recall_sweep.csv
set -uo pipefail

PSQL=/root/.pgrx-hnswsq/17.11/pgrx-install/bin/psql
TABLE="${1:-items_1m}"
QUERIES="${2:-bench_queries}"
GT="${3:-gt_1m}"
EFS="${4:-10 20 40 80 160 320 640}"
SEED=20240912
OUT_CSV=/tmp/recall_sweep.csv

q() { sudo -u pgtest "$PSQL" -h 127.0.0.1 -p 54330 -U pgtest -d postgres -X -q -At -v ON_ERROR_STOP=1 "$@"; }
log() { echo "[$(date '+%H:%M:%S')] $*"; }

drop_hnsw_indexes() {
  # The planner picks the surviving index on the column, so EVERY hnsw-family
  # index must go before building ours — a leftover index silently answers the
  # recall queries with the wrong engine at its default ef (bit us once:
  # recall flat at 0.9430 across all ef).
  for idx in $(q -c "SELECT indexname FROM pg_indexes WHERE tablename='$TABLE'
                      AND indexname <> '${TABLE}_pkey';"); do
    q -c "DROP INDEX IF EXISTS $idx;" >/dev/null
  done
}

# plan_idx: the index the planner picks for the benchmark query (sanity).
plan_idx() {
  q -c "SET enable_seqscan = off; EXPLAIN SELECT id FROM $TABLE
        ORDER BY embedding <-> (SELECT q FROM $QUERIES LIMIT 1) LIMIT 10;" \
    | grep -o "Index Scan using [a-z0-9_]*" | awk '{print $4}'
}

# recall <guc-prefix> <ef>
recall() {
  q -c "SET enable_seqscan = off; SET $1.ef_search = $2;
        SELECT round(avg(rec)::numeric, 4) FROM (
          SELECT (SELECT count(*) FROM (SELECT id FROM $TABLE
                  ORDER BY embedding <-> bq.q LIMIT 10) x
                  WHERE x.id = ANY (gt.ids))::float8 / 10 AS rec
          FROM $QUERIES bq JOIN $GT gt USING (qid)) r;"
}

build_hnswsq() { # $1 = layout
  local layout="$1"
  local idx="items_1m_recall_${layout}"
  drop_hnsw_indexes
  log "build hnswsq layout=$layout"
  q -c "SET maintenance_work_mem = '8GB'; SET hnswsq.build_seed = $SEED;
        CREATE INDEX $idx ON $TABLE USING hnswsq (embedding vector_l2_ops)
        WITH (m = 16, ef_construction = 64, storage_layout = $layout);" >/dev/null
  [ "$(plan_idx)" = "$idx" ] || log "WARNING: planner picked $(plan_idx), expected $idx"
  for ef in $EFS; do
    echo "hnswsq,$layout,$ef,$(recall hnswsq "$ef")"
  done
  q -c "DROP INDEX IF EXISTS $idx;" >/dev/null
}

build_pgvector() {
  local idx="items_1m_recall_hnsw"
  drop_hnsw_indexes
  log "build pgvector"
  q -c "SET maintenance_work_mem = '8GB';
        CREATE INDEX $idx ON $TABLE USING hnsw (embedding vector_l2_ops)
        WITH (m = 16, ef_construction = 64);" >/dev/null
  [ "$(plan_idx)" = "$idx" ] || log "WARNING: planner picked $(plan_idx), expected $idx"
  for ef in $EFS; do
    echo "hnsw,plain,$ef,$(recall hnsw "$ef")"
  done
  q -c "DROP INDEX IF EXISTS $idx;" >/dev/null
}

{
  echo "engine,layout,ef,recall10"
  build_pgvector
  build_hnswsq plain
  build_hnswsq ieeefp16
  build_hnswsq ieeefp8
  build_hnswsq f8
  build_hnswsq sq8
  build_hnswsq sq16
} | tee "$OUT_CSV"
log "done -> $OUT_CSV"
