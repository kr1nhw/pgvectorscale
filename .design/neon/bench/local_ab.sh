#!/bin/bash
# Local (single-host) hnswsq vs pgvector A/B for the "plain not worse than
# pgvector" gate. Reproduces the methodology in
# .design/hnswsq2_perf_local.md: one i.i.d. table, one index at a time (the
# planner picks the surviving index), release builds on both sides.
#
# Usage: local_ab.sh <engine:hnswsq|hnsw> [rounds]
#
#   hnswsq -> CREATE INDEX ... USING hnswsq (vector_l2_ops) WITH
#             (m=16, ef_construction=64, storage_layout=plain)
#   hnsw   -> pgvector baseline, same m/efc
#
# Prints, per round: build seconds, index size, batch ms at each ef
# (10,000 index scans per batch), and recall@10 vs the exact seq-scan
# top-10 at ef 160/640.
#
# Environment: PGHOST (/tmp/pgtestk) PGPORT (54329) PGUSER DB (postgres)
#              PSQL (/opt/homebrew/bin/psql) TABLE (ab2) QUERIES (ab2_q)
#              DIM (128) ROWS (100000) EFS (160 640) MAINT_MEM (2GB)
#
# Traps this script enforces:
#   * statement_timeout on every run so a stalled backend cannot zombie;
#   * the index is dropped and recreated per run (no page-reuse skew);
#   * build_seed pinned so both engines' graphs are reproducible;
#   * `cargo pgrx test` overwrites the installed extension with the DEBUG
#     build — check the dylib size (< 3 MB) before trusting any number.
set -uo pipefail

ENGINE="${1:-hnswsq}"
ROUNDS="${2:-2}"
PSQL="${PSQL:-/opt/homebrew/bin/psql}"
PGHOST="${PGHOST:-/tmp/pgtestk}"
PGPORT="${PGPORT:-54329}"
PGUSER="${PGUSER:-}"
DB="${DB:-postgres}"
TABLE="${TABLE:-ab2}"
QUERIES="${QUERIES:-ab2_q}"
DIM="${DIM:-128}"
ROWS="${ROWS:-100000}"
EFS="${EFS:-160 640}"
MAINT_MEM="${MAINT_MEM:-2GB}"
SEED="${SEED:-20240912}"
GT_IDX="${GT_IDX:-false}"   # true: also compute recall@10 vs exact top-10

Q=("$PSQL" -h "$PGHOST" -p "$PGPORT" ${PGUSER:+-U "$PGUSER"} -d "$DB" -X -q -At -v ON_ERROR_STOP=1)

# ---- 0. guards --------------------------------------------------------------
SO=$(ls /opt/homebrew/lib/postgresql@18/vectorscale-*.dylib 2>/dev/null | head -1)
if [ -n "$SO" ]; then
  KB=$(( $(stat -f %z "$SO") / 1024 ))
  echo "# installed extension: $SO (${KB}KB)"
  if [ "$KB" -gt 3000 ]; then
    echo "# FATAL: that looks like the DEBUG build (~7.6MB). Restore the release dylib." >&2
    exit 1
  fi
fi
ROWS_ACTUAL=$("${Q[@]}" -c "SELECT count(*) FROM $TABLE;")
QROWS=$("${Q[@]}" -c "SELECT count(*) FROM $QUERIES;")
echo "# dataset: $TABLE rows=$ROWS_ACTUAL $QUERIES=$QROWS dim=$DIM (expected $ROWS)"
if [ "$ROWS_ACTUAL" != "$ROWS" ] || [ "$QROWS" = "0" ]; then
  echo "# FATAL: dataset missing — create it first, e.g. the SQL at the end of this file." >&2
  exit 1
fi

IDX="${TABLE}_${ENGINE}_ab"
OTHER_IDX="${TABLE}_$([ "$ENGINE" = hnswsq ] && echo hnsw || echo hnswsq)_ab"
GUC=$([ "$ENGINE" = hnswsq ] && echo hnswsq || echo hnsw)

batch() { # $1 = ef
  "${Q[@]}" -c "SET enable_seqscan = off; SET $GUC.ef_search = $1; SET statement_timeout = '300s';
    SELECT sum((SELECT count(*) FROM (SELECT id FROM $TABLE ORDER BY embedding <-> q.q LIMIT 10) t))
    FROM $QUERIES q, generate_series(1,100) g;" | head -1
}

echo "engine,round,build_s,size_bytes$(printf ',ef%s_ms' $EFS)"
for r in $(seq 1 "$ROUNDS"); do
  # The planner picks the surviving index on the column, so BOTH engines'
  # indexes must be gone before building ours — otherwise an A/B silently
  # measures the other engine's index (bit us once: identical timings at
  # every ef and recall at the default ef).
  "${Q[@]}" -c "DROP INDEX IF EXISTS $IDX;" >/dev/null 2>&1
  "${Q[@]}" -c "DROP INDEX IF EXISTS $OTHER_IDX;" >/dev/null 2>&1
  T0=$(date +%s.%N)
  if [ "$ENGINE" = hnswsq ]; then
    "${Q[@]}" -c "SET maintenance_work_mem = '$MAINT_MEM'; SET hnswsq.build_seed = $SEED;
      CREATE INDEX $IDX ON $TABLE USING hnswsq (embedding vector_l2_ops)
      WITH (m = 16, ef_construction = 64, storage_layout = plain);" >/dev/null
  else
    "${Q[@]}" -c "SET maintenance_work_mem = '$MAINT_MEM';
      CREATE INDEX $IDX ON $TABLE USING hnsw (embedding vector_l2_ops)
      WITH (m = 16, ef_construction = 64);" >/dev/null
  fi
  T1=$(date +%s.%N)
  BUILD_S=$(echo "$T1 - $T0" | bc)
  SIZE=$("${Q[@]}" -c "SELECT pg_relation_size('$IDX');")
  # Sanity: the planner must actually use the index we just built.
  PLAN_IDX=$("${Q[@]}" -c "SET enable_seqscan = off; EXPLAIN SELECT id FROM $TABLE ORDER BY embedding <-> (SELECT q FROM $QUERIES LIMIT 1) LIMIT 10;" 2>/dev/null | grep -o "Index Scan using [a-z0-9_]*" | awk '{print $4}')
  [ "$PLAN_IDX" = "$IDX" ] || echo "# WARNING: planner chose '$PLAN_IDX', expected '$IDX'" >&2
  TIMES=""
  for ef in $EFS; do
    # first batch per ef warms; time the second
    batch "$ef" >/dev/null
    MS=$( { /usr/bin/time -p "${Q[@]}" -c "SET enable_seqscan = off; SET $GUC.ef_search = $ef; SET statement_timeout = '300s';
      SELECT sum((SELECT count(*) FROM (SELECT id FROM $TABLE ORDER BY embedding <-> q.q LIMIT 10) t))
      FROM $QUERIES q, generate_series(1,100) g;" >/dev/null; } 2>&1 | awk '/real/ {print $2*1000}' )
    TIMES="$TIMES,$MS"
  done
  echo "$ENGINE,$r,$BUILD_S,$SIZE$TIMES"
done

if [ "$GT_IDX" = true ]; then
  # One session: temp tables do not survive across psql invocations.
  SQL="SET enable_indexscan = off; SET enable_bitmapscan = off;
    CREATE TEMP TABLE la_truth AS SELECT q.q AS vec, t.id AS nn FROM $QUERIES q
    CROSS JOIN LATERAL (SELECT id FROM $TABLE ORDER BY embedding <-> q.q LIMIT 10) t;"
  for ef in $EFS; do
    SQL="$SQL SET enable_indexscan = on; SET enable_seqscan = off; SET $GUC.ef_search = $ef;
    CREATE TEMP TABLE la_got_$ef AS SELECT q.q AS vec, t.id AS nn FROM $QUERIES q
    CROSS JOIN LATERAL (SELECT id FROM $TABLE ORDER BY embedding <-> q.q LIMIT 10) t;
    SELECT 'recall@10 ef=$ef: ' ||
      (SELECT count(*)::float8 / (SELECT count(*) FROM la_truth)
       FROM (SELECT vec, nn FROM la_truth INTERSECT SELECT vec, nn FROM la_got_$ef) x)::text;"
  done
  "${Q[@]}" -c "$SQL"
fi

exit 0

# Dataset SQL (run once per database):
#   CREATE TABLE ab2 (id serial PRIMARY KEY, embedding vector(128));
#   INSERT INTO ab2 (embedding)
#     SELECT ('[' || string_agg((random())::text, ',' ORDER BY s.i) || ']')::vector
#     FROM generate_series(1,100000) AS r(gs)
#     CROSS JOIN LATERAL generate_series(1,128) AS s(i)
#     GROUP BY r.gs;
#   CREATE TABLE ab2_q AS SELECT embedding AS q FROM ab2 WHERE id % 1000 = 1;
