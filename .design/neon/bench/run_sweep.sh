#!/bin/bash
# Recall@10 + p50/p99 latency sweep for one index configuration.
#
# Usage: run_sweep.sh <psql_bin> <engine:ivf|hnsw> <label> <out_csv>
#   psql_bin  psql executable; connection via libpq env vars (PGHOST/PGPORT/...)
#   engine    ivf  -> sweep ivf.probes over 1..256 (ivf.top_k fixed at 1000)
#             hnsw -> sweep hnsw.ef_search over 10..640
#   label     tag written into the CSV (e.g. ivfrq-vanilla)
#   out_csv   append-only CSV: label,engine,param,param_value,recall_at_10,p50_ms,p99_ms
#
# Requires, in the target database: items_10m (id int, embedding vector(128)),
# bench_queries (qid 0..99, q vector(128)), gt_10m (qid, ids int[] exact top-10),
# and exactly one usable index: items_10m_ivf or items_10m_hnsw.
#
# Protocol per point: 100 queries (qid 0..99), LIMIT 10, L2.
#   - recall@10 vs gt_10m computed in SQL
#   - one untimed warmup pass, then one timed pass (\timing, "Time:" lines parsed)
#   - p50/p99 over the 100 per-query wall times
#   - any psql failure aborts (exit 1)
set -euo pipefail

PSQL="$1"
ENGINE="$2"
LABEL="$3"
OUT_CSV="$4"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

collect_recall() { # $1=param_value ; echoes recall_at_10
  local val="$1"
  {
    echo 'SET enable_seqscan = off;'
    if [ "$ENGINE" = ivf ]; then
      echo "SET ivf.probes = $val;"
      echo 'SET ivf.top_k = 1000;'
    else
      echo "SET hnsw.ef_search = $val;"
    fi
    echo 'DROP TABLE IF EXISTS res_cur;'
    echo 'CREATE TABLE res_cur (qid int PRIMARY KEY, ids int[]);'
    cat <<'SQL'
INSERT INTO res_cur
SELECT qid, (SELECT array_agg(id) FROM (
  SELECT id FROM items_10m ORDER BY embedding <-> q.q LIMIT 10) t)
FROM (SELECT qid, q FROM bench_queries ORDER BY qid) q;
SELECT round(100.0 * sum(cnt) / (count(*) * 10.0), 2)
FROM (
  SELECT r.qid, count(*) AS cnt
  FROM res_cur r
  JOIN gt_10m g ON g.qid = r.qid,
  LATERAL unnest(r.ids) x(id)
  WHERE x.id = ANY (g.ids)
  GROUP BY r.qid) s;
SQL
  } > "$WORK/recall.sql"
  if ! $PSQL -X -q -v ON_ERROR_STOP=1 -f "$WORK/recall.sql" > "$WORK/recall.out"; then
    echo "recall query failed (engine=$ENGINE param=$val); tail of output:" >&2
    tail -5 "$WORK/recall.out" >&2
    exit 1
  fi
  grep -oE '[0-9]+(\.[0-9]+)?' "$WORK/recall.out" | head -1
}

collect_times() { # $1=param_value ; fills $WORK/times.txt with per-query ms
  local val="$1"
  {
    # GUCs BEFORE \timing on: \timing would otherwise emit a Time line for
    # every SET and corrupt the 100-value count below.
    echo 'SET enable_seqscan = off;'
    if [ "$ENGINE" = ivf ]; then
      echo "SET ivf.probes = $val;"
      echo 'SET ivf.top_k = 1000;'
    else
      echo "SET hnsw.ef_search = $val;"
    fi
    echo '\timing on'
    for qid in $(seq 0 99); do
      echo "SELECT id FROM items_10m ORDER BY embedding <-> (SELECT q FROM bench_queries WHERE qid=$qid) LIMIT 10;"
    done
    echo '\timing off'
  } > "$WORK/timed.sql"

  # warmup: same queries untimed
  sed 's/\\timing on/\\timing off/' "$WORK/timed.sql" > "$WORK/warm.sql"
  $PSQL -X -q -v ON_ERROR_STOP=1 -f "$WORK/warm.sql" > /dev/null \
    || { echo "warmup failed (engine=$ENGINE param=$val)" >&2; exit 1; }

  # timed pass
  $PSQL -X -q -v ON_ERROR_STOP=1 -f "$WORK/timed.sql" 2>&1 \
    | grep '^Time:' | awk '{print $2}' > "$WORK/times.txt" \
    || { echo "timed pass failed (engine=$ENGINE param=$val)" >&2; exit 1; }
  [ "$(wc -l < "$WORK/times.txt")" -eq 100 ] \
    || { echo "expected 100 times, got $(wc -l < "$WORK/times.txt") (engine=$ENGINE param=$val)" >&2; exit 1; }
}

if [ "$ENGINE" = ivf ]; then
  VALS="1 2 4 8 16 32 64 128 256"
  PARAM="probes"
else
  VALS="10 20 40 80 160 320 640"
  PARAM="ef_search"
fi

for v in $VALS; do
  recall=$(collect_recall "$v")
  [ -n "$recall" ] || { echo "empty recall value (engine=$ENGINE param=$v)" >&2; exit 1; }
  collect_times "$v"
  read -r p50 p99 <<< "$(sort -n "$WORK/times.txt" | awk -v n=100 'NR==int(0.50*n)+1{p50=$1} NR==int(0.99*n)+1{p99=$1} END{printf "%s %s", p50, p99}')"
  echo "$LABEL,$ENGINE,$PARAM,$v,$recall,$p50,$p99" | tee -a "$OUT_CSV"
done
