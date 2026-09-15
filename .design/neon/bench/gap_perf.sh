#!/bin/bash
# Flat profile of one hnsw-family index scan on the 121 scratch cluster.
#
# Usage: gap_perf.sh <engine:hnsw|hnswsq> [ef] [seconds] [table]
#
# Starts a psql loop that keeps the ANN query running, profiles the backend
# with perf (symbols are present: the release .so is not stripped), then prints
# the top symbols.  This is what attributes the query gap to concrete
# functions (buffer manager vs rkyv load vs allocation vs rkyv/executor).
set -uo pipefail

ENGINE="${1:-hnswsq}"
EF="${2:-160}"
SECS="${3:-20}"
TABLE="${4:-items_1m}"

PSQL=/root/.pgrx-hnswsq/17.11/pgrx-install/bin/psql
run() { sudo -u pgtest "$PSQL" -h 127.0.0.1 -p 54330 -U pgtest -d postgres -X -q "$@"; }

GUC=$([ "$ENGINE" = hnswsq ] && echo hnswsq.ef_search || echo hnsw.ef_search)
echo "profiling engine=$ENGINE ef=$EF for ${SECS}s on $TABLE"

# Query loop: repeat one representative query (qid 0) for the duration.
{
    echo "SET enable_seqscan = off;"
    echo "SET $GUC = $EF;"
    echo "\pset tuples_only on"
    echo "\o /dev/null"
    # ~2000 queries per profile window, cycling the query set so the whole
    # bench_queries table is touched rather than one hot query.
    for r in $(seq 1 20); do
        for qid in $(seq 0 99); do
            echo "SELECT count(*) FROM (SELECT id FROM $TABLE ORDER BY embedding <-> (SELECT q FROM bench_queries WHERE qid = $qid) LIMIT 10) s;"
        done
    done
    echo "\o"
} > /tmp/gap_perf_loop.sql

run -f /tmp/gap_perf_loop.sql > /tmp/gap_perf_loop.out 2>&1 &
LOOP=$!
sleep 2
PID=$(run -Atc "SELECT pid FROM pg_stat_activity WHERE state='active' AND query LIKE 'SELECT count(*) FROM (SELECT id FROM $TABLE%' AND pid <> pg_backend_pid() LIMIT 1")
if [ -z "$PID" ]; then
    echo "could not find the query backend" >&2
    exit 1
fi
echo "backend pid=$PID"

perf record -F 997 -o /tmp/gap_perf.data -p "$PID" -- sleep "$SECS" > /tmp/gap_perf_record.log 2>&1
kill "$LOOP" 2>/dev/null
wait "$LOOP" 2>/dev/null

echo "=== top symbols ($ENGINE ef=$EF) ==="
perf report -i /tmp/gap_perf.data --stdio --sort symbol -g none --percent-limit 0.4 2>/dev/null \
    | grep -vE '^#|^$' | head -40

echo "=== dso split ==="
perf report -i /tmp/gap_perf.data --stdio --sort dso -g none --percent-limit 0.5 2>/dev/null \
    | grep -vE '^#|^$' | head -12
