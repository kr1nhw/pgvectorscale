#!/bin/bash
# Build a local benchmark dataset at an arbitrary size (see local_dataset.sql
# for the 100k variant and the rationale).
#
# Usage: local_dataset.sh <n> [db] [queries] [clusters]
#
#   n         number of rows (table `t<n>`, ground truth `gt_<n>`)
#   db        database name (default t100kdb)
#   queries   number of query vectors (default 200 for n<=300k, else 100)
#   clusters  number of clusters (default 200)
#
# The table is `t<n>` with rows spread evenly over `clusters` Gaussian-ish
# clusters, plus a shared `bench_queries` table and `gt_<n>` with the exact
# top-10 per query (computed by seq scan, no index).
set -euo pipefail

N="${1:?rows}"
DB="${2:-t100kdb}"
CLUSTERS="${4:-200}"
if [ -z "${3:-}" ]; then
    if [ "$N" -gt 300000 ]; then QUERIES=100; else QUERIES=200; fi
else
    QUERIES="$3"
fi

PORT="${PGPORT:-54331}"
USER="${PGUSER:-$USER}"
PGBIN="${PGBIN:-/opt/homebrew/opt/postgresql@18/bin}"
PSQL="${PSQL_BIN:-$PGBIN/psql}"
PER=$(( (N + CLUSTERS - 1) / CLUSTERS ))

q() { "$PSQL" -h 127.0.0.1 -p "$PORT" -U "$USER" -d "$1" -X -q -v ON_ERROR_STOP=1 "${@:2}"; }

if ! "$PSQL" -h 127.0.0.1 -p "$PORT" -U "$USER" -d "$DB" -X -Atc "SELECT 1" >/dev/null 2>&1; then
    echo "creating database $DB"
    q postgres -c "CREATE DATABASE $DB;"
    q "$DB" -c "CREATE EXTENSION vector; CREATE EXTENSION vectorscale CASCADE;"
fi

echo "loading t$N ($N rows, $CLUSTERS clusters x $PER)"
q "$DB" -c "DROP TABLE IF EXISTS t$N; DROP TABLE IF EXISTS gt_$N;"
q "$DB" -c "CREATE TABLE t$N (id int PRIMARY KEY, embedding vector(16));"
q "$DB" -c "SELECT setseed(0.42);"
q "$DB" -c "
WITH centers AS (
    SELECT g AS cid,
           (SELECT array_agg(random() * 2 - 1) FROM generate_series(1, 16)) AS c
    FROM generate_series(1, $CLUSTERS) g
)
INSERT INTO t$N (id, embedding)
SELECT (row_number() OVER ())::int - 1,
       (SELECT array_agg((c.c[j] + (random() - 0.5) * 0.05)::float4)
        FROM generate_series(1, 16) j)::float4[]::vector(16)
FROM centers c, generate_series(1, $PER);"

if ! "$PSQL" -h 127.0.0.1 -p "$PORT" -U "$USER" -d "$DB" -X -Atc \
        "SELECT count(*) FROM t$N" | grep -qx "$N"; then
    echo "FATAL: t$N has $(q "$DB" -Atc "SELECT count(*) FROM t$N") rows, expected $N" >&2
    exit 1
fi

if ! "$PSQL" -h 127.0.0.1 -p "$PORT" -U "$USER" -d "$DB" -X -Atc \
        "SELECT to_regclass('bench_queries')" | grep -q bench_queries; then
    q "$DB" -c "CREATE TABLE bench_queries (qid int PRIMARY KEY, q vector(16));"
fi
q "$DB" -c "
WITH centers AS (
    SELECT g AS cid,
           (SELECT array_agg(random() * 2 - 1) FROM generate_series(1, 16)) AS c
    FROM generate_series(1, $QUERIES) g
)
INSERT INTO bench_queries (qid, q)
SELECT (row_number() OVER ())::int - 1,
       (SELECT array_agg((c.c[j] + (random() - 0.5) * 0.01)::float4)
        FROM generate_series(1, 16) j)::float4[]::vector(16)
FROM centers c
ON CONFLICT (qid) DO NOTHING;"

if ! "$PSQL" -h 127.0.0.1 -p "$PORT" -U "$USER" -d "$DB" -X -Atc \
        "SELECT to_regclass('gt_$N')" | grep -q "gt_$N"; then
    echo "computing ground truth gt_$N"
    q "$DB" -c "
CREATE TABLE gt_$N AS
SELECT qid, array_agg(id ORDER BY d) AS ids
FROM (
    SELECT q.qid, t.id, t.embedding <-> q.q AS d,
           row_number() OVER (PARTITION BY q.qid ORDER BY t.embedding <-> q.q) AS rn
    FROM bench_queries q, t$N t
) s
WHERE rn <= 10
GROUP BY qid;"
fi

q "$DB" -c "ANALYZE t$N;"
q "$DB" -Atc "SELECT '$DB: t$N=' || (SELECT count(*) FROM t$N) || ' gt=' ||
                      (SELECT count(*) FROM gt_$N) || ' queries=' ||
                      (SELECT count(*) FROM bench_queries);"
