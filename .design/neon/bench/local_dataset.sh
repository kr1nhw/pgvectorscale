#!/bin/bash
# Build a local benchmark dataset at an arbitrary size and dimension.
#
# Usage: local_dataset.sh <tag> [db] [queries] [clusters] [dim]
#
#   tag       dataset tag: table `t<tag>`, ground truth `gt_<tag>`
#             (e.g. `100k` for dim 16 work, `100kd128` for dim 128 work)
#   db        database name (default t100kdb)
#   queries   number of query vectors (default 200 for n<=300k, else 100)
#   clusters  number of clusters (default 200; **0 = uniform random** in [0,1],
#             which is the shape that matches BIGANN and keeps recall meaningful
#             at dim 128 — tight synthetic clusters make the true 10-NN
#             neighbourhood so small that recall collapses at any ef)
#   dim       vector dimension (default 16; 128 matches the BIGANN remote runs)
#
# Rows are spread evenly over `clusters` Gaussian-ish clusters; the shared
# `bench_queries_<dim>` table holds the query vectors and `gt_<tag>` the exact
# top-10 per query (computed by seq scan, no index).
#
# Dimension matters for the *distance* path: dim 16 hides the per-element
# decode cost that dominates a dim 128 build/scan, so kernel work must be
# measured on a dim 128 dataset (tag it separately, e.g. `100kd128`).
set -euo pipefail

TAG="${1:?tag}"
DB="${2:-t100kdb}"
CLUSTERS="${4:-200}"
DIM="${5:-16}"
if [ -z "${3:-}" ]; then
    if [ "$TAG" -gt 300000 ] 2>/dev/null; then QUERIES=100; else QUERIES=200; fi
else
    QUERIES="$3"
fi

PORT="${PGPORT:-54331}"
USER="${PGUSER:-$USER}"
PGBIN="${PGBIN:-/opt/homebrew/opt/postgresql@18/bin}"
PSQL="${PSQL_BIN:-$PGBIN/psql}"
TABLE="t$TAG"
GT="gt_$TAG"
QT="bench_queries_${DIM}"
N="${TAG%%d*}"                      # rows, when the tag is `<rows>d<dim>`
case "$N" in (*[!0-9]*|'') N=100000;; esac
if [ "$CLUSTERS" -gt 0 ]; then
    PER=$(( (N + CLUSTERS - 1) / CLUSTERS ))
else
    PER=0
fi

q() { "$PSQL" -h 127.0.0.1 -p "$PORT" -U "$USER" -d "$1" -X -q -v ON_ERROR_STOP=1 "${@:2}"; }

if ! "$PSQL" -h 127.0.0.1 -p "$PORT" -U "$USER" -d "$DB" -X -Atc "SELECT 1" >/dev/null 2>&1; then
    echo "creating database $DB"
    q postgres -c "CREATE DATABASE $DB;"
    q "$DB" -c "CREATE EXTENSION vector; CREATE EXTENSION vectorscale CASCADE;"
fi

echo "loading $TABLE ($N rows, dim $DIM, $CLUSTERS clusters x $PER)"
q "$DB" -c "DROP TABLE IF EXISTS $TABLE; DROP TABLE IF EXISTS $GT;"
q "$DB" -c "CREATE TABLE $TABLE (id int PRIMARY KEY, embedding vector($DIM));"
q "$DB" -c "SELECT setseed(0.42);"
if [ "$CLUSTERS" -eq 0 ]; then
    # Uniform random in [0,1]^dim (BIGANN-like): no cluster structure.
    # `random()` must be evaluated per (row, dimension): an uncorrelated scalar
    # subquery would be treated as an InitPlan and give every row the same
    # vector, so aggregate over an explicit dimension series instead.
    q "$DB" -c "
INSERT INTO $TABLE (id, embedding)
SELECT i - 1, array_agg(random()::float4 ORDER BY j)::float4[]::vector($DIM)
FROM generate_series(1, $N) i
CROSS JOIN generate_series(1, $DIM) j
GROUP BY i;"
else
    q "$DB" -c "
WITH centers AS (
    SELECT g AS cid,
           (SELECT array_agg(random() * 2 - 1) FROM generate_series(1, $DIM)) AS c
    FROM generate_series(1, $CLUSTERS) g
)
INSERT INTO $TABLE (id, embedding)
SELECT (row_number() OVER ())::int - 1,
       (SELECT array_agg((c.c[j] + (random() - 0.5) * 0.05)::float4)
        FROM generate_series(1, $DIM) j)::float4[]::vector($DIM)
FROM centers c, generate_series(1, $PER);"
fi

if ! "$PSQL" -h 127.0.0.1 -p "$PORT" -U "$USER" -d "$DB" -X -Atc \
        "SELECT count(*) FROM $TABLE" | grep -qx "$N"; then
    echo "FATAL: $TABLE has $(q "$DB" -Atc "SELECT count(*) FROM $TABLE") rows, expected $N" >&2
    exit 1
fi

if ! "$PSQL" -h 127.0.0.1 -p "$PORT" -U "$USER" -d "$DB" -X -Atc \
        "SELECT to_regclass('$QT')" | grep -q "$QT"; then
    q "$DB" -c "CREATE TABLE $QT (qid int PRIMARY KEY, q vector($DIM));"
fi
if [ "$CLUSTERS" -eq 0 ]; then
    q "$DB" -c "
INSERT INTO $QT (qid, q)
SELECT i - 1, array_agg(random()::float4 ORDER BY j)::float4[]::vector($DIM)
FROM generate_series(1, $QUERIES) i
CROSS JOIN generate_series(1, $DIM) j
GROUP BY i
ON CONFLICT (qid) DO NOTHING;"
else
    q "$DB" -c "
WITH centers AS (
    SELECT g AS cid,
           (SELECT array_agg(random() * 2 - 1) FROM generate_series(1, $DIM)) AS c
    FROM generate_series(1, $QUERIES) g
)
INSERT INTO $QT (qid, q)
SELECT (row_number() OVER ())::int - 1,
       (SELECT array_agg((c.c[j] + (random() - 0.5) * 0.01)::float4)
        FROM generate_series(1, $DIM) j)::float4[]::vector($DIM)
FROM centers c
ON CONFLICT (qid) DO NOTHING;"
fi

if ! "$PSQL" -h 127.0.0.1 -p "$PORT" -U "$USER" -d "$DB" -X -Atc \
        "SELECT to_regclass('$GT')" | grep -q "$GT"; then
    echo "computing ground truth $GT"
    q "$DB" -c "
CREATE TABLE $GT AS
SELECT qid, array_agg(id ORDER BY d) AS ids
FROM (
    SELECT q.qid, t.id, t.embedding <-> q.q AS d,
           row_number() OVER (PARTITION BY q.qid ORDER BY t.embedding <-> q.q) AS rn
    FROM $QT q, $TABLE t
) s
WHERE rn <= 10
GROUP BY qid;"
fi

q "$DB" -c "ANALYZE $TABLE;"
q "$DB" -Atc "SELECT '$DB: $TABLE=' || (SELECT count(*) FROM $TABLE) || ' gt=' ||
                      (SELECT count(*) FROM $GT) || ' queries=' ||
                      (SELECT count(*) FROM $QT);"
