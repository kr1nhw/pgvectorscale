#!/bin/bash
# Transfer BIGANN-10M (items_10m) + 100 queries + ground truth from the
# vanilla PG17 bench db (5432) into the Neon compute (55432).
set -euxo pipefail

VANILLA="/data1/pg17/bin/psql -h 127.0.0.1 -p 5432 -U postgres -d bench -X -q"
NEON() {
  su pg17test -s /bin/bash -c "/data1/neon-test/pg_install/v17/bin/psql -h 127.0.0.1 -p 55432 -U cloud_admin -d postgres -X -q -v ON_ERROR_STOP=1 \"$1\""
}

echo "== export from vanilla =="
$VANILLA -c "ALTER EXTENSION vector UPDATE;" 2>&1 | tail -1 || true
$VANILLA -c "\copy items_10m TO '/tmp/items10m.copy' BINARY"
$VANILLA -c "\copy (SELECT qid, q FROM bench_queries WHERE qid < 100 ORDER BY qid) TO '/tmp/bq100.copy' BINARY"
$VANILLA -c "\copy gt_10m TO '/tmp/gt10m.copy' BINARY"
ls -la /tmp/items10m.copy /tmp/bq100.copy /tmp/gt10m.copy
chown pg17test:pg17test /tmp/items10m.copy /tmp/bq100.copy /tmp/gt10m.copy

echo "== import into neon =="
NEON "ALTER EXTENSION vector UPDATE;"
NEON "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
NEON "DROP TABLE IF EXISTS items_10m; CREATE TABLE items_10m (id integer PRIMARY KEY, embedding vector(128) NOT NULL);"
NEON "DROP TABLE IF EXISTS bench_queries; CREATE TABLE bench_queries (qid integer PRIMARY KEY, q vector(128) NOT NULL);"
NEON "DROP TABLE IF EXISTS gt_10m; CREATE TABLE gt_10m (qid integer PRIMARY KEY, ids integer[] NOT NULL);"
NEON "\copy items_10m FROM '/tmp/items10m.copy' BINARY"
NEON "\copy bench_queries FROM '/tmp/bq100.copy' BINARY"
NEON "\copy gt_10m FROM '/tmp/gt10m.copy' BINARY"
NEON "ANALYZE items_10m; ANALYZE bench_queries;"
NEON "SELECT count(*) FROM items_10m;"
echo "== transfer done =="
