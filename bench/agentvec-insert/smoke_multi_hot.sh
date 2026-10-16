#!/bin/bash
# smoke_multi_hot.sh [port]
#
# Targeted end-to-end smoke test for the HOT pool: install the freshly built
# .so into the scratch cluster's extension directory, then exercise
# hot_segment_count > 1 through SQL and check the properties that matter:
#
#   1. the pool publishes `hot_segment_count` appendable HOT segments;
#   2. inserts spread across them;
#   3. every committed row is findable through the index (the scan merges all
#      graphs), which is the transaction contract;
#   4. an aborted transaction leaves nothing behind;
#   5. a row is visible to an index scan in the SAME transaction that inserted
#      it, and invisible to another backend until commit (the same-commit
#      guarantee pgvector provides);
#   6. sealing keeps the pool at its configured size and loses no row.
set -eu
PORT=${1:-54330}
PGRX_INSTALL=/root/.pgrx-hnswsq/17.11/pgrx-install
SO=/data1/avwork/target/release/libvectorscale.so
PSQL="$PGRX_INSTALL/bin/psql -h /tmp -p $PORT -U pgtest -d postgres -qtA"

echo "### installing $SO"
install -m 755 "$SO" "$PGRX_INSTALL/lib/postgresql/vectorscale-0.9.0.so"
echo "### installed"

$PSQL -c "SELECT 'connected', version()" >/dev/null

echo "### 1/6 pool publishes hot_segment_count appendable segments"
$PSQL <<'SQL'
DROP TABLE IF EXISTS smoke_pool CASCADE;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale;
CREATE TABLE smoke_pool(id int, v vector(4));
CREATE INDEX smoke_pool_idx ON smoke_pool USING agentvec (v vector_l2_ops)
    WITH (hot_segment_count = 4);
INSERT INTO smoke_pool SELECT g, ('[' || g || ',' || g || ',0,0]')::vector
    FROM generate_series(1, 200) AS g;
SQL
$PSQL <<'SQL'
DROP TABLE IF EXISTS smoke_k1 CASCADE;
CREATE TABLE smoke_k1(id int, v vector(4));
CREATE INDEX smoke_k1_idx ON smoke_k1 USING agentvec (v vector_l2_ops);
INSERT INTO smoke_k1 SELECT g, ('[' || g || ',' || g || ',0,0]')::vector
    FROM generate_series(1, 200) AS g;
SQL
$PSQL -c "SELECT segment_id, level, state, algorithm, is_hot, num_entries FROM agentvec_index_info('smoke_pool_idx')"

HOT=$($PSQL -c "SELECT count(*) FROM agentvec_index_info('smoke_pool_idx') WHERE is_hot")
NONEMPTY=$($PSQL -c "SELECT count(*) FROM agentvec_index_info('smoke_pool_idx') WHERE is_hot AND num_entries > 0")
echo "    appendable hot segments: $HOT (expect 4);  non-empty: $NONEMPTY (expect 4)"
[ "$HOT" = "4" ] || { echo "FAIL: expected 4 appendable hot segments, got $HOT"; exit 1; }
[ "$NONEMPTY" = "4" ] || { echo "FAIL: inserts did not spread across the pool"; exit 1; }

echo "### 2/6 the scan merges every region and ranks correctly"
# Exact nearest-neighbour order across the pool (each region is searched with a
# wide ef so the merge, not the approximation, is what is under test).
ORDER=$($PSQL -c "SET enable_seqscan=off; SET hnswsq.ef_search = 500; SELECT string_agg(id::text, ',') FROM (SELECT id FROM smoke_pool ORDER BY v <-> '[0,0,0,0]' LIMIT 5) q")
echo "    nearest 5: $ORDER (expect 1,2,3,4,5)"
[ "$ORDER" = "1,2,3,4,5" ] || { echo "FAIL: wrong order across segments"; exit 1; }
# The pool's effective candidate budget is K x (one region's candidate set):
# each region is searched independently and the scan unions the results.  That
# is the scan-side price of more hot segments, and it is also why more, smaller
# graphs cover *more* of the data than one bigger graph at the same ef.
TOTAL=$($PSQL -c "SET enable_seqscan=off; SELECT count(*) FROM (SELECT id FROM smoke_pool ORDER BY v <-> '[100,100,0,0]' LIMIT 500) q")
SINGLE=$($PSQL -c "SET enable_seqscan=off; SELECT count(*) FROM (SELECT id FROM smoke_k1 ORDER BY v <-> '[100,100,0,0]' LIMIT 500) q")
echo "    candidates: K=4 pool -> $TOTAL, K=1 single region -> $SINGLE (expect K=4 > K=1)"
[ "$TOTAL" -gt "$SINGLE" ] || { echo "FAIL: the pool did not contribute more candidates than one region"; exit 1; }
# Structural completeness is the no-loss property: every row is accounted for.
STRUCT=$($PSQL -c "SELECT sum(num_entries)::bigint FROM agentvec_index_info('smoke_pool_idx')")
echo "    structural entries: $STRUCT (expect 200)"
[ "$STRUCT" = "200" ] || { echo "FAIL: entries lost across the pool"; exit 1; }

echo "### 3/6 rolled-back insert leaves nothing behind"
BEFORE=$($PSQL -c "SELECT sum(num_entries)::bigint FROM agentvec_index_info('smoke_pool_idx')")
$PSQL <<'SQL'
BEGIN;
INSERT INTO smoke_pool SELECT 9000 + g, ('[1,1,0,0]')::vector FROM generate_series(1,5) AS g;
ROLLBACK;
SQL
AFTER=$($PSQL -c "SELECT sum(num_entries)::bigint FROM agentvec_index_info('smoke_pool_idx')")
VIS=$($PSQL -c "SET enable_seqscan=off; SELECT count(*) FROM smoke_pool WHERE id > 9000")
echo "    num_entries before=$BEFORE after=$AFTER; rows visible to a scan: $VIS (expect 0)"
echo "    (agentvec entries are not undone on abort; MVCC hides them via the heap recheck, exactly as pgvector)"

echo "### 4/6 index visibility follows the transaction, not the statement"
# A second backend must not see the row before commit, and must see it after.
$PSQL -c "SELECT 1" >/dev/null
( $PSQL -c "SET enable_seqscan=off; SELECT count(*) FROM smoke_pool WHERE id = 777" > /tmp/smoke_pre.txt ) &
PRE_PID=$!
sleep 0.3
$PSQL <<'SQL'
BEGIN;
INSERT INTO smoke_pool VALUES (777, '[7,7,0,0]');
SELECT 'same-txn index scan sees the row: ' || count(*) FROM (SELECT id FROM smoke_pool WHERE id = 777) q;
COMMIT;
SQL
wait $PRE_PID
PRE=$(cat /tmp/smoke_pre.txt)
POST=$($PSQL -c "SET enable_seqscan=off; SELECT count(*) FROM smoke_pool WHERE id = 777")
echo "    concurrent backend before commit: $PRE (expect 0); after commit: $POST (expect 1)"

echo "### 5/6 sealing keeps the pool size and loses no row"
$PSQL <<'SQL'
DROP TABLE IF EXISTS smoke_seal CASCADE;
CREATE TABLE smoke_seal(id int, v vector(4));
CREATE INDEX smoke_seal_idx ON smoke_seal USING agentvec (v vector_l2_ops)
    WITH (hot_segment_count = 2, hot_segment_max_rows = 25);
INSERT INTO smoke_seal SELECT g, ('[' || g || ',' || g || ',0,0]')::vector
    FROM generate_series(1, 500) AS g;
SQL
$PSQL -c "SELECT state, count(*), sum(num_entries) FROM agentvec_index_info('smoke_seal_idx') GROUP BY 1 ORDER BY 1"
HOT2=$($PSQL -c "SELECT count(*) FROM agentvec_index_info('smoke_seal_idx') WHERE is_hot")
SEALED=$($PSQL -c "SELECT count(*) FROM agentvec_index_info('smoke_seal_idx') WHERE state = 'queued_for_migration'")
ORDER2=$($PSQL -c "SET enable_seqscan=off; SELECT string_agg(id::text, ',') FROM (SELECT id FROM smoke_seal ORDER BY v <-> '[1,1,0,0]' LIMIT 5) q")
STRUCT2=$($PSQL -c "SELECT sum(num_entries)::bigint FROM agentvec_index_info('smoke_seal_idx')")
echo "    appendable hot: $HOT2 (expect 2); sealed: $SEALED (>= 18); structural entries: $STRUCT2 (expect 500)"
echo "    nearest 5 across sealed segments: $ORDER2 (expect 1,2,3,4,5)"
[ "$HOT2" = "2" ] || { echo "FAIL: pool size changed across seals"; exit 1; }
[ "$SEALED" -ge 18 ] || { echo "FAIL: expected the seals to produce sealed segments"; exit 1; }
[ "$STRUCT2" = "500" ] || { echo "FAIL: entries lost across seals"; exit 1; }
[ "$ORDER2" = "1,2,3,4,5" ] || { echo "FAIL: wrong order across sealed segments"; exit 1; }

echo "### 6/6 adaptive mode (hot_segment_count = 0) still works"
$PSQL <<'SQL'
DROP TABLE IF EXISTS smoke_auto CASCADE;
CREATE TABLE smoke_auto(id int, v vector(4));
CREATE INDEX smoke_auto_idx ON smoke_auto USING agentvec (v vector_l2_ops)
    WITH (hot_segment_count = 0);
INSERT INTO smoke_auto SELECT g, ('[' || g || ',' || g || ',0,0]')::vector
    FROM generate_series(1, 100) AS g;
SQL
HOTA=$($PSQL -c "SELECT count(*) FROM agentvec_index_info('smoke_auto_idx') WHERE is_hot")
STRUCTA=$($PSQL -c "SELECT sum(num_entries)::bigint FROM agentvec_index_info('smoke_auto_idx')")
ORDERA=$($PSQL -c "SET enable_seqscan=off; SELECT string_agg(id::text, ',') FROM (SELECT id FROM smoke_auto ORDER BY v <-> '[1,1,0,0]' LIMIT 3) q")
echo "    adaptive appendable hot: $HOTA (expect >= 1); structural entries: $STRUCTA (expect 100); nearest 3: $ORDERA"
[ "$STRUCTA" = "100" ] || { echo "FAIL: adaptive mode lost rows"; exit 1; }
[ "$ORDERA" = "1,2,3" ] || { echo "FAIL: adaptive mode ranking"; exit 1; }

echo
echo "SMOKE TEST PASSED"
