#!/bin/bash
# Concurrent DML + query smoke test against the Neon compute (55432)
PSQL="/data1/neon-test/pg_install/v17/bin/psql -h 127.0.0.1 -p 55432 -U cloud_admin -d postgres -q -v ON_ERROR_STOP=1"
PIDS=()

# 4 concurrent inserters, 100 rows each
for w in 1 2 3 4; do
  (
    for i in $(seq 1 100); do
      $PSQL -c "INSERT INTO items (embedding) SELECT (ARRAY[3+(random()-0.5),3+(random()-0.5),3+(random()-0.5),3+(random()-0.5),3+(random()-0.5),3+(random()-0.5),3+(random()-0.5),3+(random()-0.5)]::real[])::vector;" >/dev/null 2>&1 || echo "INS-W$w-FAIL"
    done
  ) &
  PIDS+=($!)
done

# 2 concurrent queriers, 50 queries each
for w in 1 2; do
  (
    for i in $(seq 1 50); do
      $PSQL -c "SELECT count(*) FROM (SELECT id FROM items ORDER BY embedding <-> '[3,3,3,3,3,3,3,3]'::vector LIMIT 10) s;" >/dev/null 2>&1 || echo "QRY-W$w-FAIL"
    done
  ) &
  PIDS+=($!)
done

# 1 deleter that removes its own inserts again (MVCC churn on the index)
(
  for i in $(seq 1 100); do
    $PSQL -c "DELETE FROM items WHERE id IN (SELECT id FROM items WHERE id > 20010 ORDER BY id DESC LIMIT 3);" >/dev/null 2>&1
  done
) &
PIDS+=($!)

FAIL=0
for p in "${PIDS[@]}"; do
  wait $p || FAIL=1
done

echo "concurrent test finished, fail=$FAIL"
$PSQL -c "SELECT count(*) AS nrows FROM items;"
$PSQL -c "SET enable_seqscan=off; SELECT id, round((embedding <-> '[3,3,3,3,3,3,3,3]'::vector)::numeric,3) AS dist FROM items ORDER BY embedding <-> '[3,3,3,3,3,3,3,3]'::vector LIMIT 3;"
$PSQL -c "VACUUM items; SELECT count(*) AS nrows_after_vacuum FROM items;"
$PSQL -c "SET enable_seqscan=off; SELECT id, round((embedding <-> '[3,3,3,3,3,3,3,3]'::vector)::numeric,3) AS dist FROM items ORDER BY embedding <-> '[3,3,3,3,3,3,3,3]'::vector LIMIT 3;"
