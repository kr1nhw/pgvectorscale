#!/bin/bash
# Benchmark recall@10 + latency for the current IVF index at several probes.
# Usage: bench_num_bits.sh <suffix> <probe1,probe2,...>
set -e
SUF="$1"
PROBES="$2"
PSQL="/data1/pg17/bin/psql -h /tmp -U postgres -d bench -X -v ON_ERROR_STOP=1"

# Load the library so ivf.* GUCs register.
$PSQL -c "EXPLAIN (COSTS OFF) SELECT id FROM items_10m ORDER BY embedding <-> (SELECT q FROM bench_queries WHERE qid=0) LIMIT 10;" >/dev/null 2>&1 || true

for np in $(echo "$PROBES" | tr ',' ' '); do
  restable="res_final_${SUF}_np${np}"
  $PSQL <<SQL 2>&1 | grep -E '^Time:' | sed "s/^/np=$np /"
SET ivf.probes = $np;
SET ivf.top_k = 1000;
DROP TABLE IF EXISTS ${restable};
CREATE TABLE ${restable} (qid int PRIMARY KEY, ids int[]);
\timing on
INSERT INTO ${restable}
SELECT qid, (SELECT array_agg(id) FROM (
  SELECT id FROM items_10m ORDER BY embedding <-> q.q LIMIT 10) t)
FROM (SELECT qid, q FROM bench_queries WHERE qid < 100) q;
\timing off
SQL
done
