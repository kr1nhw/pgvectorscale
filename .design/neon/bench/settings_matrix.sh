#!/bin/bash
# Per-configuration matrix on the 1M BIGANN table: build time, index size,
# query latency at ef 10/40/160/640 (LIMIT 10), and incremental-insert
# throughput/latency — for pgvector and every hnswsq storage layout.
#
# Usage: settings_matrix.sh   (run on the box, as root; sudo -u pgtest psql)
#
# For each configuration this builds the index fresh (m=16, efc=64, mwm 8GB,
# pinned build_seed for hnswsq), sweeps query latency with gap_sweep.py, then
# inserts INSERT_N fresh vectors (sliced per config from
# bigann_base_100M.bvecs rows 1M+, so no duplicates across configs) in BATCH-
# row statements, timed with clock_timestamp().
#
# Results: CSV on stdout and /tmp/settings_matrix.csv
set -uo pipefail

PSQL="${PSQL:-/root/.pgrx-hnswsq/17.11/pgrx-install/bin/psql}"
PORT="${PORT:-54330}"
BENCH="$(cd "$(dirname "$0")" && pwd)"
TABLE=items_1m
EFS="10 40 160 640"
INSERT_N=50000
BATCH=1000
SEED=20240912
OUT_CSV=/tmp/settings_matrix.csv
STAGING=/tmp/insert_staging.txt
BCONF=0

q() { sudo -u pgtest "$PSQL" -h 127.0.0.1 -p "$PORT" -U pgtest -d postgres -X -q -At -v ON_ERROR_STOP=1 "$@"; }
qin() { sudo -u pgtest "$PSQL" -h 127.0.0.1 -p "$PORT" -U pgtest -d postgres -X -q -At -v ON_ERROR_STOP=1; }

# ---- 0. staging: 5 x INSERT_N fresh vectors from the 100M base file -------
python3 - "$STAGING" "$INSERT_N" <<'PY'
import struct, sys
out, per = sys.argv[1], int(sys.argv[2])
total = 5 * per
start = 1_000_000  # items_1m holds rows 0..999999
# bigann_base_100M.bvecs: 4-byte dim + 128 uint8 per row (132 B/row)
with open('/data1/bigann/bigann_base_100M.bvecs', 'rb') as f, open(out, 'w') as w:
    d = struct.unpack('<i', f.read(4))[0]
    assert d == 128, d
    f.seek(start * (4 + d))
    for _ in range(total):
        raw = f.read(4 + d)
        vec = struct.unpack('<128B', raw[4:])
        w.write('[' + ','.join(map(str, vec)) + ']\n')
print(f"staged {total} vectors -> {out}")
PY

drop_indexes() {
  for idx in $(q -c "SELECT indexname FROM pg_indexes WHERE tablename='$TABLE' AND indexname <> '${TABLE}_pkey';"); do
    q -c "DROP INDEX IF EXISTS $idx;" >/dev/null
  done
}

# query sweep via gap_sweep.py; echoes "q10,q40,q160,q640" (mean ms, LIMIT 10)
query_sweep() { # $1 = engine name
  python3 "$BENCH/gap_sweep.py" "$1" "$(echo $EFS | tr ' ' ',')" "$TABLE" 100 /tmp/gap_sweep_matrix.csv \
    | awk -F, '$3==10 {printf "%s,", $5} END {print ""}' | sed 's/,$//'
}

# insert bench; echoes "elapsed_s,rows_per_s,row_ms_mean,row_ms_p50,row_ms_p99"
insert_bench() {
  local skip=$((BCONF * INSERT_N))
  q -c "DROP TABLE IF EXISTS insert_staging;" >/dev/null
  q -c "CREATE UNLOGGED TABLE insert_staging (rn bigserial PRIMARY KEY, embedding vector(128));" >/dev/null
  q -c "\copy insert_staging (embedding) FROM PROGRAM 'tail -n +$((skip + 1)) $STAGING | head -n $INSERT_N'" >/dev/null
  local staged
  staged=$(q -c "SELECT count(*) FROM insert_staging;")
  if [ "$staged" != "$INSERT_N" ]; then
    echo "staged $staged rows, expected $INSERT_N" >&2
    return
  fi

  local t0 t1 elapsed stats
  t0=$(date +%s.%N)
  # One session: the temp table must survive from the DO block to the stats.
  stats=$(qin <<SQL
CREATE TEMP TABLE batch_times (batch int, ms float8);
DO \$\$
DECLARE
    i int := 0;
    done int;
    t0 timestamptz;
    base_id int := (SELECT coalesce(max(id), 0) FROM $TABLE);
BEGIN
    LOOP
        t0 := clock_timestamp();
        INSERT INTO $TABLE (id, embedding)
        SELECT base_id + rn, embedding FROM insert_staging WHERE rn > i AND rn <= i + $BATCH ORDER BY rn;
        GET DIAGNOSTICS done = ROW_COUNT;
        EXIT WHEN done = 0;
        INSERT INTO batch_times VALUES (i / $BATCH, extract(epoch FROM clock_timestamp() - t0) * 1000.0);
        i := i + done;
    END LOOP;
END
\$\$;
WITH b AS (SELECT ms / $BATCH AS row_ms FROM batch_times)
SELECT (SELECT count(*) FROM batch_times) * $BATCH || ',' ||
       round(1000.0 / (SELECT avg(row_ms) FROM b))::bigint || ',' ||
       round((SELECT avg(row_ms) FROM b)::numeric, 3) || ',' ||
       round((SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY row_ms) FROM b)::numeric, 3) || ',' ||
       round((SELECT percentile_cont(0.99) WITHIN GROUP (ORDER BY row_ms) FROM b)::numeric, 3);
SQL
)
  t1=$(date +%s.%N)
  elapsed=$(awk "BEGIN {printf \"%.3f\", $t1 - $t0}")
  printf "%s,%s" "$elapsed" "$stats"
  q -c "DROP TABLE insert_staging;" >/dev/null
}

run_config() { # $1 = engine  $2 = layout label  $3 = DDL
  local engine="$1" layout="$2"
  drop_indexes
  local t0 t1
  t0=$(date +%s.%N)
  qin <<SQL
SET maintenance_work_mem = '8GB';
$([ "$engine" = hnswsq ] && echo "SET hnswsq.build_seed = $SEED;")
$3
SQL
  t1=$(date +%s.%N)
  local build_s size qms ins
  build_s=$(awk "BEGIN {printf \"%.3f\", $t1 - $t0}")
  size=$(q -c "SELECT pg_relation_size('items_1m_matrix');")
  qms=$(query_sweep "$engine")
  ins=$(insert_bench)
  echo "$engine,$layout,$build_s,$size,$qms,$ins"
  BCONF=$((BCONF + 1))
}

{
  echo "engine,layout,build_s,size_bytes,qms_ef10,qms_ef40,qms_ef160,qms_ef640,ins_elapsed_s,ins_rows_per_s,ins_row_ms_mean,ins_row_ms_p50,ins_row_ms_p99"
  run_config hnsw plain "CREATE INDEX items_1m_matrix ON $TABLE USING hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=64);"
  run_config hnswsq plain "CREATE INDEX items_1m_matrix ON $TABLE USING hnswsq (embedding vector_l2_ops) WITH (storage_layout='plain', m=16, ef_construction=64);"
  run_config hnswsq ieeefp16 "CREATE INDEX items_1m_matrix ON $TABLE USING hnswsq (embedding vector_l2_ops) WITH (storage_layout='ieeefp16', m=16, ef_construction=64);"
  run_config hnswsq ieeefp8 "CREATE INDEX items_1m_matrix ON $TABLE USING hnswsq (embedding vector_l2_ops) WITH (storage_layout='ieeefp8', m=16, ef_construction=64);"
  run_config hnswsq f8 "CREATE INDEX items_1m_matrix ON $TABLE USING hnswsq (embedding vector_l2_ops) WITH (storage_layout='f8', m=16, ef_construction=64);"
} | tee "$OUT_CSV"
