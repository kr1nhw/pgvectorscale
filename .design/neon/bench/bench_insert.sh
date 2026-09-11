#!/bin/bash
# Incremental-insert benchmark: how long does it take to add fresh rows to an
# index that already holds the dataset?
#
# Usage: bench_insert.sh <psql_bin> <engine:hnsw|hnswsq> <db> <label> <out_csv>
#
# Environment:
#   PGPORT        server port            (default 54329)
#   TABLE         target table           (default items_10m, PK id bigserial)
#   STAGING_FILE  text file with one vector literal per line (default
#                 /data1/bigann_100m_vectors.txt); rows are inserted in order
#   N             rows to insert          (default 100000)
#   BATCH         rows per timed batch    (default 1000)
#   SKIP          lines to skip in the staging file (default 0)
#
# Appends to out_csv:
#   label,engine,base_rows,inserted,elapsed_s,rows_per_s,row_ms_mean,row_ms_p50,row_ms_p99,idx_size_bytes
#
# Timing is measured inside one session with clock_timestamp() around each
# BATCH-row INSERT, so it captures the real per-row index maintenance cost
# (aminsert + backlink updates + WAL) without per-statement client overhead.
set -euo pipefail

PSQL="$1"
ENGINE="$2"
DB="$3"
LABEL="$4"
OUT_CSV="$5"

PORT="${PGPORT:-54329}"
TABLE="${TABLE:-items_10m}"
STAGING_FILE="${STAGING_FILE:-/data1/bigann_100m_vectors.txt}"
N="${N:-100000}"
BATCH="${BATCH:-1000}"
SKIP="${SKIP:-0}"

PSQL_DB="$PSQL -h 127.0.0.1 -p $PORT -U pgtest -d $DB -X -q -v ON_ERROR_STOP=1"

BASE_ROWS=$($PSQL_DB -At -c "SELECT count(*) FROM $TABLE;")
IDX="$($PSQL_DB -At -c "SELECT indexrelid::regclass FROM pg_index WHERE indrelid='$TABLE'::regclass AND indexrelid::regclass::text LIKE '${TABLE}_%';" | head -1)"
IDX_BEFORE=$($PSQL_DB -At -c "SELECT pg_relation_size('$IDX');")

# Stage the fresh vectors (untimed; this is data movement, not index work).
$PSQL_DB -c "DROP TABLE IF EXISTS insert_staging;" >/dev/null
$PSQL_DB -c "CREATE UNLOGGED TABLE insert_staging (rn bigserial PRIMARY KEY, embedding vector(128));" >/dev/null
if [ "$SKIP" -gt 0 ]; then
  $PSQL_DB -c "\copy insert_staging (embedding) FROM PROGRAM 'tail -n +$((SKIP+1)) $STAGING_FILE | head -n $N'" >/dev/null
else
  $PSQL_DB -c "\copy insert_staging (embedding) FROM PROGRAM 'head -n $N $STAGING_FILE'" >/dev/null
fi
STAGED=$($PSQL_DB -At -c "SELECT count(*) FROM insert_staging;")
[ "$STAGED" = "$N" ] || { echo "staged $STAGED rows, expected $N" >&2; exit 1; }

$PSQL_DB -c "DROP TABLE IF EXISTS batch_times; CREATE TEMP TABLE batch_times (batch int, ms float8);" >/dev/null

START_MS=$(date +%s%3N)
$PSQL_DB -c "
DO \$\$
DECLARE
    i int := 0;
    done int;
    t0 timestamptz;
BEGIN
    LOOP
        t0 := clock_timestamp();
        INSERT INTO $TABLE (embedding)
        SELECT embedding FROM insert_staging WHERE rn > i AND rn <= i + $BATCH ORDER BY rn;
        GET DIAGNOSTICS done = ROW_COUNT;
        EXIT WHEN done = 0;
        INSERT INTO batch_times VALUES (i / $BATCH, extract(epoch FROM clock_timestamp() - t0) * 1000.0);
        i := i + done;
    END LOOP;
END
\$\$;
" >/dev/null
END_MS=$(date +%s%3N)
ELAPSED_S=$(( (END_MS - START_MS) / 1000 ))

IDX_AFTER=$($PSQL_DB -At -c "SELECT pg_relation_size('$IDX');")

STATS=$($PSQL_DB -At -F' ' -c "
WITH b AS (SELECT ms / $BATCH AS row_ms FROM batch_times)
SELECT (SELECT count(*) FROM batch_times) * $BATCH,
       round(1000.0 / (SELECT avg(row_ms) FROM b))::bigint,
       round((SELECT avg(row_ms) FROM b)::numeric, 3),
       round((SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY row_ms) FROM b)::numeric, 3),
       round((SELECT percentile_cont(0.99) WITHIN GROUP (ORDER BY row_ms) FROM b)::numeric, 3);")
read -r INSERTED RPS MEAN P50 P99 <<< "$STATS"

echo "engine=$ENGINE base=$BASE_ROWS inserted=$INSERTED elapsed_s=$ELAPSED_S rows_per_s=$RPS row_ms(mean/p50/p99)=$MEAN/$P50/$P99 idx_bytes=$IDX_BEFORE->$IDX_AFTER"
echo "$LABEL,$ENGINE,$BASE_ROWS,$INSERTED,$ELAPSED_S,$RPS,$MEAN,$P50,$P99,$IDX_AFTER" >> "$OUT_CSV"

$PSQL_DB -c "DROP TABLE insert_staging;" >/dev/null
