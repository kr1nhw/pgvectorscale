#!/bin/bash
# Insert-throughput benchmark for an existing hnsw/hnswsq index.
#
# Usage: bench_insert.sh <psql_bin> <engine:hnsw|hnswsq> <db> <label> <out_csv>
# Appends 100K fresh vectors (from /data1/bigann_100m_vectors.txt, skipping
# the first 10M rows so they are new to both bench DBs) into items_10m and
# times the INSERT.  Reports rows/sec and the index size delta.
#
# out_csv columns: label,engine,base_rows,inserted,elapsed_s,rows_per_s,idx_size_bytes
set -euo pipefail

PSQL="$1"
ENGINE="$2"
DB="$3"
LABEL="$4"
OUT_CSV="$5"

PSQL_DB="$PSQL -h 127.0.0.1 -p 54329 -U pgtest -d $DB -X -q -v ON_ERROR_STOP=1"

BASE_ROWS=$($PSQL_DB -At -c "SELECT count(*) FROM items_10m;")
IDX="$($PSQL_DB -At -c "SELECT indexrelid::regclass FROM pg_index WHERE indrelid='items_10m'::regclass AND indexrelid::regclass::text LIKE 'items_10m_%';" | head -1)"
IDX_BEFORE=$($PSQL_DB -At -c "SELECT pg_relation_size('$IDX');")

$PSQL_DB -c "CREATE UNLOGGED TABLE IF NOT EXISTS insert_staging (embedding vector(128));" >/dev/null
$PSQL_DB -c "TRUNCATE insert_staging;" >/dev/null
$PSQL_DB -c "\copy insert_staging (embedding) FROM PROGRAM 'tail -n +10000001 /data1/bigann_100m_vectors.txt | head -100000'" >/dev/null
N=$($PSQL_DB -At -c "SELECT count(*) FROM insert_staging;")

START_MS=$(date +%s%3N)
$PSQL_DB -c "INSERT INTO items_10m (embedding) SELECT embedding FROM insert_staging;" >/dev/null
END_MS=$(date +%s%3N)
ELAPSED_S=$(( (END_MS - START_MS) / 1000 ))

IDX_AFTER=$($PSQL_DB -At -c "SELECT pg_relation_size('$IDX');")
RPS=$(( N / (ELAPSED_S > 0 ? ELAPSED_S : 1) ))

echo "engine=$ENGINE base=$BASE_ROWS inserted=$N elapsed_s=$ELAPSED_S rows_per_s=$RPS idx_bytes=$IDX_BEFORE->$IDX_AFTER"
echo "$LABEL,$ENGINE,$BASE_ROWS,$N,$ELAPSED_S,$RPS,$IDX_AFTER" >> "$OUT_CSV"

$PSQL_DB -c "DROP TABLE insert_staging;" >/dev/null
