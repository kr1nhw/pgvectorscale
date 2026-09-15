#!/bin/bash
# Build (and time) the benchmark index for one engine.  hnswsq A/B variant
# of build_indexes.sh: engines are `hnsw` (pgvector) and `hnswsq`.
#
# Usage: build_indexes_hnswsq.sh <psql_bin> <engine:hnsw|hnswsq> <builds_csv>
#   hnsw   -> CREATE INDEX ... USING hnsw (embedding vector_l2_ops)
#             WITH (m=16, ef_construction=64); maintenance_work_mem raised
#   hnswsq -> CREATE INDEX ... USING hnswsq (embedding vector_l2_ops)
#             WITH (m=16, ef_construction=64, storage_layout=plain)
#
# Both builds run with maintenance_work_mem=8GB: hnswsq is targeted at
# serverless-scale datasets (1M-10M rows); at 8GB the 1M in-memory graph
# builds in a single pass with no spilling.
set -euo pipefail

PSQL="$1"
ENGINE="$2"
BUILDS_CSV="$3"

if [ "$ENGINE" = hnsw ]; then
  IDX_NAME="items_10m_hnsw"
  DDL="CREATE INDEX $IDX_NAME ON items_10m USING hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=64);"
else
  IDX_NAME="items_10m_hnswsq"
  DDL="CREATE INDEX $IDX_NAME ON items_10m USING hnswsq (embedding vector_l2_ops) WITH (m=16, ef_construction=64, storage_layout=plain);"
fi

# Recreate from scratch
$PSQL -X -q -v ON_ERROR_STOP=1 -c "DROP INDEX IF EXISTS $IDX_NAME;" >/dev/null

START_MS=$(date +%s%3N)
if [ "$ENGINE" = hnsw ]; then
  $PSQL -X -q -v ON_ERROR_STOP=1 -c "SET maintenance_work_mem='8GB';" -c "$DDL" >/dev/null
else
  $PSQL -X -q -v ON_ERROR_STOP=1 -c "SET maintenance_work_mem='8GB';" -c "$DDL" >/dev/null
fi
END_MS=$(date +%s%3N)
BUILD_S=$(( (END_MS - START_MS) / 1000 ))

# validate + size
SIZE_B=$($PSQL -X -At -c "SELECT pg_relation_size('$IDX_NAME');")
VALID=$($PSQL -X -At -c "SELECT indisvalid FROM pg_index WHERE indexrelid = '$IDX_NAME'::regclass;")

echo "engine=$ENGINE build_s=$BUILD_S size_bytes=$SIZE_B valid=$VALID"
echo "${BUILD_LABEL:-$(hostname)},$ENGINE,$BUILD_S,$SIZE_B,$VALID" >> "$BUILDS_CSV"
