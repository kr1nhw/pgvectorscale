#!/bin/bash
# Build (and time) the benchmark index for one engine.
#
# Usage: build_indexes.sh <psql_bin> <engine:ivf|hnsw> <builds_csv>
#   psql_bin    psql executable path; connection comes from the standard
#               libpq env vars (PGHOST/PGPORT/PGUSER/PGDATABASE)
#   ivf  -> CREATE INDEX ... USING ivf (embedding vector_l2_ops)
#           WITH (lists=1000, num_bits=1)
#   hnsw -> CREATE INDEX ... USING hnsw (embedding vector_l2_ops)
#           WITH (m=16, ef_construction=64); maintenance_work_mem raised for the build
set -euo pipefail

PSQL="$1"
ENGINE="$2"
BUILDS_CSV="$3"

if [ "$ENGINE" = ivf ]; then
  IDX_NAME="items_10m_ivf"
  DDL="CREATE INDEX $IDX_NAME ON items_10m USING ivf (embedding vector_l2_ops) WITH (lists=1000, num_bits=1);"
else
  IDX_NAME="items_10m_hnsw"
  DDL="CREATE INDEX $IDX_NAME ON items_10m USING hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=64);"
fi

# Recreate from scratch
$PSQL -X -q -v ON_ERROR_STOP=1 -c "DROP INDEX IF EXISTS $IDX_NAME;" >/dev/null

START_MS=$(date +%s%3N)
if [ "$ENGINE" = hnsw ]; then
  $PSQL -X -q -v ON_ERROR_STOP=1 -c "SET maintenance_work_mem='2GB';" -c "$DDL" >/dev/null
else
  $PSQL -X -q -v ON_ERROR_STOP=1 -c "$DDL" >/dev/null
fi
END_MS=$(date +%s%3N)
BUILD_S=$(( (END_MS - START_MS) / 1000 ))

# validate + size
SIZE_B=$($PSQL -X -At -c "SELECT pg_relation_size('$IDX_NAME');")
VALID=$($PSQL -X -At -c "SELECT indisvalid FROM pg_index WHERE indexrelid = '$IDX_NAME'::regclass;")

echo "engine=$ENGINE build_s=$BUILD_S size_bytes=$SIZE_B valid=$VALID"
echo "${BUILD_LABEL:-$(hostname)},$ENGINE,$BUILD_S,$SIZE_B,$VALID" >> "$BUILDS_CSV"
