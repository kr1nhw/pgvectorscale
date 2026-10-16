#!/bin/bash
# setup_multi.sh <K> [prefill_rows] [layout]
#
# Creates K independent insert targets avb_mk_1..avb_mk_K, each with its own
# agentvec index and each prefilled to <prefill_rows> so that every emulated
# "hot segment" holds a graph of the same size.
set -eu
cd "$(dirname "$0")"
. ./env.sh

K=$1
PREFILL=${2:-20000}
LAYOUT=${3:-f8}

for i in $(seq 1 "$K"); do
    t="avb_mk_$i"
    $PSQLX -q -c "DROP TABLE IF EXISTS $t CASCADE" 2>/dev/null
    $PSQLX -q -c "CREATE TABLE $t (id bigserial PRIMARY KEY, embedding vector(128))"
    $PSQLX -q -c "CREATE INDEX ${t}_idx ON $t USING agentvec (embedding)
        WITH (ivf_lists='2000', rabitq_bits='1', hot_segment_max_rows='50000',
              ivf_probes='256', search_candidates='8000',
              hot_storage_layout='$LAYOUT')"
    if [ "$PREFILL" -gt 0 ]; then
        $PSQLX -q -c "INSERT INTO $t (embedding) SELECT embedding FROM avb_pool ORDER BY k LIMIT $PREFILL"
    fi
done
echo "created $K targets (layout=$LAYOUT, prefill=$PREFILL)"
