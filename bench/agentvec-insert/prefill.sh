#!/bin/bash
# prefill.sh <target> <rows>
#
# Fills the target with <rows> rows from avb_pool in one set-based transaction.
# For avb_t_av this is what grows the single HOT hnswsq segment to a given
# graph size before a timed insert run.  Keep <rows> below
# hot_segment_max_rows or the segment seals and a fresh empty one opens.
set -eu
cd "$(dirname "$0")"
. ./env.sh

TBL=$1; N=$2
$PSQLX -v ON_ERROR_STOP=1 -c \
  "INSERT INTO $TBL (embedding) SELECT embedding FROM avb_pool ORDER BY k LIMIT $N" >/dev/null
CNT=$($PSQL -c "SELECT count(*) FROM $TBL")
echo "prefill $TBL rows=$CNT"
if [ "$TBL" = "avb_t_av" ]; then
    $PSQLX -c "SELECT segment_id, level, algorithm, is_hot, num_entries FROM agentvec_index_info('avb_t_av_idx')" \
        | sed 's/^/    /'
fi
