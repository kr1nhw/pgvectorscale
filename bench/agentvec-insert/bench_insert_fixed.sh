#!/bin/bash
# bench_insert_fixed.sh <target> <clients> <total_rows> <nk> [tag] [pgbench-extra...]
#
# Like bench_insert.sh but runs a FIXED number of transactions in total
# (-t = total_rows/clients per client) instead of a fixed duration.  Every
# client count therefore inserts the same number of rows, so the hnswsq HOT
# graph grows by the same amount in every run and the scaling curve is not
# confounded by a different average graph size.
#
# Reports rows/s computed from the measured wall time, plus pgbench's own tps.
set -u
cd "$(dirname "$0")"
. ./env.sh
avb_header

TBL=$1; C=$2; TOTAL=$3; NK=${4:-200000}; TAG=${5:-fixed}; shift 5 || true
EXTRA=("$@")
J=$C; [ "$J" -gt 16 ] && J=16
T=$(( (TOTAL + C - 1) / C ))
[ "$T" -lt 1 ] && T=1

[ -f "ins_$TBL.sql" ] || ./gen_scripts.sh "$TBL" >/dev/null

T0=$(date +%s%N)
OUT=$($PGBENCH -n -c "$C" -j "$J" -t "$T" -D nk="$NK" "${EXTRA[@]+"${EXTRA[@]}"}" \
        -f "ins_$TBL.sql" 2>&1)
T1=$(date +%s%N)
MS=$(( (T1 - T0) / 1000000 ))
ROWS=$(( T * C ))
TPS=$(echo "$OUT" | awk '/tps = /{gsub(/,/,"",$3); print $3}' | head -1)
LAT=$(echo "$OUT" | awk -F= '/latency average/{gsub(/ /,"",$2); print $2}' | head -1)
FAIL=$(echo "$OUT" | awk -F= '/number of failed transactions/{gsub(/ /,"",$2); print $2}' | head -1)
if [ -z "$TPS" ]; then
    echo "FAILED: $TBL c=$C"; echo "$OUT" | tail -20 >&2; exit 1
fi
ROWSPS=$(awk -v r="$ROWS" -v ms="$MS" 'BEGIN{ if (ms>0) printf "%.0f", r*1000/ms; else print "?" }')
echo "insert  target=$TBL clients=$C rows=$ROWS wall=${MS}ms  rows/s=$ROWSPS  tps=$TPS  lat_avg_ms=${LAT:-?}  failed=${FAIL:-0}  tag=$TAG"
avb_record "insert-fixed" "$TBL" "$C" "$MS" "$ROWSPS" "${LAT:-}" "$TAG rows=$ROWS"
# Keep the per-client latency distribution for the record.
echo "$OUT" | grep -E '^ *[0-9]+(\.[0-9]+)?%' | sed "s/^/    $TBL c=$C /" >&2
