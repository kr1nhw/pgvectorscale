#!/bin/bash
# bench_insert.sh <target> <clients> <seconds> <nk> [tag]
#
# Runs pgbench with the one-row-per-transaction insert script and reports
# rows/s (= tps) plus the average client-side transaction latency.
set -u
cd "$(dirname "$0")"
. ./env.sh
avb_header

TBL=$1; C=$2; SECS=$3; NK=${4:-200000}; TAG=${5:-insert}
J=$C; [ "$J" -gt 16 ] && J=16

[ -f "ins_$TBL.sql" ] || ./gen_scripts.sh "$TBL" >/dev/null

OUT=$($PGBENCH -n -c "$C" -j "$J" -T "$SECS" -D nk="$NK" -f "ins_$TBL.sql" 2>&1)
TPS=$(echo "$OUT"  | awk '/tps = /{gsub(/,/,"",$3); print $3}' | head -1)
LAT=$(echo "$OUT"  | awk -F= '/latency average/{gsub(/ /,"",$2); print $2}' | head -1)
FAIL=$(echo "$OUT" | awk -F= '/number of failed transactions/{gsub(/ /,"",$2); print $2}' | head -1)
if [ -z "$TPS" ]; then
    echo "FAILED: $TBL c=$C"; echo "$OUT" | tail -20 >&2; exit 1
fi
echo "insert  target=$TBL clients=$C secs=$SECS  rows/s=$TPS  lat_avg_ms=${LAT:-?}  failed=${FAIL:-0}  tag=$TAG"
avb_record "insert" "$TBL" "$C" "$SECS" "$TPS" "${LAT:-}" "$TAG"
