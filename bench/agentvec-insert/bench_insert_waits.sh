#!/bin/bash
# bench_insert_waits.sh <target> <clients> <seconds> <nk> [tag]
#
# Runs the insert workload and samples pg_stat_activity wait events during it.
set -u
cd "$(dirname "$0")"
. ./env.sh
avb_header

TBL=$1; C=$2; SECS=$3; NK=${4:-200000}; TAG=${5:-waits}
J=$C; [ "$J" -gt 16 ] && J=16
[ -f "ins_$TBL.sql" ] || ./gen_scripts.sh "$TBL" >/dev/null

WOUT=$AVB_DIR/.waits.$$
./waits.sh "$SECS" >"$WOUT" 2>&1 &
WPID=$!
sleep 1
OUT=$($PGBENCH -n -c "$C" -j "$J" -T "$SECS" -D nk="$NK" -f "ins_$TBL.sql" 2>&1)
wait $WPID 2>/dev/null

TPS=$(echo "$OUT" | awk '/tps = /{gsub(/,/,"",$3); print $3}' | head -1)
LAT=$(echo "$OUT" | awk -F= '/latency average/{gsub(/ /,"",$2); print $2}' | head -1)
echo "insert  target=$TBL clients=$C  rows/s=${TPS:-FAIL}  lat_avg_ms=${LAT:-?}  tag=$TAG"
cat "$WOUT"
cp "$WOUT" "$AVB_DIR/waits_${TBL}_c${C}.txt" 2>/dev/null
rm -f "$WOUT"
