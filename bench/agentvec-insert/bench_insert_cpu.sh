#!/bin/bash
# bench_insert_cpu.sh <target> <clients> <seconds> <nk> [tag]
#
# Runs the insert workload with pgbench and samples CPU utilisation at the
# same time, so a run can be classified as CPU-bound or contention-bound.
set -u
cd "$(dirname "$0")"
. ./env.sh
avb_header

TBL=$1; C=$2; SECS=$3; NK=${4:-200000}; TAG=${5:-insert-cpu}
J=$C; [ "$J" -gt 16 ] && J=16

[ -f "ins_$TBL.sql" ] || ./gen_scripts.sh "$TBL" >/dev/null

CPUOUT=$AVB_DIR/.cpu.$$
./cpu_sample.sh "$SECS" >"$CPUOUT" 2>&1 &
CPUPID=$!
sleep 1

OUT=$($PGBENCH -n -c "$C" -j "$J" -T "$SECS" -D nk="$NK" -f "ins_$TBL.sql" 2>&1)

wait $CPUPID 2>/dev/null
TPS=$(echo "$OUT" | awk '/tps = /{gsub(/,/,"",$3); print $3}' | head -1)
LAT=$(echo "$OUT" | awk -F= '/latency average/{gsub(/ /,"",$2); print $2}' | head -1)
CPU=$(cat "$CPUOUT"); rm -f "$CPUOUT"
[ -z "$TPS" ] && { echo "FAILED: $TBL c=$C"; echo "$OUT" | tail -20 >&2; exit 1; }
echo "insert  target=$TBL clients=$C  rows/s=$TPS  lat_avg_ms=${LAT:-?}  $CPU  tag=$TAG"
avb_record "insert-cpu" "$TBL" "$C" "$SECS" "$TPS" "${LAT:-}" "$TAG | $CPU"
