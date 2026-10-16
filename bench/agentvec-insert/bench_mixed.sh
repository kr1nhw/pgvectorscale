#!/bin/bash
# bench_mixed.sh <ins_target> <ins_clients> <q_target> <q_clients> <seconds> [nk] [nq] [tag]
#
# Runs an insert workload and a query workload CONCURRENTLY (two pgbench
# processes) and reports both throughputs, plus CPU utilisation over the
# overlapping window.  This is the "insertion + query combined" measurement:
# what happens to query QPS while the index is taking inserts, and vice versa.
#
# The insert run gets a one-second head start (and is one second longer) so the
# query window measures steady insert load rather than ramp-up.
set -u
cd "$(dirname "$0")"
. ./env.sh
avb_header

IT=$1; IC=$2; QT=$3; QC=$4; SECS=$5; NK=${6:-200000}; NQ=${7:-10000}; TAG=${8:-mixed}
IJ=$IC; [ "$IJ" -gt 16 ] && IJ=16
QJ=$QC; [ "$QJ" -gt 16 ] && QJ=16

[ -f "ins_$IT.sql" ] || ./gen_scripts.sh "$IT" >/dev/null
[ -f "q_$QT.sql" ]   || ./gen_scripts.sh "$QT" >/dev/null

IOUT=$AVB_DIR/.mixed_ins.$$; QOUT=$AVB_DIR/.mixed_q.$$
CPUOUT=$AVB_DIR/.mixed_cpu.$$

read_cpu() { awk '/^cpu /{print $2, $3, $4, $7, $8, $5, $6, $9}' /proc/stat; }
CLK=$(getconf CLK_TCK 2>/dev/null || echo 100)
read -r u0 n0 s0 i0 si0 idle0 w0 st0 <<<"$(read_cpu)"

$PGBENCH -n -c "$IC" -j "$IJ" -T "$((SECS + 1))" -D nk="$NK" \
    --progress=1 -f "ins_$IT.sql" >"$IOUT" 2>&1 &
IPID=$!
sleep 1
$PGBENCH -n -c "$QC" -j "$QJ" -T "$SECS" -D nq="$NQ" \
    --progress=1 -f "q_$QT.sql" >"$QOUT" 2>&1 &
QPID=$!
wait $QPID
read -r u1 n1 s1 i1 si1 idle1 w1 st1 <<<"$(read_cpu)"
QT1=$(awk '/tps = /{gsub(/,/,"",$3); print $3}' "$QOUT" | head -1)
QLAT=$(awk -F= '/latency average/{gsub(/ /,"",$2); print $2}' "$QOUT" | head -1)
wait $IPID

ITPS=$(awk '/tps = /{gsub(/,/,"",$3); print $3}' "$IOUT" | head -1)
ILAT=$(awk -F= '/latency average/{gsub(/ /,"",$2); print $2}' "$IOUT" | head -1)
# Steady-state insert tps over the window overlapping the query run.
# pgbench progress line: "progress: 1.0 s, 1234.5 tps, lat 1.23 ms stddev ..."
ITPS_SS=$(awk '/^progress/{gsub(/,/,"",$4); print $4}' "$IOUT" | tail -n "$SECS" | \
          awk '{s+=$1; n++} END{if(n)printf "%.0f", s/n}')

BUSY=$(( (u1-u0)+(n1-n0)+(s1-s0)+(i1-i0)+(si1-si0) ))
TOTAL=$(( BUSY + (idle1-idle0) + (w1-w0) + (st1-st0) ))
[ "$TOTAL" -le 0 ] && TOTAL=1
CORES=$(awk -v b="$BUSY" -v c="$CLK" -v s="$SECS" 'BEGIN{printf "%.1f", b/c/s}')
BUSYPCT=$(awk -v b="$BUSY" -v t="$TOTAL" 'BEGIN{printf "%.1f", 100*b/t}')

[ -z "$QT1" ] && { echo "FAILED(query)"; tail -5 "$QOUT" >&2; }
[ -z "$ITPS" ] && { echo "FAILED(insert)"; tail -5 "$IOUT" >&2; }
rm -f "$IOUT" "$QOUT" "$CPUOUT"

echo "mixed   ins=$IT c=$IC rows/s=${ITPS_SS:-$ITPS} (avg $ITPS) lat_ms=${ILAT:-?} | query=$QT c=$QC q/s=${QT1:-?} lat_ms=${QLAT:-?} | cores=${CORES}/${BUSYPCT}%  tag=$TAG"
avb_record "mixed-insert" "$IT" "$IC" "$SECS" "${ITPS_SS:-$ITPS}" "${ILAT:-}" "$TAG q=$QT/$QC qps=${QT1:-}"
avb_record "mixed-query"  "$QT" "$QC" "$SECS" "${QT1:-}" "${QLAT:-}" "$TAG ins=$IT/$IC rows/s=${ITPS_SS:-$ITPS}"
