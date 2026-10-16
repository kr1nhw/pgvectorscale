#!/bin/bash
# cpu_per_row.sh <target> <clients> <seconds> <nk> [tag] [pgbench-extra...]
#
# Measures the CPU time actually consumed per committed row.
#
# Snapshots the aggregate CPU counters from /proc/stat (user+nice+sys+irq+softirq,
# in clock ticks) around the pgbench window and divides by the rows inserted.
# The database is the only significant CPU consumer on this host, so the
# system-wide delta is a close enough attribution.
#
# This is the number that decides the right fix:
#   * high CPU-us/row and the box near 100% busy  -> the per-row work must shrink
#   * low  CPU-us/row and the box well under 100% -> the row is stalled on a
#     serialized section (WAL flush, one hot page) and more write parallelism
#     (e.g. several hot segments) can recover throughput
set -u
cd "$(dirname "$0")"
. ./env.sh
avb_header

TBL=$1; C=$2; SECS=$3; NK=${4:-200000}; TAG=${5:-cpu-per-row}; shift 5 || true
EXTRA=("$@")
J=$C; [ "$J" -gt 16 ] && J=16
[ -f "ins_$TBL.sql" ] || ./gen_scripts.sh "$TBL" >/dev/null

CLK=$(getconf CLK_TCK 2>/dev/null || echo 100)

# (user, nice, system, irq, softirq, idle, iowait, steal) from /proc/stat.
read_cpu() {
    awk '/^cpu /{print $2, $3, $4, $7, $8, $5, $6, $9}' /proc/stat
}

read -r u0 n0 s0 i0 si0 idle0 w0 st0 <<<"$(read_cpu)"
T0=$(date +%s%N)

PGOUT=$AVB_DIR/.cpr.$$
$PGBENCH -n -c "$C" -j "$J" -T "$SECS" -D nk="$NK" "${EXTRA[@]+"${EXTRA[@]}"}" \
    -f "ins_$TBL.sql" >"$PGOUT" 2>&1

T1=$(date +%s%N)
read -r u1 n1 s1 i1 si1 idle1 w1 st1 <<<"$(read_cpu)"

TPS=$(awk '/tps = /{gsub(/,/,"",$3); print $3}' "$PGOUT" | head -1)
LAT=$(awk -F= '/latency average/{gsub(/ /,"",$2); print $2}' "$PGOUT" | head -1)
NROWS=$(awk -F: '/number of transactions actually processed/{gsub(/,/,"",$2); print $2+0}' "$PGOUT")
rm -f "$PGOUT"
if [ -z "$TPS" ]; then echo "FAILED: $TBL c=$C" >&2; exit 1; fi

BUSY=$(( (u1-u0)+(n1-n0)+(s1-s0)+(i1-i0)+(si1-si0) ))
TOTAL=$(( BUSY + (idle1-idle0) + (w1-w0) + (st1-st0) ))
WALLMS=$(( (T1 - T0) / 1000000 ))
[ "$TOTAL" -le 0 ] && TOTAL=1

CPUR=$(awk -v b="$BUSY" -v c="$CLK" -v r="$NROWS" 'BEGIN{ if(r>0) printf "%.0f", b*1000000/c/r; else print "?" }')
BUSYPCT=$(awk -v b="$BUSY" -v t="$TOTAL" 'BEGIN{printf "%.1f", 100*b/t}')
CORES=$(awk -v b="$BUSY" -v c="$CLK" -v w="$WALLMS" 'BEGIN{ if(w>0) printf "%.1f", b*1000/c/w; else print "?" }')
IOWPCT=$(awk -v w="$((w1-w0))" -v t="$TOTAL" 'BEGIN{printf "%.1f", 100*w/t}')

echo "cpu     target=$TBL clients=$C  rows/s=${TPS}  rows=$NROWS  lat_avg_ms=${LAT:-?}  cpu_us/row=${CPUR}  cores_busy=${CORES}/32 (${BUSYPCT}%)  iowait=${IOWPCT}%  tag=$TAG"
avb_record "cpu-per-row" "$TBL" "$C" "$SECS" "$CPUR" "${LAT:-}" "$TAG rows/s=$TPS rows=$NROWS cores=$CORES busy=$BUSYPCT iowait=$IOWPCT"
