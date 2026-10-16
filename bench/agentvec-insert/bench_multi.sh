#!/bin/bash
# bench_multi.sh <K> <total_clients> <seconds> <nk> [tag] [--prefill N]
#
# Emulates "K hot segments" using K independent tables, each with its own
# agentvec index (i.e. K independent hnswsq regions, each with its own region
# metapage, insert page, segment header page and entry point).  The total
# client count is split evenly across the K tables and the per-table tps is
# summed, so the comparison against K=1 answers:
#
#   "if the foreground write path were sharded across K hot graphs, how much
#    aggregate insert throughput would that buy?"
#
# The heap is per-table too, but the heap+btree path costs ~70 us of CPU per
# row versus ~1300-1800 us for the vector index (measured: cpu_per_row.sh), so
# it does not confound the comparison.
set -u
cd "$(dirname "$0")"
. ./env.sh
avb_header

K=$1; C=$2; SECS=$3; NK=${4:-200000}; TAG=${5:-multi}
PREFILL=20000
[ "${6:-}" = "--prefill" ] && PREFILL=${7:-20000}

J=$C; [ "$J" -gt 16 ] && J=16

# Split clients evenly; the first (C mod K) tables get one extra.
TABLES=()
for i in $(seq 1 "$K"); do TABLES+=("avb_mk_$i"); done

for t in "${TABLES[@]}"; do
    [ -f "ins_$t.sql" ] || ./gen_scripts.sh "$t" >/dev/null
done

read_cpu() { awk '/^cpu /{print $2, $3, $4, $7, $8, $5, $6, $9}' /proc/stat; }
CLK=$(getconf CLK_TCK 2>/dev/null || echo 100)
read -r u0 n0 s0 i0 si0 idle0 w0 st0 <<<"$(read_cpu)"
T0=$(date +%s%N)

PIDS=(); OUTS=()
for i in $(seq 1 "$K"); do
    base=$(( C / K )); extra=$(( C % K ))
    ci=$base; [ "$i" -le "$extra" ] && ci=$(( ci + 1 ))
    [ "$ci" -lt 1 ] && continue
    o=$AVB_DIR/.mk_${i}.$$
    OUTS+=("$o")
    $PGBENCH -n -c "$ci" -j "$ci" -T "$SECS" -D nk="$NK" -f "ins_${TABLES[$((i-1))]}.sql" >"$o" 2>&1 &
    PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
T1=$(date +%s%N)
read -r u1 n1 s1 i1 si1 idle1 w1 st1 <<<"$(read_cpu)"

TOTAL=0; LATSUM=0
for o in "${OUTS[@]}"; do
    t=$(awk '/tps = /{gsub(/,/,"",$3); print $3}' "$o" | head -1)
    l=$(awk -F= '/latency average/{gsub(/ /,"",$2); print $2}' "$o" | head -1)
    TOTAL=$(awk -v a="$TOTAL" -v b="${t:-0}" 'BEGIN{printf "%.1f", a+b}')
    LATSUM=$(awk -v a="$LATSUM" -v b="${l:-0}" 'BEGIN{printf "%.3f", a+b}')
    rm -f "$o"
done
AVGLAT=$(awk -v s="$LATSUM" -v k="$K" 'BEGIN{printf "%.3f", s/k}')
MS=$(( (T1 - T0) / 1000000 ))
BUSY=$(( (u1-u0)+(n1-n0)+(s1-s0)+(i1-i0)+(si1-si0) ))
TOT=$(( BUSY + (idle1-idle0) + (w1-w0) + (st1-st0) ))
[ "$TOT" -le 0 ] && TOT=1
CORES=$(awk -v b="$BUSY" -v c="$CLK" -v w="$MS" 'BEGIN{printf "%.1f", b*1000/c/w}')
BUSYPCT=$(awk -v b="$BUSY" -v t="$TOT" 'BEGIN{printf "%.1f", 100*b/t}')

echo "multi   K=$K clients=$C  aggregate_rows/s=$TOTAL  avg_lat_ms=$AVGLAT  wall=${MS}ms  cores=${CORES}/32 (${BUSYPCT}%)  tag=$TAG"
avb_record "multi-hot-emul" "K=$K" "$C" "$SECS" "$TOTAL" "$AVGLAT" "$TAG cores=$CORES busy=$BUSYPCT"
