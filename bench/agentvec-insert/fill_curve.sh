#!/bin/bash
# fill_curve.sh <target> <clients> <chunk_rows> <chunks> <nk> [tag]
#
# Measures insert throughput while the HOT segment fills up, by running one
# fixed-size pgbench batch at a time and recording the segment's row count
# before/after each batch.  This is the sizing input for "how big may a HOT
# segment be before inserts slow down", and it also exposes the seal event
# (at hot_segment_max_rows) and any maintenance interference.
set -u
cd "$(dirname "$0")"
. ./env.sh
avb_header

TBL=$1; C=$2; CHUNK=$3; CHUNKS=$4; NK=${5:-200000}; TAG=${6:-fill}
IDX="${TBL}_idx"
[ -f "ins_$TBL.sql" ] || ./gen_scripts.sh "$TBL" >/dev/null

hot_rows() {
    $PSQL -c "SELECT coalesce(max(num_entries) FILTER (WHERE is_hot), 0) FROM agentvec_index_info('$IDX')" 2>/dev/null | tr -d ' '
}
nseg() {
    $PSQL -c "SELECT count(*) FROM agentvec_index_info('$IDX')" 2>/dev/null | tr -d ' '
}

echo "fill    target=$TBL clients=$C chunk=$CHUNK chunks=$CHUNKS"
printf '    %-10s %-8s %-10s %-10s %-8s\n' seg_hot_before seg_after rows/s lat_ms note
for i in $(seq 1 "$CHUNKS"); do
    B=$(hot_rows); S0=$(nseg)
    OUT=$($PGBENCH -n -c "$C" -j "$C" -t "$((CHUNK / C))" -D nk="$NK" -f "ins_$TBL.sql" 2>&1)
    T=$(awk '/tps = /{gsub(/,/,"",$3); print $3}' <<<"$OUT" | head -1)
    L=$(awk -F= '/latency average/{gsub(/ /,"",$2); print $2}' <<<"$OUT" | head -1)
    A=$(hot_rows); S1=$(nseg)
    NOTE=""
    [ "$S1" != "$S0" ] && NOTE="SEAL(+segment)"
    note_fill="fill"
    printf '    %-14s %-10s %-10s %-8s %s\n' "$B" "$A" "${T:-FAIL}" "${L:-?}" "$NOTE"
    avb_record "fill-curve" "$TBL" "$C" "$CHUNK" "${T:-}" "${L:-}" "$TAG before=$B after=$A segs=$S1 $NOTE"
    [ -z "$T" ] && { echo "$OUT" | tail -5 >&2; break; }
done
