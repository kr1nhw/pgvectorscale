#!/bin/bash
# bench_query.sh <target> <clients> <seconds> <nq> [tag]
#
# Runs pgbench with the 10-NN query script; reports queries/s.
set -u
cd "$(dirname "$0")"
. ./env.sh
avb_header

TBL=$1; C=$2; SECS=$3; NQ=${4:-10000}; TAG=${5:-query}
J=$C; [ "$J" -gt 16 ] && J=16

[ -f "q_$TBL.sql" ] || ./gen_scripts.sh "$TBL" >/dev/null

OUT=$($PGBENCH -n -c "$C" -j "$J" -T "$SECS" -D nq="$NQ" -f "q_$TBL.sql" 2>&1)
TPS=$(echo "$OUT" | awk '/tps = /{gsub(/,/,"",$3); print $3}' | head -1)
LAT=$(echo "$OUT" | awk -F= '/latency average/{gsub(/ /,"",$2); print $2}' | head -1)
if [ -z "$TPS" ]; then
    echo "FAILED(query): $TBL c=$C"; echo "$OUT" | tail -20 >&2; exit 1
fi
echo "query   target=$TBL clients=$C secs=$SECS  q/s=$TPS  lat_avg_ms=${LAT:-?}  tag=$TAG"
avb_record "query" "$TBL" "$C" "$SECS" "$TPS" "${LAT:-}" "$TAG"
