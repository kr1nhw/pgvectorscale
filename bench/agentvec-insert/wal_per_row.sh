#!/bin/bash
# wal_per_row.sh <target> <rows> [tag]
#
# Measures WAL generated per inserted row, at one client (so the LSN delta is
# attributable to this workload), for each insert-target variant.  The index
# entry, the heap tuple and the commit record all land in WAL, so this is the
# "how much does one committed row cost the WAL stream" number.
#
# Run after a CHECKPOINT so the full-page-write baseline is comparable.
set -u
cd "$(dirname "$0")"
. ./env.sh
avb_header

TBL=$1; ROWS=${2:-5000}; TAG=${3:-wal}

[ -f "ins_$TBL.sql" ] || ./gen_scripts.sh "$TBL" >/dev/null

$PSQLX -c "CHECKPOINT" >/dev/null
L0=$($PSQL -c "SELECT pg_current_wal_insert_lsn()")
T0=$(date +%s%N)
$PGBENCH -n -c 1 -j 1 -t "$ROWS" -D nk=200000 -f "ins_$TBL.sql" >/dev/null 2>&1
T1=$(date +%s%N)
L1=$($PSQL -c "SELECT pg_current_wal_insert_lsn()")

BYTES=$($PSQL -c "SELECT pg_wal_lsn_diff('$L1','$L0')::bigint")
MS=$(( (T1 - T0) / 1000000 ))
BPR=$(awk -v b="$BYTES" -v r="$ROWS" 'BEGIN{printf "%.0f", b/r}')
ROWSPS=$(awk -v r="$ROWS" -v ms="$MS" 'BEGIN{printf "%.0f", r*1000/ms}')
echo "wal     target=$TBL rows=$ROWS  WAL/row=${BPR} B  total=$((BYTES/1024)) KiB  rows/s=$ROWSPS  wall=${MS}ms  tag=$TAG"
avb_record "wal-per-row" "$TBL" "1" "$MS" "$BPR" "$ROWSPS" "$TAG rows=$ROWS"
