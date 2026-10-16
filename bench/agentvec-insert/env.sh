#!/bin/bash
# Shared environment for the agentvec insertion benchmark.
# Source this from the other scripts; do not execute directly.

: "${PGBIN:=/root/pg17-release/bin}"
: "${PGHOST_:=/tmp}"
: "${PGPORT_:=54331}"
: "${PGUSER_:=pgtest}"
: "${PGDATABASE_:=postgres}"

PSQL="$PGBIN/psql -h $PGHOST_ -p $PGPORT_ -U $PGUSER_ -d $PGDATABASE_ -qtA"
PSQLX="$PGBIN/psql -h $PGHOST_ -p $PGPORT_ -U $PGUSER_ -d $PGDATABASE_"
PGBENCH="$PGBIN/pgbench -h $PGHOST_ -p $PGPORT_ -U $PGUSER_ -d $PGDATABASE_ --protocol=prepared"

# Where the scripts and their logs live.
AVB_DIR=${AVB_DIR:-/data1/avbench}
AVB_RESULTS=${AVB_RESULTS:-$AVB_DIR/results.csv}

# Every benchmark connection runs with seqscan disabled and jit off, like the
# production query benchmark (kr_lat.sh).
export PGOPTIONS="-c enable_seqscan=off -c jit=off"

avb_log() { echo "$*" | tee -a "$AVB_DIR/run.log" >&2; }

# Append one measured data point.
# avb_record <phase> <target> <clients> <secs> <tps> <lat_ms> <note>
avb_record() {
    printf '%s,%s,%s,%s,%s,%s,%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" "$4" "$5" "$6" "${7:-}" \
        >> "$AVB_RESULTS"
}

avb_header() {
    [ -s "$AVB_RESULTS" ] && return
    echo "ts,phase,target,clients,secs,tps,lat_ms,note" >> "$AVB_RESULTS"
}
