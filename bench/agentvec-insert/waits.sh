#!/bin/bash
# waits.sh <seconds>
#
# Samples pg_stat_activity while a workload runs and prints the distribution of
# wait events across active client backends.  This is what tells us *what* the
# inserters are blocked on: LWLock:BufferContent (page contention inside one
# hnswsq region), Lock:page / Lock:extension (the region's heavyweight page
# lock), LWLock:WALInsert, IO:*, etc.
#
# Sampling is at ~50 ms; the output is a share of samples per (type, event).
set -u
cd "$(dirname "$0")"
. ./env.sh

SECS=${1:-10}
N=$(( SECS * 20 ))
$PSQL -c "
SELECT wait_event_type || ':' || coalesce(wait_event,'') AS ev, count(*) AS n
FROM pg_stat_activity
WHERE backend_type = 'client backend' AND state = 'active'
  AND application_name NOT LIKE 'waits%'
GROUP BY 1
" >/dev/null 2>&1

TMP=$(mktemp)
for _ in $(seq 1 "$N"); do
    $PSQL -c "
    SELECT coalesce(wait_event_type,'CPU') || ':' || coalesce(wait_event,'run') AS ev
    FROM pg_stat_activity
    WHERE backend_type = 'client backend' AND state = 'active'
      AND pid <> pg_backend_pid()
    " >>"$TMP" 2>/dev/null
    sleep 0.05
done
echo "wait-event distribution over ${SECS}s (samples: $(wc -l < "$TMP")):"
sort "$TMP" | uniq -c | sort -rn | head -15 | awk '{printf "    %6d  %5.1f%%  %s\n", $1, 100*$1/NR, $2}'
rm -f "$TMP"
