#!/bin/bash
# reset.sh [target ...]
#
# Returns the insert targets to a pristine state: empty table, empty (or
# freshly rebuilt) index.  For agentvec a TRUNCATE alone would leave the old
# segment directory behind, so the index is rebuilt.
set -eu
cd "$(dirname "$0")"
. ./env.sh

TARGETS=${*:-"avb_t_heap avb_t_btree avb_t_av avb_t_pgv"}

for t in $TARGETS; do
    case "$t" in
        *_pkey|*_idx) idx=$t ;;
        *) idx="" ;;
    esac
    if [ -z "$idx" ]; then
        idx=$($PSQL -c "SELECT c.relname FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid JOIN pg_class t ON t.oid=i.indrelid WHERE t.relname='$t' AND c.relam <> (SELECT oid FROM pg_am WHERE amname='btree') LIMIT 1")
    fi
    $PSQLX -c "TRUNCATE $t RESTART IDENTITY" >/dev/null
    if [ -n "$idx" ]; then
        $PSQLX -c "REINDEX INDEX $idx" >/dev/null
        echo "reset $t (reindexed $idx)"
    else
        echo "reset $t"
    fi
done
