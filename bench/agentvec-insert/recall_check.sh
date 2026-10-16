#!/bin/bash
# recall_check.sh <table> <index_col_table> <nq> [tag]
#
# Compares the agentvec-indexed top-10 against an exact (seqscan) top-10 for
# the first <nq> queries of avb_qpool, and reports recall@10.  Used to check
# that a HOT storage-layout change does not cost recall.
#
# The exact side is forced with enable_seqscan=on and the index side with
# enable_seqscan=off, in separate transactions.
set -u
cd "$(dirname "$0")"
. ./env.sh

TBL=$1; NQ=${2:-200}; TAG=${3:-recall}

EXACT=$($PSQL -c "
SET enable_seqscan=on; SET jit=off;
WITH r AS (
  SELECT q.qk AS qk,
         ARRAY(SELECT t.id FROM $TBL t ORDER BY t.embedding <-> q.q LIMIT 10) AS got
  FROM avb_qpool q WHERE q.qk <= $NQ
) SELECT r.qk || ':' || array_to_string(r.got, ',') FROM r")

APPROX=$($PSQL -c "
SET enable_seqscan=off; SET jit=off;
WITH r AS (
  SELECT q.qk AS qk,
         ARRAY(SELECT t.id FROM $TBL t ORDER BY t.embedding <-> q.q LIMIT 10) AS got
  FROM avb_qpool q WHERE q.qk <= $NQ
) SELECT r.qk || ':' || array_to_string(r.got, ',') FROM r")

REC=$(paste <(echo "$EXACT") <(echo "$APPROX") | awk -F'\t' '
{
    split($1, a, ":"); split($2, b, ":");
    if (a[1] != b[1]) next;
    n = split(a[2], x, ","); m = split(b[2], y, ",");
    hit = 0;
    for (i = 1; i <= m; i++) for (j = 1; j <= n; j++) if (y[i] == x[j]) { hit++; break }
    s += hit / n; c++
}
END { if (c) printf "%.4f", s/c; else print "?" }')

echo "recall  table=$TBL nq=$NQ  recall@10=$REC  tag=$TAG"
avb_record "recall" "$TBL" "-" "$NQ" "$REC" "-" "$TAG"
