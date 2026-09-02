#!/usr/bin/env bash
# Run the pgvectorscale ivf/RaBitQ functional + concurrent tests against a
# running PostgreSQL/Neon compute.
#
# Usage: test-neon.sh [psql]
#   psql   full psql command to use
#          (default: the Neon fork's psql against 127.0.0.1:55432 as cloud_admin)
#
# The functional test (tests/ivf_functional.sql) creates its own `items`
# table, builds ivf indexes with num_bits 1/2/4/8, exercises DML + VACUUM +
# MVCC, and measures recall against brute force.
# The concurrent test (tests/concurrent_dml.sh) then runs parallel
# inserters/queriers/deleters against that table.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSQL="${1:-/data1/neon-test/pg_install/v17/bin/psql -h 127.0.0.1 -p 55432 -U cloud_admin -d postgres}"
export PGHOST="${PGHOST:-127.0.0.1}" PGPORT="${PGPORT:-55432}"

echo "=== functional + recall test ==="
$PSQL -v ON_ERROR_STOP=1 -f "$HERE/tests/ivf_functional.sql"

echo
echo "=== concurrent DML test ==="
"$HERE/tests/concurrent_dml.sh" "$PSQL"

echo
echo "All tests finished."
