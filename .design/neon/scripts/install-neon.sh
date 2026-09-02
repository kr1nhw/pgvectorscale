#!/usr/bin/env bash
# Install the packaged pgvectorscale into a PostgreSQL install tree.
#
# Usage: install-neon.sh [PG_PREFIX] [ARTIFACT_DIR]
#   PG_PREFIX      prefix of the target PG install
#                  (default: /data1/neon-test/pg_install/v17)
#   ARTIFACT_DIR   pgrx package output dir
#                  (default: $CARGO_TARGET_DIR/release/vectorscale-pg17 or
#                   <repo>/target-neon/release/vectorscale-pg17)
#
# pgrx lays the package out mirroring the pg_config prefix (e.g.
# vectorscale-pg17/data1/neon-test/pg_install/v17/{lib,share}/...), so the
# script locates files by name regardless of that nesting.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PG_PREFIX="${1:-/data1/neon-test/pg_install/v17}"
ARTIFACT_DIR="${2:-${CARGO_TARGET_DIR:-$REPO_ROOT/target-neon}/release/vectorscale-pg17}"

if [ ! -d "$ARTIFACT_DIR" ]; then
    echo "error: artifact dir not found: $ARTIFACT_DIR (run build-neon.sh first)" >&2
    exit 1
fi

LIB_DIR="$PG_PREFIX/lib/postgresql"
EXT_DIR="$PG_PREFIX/share/postgresql/extension"
mkdir -p "$LIB_DIR" "$EXT_DIR"

find "$ARTIFACT_DIR" -name 'vectorscale-*.so' -exec cp -v {} "$LIB_DIR/" \;
find "$ARTIFACT_DIR" -name 'vectorscale.control' -exec cp -v {} "$EXT_DIR/" \;
find "$ARTIFACT_DIR" -name 'vectorscale--*.sql' -exec cp -v {} "$EXT_DIR/" \;

echo
echo "Installed into $LIB_DIR and $EXT_DIR."
echo "No compute restart is needed: vectorscale is not preloaded,"
echo "CREATE EXTENSION loads it from \$libdir on demand."
