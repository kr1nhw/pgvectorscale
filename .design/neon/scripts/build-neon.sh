#!/usr/bin/env bash
# Build pgvectorscale against a Neon PostgreSQL fork install.
#
# Usage: build-neon.sh [NEON_PG_CONFIG]
#   NEON_PG_CONFIG   path to the Neon fork's pg_config
#                    (default: /data1/neon-test/pg_install/v17/bin/pg_config)
#
# Environment:
#   CARGO_TARGET_DIR  separate cargo target dir (default: <repo>/target-neon)
#   PGRX_HOME         pgrx home (default: ~/.pgrx)
#
# Prerequisites:
#   - cargo-pgrx == the pgrx version pinned in pgvectorscale/Cargo.toml
#     (cargo install --locked cargo-pgrx --version 0.16.1)
#   - a writable PGRX_HOME
#   - the pgvectorscale source already carries the `neon` feature
#     (run scripts/patch_neon.py first if it is an older checkout)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NEON_PG_CONFIG="${1:-/data1/neon-test/pg_install/v17/bin/pg_config}"
TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target-neon}"
PGRX_HOME="${PGRX_HOME:-$HOME/.pgrx}"
export PGRX_HOME

if [ ! -x "$NEON_PG_CONFIG" ]; then
    echo "error: pg_config not found at $NEON_PG_CONFIG" >&2
    exit 1
fi

mkdir -p "$PGRX_HOME"

# Register the fork in pgrx. This validates the pg_config, writes
# $PGRX_HOME/config.toml, and runs initdb under $PGRX_HOME/data-17
# (harmless for packaging, and the Neon fork's initdb works standalone).
cargo pgrx init --pg17 "$NEON_PG_CONFIG"

cd "$REPO_ROOT"

# Feature notes:
#   pg17              - compile against PostgreSQL 17 headers
#   neon              - our portability gate for Neon's 3-arg smgropen()
#   pgrx/unsafe-postgres
#                     - Neon redefines FMGR_ABI_EXTRA to "Neon Postgres";
#                       pgrx refuses it unless this escape hatch is enabled
#   --no-default-features
#                     - the crate's default is pg18
CARGO_TARGET_DIR="$TARGET_DIR" cargo pgrx package \
    --pg-config "$NEON_PG_CONFIG" \
    --package vectorscale \
    --features "pg17 build_parallel neon pgrx/unsafe-postgres" \
    --no-default-features

echo
echo "Artifacts written under: $TARGET_DIR/release/vectorscale-pg17/"
