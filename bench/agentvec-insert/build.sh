#!/bin/bash
# build.sh [--install]
#
# Builds the vectorscale extension from /data1/avwork with the same flags the
# production x86 build uses (release + AVX2/FMA + debuginfo), and optionally
# installs it into the release PG 17.11 tree used by the benchmark cluster.
#
# The full compiler output goes to build_full.log; only the tail is echoed, so a
# failure can still be inspected after the fact.
set -u
cd /data1/avwork
export PGRX_HOME=/root/.pgrx-hnswsq
export RUSTFLAGS="-C target-feature=+avx2,+fma -C debuginfo=2"
export PATH="/root/.cargo/bin:$PATH"
LOG=/data1/avwork/build_full.log

cargo build --release -p vectorscale --features pg17,build_parallel --no-default-features >"$LOG" 2>&1
RC=$?
echo "cargo exit: $RC"
grep -E "^error" "$LOG" | head -20
tail -3 "$LOG"

SO=target/release/libvectorscale.so
if [ "$RC" -ne 0 ]; then
    echo "BUILD FAILED (full log: $LOG)"
    exit "$RC"
fi
ls -la "$SO"

if [ "${1:-}" = "--install" ]; then
    DEST=/root/pg17-release/lib/postgresql/vectorscale-0.9.0.so
    cp -f "$DEST" "$DEST.bak" 2>/dev/null || true
    install -m 755 "$SO" "$DEST.new"
    mv -f "$DEST.new" "$DEST"
    echo "installed to $DEST (previous saved as $DEST.bak)"
fi
