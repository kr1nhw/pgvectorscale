#!/usr/bin/env python3
"""Apply the Neon-fork compatibility patch to a pgvectorscale source tree.

The patch adds a `neon` cargo feature that gates the one place where Neon's
PostgreSQL fork changes the smgr ABI: `smgropen()` takes an extra
`relpersistence` argument (Neon's pagestore smgr needs it to route reads).

It is idempotent: run it again and it reports "already patched".

Usage: patch_neon.py [repo_root]
  repo_root  path to the pgvectorscale checkout (default: <this file>/../../..)
"""
import sys
from pathlib import Path

REPO_ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else
                 Path(__file__).resolve().parent.parent.parent.parent)

CARGO_TOML = REPO_ROOT / "pgvectorscale" / "Cargo.toml"
ENTRY_RS = REPO_ROOT / "pgvectorscale" / "src" / "access_method" / "ivf" / "entry.rs"

FEATURE_LINE = 'neon = []\n'

OLD_ENTRY = """            let reln = (*rel).rd_smgr;
            let reln = if reln.is_null() {
                pg_sys::smgropen((*rel).rd_locator, (*rel).rd_backend)
            } else {
                reln
            };"""

NEW_ENTRY = """            let reln = (*rel).rd_smgr;
            let reln = if reln.is_null() {
                // Neon's fork extends `smgropen` with a `relpersistence`
                // argument; vanilla PostgreSQL does not.
                #[cfg(feature = "neon")]
                {
                    pg_sys::smgropen(
                        (*rel).rd_locator,
                        (*rel).rd_backend,
                        (*(*rel).rd_rel).relpersistence,
                    )
                }
                #[cfg(not(feature = "neon"))]
                {
                    pg_sys::smgropen((*rel).rd_locator, (*rel).rd_backend)
                }
            } else {
                reln
            };"""


def main() -> int:
    changed = False

    s = CARGO_TOML.read_text()
    if FEATURE_LINE in s:
        print(f"ok: {CARGO_TOML.name} already has the `neon` feature")
    else:
        old = "build_parallel = []\n"
        assert s.count(old) == 1, f"anchor line not found exactly once in {CARGO_TOML}"
        new = (
            old
            + "# Build against Neon's PostgreSQL fork, which has a modified smgr ABI\n"
            + "# (`smgropen` takes an extra `relpersistence` argument) and a different\n"
            + "# `FMGR_ABI_EXTRA` (requires `pgrx/unsafe-postgres`).\n"
            + FEATURE_LINE
        )
        CARGO_TOML.write_text(s.replace(old, new, 1))
        print(f"patched: {CARGO_TOML}")
        changed = True

    s2 = ENTRY_RS.read_text()
    if OLD_ENTRY in s2:
        assert s2.count(OLD_ENTRY) == 1, f"block found more than once in {ENTRY_RS}"
        ENTRY_RS.write_text(s2.replace(OLD_ENTRY, NEW_ENTRY, 1))
        print(f"patched: {ENTRY_RS}")
        changed = True
    else:
        print(f"ok: {ENTRY_RS.name} already patched (or call site moved)")

    print("patch already applied, nothing to do" if not changed else "patch applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
