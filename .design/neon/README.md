# Installing & patching pgvectorscale — vanilla and Neon (ivf/RaBitQ branch)

This guide documents how to build and install **this branch** of pgvectorscale
(ivf access method + 1/2/4/8-bit RaBitQ, branch `ivf-rabitq`) and, specifically,
how to run it on **Neon** — Neon's PostgreSQL fork and its pageserver-backed
dev stack. The Neon-specific "patch" is the `neon` cargo feature described in
[Why a patch is needed](#why-a-patch-is-needed); it is **already committed** to
this branch (`e1bcabc`), so a fresh checkout needs no source edits — you only
enable the right build features.

Everything below was validated end-to-end on an x86_64 Neon dev stack
(pageserver + safekeeper + storage_broker + storage_controller + compute,
PostgreSQL 17.5 fork). See [Validation results](#validation-results).

---

## 1. Standard (vanilla PostgreSQL) install

Unchanged from upstream; for completeness:

```bash
git clone <this repo> && cd pgvectorscale
cargo install --locked cargo-pgrx --version 0.16.1   # must match the pgrx pin in Cargo.toml
cargo pgrx init --pg17 "$(which pg_config)"
cargo pgrx install --release --package vectorscale \
    --no-default-features --features "pg17 build_parallel"
```

This is the only build that needs no Neon handling: the `neon` feature is off,
so `smgropen()` uses the vanilla 2-argument form.

---

## 2. Why a patch is needed (Neon fork ABI differences)

Neon's PostgreSQL fork (`neondatabase/postgres`, branch `REL_17_STABLE_neon`)
changes two things that matter to a pgrx extension:

### 2.1 `FMGR_ABI_EXTRA` is `"Neon Postgres"`

`src/include/pg_config_manual.h`:

```c
#define FMGR_ABI_EXTRA        "Neon Postgres"
```

pgrx asserts at compile time that the ABI string is the stock one:

```text
error[E0080]: evaluation panicked: Unsupported Postgres ABI.
              Perhaps you need `--features unsafe-postgres`?
```

Fix: build with pgrx's escape hatch — the `pgrx/unsafe-postgres` feature. This
is safe here because `pgrx-pg-sys` bindings are generated from the *same* Neon
headers, so struct layouts stay consistent; only the string check is bypassed.

### 2.2 `smgropen()` takes a third `relpersistence` argument

Neon's smgr replacement (see [How the smgr dispatch works](#how-the-smgr-dispatch-works))
must know each relation's persistence to route I/O:

```c
/* vanilla PG17 */  SMgrRelation smgropen(RelFileLocator rlocator, BackendId backend);
/* Neon fork    */  SMgrRelation smgropen(RelFileLocator rlocator, ProcNumber backend, char relpersistence);
```

`pgvectorscale/src/access_method/ivf/entry.rs` opens the smgr relation itself
for the FastScan `smgrreadv` bulk read, so it must pass the extra argument on
Neon. That is the whole patch — a `neon` cargo feature gating the call:

```rust
let reln = (*rel).rd_smgr;
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
};
```

```toml
# pgvectorscale/Cargo.toml
[features]
...
build_parallel = []
# Build against Neon's PostgreSQL fork, which has a modified smgr ABI
# (`smgropen` takes an extra `relpersistence` argument) and a different
# `FMGR_ABI_EXTRA` (requires `pgrx/unsafe-postgres`).
neon = []
```

**Current HEAD already contains this.** If you are applying it to an older
checkout or another fork, `scripts/patch_neon.py` applies exactly these two
edits (idempotent):

```bash
.design/neon/scripts/patch_neon.py /path/to/pgvectorscale
```

No other Neon accommodations exist anywhere in the extension: buffer writes
(`smgrwrite`/`GenericXLog`), `FlushRelationBuffers` flush-before-publish, and
the `smgrreadv` scan path all go through stock PostgreSQL APIs.

---

## 3. Prerequisites for the Neon build

On the machine that hosts the Neon stack (all services run as the same
non-root user, `pg17test` on our reference box):

1. **Neon fork install** — a configured + installed `neondatabase/postgres`
   checkout, e.g. `/data1/neon-test/pg_install/v17` (PG 17.5, commit
   `1e01fcea2a6b38180021aa83e0051d95286d9096`, which matches neon-repo HEAD's
   `vendor/revisions.json` pin).
2. **The Neon pgxn extensions built against that fork**: `neon.so`,
   `neon_rmgr.so`, `neon_walredo.so` in `pg_install/v17/lib/postgresql/`
   (they provide the pagestore smgr, WAL redo, and resource manager).
3. **pgvector** (`vector.control` + `vector.so`) in the same install — the
   extension `requires = 'vector'`.
4. **Rust + cargo-pgrx**:

   ```bash
   cargo install --locked cargo-pgrx --version 0.16.1
   ```

   (`rustc 1.98` was verified working; the E0080 in §2.1 is an ABI assert,
   not a toolchain incompatibility.)
5. **A writable `PGRX_HOME`** (default `~/.pgrx`) — the build fails early
   with `$PGRX_HOME does not exist` otherwise.
6. A running **compute endpoint** to test against (the Neon dev stack: see
   [Reference server layout](#reference-server-layout)).

---

## 4. Build and install (the short version)

```bash
# 1. (one-time) register the fork with pgrx — done inside the script too
PGRX_HOME=~/.pgrx cargo pgrx init --pg17 /data1/neon-test/pg_install/v17/bin/pg_config

# 2. package the extension (runs `cargo pgrx schema` and compiles the .so)
.design/neon/scripts/build-neon.sh /data1/neon-test/pg_install/v17/bin/pg_config

# 3. copy vectorscale-*.so, vectorscale.control, vectorscale--*.sql into the install
.design/neon/scripts/install-neon.sh /data1/neon-test/pg_install/v17
```

What step 2 runs, explicitly:

```bash
CARGO_TARGET_DIR=<repo>/target-neon cargo pgrx package \
    --pg-config /data1/neon-test/pg_install/v17/bin/pg_config \
    --package vectorscale \
    --features "pg17 build_parallel neon pgrx/unsafe-postgres" \
    --no-default-features
```

Notes:

- `--package vectorscale` is required: the workspace root is virtual and the
  package is named `vectorscale` (the directory is `pgvectorscale/`).
- `--no-default-features` is required: the crate's default is `pg18`.
- Output lands in `target-neon/release/vectorscale-pg17/...`, nested to mirror
  the pg_config prefix; `install-neon.sh` finds the files by name.
- No compute restart is needed after installing — `vectorscale` is not
  preloaded; `CREATE EXTENSION` loads it from `$libdir` on demand.

---

## 5. Verify

```bash
# on the compute (or anywhere the psql can reach it):
.design/neon/scripts/test-neon.sh \
  "/data1/neon-test/pg_install/v17/bin/psql -h 127.0.0.1 -p 55432 -U cloud_admin -d postgres"
```

`tests/ivf_functional.sql` covers, in order:

1. `CREATE EXTENSION vector; CREATE EXTENSION vectorscale;`
2. 20k x 8-dim table in 8 well-separated clusters
3. `CREATE INDEX ... USING ivf (embedding vector_l2_ops) WITH (lists=8, num_bits=1/2/4/8)`
4. index-scan queries with distance spot checks
5. DML: `UPDATE` 500 rows, `DELETE` ~490, `VACUUM`, `ANALYZE`
6. MVCC: rolled-back insert stays invisible; committed insert is found
7. per-bit-width recall vs brute-force top-10 (index rebuilt one width at a
   time, `ivf.probes = 8`)

`tests/concurrent_dml.sh` then runs 4 inserters + 2 queriers + 1 deleter in
parallel against the same table.

**Important:** the ivf operator classes are intentionally not `DEFAULT`
(same convention as pgvector's hnsw/ivfflat), so always name the opclass:
`USING ivf (embedding vector_l2_ops)`.

### The Neon-specific durability check

The property that matters most on Neon — that our `GenericXLog` + flush-before-
publish writes are durable in the pageserver — is verified by a full wipe and
re-bootstrap of the compute:

```bash
neon_local endpoint stop main
mv .neon/endpoints/main/pgdata .neon/endpoints/main/pgdata.bak   # force a fresh basebackup
neon_local endpoint start main --start-timeout 600s

psql ... -c "SELECT count(*) FROM items;"                        # rows intact
psql ... -c "SET enable_seqscan=off; EXPLAIN (COSTS OFF)
             SELECT id FROM items ORDER BY embedding <-> '[...]'::vector LIMIT 5;"
# → index scan used, distances identical to before the wipe
```

---

## 6. Validation results

Validated on the reference Neon stack (PG 17.5 fork, compute on `127.0.0.1:55432`):

| Check | Result |
|---|---|
| Build + `CREATE EXTENSION vectorscale` (with vector 0.8.0) | ✅ |
| `CREATE INDEX USING ivf ... WITH (num_bits = 1/2/4/8)` | ✅ all valid, planner uses them |
| Index-scan distances | ✅ correct (top-1 at 0.000 for the exact centroid row) |
| Recall vs brute force, `ivf.probes = 8` | ✅ **100.0% for all four bit widths** |
| UPDATE 500 / DELETE 487 / VACUUM / ANALYZE | ✅ |
| MVCC (rollback invisibility, commit visibility) | ✅ |
| Concurrent: 4 inserters + 2 queriers + 1 deleter | ✅ 0 failures |
| Durability: stop → wipe pgdata → fresh basebackup (LSN 0/2D5DFE8) | ✅ extensions, rows, index, and byte-identical results restored; post-restore `CREATE INDEX` + `VACUUM` fine |

---

## 7. How the smgr dispatch works (why no Neon source changes were needed)

The FastScan path in `entry.rs` bypasses the buffer manager with one vectored
call to the core smgr API — `smgrreadv()` is vanilla PG 17, not a Neon hook:

```rust
pg_sys::smgrreadv(reln, pg_sys::ForkNumber::MAIN_FORKNUM,
                  start_page, ptrs.as_mut_ptr(), num_blocks as BlockNumber);
```

PostgreSQL core dispatches it through a vtable (`smgr.c`):

```c
void smgrreadv(SMgrRelation reln, ForkNumber forknum, BlockNumber blocknum,
               void **buffers, BlockNumber nblocks)
{
    (*reln->smgr).smgr_readv(reln, forknum, blocknum, buffers, nblocks);
}
```

Neon's fork adds one hook — `smgr_hook` in `storage/smgr.h` — which `neon.so`
sets during shared-preload init (`libpagestore.c`):

```c
smgr_hook = smgr_neon;
```

`smgr_neon` returns Neon's own `f_smgr` vtable (`pagestore_smgr.c`), which
implements the full contract including

```c
static const struct f_smgr neon_smgr = {
    ...
#if PG_MAJORVERSION_NUM >= 17
    .smgr_readv  = neon_readv,
    .smgr_writev = neon_writev,
#endif
    .smgr_nblocks = neon_nblocks,
    ...
};
```

`neon_readv` batches the blocks into libpagestore page requests to the
pageserver (and falls back to local `mdreadv` for unlogged-build relations,
which is why it needs `reln->smgr_relpersistence` — the very reason
`smgropen` gained its third argument). So the entire read path — and equally
`smgrwrite`/`FlushRelationBuffers` on the write side — works through
pre-existing extension points. **This project makes zero changes to Neon
sources.**

---

## 8. Troubleshooting

Real issues hit while bringing this up, with fixes:

| Symptom | Cause / fix |
|---|---|
| `pgrx requires a root package in a workspace` | add `--package vectorscale` (package name ≠ directory name) |
| `Could not find package pgvectorscale` | same — the package is `vectorscale` |
| `$PGRX_HOME does not exist` / `config.toml not found` | `mkdir -p ~/.pgrx` and run `cargo pgrx init --pg17 <pg_config>` |
| `Permission denied (os error 13)` during package | repo/target dir not writable by the build user; `chown -R` the checkout (on Neon boxes, **always** build and run as the service user, not root — root-owned files crash the pageserver) |
| `error[E0080]: ... Unsupported Postgres ABI` | Neon's `FMGR_ABI_EXTRA`; add `pgrx/unsafe-postgres` |
| `error[E0061]: smgropen ... argument #3 of type i8 is missing` | missing the `neon` feature (run `scripts/patch_neon.py`) |
| `data type vector has no default operator class for access method "ivf"` | name the opclass explicitly: `USING ivf (embedding vector_l2_ops)` |
| Compute dies right after "ready"; `background worker "WAL proposer" ... signal 11` | pageserver's sticky in-memory `corruption_detected` flag (set by an earlier `critical_timeline!` walredo failure) rides in every feedback; neon HEAD then hits its own NULL-deref bug in `record_pageserver_feedback` (`databricks_metrics_shared` is NULL unless `lakebase_mode=on`). Fix: **restart the pageserver** to clear the flag — after making sure the pgxn extensions really match the fork. |
| Basebackup/redo slow, `compute startup timed out` | `neon_local endpoint start main --start-timeout 600s` |
| `could not access file "neon" / "neon_rmgr" / "neon_walredo"` | those pgxn extensions are missing/mismatched in the fork's `lib/postgresql`; rebuild them against the exact fork revision the neon-repo HEAD pins (beware hand-edited `vendor/revisions.json`) |
| `VACUUM cannot run inside a transaction block` | psql `-c` wraps statements in a transaction; run VACUUM via `-f` |
| `scp: dest open ... Permission denied` | the target file was previously `chown`ed; `rm` it first |

---

## 9. Reference server layout (our test box)

| Item | Path / value |
|---|---|
| Neon source + binaries | `/data1/neon-test` (`target/release/{pageserver,safekeeper,storage_broker,storage_controller,neon_local}`) |
| Neon-patched PG install | `/data1/neon-test/pg_install/v17` (fork `1e01fcea`, PG 17.5, `--without-icu`) |
| pgxn extensions | `/data1/neon-test/pgxn/{neon,neon_rmgr,neon_walredo}` → `pg_install/v17/lib/postgresql/*.so` |
| Compute endpoint | `main`, `127.0.0.1:55432`, user `cloud_admin` (compute_ctl HTTP on 55433) |
| Controller metadata PG | port 1235, db `storage_controller`, user `pg17test` |
| Pageserver / safekeeper / broker / controller | 9898 / 5454 / 50051 / 1234 |
| Service user | `pg17test` (uid 1001) — everything runs as this user |
| Start/stop services | `su pg17test -c "cd /data1/neon-test && bash start_neon.sh"`; endpoints via `neon_local endpoint start/stop main` |

Useful debugging recipes:

```bash
# core dumps for the walproposer (SIGSEGV): enable before starting the endpoint
ulimit -c unlimited   # core_pattern is /tmp/core.%e.%p on that box

# backtrace
gdb -batch -q /data1/neon-test/pg_install/v17/bin/postgres /tmp/core.postgres.<pid> \
    -ex "set solib-search-path /data1/neon-test/pg_install/v17/lib" -ex "bt 15"
```
