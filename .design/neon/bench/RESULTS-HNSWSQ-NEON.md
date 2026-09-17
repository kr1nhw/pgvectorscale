# hnswsq on Neon — 1M benchmark, perf attribution, and plan

Measured 2026-09-17 on host-121 (121.37.117.106, 32 vCPU) with a local Neon
dev stack built from source: pageserver + safekeeper + storage-broker +
storage-controller + a compute running the Neon PostgreSQL fork **17.5**
(`vendor/postgres-v17`, commit `1e01fcea2a6b38180021aa83e0051d95286d9096`,
built with the Makefile's release profile — note it still enables
`--enable-debug`/asserts).  Dataset: BIGANN 1M×128 (the same `items_1m` /
`bench_queries` / `gt_1m` as the vanilla study), `hnswsq` sq8
(storage_layout=sq8, m=16, efc=64), extension built with
`--features "pg17 build_parallel neon pgrx/unsafe-postgres"`.

## Measured

| config | build s | qms ef10/40/160/640 (p50) | recall@10 ef160/640 |
|---|---|---|---|
| vanilla release PG 17.11 (4 workers, settings matrix) | 114.4 | 0.508 / 0.976 / 2.278 / 5.448 | 0.994 / 0.999 |
| **Neon, tuned compute** (2 GB shared_buffers, 8 GB LFC, aggressive PS compaction) | **50.3** (8 workers) | 4.3 / 7.1 / 11.4 / 20.7 | 0.992 / 1.0 |
| **Neon, default compute** (1 MB shared_buffers, LFC off — the dev defaults) | — (see the crash note) | 471 / 2365 (ef40/ef640) | — |

Same-query apples-to-apples (single `EXPLAIN (ANALYZE, BUFFERS)` at ef 640,
warm, LIMIT 10): vanilla 18.8 ms / Neon 21.6 ms (**1.15x**) with identical
buffer counts (**13,379 vs 13,401 shared hits**) — when the working set is
cached, the pageserver is not in the query path at all and Neon is within
15% of vanilla.  The default-config 2.4 s query is the same 13.4 K page
touches paying one pageserver round trip each (~0.18 ms/request on the
loopback) — the cold path, not the ALU.

The build is **faster** on Neon (50.3 s vs 114.4 s) for two reasons: 8
parallel workers vs 4, and the deferred WAL (the whole graph is logged once
with `log_newpage_range` at the end instead of shipping per-page WAL during
the build).

## perf attribution

ef-640 query backend on the Neon compute (30 s sample):

| symbol | share |
|---|---|
| LWLockRelease | 28.8% |
| `load_element_impl` (hnswsq) | 21.1% |
| PinBuffer | 16.7% |
| LWLockAttemptLock | 7.0% |
| `Visited::insert_key_hash` | 3.1% |
| UnpinBufferNoOwner | 2.3% |

The on-CPU shape is identical to vanilla (buffer-pin/lock management ~55% +
element loading ~21%) — the Neon compute adds nothing per-page on the hit
path.  The wall-clock gap on the default config is entirely the pagestore
waits (the same 7% CPU-utilization signature documented in
`OPTIMIZATION-PLAN.md` for the 10 M study).

## Findings / bugs

1. **The build's deferred WAL logging is Neon-fragile.**  The port (like
   pgvector) writes the whole index with `MarkBufferDirty` and logs the
   page range once at the end (`build.rs` → `log_newpage_range`).  If a
   checkpoint lands mid-build, the checkpointer evicts a dirty page whose
   LSN is still `InvalidXLogRecPtr` and Neon's smgr PANICs:
   `[NEON_SMGR] Page 0 ... is evicted with zero LSN` — reproduced twice
   (~23 s and ~2 min into builds; the timing is the 5-min checkpoint from
   the preceding load).  pgvector's build survives only by timing luck —
   it has the same latent pattern.  The bench workaround was
   `checkpoint_timeout=1h` + `max_wal_size=32GB`; the real fix is to WAL
   each page as it is created (or emit `log_newpage_range` incrementally)
   under `cfg(feature = "neon")`.
2. **The scan touches 13.4 K buffers per ef-640 query** — ~21 buffer ops
   per visited candidate (element page + neighbor-tuple page + recheck
   heap fetches).  This is the same on vanilla and Neon, and it is the
   quantity the cold path multiplies by the round-trip cost.
3. **The recheck heap-fetch amplification** (the executor fetches heap
   tuples for the candidate set) is the known 2-CU cost from the 10 M
   study; at 1 M with a 2 GB buffer pool it is the residual Neon cost.

## Plan

1. **Fix the zero-LSN fragility (correctness first).**  Gate the build's
   page writes: under `cfg(feature = "neon")` emit `log_newpage_buffer`
   per page (or periodic `log_newpage_range` flushes) instead of the
   single final range log.  Accept the extra WAL volume (the index is
   written once either way).
2. **Shrink the 13.4 K-buffer footprint (helps vanilla too).**  The
   per-candidate neighbor-tuple reads and the recheck heap fetches are the
   bulk; batch the neighbor-tuple loads per page (they cluster on the
   graph pages) and re-examine the iterative-scan batch sizing.
3. **Neon read-path levers, in order of measured value (from the 10 M
   study, confirmed here):** (a) pageserver compaction right after the
   build (removes per-page walredo — already applied), (b) LFC sized to
   the working set (1 M sq8 = 376 MB → fits a 2-CU compute), (c) the
   pageserver page cache as the second tier, (d) prefetch/batched getpage
   for the HNSW's *random* neighbor-page access — the ivf study measured
   prefetch as a no-gain on its sequential bursts, but the random-access
   HNSW pattern has a different round-trip structure and is the one case
   worth re-testing.
4. **CU budget:** 1 M sq8 (376 MB) fits a 2-CU LFC; the 10 M graph is
   ~3.8 GB in sq8 vs 7.9 GB in plain — sq8 moves the "hnsw needs a big
   compute" boundary down roughly one CU tier.

## Reproducing

The stack lives at `/data1/neon1711` (built as the `neon1711` user;
`make POSTGRES_VERSIONS=v17 BUILD_TYPE=release` + the cargo git/registry
caches mirrored from the build Mac; protoc via a `grpc_tools.protoc`
wrapper; `PG_MAJORVERSION=17` in the build env; the v14–v16 `pg_install`
entries symlinked to v17 for the bindgen fallback).  The endpoint is
`neon_local endpoint create main --tenant-id ... --branch-name main`;
the tuned `postgresql.conf` and the pageserver `compaction_period=2s /
compaction_threshold=1 / page_cache_size=1M` settings are documented in
`scripts/RECOMMENDED-SETUP.md`.  `scripts/build-neon.sh` /
`install-neon.sh` / `test-neon.sh` are the extension build/install/verify
recipe (the `neon` cargo feature gates the smgropen ABI difference; the
`pgrx/unsafe-postgres` feature passes the `FMGR_ABI_EXTRA` check).
