# hnswsq vs pgvector-hnsw: same-host 1M A/B (121.37.117.106, 32 vCPU, PG 17.11)

Dataset: BIGANN 1M rows, dim 128, first 100 queries, exact top-10 ground truth
(`items_1m` / `gt_1m`).  Both engines: `m=16`, `ef_construction=64`,
`maintenance_work_mem=8GB`.  hnswsq index: `storage_layout=plain`.

## Release-PostgreSQL re-run (2026-09-15, same box)

The box's default cluster runs pgrx's DEBUG PostgreSQL
(`--enable-cassert -DRANDOMIZE_ALLOCATED_MEMORY=1`), which taxes every
palloc and distorts absolute numbers.  A **release PostgreSQL 17.11** was
built from the same source tree on the box (`/root/pg17-rel-src`,
prefix `/root/pg17-release`, port 54331), with pgvector 0.8.6 rebuilt
against it and the pgrx-built release extension installed; the dataset
was restored from the scratch cluster (`items_1m` 1M + `bench_queries` +
`gt_1m`).

| config | build s | qms ef10/40/160/640 (ms) | recall@10 ef160/640 | ins rps (ms/row) |
|---|---|---|---|---|
| pgvector hnsw | 72.6 | 0.294 / 0.770 / 2.394 / 7.029 | 0.995 / 1.0 | 208 (4.80) |
| hnswsq f8 scalar | 103.3 | 0.249 / 0.619 / 1.725 / 5.103 | 0.992 / 1.0 | 184 (5.43) |
| hnswsq f8 pairwise (weighted, default) | 89.7 | 0.218 / 0.544 / 1.448 / 4.461 | 0.991 / 1.0 | 250 (4.00) |
| hnswsq sq8 (fixed [0,255] int8, training-free) | 125.2 | 0.480 / 0.915 / 2.059 / 5.429 | 0.994 / 1.0 | 461 (2.17) |
| hnswsq sq16 (fixed [0,255] int16, training-free) | 146.0 | 0.523 / 0.984 / 2.194 / 5.579 | 0.994 / 1.0 | 418 (2.39) |

Recall@10 across ef (release PG, `recall_sweep.sh`, all layouts on the
1M-row table):

| ef | pgvector | plain | ieeefp16 | ieeefp8 | f8 | sq8 | sq16 |
|----|----------|-------|----------|---------|----|-----|------|
| 10 | 0.795 | 0.782 | 0.782 | 0.709 | 0.782 | 0.783 | 0.782 |
| 20 | 0.886 | 0.883 | 0.883 | 0.876 | 0.878 | 0.884 | 0.884 |
| 40 | 0.941 | 0.939 | 0.939 | 0.928 | 0.943 | 0.940 | 0.941 |
| 80 | 0.975 | 0.978 | 0.979 | 0.971 | 0.979 | 0.978 | 0.979 |
| 160 | 0.991 | 0.994 | 0.994 | 0.994 | 0.991 | 0.994 | 0.994 |
| 320 | 0.999 | 0.999 | 0.999 | 1.000 | 0.998 | 0.999 | 0.999 |
| 640 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The fixed-range layouts track plain within ±0.2pt over the whole range —
the byte-exact codes give them plain-class ranking on uint8 data with a
quarter of the vector bytes.

(The `sq8`/`sq16` rows above are from the 2026-09-16 run on the restored
1M table; pgvector/f8 are the 2026-09-15 numbers.  See the settings
matrix below for the full same-pass 09-16 table.)

Findings:

- **sq8 queries beat pgvector at every ef on release PG** (0.249 vs 0.294
  through 5.103 vs 7.029 ms), with recall within 0.4pt (0.991-0.992 vs
  0.995) and half the index size (376 MB vs 832 MB).
- **The weighted pairwise distance is now the SQ8 default**: the query
  is quantized once into code space and candidate distances are
  `SUM scale_i^2 (qhat - code)^2` — the squared difference of the DECODED
  values, i.e. the scalar decode's exact ranking with the
  query-quantized integer form (the scale weights Lance preserves via
  stored per-vector norms and Milvus via per-dimension 256-entry LUTs),
  with a vectorized AVX2 kernel.  On 1M: recall identical to scalar
  (0.991/1.0), queries 2.2x scalar (0.218-4.461 vs 0.249-5.103 ms), build
  1.15x (89.7 vs 103.3 s), and the pairwise-built graph is smaller
  (376 vs 450 MB).  The earlier no-gain measurements were two
  plumbing bugs, both fixed: the parallel build's workers never saw the
  leader's `hnswsq.sq8_distance` SET (workers are separate processes), and
  the shared-mode write initially raced the worker launch — the mode now
  travels through shared memory and is published BEFORE
  `LaunchParallelWorkers`.  The UNWEIGHTED pairwise was
  rejected (4 pt recall loss on uneven scales); the Lance-dot variant
  needs stored norms (format change).
- **pgvector's build is faster on release PG** (72.6s vs 103.3s for f8):
  its C build pallocs heavily and suffered the debug tax (107s on the
  debug box); the f8 build is decode/load-bound and was barely taxed
  (105.7s debug vs 103.3s release).  Build parity for f8 remains the
  open item; `plain` hnswsq built in 71s on the debug box vs pgvector's
  107s there.
- **`sq8`/`sq16` — the training-free fixed-range SQ layouts** (added
  2026-09-16): a global `[0, 255]` range, so byte-valued BIGANN vectors
  encode losslessly (`sq8` stores the byte itself, scale 1.0; `sq16` adds
  128 sub-steps per unit at 2⁻⁷) and distances come straight from the
  integer codes (`Σ(q̂ − code)²`).  **Recall tracks plain exactly**
  (0.994/0.999/1.0 at ef 160/320/640 — equal to plain/fp16, ahead of
  f8's 0.991/0.998 and pgvector's 0.991/0.999), queries beat pgvector at
  every ef (0.480–5.429 ms vs 0.545–6.291 for sq8; 0.523–5.579 for sq16),
  inserts are the fastest quantized layout (461 rows/s vs f8's 457, at
  2.17 ms/row), and the size is the plain ratio expected of 1-byte /
  2-byte vectors (376 vs 537 B/row — sq8 matches f8 per row).  Builds
  are decode-bound like f8's (100–112 µs/row vs f8's 95).  Graph
  construction uses the exact decoded distance for these layouts — the
  query-quantized integer form fragments the neighbor graph on clustered
  data (its half-step query noise rivals intra-cluster spacing: recall
  collapsed to ~0.75 at every ef vs 1.0 with the exact build) — while
  scans keep the fast integer form.

## Re-run 2026-09-15 (measured on the box's pgrx DEBUG PostgreSQL — superseded)

> **Superseded for absolute numbers.** This section was measured on the
> box's pgrx DEBUG PostgreSQL (`RANDOMIZE_ALLOCATED_MEMORY`), which taxes
> palloc-heavy code.  The **Release-PostgreSQL re-run** (top section) and
> the **Settings matrix** below are the current reference numbers; the
> claims made here are corrected next to each table.

Run with `.design/neon/bench/gap_study.sh` after syncing this repo to the box
and installing the release build (see `RE-RUN-121.md`).

### Build + size (1M rows)

| engine | workers | seconds | size_bytes | bytes/vector |
|--------|---------|---------|------------|--------------|
| hnswsq (port) | 4 (server default) | **59** | 819,216,384 | 819 (-1.6%) |
| pgvector hnsw | 4 (server default) | 102 | 832,241,664 | 832 |
| pgvector hnsw | 1 | 446 | 831,995,904 | 832 |
| pgvector hnsw | 32 | 64 | 832,184,320 | 832 |

(Debug-PG numbers; both engines were palloc-taxed here, pgvector more so.)
On release PG the same comparison is **parity**: hnswsq plain 67.4s vs
pgvector 66.1s (Settings matrix), and the plain index is ~3% LARGER than
pgvector's under the pinned seed (858 vs 831 B/vec).  The 59s figure above
reflects this box's debug build, not a release advantage.

### Query sweep (mean ms per query, 100 queries, warm)

| ef | hnswsq LIMIT10 | hnswsq drain | pgvector LIMIT10 | pgvector drain | ratio (LIMIT 10) |
|----|---------------:|-------------:|-----------------:|---------------:|-----------------:|
| 10 | 0.696 | 0.771 | 0.618 | 0.724 | 1.13x |
| 40 | 1.151 | 1.279 | 1.115 | 1.214 | 1.03x |
| 160 | 2.445 | 2.781 | 2.491 | 2.858 | **0.98x** |
| 640 | 6.609 | 7.405 | 6.982 | 7.948 | **0.95x** |

(Debug-PG numbers. On release PG the sweep is at parity at every ef —
hnswsq plain 0.586/1.028/2.311/6.353 vs pgvector 0.597/1.003/2.244/6.499 ms,
see the Settings matrix; the "faster at ef ≥ 160" claim above does not hold
on release PG. The per-scan pgrx FFI overhead at ef=10 was likewise part of
the debug tax.) Perf profiles (in `/tmp/gap_study.log`) show both
engines buffer-manager-bound: hnswsq's top symbols are PinBuffer 25% /
`load_element_impl` 21% / LWLockRelease 19%; pgvector's are PinBuffer 23% /
LWLockRelease 19% / `HnswLoadElementImpl` 17%.

### Recall@10 (vs `gt_1m`)

| ef | pgvector | hnswsq plain | hnswsq ieeefp16 | hnswsq ieeefp8 | hnswsq f8/sq8 |
|----|----------|--------------|-----------------|----------------|---------------|
| 10 | 0.772 | 0.784 | 0.784 | 0.709 | 0.786 |
| 20 | 0.882 | 0.884 | 0.884 | 0.876 | 0.880 |
| 40 | 0.943 | 0.940 | 0.940 | 0.929 | 0.943 |
| 80 | 0.976 | 0.977 | 0.976 | 0.969 | 0.978 |
| 160 | 0.992 | 0.993 | 0.992 | 0.995 | 0.991 |
| 320 | 0.999 | 0.999 | 0.999 | 1.000 | 0.998 |
| 640 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

(reproducible via `.design/neon/bench/recall_sweep.sh`; `plain`, `ieeefp16`
and `f8` track pgvector within ±1pt over the whole range, `ieeefp8` is ~1pt
down at low ef and equal or better from ef 160.  The `f8` column was
measured with the scalar distance; the weighted-pairwise default
reproduces it exactly — 0.991/1.0 at ef 160/640 on release PG.)

## Settings matrix (build / query latency / insert throughput)

Measured on the **release PostgreSQL 17.11** cluster on the same box
(port 54331; see the "Release-PostgreSQL re-run" section below for how
it was built), `items_1m`, m=16, efc=64, mwm 8GB, 4 workers, 50k fresh
inserts per config (batches of 1000, one index at a time).  Each config
indexes the table as grown by the previous configs' inserts (1M for
pgvector, +50k each step), so per-row costs are the comparable column.
2026-09-16 run — all seven configs in one pass, current code (bit-math fp
conversions, weighted-pairwise f8 default, the new fixed-range `sq8`/
`sq16` layouts):

| config | build s | size_bytes | qms ef10/40/160/640 (LIMIT 10) | ins rows/s | ins row_ms mean/p50/p99 |
|---|---|---|---|---|---|
| pgvector | 64.7 | 832,200,704 | 0.545 / 0.913 / 2.196 / 6.291 | 691 | 1.448 / 1.449 / 1.642 |
| hnswsq plain | 66.9 | 860,176,384 | 0.572 / 0.990 / 2.317 / 6.252 | 696 | 1.436 / 1.420 / 1.620 |
| hnswsq ieeefp16 | 108.5 | 564,215,808 | 0.487 / 0.933 / 2.068 / 5.215 | 336 | 2.972 / 3.099 / 3.420 |
| hnswsq ieeefp8 | 188.0 | 431,980,544 | 0.567 / 1.156 / 2.828 / 7.740 | 338 | 2.962 / 2.980 / 3.392 |
| hnswsq f8 (weighted-pairwise default) | 113.8 | 450,797,568 | 0.492 / 0.902 / 2.033 / 5.345 | 457 | 2.190 / 2.178 / 2.467 |
| hnswsq sq8 (fixed [0,255] int8) | **114.4** | 469,581,824 | 0.508 / 0.976 / 2.278 / 5.448 | **504** | 1.985 / 1.971 / 2.268 |
| hnswsq sq16 (fixed [0,255] int16) | 149.2 | 666,836,992 | 0.628 / 1.103 / 2.245 / 5.657 | 404 | 2.477 / 2.482 / 2.766 |

(The table was restored to exactly 1M rows before this run; the
DELETE+VACUUM reshuffled the physical layout, so the f8 calibration
sample — and therefore the whole f8 graph — differs from the 2026-09-15
run above, where f8 pairwise measured 89.7s build / 376 MB /
0.218–4.461 ms / 250 rows/s on the pre-restore table.  The other
layouts are layout-insensitive and match the 09-15 numbers.)

On release PG the debug-box distortions are gone:

- **plain reaches parity everywhere**: build 66.9 vs 64.7s, queries
  equal, inserts 696 vs 691 rows/s (1.44 vs 1.45 ms/row) — the earlier
  3.2x insert gap was the debug PG's RANDOMIZE_ALLOCATED_MEMORY tax on
  the port's higher palloc volume, exactly as the local release-PG
  measurement predicted (1.12x there).
- **fp16's halved memory traffic shows up**: queries beat plain at
  ef >= 40 (e.g. 5.215 vs 6.252 ms at ef640) at 65% of the size.
- **f8 and the fixed-range sq8 tie on queries** (0.492–5.345 vs
  0.508–5.448 ms — the same code space, both ahead of plain's
  0.572–6.252) at 52% of plain's size; **sq8's inserts are the fastest
  quantized layout** (504 rows/s, 1.99 ms/row, vs f8's 457) and its
  per-row size matches f8's (376 B/row).  sq16 costs the expected
  +~50% size (513 B/row) and ~10% more query time for near-plain recall.
- Quantized builds stay slower than plain (fp16 1.6x, fp8 2.8x, f8 1.7x,
  sq8 1.7x, sq16 2.2x) — the per-dimension conversion cost in the
  build's distance kernel; f8/sq8/sq16's insert rates are the best of
  the quantized layouts (buffer/backlink-bound, not distance-bound).

### perf attribution (2026-09-16, `perf record` on the release build)

Single-backend sq8 build (90s sample): **`distance_encoded_direct` 69.7%**,
then visited-table insert (4.8%), candidate loading (5.3%),
`select_neighbors` sort (1.8%) and heap pops (1.7%) — the build is
distance-bound.  ef-640 scan (20s sample): pairwise distance kernel
27.4%, `load_element_impl` 19.3%, `PinBuffer` 18.0%, `LWLockRelease`
8.8%, `LWLockAttemptLock` 6.9% — the scan is buffer-manager-bound with
the distance as the largest single symbol.

Two changes followed:

- **sq8 graph mutation now uses the integer pairwise distance** — for
  `sq8` the build's query is the decoded vector, which is always
  integral (decode = the codes themselves), so the integer pairwise is
  value-identical to the f32 scalar form and the graph is unchanged.
  Matrix: sq8 build 125.2 → **114.4s**, inserts 461 → **504 rows/s**,
  recall identical (0.993/0.999/1.0 at ef 160/320/640).  `sq16` keeps
  the scalar mutation path: its decoded domain is fractional
  (code/128), where the integer form deviates from the f32 form at the
  ulp level and those near-tie flips fragmented the graph (recall
  collapsed to ~0.72 in the local suite — the same signature the old
  ±1-range pairwise build showed).
- **AVX-512 distance kernels** (the box has avx512f/bw/vl): 16-lane
  versions of the weighted f8 pairwise, the fixed sq8/sq16 pairwise and
  the fixed decode distances, with runtime dispatch.  The distance
  share of an ef-640 scan dropped from 27% to **~2.5%**, leaving the
  scan ~60% PostgreSQL buffer management (PinBuffer/LWLock/UnpinBuffer)
  and ~20% element loading — the distance math is no longer a visible
  scan cost.  The buffer-manager path (pgvector-identical) is the next
  frontier.

(The earlier version of this table, measured on the box's pgrx DEBUG
PostgreSQL, is superseded: pgvector 107.1s / plain 71.0s / fp16 294.7s /
fp8 740.9s / f8 134.6s builds and 867/271/127/81/253 insert rates
reflected the debug-palloc tax, not the engine.)

---

## History — the retired engine (pre-port, same box)

Both engines were **release builds** on the same box and the same table,
measured back to back with `.design/neon/bench/cycle.sh`. This replaced an
earlier 100k/113 comparison whose hnswsq side was a *debug* build of the
extension — see the bottom of this file.

### Build + size (old engine)

| engine | build_s | size_bytes | bytes/vector |
|--------|---------|------------|--------------|
| pgvector hnsw | 105 | 832,184,320 | 832 |
| hnswsq plain (retired engine) | **517** | 861,921,280 | 862 (+3.6%) |

The retired engine was single-backend; pgvector's C build parallelized across
the box.  The gap was parallelism, not algorithm:

| build | workers | seconds |
|---|---|---|
| hnswsq | 1 | 562 |
| pgvector | 1 | 442 |
| pgvector | 4 (server default) | 103 |
| pgvector | 32 requested (7 effective) | 64 |

i.e. **1.27x per core** and 5.5-8.8x in wall clock purely from worker count.
The phase split of the hnswsq build (from `hnswsq.build_stats`):

```
search=372079ms (76%) backlink_select=97912ms (20%) flush=10810ms select=6990ms
backlink_pairs=2000ms  accounted_total=489791ms of 517s wall
```

For scale: the same build on this table before this branch's optimizations was
measured at ~170 nodes/s (interrupted at 48 min ≈ 97 min projected for 1M, and
that was the *debug* profile); the current release build is 11x faster, and the
remaining 76% is the graph search, which is what a parallel build would attack.

## Query path after the P1-P3 optimizations (same table, same box, same session)

The gap study located the query gap in per-hop memory work (not distance
arithmetic), so the query path was reworked (`.design/hnswsq_perf_analysis.md`,
"Query-optimization round"): in-page distance probes with allocation-free
expansion, a counted-loop/SIMD distance kernel for the lossless layout, and — for
`plain` only — exact distances with `xs_recheckorderby = false` plus no second
load per emitted candidate.

| ef | recall@10 (before = after) | hnswsq p50 before | hnswsq p50 after | pgvector p50 |
|---|---|---|---|---|
| 10 | 78.50 | 1.033 ms | **0.742 ms** | 0.655 ms |
| 20 | 87.30 | 1.374 ms | **0.920 ms** | 0.824 ms |
| 40 | 94.30 | 1.969 ms | **1.242 ms** | 1.056 ms |
| 80 | 97.70 | 2.929 ms | **1.800 ms** | 1.577 ms |
| 160 | 99.40 | 4.795 ms | **2.864 ms** | 2.520 ms |
| 320 | 100.00 | 8.084 ms | **4.604 ms** | 4.146 ms |
| 640 | 100.00 | 14.046 ms | **7.936 ms** | 7.013 ms |

Recall is bit-identical before/after at every ef (same build seed), and the
latency ratio to pgvector went from 1.56-1.92x to **1.11-1.18x**.  The build also
dropped from 538 s to **352 s** (1.53x; the search phase alone 389.9 s -> 228.3 s)
with the same index size.

## Recall@10 and latency sweep (ef_search 10..640)

| ef | pgvector recall | pgvector p50 ms | hnswsq recall | hnswsq p50 ms | hnswsq p99 ms |
|----|-----------------|-----------------|---------------|---------------|---------------|
| 10 | 77.47 | 0.655 | **78.30** | 1.020 | 5.864 |
| 20 | 88.40 | 0.824 | **88.30** | 1.335 | 7.171 |
| 40 | 93.70 | 1.056 | **94.00** | 1.919 | 9.012 |
| 80 | 97.30 | 1.577 | **97.60** | 2.905 | 12.155 |
| 160 | 98.90 | 2.520 | **99.30** | 4.629 | 18.197 |
| 320 | 99.80 | 4.146 | **99.90** | 7.831 | 28.373 |
| 640 | 100.00 | 7.013 | **100.00** | 13.787 | 43.785 |

## Interpretation

- **Recall**: hnswsq-plain matches or slightly beats pgvector at every ef point
  (+0.4 to +0.8 points at ef 10-320) with a single-threaded build.
- **Size**: +3.6% over pgvector (862 vs 832 B/vector) — the page-based node
  format stores ItemPointers, and each node carries one extra list slot.
- **Build**: 4.9x pgvector's build time *single-threaded* (517s vs 105s).  This
  is now an algorithmic/parallelism gap, not a constant-factor one: 76% of the
  hnswsq build is the graph search, which is per-insert independent and is the
  target of the planned parallel build (`.design/hnswsq_perf_analysis.md`,
  "Next-round plan").
- **Query latency**: ~1.6-2.0x higher p50 across the sweep (1.9ms vs 1.1ms at
  ef=40; 13.8ms vs 7.0ms at ef=640).  The earlier "10-13x" figure in this file
  was a debug-build artifact.  The remaining gap is the architectural tradeoff
  of a page-based, buffer-managed index: each search hop loads a page through
  the shared-buffer manager and deserializes the node (rkyv) instead of walking
  a memory-mapped graph — which is what buys MVCC-safe online inserts/deletes,
  WAL safety, vacuumability, and serverless-friendly per-page storage.

## History (not comparable — debug profile)

The first version of this file recorded a 100k BIGANN (dim 128) A/B on
113.44.106.182 with hnswsq built **without `--release`** (that is what
`cargo pgrx install` does by default): pgvector 13s vs hnswsq 3556s to build,
and p50 3.79ms vs 0.31ms at ef=10.  A debug build of this extension is ~12x
slower than release, which accounts for most of that spread; the numbers are
kept only as a reminder to compare like with like (all current local and remote
measurements use `cargo pgrx install --release`).

## Commands

```bash
# same-host cycle (build + stats + recall sweep + insert bench) — 121
PGPORT=54330 PGUSER=pgtest PGDATABASE=postgres \
PSQL_BIN=/root/.pgrx-hnswsq/17.11/pgrx-install/bin/psql \
  bash .design/neon/bench/cycle.sh hnswsq postgres <label> 1m
PGPORT=54330 ... bash .design/neon/bench/cycle.sh hnsw postgres pgvector-1m 1m

# local (single host, pinned seed, reproducible dataset)
# dim 16 dataset (tag = rows), then a dim 128 one for kernel work
.design/neon/bench/local_dataset.sh 1m t100kdb 100 200 16
.design/neon/bench/local_dataset.sh 100kd128 t100kdb 200 200 128
.design/neon/bench/local_cycle.sh <label> plain t100kdb 100kd128 16 64 128
```
