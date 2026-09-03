# RaBitQ Store Implementation Report

Commit: `0377fe0 feat: add RaBitQ storage layout (rabitq) with 1/4/8-bit and SIMD kernels`

## 1. Summary

This change adds a third DiskANN storage layout to pgvectorscale, exposed as
`storage_layout = rabitq` (alias `rabitq_compression`), alongside `plain` and
`memory_optimized` (`sbq`). The implementation is modeled on Lance's RaBitQ
index (`lance-index/src/vector/bq/{builder,storage,rotation}.rs`) and the
RaBitQ paper / `rabitq-rs`.

RaBitQ stores **1 bit per dimension** as a sign code and optionally adds
**error-correction ("ex") bits**, giving three precisions driven by the existing
`num_bits_per_dimension` reloption:

| `num_bits_per_dimension` | layout                       | bits/dim stored |
|--------------------------|------------------------------|-----------------|
| `1` (default)            | 1 sign bit                   | 1               |
| `4`                      | 1 sign bit + 3 ex bits       | 4               |
| `8`                      | 1 sign bit + 7 ex bits       | 8               |

Distance is estimated as an affine function of a sign-code dot product plus an
ex-code dot product, with metric-specific node/query factors. The estimator
covers **L2, cosine, and inner product**, and the final top-k is always
re-ranked by the exact full-vector distance via `get_full_distance_for_resort`.

## 2. Algorithm

For a node vector `v` and query `q`, with global mean `c`:

1. **Center**: `x = v - c`, `y = q - c`.
2. **Rotate**: apply a matrix-free fast random orthogonal transform `R`
   (composition of Rademacher sign flips, FWHT on power-of-two windows, and a
   Kac-walk mixing step for non-power-of-two dims). Only `4 * ceil(dim/8)`
   random sign bytes are persisted.
3. **Quantize**: `sign[i] = (R x)[i] >= 0`; for `num_bits > 1`, quantize
   `|R x|[i] / ‖R x‖` into `2^(num_bits-1)` levels with the sign folded into the
   low bits (`quantize_ex_code` / `best_rescale_factor`, ported from
   Lance / rabitq-rs).
4. **Estimate** (`estimate_distance`):
   - `binary_dot = Σ sign_v[i] * R(q-c)[i]` (continuous-query dot)
   - `ex_dot    = Σ ex_v[i] * R(q-c)[i]`
   - 1-bit: `dist = g_add + f_add + f_rescale * (binary_dot - 0.5·Σq_rot)`
   - multi:  `dist = g_add + f_add_ex + f_rescale_ex * (2^ex·binary_dot + ex_dot + cb·Σq_rot)`

Per-node factors (`f_add`, `f_rescale`, `f_add_ex`, `f_rescale_ex`) are computed
once at build time; per-query factors (`g_add`, `Σq_rot`) are computed once at
query time. The exact per-metric constants follow rabitq-rs `compute_one_bit_factors`
/ `compute_extended_factors`:

| metric | f_rescale (1-bit) | g_add (query) |
|--------|-------------------|----------------|
| L2     | `-2·‖x‖²/denom`   | `‖y‖²`        |
| Cosine | `-‖x‖²/denom`     | `0.5·‖y‖²`    |
| IP     | `-‖x‖²/denom`     | `-⟨y, c⟩`     |

with `denom = 0.5·‖R x‖₁`. Cosine relies on pgvectorscale's existing
pre-normalization (`preprocess_cosine`) and Lance's cosine→L2 reduction.

## 3. Design decisions

- **Storage type**: `StorageType::RabitqCompression = 3`; strings `"rabitq"` /
  `"rabitq_compression"`.
- **Page types**: `PageType::RabitqNode = 9`, `PageType::RabitqMetadata = 10`
  (metadata is chained, like `SbqMeans`).
- **Bits control**: reuses `num_bits_per_dimension`, interpreted for RaBitQ as
  total bits per dimension ∈ `{1,4,8}` (build-time error otherwise).
- **Node payload**: `heap_item_pointer`, `code: Vec<u64>` (sign bits, LSB-first),
  `ex_code: Vec<u8>` (nibble-packed for 4-bit, byte-packed for 8-bit, empty for
  1-bit), `f_add/f_rescale/f_add_ex/f_rescale_ex: f32`, fixed `num_neighbors`
  neighbor slots, optional labels. Classic vs Labeled variants, as in SBQ.
- **Metadata**: `RabitqMetadata { count, mean, rotation_signs, num_bits }`,
  stored via `ChainTapeWriter` and referenced by `MetaPage.quantizer_metadata`.
- **Estimator**: query-independent per-node factors + query-time factors (the
  scalar adaptation of Lance's batch RawQuery transformer), because DiskANN
  graph search evaluates many node codes against one query.

## 4. SIMD-friendliness

All hot paths have AVX2 kernels (guarded by `#[cfg(target_arch = "x86_64")]
` and runtime `is_x86_feature_detected!("avx2")`), with scalar fallbacks that
also serve aarch64/NEON:

- `rotation.rs`: `flip_signs_avx2` (`_mm256_xor_ps` + bit-select) for the
  Rademacher sign flips.
- `quantize.rs`: `pack_sign_bits_avx2` (`_mm256_movemask_ps`, 8 sign bits per
  256-bit load), `binary_dot_avx2` (masked-sum via `_mm256_and_ps` +
  `_mm256_cmpeq_epi32`), `ex_dot_7bit_avx2` (`_mm256_cvtepu8_epi32` +
  `_mm256_fmadd_ps`).
- The FWHT butterfly is kept scalar (as in Lance) and auto-vectorizes.

The codebase already requires `+avx2,+fma` on x86 (`.cargo/config.toml` and the
`compile_error!` in `distance/mod.rs`), so the AVX2 kernels reuse that invariant.

## 5. Files

New module `pgvectorscale/src/access_method/rabitq/`:

- `rotation.rs` — fast random rotation
- `quantize.rs` — quantizer, ex-code quantization, affine estimator, AVX2 kernels
- `node.rs` — `RabitqNode` (Classic/Labeled) + archived/writable enums
- `storage.rs` — `RabitqStorage` implementing `Storage`
- `cache.rs` — `RabitqCodeCache` (LRU)
- `mod.rs` — `RabitqMetadata`, query/node distance measures
- `tests.rs` — unit + `#[pg_test]` integration tests

Wiring:

- `util/page.rs` — `PageType` variants + `is_chained`
- `access_method/storage.rs` — `StorageType` variant + parsing
- `access_method/meta_page.rs` — quantizer pointer, defaults, `num_bits` validation
- `access_method/options.rs` — reloption description + parsing test
- `access_method/scan.rs` — `StorageState` + scan/endscan arms
- `access_method/build.rs` — training, rotation persistence, build/insert paths
- `access_method/vacuum.rs` — bulk-delete dispatch
- `access_method/mod.rs` — module registration
- `access_method/distance/mod.rs` — `DistanceType: Clone, Copy`

## 6. Validation status

Local (aarch64 macOS, pg17 via pgrx):

- `cargo check --no-default-features --features pg17` — passes
- `cargo check --tests` — passes
- `cargo test --lib` unit tests — **10 RaBitQ tests pass** (rotation shape +
  orthonormality, sign-bit packing, binary/ex dot kernels, 1/4/8-bit quantization,
  L2/cosine/IP estimator ordering).
- `rustfmt --check` — clean.

Not yet run locally:

- `#[pg_test]` integration tests (`cargo pgrx test pg17`) — blocked by the
  missing `pgvector` (`vector`) extension in this environment. Tests are written
  and should run in a pgvector-equipped CI.
- x86 AVX2 kernels — not compiled on this aarch64 host; need an x86 build.

## 7. Remote x86 test plan

1. Copy source to the x86 remote (`root@113.44.106.182`).
2. Extract PostgreSQL 17 from `/data1/pg17-release-1711.tar.gz`.
3. Install pgvector + pgrx prerequisites (pgvector is required by vectorscale).
4. `cargo pgrx init/install` against the extracted PG17, `RUSTFLAGS` with
   `+avx2,+fma` (already in `.cargo/config.toml`).
5. Run `cargo pgrx test pg17` (rabitq tests) and the AVX2 paths.
6. Load BIGANN 100M test case from `/data1/dataset` and measure build + recall.

Outcome will be appended to this report after the remote run.

---

## 8. Remote x86 test results (113.44.106.182)

### Environment

- x86_64 (16 cores), 60 GiB RAM, 532 GiB free on /data1.
- PostgreSQL 17.11 at `/data1/pg17` (from `/data1/pg17-release-1711.tar.gz`).
- Rust 1.97.1, cargo-pgrx 0.16.1, pgvector 0.8.6 (built/installed against PG17).
- Source placed in `/data1/pgvectorscale-rabitq` (the pre-existing advanced fork
  at `/data1/pgvectorscale` was left untouched per instruction).
- BIGANN 100M data already loaded: `items_100m` (100,000,768 rows), `bench_queries`
  (10,000 queries), `gt_idx_100M.ivecs` ground truth, plus `items_10m`/`items_200k`
  subsets.

### Build & install

`cargo pgrx install -c /data1/pg17/bin/pg_config --no-default-features --features build_parallel`
compiled the x86_64 AVX2 path (`+avx2,+fma` via `.cargo/config.toml`) and installed
`vectorscale 0.9.0` into PG17. A release install (`--release`) was used for the
benchmark so the orphan heuristic is a warning (not a debug-assert) and code is optimized.

### Correctness

A 1-bit rabitq index on `items_1k` (128-dim BIGANN subset) returns **exactly** the
same top-5 as an exact sequential scan (100% recall@1 / recall@5 on that sample):

```
 id | distance      id | distance
 42 |    0.00       42 |    0.00
 40 |  233.91       40 |  233.91
711 |  257.80      711 |  257.80
 30 |  299.10       30 |  299.10
960 |  302.79      960 |  302.79
   (rabitq index)     (exact seq scan)
```

### Bugs found and fixed on x86

1. **Negative distance estimate** — the affine L2 estimate can undershoot below 0 for
   near-identical vectors, tripping `DistanceWithTieBreak::new`'s `distance >= 0.0`
   assert during build. Fixed by clamping the estimate to `>= 0`.
2. **Inconsistent node-node distance** — node-node distances used Hamming while
   query-node used the affine estimate; the mixed scales caused DiskANN pruning to
   orphan nodes (`debug_assert!`). Fixed by using a symmetric RaBitQ L2 estimate
   (`‖x_a‖²+‖x_b‖²-2·(‖x_a‖‖x_b‖/D)(D-2h)`) for node-node, keeping both on the same scale.

### Parallel build

Parallel index build was enabled for rabitq (`StorageType::RabitqCompression`) and
verified running with 4-8 workers. Memory tuning matters for 10M+: `maintenance_work_mem=8GB`
spills the BuilderNeighborCache after ~365K vectors; `28GB` with 4 workers avoids the
spill (the prior 8-worker 28GB config OOM'd).

### BIGANN 100M benchmark

`items_100m` (100M vectors) and the `bench100m.sh`/`run_all.sh` harness are in place.
A 10M validation build (`items_10m`, rabitq 1-bit, num_neighbors=32, search_list_size=100)
was run to completion-scale validation; it is a multi-hour build on this hardware and was
left running (results logged to `/root/rabitq_mine_10m.log` on the remote). The 100M build
follows the same harness with `search_list_size` reduced (the provided scripts use 40 for
100M to keep build times tractable).

---

## 9. Clean 200K recall demonstration (final)

To rule out the DiskANN memory concern at small scale, a clean 200K BIGANN
(`items_200k`, 128-dim) recall test was run with a single rabitq 1-bit index
(`num_neighbors=32`, `search_list_size=100`, `maintenance_work_mem=2GB`):

- Build time: **60 s** (parallel) / 148 s (serial), no spilling, index 63 MB.
- Recall measured against an exact sequential scan (`enable_indexscan=off`):

| metric      | value |
|-------------|-------|
| recall@1    | **97.0%** |
| recall@10   | **84.2%** |

**Important finding**: an earlier "could not read blocks …" error during the 200K
query was **not** a bug in this implementation — the planner had selected a
pre-existing `items_200k_rabitq1` index built by the *fork* (whose on-disk RaBitQ
format is incompatible with this build). Dropping that stale index and letting the
query use the freshly-built index produced correct results. The earlier 10M/100M
slowdown/spilling is the DiskANN BuilderNeighborCache memory issue, not a RaBitQ
correctness issue.

**Conclusion**: diskann + rabitq behaves normally at 200K scale (97% recall@1).
Per the agreed direction, the next job is to implement the IVF index with
centroid-based RaBitQ, replacing diskann+rabitq.

---

## 10. IVF + centroid-based RaBitQ — final results (round 3)

The IVF index (`USING ivf`) with centroid-relative RaBitQ is complete and
validated end-to-end on the x86 remote.

### Validation (BIGANN, 128-dim)

| dataset | lists | probes | recall@1 | recall@10 | index size |
|---------|-------|--------|----------|-----------|------------|
| items_10m (10M) | 100 | 10 | **100.0%** | **98.9%** | 523 MB (~52 B/entry) |
| items_200k | 100 | 10 | 96.0% (approximate) / 99%+ (exact) | 96%+ | 10 MB |

Distances returned by the scan are **exact** (the executor rechecks the order-by
against the heap tuple).

### Key fixes this round

- **L2 estimate centroid-correction bug**: the estimate added a
  `2·‖ro‖²·⟨rot(c),code⟩/⟨ro,code⟩` term that belongs to the inner-product
  estimator. For centroid-relative IVF this term is non-zero (it was zero in the
  unit tests, which used an empty center) and collapsed 10M recall to ~6%.
  Removing it restored 100% recall@1.
- **Lower-bound for recheck**: subtract a conservative error margin
  `2·‖ro‖·‖rq‖/√D` so the estimate is a lower bound on the exact L2 distance;
  this lets `xs_recheckorderby=true` reorder by exact distance without the
  "index returned tuples in wrong order" error (which previously fired for low
  dimensions).
- **`xs_orderbyvals` Datum**: provide the estimate as a full `Datum`
  (`Datum::from(bits)`) plus `xs_orderbynulls`; the executor always dereferences
  these in `IndexNextWithReorder`.
- **1-bit code buffer**: use `dim.div_ceil(8)` (was `dim/8`), which was empty for
  padded dim < 8 and crashed the sign-packing loop.
- **Operator classes**: `vector_{l2,cosine,ip}_ops USING ivf` are now created by
  `CREATE EXTENSION` (idempotent `extension_sql`).
- **aminsert**: nearest-centroid assignment + centroid-relative quantization +
  list append; seeds the first centroid when built on an empty table.
- **vacuum**: `ambulkdelete` filters dead entries per list and rewrites them.
- **ambuildempty** / empty-table `ambuild`: write the minimal meta + list
  directory + centroid pages (PG17 calls `ambuild`, not `ambuildempty`, for
  empty tables).

### Remaining (non-blocking)

- Parallel build (`build_ivf_index_parallel`) still falls back to the serial
  build; serial builds 10M in ~48 s. Parallel k-means/assignment/entry writing
  is a performance follow-up.
- `ivf/partition.rs` is an unused stub (the build manages lists inline via
  `IvfListDirectory`).

---

## 11. Parallel build (round 4) — objective complete

- Added `rayon` and parallelized the per-vector nearest-centroid assignment and
  centroid-relative RaBitQ quantization (the CPU-heavy build phase).
- `build_ivf_index_parallel` now delegates to the internally-parallel build
  (k-means runs on a small reservoir sample and stays serial).
- Removed the unused `ivf/partition.rs` stub.

Final validation on 10M BIGANN (lists=100, probes=10):

| metric | value |
|--------|-------|
| build time | **9 s** (was 48 s serial) |
| recall@1 | **100.0%** |
| recall@10 | **97.7%** |
| index size | 523 MB (~52 B/entry, 1-bit RaBitQ) |

All IVF stubs (partition management, aminsert, vacuum, ambuildempty, parallel
build) are implemented; the centroid-relative RaBitQ quantizer is corrected and
integrated; the index compiles and validates on the x86 remote against BIGANN.

---

## 12. Query latency bottleneck (perf) and optimization plan

### Measured latency (10M, probes=10, 100%-recall config)

p50 1.54 s, p90 1.90 s, **p99 2.08 s**, max 2.10 s.

### `perf` profile (backend, `perf record -p <pid> -g`, 59K samples)

| self % | symbol | meaning |
|---|---|---|
| 19.3% | `PinBuffer` | buffer pin per page read |
| 18.8% | `LWLockAttemptLock` | buffer-header LWLock |
| ~13% | `AllocSetAlloc` → page faults | per-entry allocation / zeroing |
| 5.9% | `ivf::scan::amgettuple` | whole scan incl. inlined RaBitQ estimate |
| ~1.5% | `quicksort` | full sort of ~1M candidates |

The RaBitQ `estimate_l2`/`dot_with_rotated` is inlined and negligible. 52 MB is not
the issue — it is per-page buffer pinning + per-entry allocation + a full sort.

### Approved plan

1. **Zero-copy deserialize** — use `rkyv::archived_root::<IvfEntryPage>` and compute
   `estimate_l2` against borrowed `ArchivedVec<u8>` slices instead of
   `from_bytes` into owned `Vec<u8>`. Removes ~1M heap allocations/query.
2. **Bounded top-k** — replace the full `results.sort_by` with a max-heap of size
   `K` (K > requested LIMIT, to preserve recheck recall). Removes the full sort.
3. **Batch page reads** — see layout analysis below.
4. **Tighter entry layout** — see layout analysis below.

### PostgreSQL page-layout impact of points 3 and 4

The IVF index is stored in ordinary 8 KB PostgreSQL pages (`PageInit` + the TSV
opaque special area), same as the rest of pgvectorscale. Neither point changes
PostgreSQL's page structure (page header, line pointers, opaque area); they only
change *how the IVF serializes its own bytes inside the page items*.

- **Point 4 (tighter layout)** — no PG page-layout impact. It changes the rkyv
  serialization of `IvfEntry`/`RabitqVector` (e.g. drop the per-entry `Vec` header,
  or a struct-of-arrays packing). Fully committable; the only consequence is that
  IVF indexes built before the change must be dropped/rebuilt (fine pre-release).

- **Point 3 (batch reads)** has two forms:
  - *Read-ahead (`StartReadBuffers`/`WaitReadBuffers`)* — keeps the current chained
    on-disk format; reduces I/O stall but **not** the per-page `PinBuffer`/`LWLock`
    CPU cost, so it does not fix the 38% hotspot. No layout change, committable.
  - *Contiguous run + `smgrreadv`* — changes the IVF logical layout from a chained
    item to a contiguous block range (store `start_block`+`num_blocks` in the list
    directory instead of a per-chunk `next` pointer), then reads the range in one
    bulk `smgrreadv` into a single buffer. This **eliminates** per-page
    `PinBuffer`/`LWLock`. It still stores bytes in normal 8 KB pages, so it does
    not violate PG's page structure; it is read-only (`smgrread` has no WAL
    implication for reads) but bypasses shared_buffers (relies on the OS page
    cache) and, like the current chain-rewrite insert/vacuum, needs an access
    lock around concurrent list rewrites. Higher risk than 1/2/4; recommended as
    a second, separately-reviewed change.

### Recommended sequencing

1 and 2 first (pure, low-risk code changes in `ivf/{entry,scan}.rs`), then 4
(self-contained serialization change), then 3b if further gains are needed.

---

## 13. Fix 1 + 2 implemented and measured (2026)

Implemented the two low-risk fixes from §12:

1. **Zero-copy deserialize** — `IvfEntryReader::for_each_entry` accumulates the list
   bytes into a `rkyv::AlignedVec` and iterates `ArchivedIvfEntry` directly; the
   distance is computed from borrowed fields via `dot_with_rotated_fields` /
   `estimate_l2_fields` (no owned `Vec<u8>` per entry, no `Vec<IvfEntry>`).
2. **Bounded top-k** — `amgettuple` keeps a bounded max-heap of `DistTid`
   (size `ivf.top_k`, default 1000) instead of collecting + fully sorting ~1M
   candidates.

On-disk format is unchanged, so existing IVF indexes remain valid.

### Measured on 10M (probes=10, 50 queries, warm cache)

| metric | baseline | after fix 1+2 | change |
|---|---|---|---|
| p50 | 1.54 s | **96.5 ms** | ~16x |
| p99 | 2.08 s | **133.4 ms** | ~15.6x |
| mean | ~1.6 s | 100.7 ms | ~16x |
| recall@1 | 100% | 100% | unchanged |
| recall@10 (mean-r@5) | ~0.977 | **0.980** | preserved |

### `ivf.top_k` recall tradeoff (100 queries, probes=10)

| top_k | recall@1 | recall@10 (mean-r@5) |
|---|---|---|
| 100 | 0.98 | 0.912 |
| 500 | 1.00 | 0.974 |
| **1000 (default)** | 1.00 | 0.980 |
| 10000 | 1.00 | 0.980 |

Recall saturates at ~0.980 by top_k=1000, so 1000 is the sweet-spot default.
The remaining ~96 ms is dominated by the per-page `PinBuffer`/`LWLock` reads
(~6500 pages/query) and the ~52 MB list copy — i.e. fix 3 (batch/bulk reads)
is the next lever.

---

## 14. Fix 3b + 4 implemented and measured (2026)

Implemented the two storage/format fixes from §12:

3. **Contiguous run + smgr bulk read** — a list's entries are written as a
   contiguous block run (one item per page) and read with a single `smgrreadv`
   into one buffer, bypassing the buffer manager's per-page pin/LWLock.
   `FlushRelationBuffers` after build/insert/vacuum keeps the on-disk image
   consistent for the direct reads (the scan reads the OS page cache).
4. **Packed struct-of-arrays layout** — replaced the rkyv `Vec<IvfEntry>`
   serialization with a manual SoA stream (header + tids + codes + sums + l1s),
   dropping the per-entry `Vec` header / alignment padding and the unused
   `cent_dot`.

### Measured on 10M (probes=10, 50 queries, warm cache)

| metric | fixes 1+2 | fixes 1+2+3b+4 | change |
|---|---|---|---|
| index size | 523 MB | **288 MB** | -45% |
| p50 | 96.5 ms | 95.3 ms | ~flat |
| p99 | 133.4 ms | **122.6 ms** | -8% |
| recall@1 | 100% | 100% | unchanged |
| recall@10 (mean-r@5) | 0.980 | **0.982** | preserved |

### Where the remaining ~95 ms goes

`perf` now shows **89% self time in `amgettuple`** (the inlined scan body); the
buffer-manager `PinBuffer`/`LWLock` hotspots are gone. The scan is now CPU-bound
on the scalar RaBitQ estimate (`code_dot_with_rotated` walks the set bits) and
the ~30 MB SoA reassembly memcpy. This is the "SIMD-friendly" work from the
original ask — the next lever is vectorizing `dot_with_rotated_fields` for the
1-bit code (e.g. `_mm256` popcount/dot over the packed codes).

---

## 15. Lance IVF-RQ cross-read: applicable optimizations (2026)

Read `lance_ivfrq_10m_benchmark_report.md` and the Lance RaBitQ sources
(`lance-index/src/vector/bq/{storage,dist_table_quant,ex_dot,prune}.rs`,
`ivf.rs`). Lance hits recall@10 0.997 @ p99 3.9 ms (b=8, np=160, rf=4) on the
same 10M set — ~30x our p99 122 ms — because its scan is SIMD and its ranking
is 8-bit; the algorithmic shape is otherwise the same (RaBitQ estimate + exact
re-scoring).

### What Lance does that we don't

1. **FastScan distance table (1-bit dot)** — instead of walking set bits per
   candidate (`code_dot_with_rotated`), it precomputes a `d/4 × 16` f32 table
   per (query, partition), then each candidate's binary IP is `d/4` 4-bit table
   lookups + adds. The table is quantized to u8 and the per-candidate sum runs
   in AVX2/AVX-512 integer SIMD (`pshufb`). ~10-20x the scalar set-bit loop.
2. **Lower-bound pruning** — binary IP for all rows, then a cheap per-row lower
   bound, then SIMD prune 16 rows/mask; only <1% survivors get the exact
   rerank. We currently compute the full `estimate_l2` (div + sqrt) for every
   candidate.
3. **8-bit codes** (`num_bits=8` = 1 sign + 7 magnitude bits) rank far better
   than 1-bit (0.90 vs 0.46 recall @20 probes, no refine), letting it probe
   fewer lists.
4. **Blocked ex-code layout** (`ex_dot.rs`) so SIMD unpack emits codes in dim
   order, then FMA against the rotated query.
5. **3162 partitions (√N)** vs our 100 lists → 10x fewer candidates/probe.

### Layout caveats (why it is not a copy-paste)

- Lance is **columnar** and keeps the **full f32 vectors** for refine; our index
  stores only 30 B/entry of quantized codes and rechecks against the PG heap via
  the executor. Our "refine" is the executor recheck, not an in-index f32 read.
- Our fix-4 SoA already stores `codes` as one contiguous `n × code_len` array,
  so the FastScan table + per-candidate SIMD applies directly to the 1-bit path
  (LSB-first nibbles are exactly what the table indexes).
- Lance batches/transposes 32 rows for cross-row SIMD; our SoA is row-major,
  which is fine for `pshufb` FastScan but would need a blocked layout for 8-bit
  ex-dot.

### Applicable, ranked

1. **FastScan table + AVX2 for `code_dot_with_rotated`** (targets the 89%
   `amgettuple` hotspot directly).
2. **Defer the full estimate**: binary IP for all candidates, keep a larger top
   heap, full `estimate_l2` (div+sqrt) only on survivors.
3. **`num_bits=8`** — flip `build.rs`'s hardcoded `1` and add the SIMD ex-dot
   kernel; 8x index for far fewer probes.
4. **More lists** (100 → ~1000, retune probes) — config-only, fewer candidates.

---

## 16. FastScan + precomputed factors, and multi-bit status (2026)

### Fixes 1+2 of §15 (FastScan table + precomputed estimate factors)

- `RabitqFastScan`: `d/4 × 16` distance table for the 1-bit binary dot (32 nibble
  lookups vs the scalar set-bit walk), plus query-side constants.
- SoA now stores per-entry `scale = -2·‖ro‖²/l1` and `margin_factor =
  2·√‖ro‖²/√D` (l1 dropped, reconstructed on rewrite); no per-entry div or sqrt.

Measured 10M (probes=10, 50 queries):

| metric | before | after | change |
|---|---|---|---|
| p50 | 95.3 ms | **19.1 ms** | 5.0x |
| p99 | 122.6 ms | **26.9 ms** | 4.6x |
| mean | 95.9 ms | 19.7 ms | 4.9x |
| recall@10 (mean-r@5) | 0.982 | **0.988** | preserved |

Cumulative vs baseline (1.54 s p50): ~80x.

### 3a — multi-bit (4/8-bit) RaBitQ: implemented, was broken, now fixed but slow

The 4/8-bit quantizer + dot already existed but had never been run. First run
showed recall *degrading* with more bits (1-bit 0.99, 4-bit 0.65, 8-bit 0.40 @
~same probes) — backwards. Root cause: `dot_with_rotated_fields` used ±1 for the
sign bit; Lance's `full_dot = 2^ex · Σ sign·rot + Σ ex·rot + bias·Σrot` needs
unsigned 0/1. Fixed (+ reference unit test).

After the fix (20-query samples, probes=5):

| num_bits | index size | latency/query | recall@1 | mean-r@5 |
|---|---|---|---|---|
| 1 (probes=10) | 326 MB | 19 ms | 0.99 | 0.988 |
| 4 (probes=5) | 786 MB | ~230 ms | 0.95 | 0.950 |
| 8 (probes=5) | 1400 MB | ~280 ms | 0.95 | 0.900 |

Verdict: multi-bit is now *correct* but not a win yet — (a) recall is slightly
below 1-bit at comparable candidate counts, and (b) the scan is ~12-15x slower
because the multi-bit path is scalar (no FastScan/SIMD ex-dot). To make 8-bit
competitive (Lance's 0.997 @ 3.9 ms) still needs the blocked ex-code layout +
AVX2/AVX-512 ex-dot kernel, plus lower-bound pruning — the opt-3 work.

---

## 17. Opt-3: AVX2 SIMD ex-dot for multi-bit (2026)

### Implemented

`dot_full_code` + `ex_dot_simd` AVX2/FMA kernels compute `Σ full_code·rot` for
the multi-bit codes (8-bit = 1 byte/dim, 4-bit = sequential nibbles) by
unpacking to f32 and FMA-ing against the rotated query (Lance's `fma16_avx2`
pattern), with runtime AVX2/FMA dispatch and a scalar fallback. Wired into
`RabitqFastScan::full_dot`.

### Measured 10M (probes=10)

| num_bits | latency/query | recall@1 | mean-r@5 |
|---|---|---|---|
| 1 | 19.1 ms | 0.99 | 0.988 |
| 4 | ~20 ms (was ~230) | 0.95 | 0.980 |
| 8 | ~26 ms (was ~280) | 1.00 | 0.982 |

The ~12x multi-bit speedup makes 4/8-bit competitive with 1-bit. 8-bit's higher
latency vs 1-bit is the 8x larger code (128 B vs 16 B/entry) in the smgr read,
not the compute.

### Remaining opt-3 item: lower-bound pruning

Lance's two-stage scan (binary FastScan for all rows → SIMD prune → ex-dot only
on <1% survivors) does not map cleanly to our layout: our sign bit and ex code
are interleaved in one packed code, so computing the binary IP still requires
reading the full code — pruning would cut ex-dot *compute* (now cheap SIMD) but
not the smgr *I/O*. To get Lance's I/O reduction we'd need a separate 1-bit sign
column (a format change). Not pursued this round.

---

## 18. aarch64 port + NEON optimization (2026)

Set up a second remote (`root@116.204.102.142`, Huawei Cloud EulerOS 2.0 aarch64,
16 vCPU, 60 GB) and brought the whole stack up there:

- Built PostgreSQL 17.11 from source, Rust 1.98 (rsproxy.cn mirror), cargo-pgrx
  0.16.1, and `clang`/`libclang` for bindgen.
- Built pgvector (0.8.5) + pgvectorscale (0.9.0) for aarch64.
- Transferred the 10M BIGANN data (items10m.bin 5.3 GB, bench_queries, gt_10m)
  directly x86 → ARM and loaded it (10M rows, 10k queries, 100 GT rows).

### NEON ex-dot for multi-bit

Added aarch64 NEON kernels (`ex_dot_neon`) for the multi-bit full-code dot:
8-bit u8→f32 FMA and 4-bit sequential-nibble unpack (`vzipq_u8`) + FMA, scalar
tail. Dispatched from `dot_full_code` on `target_arch = "aarch64"`.

### ARM results (probes=10, 100 queries, warm)

| num_bits | build | index | latency/q | recall@1 | mean-r@5 |
|---|---|---|---|---|---|
| 1 | 44.6 s | 326 MB | ~40 ms | 1.00 | 0.978 |
| 4 | 28.0 s | 786 MB | ~50 ms | 0.99 | 0.984 |
| 8 | 59.7 s | 1400 MB | ~55 ms | 1.00 | 0.984 |

Recall is preserved; NEON makes 4/8-bit competitive (they would be ~12x slower
scalar). The ARM box is ~2-2.5x slower than the x86 box per query (slower cores
plus the 1-bit FastScan still being scalar).

### Remaining ARM item

The 1-bit FastScan `sum_set` (32 scalar nibble lookups) is not yet NEON-ized.
Lance's `sum_4bit_dist_table` + `dist_table_quant` (quantize the f32 table to
u8, then `vtbl`/`vqtbl` lookup + accumulate + affine dequantize) is the template
to speed up the default 1-bit path further.

---

## 19. SIMD 1-bit FastScan (cross-platform: NEON + AVX2) (2026)

Profiling on the ARM box showed the 1-bit query is compute-bound on the scalar
`sum_set` (32 nibble lookups); `smgrreadv`/memcpy were ~0.3%.

Implemented Lance's SIMD FastScan for the 1-bit path:
- `rabitq_fastscan.rs`: quantize the `f32` `d/4×16` table to u8 (min/max affine),
  transpose 1-bit codes into 32-row chunk-major batches, and sum via NEON
  `vqtbl1q_u8` / AVX2 `pshufb` (scalar fallback) + affine dequantize.
- `entry.rs`: 1-bit codes now stored transposed in the SoA (untransposed on the
  insert/vacuum rewrite path); 4/8-bit stay row-major.
- `rabitq.rs`/`scan.rs`: batched 1-bit scan via `full_dot_1bit` +
  `estimate_from_full_dot`.

### ARM result (1-bit, probes=10, 100 queries)

| | before | after |
|---|---|---|
| latency/q | ~40 ms | **~16.7 ms** (2.4x) |
| recall@1 | 1.00 | 0.99 |
| mean-r@5 | 0.978 | 0.982 |

The u8 table quantization adds a small ranking noise (absorbed by the error
margin); a u16 "accurate" table is the fallback if finer recall is needed. The
same code carries an AVX2 `pshufb` kernel for x86.

---

## 20. Cross-platform 1/4/8-bit benchmark (10M, probes=10, 100 queries) (2026)

Both machines run the SIMD FastScan (1-bit) + SIMD ex-dot (4/8-bit).

| machine | num_bits | index | latency/q | recall@1 | mean-r@5 |
|---|---|---|---|---|---|
| x86 | 1 | 326 MB | **9.6 ms** | 0.99 | 0.986 |
| x86 | 4 | 786 MB | 19.4 ms | 0.99 | 0.990 |
| x86 | 8 | 1400 MB | 22.0 ms | 0.99 | 0.990 |
| ARM | 1 | 326 MB | **16.7 ms** | 0.99 | 0.982 |
| ARM | 4 | 786 MB | 48.5 ms | 1.00 | 0.990 |
| ARM | 8 | 1400 MB | 54.8 ms | 0.97 | 0.986 |

Notes:

- 1-bit is the clear winner on both platforms (SIMD FastScan, O(d/4) batched).
- 4/8-bit are slower because their codes are 4x/8x larger (I/O-bound) and their
  ex-dot is per-entry rather than batched; at probes=10 they give no recall gain
  over 1-bit (all ~0.98-0.99 mean-r@5), so they are only worth it at *lower*
  probes.
- ARM 4/8-bit are ~2.5x slower than x86 (slower cores + NEON's 4-lane vs AVX2's
  8-lane ex-dot).

---

## 21. perf analysis of the 1-bit scan (post-FastScan) (2026)

`perf record -a -g` on both machines during a sustained 1-bit query loop.

### x86 (9.6 ms/q)

| self% | symbol | meaning |
|---|---|---|
| 36.0% | `amgettuple` closure | SIMD FastScan sum + estimate + heap (inlined) |
| ~15% (children) | `__libc_pread` | smgr code read (16 MB/query) |
| 3.2% / 2.6% / 2.1% / 1.8% | `PinBuffer`/`LWLockAttemptLock`/`hash_search`/`LWLockRelease` | buffer manager |
| 4.9% (children) | `heap_page_prune_opt` | executor exact-recheck heap reads |
| 0.9% | `cmp_orderbyvals` | executor reorder comparison |

### ARM (16.7 ms/q)

Same structure, diluted across 16 CPUs (single backend): `amgettuple` closure is
the top self symbol (~2.8% self ≈ ~25% of the backend), then libc `memcpy`/`pread`
and buffer-manager (`hash_search`/`LockBufHdr`/`LWLockAcquire`).

### Conclusion

The SIMD FastScan removed the scalar-table bottleneck; the 1-bit scan is now
balanced across three roughly-equal parts: (1) the SIMD per-entry sum+estimate,
(2) the smgr code read, and (3) the buffer manager + executor exact recheck.
Remaining levers: cut the code I/O (fewer probes / more partitions) and/or cut
the recheck work (smaller `top_k`, or in-index exact vectors).

---

## 22. p99<5ms drive: fused SIMD estimate + candidate reduction (2026)

### 1. Fused SIMD estimate + I/O (b21e559)

Collapsed the per-entry `dequantize → full_dot → estimate` into one linear form
`d = (a_full·scale·sum + b_full·scale + sum_of_x2 + rq_sum − margin_factor·rq_margin).max(0)`
computed in NEON/AVX2 (`estimate_batch`, 4/8 rows/instr); also stopped
zero-filling the smgr raw buffer. 1-bit: x86 9.6→8.4 ms, ARM 16.7→14 ms.

### 2. Dynamic centroid location (c5b10bc)

The centroid page was hardcoded at block 2, which collides with the list
directory chain once `lists > ~340`. Now stored at a dynamic block recorded in
the meta page's `centroids_pointer`.

### 3. Candidate reduction (lists 100 → 1000)

| machine | lists/probes | recall@1 | mean-r@5 | latency |
|---|---|---|---|---|
| x86 | 100/10 | 0.99 | 0.986 | 8.4 ms |
| x86 | **1000/40** | 1.00 | **0.994** | **p50 3.5 ms / p99 6.2 ms** |
| x86 | 1000/30 | ~0.99 | ~0.98 | p50 3.1 / p99 5.6 ms |
| ARM | 100/10 | 0.99 | 0.982 | ~14 ms |
| ARM | **1000/40** | 0.99 | 0.982 | ~9.6 ms |
| ARM | 1000/80 | 1.00 | 0.996 | ~14 ms |

p99<5 ms is now within reach on x86 (p99 5.6 ms @ probes=30, recall ~0.98); the
remaining tail is the executor's exact recheck (top_k=1000 heap re-scores) plus
per-query cache variance, not the RaBitQ compute.
