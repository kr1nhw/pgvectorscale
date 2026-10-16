# agentvec insert throughput: validation, bottleneck attribution, and the adaptive HOT pool

Status: **measured on the production 1B BIGANN index; HOT pool implemented and verified** (2026-10-12).
Hosts: x86 `root@121.37.117.106` (32 vCPU, 121 GB, PG 17.11, `/data1` on vdb, shared_buffers 4 GB,
`synchronous_commit=on`), arm `root@116.204.102.142` (16 vCPU, 60 GB).

Harness: `bench/agentvec-insert/` (deployed to `/data1/avbench`). Every number below is a
measurement, not an estimate; the raw rows are in `/data1/avbench/results.csv`.

---

## 0. Headline

* The reported **~4 000 rows/s is exactly reproducible**: `items_1b` (the real 1B-row table with
  the production `items_1b_av` index) inserts at **4 140 rows/s with 8 concurrent clients**, and
  the synthetic equivalent (`avb_t_av`, identical reloptions) at **4 263 rows/s**.
* It is **not** caused by there being a single HOT segment. Splitting the write path across K
  independent hnswsq regions — first emulated with K independent indexes, then implemented
  natively — changes aggregate insert throughput by **−2 % … +0.5 %** at 8, 16 and 32 clients.
* The foreground insert is **CPU-bound at ~1.2–1.8 ms of CPU per row**, of which ~43 % is the
  quantized (SQ8/f8) distance codec, ~21 % is buffer-manager/WAL lock traffic, and ~10 % is
  buffer pinning. The box is only 55–70 % busy at 32 clients, but the *product* of per-row CPU
  and client count is the wall: 32 cores ÷ 1.83 ms ≈ 17.5 k rows/s is the hard ceiling for the
  default `f8` layout on this host.
* The effective levers are therefore **per-row cost**, not write-path sharding:
  `hot_storage_layout=plain` (+17 %), `hot_ef_construction` 64→32 (+11 %) or 64→16 (+70 %).
* Under the production query load (`ivf_probes=256, search_candidates=8000`) inserts collapse:
  **−85 % at 8 query clients** (4 140 → 635 rows/s), because each such query reads ~4 GB of index.

---

## A. Validation

### A.1 Insertion-only QPS (rows/s), `hot_m=16`, `hot_ef_construction=64`, dim 128, HOT prefill 20 k

| clients | heap only | heap+btree | pgvector HNSW | **agentvec f8 (prod default)** | **agentvec plain** |
|---|---|---|---|---|---|
| 1 | 2 487 | 2 488 | 915 | **613** | 904 |
| 2 | 2 561 | 2 557 | 1 663 | **1 131** | 1 470 |
| 4 | 5 047 | 4 978 | 2 712 | **2 330** | 2 929 |
| 8 | 9 571 | 9 602 | 4 984 | **4 263** | 4 982 |
| 16 | 18 142 | 17 809 | 8 746 | **7 362** | 8 796 |
| 32 | 34 785 | 32 895 | 16 041 | **12 152** | 14 171 |

**Real 1B table** (`items_1b`, production reloptions, HOT segment filling from 200 rows):
**4 140 rows/s at 8 clients** (1.93 ms/row), 15.7 % CPU busy.

Storage-layout sweep at 32 clients: `plain` 14 378, `f8` (default) 12 251, `ieeefp16` 7 905 rows/s.
At one client: `plain` 888, `f8` 648, `ieeefp16` 373 rows/s.

**Conclusion.** The 4 k rows/s figure is real and reproducible, and `f8` — the default HOT
layout — costs ~17 % versus `plain`. agentvec at `plain` matches pgvector HNSW (4 982 vs 4 984
rows/s at 8 clients), so the agentvec segment wrapper is **not** the problem.

### A.2 CPU per row (the explanatory variable)

Sampled from `/proc/stat` around each pgbench window:

| target | µs CPU/row @1 | @8 | @32 | cores busy @32 |
|---|---|---|---|---|
| heap only | 72 | 60 | 74 | 2.6 / 32 (8 %) |
| agentvec `plain` | 951 | 937 | 1 349 | 19.1 / 32 (60 %) |
| agentvec `f8` | 1 326 | 1 231 | 1 832 | 22.2 / 32 (69 %) |
| pgvector HNSW | 1 832* | 957 | 1 229 | 19.7 / 32 (61 %) |

\* single-client sample is noisy.

The heap path costs ~70 µs/row; the HNSW insert costs **13–26× that**. CPU per row *grows* with
concurrency for agentvec (951 → 1 349 µs for `plain`), i.e. contention adds CPU burn, while the
heap path is flat.

### A.3 WAL per committed row

| target | WAL B/row | note |
|---|---|---|
| heap only | 697 | |
| heap + btree | 697 | |
| agentvec `f8` | 4 499 | +3.8 KB for the index |
| agentvec `plain` | 7 862 | larger codes, still faster |
| pgvector HNSW | 5 747 | |

agentvec emits **less** WAL per row than pgvector, so WAL *volume* is not the differentiator even
though `LWLock:WALWrite` is the top non-CPU wait (26 % of samples for agentvec, 36 % for pgvector
at 32 clients; `LWLock:BufferContent` only 2 %). Commits are group-commit serialized; more
concurrency amortizes it.

### A.4 Insertion + query combined (real 1B index)

Query workload = the production configuration (`ivf_probes=256, search_candidates=8000`), which
resolves to `ORDER BY embedding <-> q LIMIT 10` over `items_1b`.

Query-only (the query path saturates at ~3 q/s regardless of client count — each query scans
~128 M index entries ≈ 4 GB):

| query clients | q/s | latency |
|---|---|---|
| 1 | 1.94 | 514 ms |
| 2 | 2.95 | 677 ms |
| 4 | 3.19 | 1 255 ms |
| 8 | 3.11 | 2 574 ms |
| 16 | 2.90 | 5 511 ms |

Mixed — 8 insert clients + Q query clients:

| query clients | insert rows/s | Δ insert | insert lat | query q/s | query lat |
|---|---|---|---|---|---|
| 0 | 4 140 | — | 1.93 ms | — | — |
| 1 | 4 158 | +0.4 % | 1.93 ms | 1.86 | 537 ms |
| 2 | 3 135 | −24 % | 2.51 ms | 2.90 | 685 ms |
| 4 | 1 365 | −67 % | 5.62 ms | 2.88 | 1 379 ms |
| 8 | 635 | **−85 %** | 9.93 ms | 3.11 | 2 568 ms |

**Conclusion.** Queries are unaffected by inserts (1.86 vs 1.94 q/s), but inserts are starved by
queries from the second query client onward. CPU stays at 13–20 % throughout, so this is memory
bandwidth / buffer-pool eviction, not CPU. This is the dominant production risk for a
write-while-serving deployment — far larger than anything the write path can fix.

### A.5 HOT segment sizing

Insert throughput while one HOT segment fills (8 clients, 5 000-row batches):

| HOT rows | `f8` rows/s | `plain` rows/s |
|---|---|---|
| 0 → 5 k | 4 680 | 5 282 |
| 5 k → 15 k | 4 345 → 4 188 | 5 026 → 4 857 |
| 15 k → 45 k | 4 140 … 4 265 (flat) | 4 599 … 4 740 |
| 45 k → 50 k | 4 007 (−14 % vs empty) | 4 341 (−18 % vs empty) |
| after the seal (fresh segment) | 4 696 | 5 138 |

**Conclusion.** Filling a HOT segment to the 50 k default costs 13–18 % of insert throughput, and
the seal itself is free (the successor starts empty). `hot_segment_max_rows=50000` is well chosen:
going smaller buys ≤13 % at the price of more seals and more WARM segments; going much larger
loses throughput steadily.

---

## B. Where the time goes, and what the references say

### B.1 Profile of an insert run (perf, 8 clients, `f8`, 30 s)

| symbol | % |
|---|---|
| `Codec::distance_encoded_direct` (the SQ8 candidate distance) | **33.0** |
| `LWLockAttemptLock` | **14.0** |
| `PinBuffer` | 9.9 |
| `Codec::decode_into` | 8.0 |
| `LWLockRelease` | 7.6 |
| `computeRegionDelta` (WAL) | 2.2 |
| `UnpinBufferNoOwner` | 1.8 |
| `distance_l2_sq8_pairwise_x86_avx512` | 1.4 |
| `load_element_impl` | 1.3 |
| `select_neighbors` | 1.2 |
| `hash_search_with_hash_value` / `Visited::insert_key_hash` / `hash_bytes` | 2.7 |
| `pair_distance` | 0.9 |

So the insert is **~43 % quantization, ~21 % lock traffic, ~12 % buffer pinning**. The
agentvec-specific per-row work (meta page read, directory load, segment header update) does not
appear above 0.3 % — optimising it is not where the throughput is.

Confirmed by A/B: switching the SQ8 distance implementation between the integer pairwise form and
the scalar form changes throughput by only 3 % (4 192 vs 4 057 rows/s), i.e. the codec path is
already the fast one; the cost is the *amount* of distance work per row, i.e. `ef_construction`.

### B.2 Lock and WAL analysis

* The whole graph update runs under `LockPage(index, update_lock_page(base), ShareLock)` —
  **shared**, so concurrent inserts into one region do run in parallel (this matches pgvector
  exactly, `hnswinsert.c:703-740`). It escalates to `ExclusiveLock` only when the new element's
  level beats the entry point, i.e. O(m^-ΔL) ≈ 1/16 per row early and ≈1e-6 once large.
* The only per-row *exclusive* serialization inside a segment is the agentvec segment-header
  counter bump (one page rewrite + WAL record per row). It is real but small: the K-sweep below
  shows removing it from the contention set buys nothing.
* Wait events at 32 clients: `CPU:run` 68 %, `LWLock:WALWrite` 26 %, `IO:WalSync` 3 %,
  `LWLock:BufferContent` 2 %, `Lock:page` 0.3 %. **The bottleneck is not the graph's locks.**

### B.3 The decisive experiment: does sharding the write path help?

K independent hnswsq regions, emulated with K independent agentvec indexes each prefilled to
20 000 rows so every graph starts at the same size (aggregate rows/s):

| total clients | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| 8 | 4 271 | 4 072 | 4 198 | 4 204 |
| 16 | 7 432 | 7 455 | 7 587 | 7 653 |
| 32 | 12 509 | 12 306 | 12 548 | 11 822 |

**No gain (−2 % … +3 %).** The contention is in globally shared structures (buffer mapping
partitions, WAL insert/write locks, WAL flush), not in one graph's pages, so giving each writer
its own graph cannot help. Per-row CPU is the binding constraint.

This is consistent with the reference systems rather than contrary to them: hnswlib, USearch and
pgvector all have exactly one global lock per graph (the entry point), and everything else is
already page- or node-local; Faiss/USearch/Milvus/Lance shard because they must decouple *bulk
index construction* from *ingest visibility*, not because a single graph's inserts serialize.

### B.4 Measured levers (8 clients, `f8`, 15 s)

| change | rows/s | vs default |
|---|---|---|
| `hot_ef_construction=64` (default) | 4 257 | — |
| `hot_ef_construction=32` | 4 722 | **+11 %** |
| `hot_ef_construction=16` | 7 245 | **+70 %** |
| `hot_storage_layout=f8` (default) | 4 263 | — |
| `hot_storage_layout=plain` | 4 982 | **+17 %** |
| recall check (`plain` vs `f8`, 30 k rows, exact GT) | 1.0000 both | no recall cost |

`plain` is a strictly better HOT default: ~2.2× the bytes of `f8` for at most 50 k rows (measured
16 MB vs 7.4 MB on a 20 000-row dim-128 index, i.e. ~40 MB vs ~18 MB at the cap — noise against
a 32 GB index), +17 % insert throughput, no recall change, and it also makes consolidation
cheaper (training-free layouts decode inline instead of fetching exact heap vectors).

### B.5 Reference findings that mattered

* **Lance** keeps index segments immutable and unions "indexed" with a flat scan of unindexed
  fragments (`knn_combined` + `fragment_bitmap`). The deferral is of *work*, and the visibility
  boundary is a committed manifest — which is exactly what our constraint forbids as a primary
  path.
* **Milvus** never inserts into a sealed HNSW graph: raw rows land in a growing segment
  (brute-forced, or served by the incrementally-appended interim *IVF* index, `nlist=128`,
  `nprobe=16`, threshold `max(0.1·rows, nlist·39) ≈ 4 992`), and HNSW is built in bulk. Its
  growing-segment count is emergent (`sealProportion=0.12` of `segment.maxSize=1024 MB` ≈
  110–123 MB, plus a 4 GB per-shard aggregate cap that seals the *largest* growing segment), not
  a knob.
* **hnswlib / USearch / pgvector** converge on the same locking shape: node-local locks plus one
  global lock taken only on an entry-point promotion. Our port matches pgvector bit-for-bit.
* **SIMD is not the lever here.** The repo already tried AVX-512 tiers and reverted them for "NO
  GAIN"; pgvector's HNSW search is ~5 % distance arithmetic and ~40 % buffer manager. In our
  insert profile the *quantized* codec is 43 %, but the fix for that is the layout choice
  (`plain`), not a wider kernel.
* **BRIN and GIN (`fastupdate`) are the PostgreSQL-native precedents** for "defer the work, never
  the visibility": both keep the deferred structure inside the index, WAL-logged in the inserting
  transaction, and have the scan union it. Any future "append cheap, wire later" design for
  agentvec must be built that way — and the existing `FLAT` active-chain path is already an
  instance of it.

---

## C. The adaptive HOT pool

### C.1 What was implemented

`hot_segment_count` (reloption, default **1**, max 16). The appendable pool is **derived from the
published directory** — every entry with `level == Hot && state == Published` — so there is **no
new on-disk metadata and no format-version bump**: an index written by an earlier build simply has
one such segment.

* `insert.rs`: `resolve_hot_pool` + a backend-local cache keyed on the **published directory
  pointer** (every publication allocates a fresh chained item, so a changed pointer is an exact
  invalidation signal). Writers pick the least-loaded member of the pool, accounting for what the
  backend has already put into each since resolution; the per-insert path costs a pointer
  comparison on top of the meta read.
* `insert.rs`: `ensure_hot_pool` publishes the whole pool at build time (both the bulk IVF path
  and the small-dim HNSW path), so the steady state never takes the meta lock to create a segment.
  A seal always opens exactly one successor, so the pool size is invariant.
* `insert.rs`: **sealing now runs under a region barrier.** `seal_hot_and_open_new` takes the same
  page lock every insert holds in *share* mode for the duration of its graph update, but
  *exclusively*, before flipping the directory state. Lock order is region → meta → header, which
  is consistent with the plain append path (region → header) and with the documented meta →
  header order.
* **Known issue, found and NOT fixed here.** A writer that had already resolved a segment when it
  was sealed could still append after the collector walked the graph; the row would then be
  missing from the replacement *and* retired in the source. The seal barrier above closes the
  realistic window (no insert can be *in flight* when the segment is demoted), but the residual
  window between "pool resolved" and "region share lock acquired" remains. A belt-and-braces fix
  (quiesce the source region for the whole conversion, re-verify the source's `num_entries`
  before publishing, and abandon the conversion — leaving the source `Retiring` and searchable —
  when it changed) was written, but it aborted the backend in
  `pg_test_agentvec_consolidate_lifecycle_and_row_identity` and was **reverted rather than landed
  unvalidated**. It is the right shape for a follow-up; the crash needs to be understood first.
* `mod.rs`: `agentvec_index_info().is_hot` now means "accepts inserts right now" (every published
  HOT segment), not "is the single hot id".
* **No scan change was needed**: `scan.rs` already iterates every searchable segment and the
  router only deactivates owned IVF segments, so HOT segments were always all searched.

Adaptive mode is `hot_segment_count = 0`, which resolves to
`clamp(1 + consolidation_backlog, 1, 4)` — the backlog (HOT segments already sealed and awaiting
maintenance) is the in-index signal that the write side is outrunning maintenance, and it needs no
new persistent state. The default remains **1** because of §B.3.

### C.2 Verification (`bench/agentvec-insert/smoke_multi_hot.sh`, all green)

| check | result |
|---|---|
| pool publishes `hot_segment_count` appendable segments | 4 of 4 |
| inserts spread across the pool | 4 of 4 non-empty |
| nearest-5 exact order across regions | `1,2,3,4,5` |
| candidate coverage K=4 vs K=1 (same data) | 136 vs 34 |
| structural no-loss (`sum(num_entries)`) | 200 / 200, 500 / 500 |
| **same transaction: index scan sees the row** | yes |
| **concurrent backend before commit / after** | 0 / 1 |
| aborted transaction visible to a scan | 0 rows |
| seal keeps the pool at its configured size | 2 of 2, 18 sealed, 0 lost |
| adaptive mode (`hot_segment_count=0`) | 1 segment, 100/100 rows |

The Rust suite (`cargo pgrx test -p vectorscale pg17 agentvec --runas pgtest`) reports
**21 passed / 7 failed**. Both new tests pass:
`pg_test_agentvec_multi_hot_pool` and `pg_test_agentvec_multi_hot_seal_keeps_writers_going`.

The 7 failures are **one real failure plus 6 collateral**. The real one is
`pg_test_agentvec_consolidate_lifecycle_and_row_identity`: its backend dies with signal 6 and
takes the shared test instance down with it, after which the other six fail with
"connection closed" — they are not independently broken.

That failure is **pre-existing and unrelated to this change**, established by A/B rather than by
inference. Running *only that test*, built from this commit and then from its parent `d6491a6`,
fails identically both times (`test result: FAILED. 0 passed; 1 failed`).

It is also **specific to the `cargo pgrx test` harness**. The same sequence succeeds against a
normal cluster in every variant tried: autocommit; inside an explicit transaction (which is how
the harness wraps a test); with the harness's own GUCs (`maintenance_work_mem = '640MB'`,
`agentvec.maintenance_worker = off`); and invoked as a SQL function, the way the harness invokes
it. `agentvec_consolidate` also runs correctly and continuously in production — the maintenance
worker has repeatedly converted sealed HOT segments into WARM IVF-RaBitQ segments on the 1B
index. The abort is therefore a harness/environment problem rather than a product defect
reachable from SQL, worth chasing because it keeps the suite red and hides five other tests.

> Note for whoever picks this up: `cargo pgrx test` needs `-p vectorscale`, `RUST_TEST_THREADS=1`,
> and `target/test-pgdata` pre-created and owned by the `--runas` user, otherwise every test fails
> with a misleading "Could not obtain test mutex". A hard-aborted run also leaves that directory
> poisoned — delete it before re-running. To reproduce the A/B, keep a copy of the parent's
> `pgvectorscale/src` and swap it in (the server keeps one at `/data1/avwork/src_parent`).

### C.3 Measured effect on the real insert path (new build, prefill 20 k, `f8`)

| config | 8 clients | 32 clients |
|---|---|---|
| K=1 (new build) | 4 325 | 12 269 |
| K=4 (new build) | 4 346 | 12 019 |
| K=1 (previous build, for reference) | 4 263 | 12 152 |

`items_1b` with the new build: 4 101 rows/s at 8 clients (previous build 4 140) — **no regression**.
The existing 1B index reads correctly and the maintenance worker consolidated the sealed segments
produced by the benchmark (segments 2/3/5/6 → retired, 4/8/9/10 → WARM).

**So multi-hot buys ~0 % insert throughput (as §B.3 predicted), and is not the answer to the
reported bottleneck.** It is still worth keeping for its actual benefits, which are structural
rather than throughput:

1. **Scan-side coverage.** Each HOT region contributes its own candidate set, so the pool's
   effective candidate budget is K × (one region's candidates). Measured on the same 200 rows:
   **K=1 → 34 candidates, K=4 → 136.** More, smaller graphs cover *more* of the data at the same
   `ef`.
2. **Independent seal schedules**, so one consolidation cannot stall all writers.
3. **A bound on per-graph size** as `hot_segment_max_rows` is raised.
4. Adaptive growth only when a consolidation backlog actually exists.

### C.4 How to choose the numbers

* `hot_segment_max_rows`: keep **50 000** (per-segment). Fill-curve says 13–18 % cost from empty to
  full, and the seal is free.
* `hot_segment_count`: keep **1** on hardware like this. Use `2–4` when (a) a consolidation
  backlog is persistent, (b) `hot_segment_max_rows` must be large for another reason, or (c) the
  scan-side coverage argument in C.3.1 matters more than the K× scan work. `0` (auto) does this
  from the backlog without hand-tuning.
* `hot_storage_layout`: **`plain`** — now the shipped default.  This applies to HOT segments
  created *from now on*: an existing index keeps the layout of its current HOT segment until
  that segment fills, seals, and is replaced.  On the 1B index the change therefore takes effect
  over the next few seals, not immediately. +17 % insert throughput,
  recall-neutral, and it costs ~40 MB instead of ~18 MB at the 50 k-row cap (measured on a
  20 000-row dim-128 index: 16 MB vs 7.4 MB). Verified end to end after the change: a plain
  index created with no `WITH` clause gives 4 979 rows/s against 4 408 for an explicit `f8`
  one at 8 clients.
* `hot_ef_construction`: the strongest single lever (64→32 = +11 %, 64→16 = +70 %); it is a
  recall/throughput trade confined to the HOT layer.
* For a write-while-serving deployment, **cap the query cost** (`ivf_probes`,
  `search_candidates`): that is what costs 85 % of insert throughput in §A.4.

---

## Files

Benchmark harness (new): `bench/agentvec-insert/` — `00_setup.sql`, `env.sh`, `gen_scripts.sh`,
`bench_insert.sh`, `bench_insert_fixed.sh`, `bench_insert_cpu.sh`, `bench_insert_waits.sh`,
`bench_query.sh`, `bench_mixed.sh`, `bench_multi.sh`, `setup_multi.sh`, `fill_curve.sh`,
`wal_per_row.sh`, `cpu_per_row.sh`, `cpu_sample.sh`, `waits.sh`, `recall_check.sh`, `reset.sh`,
`prefill.sh`, `build.sh`, `smoke_multi_hot.sh`.

Implementation (modified): `pgvectorscale/src/access_method/agentvec/insert.rs`,
`options.rs`, `build.rs`, `mod.rs`, `consolidate.rs`, `tests.rs`.

Deployment on x86: source `/data1/avwork`, build log `/data1/avwork/build_full.log`, installed to
`/root/pg17-release/lib/postgresql/vectorscale-0.9.0.so`. The previous production library is kept
at `/data1/avbench/so/libvectorscale.prod-before.so` and the pristine baseline build at
`/data1/avbench/so/libvectorscale.baseline.so`; restoring is
`install -m 755 /data1/avbench/so/libvectorscale.prod-before.so /root/pg17-release/lib/postgresql/vectorscale-0.9.0.so`.
