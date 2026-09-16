# hnswsq — the HNSW index access method with reduced-precision storage

`hnswsq` is this repository's HNSW (hierarchical navigable small world) index
over pgvector `vector` columns, with **six node-vector storage layouts**. The
graph core is a **function-for-function Rust port of pgvector's HNSW**
(build, insert, scan, vacuum, relptr addressing, the same on-disk tuple
format); this repository's contribution is the reduced-precision layouts and
the scan contract that goes with them:

| `storage_layout` | format | bytes/dim | size vs plain | training |
|---|---|---|---|---|
| `plain` | IEEE f32 verbatim | 4 | 1× | none |
| `ieeefp16` (alias `f16`) | IEEE 754 binary16 (half) | 2 | ~1/2× | **none — stateless cast** |
| `ieeefp8` | OCP FP8 **E4M3** | 1 | ~1/4× | **none — stateless cast** |
| `f8` | Lance-style SQ8: per-dimension min/max linear | 1 | ~1/4× | calibrated at CREATE INDEX |
| `sq8` | fixed-range int8: `round(clamp(x, ±1) · 127)` | 1 | ~1/4× | **none — global fixed range** |
| `sq16` | fixed-range int16: `round(clamp(x, ±1) · 32767)` | 2 | ~1/2× | **none — global fixed range** |

The IEEE layouts and the fixed-range `sq8`/`sq16` are **training-free**: no
calibration artifact, no range drift, identical behavior on bulk builds and on
empty-start/incremental indexes (CREATE INDEX on an empty table, rows appended
over time).  `f8`/SQ8 freezes a per-dimension `[min, max]` range from a
reservoir sample at build time; inserts outside the range clamp (accuracy loss
only, never a correctness issue), and an empty-start SQ8 index gets a
provisional `[-1, 1]` range until `REINDEX` retrains from real data.

`sq8`/`sq16` quantize against a **global, fixed `[-1, 1]` range** — the natural
range of cosine-normalized embeddings — so their distances are computable
**directly from the integer codes**: `Σ (q̂ − code)²` is the decoded-domain L2
distance of the quantized query times one global constant (the stored side is
exact; the query carries only the standard half-step error).  The graph
itself is always constructed with the exact decoded distance (graph mutation
never uses the query-quantized form: on clustered data the half-step noise
rivals intra-cluster spacing and fragments the neighbor graph), while scans
use the fast integer-code form under `hnswsq.sq8_distance = pairwise`.

The index is built to the same operational bar as the `ivf` (IVF-RaBitQ) work:
append-only node storage, WAL-logged transactional mutations, executor-driven
MVCC visibility, autovacuum support, and pgvector-style concurrent inserts
(no global writer lock).  All reads go through the buffer manager — no `smgr`
tricks — so it works unchanged on vanilla PostgreSQL and Neon.

## 1. Setup

```sql
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector provides the vector type
CREATE EXTENSION IF NOT EXISTS vectorscale; -- registers the hnswsq access method
```

## 2. Creating an index

```sql
CREATE INDEX items_idx ON items
  USING hnswsq (embedding vector_l2_ops)
  WITH (storage_layout = ieeefp16, m = 16, ef_construction = 64);
```

Notes:

- The operator class must be named explicitly: `vector_l2_ops` (**default**
  for the AM, `<->`), `vector_cosine_ops` (`<=>`), `vector_ip_ops` (`<#>`).
  Cosine vectors are normalized at insert/query time; inner product is
  supported on all layouts.
- The build is **in-memory** while the estimated graph footprint fits
  `maintenance_work_mem` (each node ≈ `dim × elem_bytes` + `2·m + m` neighbor
  ids), then spills to a single sequential writeout; if the budget is
  exhausted mid-build, the remaining rows flow through the regular disk
  insert path (bounded memory).  `CREATE INDEX CONCURRENTLY` always uses the
  disk insert path.
- The indexed column must have a fixed dimension (e.g. `vector(128)`).
  Dimension limits per layout: a node must fit one page item — roughly
  `plain` ≤ ~1900, `ieeefp16`/`sq16` ≤ ~3900, `ieeefp8`/`f8`/`sq8` ≤ ~7900
  at `m = 16` (the exact bound is computed from `m` and reported in the error
  message).

### Options (WITH clause)

| option | range | default | meaning |
|---|---|---|---|
| `storage_layout` | `plain`, `ieeefp16`, `ieeefp8`, `f8`, `sq8`, `sq16` | `plain` | node-vector precision (alias `f16`) |
| `m` | 4 – 100 | 16 | max neighbors per node per upper layer; layer 0 gets `2·m` |
| `ef_construction` | 4 – 1000 | 64 | search width during build and insert |
| `sample_size` | 0 – 1000000 | 0 (= auto, 30000) | vectors reservoir-sampled for `f8` calibration only |

## 3. Querying

The index serves `ORDER BY embedding <op> query LIMIT k` plans.  The useful
GUC:

| GUC | range | default | meaning |
|---|---|---|---|
| `hnswsq.ef_search` | 1 – 1000 | 40 | layer-0 search width — the main recall/speed dial; must be ≥ the query LIMIT for full recall |
| `hnswsq.iterative_scan` | `off` / `relaxed` / `strict` | `relaxed` | pgvector-style iterative scan: a full search up front (`off`), or emit from the first batch and resume from discarded candidates on demand (`relaxed`; `strict` additionally forces non-decreasing distances) |
| `hnswsq.max_scan_tuples` | −1 – INT_MAX | −1 (unbounded) | cap on tuples one scan may visit before falling back to the discarded heap; −1 disables |
| `hnswsq.build_seed` | −1 – INT_MAX | −1 (entropy) | seeds the level RNG; pin it to make two builds' graphs (and timings) comparable |
| `hnswsq.sq8_distance` | `scalar` / `pairwise` | `pairwise` | scan-time distance form for the SQ layouts (`f8`, `sq8`, `sq16`): decode-then-compare, or the integer code-pairwise form.  Graph construction always uses the exact decoded distance. |

```sql
SET enable_seqscan = off;   -- force the index on small tables
SET hnswsq.ef_search = 100;

SELECT id FROM items
ORDER BY embedding <-> '[0.1, 0.2, ...]'::vector
LIMIT 10;

EXPLAIN (ANALYZE, BUFFERS) SELECT id FROM items
ORDER BY embedding <-> '[0.1, 0.2, ...]'::vector LIMIT 10;
```

The index is only used for ORDER BY queries: `amcostestimate` refuses paths
without orderby keys (it would undercount plain scans and could expose stale
TIDs to index-only fetches), so `count(*)`-style queries fall back to a
sequential scan.

## 4. Choosing a storage layout

- **`plain`** — exact stored distances, 4 bytes/dim.  The baseline, and the
  only layout whose scan needs no executor recheck (see §5).
- **`ieeefp16`** — ~2× smaller; per-element relative error ≤ 2^-11, which is
  effectively lossless for ANN ranking.  The recommended default for
  incremental workloads.
- **`ieeefp8`** — ~4× smaller; FP8 E4M3 has a 3-bit mantissa (≤ 2^-4 relative
  error) and a ±448 range: values outside the range clamp on encode, which
  can cost recall on unnormalized data.  Great for cosine-normalized
  embeddings and memory-bound (Neon-style) deployments.
- **`f8`** (SQ8) — ~4× smaller; per-dimension min/max linear quantization
  (Lance-style) gives the best 1-byte accuracy for in-range data, but it is
  the only layout that needs calibration: build it over representative data,
  or accept the provisional `[-1, 1]` range on empty-start indexes until
  REINDEX.
- **`sq8`** — ~4× smaller; fixed-range int8 (`round(clamp(x, ±1) · 127)`,
  one global step).  No calibration at all — the training-free 1-byte layout.
  Distances are computed directly from the codes (`Σ (q̂ − code)²`, pure
  integer arithmetic), and the stored side is exact: the fixed global range
  means the code-pairwise form reproduces the decoded-domain ranking with no
  per-dimension weights.  Components outside ±1 clamp on encode (the vector
  loses its provable lower bound, never its validity), so it is aimed at
  normalized data — cosine-normalized embeddings land in ±1 naturally.
- **`sq16`** — ~2× smaller; the same fixed-range design at int16 precision
  (~2^-15 steps): near-plain accuracy with half the bytes and training-free.
  For normalized data this is effectively lossless for ANN ranking.

Size note: neighbor pointers are shared across layouts (each node carries
`2·m + m·level` ItemPointers), so the 2×/4× ratios hold for the vector bytes;
total index size shrinks a bit less for small `dim`.

## 5. Concurrency & MVCC

- **Visibility** is the executor's snapshot check: the index returns heap
  TIDs, the executor fetches each heap tuple (MVCC) and drops invisible rows.
  Ordering depends on the storage layout: for the lossless `plain` layout the
  index emits the operator's own distance and PostgreSQL trusts that order
  (`xs_recheckorderby = false`, exactly as pgvector's hnsw does for `vector`
  columns); for the reduced-precision layouts (`ieeefp16`, `ieeefp8`, `f8`,
  `sq8`, `sq16`) the emitted value is a provable **lower bound**,
  `xs_recheckorderby = true`, and
  the executor recomputes the exact operator value per tuple and restores exact
  ordering over the candidates the graph search produced.  Uncommitted /
  aborted / dead rows are therefore invisible either way, and aborted inserts
  leave at most an orphan index node that vacuum cleans up.
- **Inserts are concurrent** (pgvector-style): no global writer lock.  Node
  slots are appended under per-page exclusive locks; neighbor lists are
  updated with a two-phase optimistic protocol (snapshot under a share lock,
  validate-and-write under one exclusive lock, retry, then append-if-room);
  the entry point is promoted under the meta page's exclusive lock.  At most
  ONE buffer content lock is ever held at a time, which is what makes the
  protocol deadlock-free (PostgreSQL buffer content locks are not
  deadlock-detected).
- **Deletes** are visible immediately (heap rows), and vacuum tombstones the
  matching nodes; between delete and vacuum the tombstones keep routing the
  search (deleted rows stop appearing in results as soon as they are dead in
  the heap).

## 6. Vacuum / autovacuum

`ambulkdelete` walks the node pages linearly (so even crash-orphaned
unreachable nodes are cleaned), tombstones dead nodes in place, repairs the
graph (dead references removed from live lists, their live neighbors spliced
in to preserve connectivity), fixes a tombstoned entry point, and puts
fully-dead pages on a free list for insert reuse — repeated
delete→vacuum→reload cycles hold the index size stable.  Autovacuum works
out of the box (standard callbacks, no reloption tuning required).

## 7. Limitations

- Single `vector` column only (no label filtering, no multi-column indexes).
- No `amgetbitmap`.  The iterative scan (§3) bounds memory like pgvector's,
  with the same tuple/memory caps.
- Builds parallelize through PostgreSQL's **standard** parallel index-build
  machinery (`plan_create_index_workers`, capped by
  `max_parallel_maintenance_workers`) — no custom GUC, see §11.
- The parallel path has no mid-build spill, so it needs
  `maintenance_work_mem` to hold the whole graph: when the budget runs out
  mid-build, the pages built so far are flushed to disk and the remaining
  rows flow through the regular disk insert path (NOTICE logged, build takes
  significantly longer — the same fallback pgvector has).
- `ieeefp8`'s accuracy assumes in-range data (±448); out-of-range components
  clamp (cosine-normalized data is unaffected).  `sq8`/`sq16` assume the
  tighter ±1 range for the same reason: components outside it clamp on
  encode, which is fine for normalized data and costs recall elsewhere.
- As with any approximate index, crash windows are transactional: a crash
  rolls back the inserting transaction, so committed rows are never lost;
  an orphaned node (dead TID) is removed by the next vacuum.

## 8. Tuning: recall vs latency

- Raise `hnswsq.ef_search` for more recall (must be ≥ LIMIT).
- Raise `m`/`ef_construction` at CREATE INDEX for a better graph (bigger
  index, slower inserts).
- `hnswsq.ef_search` only bounds the number of LIVE candidates returned:
  tombstone-heavy indexes between vacuums still return up to `ef_search`
  live rows.

## 9. Testing

The full hnswsq test suite (54 tests: types/pages layout, recall matrix across
the four storage layouts and three distance types, incremental builds,
transaction rollback, planner behavior, dimension limits, NULLs, REINDEX, GUCs,
SQ8 clamp behavior, size ratios, one-node-per-page packing, vacuum lifecycle,
and the cross-process parallel-build scaffold) runs with:

```bash
cd pgvectorscale && RUST_TEST_THREADS=1 cargo pgrx test pg18 hnswsq
```

Two environment notes:

- `RUST_TEST_THREADS=1` is required (pgrx's test binary initializes only
  from the libtest main thread, and the vacuum scaffolds need their deleted
  rows to be globally dead).
- On Linux the unit-test binary references PostgreSQL backend symbols from
  `#[pg_test]` bodies that `--gc-sections` cannot discard (Rust does not emit
  per-function sections).  `test_stubs.c` + `build.rs` + the
  `#[cfg(all(test, target_os = "linux"))] #[link]` block in `src/lib.rs`
  link weak stubs into test binaries only; the extension shared library
  itself never sees them.  Run the suite as a non-root user
  (`cargo pgrx test pg17 --runas <user>`; initdb refuses root).

The multi-backend concurrency stress (concurrent inserts, mixed
INSERT/SELECT/DELETE/VACUUM, quantized layouts) is in
`tests/test_hnswsq_concurrent.py`; run it against a scratch instance with
`CREATE EXTENSION vector; CREATE EXTENSION vectorscale;` and
`DB_HOST/DB_PORT/DB_USER/DB_NAME` pointing at it.

## 10. Build performance

The port is a faithful translation of pgvector's build (`.design/hnswsq2_port.md`
lists the sanctioned divergences — deterministic `BinaryHeap` candidates with a
packed-TID tie-break instead of pairing heaps, a reimplemented relptr, codec-based
distances, a caller-owned scratch instead of reset contexts, and the SQ8
calibration chain written before the parallel phase). The retired engine's
bespoke in-memory machinery (`search_layer_mem`, `DistBuf`, the incremental
backlink re-prune) is gone with it.

Measured builds, **release builds**, `m=16`, `ef_construction=64`:

| dataset | box | hnswsq | pgvector | ratio |
|---|---|---|---|---|
| 100k × 128 i.i.d., mwm 2GB | local (Apple M4 Pro, PG 18) | **6.9 s** | 7.5 s | **0.92x** |
| 1M BIGANN dim 128, mwm 8GB, 4 workers | 121.37.117.106 (32 vCPU, PG 17.11) | **59 s** | 102 s | **0.58x** |
| 1M BIGANN, single backend | 121.37.117.106 | – | 446 s | – |
| 1M BIGANN, 32 workers | 121.37.117.106 | – | 64 s | – |

Index size on the 1M table: hnswsq **819 B/vec** vs pgvector **832 B/vec**
(−1.6%).  Query sweep on the same table (mean ms, LIMIT 10, 100 queries):
0.696 / 1.151 / 2.445 / 6.609 at ef 10/40/160/640 vs pgvector
0.618 / 1.115 / 2.491 / 6.982 — **0.95–0.98x at ef ≥ 160**.  Recall@10:
99.3% vs 99.2% at ef 160, 100.0% for both at ef 640.  Full numbers and both
engines' `perf` profiles (both buffer-manager-bound: PinBuffer /
`load_element_impl` / LWLockRelease dominate) in
`.design/neon/bench/RESULTS-HNSWSQ.md`; the local-methodology A/B and the
storage-layout size/latency table are in `.design/hnswsq2_perf_local.md`.

**Always benchmark a release build**: `cargo pgrx test` installs a debug build
of the extension over the release one, which makes every build roughly 25x
slower and silently invalidated a whole analysis pass (the bench scripts now
refuse to run against a debug `.so`).

## 11. Parallel builds

Builds parallelize through PostgreSQL's standard parallel `CREATE INDEX`
machinery — pgvector's shared-arena design, ported: workers scan disjoint
block ranges into one `shm_toc`-allocated graph arena, each insert searches
the shared graph under per-element LWLocks (a searching node sees in-flight
inserts), the entry point is handed over with pgvector's entry-lock/wait-lock
protocol, and the leader joins the scan itself before flushing the graph.

```sql
SET maintenance_work_mem = '8GB';   -- must hold the whole graph: no mid-build spill
-- workers come from the server's standard pool:
SHOW max_parallel_maintenance_workers;

CREATE INDEX ON items USING hnswsq (embedding vector_l2_ops)
    WITH (storage_layout = 'plain', m = 16, ef_construction = 64);
```

Measured on the 1M BIGANN box (32 vCPU, release, same server settings for both
engines):

| engine | workers | wall |
|---|---|---|
| hnswsq | 4 (server default) | **59 s** |
| pgvector | 4 (server default) | 102 s |
| pgvector | 1 | 446 s |
| pgvector | 32 | 64 s |

The cross-process path is exercised by the test suite (parallel-build scaffold)
and was A/B-verified against the single-builder path during the port; recall
does not degrade with the worker count.

Constraints worth knowing:

- Not used for `CREATE INDEX CONCURRENTLY`: live inserters would race the bulk
  writeout, exactly as on the single-builder path, so concurrent builds stay on
  the disk path.
- No mid-build spill (see §7): size `maintenance_work_mem` for the whole graph,
  or let the build fall back to the disk insert path.
- A worker that fails, or an arena that fills, fails the build rather than
  writing a partial graph.
- "Workers launched" is capped by `max_parallel_maintenance_workers` and the
  standard worker pool — read the timing with the count the build reports.
