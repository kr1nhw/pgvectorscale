# hnswsq — the HNSW index access method with reduced-precision storage

`hnswsq` is this repository's HNSW (hierarchical navigable small world) index
over pgvector `vector` columns, with **four node-vector storage layouts**:

| `storage_layout` | format | bytes/dim | size vs plain | training |
|---|---|---|---|---|
| `plain` | IEEE f32 verbatim | 4 | 1× | none |
| `ieeefp16` (alias `f16`) | IEEE 754 binary16 (half) | 2 | ~1/2× | **none — stateless cast** |
| `ieeefp8` | OCP FP8 **E4M3** | 1 | ~1/4× | **none — stateless cast** |
| `f8` (alias `sq8`) | Lance-style SQ8: per-dimension min/max linear | 1 | ~1/4× | calibrated at CREATE INDEX |

The IEEE layouts are **training-free**: no calibration artifact, no range
drift, identical behavior on bulk builds and on empty-start/incremental
indexes (CREATE INDEX on an empty table, rows appended over time).  `f8`/SQ8
freezes a per-dimension `[min, max]` range from a reservoir sample at build
time; inserts outside the range clamp (accuracy loss only, never a
correctness issue), and an empty-start SQ8 index gets a provisional `[-1, 1]`
range until `REINDEX` retrains from real data.

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
  `plain` ≤ ~1900, `ieeefp16` ≤ ~3900, `ieeefp8`/`f8` ≤ ~7900 at `m = 16`
  (the exact bound is computed from `m` and reported in the error message).

### Options (WITH clause)

| option | range | default | meaning |
|---|---|---|---|
| `storage_layout` | `plain`, `ieeefp16`, `ieeefp8`, `f8` | `plain` | node-vector precision (aliases `f16`, `sq8`) |
| `m` | 4 – 100 | 16 | max neighbors per node per upper layer; layer 0 gets `2·m` |
| `ef_construction` | 4 – 1000 | 64 | search width during build and insert |
| `sample_size` | 0 – 1000000 | 0 (= auto, 30000) | vectors reservoir-sampled for SQ8 calibration only |

## 3. Querying

The index serves `ORDER BY embedding <op> query LIMIT k` plans.  The useful
GUC:

| GUC | range | default | meaning |
|---|---|---|---|
| `hnswsq.ef_search` | 1 – 1000 | 40 | layer-0 search width — the main recall/speed dial; must be ≥ the query LIMIT for full recall |

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

- **`plain`** — exact stored distances, 4 bytes/dim.  The baseline.
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

Size note: neighbor pointers are shared across layouts (each node carries
`2·m + m·level` ItemPointers), so the 2×/4× ratios hold for the vector bytes;
total index size shrinks a bit less for small `dim`.

## 5. Concurrency & MVCC

- **Visibility** is the executor's snapshot check: the index returns heap
  TIDs (`xs_recheckorderby = true`), the executor fetches each heap tuple
  (MVCC) and recomputes the exact distance with the operator, restoring exact
  ordering over the candidates the graph search produced.  Uncommitted /
  aborted / dead rows are therefore invisible, and aborted inserts leave at
  most an orphan index node that vacuum cleans up.
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
- No `amgetbitmap`; no iterative scan for WHERE-filtered queries (a pgvector
  0.8 feature) — `ef_search` bounds the candidate set.
- Builds are single-backend (the in-memory phase is sequential; parallel
  build is future work).
- `ieeefp8`'s accuracy assumes in-range data (±448); out-of-range components
  clamp (cosine-normalized data is unaffected).
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

The full hnswsq test suite (53 tests: recall matrix across the four storage
layouts and three distance types, incremental builds, transaction rollback,
planner behavior, dimension limits, NULLs, REINDEX, vacuum lifecycle, and
page-packing extremes) runs with:

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

## 10. Build performance notes

The in-memory build path has two optimizations that keep 1M-row builds
practical on serverless-class boxes:

- **Decode-free distance kernels** (`Codec::distance_encoded_direct`):
  distances are accumulated directly over the encoded bytes (f32/f16/fp8/sq8)
  with no per-call scratch decode copy; semantics match the SIMD kernels
  exactly (L2 is squared-sum, IP is negative — `<#>` ordering — and cosine
  is `1 − Σq·v` clamped at zero).
- **Build-scoped pair-distance cache**: backlink re-pruning recomputed every
  neighbor-list pair distance on each revision; the cache turns that into one
  computation per unordered pair (bounded at ~4M entries ≈ 250MB, cleared on
  overflow), roughly halving build time.

Builds are still single-backend and slower per node than pgvector's parallel
C build; see `.design/neon/bench/RESULTS-HNSWSQ.md` for the 100K A/B numbers
(recall parity within ±1%, size parity within +4%, query latency ~10-13x
higher p50 — the cost of the page-based, buffer-managed design that buys
MVCC-safe online mutation, WAL safety, and vacuumability).
