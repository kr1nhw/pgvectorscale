# hnswsq concurrent graph — design & implementation notes

Status: **implemented and validated on local PG 18.4 / arm64** (this branch).
On-disk format version 1; the AM is registered as `hnswsq` (pgvector already
owns the `hnsw` name and both extensions coexist).

## Goal

An HNSW index with reduced-precision node storage (`plain` / `ieeefp16` /
`ieeefp8` / `f8`-SQ8) that matches the operational bar of the IVF-RaBitQ work:
append-only node storage, WAL-logged transactional mutations, MVCC via the
executor, autovacuum support, and **pgvector-style concurrent inserts** (no
global writer lock) — all reads through the buffer manager (no `smgr` tricks,
Neon-compatible).

## On-disk layout

```
block 0  : HnswMeta           single-item page; locked update() RMW
             { magic "HNSQ", version, extension_version, distance_type,
               num_dimensions, precision (plain|ieeefp16|ieeefp8|f8),
               m, m0=2m, ef_construction, ml=1/ln(m), max_level,
               entry_point (ItemPointer), entry_level,
               node_count, deleted_count,
               insert_page (append hint), free_pages_head,
               calibration (chain-item ptr; SQ8 only) }
dynamic  : HnswCalibration    chained item (SQ8 per-dim min/max + provisional flag)
dynamic  : HnswNode pages     multi-item pages; one rkyv item per node
dynamic  : HnswFreePages      threaded free list: each freed page stores the
                              next free block as its only item
```

Node item (constant serialized size per node — the `PlainNode` padding trick):

```text
heap_tid: ItemPointer   level: u8   deleted: u8
vector: dim × elem_bytes (4/2/1/1)
neighbors: (level+1) lists; layer 0 padded to m0, upper layers to m,
           padding = Invalid ItemPointers, valid entries form a prefix
```

- Node identity = `ItemPointer`; nodes are **never moved** — inserts append,
  vacuum tombstones in place (deleted flag + invalidated heap TID) and frees
  only fully-dead pages.
- Neighbor lists are mutated in place under one exclusive page lock
  (GenericXLog WAL); the node payload is written once.
- `max_level` is the largest level whose worst-case node fits a page item
  (probe-serialized at CREATE INDEX); the random level is clamped to it.  A
  level-0 node that doesn't fit = dimension-limit error (layout-specific
  caps).

## Invariants

1. **Single-content-lock rule**: a backend holds at most ONE buffer content
   lock at a time and never acquires any other lock (content, extension,
   meta, advisory) while holding an exclusive content lock.  Reads are
   snapshot-and-release (`load_node_view` copies the item out).  Buffer
   content locks are not deadlock-detected by PostgreSQL — this rule is what
   makes concurrent inserts hang-free; the hnswsq paths take NO advisory
   locks at all.
2. **Two-phase optimistic neighbor updates**: every list mutation snapshots
   the target under a share lock (phase 1), computes the new list with no
   locks held, then re-validates identity (`heap_tid`, `level`, `deleted`)
   and list equality under one exclusive lock (phase 2), retrying on a race
   (bounded), falling back to append-if-room / remove-only.
3. **Crash safety via transaction atomicity**: the node, its lists, and its
   backlinks are all written before the inserting transaction commits, so a
   crash rolls the heap row back with them.  An orphaned index node points at
   a dead TID and is tombstoned by vacuum's LINEAR page walk.
4. **Stale pointers are safe**: `load_node_view` validates page type, offset
   range, and line-pointer flags; a freed-and-reused page resolves to "gone"
   or to a valid live node (approximate-search semantics, identical to
   pgvector's deleted-page reuse).

## Insert (pgvector-style, no global writer lock)

1. Draw level (clamped to `max_level`); encode the vector (SQ8 loads the
   calibration chain; IEEE layouts are stateless casts).
2. Allocate a node slot: `insert_page` hint → free-list pop → relation
   extension (the only heavyweight lock, taken alone); write the node with
   capacity-padded empty lists; publish the hint via meta RMW when rotated.
3. Empty index: claim the entry point under the meta RMW (loser of a
   first-insert race re-reads and links into the winner's graph, so no
   committed row is ever left unreachable).
4. Greedy ef=1 descent from the entry point to `level+1`; `search_layer` with
   `ef_construction` at each layer `min(level, entry_level)..0`; per-layer
   `select_neighbors_heuristic` (occlusion rule + closest-pruned backfill).
5. Fill the node's own lists (one exclusive lock, one commit).
6. Backlinks: two-phase per selected neighbor (invariant 2).
7. Promote the entry point iff `level > entry_level` (meta RMW re-checks the
   level under the lock; concurrent promoters serialize, highest wins).

## Search / scan

- Layered search, layer 0 with `hnswsq.ef_search`; the result heap holds
  **LIVE nodes only** — tombstones route the search but never consume an
  `ef` slot, so a delete-heavy index still returns `ef_search` live rows
  (bounded by the connected component).
- Distances are computed on the stored (decoded) vectors.  `amgettuple`
  emits `xs_orderbyvals` as **provable lower bounds of the exact operator
  value, in the operator's units** — the contract `nodeIndexscan.c` enforces
  (it errors with "index returned tuples in wrong order" when the index value
  exceeds the recomputed one).  Bounds per layout:
  - L2: `sqrt(max(0, d − 2√d·e − e² − slack))`,
  - cosine: `max(0, d − ‖q‖·e − slack)`,
  - inner product: `d − ‖q‖·e − slack`,
  where `e = rel_err·margin·‖v̂‖` for the IEEE layouts (fp16: 2^-11·1.01,
  fp8: 2^-4·1.15) or `e = ‖scales‖/2` for SQ8, and `slack` covers SIMD
  accumulation differences against the executor's recomputation.  `plain`
  emits the exact value minus `slack`.
- **The orderbyval datum must be a float8** — pgvector's distance operators
  return `float8`; raw f32 bits are reinterpreted as a garbage double
  (harmless-looking for non-negative values, catastrophic for negative inner
  products).  `xs_recheckorderby = true` for all layouts: the executor's
  reorder queue restores exact ordering from the heap recheck.

## Vacuum (autovacuum-ready)

`ambulkdelete` (linear walk over node pages — reachability-independent, so
crash orphans are cleaned):

1. **Mark**: peek each page under a share lock; items whose heap TID the
   callback reports dead are tombstoned under ONE exclusive lock (GenericXLog
   WAL).  No cleanup locks: tombstones keep their items (routing stays
   intact) and every read path re-validates locations atomically.
2. **Repair**: for each fresh tombstone, its live neighbors lose the
   reference (remove-only fallback after racing retries) and gain the
   tombstone's other live neighbors as candidates (heuristic-pruned,
   connectivity splice).  Tombstones keep routing until their page is freed.
3. **Entry fix**: a tombstoned entry is replaced by the highest-level live
   node seen during the walk (promote-only; never demotes a concurrently
   promoted higher entry — the meta RMW compares against the entry vacuum
   inspected).
4. **Free**: fully-dead pages go onto the threaded free list.  `push_free_pages`
   RE-VERIFIES each page under its exclusive lock inside the meta RMW (a
   concurrent insert with a stale `insert_page` hint may have revived it).
   After repair no live list references a freed page; stale scan pointers
   resolve through `load_node_view`'s guards.

`amvacuumcleanup` refreshes `num_index_tuples`/`num_pages` (the latter is what
updates `pg_class.relpages` — forgetting it makes the planner think the index
shrank to zero).

## Concurrency validation

- Local pg_test suite (54 tests, PG 18.4 arm64): recall matrix across 4
  layouts × 3 distance types, empty-start incremental lifecycle, transaction
  rollback, planner behavior, dimension limits, page-packing extremes, a
  5-build dual-recall probe (pure disk-search membership vs executor recall
  on the same build), and delete→vacuum→reload lifecycle with page-reuse
  (`relpages`) checks for every layout.  The hnswsq integration tests are
  serialized by a suite mutex: parallel pg_tests hold long transactions that
  block VACUUM's dead-tuple removal (a test-harness artifact, not an index
  defect).
- Python concurrency tests (`tests/test_hnswsq_concurrent.py`): concurrent
  INSERTs (plain + ieeefp8), mixed INSERT/SELECT/DELETE/VACUUM with
  probe-based "no lost rows" verification.

## Known limitations

- Tombstones occupy space and routing until vacuum frees their pages; very
  heavy delete ratios between vacuums increase search cost (the live-only ef
  heap bounds result count, not traversal).
- `ieeefp8` out-of-range values clamp (±448) — the lower-bound proofs and
  accuracy assume in-range data.
- No iterative scan for filtered queries; single column only.
