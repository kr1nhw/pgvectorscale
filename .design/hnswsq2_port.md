# hnswsq2 — port of pgvector's HNSW core (Phase 0 recon)

Reference: pgvector v0.8 vendored at `.design/reference/pgvector/`
(`hnsw.h` 16k, `hnswbuild.c` 33k, `hnswinsert.c` 20k, `hnswutils.c` 33k,
`hnswscan.c` 8k, `hnswvacuum.c` 22k — 5 122 lines total).

## What pgvector does, in one paragraph

The graph is a **linked list of elements** (`HnswElementData`), each holding its
heaptid, level, a vector datum and a neighbor array (a "V2" buffer of
`(distance, heaptid, element)` triples).  The same struct addresses an element in
**three places**: private memory, a shared memory region, or an index page —
via `HnswPtr`, a PG `relptr` union (`ptr` when `base == NULL`, `relptr_off`
otherwise).  The build inserts into private/shared elements protected by **one
LWLock per element**; the finished index stores element/neighbor *tuples* on
pages protected by **ordinary buffer locks**.  The heuristic is classic HNSW
(Paper-1 distance + occlusion diversity), pruned incrementally in-place.

## Decisive answers from the source

1. **On-disk elements are locked with `LockBuffer`, not per-element LWLocks.**
   `hnswutils.c`/`hnswinsert.c` take `BUFFER_LOCK_SHARE` to read a page and
   `BUFFER_LOCK_EXCLUSIVE` to modify it (lines 186/306/382/542/770 and
   30/79/185/485/596 respectively); the per-element `LWLock`
   (`hnswutils.c:740`) is acquired only for **in-memory elements**.
   Consequence for the port: my `NodeLocks`/tranche machinery is needed for the
   build's in-memory phase only — the scan path needs none of it.

2. **`HnswPtr` is a relptr.**  `HnswPtrAccess(base, hp)` is
   `base == NULL ? hp.ptr : relptr_access(base, hp.relptr)` and
   `HnswPtrStore` is the inverse.  PG's `relptr_store`/`relptr_access` are
   `static inline` and therefore absent from pgrx bindings, so the port
   implements them as trivial offset arithmetic in one small unsafe module:
   `access(base, rp) = (Type*)((char*)base + rp.relptr_off)`,
   `store(base, rp, p) = (rp.relptr_off = (char*)p - (char*)base)`.

3. **The graph is built twice-over in `hnswbuild.c`.**  `InsertTuple` →
   `InsertTupleInMemory` (private/shared elements, per-element locks) until the
   build ends, then `WriteNeighborTuples`/`FlushPages` move it onto pages and
   `CreateMetaPage` records the entry.  The parallel build
   (`HnswParallelBuildMain(dsm_segment*, shm_toc*)`) scans a share of the heap
   into the *same* in-memory structure in a shared region, with a
   `HnswLeader` handshake (entryLock + entryWaitLock for the entry point).

4. **The scan is a state machine over pages** (`hnswscan.c`), resumable via
   `HnswScanOpaqueData` — pgvector's iterative-scan machinery lives here.

5. **Vacuum** (`hnswvacuum.c`) marks deleted elements, repairs their neighbors'
   lists (`RepairGraph`), and fixes the entry point if it was deleted — the
   model hnswsq2 adopts wholesale.

## Port surface

| pgvector file | Rust module | contents |
|---|---|---|
| `hnsw.h` | `types.rs` | metapage, page opaque, element/neighbor tuples, `HnswPtr`, element data, candidate heaps, options, graph |
| `hnswutils.c` | `utils.rs` | search layer, neighbor selection heuristic, connection update/prune, entry handling, buffers/pages |
| `hnswinsert.c` | `insert.rs` | in-memory → on-disk insert, page management, duplicate handling, tuple versioning |
| `hnswbuild.c` | `build.rs` | ambuild, in-memory build, shared region, per-element locks, worker main, flush + metapage |
| `hnswscan.c` | `scan.rs` | beginscan/gettuple/rescan/endscan, ef_search |
| `hnswvacuum.c` | `vacuum.rs` | bulkdelete, repair, cleanup |

Deliberate divergences from pgvector, and only these:

* the element tuple gains a **layout byte + encoded element width**, so
  `plain`/`f16`/`fp8`/`sq8` use the existing `Codec`/quantization;
* the candidate heaps may use Rust `BinaryHeap`s with an identical comparator
  rather than porting `pairingheap_container` macros (no observable
  difference in the nearest-first expansion, verified in the recall gate);
* `relptr_store`/`relptr_access` re-implemented in one `ptr.rs` module.

Not carried over from hnswsq: the append-only/segment machinery, the arena
build, the fingerprint/determinism gate (pgvector levels come from
`HnswGetRandomLevel`, entropy-seeded — hnswsq2's gates are recall and structure).

## Rust representation decisions

* `repr(C)` for everything that crosses shared memory or disk
  (metapage, page opaque, element/neighbor tuple data).
* In-memory elements: `repr(C)` too, because they cross the shared region.
* All pointer arithmetic confined to `ptr.rs` + the page helpers; every offset
  gets a unit test (this is where a port rots).

## What M3 work is reused, verified already

Worker entry signature `(dsm_segment*, shm_toc*)`, the versioned library name,
toc keying + `BuildParams`-style parameters, the shared scan descriptor, the
worker-failure flag + per-worker stats reporting, and the `ivf`/`ivfrq`
two-access-methods pattern.

## Gates (per plan)

1. types/pages: encode/decode + layout-offset unit tests; create/drop empty index.
2. single-writer insert: recall parity vs pgvector on a small dataset (same
   data/params), index size within a few percent.
3. parallel build: cross-process test; host-121 A/B vs pgvector (103.3 s) and
   vs the old engine (179.5 / 374.7 s), 1/2/4/8 workers.
4. scan+vacuum: query correctness; delete → vacuum → recall; hnswsq
   transactional tests re-pointed at hnswsq2.
5. retirement: hnswsq2 renamed to hnswsq, old engine deleted, full suite green,
   host-121 numbers re-run.
