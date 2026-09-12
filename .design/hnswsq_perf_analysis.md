# hnswsq performance analysis (vs pgvector hnsw, vs Lance)

Goal: understand why hnswsq builds and queries are slower than the
alternatives, and what to change.  Numbers below are on vanilla PostgreSQL
17.11 (pgrx-managed builds) and lancedb 0.37.1, on the same cloud boxes.

## Measured results

### 100K BIGANN (113.44.106.182, 16 vCPU / 60GB, shared-CPU VM)

| engine | build | size | recall@10 (ef/nprobes 10→640) | p50 ms (ef 10→640) |
|--------|-------|------|-------------------------------|--------------------|
| pgvector hnsw (m16/efc64) | 13 s | 832 B/vec | 85.2% → 100% | 0.31 → 3.72 |
| hnswsq plain (m16/efc64) | 3556 s | 861 B/vec | 84.2% → 100% | 3.79 → 68.0 |
| Lance IVF_HNSW_SQ p16/sq8 | 1.6 s | 788 B/vec | 58.4% → 86.2% (np 1→32) | 1.45 → 1.50 |
| Lance IVF_SQ p64/sq8 | 0.3 s | 1171 B/vec | 58.2% → 98.4% (np 1→32) | 1.45 → 1.72 |

### 1M BIGANN (113 for Lance; 121.37.117.106 (32 vCPU) for pgvector)

| engine | build | size | recall@10 | p50 ms |
|--------|-------|------|-----------|--------|
| pgvector hnsw (m16/efc64) | 101 s | 832 B/vec | 78.9% → 100% (ef 10→640) | 0.70 → 7.03 |
| hnswsq plain (m16/efc64) | ~97 min (interrupted at 48 min; ~170 nodes/s) | — | — | — |
| Lance IVF_HNSW_SQ p16/sq8 | 19.6 s | 823 B/vec | 60.6% → 84.6% (np 1→32) | 1.56 → 1.53 |
| Lance IVF_HNSW_SQ p64/sq8 | 15.4 s | 1113 B/vec | 51.6% → 87.1% | 2.74 → 1.68 |
| Lance IVF_SQ p64/sq8 | 0.85 s | 1244 B/vec | 61.5% → 97.9% | 1.77 → 4.59 |

Incremental insert (100K fresh rows into the existing index): pgvector
~2.5-4 ms/row on 113; hnswsq TBD (page-based backlink updates); Lance does
no per-row index maintenance at all (append-only fragments; bulk
compaction/optimize instead).

## Phase-0 instrumented baseline (`hnswsq.build_stats`)

Counters added to the build path (GUC `hnswsq.build_stats = on`), 3000 nodes,
dim 16, m=16, ef_construction=64, plain layout, local macOS/pg18:

```
nodes=3000 backlink_lists=99104 pair_lookups=52196960 pair_misses=98862
search=794.9ms select=343.8ms backlink_pairs=1003.7ms backlink_select=10615.0ms
flush=22.5ms accounted_total=12779.9ms rows_total=3000 disk_mode=false
```

Reading: the pair-distance cache is working (98 862 misses for 52 196 960
lookups = 0.19%), so distance *computation* is no longer the cost.  The cost
is the number of pair *comparisons* the backlink re-prune performs —
**83% of build time is `select_neighbors_heuristic` runs on backlink lists**,
~527 occlusion checks per list, 33 lists per inserted node.

This is an algorithmic constant, not an allocation or kernel problem: each
backlink list is re-derived from scratch even though the existing list is
already the output of the same heuristic, so all of its internal occlusion
relations are already known.

## Phase-1 result: exact incremental backlink prune

`backlink_prune_mem` replaces the per-backlink re-run of the full heuristic.
The existing list is itself heuristic output, so an incoming link can only
*add* occlusion relations: entries before the new node keep their status, and
entries after it need exactly one new check (against the new node, if it was
accepted).  A per-list mask records which entries the occlusion rule actually
accepted, so entries that were only re-added by the closest-pruned backfill
are re-checked instead of trusted.  The result is bit-identical to the full
heuristic — asserted by `test_backlink_prune_fuzz` (2000 randomized
function-level cases), `test_backlink_prune_matches_full_heuristic` and
`test_backlink_incremental_matches_every_insert` (whole graphs, lists +
distances + masks compared after every insert).

Same 3000-node/dim-16 measurement as above, after the change:

| phase | before | after |
|---|---|---|
| pair-distance lookups | 52 196 960 | 1 739 703 (30x fewer) |
| `backlink_select` | 10 615 ms | 297 ms (36x) |
| `backlink_pairs` | 1 004 ms | 232 ms (4.3x) |
| `search` | 795 ms | 753 ms |
| `select` (own lists) | 344 ms | 336 ms |
| `flush` | 22 ms | 22 ms |
| **accounted total** | **12 780 ms** | **1 639 ms (7.8x)** |

The build is now dominated by the actual graph searches (46% search, 20%
neighbor selection for the new node's own lists) rather than by list
maintenance.

### The split changes with scale

100k rows, dim 128, plain, 32-vCPU box (121.37.117.106), `build_stats` on:

```
nodes=100000 backlink_lists=3306955 pair_lookups=185980982 pair_misses=66123417
search=630758.6ms select=153311.9ms backlink_pairs=12857.4ms backlink_select=496008.6ms
flush=3398.1ms accounted_total=1296334.7ms
```

Total 1 296 s (21.6 min for 100k on that box), split: **search 49 %,
backlink_select 38 %, select 12 %, backlink_pairs 1 %, flush 0.3 %**.  Two
things follow:

* the pair-distance cache thrashes at this size (66M misses / 186M lookups =
  36 % miss) because `PAIR_CACHE_MAX` clears it repeatedly; entry lists are
  also much longer relative to the cache window;
* the backlink path is still material, because entries whose mask bit is clear
  (never occlusion-evaluated, or re-added by a backfill) take the full re-check
  path, and the "extras" that those accept add further checks.

So the next levers, in order, are (a) the search path itself (allocation and
hash churn per visited node — 49 % and growing with dataset size), (b) the
backlink entries that still take the full re-check, and (c) parallelism, which
is the only route to the single-digit-minute 1M builds the Lance reference
shows (IVF_HNSW_SQ: 19.6 s for 1M) without giving up the page-based,
transactional design.  In-memory build benchmarks (600-vector clustered graphs,
`cargo test mem_build`): 91 s → 42 s (pair cache) → **34 s** (incremental
prune) on the same machine.

## Phase-0/A instrumented baseline, 100k rows dim 16 (local, 2GB budget)

The per-caller counters split the remaining cost unambiguously:

```
nodes=100000 accounted_total=58362ms rows_total=100000
search=26654.6ms (46%)  select=12277.2ms (21%)
backlink_select=10144.9ms (17%)  backlink_pairs=8430.3ms (14%)  flush=855.6ms
fast_entries=104037120  full_entries=0  extras_seen=0
pair(select)  = 50533540 lookups /     849 misses   (99.998% hit)
pair(backlink)=  3304313 lookups / 3284915 misses   (99.4% miss)
search_calls=106575  search_hits=3510129
```

Reading, and what each number implies:

* **`select` is pure lookup cost.**  50.5 M pair-cache lookups with essentially
  no misses: the own-list heuristic's O(candidates × selected) occlusion checks
  (~500 per list) each pay a hash probe for a distance that is already known.
  The fix is not a better cache but *not* using one: decode the candidate set
  once into a contiguous scratch and run SIMD distances (Lance's storage-backed
  `DistCalculator` does exactly this).
* **`backlink_pairs` is write-once traffic.**  3.3 M lookups, 99.4% misses: these
  are the `(target, new_node)` distances, one per backlink list, never queried
  again.  They should be computed decode-free (`distance_encoded_direct`)
  without touching or polluting the cache.
* **`backlink_select` is dominated by re-sorting**, not by checks:
  `fast_entries = 104 037 120` over 3.3 M lists means every backlink re-walks and
  re-sorts the whole ~33-entry list, and `full_entries = 0` shows the
  mask/extras machinery now costs nothing.  Lance's `cutoff` admission avoids
  most of this work entirely by skipping edges that cannot enter the list.
* `extras_seen = 0` at this scale — the extras path is a rare-case guard only.

## Phase-C result: memory beam search (Lance's VisitedGenerator + borrowed access)

The memory build's search called the generic `search_layer`, which clones the
encoded vector *and* the neighbour list per visited node, hashes into a
`HashSet` + `HashMap` cache per search, and allocates fresh heaps each call.

`search_layer_mem` replaces that for the memory graph (which has no tombstones
and no vanished ids) with: epoch-stamped visited/expanded mark arrays (no
hashing, O(visited) reset — Lance's `VisitedGenerator` pattern), borrowed
`&[u8]`/`&[u32]` access, and heaps reused across the layered searches of one
insert.  `test_search_layer_mem_matches_generic` asserts it returns exactly the
same hits, in the same order, as the generic path.

100k dim 16, same machine, same dataset:

| phase | before | after |
|---|---|---|
| `search` | 26 655 ms | **5 216 ms (5.1x)** |
| `select` | 12 277 ms | 11 542 ms |
| `backlink_select` | 10 145 ms | 9 887 ms |
| `backlink_pairs` | 8 430 ms | 8 210 ms |
| flush | 856 ms | 907 ms |
| **accounted total** | **58 362 ms** | **35 762 ms (1.63x)** |

The build is no longer search-bound: the remaining cost is the own-list
heuristic's pair-cache lookups (`select`, 50.5 M lookups for 853 misses — pure
hashing, to be replaced by a decoded candidate buffer + SIMD distances) and the
write-once backlink distances (`backlink_pairs`, 3.3 M lookups with 99.4 %
misses — to be computed decode-free without touching the cache).

## Phase-C/D result: decode-once distance buffer replaces the pair cache

The own-list selection and the backlink re-prune drove every pair distance
through a build-scoped `HashMap<(u32, u32), f32>`.  At 100k rows that table had
grown to ~100MB, so each probe was a random access into main memory: 3.3M
`(target, new_node)` distances with a 99.4% miss rate cost 8.2s, and the own-list
selection's 50.5M probes — 99.998% of them *hits* — cost 11.5s.  Both loops only
ever need the vectors of a small set of ids (one candidate set, or one neighbour
list plus the incoming node), so `DistBuf` decodes each of those vectors exactly
once into a flat `f32` buffer and evaluates pairs straight through the SIMD
`dist_fn`.  No hashing, no per-pair allocation, working set in L1/L2.

`DistBuf::dist` returns `dist_fn` over the same two decoded vectors the map
stored, so the change is bit-identical, and the counters prove it (same
`pair_dists` as the old `pair_lookups` to the digit, same fast/full entry counts,
same recall at every ef).

### Dataset change (and a methodology fix)

The scratch database holding the previous 100k/dim-16 dataset was dropped, so the
old `35.8 s` figure is not comparable to these runs.  The dataset is now scripted
and reproducible (`.design/neon/bench/local_dataset.sql`: 100k rows, dim 16, 200
clusters x 500 rows, 200 queries + exact top-10), the harness pins
`hnswsq.build_seed = 20240912` (an unpinned build re-seeds from entropy, which
moved recall@10 by up to 0.30 between runs), and `.design/neon/bench/local_cycle.sh`
records build time + stats + recall sweep in one CSV row.

### 100k, dim 16, m16/efc64, pinned seed — release profile

Release is the profile that matters (pgvector and Lance are C/Rust release
builds; a debug build of this extension is ~12x slower, which is what every
earlier local number in this file was):

| phase | before | after | |
|---|---|---|---|
| `search` | 7 659.7 ms | 6 074.7 ms | 1.26x |
| `select` (own lists) | 3 954.0 ms | **320.5 ms** | **12.3x** |
| `backlink_select` | 23 101.4 ms | **4 956.3 ms** | **4.66x** |
| `backlink_pairs` | 264.7 ms | 73.5 ms | 3.6x |
| flush | 359.8 ms | 379.9 ms | |
| **accounted total** | **35 339.5 ms** | **11 804.9 ms** | **2.99x** |
| **wall (CREATE INDEX)** | **36.28 s** | **12.48 s** | **2.91x** |

Recall@10 is identical before/after (ef 10/40/160/640 = 0.50 / 0.80 / 1.00 /
1.00) and so is every counter: `pair_dists = 233 686 708` (was
`pair_lookups`), `fast_entries = 60 853 299`, `full_entries = 43 205 713`,
`search_hits = 6 824 040`.

Same change, debug profile (kept for the record — it is what the test suite
runs): 439.3 s -> 321.5 s wall, `select` 61.4 s -> 18.6 s, `backlink_select`
242.4 s -> 170.7 s.

### Scaling: 100k -> 300k -> 1M (release, local 12-core Mac, pinned seed)

Same clustered dataset generator at each size, `m=16`, `ef_construction=64`,
`maintenance_work_mem=2GB`, single-threaded build:

| rows | wall | `search` | `backlink_select` | `select` | flush | pair distances |
|---|---|---|---|---|---|---|
| 100k | 12.48 s | 6.07 s (51%) | 4.96 s (42%) | 0.32 s | 0.38 s | 234 M |
| 300k | 45.44 s | 25.43 s (60%) | 15.22 s (36%) | 0.97 s | 0.77 s | 694 M |
| **1M** | **182.59 s** | **110.02 s (65%)** | **51.20 s (30%)** | 3.32 s | 3.50 s | 2 291 M |

Recall@10 at ef 160/640 is 1.000 at every size (ef 10/40 = 0.500/0.508 — this
clustered dataset needs a large ef at `m=16`, which is a property of the data,
not of the build).  For reference, the same 1M build in the *debug* profile
(none of the pair-distance work done) was interrupted after 97 min, so the
release profile alone is a ~30x methodology correction on top of the algorithmic
work.

### Ranked admission re-measured with cheap distances (Phase-B revisit)

The Phase-B verdict (Lance's ranked/cutoff list loses) was measured while a pair
lookup cost ~2.5us.  With distances at SIMD cost the cost model changed, so it
was re-run on the same 100k/pinned-seed/release setup:

| | exact (default) | ranked (cutoff) |
|---|---|---|
| wall | **12.48 s** | 13.56 s |
| pair distances | 234 M | 664 M |
| edges skipped by cutoff | 0 | 471 000 |
| admits / prunes | - | 2 835 872 / 2 834 936 |
| recall@10 (ef 10/40/160/640) | 0.500 / **0.800** / 1.000 / 1.000 | 0.500 / **0.508** / 1.000 / 1.000 |

Still slower *and* worse recall, so the default stays `Exact`.  The reason is
unchanged: the cutoff test is cheap, but every admitted edge then pays a full
neighbor-selection heuristic over the merged list (664M pair distances for 2.8M
admits), whereas the exact incremental re-prune resolves an admitted edge with
an O(len) walk over state it already has.

### What is left at 100k (release)

`search` 6.07 s (51%) and `backlink_select` 4.96 s (42%) now account for
essentially the whole build; everything else is 0.8 s.  Two consequences for the
next phases:

* the expensive part of `backlink_select` is no longer distance *lookup* but the
  43.2M entries that cannot take the O(1) fast path (mask bit clear: never
  occlusion-evaluated, or re-added by the closest-pruned backfill) and are
  re-checked against the accepted prefix — 198.8M pair distances in total;
* the Phase-B verdict on Lance's ranked/cutoff admission was measured while a
  pair lookup cost ~2.5us, i.e. under a cost model that no longer holds; it has
  to be re-measured with distances at SIMD cost before the default stays exact.

## Next-round plan, driven by the measurements above

Three measured facts set the order:

1. `search` is the largest phase at every size (51% at 100k, 60% at 300k, 65% at
   1M) and grows faster than the row count.  At 1M it is 103us per insert for 64
   hits (1.6us per hit), which is ~10x the cost of the distance kernels
   themselves: the phase is memory-latency bound.  Every visited node touches
   `vectors[id]` (its own heap allocation) and `neighbors[id][layer]` (two levels
   of `Vec` indirection).
2. Backlink admission is 30-42% and is dominated by the entries that cannot take
   the O(1) fast path (357M of them at 1M): a candidate whose mask bit is clear
   (never occlusion-evaluated, or re-added by the closest-pruned backfill) is
   re-checked against the whole accepted prefix.
3. Nothing in the two phases above is inherently sequential, but the build is
   single-threaded; that is the gap to Lance (19.6s for a 1M IVF_HNSW_SQ build).

Next steps, in benefit order:

**(a) Contiguous memory-graph layout** (Lance's flat adjacency +
`prefetch_distance`): one vector arena with `stride = dim x elem_bytes`, and flat
per-`(node, layer)` neighbour/distance/mask slabs.  Visiting a node then touches
one cache line instead of chasing allocations.  Expected 1.3-2x on `search`
(15-30% of total build time).  No semantic change, no on-disk format change; the
existing equivalence tests keep it honest.

**(b) Parallel build (Phase E) — attempted, measured, rejected; needs the
per-node-lock design.**  The cheap variant was implemented and does not work:

*Design tried.*  Rows are buffered on the main thread (the row stream is a
PostgreSQL table scan and has to stay there), each batch is *planned* by N
worker threads over a read-only graph, then the plans are *applied* one after
another on the main thread (so every mutation still happens in one order, and no
locking is needed anywhere).  Levels are drawn on the main thread, so node ids
and heap TIDs stay deterministic.

*Why it fails.*  A node planned inside a batch cannot see its batch-mates'
edges, so all of them attach to the same pre-batch nodes; those layer-0 lists
are already full (`m0` = 32), the excess is pruned away, and whole groups of
nodes end up with **no incoming edge at all** — disconnected components, which
is exactly what recall measures.  Clustered 1000-node harness, `m=16`, `m0=32`,
same seed (`test_mem_build_parallel_recall_floor` asserted the shipped width,
the other rows come from the diagnostics test that was used to find it):

| batch (rows invisible to each other) | threads | nodes with no incoming edge | recall@10 |
|---|---|---|---|
| 1 (sequential) | 1 | 0 | 0.9550 |
| 2 | 2 | 11 | 0.9400 |
| 32 | 4 | 201 | 0.7950 |
| 64 | 4 | 402 | 0.6250 |
| 256 | 8 | 452 | 0.5800 |

The loss starts at a batch width of 2 and grows roughly linearly with the width,
while throughput needs a width of at least the thread count — so the two-phase
design has no useful operating point.  It was reverted rather than shipped
behind a default-off GUC.

*What a working parallel build needs*: search-time visibility of in-flight
inserts, i.e. the Lance model (`Arc<RwLock<GraphBuilderNode>>`, edges published
as each node is linked, backlinks taken with a per-node write lock, searches
with read locks) so that a node being inserted concurrently is already reachable
while its peers search.  That is a structural change to `MemGraph` (per-node
locks or shared-memory pages) and has to be validated with the same
connectivity/recall checks used above.

*And why not PostgreSQL parallel workers* (the mechanism the diskann path in this
repo already uses): those workers share state through DSM, and the hnswsq memory
graph is plain process-local Rust data (`Vec`/`Box`), not shared-memory pages, so
it would have to be rewritten as a shared-memory arena with swizzled offsets.

**(c) Occluder memo for clear-mask backlink entries**: store the id of the entry
that occluded each rejected candidate (a build-only side table, like
`list_masks`), so a clear-mask entry that is still occluded by an entry that is
still accepted resolves in O(1) instead of a re-check against the whole accepted
prefix.  Attacks most of `backlink_select`; the "an accepted entry never loses
acceptance" argument is exactly what the full-heuristic equivalence tests
(`test_backlink_prune_*`) can validate.

**(d) SQ integer kernels for `f8`** (Lance's `dot_u8`/`l2_u8` with a pre-folded
query and closed-form bias) for the scan-side distance cost of the quantized
layouts.

## Phase-B result: Lance's ranked/cutoff admission is not a win here

`lance-index` admits a backlink edge only if it beats the target's current worst
neighbour (`GraphBuilderNode::cutoff`) and prunes that list only when it
overflows.  Implemented behind `hnswsq.build_backlink_mode` (0 = ranked,
1 = exact; **exact stays the default**) and measured on the same 100k/dim-16
dataset and on the clustered memory harness:

| | exact (default) | ranked (cutoff) |
|---|---|---|
| build accounted total, 100k | **58.2 s** | 60.9 s |
| backlink lists skipped | 0 | 3 199 342 (97 %) |
| pair lookups in own-list select | 50.5 M / 855 misses | 100.8 M / 0 misses |
| backlink_select, 1k-node harness | 294 ms | 508 ms |
| recall@10 (same build seed, clustered) | 1.000 | 0.970-0.995 |

Why it loses: the cutoff test is genuinely cheap and skips 97 % of backlink
lists, but every *admitted* edge pays a full neighbor-selection heuristic over
the merged list (that is what the doubling of select-path lookups is), while the
exact incremental re-prune resolves an admitted edge with an O(len) walk over
state it already has.  A second finding: the cutoff rule alone starves nodes in
already-saturated dense clusters of *incoming* edges (recall fell to 0.63-0.76);
admitting the new node's nearest backlink unconditionally restores it to
0.97-0.995, still short of exact.

Conclusion: keep the ranked mode as an experiment knob (`build_backlink_mode = 0`)
guarded by `test_backlink_ranked_mode_recall_floor`, and keep the exact mode as
the default.  The Lance items that *did* carry over are the ranked-list
representation (lists are now kept globally sorted, which is what makes a
`cutoff`-style check meaningful at all) and the mode switch itself for A/B work.

## Where the time goes (profiling)

### Build

Instrumented experiments on the in-memory graph (24K inserts, local):

| variant | time |
|---------|------|
| full build (m=16, efc=64) | 91 s |
| heuristic → plain top-cap (self lists) | 88 s |
| backlinks disabled entirely | **0.62 s** |

- **Backlink maintenance is >99% of build time.**  Each insert touches ~32
  backlink lists; each list re-prune recomputes ~33 pairwise distances and
  re-decodes ~33 vectors, i.e. ~1K distance computations + ~1K decode/copies
  per inserted node — repeated on every list revision (each pair is
  recomputed ~10-20x over the build).
- Landed fixes: decode-free distance kernels (`distance_encoded_direct`)
  and a build-scoped pair-distance cache (bounded ~250MB) → 91 s → 43 s on
  the same experiment; full pgrx suite 264 s → 124 s.  Still: ~1K hash
  lookups + cache misses per insert, single-threaded, scalar-loop
  accumulation in the direct kernels (no SIMD over encoded bytes yet).
- The build is **single-backend**; pgvector builds with parallel workers and
  Lance partitions the problem (16-64 centroids → tiny HNSW graphs).

### Query (scan)

- p50 is ~10-13x pgvector at equal recall.  Each search hop: shared-buffer
  lookup + page pin + rkyv node deserialize + per-neighbor decode into
  scratch.  pgvector keeps the whole graph memory-mapped; a hop is a few
  pointer chases.  Lance's hop set is tiny (a graph over 16-64 centroids)
  and the scan itself is a SIMD pass over contiguous SQ8 codes — ~1.5 ms
  total.
- VisitData clones the neighbor list + encoded vector per visited node;
  distance kernels re-decode per neighbor (direct kernels removed the
  scratch copy but still run scalar per-element loops).

### Insert

- One aminsert: search (page hops) + one exclusive node-page write + ~32
  backlink two-phase updates, each a page pin + validate + GenericXLog +
  WAL.  This is inherent to the transactional page-based design (MVCC-safe
  online mutation, crash-safe) — pgvector rebuilds its in-memory graph and
  Lance appends immutable fragments with no per-row transactional index
  maintenance at all.

## Lance reference architecture (what to borrow)

- Columnar, memory-mapped storage: vector batches are Arrow buffers read
  with zero copy; no per-row deserialization.
- `IVF_HNSW_SQ`: HNSW graph over only the *partition centroids* (tiny), each
  partition's vectors SQ8-compressed; queries do a few graph hops then SIMD
  scans of SQ8 codes.  Builds of 1M rows take 15-20 s in Rust; queries are
  ~1.5 ms p50 at 85% recall, 2-5 ms at 97%+ (IVF_SQ, more probes).
- Immutable data files + versioned manifests: inserts are appends with no
  per-row index mutation; compaction/optimize rebuilds indexes in bulk.
- SIMD distance kernels over contiguous, aligned buffers.

## Optimization directions for hnswsq (ordered by impact/effort)

> Superseded by "Next-round plan, driven by the measurements above": the phases
> below were the plan written *before* the instrumented runs, and the measured
> splits moved several of them (the pair cache became a decode-once buffer, the
> ranked/cutoff admission was measured negative twice, and the parallel build
> was attempted and rejected).  Kept for the record.

1. **Build: prune backlinks once per node, not once per insert.**
   Accumulate the incoming-link set per node during the build and run the
   re-prune heuristic once when a node's list actually changes (or at flush
   time).  Removes the dominant 10-20x recomputation (the profiling table
   above shows the entire build collapses to ~1% of its time without the
   backlink path).
2. **Build: cache decoded vectors in the MemGraph** (one decode per node per
   build, not per heuristic run).  Partially mitigated by the pair cache;
   a flat `Vec<Vec<f32>>` keyed by id removes the HashMap churn.
3. **SIMD batch kernels**: distance over an encoded 128-float row vs a small
   candidate batch (e.g. 32 rows) in one AVX2/NEON pass, including the
   fp16/fp8/sq8 decode-on-the-fly forms — the Lance-SQ8-style lever.
   (Direct kernels exist; they are scalar per-element today.)
4. **Query: in-memory node cache.**  A bounded shared-memory cache of
   (vector bytes + neighbor lists) for hot nodes removes the pin+deserialize
   per hop — the biggest p50 multiplier — while keeping page-based
   durability.  Alternative: pack nodes so a whole frontier expansion is one
   page read, and prefetch.
5. **Quantized layouts for the scan path.**  ieeefp16/ieeefp8/f8 cut the
   page bytes and decode work per hop; combined with SIMD decoding this is
   the Lance-SQ8-style lever for serverless (CPU-bound) queries.
6. **Insert: amortize backlink maintenance.**  (a) Coalesce all backlink
   updates of one insert into one WAL record per page (full-page images
   already do this per page); (b) a write-ahead "pending links" delta merged
   lazily by vacuum/rebuild — trades exact-neighbor freshness for insert
   throughput, the trade Lance makes wholesale.
7. **Build parallelism** (longer term): partition-sharded build like Lance
   (build HNSW over sampled centroids, then per-shard graphs) or pgvector's
   parallel-worker build with subgraph merge; the current single-backend
   build caps throughput at one core.

## Expected outcome of 1+2+3+4

- Build: ~5-10x faster (1M from ~100 min toward ~10-20 min on weak VMs;
  proportional on real hardware; a sharded/parallel build gets to the
  Lance 15-20 s class only with 7).
- Query p50: ~3-5x faster with a hot-node cache + SIMD kernels (toward
  ~2-3x pgvector; still page-based).
- Insert: 2-3x with coalesced backlink WAL; order-of-magnitude only with
  lazy link maintenance (6).
