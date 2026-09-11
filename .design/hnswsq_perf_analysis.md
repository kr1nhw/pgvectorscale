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
maintenance.  In-memory build benchmarks (600-vector clustered graphs,
`cargo test mem_build`): 91 s → 42 s (pair cache) → **34 s** (incremental
prune) on the same machine.

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
