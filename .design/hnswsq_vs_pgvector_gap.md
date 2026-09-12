# hnswsq vs pgvector-hnsw: where the remaining gap comes from

Measured on 121.37.117.106 (32 vCPU, PG 17.11), same table (`items_1m`, BIGANN
1M rows, dim 128), same index parameters (`m=16, ef_construction=64`,
`maintenance_work_mem=8GB`), **both engines release builds**, same session, with
`hnswsq.build_stats` / `perf` / `EXPLAIN (ANALYZE, BUFFERS)` instrumentation.

Reproduce: `.design/neon/bench/gap_study.sh` (builds + sweeps + profiles),
`gap_sweep.py` (per-query sweep), `gap_perf.sh` (flat profile of one scan).

## 1. Headline numbers

| | hnswsq | pgvector hnsw | ratio |
|---|---|---|---|
| build, 1 backend | **562 s** | **442 s** | 1.27x |
| build, default parallelism (4 workers) | – (single-backend) | **103 s** | 5.5x |
| build, 32 workers requested (7 effective) | – | **64 s** | 8.8x |
| index size | 861,880,320 B (862 B/vec) | 832,143,360 B (832 B/vec) | 1.036x |
| recall@10 @ ef 160 | 99.3 % | 98.9 % | – |
| query mean ms, LIMIT 10, ef 10/40/160/640 | 1.01 / 1.94 / 4.71 / 13.58 | 0.65 / 1.15 / 2.53 / 7.08 | 1.56x / 1.69x / 1.86x / 1.92x |
| query mean ms, full candidate drain, same ef | 0.91 / 2.02 / 5.19 / 15.13 | 0.72 / 1.24 / 2.90 / 8.04 | 1.26x / 1.63x / 1.79x / 1.88x |

Two things fall out immediately:

* **The build gap is parallelism, not algorithm.** Per core we are 1.27x slower
  (562 s vs 442 s), i.e. 79 % of pgvector's single-core throughput, and our graph
  is slightly better (99.3 % vs 98.9 % recall at ef 160) with an exact backlink
  re-prune. pgvector's 4-worker build is 4.29x faster than its own single-core
  build (442 -> 103 s) and ~7 workers give 6.9x (-> 64 s). The visible 5.5-8.8x
  is that parallelism, which hnswsq does not have yet.
* **The query gap is per-candidate cost, not fixed overhead.** Fitting the sweep:
  marginal cost per extra unit of `ef_search` is 19.9 us (hnswsq) vs 10.2 us
  (pgvector) = **1.95x**, while the fixed part is 0.81 ms vs 0.55 ms = 1.49x.
  Whatever we fix has to be per visited/emitted candidate.

## 2. Where a query spends its time (perf, ef=160, 20 s of samples)

Flat profile of the backend running `ORDER BY embedding <-> q LIMIT 10`:

| symbol (work) | hnswsq | pgvector |
|---|---|---|
| `PinBuffer` (buffer pin) | 13.2 % | 23.6 % |
| `PinBuffer`-adjacent: unpin + private refcount | 1.5 % | 2.6 % |
| buffer hash lookup (`hash_search_with_hash_value`) | 5.1 % | 5.7 % |
| content lock (`LWLockRelease` + `LWLockAttemptLock`) | 10.5 % | 8.1 % |
| `randomize_mem` (buffer manager memset) | 2.7 % | 9.7 % |
| **buffer manager subtotal** | **~33 %** | **~50 %** |
| page opaque parse (`TsvPageOpaqueData::read_from_page`) | 9.5 % | (inline flag test) |
| element load: copies + Vec growth (`Vec<ItemPointer>::from_iter`, `finish_grow`, `realloc`) | 19.0 % | 25.8 % (`HnswLoadElementImpl`, mostly in-page, no copies) |
| distance kernel | 4.5 % (`distance_encoded_direct`) | 5.0 % (`vector_l2_squared_distance`) |
| visited hash insert | 2.6 % (`hash_one` + `hash_bytes` + `HashMap::insert`) | 3.2 % (`tidhash_insert`) |
| **libc malloc/free** | **15.6 %** | **1.4 %** |
| dso split | vectorscale 45.3 % / postgres 37.1 % / libc 15.6 % | postgres 59.9 % / vector.so 36.7 % / libc 1.4 % |

Scale-free reading (absolute ms per query at ef=160, from the sweep means):

| component | hnswsq (4.71 ms) | pgvector (2.53 ms) | delta |
|---|---|---|---|
| buffer pin + lookup + locks | 1.36 ms | 0.94 ms | +0.42 |
| buffer manager `randomize_mem` | 0.13 ms | 0.25 ms | -0.12 |
| page opaque parse | 0.45 ms | ~0 | +0.45 |
| element/neighbour materialization | 0.90 ms | 0.65 ms | +0.25 |
| libc allocation | 0.74 ms | 0.04 ms | +0.70 |
| distance arithmetic | 0.21 ms | 0.13 ms | +0.08 |
| visited hashing | 0.12 ms | 0.10 ms | +0.02 |
| **accounted** | **3.91 ms** | **2.11 ms** | **+1.80 ms** |

So the 2.18 ms gap is *not* distance arithmetic (0.08 ms of it) and not the hash
sets (0.02 ms). It is:

1. **memory allocation and copying per visited node** — 1.64 ms of the 2.18 ms
   (`libc` 0.70 + materialization 0.25 + the Rust alloc glue inside
   `finish_grow`/`realloc` that lands in the "materialization" row),
2. **re-parsing the page opaque header on every load** (0.45 ms),
3. buffer-manager mechanics we do slightly more expensively (0.42 ms).

## 3. Why: the two implementations' per-hop work

Both engines pay one `ReadBuffer` + `LockBuffer(SHARE)` + `UnlockReleaseBuffer`
per hop and one hash probe per visited id — that part is the same by design.
What differs is what happens around it.

**pgvector** (`hnswutils.c`):

* `HnswLoadElementImpl(blkno, offno, ...)`: `ReadBuffer` + `LockBuffer(SHARE)`,
  then **computes the distance straight out of the page tuple**
  (`HnswGetDistance(q->value, PointerGetDatum(&etup->data))`) — no copy, no
  allocation; and it **materialises the element only if the neighbour is
  admitted** (`eElement = NULL;` … `if (eElement == NULL) continue;`), into a
  palloc'd element (their own allocator, not malloc).
* The neighbour *list* lives in a separate page and is read once per expansion
  (`HnswLoadUnvisitedFromDisk`), already filtered against the visited hash.
* Results are the elements built during the search, so emitting a tuple is
  `element->heaptids[--element->heaptidsLength]` — **no second pass over the
  graph**.
* `hnswgettuple` sets `xs_recheckorderby = false` (hnswscan.c:325): the executor
  trusts the index order, so each returned tuple costs one heap fetch for MVCC
  visibility and nothing else.

**hnswsq** (`node.rs`, `graph.rs`, `scan.rs`):

* `load_node_view`: `ReadBuffer` + share lock, `rkyv::archived_root`, then it
  **copies the whole node out of the page**: `neighbors` into a fresh
  `Vec<Vec<ItemPointer>>` (one allocation for the outer vec + one per layer) and
  the encoded vector into a fresh `Vec<u8>` (512 B at dim 128). Thick pages
  (`get_type()` -> `TsvPageOpaqueData::read_from_page` -> `verify()`) are
  re-parsed on every load.
* `search_layer::<DiskGraph>` additionally keeps a `visited: HashSet` **and** a
  `cache: HashMap<Id, VisitData>` of those copied views, and clones the
  neighbour list *again* per expansion (`vd.neighbors.clone()`), hashing
  `ItemPointer`s with SipHash (RandomState).
* `compute_results` then runs a **second pass over the results**: for each of
  the up-to-`ef` hits it calls `load_node_view` *again* (another page read,
  another copy), `codec.decode`s the vector into a fresh `Vec<f32>`, computes
  `‖v̂‖` and a per-node error bound.
* `amgettuple` sets `xs_recheckorderby = true` and reports **provable lower
  bounds**. The executor then, per returned tuple: fetches the heap tuple,
  recomputes the exact operator value, verifies it against our reported value
  ("index returned tuples in wrong order" otherwise), and pushes a palloc'd copy
  into a **reorder pairing heap** that can only drain when
  `topmost_exact <= last_reported_by_index`
  (`IndexNextWithReorder`, `nodeIndexscan.c`). Because our reported values are
  lower bounds (≤ exact), that condition is harder to satisfy than it would be
  with exact values, so the queue pulls more tuples than pgvector's scan does.

The three structural differences, in order of size:

| | pgvector | hnswsq | cost |
|---|---|---|---|
| per-hop vector access | distance computed in page | vector copied to `Vec<u8>` | allocation + memcpy per hop |
| per-hop neighbour list | read from its own page, filtered by visited | copied into `Vec<Vec<ItemPointer>>`, then cloned per expansion | 2-3 allocations + 2 copies per hop |
| emitting results | reuse elements from the search | re-load + decode + norm per hit | up to `ef` extra page loads, decodes, norms |
| order-by contract | exact values, `xs_recheckorderby = false` | lower bounds, `xs_recheckorderby = true` | executor recompute + reorder queue per returned tuple |
| result of the query | 2.53 ms (ef 160) | 4.71 ms | 1.86x |

## 4. Where the build time goes (and why the gap is parallelism)

`hnswsq.build_stats` on the same 1M build (single backend):

```
search=372.1s (76%)  backlink_select=97.9s (20%)  flush=10.8s  select=7.0s
```

* 76 % is the graph search per inserted row — the part a parallel build attacks
  directly; the same code path is what the earlier parallel attempt failed to
  parallelize *safely* (see the perf-notes "parallel build" section: the batched
  plan/apply design loses connectivity, so a parallel build needs search-time
  visibility of in-flight inserts, i.e. the per-node-lock design).
* 20 % is the exact backlink re-prune. pgvector does not do this at all: it
  appends and only repairs a list when it overflows (`HnswUpdateConnection`),
  which is cheap per insertion but leaves a lower-quality graph — its recall at
  ef 160 is 98.9 % against our 99.3 %.
* Measured per-core: 562 s vs pgvector's 442 s. Our extra 120 s is roughly the
  backlink work (98 s) plus the fact that we keep the whole graph in per-node
  heap allocations (1.5 GB RSS at 1M nodes) instead of writing elements into
  pages as we go.

## 5. Fixes, ranked by measured benefit

Each item names the measurement that justifies it and the size of the prize on
the 1M/dim-128 workload above.

1. **Stop re-loading every emitted candidate (~15-35 % of query time).**
   `compute_results` re-reads each hit's page although `search_layer` already had
   its `VisitData` (heap TID + encoded vector) in hand. Carry the view into the
   result instead of re-loading: removes up to `ef` page loads, `rkyv` parses,
   `Vec<u8>` copies and `Vec<f32>` decodes per query (at ef 640 that is ~640 of
   the query's page touches). No format or contract change.
2. **Make the per-hop path allocation-free (~20-25 %).** Reuse a scratch node
   view across hops (`&[u8]` slices into the pinned page + fixed-size neighbour
   buffers), pass the page's vector bytes straight to `distance_encoded_direct`
   (it already takes `&[u8]`), and replace the `HashSet`/`HashMap` visited+cache
   pair with the epoch-stamped marks already used by the in-memory search
   (`hnswsq.scan` needs an id->ordinal map, but the node page/offset pair can be
   hashed with a cheap hasher). This removes the 15.6 % libc time and most of the
   8 % `finish_grow`/`realloc` time.
3. **Report exact distances and clear `xs_recheckorderby` for `plain`** (and only
   for the lossless layout; the quantized layouts keep the lower-bound path).
   `plain` stores f32 verbatim, so the index's distance *is* the operator value —
   that is exactly what pgvector does for `vector`. This drops the executor's
   per-tuple exact recomputation, its cmp check and the reorder pairing heap, and
   lets the scan stop as soon as k visible tuples are produced. The 1e-4 slack
   and the `f32::NEG_INFINITY` clamping machinery become unnecessary on the
   `plain` path.
4. **A real plain-layout distance kernel (top build cost).** `distance_encoded_direct`
   walks `bytes.chunks_exact(4)` through `Map`/`Enumerate` iterator adapters
   calling `f32::from_le_bytes`. The *build* profile shows this chain at ~65 % of
   samples while the SIMD kernel it feeds (`distance_l2_x86_avx2`) is 3.3 %:
   for `plain` (and `ieeefp16`, where half→f32 is vectorizable) the bytes can be
   read directly as `&[f32]` and handed to `dist_fn`, skipping the per-element
   decode entirely. This is the single biggest CPU item in a dim-128 build.
5. **Cheapen the page access itself.** `TsvPageOpaqueData::read_from_page`
   (`verify()` with an `assert_eq!`) shows up at 9.5 % of the query profile; it
   is called once per hop and only needs a byte compare. Add
   `PrefetchBuffer` for the next candidate's page (Lance's `prefetch_distance`).
6. **Parallel build** — the only way to close the 5.5x wall-clock build gap, and
   the measured constraint is now explicit: per-core we are within 1.27x, so the
   win is entirely `workers x per-core throughput`; a design that cannot show
   in-flight inserts to concurrent searches loses 20-40 points of recall
   (measured), so it has to be the per-node-lock/Lance form.

## 6. Methodology trap found while doing this

`cargo pgrx test` **installs a debug build of the extension over the release
one** in the pgrx prefix. On this box that silently turned a 562 s 1M build into
an aborted ~30-minute one and made every subsequent measurement ~25x too slow.
`gap_study.sh` now refuses to run unless the installed `.so` is release-sized,
and any benchmark after a test-suite run must reinstall with
`cargo pgrx install --release`.
