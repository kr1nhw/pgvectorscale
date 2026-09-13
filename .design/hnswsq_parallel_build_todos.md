# Parallel hnswsq build — TODO analysis

Scope: make the **in-memory** build phase of `hnswsq` use several workers, to
close the last measured gap to pgvector hnsw.  Everything below is grounded in
the same-host gap study (`.design/hnswsq_vs_pgvector_gap.md`), in the parallel
build that already ships in this repo for diskann
(`src/access_method/build.rs`: `ParallelShared`, `PARALLEL_BUILD_MAIN`,
`min_vectors_for_parallel_build()`, `do_heap_scan(.., ParallelBuildInfo)`), and in
pgvector's `hnswbuild.c`, whose design is documented in its own header comment.

## 1. What it must buy, and the constraint that shapes it

Measured on 121 (32 vCPU, 1M BIGANN dim 128, both engines release):

| build | workers | seconds |
|---|---|---|
| hnswsq (after P1) | 1 | **352** |
| pgvector | 1 | 442 |
| pgvector | 4 (`max_parallel_maintenance_workers` default) | 103 (4.29x) |
| pgvector | ~7 | 64 (6.9x) |

So single-core we are already faster than pgvector; **the entire remaining build
gap is worker count**.  Acceptance for this work: 1M build at 4 workers
**≤ 160 s**, recall within 0.005 of the single-backend build, index size
unchanged, connectivity invariants intact, `workers = 0` (today's behaviour)
untouched.

The hard constraint comes from the rejected attempt (recorded in the perf notes):
a node being inserted must be **visible to concurrent searches as soon as its own
list is published**.  A batched "plan everything in parallel, then apply in
order" design hides batch-mates, the batch-mates all attach to the same already
full pre-batch lists, and nodes end up with no incoming edge at all:

| rows invisible to each other | threads | nodes with no incoming edge | recall@10 |
|---|---|---|---|
| 1 (sequential) | 1 | 0 | 0.9550 |
| 2 | 2 | 11 | 0.9400 |
| 64 | 4 | 402 | 0.6250 |
| 256 | 8 | 452 | 0.5800 |

Therefore: **per-node read/write locking with live publication is mandatory**,
in every design below.

## 2. Two designs

**A. PostgreSQL parallel workers + shared memory (pgvector's model, already the
pattern in this repo).**  The graph lives in one DSM chunk; every pointer in it
is a *relative* pointer (offset from the chunk base), each node is guarded by its
own LWLock, the entry point has its own lock plus a "wait for it to exist" lock,
and elements are allocated from a shared bump allocator under an allocator lock
with a margin so allocation cannot fail mid-build.  Workers are real PG
processes, each scanning its own slice of the heap through a shared
`ParallelTableScanDesc`; the leader waits for them and then materialises the
graph to disk exactly as today.

**B. In-process `std::thread` workers with a `RwLock` per node.**  The Rust graph
stays `Vec`-based, wrapped in per-node locks; the leader buffers rows and hands
them to a scoped thread pool.

| | A (PG workers + DSM) | B (threads + RwLock) |
|---|---|---|
| memory model | one sized arena, relative pointers | today's Vecs, unsized, per-node allocations |
| worker count | `ii_ParallelWorkers` / `max_parallel_maintenance_workers` (4 on 121) | our own GUC, up to nproc |
| lifecycle, cancellation, errors | PG owns them; workers get signals, `shm_mq` error propagation | we own them; a cancel must be noticed by the leader while workers run |
| PG APIs inside a worker | allowed (real backends) | **forbidden** (no palloc/ereport/buffer manager off-thread) |
| code to reuse from this repo | diskann's whole parallel scaffolding + progress reporting + forced-worker GUC | none |
| new code | arena + relative pointers + locks + rendezvous | locks + thread pool + batching + arena flattening (for lock granularity) |
| scale ceiling | bounded by `max_worker_processes` / `max_parallel_maintenance_workers` | bounded by cores |
| proven at this task | yes — pgvector's 4.29x/6.9x is this design | not in this codebase |

**Recommendation: A.**  It matches the reference that produced the numbers we are
chasing and the scaffolding this repository already ships; it keeps workers as
PG processes (so cancellation, error propagation and the worker-count decision
are Postgres's problem); and B's only advantage — staying with `Vec`s — is lost
anyway once per-node locking forces a flat, fixed-capacity node layout.  The one
thing A costs is the relative-pointer discipline, which is containable: keep the
arena types in a single module behind a `Rel<T>` wrapper and do not let them leak
into the algorithm code.

## 3. TODO list

Sizes: S ≤ half a session, M ≈ one session, L ≈ a session or more with debugging.

**T0 — Freeze the graph interface (M).**  Put the memory-graph accessor surface
behind a trait (`level/tid/clamped/vector/probe/expand/set_list/entry`) so
`mem_plan`/`mem_apply`/`search_layer_mem`/`backlink_prune_mem` compile against
either the existing `MemGraph` or the arena.  *Why first:* the arena port must not
touch the algorithms, or every later step becomes a mixed change.
*Gate:* suite green, no behaviour change, 100k build time unchanged.

**T1 — Relative-pointer arena (L).**  `HnswArena` in one `shm_toc` chunk: header,
node slots (level, tid, clamped, `vector_bytes` payload, per-layer list slabs),
a free/bump cursor guarded by `allocatorLock`, and a reserved margin.  `Rel<T>`
for every internal pointer.  Size it from `maintenance_work_mem` minus a margin
before any worker starts.  *Gate:* unit tests for allocation, exhaustion, and
offset stability under a simulated remap; 1M arena fits and reports its high-water
mark.

**T2 — Per-node locks (M).**  One LWLock per node slot in the same chunk
(tranches if the slot count ever dwarfs the lock budget), plus `entryLock` and
`entryWaitLock` as pgvector does.  Rule enforced by construction: **hold at most
one node lock at a time**, always in the same order (own node → then each
backlink target, released in between).  *Gate:* debug assertion that detects a
second simultaneous node lock; fuzz test with two writers on one list.

**T3 — Entry-point rendezvous (M).**  Reuse `initial_start_nodes_count()` and the
condition variable already in `ParallelShared`: one worker (or the leader)
inserts the first N nodes alone while the others wait, then everybody searches a
graph that already has an entry.  A worker that still finds no entry must wait,
never synthesise one.  *Gate:* test with workers that start in reverse order.

**T4 — Parallel heap scan (M).**  Switch hnswsq pass 2 from `IndexBuildHeapScan`
to the `table_index_build_scan` + shared `ParallelTableScanDesc` shape that
diskann already uses (`do_heap_scan(.., ParallelBuildInfo)`), keeping the current
single-backend path for `CREATE INDEX CONCURRENTLY` (which cannot go parallel and
already forces disk mode) and for `build_workers = 0`.  *Gate:* row counts and
`pg_stat_progress_create_index` match the single-backend run.

**T5 — Concurrent insert into the arena (L).**  `arena_plan` reads with shared
locks (`probe`/`expand` over arena slots); `arena_apply` takes its own node's
write lock, publishes the list, then walks the backlinks taking one target lock
at a time.  Decide the conflict policy and measure it: pgvector re-checks the
element's version and retries the whole link; Lance's `try_write`-and-skip is
cheaper but drops edges.  *Gate:* the connectivity invariant (no node without an
incoming layer-0 edge) and recall within 0.005 of sequential at workers 1/2/4/8 in
the in-memory harness.

**T6 — Backlink policy under concurrency — DECIDED: pgvector's append/shrink
policy, in the arena/parallel path only (M).**

Exact algorithm (read from `hnswutils.c:HnswUpdateConnection`, which the insert
path calls per selected neighbour from `hnswinsert.c:441`; note this is *not* the
"replace the farthest" shortcut one might expect):

```
arena_backlink(target, x, layer, cap, d_self):
    lock target for write (the only lock held; d_self computed before locking)
    if target.list.len < cap:
        append (x, d_self)                      # O(1), the common case
    else:
        candidates = target.list ∪ {(x, d_self)}
        pruned     = SelectNeighbors(candidates, cap)   # the SAME diversification
                                                        # heuristic we run, with a
                                                        # reusable closer-set bitset
        replace the entry equal to `pruned` with x      # in place, list stays full
    unlock
```
No version/retry is needed in the arena: with a per-node write lock the
read-modify-write is atomic, and the only input that needs the lock is the list
itself (`d_self` is precomputed from the two vectors).  Dedup check (never link a
node to itself, never append a duplicate) is an assertion.

Consequences (what changes in T1/T5):
* the arena slabs hold **ids only** — no `list_dists`, no `list_masks`, so ≈150 MB
  of the 1M arena disappears and the slab layout is just `[u32; cap] + len`;
* `arena_apply` becomes: publish own list, then one target lock at a time, append
  or shrink-and-replace — no extras bookkeeping, no merged-list sort, no masks to
  maintain across revisions;
* the policy needs the target's *current* distances only when it is full, which is
  where the precomputed `d_self` plus per-neighbour distances come from the same
  probe path the search already uses.

Measured cost of the decision (sequential, single-backend, so it is the policy
alone): 99.3% vs 98.9% recall@10 at ef 160 on 1M BIGANN, and the low-ef effect is
sharper (0.508 vs 0.800 at ef 40 in the dim-16 ranked-mode A/B).  Cause: while a
list is *unsaturated* pgvector's list is "the first M arrivals", not the
heuristic's diversified set; the heuristic only runs on overflow.  Acceptance
therefore splits in two:

1. **Policy cost** (recorded, not a pass/fail): parallel-path policy vs today's
   exact policy at 1 worker, recall sweep at 100k and 1M — expect a few tenths of
   a point at low/mid ef.
2. **Concurrency cost** (the gate): workers 1/2/4/8 with the *same* policy must
   agree within 0.005 recall, with zero nodes lacking an incoming layer-0 edge.

The single-backend path keeps the exact incremental re-prune (it is today's
shipped behaviour, it is better, and the acceptance in §1 requires `workers = 0`
to be untouched).  That leaves three recorded policies — exact (default
single-backend), pgvector append/shrink (arena/parallel), Lance ranked/cutoff
(`hnswsq.build_backlink_mode = 0`, measured slower and lower recall twice) — so
T10 must document which applies when.  Unifying the single-backend path onto the
arena + append/shrink later is a separate decision to take on measurement, not a
prerequisite.

**T7 — Writeout, spill and failure paths (M).**  Leader does
`WaitForParallelWorkersToFinish`, then the *existing* `flush_mem_graph` +
`HnswMetaPage::update` over the arena (ids are claimed from a shared counter, so
the id order needed by the writeout still exists).  Two failure paths must be
handled explicitly: **arena exhaustion mid-build** (stop workers, flush what
exists, finish the remaining rows through the on-disk `insert_vector` path — the
same transition `spill_to_disk` already performs) and **worker error** (propagate
via the parallel context; the build is transactional, so no half-index survives).
*Gate:* forced-exhaustion test at 100k, worker-error injection, statement cancel
during a build.

**T8 — Sizing and worker-count decision (M).**  Estimate the arena need (rows ×
per-node bytes) from the table's `reltuples` and the index options; only go
parallel when it fits, otherwise keep today's path.  Worker count from
`ii_ParallelWorkers`, with an override GUC mirroring
`tsv_force_parallel_workers`, and a floor (reuse
`min_vectors_for_parallel_build`).  *Gate:* a build that does not fit never spawns
workers; a forced-worker run at 100k works.

**T9 — Stats and progress (S).**  Per-worker `BuildStats` merged at the end so
`hnswsq.build_stats` keeps its current one-line aggregate; per-worker
`pgstat_progress_update_param` following diskann.  *Gate:* the existing bench
harness parses the line unchanged.

**T10 — Surface and docs (S).**  `hnswsq.build_workers` (0 = off, default) as the
override; document that it applies only to non-concurrent in-memory builds, that
graph *edge sets* (not node ids) vary run to run, and that
`max_parallel_maintenance_workers` caps the win.  *Gate:* docs reviewed against
the measured tables.

**T11 — Cancellation and interrupt latency (M).**  The leader must stay
responsive: `CHECK_FOR_INTERRUPTS` between batches/phases, then
`WaitForParallelWorkersToFinish` before erroring.  *Gate:* cancel a 1M build and
assert a clean error, no leaked workers, no index left behind.

**T12 — Validation harness (M).**  Extend `gap_study.sh` with a worker sweep
(1/2/4/8) that records build time, recall sweep, size and connectivity; port
`assert_graph_invariants` (build.rs mem tests) to the arena; run the 4-host
suite and `tests/test_hnswsq_concurrent.py`.  *Gate:* the acceptance table in §1.

### Suggested order

T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8/T9/T10 → T11 → T12, i.e. build the
arena and the locking discipline *before* touching the scan plumbing, and keep the
single-backend path as the default and as the reference to diff against at every
step.  A useful intermediate milestone is T1+T2 only, measured with one worker:
if the arena costs more than ~5% single-threaded, fix the layout before adding
concurrency (the arena also removes the per-node `Vec` allocations that the gap
study flagged as build CPU).

## 4. Test plan and gates

* **Unit:** arena allocation/exhaustion/margin, `Rel<T>` round-trip, lock-order
  assertion, version/retry correctness under two concurrent writers, entry
  rendezvous.
* **In-memory harness:** workers 1/2/4/8 → recall within 0.005 of sequential,
  zero nodes without an incoming layer-0 edge, no self-loops/duplicates/over-cap
  lists.
* **pg-level:** 100k and 1M at workers 1/4, pinned `build_seed` where determinism
  allows, recall sweep + size + `EXPLAIN` sanity; vacuum, rollback, REINDEX and
  the concurrency suite unchanged.
* **Negative paths:** arena exhaustion, worker failure, cancel.
* **Regression guard:** `workers = 0` must stay behaviourally and
  performance-wise identical (assert an unchanged graph for a pinned seed, and no
  build-time regression at 100k/1M).

## 5. Open decisions

1. **Design A vs B** — recommend A (§2).
2. **Default** — recommend off (`build_workers = 0`) until the recall and
   connectivity gates pass, then auto-enable above
   `min_vectors_for_parallel_build` and a fits-in-budget check.
3. **Backlink policy in the parallel path** — exact re-prune in the arena vs
   pgvector-style repair-on-overflow (T6; needs a measured call).
4. **Spill semantics** — stop-and-flush on arena exhaustion (recommended, matches
   pgvector's "flush when out of memory, then continue on disk") vs pre-sizing so
   a spill cannot happen.
5. **Determinism** — accept that edge sets vary between runs at workers > 1
   (node ids and levels stay deterministic), and document it.
6. **Flatten the single-backend graph too?** — recommended yes: it is a
   prerequisite of the arena layout and a standalone win (fewer allocations per
   node, better locality in `search_layer_mem`).

## 6. Risks

| risk | mitigation |
|---|---|
| Recall regression from concurrency (the earlier failure mode) | connectivity + recall gates at every worker count; the exact policy available as a fallback |
| DSM sizing: a 1M graph is ~1.5 GB, `shm_toc`/`shared_memory_type` limits | size check before spawning (T8), spill path as the escape hatch |
| Worker-count ceiling on the target boxes (`max_parallel_maintenance_workers = 4`, `max_worker_processes = 8`) | document the tuning; the win at 4 workers is already 4.3x |
| Two build paths drifting apart | single path stays the reference; diff graphs for pinned seeds; keep the parallel path opt-in until validated |
| Scope creep into the on-disk insert path | explicitly out of scope: its two-phase protocol, WAL and locking stay untouched |

## 7. Effort

Core (T1/T2/T5) is where the risk concentrates; T3/T4/T7/T8 are plumbing with
existing patterns to copy (diskann, pgvector).  Realistic: **3-5 focused
sessions** including the validation cycles, with the first measurable milestone
(arena at one worker, no regression) after T1+T2.
