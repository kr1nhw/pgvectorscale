# Parallel hnswsq build — TODO analysis

## 0. Status (implementation log, newest last)

**Implemented and measured** (all committed; numbers are 100k dim-128 uniform,
pinned seed, release, same host, unless stated):

| piece | state | evidence |
|---|---|---|
| legacy engine (exact re-prune) | unchanged, still the default | build 21.5 s, fingerprint `e4aaa8fa5d6b7f25`, `no_incoming=384` |
| flat engine: flat slabs, ids only, policy-free | done | `flat_graph.rs`, 15 tests |
| flat engine: search / select / plan / apply (append-shrink) | done | `flat_engine.rs`, 10 tests; build 20.9 s, fingerprint `30db11c54b7edd31`, `no_incoming=519` |
| driver seam + temporary `hnswsq.build_engine` | done | both engines A/B in one binary; per-phase split + fingerprint in the stats line |
| determinism | proven | each engine reproduces its fingerprint exactly across runs |
| policy cost (append/shrink vs exact) | measured | apply +2.1 s, plan −1.35 s, net build ~0.6 s faster, recall identical here |
| writeout bridge (flat → MemGraph → existing flush) | done, temporary | replaced by an arena-native writeout during the storage swap |
| `arena.rs`: node locks + one-write-lock enforcement | done | 8 tests |
| fixed capacity + spill hand-off | done | `with_limits`/`try_push_node`; end-to-end `disk_mode=false`, same fingerprint |
| claim/publish + published watermark | done | 20 tests; readers bound-check the watermark |
| capacity planned from the byte budget | done | `plan_capacity`; never under-counts (test) |
| chunk layout + `Chunk` backing store | done | offset-addressed, single allocation, alignment asserted at the accessor |
| structural gate (`check_lists`, both engines) | done | wired into the stats line; found the connectivity finding below |
| connectivity control (legacy vs flat) | done | legacy 384 / flat 519 at 100k: a 0.14 pp policy delta, not a defect ⇒ the gate is *relative* (§3e) |
| backfill knob (`hnswsq.build_backfill`) | done | measured: +51% build, +9.5 recall pts at ef 40 here; decision deferred to 1M BIGANN |

**Remaining, in order:**

1. **M6 decision: `build_backfill` 0 vs 1 on 1M BIGANN** (recall sweep at ef 10..640,
   build seconds, `no_incoming`, fingerprint).  The knob exists; the run is the work.
2. **M3 storage swap**: point `FlatGraph`'s arrays at `Chunk` regions behind the same
   accessors, allocate the chunk from `shm_toc`, replace `NodeLocks`'s `RwLock`s with
   LWLock tranches.  No open design questions — `arena_layout`, `plan_capacity`,
   `Chunk` and the watermark are all in place and tested.
3. **M3 driver**: entry rendezvous (`initial_start_nodes_count` + the `ParallelShared`
   condition variable), `table_index_build_scan` with a shared `ParallelTableScanDesc`,
   `amcanbuildparallel` under `feature = "build_parallel"`, a pre-drawn level table for
   determinism, then the worker sweep 1/2/4/8 with the relative gates above.
4. **M4–M5**: arena-native writeout, exhaustion/error/cancel paths, per-worker stats,
   worker-count sizing surface.
5. **M7**: settle the surviving policy (append/shrink only, or also exact) with the 1M
   measurement, then delete the legacy engine, the adapter and `build_engine`.

## 3g. Same-dataset A/B at 1M: the flat engine is 1.78x faster, with better low-ef recall

Same local 1M dataset (dim 16), pinned seed, release, back to back:

| engine | build | plan (search) | apply (backlinks) | `no_incoming` | recall@10 ef 40 / 160 |
|---|---|---|---|---|---|
| legacy (exact re-prune, `list_dists`/`list_masks`) | **174.0 s** | 105.2 s | 50.5 s | 0 | 0.8000 / 1.0000 |
| flat (append/shrink, ids-only) | **97.8 s** | 73.4 s | 19.4 s | 1 | **1.0000** / 1.0000 |

**1.78x faster single-threaded**, split across both halves: plan 105.2 -> 73.4 s (1.43x,
the flat layout's locality plus ids-only writes and no backfill) and apply 50.5 -> 19.4 s
(2.6x).  Recall is *better* at ef 40 (1.0000 vs 0.8000) and equal at ef 160, with
connectivity equivalent (1 vs 0 nodes without an incoming edge out of a million).

Two things this changes:

* **the parallel-build acceptance math gets easier**: a 97.8 s single-core 1M build at
  this size means the 4-worker target is plausible without exotic scaling, and the
  single-core comparison against pgvector (442 s on BIGANN) now favours us by ~4.5x rather
  than 1.27x -- the BIGANN number still has to be re-measured with the flat engine before
  that is quoted anywhere;
* **the apply-half comparison is now dataset-dependent in *both* directions**: flat's apply
  is 2.6x *cheaper* here (many targets stay unsaturated, so appends dominate) and 1.4x
  *dearer* on the 100k dim-128 set (tight clusters saturate targets, so every backlink
  pays the overflow re-measure).  Any claim about the policy's cost has to name the data
  shape; the phase split in the stats line is what makes that checkable per run.

Also worth noting for the legacy engine's own quality: it backfills unconditionally, and
the 1M backfill experiment (§3f) showed backfill costing 20 recall points at ef 40 on this
dataset -- i.e. the flat engine's ef-40 advantage here is plausibly the *legacy* engine's
backfill being a liability at low search width, not the flat engine being clever.

## 3f. Backfill decision measured at 1M (local, dim 16) -- keep it OFF

Same harness, 1M rows (the local `t1000000` dataset), pinned seed, release, flat engine
(`build_engine = 1`), the two backfill settings back to back:

| `build_backfill` | build | plan (search) | apply (backlinks) | `no_incoming` | `reachable` | recall@10 ef 40 / 160 |
|---|---|---|---|---|---|---|
| **0 (decided policy)** | **97.8 s** | 73.4 s | **19.4 s** | **1** | 999 999 | **1.0000** / 1.0000 |
| 1 (closest-pruned backfill) | 143.7 s | 79.5 s | 60.6 s | 0 | 1 000 000 | 0.8000 / 1.0000 |

Three conclusions, and they settle the question for the default:

1. **Backfill is not needed for connectivity.**  At 1M the decided policy leaves *one*
   node without an incoming edge out of 1M (999 999 reachable) — the 519-node figure was
   a 100k artifact of short young lists, not a property that persists at scale.  So the
   candidate fix (2) (`always_admit`) has no case at all, and the relative gate from §3e
   is the right shape.
2. **It costs a lot:** +47% build time (97.8 -> 143.7 s), concentrated in the apply half
   (19.4 -> 60.6 s, 3.1x), because every backlink then lands on a saturated target and
   pays the full re-measure plus occlusion walk instead of an O(1) append.
3. **Its recall effect is dataset-dependent and can be negative:** at 1M dim-16 it *hurts*
   at ef 40 (0.8000 vs 1.0000) because filling lists with occluded, non-diverse
   candidates degrades the layer-0 neighbourhood at low search width; on the 100k dim-128
   dataset (§3d) it helped (0.2000 -> 0.2945 at ef 40).  Two datasets, opposite signs,
   with the easy one at the operating point where recall is already saturated.

Decision: **`hnswsq.build_backfill` stays 0.**  The knob is kept (it is one GUC, useful
for the BIGANN operating-point check when host 121 is reachable again), with the
dataset-dependence documented rather than generalised from a single measurement.

**Blocked step, recorded so the next session does not re-discover it:** the 1M BIGANN
`build_backfill` 0-vs-1 decision run (item 1 above) could not be started — host
121.37.117.106 was unreachable over SSH for two consecutive attempts (connection
timeout; it had also failed a banner exchange earlier in this session).  Everything
needed for that run is committed and installable: the knob, the pinned-seed harness,
and `run_sweep_hnswsq.sh`; the sequence is sync → `cargo pgrx install --release` →
two builds (`hnswsq.build_engine = 1`, `build_backfill` 0 then 1, pinned seed) → sweep.
No code or measurement depends on anything else being done first, so it can run as soon
as the host answers.

**Two traps this work has already paid for**, both worth re-reading before touching the
harness: `cargo pgrx test` installs a *debug* extension over the release one (the cycle
scripts now refuse to run against it), and recall alone cannot see connectivity damage
(0.52% invisible nodes looked exactly like 0.38% in every recall number).

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

**Approach change (decided): build the new memory graph *alongside* the old one.**

Rather than freezing an interface over the existing `MemGraph` and then porting
it in place (the original T0→T1 order), the new implementation is written as a
separate module and the legacy one is deleted only once the new one passes every
gate.  Why this is better here:

* the window in which the build path is half-ported disappears — the legacy path
  keeps compiling, keeps passing tests and keeps being the default throughout;
* the legacy implementation becomes a *live* A/B reference instead of a
  historical one (recall, invariants, size and build time can be compared in the
  same binary, same dataset, same seed);
* the algorithms are private to each implementation, so the shared surface is
  only what the *build driver* needs (`ambuild`, the spill/budget check, the
  writeout, the stats line) — a much narrower trait than the full accessor
  surface the original T0 planned, and one that can be shaped once the new graph
  exists rather than guessed up front.

Consequences to honour while both exist:

* a temporary GUC selects the engine (e.g. `hnswsq.build_engine = legacy|arena`),
  removed together with the legacy module;
* the A/B is **statistical, not bit-identical**: the two implementations
  deliberately differ in backlink policy (exact re-prune vs append/shrink), so
  graph equality cannot be asserted — the comparison is recall sweep, connectivity
  invariants, size and build time, plus bit-identical *self*-comparison per engine
  (same seed) to catch nondeterminism;
* `workers = 0` keeps routing to the legacy engine until the new one passes, and
  the query path is untouched by both.

**Deletion decision (must be taken *before* dropping the legacy engine).**  Once
the legacy `MemGraph` is deleted, whatever policy the surviving implementation
implements becomes the *default single-backend* behaviour.  If the new engine
implements only append/shrink, the default build loses the exact policy's
advantage — measured at 0.4-0.5 points of recall@10 (99.3 vs 98.9 at ef 160 on 1M
BIGANN; sharper at low ef in the dim-16 ranked A/B) — for every user, not just
those who opt into `build_workers > 0`.  Two ways out, to be settled with a
measurement on 1M BIGANN rather than by default:

1. accept append/shrink everywhere (simplest; the ids-only slab stays, no
   metadata anywhere) and document the recall shift, or
2. let the surviving engine implement both policies (exact needs `list_dists` +
   `list_masks` back in the slab — the ~150 MB/complexity the T6 decision was
   meant to avoid) and keep exact as the default.

**T0 — Freeze the graph interface (M) — superseded by the approach above.**  Put the memory-graph accessor surface
behind a trait (`level/tid/clamped/vector/probe/expand/set_list/entry`) so
`mem_plan`/`mem_apply`/`search_layer_mem`/`backlink_prune_mem` compile against
either the existing `MemGraph` or the arena.  *Why first:* the arena port must not
touch the algorithms, or every later step becomes a mixed change.
*Gate:* suite green, no behaviour change, 100k build time unchanged.

**T0 design (interface shape decided; the mechanical rewiring is what remains).**
Seven consumers reach into the graph today — `DistBuf::push` (build.rs:361),
`select_neighbors_heuristic_mem` (:431), `search_layer_mem` (:592),
`backlink_prune_mem` (:810), `mem_plan`/`mem_apply`, and `flush_mem_graph`
(:1551) — all against concrete `MemGraph` fields.

The trait must be implementable by *both* a `Vec`-backed single-threaded graph
and a lock-guarded shared arena, and those two want opposite receivers: the arena
mutates through `&self` (interior locks, because workers only ever hold a shared
reference to the chunk), while `MemGraph`'s writes are plain `&mut self`.  Two
consequences, both decided here so the rewiring is mechanical:

1. **All trait methods take `&self`, including the write side** (`set_list`,
   `set_entry`, and the per-target snapshot).  `MemGraph` implements them through
   a small `RefCell`-scoped adapter (single-threaded; the borrow flag costs a
   couple of instructions against a once-per-list write), the arena implements
   them with its per-node locks.  Consumers take `&G` (or `&mut G` where they also
   hold scratch) and never touch fields.
2. **Reads copy into caller-owned buffers, not borrowed slices**:
   `copy_neighbors(&self, id, layer, out: &mut Vec<u32>)`,
   `copy_vector(&self, id, out: &mut Vec<u8>)`, `copy_list_meta(&self, id, layer,
   &mut Vec<f32>, &mut [u64; 2])`.  A borrowed `&[u32]` cannot escape a method
   that must hold a read guard for the duration of the borrow, and this is
   already the shape the disk path settled on in P2 (`probe`/`expand`).  Cost in
   the memory search: one ≤128 B copy per expansion, which the ±2 % build-time
   gate covers.

Work items for T0 itself: add `trait MemoryGraph` with `len/entry/entry_level/
level/tid/clamped/copy_vector/copy_neighbors/copy_list_meta/set_list/set_entry`;
implement it for `MemGraph` (plus the `RefCell` adapter); switch the seven
consumers to it; keep `MemGraph`'s public field access private to the module.
*Gate:* suite green (61), identical counters for the pinned seed, 100k dim-128
local build within 2 % of 22.3 s.

**M3 finding that de-risks T1: the flat graph is already offset-addressed.**
Every access in `FlatGraph` is index arithmetic over flat arrays — `ids[slab * cap +
k]`, `lens[slab]`, `vectors[id * stride ..]` — with no pointers anywhere.  So the
DSM port is *not* pointer swizzling: it is (a) making the four arrays (vectors, ids,
lens, levels/tids/clamped) slices of one `shm_toc` chunk instead of process-local
`Vec`s, (b) a bump allocator over a fixed byte budget with a reserved margin, and
(c) the lock array.  Consequences the arena must honour and the prototype must
already respect:

* **nothing may grow** — the arrays are sized once from the budget, and running out
  is a *normal* outcome that stops the workers, flushes what exists and continues on
  the disk path (the existing `spill_to_disk` transition), not an error path;
* **node data must be published before it is observable**: a worker claims an id
  from the shared counter and only then writes level/tid/clamped/vector, so readers
  bound-check against the *graph's* length (the flat search already does) and the
  arena needs a published-watermark check before trusting a slot's contents;
* the per-node lock array must never be reallocated while a worker holds a guard,
  which is why `NodeLocks` has `grow_to` (documented: extend between phases, under
  the driver's serialization, never under a held guard).

First piece implemented: `arena.rs` — `NodeLocks` with read/write guards where
`write` counts the current thread's write guards and panics on a second one, so the
"one target lock at a time" rule that keeps the backlink order acyclic is enforced
rather than assumed (`std::sync::RwLock` here, LWLock tranches in the arena behind
the same API).  4 unit tests.

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

## 3b. M1/M2 measurement record (100k dim-128 uniform, pinned seed, release)

Both engines are selectable with `hnswsq.build_engine` and now report a per-phase
split plus a stable graph fingerprint under `hnswsq.build_stats`, so the policy cost
is attributable and determinism is checkable directly rather than by inference:

| engine | run | build | plan (beam search + selection) | apply (own lists + backlinks) | fingerprint |
|---|---|---|---|---|---|
| legacy (0) | 1 | 21.50 s | 13 230 ms | 5 807 ms | `e4aaa8fa5d6b7f25` |
| legacy (0) | 2 | 21.52 s | 13 244 ms | 5 816 ms | `e4aaa8fa5d6b7f25` |
| flat (1) | 1 | 20.90 s | 11 879 ms | 7 897 ms | `30db11c54b7edd31` |
| flat (1) | 2 | 20.83 s | 11 842 ms | 7 878 ms | `30db11c54b7edd31` |

Reading:

* **determinism proven directly** — each engine reproduces its fingerprint exactly
  across runs; the two fingerprints differ, which is the point of the switch
  (append/shrink vs exact re-prune);
* **the policy cost is +2.1 s of apply time** (7.9 s vs 5.8 s): on a saturated list
  the flat engine re-measures `d(target, member)` for every member instead of
  reading cached `list_dists`/`list_masks`;
* **the flat plan half is 1.35 s cheaper** (11.9 s vs 13.2 s): selection writes ids
  only (no dists, no mask) and does not backfill, so lists are shorter while the
  graph is young;
* **net: flat is ~0.6 s faster** (20.9 s vs 21.5 s) at identical recall
  (ef 10/40/160/640 = 0.0000 / 0.2000 / 0.2000 / 0.5000 on this dataset).

Caveat for M6: this dataset is hard (recall 0.2-0.5), so it cannot separate the two
policies' *quality*; the 1M BIGANN sweep is the gate for that.

## 3c. First structural measurement of the flat engine: 519 invisible nodes

Wiring `check_lists` into the writeout and running a real 100k dim-128 build with the
flat engine (append/shrink, no backfill in own-list selection) gave:

```
fingerprint=30db11c54b7edd31
checks(published=100000 no_incoming=519 reachable=99477 self_links=0 duplicates=0 max_len=32)
```

So the graph is well formed (no self-links, no duplicates, capacity respected) but
**0.52% of nodes have no incoming edge at all** -- invisible to every search -- and
523 nodes are unreachable from the entry.  The recall sweep could not see this: at
this dataset's granularity the numbers were identical to the legacy engine
(0.0000 / 0.2000 / 0.2000 / 0.5000 at ef 10/40/160/640), which is exactly why the
structural gate is a gate and recall is not.

Likely cause, to be confirmed by measurement: **the missing backfill**.  The legacy
engine's own lists are always filled to `cap` (occlusion-accepted entries plus the
closest-pruned backfill), so every new node attempts `cap` backlinks; the flat engine
selects the heuristic's output alone, so a node whose own list is short attempts few
backlinks, and on a saturated target an occluded newcomer is admitted by none of
them.  pgvector's forward list is also heuristic-only, so it plausibly has the same
property -- its recall being 0.4-0.5 points below ours at ef 160 is consistent with
that, not with a pure policy difference.

Candidate fixes, to be measured (M3/M6), in order of least semantic change:

1. restore the closest-pruned **backfill in `select_neighbors_flat`** (lists stay
   full, so backlink attempts stay at `cap`), keeping the crash-free append/shrink
   backlink policy -- the ids-only slab and the no-metadata property are unaffected,
   since backfill only changes *which* ids are in the list;
2. make admission of the *nearest* backlink target unconditional (a documented
   exception shaped like ranked mode's `always_admit`, which was added for exactly
   this failure mode earlier: without it recall fell to 0.63-0.76);
3. accept it, and re-check on 1M BIGANN where a 0.5% invisible fraction costs recall
   directly, before deciding.

Either way the gate stands as the acceptance criterion for `workers > 0`: zero nodes
without an incoming edge, everything reachable from the entry.

## 3d. Fix (1) measured: backfill does not fix connectivity and costs 51% build time

Restoring the closest-pruned backfill in `select_neighbors_flat` (own lists filled to
`cap` with the pruned candidates, as the legacy engine does) and re-running the same
100k dim-128 build:

| | heuristic-only (decided policy) | with backfill |
|---|---|---|
| build | 20.65 s | **31.14 s (+51%)** |
| plan (beam search + selection) | 11.9 s | 12.46 s |
| apply (own lists + backlinks) | 7.9 s | **17.68 s (2.2x)** |
| `no_incoming` | 519 | **426** |
| `reachable` | 99 477 | 99 572 |
| recall@10 ef 10/40/160/640 | 0.0000 / 0.2000 / 0.2000 / 0.5000 | **0.1000 / 0.2945 / 0.3000 / 0.5000** |

Two conclusions:

* **the hypothesis was mostly wrong** — backfill is not the cause of the invisible
  nodes: it removed 93 of 519 and left 426 (0.43%).  The mechanism must be something
  else (a newcomer occluded from every accepted entry of every target it selected is
  simply never admitted when those targets are saturated);
* **the cost is real**: full lists mean every backlink lands on a saturated target, so
  the apply half does the full re-measure + occlusion walk over `cap+1` candidates
  instead of an O(1) append: apply 2.2x, total build +51%.  In exchange recall at
  this dataset's low/mid ef improves a lot (ef 40: 0.2000 -> 0.2945).

That is a *quality-versus-throughput* trade, not a defect fix, so the code was
reverted to the decided policy (heuristic-only own lists, ids-only slabs) and the
variant stays available as a measured option.  Before choosing between them, the
control experiment is missing and is the next measurement: **what is the legacy
engine's own `no_incoming` at 100k?**  `check_lists` only exists for the flat graph,
so it needs a `MemGraph` variant.  If legacy also leaves a few hundred nodes without
incoming edges at this scale, the flat engine has no connectivity defect and the
remaining question is purely the recall/build trade above; if legacy is at zero, the
append/shrink engine needs fix (2) (unconditional admission of the nearest backlink
target, the `always_admit` shape from ranked mode).

## 3e. Control experiment: the invisible nodes are normal for HNSW here, and the gate must change

`MemGraph::list_checks` mirrors `flat_graph::check_lists`, so the legacy engine now
reports the same structural numbers.  Same 100k dim-128 build, same seed:

| engine | build | `no_incoming` | `reachable` | fingerprint |
|---|---|---|---|---|
| legacy (exact re-prune) | 21.54 s | **384** (0.38%) | 99 616 | `e4aaa8fa5d6b7f25` |
| flat (append/shrink) | 20.90 s | **519** (0.52%) | 99 477 | `30db11c54b7edd31` |

Three corrections follow, and they matter more than the code change:

1. **The flat engine has no connectivity defect.**  The legacy engine -- the shipped,
   exact policy -- also leaves 384 nodes without an incoming edge at this scale and
   configuration (`m = 16`, `ef_construction = 64`, 100k rows).  The flat engine is
   135 nodes (0.14 percentage points) worse; that is a policy delta, not a bug.
2. **The gate "zero nodes without an incoming edge" is wrong as written.**  It came
   from the 1000-node in-memory harness, where it holds; at 100k it does not hold for
   either engine.  The acceptance criterion for `workers > 0` must therefore be
   *relative to the same-policy single-worker build*: `no_incoming(W)` and
   `reachable(W)` within a small tolerance of `no_incoming(1)`/`reachable(1)`, which
   is exactly the same-policy comparison the recall gate already uses.  Absolute
   connectivity stays useful as a *reported* number (and as a smoke check for the
   arena), not as a pass/fail.
3. **Fixes (1) and (2) are quality knobs, not defect fixes.**  Backfill measured as
   +51% build for +9.5 recall points at ef 40 on this dataset and -93 invisible nodes;
   unconditional admission of the nearest backlink would be a similar knob.  Both
   belong to the same decision, to be taken on 1M BIGANN where the recall effect is
   measurable at the operating point (99.4% at ef 160), not on this dataset.

Also worth recording: `max_len == 32 == m0` in both engines, and `self_links == 0`,
`duplicates == 0` in both, so the structural invariants that *must* hold do hold.

## 3h. BIGANN 1M operating point: three configurations measured (host 121 back up)

Same table (`items_1m`, dim 128), host, pinned seed and release build throughout:

| config | build | plan (search) | apply (backlinks) | `no_incoming` | recall@10 ef 10/20/40/80/160 | p50 ef 160 |
|---|---|---|---|---|---|---|
| legacy (exact re-prune, unconditional backfill) | 352 s | 228.3 s | 88.5 s | (n/a on BIGANN) | 78.5 / 87.3 / 94.3 / 97.7 / 99.4 | 2.864 ms |
| flat, `build_backfill = 0` | **178 s** | 142.2 s | **23.3 s** | 16 | 74.2 / 83.0 / 91.3 / 96.6 / 99.1 | **2.541 ms** |
| flat, `build_backfill = 1` | 308 s | 163.9 s | 130.8 s | **3** | 76.9 / 87.2 / 93.7 / 97.2 / 99.4 | 2.824 ms |

Three conclusions, all measurable rather than argued:

1. **flat + backfill = 1 dominates the shipped engine**: 1.14x faster (308 vs 352 s) with
   recall within 0.6 points at every ef and better connectivity (3 vs unknown; 384 was
   the legacy figure at 100k), i.e. the earlier "legacy is the quality reference" framing
   no longer holds at the operating point;
2. **flat + backfill = 0 is the throughput option**: 1.98x faster than legacy (178 s) but
   3-4 recall points lower at ef 10-40 (74.2 vs 78.5 at ef 10), recovering to within 0.3
   points by ef 160 -- and it has the *best* query latency (p50 2.541 ms vs 2.864);
3. **the backfill trade is dataset-dependent in both directions**, now on three datasets:
   it helps BIGANN dim-128 (+2.7/+4.2/+2.4 at ef 10/20/40 over `bf=0`), helped the hard
   100k dim-128 set (+9.5 at ef 40), and *hurt* the easy 1M dim-16 set (-20 at ef 40).
   The default therefore stays 0 (the decided policy, and the throughput choice), with the
   knob documented as the recall lever for datasets like BIGANN.

For the parallel plan this is the useful part: both flat configurations are well inside
the 4-worker build target (178 s and 308 s single-core against a <=160 s target at 4
workers), so the acceptance number is about worker scaling, not about squeezing the
sequential engine -- and if recall at low ef is the binding constraint, `bf=1` buys it
back inside the same budget.

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

## 4b. Deferred to future testing

The refinement "decide every eviction by measuring distances, with no cached
`list_dists`/`list_masks` anywhere" is recorded as an **unscheduled** idea in
`.design/future/hnswsq_backlink_measure_on_demand.md` (with its gains, costs, the
three-configuration micro-benchmark that would settle it and its acceptance
criteria).  It is not part of this plan: T6's decided policy is ids-only
append/shrink, which already removes the metadata from the arena, and the
measure-on-demand question is whether the *single-backend* path should follow
later.

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
