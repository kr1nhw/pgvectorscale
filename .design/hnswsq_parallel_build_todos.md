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
| M3 storage swap (region by region) | **complete** | all 8 regions in the chunk (`vectors`/`ids`/`lens`/`levels`/`tids`/`clamped`/`published`/`slab_off`); gate bit-identical after each step (§3j) |
| M3 step 5n: crash pinned to `index_build_range_scan` | **one call** | stages 1-4 clean with a worker running; 5-7 crash; the delta is the scan |
| M3 step 5m: leader teardown fixed, crash now in the worker's inserts | **leader verified** | stage 9 clean with 2 workers; stage 0 dies in a worker after ~3 s |
| M3 step 5l: crash bisect | **worker startup, not worker code** | a do-nothing worker crashes too; the same flow works in the pgrx test cluster |
| M3 step 5k: SQL-callable parallel build + first run | **crashes in the worker** | leader-only run is clean (3 ms); worker path segfaults |
| M3 step 5j: the callback, the scan loop, the wired entry | done (compile-verified) | two workers launched, entered, and reached the relation open |
| M3 step 5i: `worker_build_state` over the shared arena | done | from `BuildParams` + the arena; `BuildState` is crate-visible |
| M3 step 5h: params carry the backlink policy | done | a worker cannot silently build on another policy |
| M3 step 5g: params carry the dimension; meta-page ordering found | done | worker codec is derivable; leader must write meta before launching |
| M3 step 5f: level from a raw `ItemPointerData` | done | block hi/lo decoding pinned, incl. blocks > 65535 |
| M3 step 5e: one insert body, two level sources | done | `flat_insert_at_level` + `level_seed`; gate bit-identical |
| M3 step 5d: post-join entry promotion | done | `promote_best_entry`; workers never promote |
| M3 step 5c: shared scan descriptor in the toc | done | `table_parallelscan_*`; one descriptor every worker reads |
| M3 step 5b: build parameters + verified toc magic | done | `BuildParams` round-trips; `PARALLEL_MAGIC` is measured, not remembered |
| M3 step 5: parallel worker entry, cross-process verified | done | workers load the library, attach the leader's arena, touch the shared cursors |
| M3 step 4i: deterministic level table | done | `levels.rs`; levels drawn up front so worker scheduling cannot change the graph |
| M3 step 4h: two threads build one shared graph | done | `SharedGraph`; entry-cursor bug found; the whole engine runs concurrently |
| M3 step 4g: engine runs behind `Locking` | done | `apply_flat` takes `Locking::{SoleWriter, Locks}`; same heuristic either way |
| M3 step 4f: shared-write + node-lock path | done | `set_list_concurrent`/`set_list_locked`; contention test (no torn lists) |
| M3 step 4e: `FlatGraph` over a shared arena | done | cursors read the segment (`in_arena`); 2 latent cursor bugs surfaced by the shared path |
| M3 step 4d: shared cursors (`ArenaState`) | done | packed claim CAS, contiguous watermark, entry + rendezvous; 4 tests |
| M3 step 4c: arena in a `shm_toc` segment | done | `SharedArena::allocate`/`attach` by key; `pg_test` round-trips a real dsm segment |
| M3 step 4b: node locks as LWLocks | done | `NodeLocks` over `RwLock`s *or* a tranche we register at runtime; 2 `pg_test`s |
| M3 step 4a: chunk over shared memory | done | `Chunk` owns a `Vec<u64>` *or* borrows a segment (`attach`); relocation asserted by test (§3j) |
| incremental exact-match probe test | flaky, not ours | `pg_test_hnswsq_incremental_empty_start_sq8_provisional`: 1 failure in one grouped run, clean on rerun (§3k) |
| suite has 2 pre-existing red IVF tests | not ours | `ivf::options::tests::pg_test_ivf_options_{defaults,custom}`: no default opclass exists (§3j) |

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

**Baseline recipe (pin this exactly; the fingerprint is table-specific).**  Table
`t100kd128u` in `t100kdb` -- *not* `t100kd128` (same 100k/dim-128 shape, different
rows: that one gives flat `eea4810ef4123d1a` / `no_incoming=498`) and *not* `t100k`
(dim 16, fully connected: `no_incoming=0`, flat `3794aa11c860aba0`).  The in-memory
path requires a non-zero build budget, so the GUC line must include
`maintenance_work_mem`; without it the build silently takes the disk path
(`disk_mode=true`, `checks(...)` all zero, fingerprint unrelated):

```sql
SET maintenance_work_mem = '2GB';   -- else disk_mode=true and the gate is meaningless
SET hnswsq.build_stats = 1;
SET hnswsq.build_seed = 20240912;
SET hnswsq.build_engine = 1;        -- 0 = legacy MemGraph, for the side-by-side
CREATE INDEX <idx> ON t100kd128u USING hnswsq (embedding vector_l2_ops)
  WITH (storage_layout = 'plain', m = 16, ef_construction = 64);
```

Legacy baseline on the same table: `e4aaa8fa5d6b7f25`, `no_incoming=384`,
`reachable=99616`.  Both engines reproduce these exactly on repeated runs, so any
drift means a real change, either to the graph or to the gate.

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

## 3i. Storage swap: the route that does not ripple (Route C), and why the others do

The swap must not change any signature the engine, the driver or the tests use, or it
becomes the "half-ported hot file" the side-by-side approach exists to avoid.  Three
routes were considered:

* **Route A -- `FlatGraph<'a>` with slices.**  Replace the eight `Vec` fields with
  `&'a mut [T]` slices taken from the chunk.  Zero indirection change, but the lifetime
  parameter ripples through `search_layer_flat`, `select_neighbors_flat`, `plan_flat`,
  `apply_flat`, `flat_insert`, `FlatEngineState`, every test helper and the globals that
  hold them -- a wide, compile-iteration-heavy change.
* **Route B -- a storage trait with two implementations.**  The arena cannot hand out a
  borrowed `&[u32]` without leaking a lock guard, so the trait's read methods would have
  to be copy-into-buffer, which *changes* the hot search/selection loops (and their
  allocation behaviour) -- the one thing the flat layout was built to fix.
* **Route C -- keep the field list, replace the backing (recommended).**  `FlatGraph`
  holds **one** `Chunk` plus the `ArenaLayout` offsets and two cursors (`nodes`,
  `slabs`), and every existing accessor is implemented over the chunk:

  | today | after |
  |---|---|
  | `levels: Vec<u8>` | `chunk.region_bytes(&layout.levels)[..nodes]` |
  | `tids: Vec<ItemPointer>` | `chunk.region_slice::<ItemPointer>(&layout.tids)[..nodes]` |
  | `clamped`, `published_flags: Vec<bool>` | `chunk.region_bytes(..)[..nodes]` (one byte per node; `bool` only at the edges) |
  | `vectors: Vec<u8>` | `chunk.region_bytes(&layout.vectors)[..nodes * stride]` |
  | `slab_off: Vec<u32>` | `chunk.region_u32(&layout.slab_off)[..nodes]` |
  | `lens: Vec<u16>` | `chunk.region_u16(&layout.lens)[..slabs]` |
  | `ids: Vec<u32>` | `chunk.region_u32(&layout.ids)[..slabs * cap]` |

  `Chunk` needs read-only twins of the three mutable accessors (`region_bytes`,
  `region_u32`, `region_u16`, `region_slice`), which is mechanical.  Consequences:

  * **no signature changes anywhere** -- `push_node`, `try_push_node`, `claim_slot`,
    `publish`, `set_list`, `neighbors`, `copy_neighbors`, `vector`, `watermark`,
    `check_lists`, and therefore the engine, the driver and the tests all compile as-is;
  * growth mode (`usize::MAX` budgets) becomes "a chunk sized by the caller"; the
    prototype keeps a grow-mode path by reallocating a bigger chunk and copying, which is
    only used by tests and by the non-parallel driver -- or, simpler, `new()` allocates a
    chunk sized by a default cap and `with_limits` by the budget, dropping the growing
    behaviour the arena never uses;
  * `heap_bytes()` becomes `layout.total_bytes` (plus the chunk's own overhead), so budget
    accounting gets *more* accurate than the `Vec::capacity` sum it reports today;
  * verification is already in place: the 100k and 1M builds must reproduce their
    fingerprints (`30db11c54b7edd31` at 100k, `7ff861f019ba8dc3` at BIGANN 1M, both with
    `build_backfill = 0`), and `check_lists` must report the same numbers.  A swap that
    changes either has changed behaviour.

**Order of work for the next session** (each step compiles and tests on its own):
1. add `Chunk`'s read-only accessors + a couple of unit tests;
2. replace `FlatGraph`'s fields with `chunk` + `layout` + cursors, keeping the accessor
   bodies as thin wrappers (this is the one wide-ish edit, and it is confined to
   `flat_graph.rs`);
3. run `cargo pgrx test pg18 flat_` (25 tests) and the two fingerprint builds;
4. only then allocate the chunk from `shm_toc` and swap `NodeLocks` for LWLock tranches.

## 3j. Storage swap step 2, and two pre-existing red tests

The chunk conversion goes in one region at a time, each step gated on the 26
flat/arena unit tests plus the 100k in-memory fingerprint on `t100kd128u`
(`30db11c54b7edd31`, `no_incoming=519`, `reachable=99477`; recipe in §3c).  After
each step the fingerprint must be **bit-identical** -- the swap changes where the
bytes live, never which graph is built.

| step | region moved to `Chunk` | gate |
|---|---|---|
| 1 | `vectors` (encoded node vectors) | identical, 22.84 s |
| 2 | `ids` (`slabs * cap` neighbour ids) | identical |
| 3 | `lens` (slab lengths, + new `slabs_used` cursor) | identical |
| 4 | `levels` (node levels, + new `nodes_used` cursor; `len()` reads it) | identical |
| 5 | `clamped` + `published` (per-node flags, byte regions) | identical |
| 6 | `tids` + `slab_off` (heap TIDs, layer-0 slab index) | identical + integration suite |

Two consequences worth naming:

* `lens` could not simply move: it was both the length store *and* the slab
  cursor that `slab_base` counted through, so `slab_usage`/`is_full`/`try_push_node`
  all read a `Vec` length that is 0 in chunk mode.  A `slabs_used: usize` field is
  now the single authority in both modes, with the `Vec` kept in lockstep only
  while grow mode still owns the storage.  The `ids`/`lens` `Vec` reserves in
  `with_limits` are gone, saving up to `max_slabs * cap * 4` bytes of dead
  allocation.
* `levels` is the same shape of problem one level up: `len()` *was*
  `levels.len()`, and the node cursor is what the id stream, the capacity budget
  and the writeout all key off.  `nodes_used: usize` is now authoritative in both
  modes, advanced in `claim_slot` before anything else can observe the slot, with
  `level()` reading the chunk region and `slab`/`slab_base` going through that
  accessor.  Grow mode keeps `levels`/`lens` `Vec`s exactly in lockstep (asserted),
  so both modes answer identically while the conversion is in flight.
* the two per-node flags are byte regions and need no new cursor (`nodes_used`
  bounds them).  They are the one place where the arena leans on something other
  than an explicit write: the chunk arrives zeroed, so a freshly claimed node
  already reads "unclamped, unpublished" -- and publication is what makes a node
  observable, which is exactly the state the watermark relies on.  `claim_slot`
  therefore writes both zeros anyway, so correctness never rests on allocation
  zeroing.
* `tids` + `slab_off` close the swap, and each had a trap that a mechanical
  rewrite would have walked into:
  * **an all-zero `ItemPointer` is not the invalid one.**  It is block 0 /
    offset 0 -- a real-looking pointer -- so a claimed slot cannot inherit
    "unwritten" from the chunk's zeroed memory the way the flags can.  `claim_slot`
    writes `ItemPointer::new_invalid()` into the region explicitly.
  * **`slab_base` was bounded by `slab_off.get(id)`**, i.e. by a `Vec` that is empty
    in chunk mode; the region, by contrast, is `max_nodes` long, so a length-based
    bound would happily resolve slabs for unclaimed ids.  It is now bounded by
    `nodes_used`, with unit tests for both an unclaimed id (`slab_base(7) == None`)
    and claimed ones.
  * `tids` is also the one region the fingerprint gate cannot see: the fingerprint
    covers id/layer/list structure, so a broken heap TID would pass it and corrupt
    results silently.  The `access_method::hnswsq` integration tests (which build and
    scan) are the gate for it -- that is why step 6 ran them as well.

### 3j.1 Making the chunk relocatable (M3 step 4, first half)

A parallel build allocates the arena once, in the leader, from `shm_toc`, and every
participant -- worker and leader alike -- addresses those bytes at whatever address
its own mapping lands on.  For that to be sound the arena must be **relocatable**:
no absolute pointer may be stored inside the chunk, only offsets and plain data.

`Chunk` now expresses this directly instead of being a `Vec<u64>` that merely
*claims* to be a prototype of a segment:

* `Backing::Owned(Vec<u64>)` -- the local path; this handle frees on drop.
* `Backing::Borrowed { words, count }` -- built by `unsafe Chunk::attach`, which
  checks 8-byte alignment and takes the caller's word that the segment is live,
  large enough, zero (or already a graph), and not concurrently written except
  through the arena's locks.  An attached handle frees nothing, so the `Drop` of a
  worker's view can never free the leader's segment.

The property is asserted rather than assumed: `the_arena_is_valid_at_any_mapping_address`
builds a small graph, **copies the chunk's bytes into a fresh allocation elsewhere**
(asserting the address really differs, so the test cannot pass vacuously), attaches
a handle there, and requires every accessor -- vectors, ids, lens, levels, slab_base,
tids, flags, `len`/`slab_usage` -- to answer identically, then writes through the
moved arena and checks the owner sees it.  That is exactly what a dsm segment does
to a chunk, and it is why `ItemPointer` (block/offset, not a pointer) is safe to
store while e.g. a `Vec` would not be.

### 3j.2 Node locks as LWLocks (M3 step 4, second half)

`NodeLocks` now serves the same `read`/`write` API over either backing: the
prototype's `RwLock`s, or LWLocks that a parallel build places in its own segment.
The guards became an enum internally (`GuardBacking`) because an `RwLock` guard
releases by dropping while an LWLock has to be released explicitly -- so
`NodeReadGuard` grew a `Drop` it did not need before.

Two decisions worth recording, both from PostgreSQL's rules rather than preference:

* **`LWLockNewTrancheId` + `LWLockRegisterTranche`, not
  `RequestNamedLWLockTranche`.**  The latter only works while shared memory is being
  set up -- i.e. from `shared_preload_libraries` -- and hnswsq is loaded on demand,
  so a named tranche is not available to us.  Registering a tranche id at runtime and
  `LWLockInitialize`-ing our own locks in our own segment is the route parallel
  index builds use.  The registered name must be `'static`, since PostgreSQL keeps
  the pointer (it shows up in `pg_locks`) rather than copying the string.
* **A shared arena cannot grow.**  `grow_to` still works on the `RwLock` backing, but
  with LWLocks it asserts: a segment's size is fixed at `shm_toc` allocation time, so
  the node budget must come from `plan_capacity` before any worker starts, and a lock
  array that is too small is a setup bug, not a runtime condition.

The LWLock path needs a live backend (acquire/release touch `MyProc`) and LWLocks are
per-process, so contention cannot be tested from two threads in one backend and is
deliberately left to the parallel build.  What the two new `pg_test`s do assert:
tranche registration, shared and exclusive acquire/release (re-acquiring in a second
pass, which would *hang* rather than fail if a release were missing), and the
no-growth rule.

**Gotcha found here, worth remembering for any future `#[pg_test]`:** a test module
gated `#[cfg(test)]` registers the Rust test but never creates its
`tests.<name>()` SQL function -- the build script emits that SQL and compiles the
crate *without* `cfg(test)`, so the failure is "function tests.foo() does not exist"
at run time.  The module must be gated
`#[cfg(any(test, feature = "pg_test"))]` and carry `#[pgrx::pg_schema]`.

### 3j.3 The arena inside a segment (M3 step 4, third piece)

`SharedArena` is what a participant actually holds: the shared header, the chunk and
the node locks, all three living in a `shm_toc`.  The leader calls
`SharedArena::allocate(toc, stride, cap, nodes, slabs, tranche)` once; every worker
calls `unsafe SharedArena::attach(toc)` and gets the same three by **key**, never by
pointer -- which is the whole point, since a worker's mapping address differs from
the leader's.

* `ArenaHeader` is `#[repr(C)]` POD carrying the region map and capacities, so the
  two ends of a segment cannot disagree about the shape.  `Region`/`ArenaLayout` are
  `#[repr(C)]` now too: they cross a process boundary, so their field layout must be
  fixed rather than whatever the compiler prefers.
* `attach` uses the new `shared_locks_in_segment`, which does **not** initialize the
  LWLocks: re-running `init_shared_locks` would reset a lock a peer may be holding.
  Initialization is the leader's job, once.
* The three toc keys are fixed constants, and the segment magic is fixed rather than
  random -- leader and workers are the same binary, and a wrong toc fails loudly at
  the first key lookup instead of silently reading garbage.
* `SharedArena::segment_bytes` gives the leader its payload estimate for
  `shm_toc_estimate_chunk`; it deliberately excludes the toc's own key/alignment
  overhead, which `shm_toc_estimate_keys` accounts for separately in the driver.

### 3j.4 Shared cursors (`ArenaState`)

The cursors a parallel build shares now live in their own toc region, separate from
`ArenaHeader`: the header is written once by the leader and read by everyone, while
these are touched by every worker for every node.  Fields: the claim cursor, the
watermark, the entry point, the rendezvous counters, and a failure flag.

Three decisions the tests pin down:

* **One packed cursor, not two counters.**  `cursor` holds `(nodes << 32) | slabs` and
  moves in a single CAS, so a claim can never reserve a node without its slabs (or
  the reverse).  Two separate `fetch_add`s would leak one budget on every collision,
  and `plan_capacity` sizes the segment tightly enough that a leak matters.  Refusal
  is also side-effect free: the budgets are checked before the CAS, so an exhausted
  arena is not slowly walked past.
* **The watermark advances only over a contiguous prefix from zero.**  A worker that
  finishes out of order cannot expose a node whose vector is half-written.  The unit
  test sets flags 2,3,4 and asserts the watermark stays 0 -- i.e. contiguity is from
  zero, not from the lowest published id -- then fills the gaps and watches it jump.
* **Workers never initialize shared state.**  `attach` deliberately does not
  re-initialize the cursors or the locks; the leader does both once, in `allocate`.
  A worker that reset the claim cursor would hand out duplicate ids.

The failure flag exists because a worker cannot longjmp into the leader: it sets
`failed`, exits cleanly, and the leader re-raises the error itself.  That is the
hook the driver's error path needs.

### 3j.5 `FlatGraph` over a shared arena

`FlatGraph::in_arena(&SharedArena)` builds a graph whose storage and cursors both come
from the segment: `len`/`slab_usage`/`is_full`/`watermark`/`claim_slot`/`publish` go
through `ArenaState` (packed claim CAS, shared watermark), and every per-node region is
addressed in the shared chunk.  The `Vec` fields remain the single-builder path, and
`Chunk::region_bytes_concurrent` is the documented escape hatch the parallel write path
needs: several workers write sibling bytes of one chunk at a time, so there is no
`&mut Chunk` to share, and the soundness argument is the division of labour (a node's
bytes are written only by the worker that claimed it, and readers hold its node lock).

**Two latent bugs the shared path surfaced immediately**, both of the same kind -- a
cursor with two homes (the field and the segment):

* `slab_base` bounded `id` by the *field* `nodes_used`, which is 0 once the cursors
  live in the segment, so every lookup returned `None` (unreachable slabs rather than
  wrong ones -- the safe direction, but wrong);
* `try_push_node`'s budget pre-check read the field too, so it would report "room"
  until `push_node`'s claim panicked on exhaustion.  `false` is the driver's spill
  signal, so that had to become the real answer.

Both are fixed by reading through the cursor methods, and the audit for *every*
remaining raw field read is in the diff: the only ones left are in the local arms of
the two `match`es and behind `state.is_none()` guards.

**A third bug was mine, not the design's:** the same patch dropped
`self.push_node(...)` from `try_push_node` entirely, so it returned `true` without
pushing.  Two pre-existing tests (`fixed_capacity_reports_full_instead_of_growing`,
`planned_capacity_admits_more_nodes_than_its_estimate`) failed on it immediately --
which is the argument for keeping them: a "true means it worked" path with no
assertion of its own would have shipped that silently.

### 3j.6 The backlink write path, and what threads can and cannot test here

The backlink step is the only place a worker writes a node it does not own, so it is
the only place the node lock has to do real work.  It now has its own path:
`FlatGraph::set_list_concurrent` (unsafe: the caller is the only writer of that slab)
and `set_list_locked` (takes the node's write lock, then calls it).  Writes go through
typed concurrent views of the chunk (`region_u32_concurrent`/`region_u16_concurrent`),
and the length is written *after* the ids, so a reader can never see a length pointing
at ids that were not written yet.

**A thread test cannot use the arena's real locks.**  The first version of the
contention test took `arena.locks()` -- LWLocks -- from two threads, and it failed: an
LWLock is a *per-process* lock, so two threads in one backend do not contend for it the
way two worker processes do (the second acquire looks like a recursive acquire by the
same process).  The test now uses `NodeLocks::new` (thread-correct `RwLock`s) while
still writing into the real dsm segment, which is what it can honestly check: the
protocol and the concurrent write path.  **The cross-process case is only testable in
the parallel build itself**, and that is now the second thing the worker sweep has to
prove, after the speedup.

`FlatGraph` deliberately has no blanket `Send`/`Sync` impl yet: a chunk-backed graph
over a segment is shareable, but a grow-mode graph owns `Vec`s that would race, so the
impl has to be attached to the shared construction (`in_arena`) rather than to the
type.  Doing it as a blanket impl for a test's convenience would have made the
grow-mode graph quietly unsound.

### 3j.7 The engine runs behind `Locking` (M3 step 4, seventh piece)

`apply_flat` and `backlink_flat` no longer care whether they are the only writer.
`Locking::{SoleWriter, Locks(&NodeLocks)}` is the parameter, and the two writes a
build makes go through it:

* `write_own` -- the node just claimed, so no lock is needed.  `SoleWriter` uses the
  safe `set_list` (which is the *only* path that works for a grow-mode graph, whose
  `Vec` would reallocate); `Locks` uses the concurrent write.
* `write_target` -- a backlink into a node the caller does not own.  `SoleWriter` still
  uses the safe path; `Locks` takes the node's write lock.

Keeping one function with a parameter, rather than a second copy of `backlink_flat`,
means the single-builder and parallel builds cannot drift apart in *policy* -- they run
the identical heuristic, and only the exclusion differs.  That matters here because the
append/shrink policy is exactly what the fingerprint gate checks.

Entry promotion is deliberately **not** done by a worker: every insert would serialize
on the entry, and promotion only matters once the graph stops growing.  `Locking::Locks`
skips it and the driver promotes after the workers stop, which is also the first moment
a searcher may observe it.

### 3j.8 Two threads build one shared graph (and the entry cursor)

`SharedGraph` is the handle workers share: `Send`/`Sync` are asserted on this *type*,
which is only constructible from a shared arena, rather than on `FlatGraph` -- a
chunk-backed graph over a segment is shareable, a grow-mode graph owns `Vec`s that
would race, and a blanket impl would silently permit the second.

`two_threads_run_the_engine_over_one_arena` then runs the *whole* engine -- search,
plan, apply, backlink, publish -- from two threads over one arena, and asserts through
the structure gate: 40 nodes handed out with no duplicates, the watermark past all of
them, no self-links, no duplicates, capacity respected, and at least half the graph
reachable from the entry.

Four things this cost, all worth keeping:

* **The entry point was the third cursor with two homes.**  `entry()`/`promote_entry`
  read the local field, which is `None` forever in shared mode, so every worker saw an
  empty graph, planned an empty list and published a node with *no edges at all*.  The
  structure gate caught it in the plainest possible terms: `nodes=40 published=40
  max_list_len=0/4 no_incoming=40 reachable=0`.  Both accessors now go through
  `ArenaState`, and promotion writes the level before the id so a reader cannot see an
  entry without its level.
* **The first node must be an entry before any worker starts** -- otherwise there is
  nothing to search from and *every* node is born unlinked.  The test now seeds it and
  sets `start_nodes`, which is precisely the rendezvous the driver formalises.
* **PostgreSQL FFI may not be called from spawned threads** (pgrx enforces it:
  "postgres FFI may not be called from multiple threads"), and the arena's locks *are*
  FFI -- LWLocks.  A thread is not a substitute for a worker process.  The test
  therefore locks with `NodeLocks::new` (`RwLock`s) while storing into the real shared
  segment; the cross-process LWLock case stays the worker sweep's to prove.
* worker panics lose their message inside a scoped thread (the hook writes to the
  backend's stderr), so each worker catches its own and hands the payload back.  That
  is what turned "a scoped thread panicked" into the two real diagnoses above.

### 3j.9 Deterministic levels for a parallel build (`levels.rs`)

In a single-builder build, "the i-th node" and "the i-th random draw" coincide, so
drawing a level at insert time is deterministic given the seed.  That stops being true
with several workers claiming ids concurrently: which worker draws next depends on
scheduling, so the level stream -- and therefore the graph, and the fingerprint -- would
change from run to run, and the fingerprint gate could not be used for a parallel build
at all.

`LevelTable` therefore draws every level up front, in the leader, from the pinned
`build_seed`: the levels become a function of the row count alone.  It also exposes
`slabs()` (`sum(level + 1)`, what the arena's slab budget has to cover for those nodes)
and `extend()`, which *replays* rather than redraws, because a worker may already be
using the levels already handed out.

The tests pin the properties the driver will rely on: same seed gives the same table and
a different seed does not; levels stay within `max_level` and keep the geometric shape
(most nodes level 0, some above); `level(id)` past the table reads 0 rather than
panicking, which is what a search over a partially claimed arena needs; and
`extend(n)` equals `draw(n)` for the same seed, so extending cannot invalidate levels
already in flight.

Note the gap this leaves, deliberately: nothing consumes the table yet.  The
single-builder path keeps drawing at insert time (bit-identical for the same seed, and
verified by every fingerprint gate so far), and the driver switches the workers to the
table.  When it does, the check is that a table-driven build reproduces the
single-builder fingerprint -- which is only possible *because* the levels are drawn up
front.

### 3j.10 The contention test flake: instrumented, not resolved

`node_locks_serialize_concurrent_list_writes` failed about one run in three under
full-suite scheduling with the old shape, and passes when run alone.  The panic was
inside a scoped thread, so the framework could only report "a scoped thread panicked"
and the payload -- which would have said *which* assertion fired and *what* it saw -- was
lost.

That is now fixed rather than worked around: readers **record** what they observe
(`round, len, ids`) into a shared list and the main thread asserts on it, while the
writers still panic on purpose.  The split is deliberate -- if a future failure reports
observations, the read path saw a non-exclusive lock; if it reports nothing *and* a
thread panicked, the culprit is a writer.

Between the two loops after instrumenting, 14 consecutive full-suite runs passed (the old
shape failed 2 of 6).  That is evidence the failure is window-dependent and that the
instrumentation shifted the window, **not** evidence that it is gone, so the test is left
enabled rather than `#[ignore]`d: the suite keeps exercising it, and the next occurrence
reports the observation instead of losing it.

Unresolved and worth stating plainly: if the tear is real, then a plain store through
`region_u32_concurrent` paired with a plain load through `region_u32` under
`std::sync::RwLock` is not ordered as the test assumes, which would matter to the
parallel write path.  The alternative is that the old test's own bookkeeping was at
fault.  Neither has been demonstrated, and the parallel build -- which locks with the
arena's LWLocks across *processes* -- is where it will matter, so the worker sweep is the
place to settle it.

### 3j.11 Corrected: a parallel level assignment must key on the row, not the ordinal

Building the driver's scan exposed a flaw in the previous round's design.  An
ordinal-indexed `LevelTable` is enough for a single-builder build, but **not** for
`table_index_build_scan`: which worker sees which row, and in which order, depends on
scheduling, so "the i-th row processed" is no more stable than "the i-th random draw" --
the table would merely move the nondeterminism from the RNG to the scan.  The level
assignment has to key on something intrinsic to the row.

`levels::level_for_tid(seed, ml, max_level, tid)` is that: a pure function of the pinned
seed and the row's heap TID, so the same row yields the same level whichever worker gets
it, and the whole assignment becomes a function of the table's contents.  That is what the
fingerprint gate -- and the acceptance test that `workers = 0` is bit-identical -- will
need from a parallel build.

Two details the tests pin down, both about *not* being naive:

* the seed is mixed with an FNV-1a hash of the TID (plus one avalanche round) rather than
  offset by it.  Sequential TIDs are adjacent numbers; feeding those to a generator
  correlates neighbouring rows, which shows up as long constant runs or levels drifting
  together.  The test asserts the geometric shape survives *and* that the longest constant
  run stays short.
* processing order cannot change a level: the same rows in reverse order produce the
  identical assignment.  The ordinal table is kept for the single-builder path and for
  tests that want to assert a level directly, and both derive from the same
  `random_level`, so they agree whenever the order is the same.

### 3j.12 The driver's leader side: sized, allocated, and found by key

`driver.rs` holds the leader half, and it **works** as of this round: in a real parallel
context, `estimate_arena` -> `InitializeParallelDSM` -> `SharedArena::allocate` puts the
arena in the context's toc with the planned shape, `shm_toc_attach(PARALLEL_MAGIC, ...)`
returns exactly the pointer `InitializeParallelDSM` stored (so the pinned magic is verified,
not trusted), every region is findable by key through it, and a second handle built from
that toc sees the same arena.  `worker_attach` (`dsm_attach` + toc attach + by-key lookup)
is the other half, ready for the entry point.

Getting there took three corrections, and the record is worth keeping because two of them
were mine:

1. **The estimate is per allocation, not per total.**  `shm_toc_allocate` `BUFFERALIGN`s
   each allocation separately, so rounding up the summed regions once is short by up to
   eight bytes per region -- which surfaces as `ERROR: out of shared memory`, a hard failure
   for a rounding detail.  `SharedArena::allocation_sizes` is what the estimator consumes.
2. **The previous round's conclusion was wrong.**  It reported that the estimate "appears
   not to be honored at all".  Measuring says the opposite: `space_for_chunks` grew by
   exactly what we asked for.  The apparent contradiction (an instrumented variant passing
   while the plain one failed) was a red herring from that variant calling
   `InitializeParallelDSM` twice.
3. **`BUFFERALIGN` is not the whole charge.**  With per-allocation estimates, a real
   context reported `shm_toc_freespace` **56 bytes above** the aligned total the arena
   needed -- and the allocation still failed.  So each allocation also carries a small
   margin (`CHUNK_MARGIN = 64`).  Both are over-reservations by design: reserved-and-unused
   shared memory costs kilobytes, an under-estimate is a hard error.

The lesson worth carrying to the rest of the driver: PostgreSQL's own accounting is only
knowable by measuring against a real context, and when an estimate is involved, "it fits"
should be an assertion rather than an arithmetic belief.

### 3j.13 Cross-process: worker processes attach the leader's arena

The question no thread-based test could answer -- does a worker in *another process*
really see the leader's arena -- is now answered by a real parallel context:
`CreateParallelContext` -> `estimate_arena` -> `leader_setup` -> `LaunchParallelWorkers` ->
`WaitForParallelWorkersToFinish`, with two actual worker processes that load the extension,
enter `hnswsq_parallel_build_main`, attach the arena through the toc and touch the shared
cursors.  The assertions are the ones that matter: the leader's `workers_entered()` equals
PostgreSQL's `nworkers_launched`, and `active_workers()` is back to zero afterwards.

Two contract mistakes were found by running it, both of which guessing had gotten wrong:

* **the entry point's signature is `(dsm_segment *seg, shm_toc *toc)`**, not `(Datum)`.
  PostgreSQL attaches the segment itself and hands the worker both objects, so the first
  version -- taking a `Datum` and calling `dsm_attach` on it -- was treating a pointer as a
  handle and died with "could not attach the build segment".  A useful side effect: the
  worker never needs `PARALLEL_MAGIC`, since it is *given* the toc.  The constant and its
  verification stay for anything that has to re-find the toc (the writeout, say).
* **the library name PostgreSQL loads is versioned.**  pgrx installs
  `vectorscale-0.9.0.dylib`, which is what the extension's `module_pathname` points at, so
  passing the bare name to `CreateParallelContext` launched workers that died with
  `could not access file "vectorscale": No such file or directory`.  It is now derived
  (`concat!("vectorscale-", env!("CARGO_PKG_VERSION"))`) rather than hardcoded.

That is the whole shared-arena stack exercised across processes: `dsm` segment, toc by key,
`ArenaState` cursors, and the library load path.  What a worker still does *not* do is build:
the entry point attaches, counts itself and returns.  The scan and the insert loop are next,
and with them the first real measurement of the 1/2/4/8-worker sweep.

### 3j.14 The build parameters, and a toc magic that had to be measured

`BuildParams` is the contract the worker loop will read: rows, seed, heap/index oids, stride,
cap, `m`/`m0`/`ef_construction`, `ml`/`max_level`, distance type, precision, backfill and the
lock tranche, as `#[repr(C)]` POD with the `u64`s first and Oids as `u32` (the wire shape
should not depend on how pgrx wraps an Oid this release).  `publish` overwrites in place on a
second call rather than allocating again -- the estimate reserves room for exactly one copy.

Two corrections came out of writing the verification for it, and both are the kind that only
a *test* finds:

* **the pinned toc magic was wrong.**  `PARALLEL_MAGIC` was remembered as `0x50477c23`;
  `shm_toc_attach` returned NULL for it, and nothing else would have noticed, because the
  worker path no longer needs the magic at all (PostgreSQL hands the worker the toc).  The
  test now reads the magic off the live toc and requires the constant to match it, so it is
  `0x50477C7C` by measurement, and the value is verified on every run rather than recalled.
* **verification had quietly drifted away.**  An earlier round's "remove the duplicate test"
  step deleted the only assertions checking the magic and the by-key lookups, while the doc
  and the round report went on claiming they were verified.  Restoring them is why the wrong
  magic surfaced now instead of at the first worker sweep.

**Next: the scan**, and it is now scoped precisely.  `table_index_build_scan` is *not* bound
because it is `static inline` in `tableam.h`, and the same is true of
`table_parallelscan_estimate`/`initialize`, so the worker loop has to call the table AM
directly:
`relation->rd_tableam->parallelscan_estimate/initialize` and
`...->index_build_range_scan(table_rel, index_rel, index_info, allow_sync, anynulls,
is_validate, blockNum, callback, callback_state, scan)`.  `table_beginscan_parallel(Relation,
ParallelTableScanDesc)` *is* bound, and `IndexBuildCallback`'s signature is
`extern "C-unwind" fn(index, tid, values, isnull, tupleIsAlive, state)`.  The shared scan
descriptor has to be allocated in the toc so every worker sees the same one.

### 3j.15 The shared scan descriptor, and which scan API the worker actually needs

The leader now builds the table scan in the toc: `scan_bytes(heap, snapshot)` (a pure size
computation, so it can be reserved *before* `InitializeParallelDSM` -- which is why the
estimate is a separate step at all) and `leader_scan_setup`, which allocates the descriptor,
`table_parallelscan_initialize`s it, and inserts it under its key.  `scan_descriptor(toc)`
is the worker's side.  One descriptor for every worker is what makes the parallel scan hand
out *disjoint* block ranges: workers reading the same object cover the table once instead of
each scanning all of it.  The snapshot is copied into the descriptor by the initialize call,
which is why a worker needs nothing but the descriptor to start.

Verified leader-side with a real table and a real snapshot
(`the_leader_publishes_a_shared_scan_descriptor`, 3 driver tests green).

**A useful simplification for the worker loop.**  The plan was a slot-based scan, but
`table_scan_getnextslot` is `static inline` in `tableam.h` and therefore not bound -- and it
turns out not to be needed.  The build wants *values*, and `index_build_range_scan` (bound as
a `TableAmRoutine` field, with the full signature recorded above) hands them to a callback
directly.  So the worker scan is:

```
table_beginscan_parallel(heap, scan_descriptor(toc))
rd_tableam->index_build_range_scan(table_rel, index_rel, BuildIndexInfo(index_rel),
                                   allow_sync=false, anyvisible=false, progress=false,
                                   0, InvalidBlockNumber, callback, state, scan)
table_endscan(scan)
```

with each worker building its own `IndexInfo` from the index relation (a palloc'd one cannot
be shared, and it is deterministic from the relation anyway), and the heap/index relations
opened from the oids in `BuildParams`.  The callback is
`extern "C-unwind" fn(index, tid, values, isnull, tupleIsAlive, state)`, which is where the
vector for `claim_slot`/`plan_flat`/`apply_flat` comes from -- i.e. the next step is the
insert loop itself, no further ABI discovery expected.

### 3j.16 Who promotes the entry, and when

Workers never promote the entry: every insert would then serialize on one cache line, and
promotion only matters once the graph stops growing.  So the leader does it twice -- it seeds
an entry *before* launching (without one, searches find nothing and every node is born with
an empty list, a failure this project already produced once and caught only through the
structure gate), and it re-promotes *after* the join, because a higher-level node may have
appeared in between.

`driver::promote_best_entry` is the second half.  It walks ids below the **watermark**, not
below the claim cursor: a claimed-but-unpublished slot has a level but no list yet, and
promoting one would hand searches an entry that is not there.  That distinction is what the
new test pins, along with the fact that peer handles see the promoted entry (it lives in
`ArenaState`, not in one handle's fields).

Verified: 3 driver, 31 flat and 19 arena tests green.

### 3j.17 One insert body, two level sources

The insert loop does not need a new insert path: `flat_insert` already *is* the loop body
(claim, encode, plan, apply with backlinks, publish).  What the parallel path needs is a
different **level source**, so `flat_insert` now draws its level from the RNG and delegates
to `flat_insert_at_level`, which is the whole body.  A worker derives the level from the row
(`levels::level_for_tid`, seeded from `BuildParams.seed`) and enters through the same
function -- so the two paths cannot drift apart in anything except where the level came from,
which matters because the fingerprint gate has to be able to compare them.

`BuildState.level_seed` carries that seed, filled from `build_seed_value()` (the same GUC
`build_rng` reads; entropy mode draws one, which the parallel path never relies on since the
driver pins a seed).

Verified: 31 flat tests green and the **gate is bit-identical** -- flat
`30db11c54b7edd31` / `no_incoming=519`, legacy `e4aaa8fa5d6b7f25` / `384`.  That is the
correct expectation and the point of the change: the single-builder path still draws from the
RNG, so this is a behaviour-preserving refactor, and `flat_insert_at_level` is now the seam.

One small lesson from the patch: adding the field by regex hit `SampleState`'s literal as well,
and the compiler named the line.  Worth a glance at *every* insertion site when a struct
literal is built by pattern.

### 3j.18 Decoding the row's TID, and what the worker body still needs

An index-build callback holds a raw `ItemPointerData`, so `levels::level_for_item_pointer`
decodes it into the `(block, offset)` the level rule takes.  The packing is the trap: a block
number is `bi_hi << 16 | bi_lo`, and swapping them yields a plausible level for the *wrong
row* -- invisible until a table has blocks above 65535, i.e. not in any test someone writes
first.  The test therefore includes 65536, 70_000 and 1_048_576 on purpose, and checks that a
large block does not decode to its own low half.

**What the worker body still needs**, scoped by reading the code rather than guessing:

* its own `BuildState`.  The leader's construction is ~40 inline lines inside the build
  function (codec from `codec_for(&index_rel, &meta)`, `m`/`m0`/`ef_construction`/`max_level`
  from the reloptions, `ml` from the meta page, `pair_buf` from the dimensions), not a
  callable helper, so a worker either grows one from `BuildParams` plus a `table_open` of the
  index oid, or that construction is factored out of the build function first -- factoring is
  the cleaner of the two and is what the next round should do.
* the callback: `extract_vector(*values)` -> `preprocess_cosine` when the distance type is
  cosine -> the level -> `flat_insert_at_level`, mirroring the existing callback at
  `build.rs:~1545`.
* the `index_build_range_scan` loop around it, plus the leader's writeout after the join and
  `amcanbuildparallel` under `build_parallel`.

### 3j.19 What the worker must read from the index, and an ordering constraint

Sizing up the worker's `BuildState` by reading how the leader derives its own produced two
useful facts.

**Almost nothing has to be re-derived.**  `m`, `m0`, `ef_construction`, precision, distance
type, stride, `ml`, `max_level` and the seed all travel in `BuildParams` (that is what the
struct is for), so a worker builds its state from the parameters rather than from the index's
reloptions.  The one thing `BuildParams` was missing is the **dimension**: `stride = dim *
elem_bytes` and `elem_bytes` depends on the precision, so the dimension is not recoverable
from the stride alone.  It is now carried, and the driver tests still pass.

**The meta page is the exception, and it sets an ordering constraint.**  `ml` and `max_level`
come from the index's meta page, which the leader writes *during its own build*, so a parallel
build has to write block 0 and the calibration chain **before launching workers** -- otherwise
a worker would seed its state from a meta page that does not exist yet.  That is recorded on
`leader_setup`, where the driver will need it.

Still open for the worker body, and now precisely scoped: the `BuildState` literal has ~18
fields, so growing one from `BuildParams` means reading the remaining ones (`m0`, `stats`,
`reference_backlinks`, `backlink_mode`, ...) -- mechanical, but it is the next round's first
move rather than something to guess at.

### 3j.20 The worker's `BuildState`: the full recipe, and the three traps in it

`BuildState` has 21 fields; here is what each becomes for a worker, so the next round is
assembly rather than investigation:

| field | worker value |
|---|---|
| `codec` | `Codec::new(precision, params.num_dimensions)` -- both now in `BuildParams` |
| `dist_fn` / `distance_type` | from `params.dist_type` |
| `m`, `m0`, `ef_construction` | from `params` (never re-derived from reloptions) |
| `ml`, `max_level` | from `params` -- *not* the meta page, which avoids the ordering trap below |
| `budget_bytes` | the **arena's** byte capacity, not 0 |
| `mem_used` | 0 |
| `graph` | `MemGraph::new()` -- unused on the flat path, required by the struct |
| `pair_buf` | `DistBuf::new(num_dimensions)` |
| `stats` | `BuildStats::new()` (per worker; merging is M4) |
| `search_scratch` | `SearchScratch::new()` |
| `flat` | `Some(FlatEngineState { graph: <the arena's graph>, scratch, buf: FlatPairBuf::new(dim) })` |
| `reference_backlinks` | `false` |
| `backlink_mode` | from `params.backlink_mode` |
| `disk_mode` | `false` -- a worker must never take the disk path |
| `nrows`, `rng` | 0, seeded from `params.seed` (unused) |
| `level_seed` | `params.seed` |

Three traps found while deriving that, each now either fixed or recorded:

1. **`backlink_mode` was not in `BuildParams`.**  A worker would therefore have run whatever
   the default is regardless of the leader's choice -- and on a different policy it builds a
   *different graph*, which the fingerprint gate would report as an unexplained mismatch
   rather than as "the worker disagreed about the policy".  It is carried now.
2. **`budget_bytes` must not be 0.**  The single-builder path uses 0 to mean "go straight to
   disk", and a worker taking that branch would abandon the arena.  A worker has no byte
   budget of its own -- the arena's capacity is the limit, and `claim_slot` returning `None`
   is the real exhaustion signal -- so it gets the arena's size.
3. **`FlatEngineState` owns its graph, and a worker must not.**  Its `graph` field has to be
   the arena's shared graph (`FlatGraph::in_arena`), which means constructing the struct
   literally in `build.rs` (where it lives) rather than calling `FlatEngineState::new`, which
   would allocate a private graph and quietly build into it.

### 3j.21 The worker's state exists (and the visibility that blocked testing it)

`worker_build_state(params, arena)` builds a worker's `BuildState` from the parameters plus the
arena -- the field table from §3j.20, in code.  It reads no reloptions and no meta page (the
parameters carry the graph's shape, which is also what sidesteps the meta-page ordering
constraint), gives itself no byte budget (`budget_bytes` is the arena's size, because 0 means
"disk path" on the single-builder path), and constructs `FlatEngineState` literally so its graph
is the arena's shared one rather than a private allocation.

`flat_insert_at_level` now takes the `Locking` as a parameter, so the same insert body serves
the single-builder path (`SoleWriter`) and a worker (`Locks(arena.locks())`) with no second
copy of the policy.

**A visibility problem worth recording**, because it shaped the test: `BuildState` was a private
struct, so the driver could not name the type its own function returns -- and its fields are
private to `build.rs`, so a test in `driver.rs` cannot read `budget_bytes`, `disk_mode` or the
flat graph.  The type is now `pub(crate)` with private fields.  The test therefore asserts what
this side can see (the construction succeeds against index-shaped parameters and leaves the
arena untouched -- a state that claimed or published while being built would corrupt a graph the
leader may already have seeded), and the finer assertions want accessors on `BuildState` that the
driver needs anyway for the writeout; they come with it.

Also worth noting from the attempt: moving that test into `build.rs` by appending it to the end
of the file put it outside the test module and broke `mod mem_tests`'s gate -- the file's last
`}` is not the module's.  Rewriting that way was reverted rather than patched.

### 3j.22 The callback and the scan loop are in place

The last structural pieces exist: `parallel_insert_callback` (extract the vector, cosine
preprocess, level from the row, insert under `Locking::Locks(arena.locks())` -- the same insert
body as the single-builder path) and `parallel_worker_scan` (open heap and index, build this
worker's own `IndexInfo`, `table_beginscan_parallel`, then
`rd_tableam->index_build_range_scan` over the whole heap -- the *descriptor* is what divides the
heap between workers, not this call).  The worker entry point now reads the parameters and runs
the scan when they name a heap.

Running it produced the first real evidence: with two workers launched, the leader saw both
enter, and the failure came from a worker *inside* `parallel_worker_scan`, at `table_open`:

```
ERROR:  cannot open relation "par_build_test_idx"
```

**A test-harness constraint, not a driver defect**: `#[pg_test]` runs inside a transaction that
is rolled back, so the test's `CREATE TABLE`/`CREATE INDEX` are uncommitted -- and a parallel
worker gets a snapshot that cannot see uncommitted catalog rows.  The test is `#[ignore]`d with
that diagnosis in place.  Finishing it needs committed fixtures: create the table and index
outside the test transaction (harness setup SQL, or a script against a cluster that already has
such a table -- the local scratch database has `t100k` with an hnswsq index, which is exactly
the shape required).

The silver lining is that the error locates the boundary precisely: entry -> parameters ->
`worker_build_state` -> `parallel_worker_scan` -> relation open all execute in a worker process.
What remains untested is the scan itself and the inserts it feeds.

### 3j.23 The first parallel build ran, and crashed in the worker

`#[pg_test]` cannot drive this (its transaction is rolled back, so workers cannot see the
fixtures), so the driver is now reachable from SQL: **`hnswsq_parallel_build_debug(heap_oid,
index_oid, workers, dims, dist_type, budget_mb)`** runs a whole parallel build against a real
committed table and returns one line -- rows published, the structural checks, elapsed
milliseconds, and how many workers actually ran.  It is also the vehicle the worker sweep needs,
since it takes `workers` as an argument.  (Installing it against an already-created extension
needs the SQL wrapper made by hand: `CREATE FUNCTION ... AS '$libdir/vectorscale-0.9.0',
'hnswsq_parallel_build_debug_wrapper'`.)

Results against the scratch cluster's `t100k` + `t100k_idx` (100k rows, dim 16, `m=16`,
`ef_construction=64`, plain layout):

| workers | result |
|---|---|
| 0 | `published=1 slabs=1/67108855 checks(published=1 no_incoming=1 reachable=1 self_links=0 duplicates=0 max_len=0/32) elapsed_ms=3` |
| 1 | **server crash**, the backend terminated abnormally |
| 2 | same (the cluster was left in recovery) |

That is a useful split rather than a dead end: **the leader side is sound** -- sizing, allocation,
entry seeding, toc publishing and the destruction path all complete in 3 ms with a
well-formed single-node graph -- so the fault is in the worker path, which is the only part that
had never executed.  Candidates, in the order they should be eliminated:

1. `table_beginscan_parallel(heap, pscan)` in a worker: if the snapshot has to be *serialized*
   into the descriptor rather than referenced, a worker dereferencing the leader's snapshot is a
   segfault, and this is the first call that would do it.
2. `rd_tableam->index_build_range_scan` with the arguments as passed (`anyvisible=false`,
   `progress=false`, `0..InvalidBlockNumber`).
3. The callback's `extract_vector(*values)`: it assumes the index's key attribute is the vector,
   which is true for `t100k_idx` -- but if the scan hands over `values` for a different
   attribute ordering, this is where it would die.

The cheap bisect is to make the worker stop after (1), then after `BuildIndexInfo`, and so on,
because the harness already reports how far it got.

**Fix on the way here:** the index relation has to be opened with `index_open`, not `table_open`
(the leader-side attempt failed loudly with "This operation is not supported for indexes").

### 3j.24 The crash is in worker *startup*, not in the worker's code

`hnswsq.parallel_stage` (debug-only) makes a worker stop after N steps, which bisected the crash
in four runs:

| stage | what the worker does | result |
|---|---|---|
| 9 | returns from the entry point immediately | **crash** |
| 1 | opens the heap and index | crash |
| 2 | + `BuildIndexInfo` | crash |
| 3 | + `table_beginscan_parallel` | crash |
| 4 | + the worker's `BuildState` | crash |
| 0 | the whole build | crash |
| -- | `workers=0` (leader only) | clean, 3 ms |

So every stage crashes identically, including one where the worker touches nothing: **the fault is
not in the worker's code at all**. Three follow-ups, each eliminating a hypothesis cheaply:

* **arena size**: a 1 MB arena crashes the same way, so the large dsm segment at 64 MB is not the
  trigger.
* **parallel mode**: wrapping the context in `EnterParallelMode`/`ExitParallelMode` (which
  PostgreSQL's own callers always do, and which this code did not) changes nothing -- correct to
  add, but not the cause.
* **the worker's own path**: ruled out by stage 9.

What this leaves is how the context is *driven* from a SQL function, in this cluster.  The same
sequence -- `CreateParallelContext` -> `InitializeParallelDSM` -> `LaunchParallelWorkers` ->
`WaitForParallelWorkersToFinish` -> `DestroyParallelContext`, with workers successfully entering
the entry point -- already works inside the pgrx test cluster, which is the main difference
between the working and failing cases.

**Next diagnostic, and it needs the log**: the scratch cluster was started without `-l`, so the
postmaster's account of the crash was lost.  Restart it with a logfile and
`log_min_messages=debug1` (a crash names the signal and the process role), then rerun stage 9 --
that will say whether the worker dies before or inside `ParallelWorkerMain`, which is the
question no amount of further bisecting inside the worker can answer.

### 3j.25 Two crashes, found by reading the log rather than guessing

Restarting the scratch cluster with `-l` (it had been started without one, which is why the
previous round had nothing to read) made the postmaster's account of the crash decisive, and it
said something the code alone had not:

```
background worker "parallel worker" (PID 26432) exited with exit code 0
LOG:  client backend (PID 26430) was terminated by signal 11: Segmentation fault
```

The worker was fine; the **leader** was dying.  In the debug function the summary `format!` read
`arena.state()` *after* `DestroyParallelContext` -- which detaches the dsm segment, leaving the
arena handle dangling.  Reading the values before teardown fixes it, and the check is that
`stage 9` now completes with two workers:

```
workers launched=2 entered=0 failed=false published=1 slabs=1/67108855 \
  checks(published=1 no_incoming=1 reachable=1 self_links=0 duplicates=0 max_len=0/32) elapsed_ms=6
```

(The `elapsed_ms=6` is just the leader: at stage 9 the workers return before attaching, which is
why `entered=0` -- and why this is a clean control rather than a build.)

With the leader fixed, the crash *moved*: at stage 0 the failing process is now the **worker**,
and the timeline says it ran for about three seconds first --

```
14:53:26.104  starting background worker process "parallel worker for PID 27193"
14:53:29.052  background worker "parallel worker" (PID 27194) was terminated by signal 11
```

-- so the workers are scanning and inserting, and the fault is inside the insert path (the
callback, `flat_insert_at_level`, or the locked backlink write), not in startup.  That is the
first time the parallel path has executed real work.

**Next diagnostic:** extend `hnswsq.parallel_stage` into the callback -- return after
`extract_vector`, after `plan_flat`, after `apply_flat`, before `publish` -- and bisect the same
way.  The harness reports how far it got, so each stage is one run.

### 3j.26 The crash is `index_build_range_scan`, not my callback

Extending the stage GUC into the row callback separated the scan from the insert, and with the
leader teardown fixed the earlier stages are clean, which they were not before:

| stage | what the worker does | result |
|---|---|---|
| 1 | opens heap and index | clean, `entered=1` |
| 2 | + `BuildIndexInfo` | clean |
| 3 | + `table_beginscan_parallel` and `table_endscan` | clean |
| 4 | + the worker's `BuildState` | clean |
| 5 | + `index_build_range_scan` (**callback returns before reading the tuple**) | **crash** |
| 6 | + `extract_vector` | crash |
| 7 | + the level | crash |
| 0 | the whole insert | crash |

Two things follow.  First, the crash is **not in the callback**: stage 5 returns before touching
`values` or `isnull`, and still dies.  Second, the only difference between the clean stage 4 and
the crashing stage 5 is the `rd_tableam->index_build_range_scan` call -- so that call, or the scan
descriptor handed to it, is the fault.

The argument list matches PostgreSQL's own `table_index_build_scan` wrapper exactly
(`(... , allow_sync, false /*anyvisible*/, progress, 0, InvalidBlockNumber, callback, state,
scan)`), so the suspect is the **scan descriptor**: stage 3 creates it and tears it down without
ever scanning, and stage 5 is the first call that actually walks the heap through it -- i.e. the
first call that uses the *snapshot* the descriptor carries.  If `table_parallelscan_initialize`
did not serialize the snapshot into the descriptor the way this code assumes, a worker reading it
is reading something that is not there, which is exactly a segfault at the first tuple.

**Next experiment (one run):** in the worker, immediately after `table_beginscan_parallel`, log
`(*scan).rs_snapshot` and compare it with what the leader's descriptor holds.  Null or foreign
means the leader-side setup is wrong (the snapshot has to be registered/exportable before it is
copied into a descriptor that another process will read).  If it looks right, the next suspect is
`index_info`, which each worker builds with `BuildIndexInfo` -- palloc'd in the worker, which the
scan may then expect to have been prepared (`ii_ExpressionsState`) by the leader.

Still to come: the driver.  Today `FlatGraph` still owns `nodes_used`/`slabs_used` in
its own fields, so the next step is pointing it at `ArenaState` (and giving `Chunk` a
documented concurrent-write view, since sibling workers write sibling bytes without a
`&mut` borrow to share).

Asserted by `a_shared_arena_round_trips_through_a_segment`: a real `dsm_create` +
`shm_toc_create` segment, an arena allocated by key, then a second `attach` that sees
the leader's writes, shares one lock array, and re-acquires locks after releasing
them (which would hang rather than fail if a release leaked).  Still to come: the
driver -- entry rendezvous, `table_index_build_scan`, and the shared cursors that
today still live in `FlatGraph`'s own fields.

**The swap is done:** every per-node array is a chunk region, the `Vec`s are empty in
chunk mode, and only `nodes_used`/`slabs_used` are kept on the side.  The `Vec` arms
of each accessor remain as the grow-mode backing until the arena is the only mode.

## 3k. Flaky incremental exact-match probe (pre-existing, not from the swap)

`access_method::hnswsq::tests::tests::pg_test_hnswsq_incremental_empty_start_sq8_provisional`
failed in one grouped run of `cargo pgrx test pg18 access_method::hnswsq`
(95 passed / 1 failed, 349.7 s) and then passed both alone (11.3 s) and in an
immediate rerun of the same grouped command (exit 0).  So it is a flake, not a
deterministic regression, and it is on the **disk insert path**, which the storage
swap does not touch (`disk_mode` builds never construct a `FlatGraph`).

Why it is still worth chasing: the assertion is not a recall threshold but an
exact-match probe -- after each 200-row batch, `ORDER BY embedding <-> probe LIMIT 1`
must return the probe row itself, whose distance is 0.  A miss therefore means a
row that was inserted is *not reachable from the entry point*, which is the
insert-path analogue of the connectivity finding in §3c (a node with no incoming
edge is invisible to every search).  Pinning `hnswsq.build_seed` (done in
`incremental_case`) was not enough to make it reproducible, so something on that
path is still not pinned.

To capture the next occurrence -- the test already logs `got`/`expected` per batch
and dumps `hnswsq_diag('hs_i_idx')`, which is what identifies the missing node:

```sh
PGRX_HOME=... RUST_TEST_THREADS=1 cargo pgrx test pg18 access_method::hnswsq 2>&1 | \
  sed -n '/panicked/,/Client Error/p'
```

**Suite state:** `cargo pgrx test pg18` reports 247 passed, 10 ignored, and **2
failures that are pre-existing and unrelated**:
`access_method::ivf::options::tests::pg_test_ivf_options_defaults` and
`..._custom`.  Both run `CREATE INDEX ... USING ivf(encoding)` with no operator
class, which fails with `data type vector has no default operator class for
access method "ivf"`; the extension declares `vector_{cosine,l2,ip}_ops` for
`ivf` without `DEFAULT` (`src/access_method/ivf/mod.rs:133-147`) and no SQL
script contains `USING ivf` at all.  They fail identically in isolation, on a
file and opclass set this work never touches.  Until someone declares a default
opclass or gives the tests an explicit one, hnswsq changes are judged by
`cargo pgrx test pg18 access_method::hnswsq` plus the fingerprint gate, not by
the suite's exit code.

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
