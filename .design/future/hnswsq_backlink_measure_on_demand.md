# Future testing: measure-on-demand backlink eviction (no cached distances, no masks)

**Status:** recorded, not scheduled.  Written while deciding T6 of
`.design/hnswsq_parallel_build_todos.md`; the *decided* policy there is
pgvector's append/shrink with ids only (commit `e88a172`).  This document is the
next step on the same axis: what if eviction is decided by **measuring
distances every time**, with no per-list metadata at all?

Sibling documents: `.design/hnswsq_parallel_build_todos.md` (T6),
`.design/hnswsq_vs_pgvector_gap.md` (where the query/build costs come from),
`.design/hnswsq_perf_analysis.md` (measured build phase splits).

## The idea

Today's single-backend in-memory build caches two things next to every neighbour
list so that a backlink update can be O(len) instead of O(len·cap):

* `list_dists[node][layer][k]` — `d(owner, neighbour k)`, measured when the list
  was produced, used to order the merged list without re-measuring;
* `list_masks[node][layer]` — one bit per entry recording whether the occlusion
  rule *accepted* it or it was only re-added by the closest-pruned backfill, used
  to skip re-checking entries whose acceptance is already known.

The idea: **drop both permanently and recompute every eviction from freshly
measured distances** — i.e. run the full selection heuristic over
`list ∪ {new}` on every backlink update, measuring all the distances it needs.
That is what the *disk* insert path already does (`insert.rs::add_backlink`), and
what pgvector does whenever a list overflows.

## Which distances are actually in play

For a backlink update on target `T` with newcomer `x`:

| distance | purpose | cached policy measures | measure-on-demand measures |
|---|---|---|---|
| `d(T, x)` | the newcomer's rank | 1 (`d_self`) | 1 |
| `d(T, m)` for members `m` | ordering the merged list by `(distance, id)` | **0** — served from `list_dists` | `len` of them |
| `d(m_i, m_j)` | the occlusion test `d(cand, sel) < d(cand, T)` | only for `x` against the accepted prefix, plus the `extras` checks; entries with a set mask bit are *trusted* | the full walk over the merged list |

## Why this is not a quality change

`backlink_prune_mem` is *defined* to be bit-identical to running the full
heuristic over `T.list ∪ {x}` — that is exactly what
`test_backlink_prune_matches_full_heuristic_many_seeds` and
`test_backlink_incremental_matches_every_insert` assert.  Node vectors are
immutable after insert and the cached distances were produced by the same
kernel, so recomputing them yields the same values, the same ordering and the
same list.

> Where the mask's precondition holds, measure-on-demand produces the **same
> list**.  It changes cost, memory and preconditions — not the output.

And where the precondition does **not** hold, measure-on-demand is *more*
correct: the mask-based path silently assumes every mutation of a list went
through the heuristic, which is false for the disk path's `append-if-room`
fallback after repeated races, for ranked/cutoff mode
(`hnswsq.build_backlink_mode = 0`), and for **vacuum repair**, which splices a
tombstoned node's live neighbours into lists that never proposed themselves (a
spliced entry has no meaningful mask provenance).

## Gains

1. **No per-list metadata**: `list_dists` + `list_masks` disappear.  In the
   parallel-build arena that is ≈**150 MB at 1M nodes** (33 × 4 B of distances +
   16 B of mask per layer-0 list) inside a chunk that must be sized up front, and
   the slab collapses to `[u32; cap] + len`.
2. **No precondition to maintain**: the policy becomes repair-agnostic and
   race-agnostic, which is a much easier invariant story for concurrent updates
   and for any future mixed-mode list mutation.
3. **Uniform semantics with the disk insert path**, so build and incremental
   `INSERT` evict by the same rule and one behaviour needs testing rather than
   two.
4. **In the arena the extra measurements are cheap in a way they are not on
   disk**: stored vectors are in shared memory, so "measure" is a SIMD kernel
   call.  On disk the same policy means `ReadBuffer` + rkyv access +
   `codec.decode` per member — which is precisely why the disk path pays what it
   pays and is fine only for row-at-a-time inserts.

## Costs

1. **CPU per update**: roughly `len` (≈32) extra SIMD distances for the ordering
   plus the occlusion checks the mask used to skip.  Order **+1–2 µs per update**
   at dim 128, against the measured 2.7 µs (1M/dim-128: 33.07 M updates in
   88.5 s), i.e. the backlink phase grows 40–60 % and the single-backend build
   ~10–15 %.  In a parallel build most of it is spread across workers.  **This is
   an estimate, not a measurement.**
2. **Lock hold time**: if the recompute happens under the target's write lock,
   hold time goes from "a few checks" to "measure + full walk".  Mitigation is
   the disk path's pattern — snapshot under a read lock, compute with the lock
   released, re-acquire the write lock and validate `list unchanged`, retry on
   change — at the cost of a retry loop.
3. **Loss of the trusted fast path**: at 100k the mask let 60.8 M of 104 M merged
   entries skip re-checking; measure-on-demand pays for those checks again, and
   the `extras` machinery disappears with it.
4. **Only a win in memory.**  The disk insert path should keep its current shape
   regardless of the outcome here.

## The experiment that would settle it

Instrument `arena_apply` (or `mem_apply`, before the arena exists) with a
distance counter and a per-update timer, and run the local `100kd128u` dataset
(100k rows, dim 128, pinned `hnswsq.build_seed`) in three configurations:

| config | list metadata | ordering distances | occlusion checks |
|---|---|---|---|
| (a) mask-cached — today | yes | cached | only `x` + `extras` |
| (b) measure on overflow — pgvector-on-overflow | none | measured when full | full walk on overflow |
| (c) measure every update — this idea | none | measured always | full walk always |

Report per configuration: distances measured per update, ns per update,
`backlink_select` ms, total build seconds, and — most importantly — **graph
quality**: recall sweep (ef 10/40/160/640) and the connectivity invariant (nodes
without an incoming layer-0 edge), because a recomputation that is cheaper than
expected but changes the accepted sets is not a win.

## Acceptance criteria if it is picked up

* (c) within 0.005 recall of (a) at 100k and on 1M BIGANN, zero nodes without an
  incoming layer-0 edge;
* single-backend 1M build no more than ~10 % slower than today, or better if the
  metadata removal also improves cache behaviour;
* `list_dists`/`list_masks` deleted from `MemGraph` and from the arena layout,
  with the exactness tests re-pointed at the reference full-heuristic
  implementation (which stays, as the oracle);
* no change to the disk insert path.
