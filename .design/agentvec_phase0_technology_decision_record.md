# AgentVec — Phase 0 technology decision record

Status: **complete** (2026-09-10, branch `agentvec`, host macOS arm64, PG 18.4 Homebrew, pgrx 0.16.1).
Companion documents: `.design/agentvec_system_design_and_implementation_plan.md` (the plan),
`.design/plugin-model.md` (external index adoption), `.design/agentvec_phase1_implementation_notes.md` (what phase 1 built).

Phase 0 exists to answer five questions before any index code is written. Each
section states the question, what was verified, and the decision taken.

---

## 0.1 Target version, license, and foundation

**Question.** Which pgvectorscale version and license are we building on, and is
the toolchain able to produce a new index AM for the target PostgreSQL?

Verified:

| Item | Finding |
|---|---|
| This repository's license | PostgreSQL License (`LICENSE`) — permissive, no obstacle to a new AM. |
| pgvector (the `vector` type owner) license | PostgreSQL License; current release 0.8.6 (2026-07-29). `vectorscale.control` already declares `requires = 'vector'`, so the type, operators and opclass conventions are available. |
| Rust/pgrx | `pgrx = 0.16.1`, `cargo-pgrx 0.16.1`; crate features `pg14`…`pg18`, default `pg18`. |
| Local PostgreSQL | 18.4 (Homebrew, arm64) plus 17.x for the older feature path. |
| Build/test loop | `PGRX_HOME=<repo>/.pgrx-home cargo pgrx test pg18 <filter>` builds, installs and runs `#[pg_test]`s against a temporary instance. Verified working. |

**Decision.** Build AgentVec as a **new index access method (`agentvec`) inside the existing
`vectorscale` crate**, not as a separate extension. Rationale: the design's §28 module layout is
about module boundaries, not process boundaries, and the pieces AgentVec must reuse —
`util::page`/`buffer`/`chain`, `access_method::distance`, `quantization::rabitq`,
`access_method::ivf::entry`/`simd`/`centroid`, `pg_vector` — are Rust items of this crate.
A separate crate would have to re-export or duplicate all of them and would ship a second `.so`
for no isolation benefit: both extensions would still be loaded into the same backend and share
the same page-type namespace. The design doc explicitly allows this adaptation
("The exact PGRX layout can be adapted to the selected pgvectorscale version").

---

## 0.2 How pgvectorscale does background maintenance, and what AgentVec may reuse

**Question.** How does this codebase run background maintenance, and what can be reused
without coupling AgentVec to StreamingDiskANN?

Verified:

* There is **no background worker anywhere in the crate**: no `RegisterBackgroundWorker`, no
  `BackgroundWorkerHandle`, no `_PG_init` hook beyond GUC/reloption registration and the
  distance-feature check. The existing AMs (`diskann`, `ivf`, `hnswsq`) are all purely
  foreground: build, insert, scan and vacuum are AM callbacks.
* The reusable infrastructure is therefore at the storage layer, not the maintenance layer:
  buffer/page wrappers with `GenericXLog` WAL (`util::page`, `util::buffer`), chained large items
  (`util::chain`), a page allocator with a free list (`ivf::meta_page`), SoA entry streams and
  bulk reads (`ivf::entry`, `ivf::simd`), k-means (`ivf::centroid`), RaBitQ
  (`quantization::rabitq*`), and the append-only segmented storage pattern (`ivf::segment`).
* `diskann`'s parallel build (`amcanbuildparallel` + `CreateParallelContext` +
  `_vectorscale_build_main`) proves that a real multi-process build works in this crate — it is
  the only parallelism in the codebase and is a pattern to copy, not to invent.

**Decision.** Phase 1 uses only the storage-layer primitives (page/buffer/chain/crypto-free
WAL), and copies the *patterns* — single-page atomic read-modify-write, copy-on-write
publication of immutable items, append-only payload pages — rather than depending on any
DiskANN code path. The maintenance worker (phase 4) will be the first background worker in the
crate; nothing in the current code assumes its absence.

---

## 0.3 Parallel index scan and parallel build on the target version

**Question.** Does the target PostgreSQL/Neon actually support a parallel index scan for a
custom AM, and how should a parallel build be done? (Plan §21, §22 and Phase 0 item 5 flag this
as needing verification.)

Verified against the PostgreSQL 18.4 sources:

* `amcanparallel` (PG 10) and `amcanbuildparallel` (PG 17) both exist in PG 18.
* **Parallel scan**: `amcanparallel` is consumed **only by the planner**
  (`indxpath.c` builds a partial path; `plancat.c` copies the flag). There is no AM-type check
  anywhere in the executor: `execParallel.c` dispatches on `nodeTag` alone, and
  `nodeIndexscan.c` calls `index_beginscan_parallel()` whenever the plan is parallel-aware.
  `index_parallelscan_estimate/initialize`, `index_beginscan_parallel` and `index_parallelrescan`
  all exist and tolerate a NULL `amestimateparallelscan`/`aminitparallelscan`/`amparallelrescan`.
  **So a custom AM on PG 18 can get a real parallel index scan**, with two caveats:
  1. all work partitioning is the AM's job (core only supplies the DSM blob — btree implements
     its own page-claiming protocol with an LWLock + condition variable);
  2. `ambeginscan` is called *before* `scan->parallel_scan` is assigned, so parallel state must be
     picked up in `amrescan`/`amgettuple`.
  PG 18 changed the signature to `amestimateparallelscan(Relation, int, int)`; PG 19-dev changes
  `index_beginscan` again (extra `flags` argument), so any code driving index scans is
  version-fragile. No AM other than btree has ever implemented this, and `parallel.sgml` still
  says "parallel index scans are supported only for btree indexes".
* **Parallel build**: `amcanbuildparallel`'s only consumer is `index_build()`, which merely fills
  `index_info->ii_ParallelWorkers`; the AM must launch workers itself with
  `CreateParallelContext(<library>, <entry point>, n)`. `LookupParallelWorkerFunction()` resolves
  any library name that is not `"postgres"` through `load_external_function()`, which is the
  extension hook. pgvector's HNSW and this crate's `diskann` both use it.

**Decision.**

* Phase 1 ships `amcanparallel = false` and `amcanbuildparallel = false`: advertising either
  without implementing the contract (and without the AM-side partitioning) would be a lie.
* Phase 9 (parallel segment search) is **feasible** on PG 18, and the immutable
  `segment_refs[]` + atomic `next_segment` design in the plan maps directly onto it, but it is
  pioneering work for a non-btree AM and needs its own spike. The cheaper intermediate step is
  self-parallelising inside one backend (rayon, as the IVF build already does) while a worker
  claims segments from the shared DSM cursor.
* Parallel *build* should copy `diskann`'s proven plumbing when the WARM/COLD build lands
  (phase 3), not be invented.

---

## 0.4 Neon constraints that shape the design

**Question.** What does the Neon deployment change about an index AM?

From this repository's `.design/neon/` measurements (PG 17.5 fork, x86_64):

* Neon compute is **read-path bound**: ~7.3 % CPU utilisation with ~93 % of wall time waiting on
  pagestore reads. Page locality and the number of pages touched per query dominate everything.
* `smgrreadv` bursts must be **chunked to ≤ 32 blocks** (Neon's `PG_IOV_MAX`); a wider burst
  fails with "Read request too large".
* The Neon fork's `smgropen()` takes an extra `relpersistence` argument, so any code calling it
  directly needs the `neon` cargo feature (this crate already has it and `ivf::entry` is gated).
* Bulk page writes that skip WAL need Neon's `smgr_start_unlogged_build` /
  `smgr_finish_unlogged_build_phase_1` / `smgr_end_unlogged_build` hooks, otherwise pages are
  evicted with zero LSN and the compute panics.
* `smgrprefetch`-based read pipelining was implemented and **measured as a noise-level
  regression, then reverted** — do not re-add it without a cold-path workload that demands it.
* Parallel maintenance workers hit a Neon-specific hazard historically
  ([neon#10184](https://github.com/neondatabase/neon/issues/10184): `apply_config` reloads can
  kill parallel workers with `max_stack_depth cannot be set during a parallel operation`), so
  retry/backoff around long parallel maintenance is prudent.
* Compute units: 1 CU = 1 vCPU + 2 GB RAM; the index working set is a first-class constraint
  (a 10M×128 RaBitQ index is ~350 MB vs ~7.9 GB for the same HNSW graph).
* The extension is loaded on demand at `CREATE EXTENSION` (it is not in
  `shared_preload_libraries`), so **no postmaster-level hooks or background workers run today**.
  Whether a hosted Neon compute permits either is an open risk.

**Decision.** AgentVec keeps the page format read-friendly (contiguous runs, bulk-readable) and
bounded-page-count per query, and it does not adopt unlogged bulk writes or prefetch. Any
background worker (phase 4) must be optional and degrade to a manual/cron entry point, because
the Neon control plane is not verified to allow preloaded workers.

---

## 0.5 Reuse boundary: what AgentVec takes, and the pgvector/HNSW decision

**Question.** Which existing pieces are reused, which are built, and where does the HOT HNSW
implementation come from?

Reuse (unchanged, by module):

| Piece | Source | Used for |
|---|---|---|
| Page/buffer wrappers with WAL | `util::page`, `util::buffer` | every AgentVec page write |
| Chained large items | `util::chain` | the segment directory item |
| Distance kernels + `DistanceType` | `access_method::distance` | search and the opclass contract |
| `vector` datum decoding | `access_method::pg_vector` | insert/build/scan |
| RaBitQ quantizer + FastScan | `quantization::rabitq*` | phase 3 WARM/COLD encoding |
| IVF entry SoA + SIMD scan | `ivf::entry`, `ivf::simd`, `ivf::centroid` | phase 3 WARM/COLD payload |
| Reloption plumbing pattern | `ivf::options`, `access_method::options` | `agentvec.*` reloptions |

Build (new): the segment directory and its copy-on-write publication, the segment header
(active/sealed chain publication point), the HOT seal lifecycle, the FLAT executor, the
`agentvec` AM callbacks, and later the router, migration planner, generations and maintenance
jobs.

**HOT HNSW: where it comes from.** The user's preference is to use pgvector's HNSW rather than
writing another graph implementation. What was verified:

* pgvector's HNSW is a normal AM in another `.so` (`hnswhandler`, PostgreSQL License). There is
  **no supported cross-extension API** for running another AM's search, but the core functions
  exist and are used by other extensions for exactly this: `index_open` + `index_beginscan` +
  `index_rescan` + `index_getnext_slot` (pg_squeeze drives a foreign AM this way;
  VectorChord drives `index_beginscan` from an `ExecutorStart_hook`). pgvector's HNSW needs
  only: an MVCC snapshot, exactly one ORDER BY key (`scan->orderByData[0].sk_argument`), and no
  index quals — and it resolves its own distance function internally.
* **No verified extension drives pgvector's HNSW from another extension's code**, and pgvector
  publishes no API for it (`hnsw.h` is not installed). Every vector extension surveyed ships its
  own AM. Driving a foreign AM is version-fragile (PG 18 added an `instrument` argument to
  `index_beginscan`; PG 19 adds `flags`).

**Decision.** Keep the HNSW-as-a-segment option open behind the segment abstraction, and choose
between the two candidate routes at phase 2 with a small spike:

* **H1 (preferred if the spike succeeds): pgvector-compatible HOT via an adopted physical index.**
  AgentVec registers a `hnsw` relation as an `External` segment (`physical_index_oid`,
  `access_method_oid` are already in the directory entry) and executes it through a
  `SegmentExecutor` adapter built on `index_beginscan`/`index_rescan`/`index_getnext_slot`.
  This reuses pgvector's proven graph code, matches `plugin-model.md`, and doubles as the
  adoption mechanism for user-provided indexes.
* **H2 (fallback): in-tree Rust HNSW** as an `Owned` segment inside the AgentVec relation,
  porting the `hnswsq` branch (page-based HNSW, tombstones, vacuum, 53-test suite) which already
  follows this crate's append-only storage conventions.

The directory format does not depend on this choice: `ownership`, `algorithm`,
`physical_index_oid` and `access_method_oid` are in the phase-1 entry.

---

## 0.6 External index adoption

Per the user's direction, adoption of a *user's pre-existing* index is **deferred**, but the
directory is designed so it fits without a format change:

* `SegmentOwnership::Owned` vs `External` from the first version;
* `physical_index_oid` / `access_method_oid` / `opclass_oid` per segment;
* `code_root` / `posting_root` separated from the segment's mutable header, so an external
  segment can carry no AgentVec-owned payload at all.

The feasibility gate for adoption (and for H1) is the same adapter prototype; it will be built
once, when the first external executor is needed.

---

## 0.7 Findings from Phase 0 that are not AgentVec's to fix

Recorded here because they were found while validating the foundation; they belong to the
pre-existing `ivf` AM on this branch.

1. **PG 18 planner hazard (`ivf` can be chosen for `count(*)`, returning an error/wrong count).**
   PG 18 replaced the `disable_cost` convention with a per-path `disabled_nodes` counter that is
   compared *before* cost. `ivf_amcostestimate` refuses non-ORDER-BY paths by returning
   `f64::MAX` only, so with `SET enable_seqscan = off` the planner picks an Index Only Scan over
   the `ivf` index for `count(*)`; executing it raises
   `assertion left == right failed: left: 128, right: 0` instead of returning the count.
   pgvector's HNSW handles this by setting `path->path.disabled_nodes = 2` on PG 18
   (pgsql-hackers "On disable_cost" thread). AgentVec implements that fix
   (`agentvec_amcostestimate`) and has a regression test for it.
2. **`ivf` writes `xs_orderbyvals` as f32 bits.** `ivf/scan.rs` stores
   `Datum::from(distance.to_bits() as usize)` while the ORDER BY expression is `float8`, so the
   executor reads a denormal (~1e-315) instead of the distance. With `xs_recheckorderby = true`
   this defeats the "value was exact" fast path (every candidate is pushed through the reorder
   queue) and risks `ERROR: index returned tuples in wrong order` in edge cases. AgentVec avoids
   the whole issue: FLAT distances are exact, so it sets `xs_recheckorderby = false` and does not
   publish `xs_orderbyvals` at all. (An exact-match reproduction attempt did not raise the error;
   the datum-type mismatch itself is unambiguous from the code.)
3. **`CREATE INDEX ... USING ivf` fails for small dimensions** with
   `assertion failed: rot.len() >= code.len() * 8` (reproduced with `vector(2)`).
4. **`WARNING: IVF amoptions: entering` on every `CREATE INDEX`/`ALTER INDEX`** — a leftover debug
   warning in `ivf/options.rs` (the audit removed the `amhandler`/`amcostestimate` warnings but
   missed this one).

None of these were changed as part of phase 1; they are listed so the decision to leave them
alone is explicit.
