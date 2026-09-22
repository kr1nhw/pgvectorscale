# AgentVec — Immutable segments & the Vamana centroid router (phases 3 + 3.5)

Status: **implemented and verified** (2026-09-22, branch `agentvec`, PG 18.4 arm64).
Milestones: **M3/M4 region of the plan** (whole-segment conversion) and **M6** (deterministic
segment router), per the revised design decision: segments are immutable from publication;
no per-tuple migration; a Vamana graph organizes the owned IVF segments' centroids.

Commits: `31c7952` (phase 3), `…` (phase 3.5).

---

## 1. Whole-segment conversion (`agentvec_consolidate`)

`SELECT agentvec_consolidate('idx')` claims the oldest sealed HOT segment
(`QueuedForMigration` or `Retiring` — a crashed claim self-heals), collects its live rows,
builds an **immutable IVF-RaBitQ payload**, and publishes it with ONE directory
republication that adds the new WARM segment and retires the source.  There is no
per-tuple migration machinery to keep idempotent, because there is no per-tuple migration.

```text
claim (meta RMW: QueuedForMigration -> Retiring)
  -> collect live rows of the embedded hnswsq region
  -> k-means (kmeans++ + Lloyd's, reservoir-bounded sample)
  -> RaBitQ encode per list
  -> write invisible IVF payload (IvfMetaPage @ rel end, centroids, SoA runs,
     list headers, list directory)
  -> FlushRelationBuffers            (bulk smgr reads need the pages on disk)
  -> allocate segment id (meta RMW)
  -> ensure router region + insert centroids (Vamana graph nodes)
  -> publish (meta RMW: retire source, add WARM segment, new directory)
```

Crash windows:

* **after claim, before publish** — the source stays `Retiring` (still searchable); the next
  call re-claims it.  Orphaned pages are reclaimed by the phase-10 compactor.
* **after centroid insertion, before publish** — the router holds nodes whose segment id
  never appears in the directory; `route` drops unknown ids.  A segment can never be
  published without its centroids being routable.
* **after publish** — everything is consistent; the swap is atomic because a directory
  republication is a single copy-on-write item referenced by the meta page.

### 1.1 Exact source vectors for calibrated HOT layouts

The embedded hnswsq region's default layout (`hot_storage_layout = f8`, i.e. calibrated
SQ8) starts from the **provisional [-1, 1] per-dimension calibration** (the hnswsq
incremental-build convention).  Any component ≥ 1 clamps to the same byte, so decoded
inline bytes are unusable as a conversion source: every structured vector ≥ 1 collapses
to one value.  For calibrated layouts the conversion therefore fetches the **exact heap
vector** per row (heap_fetch + ExecStoreBufferHeapTuple, the vector's heap attnum read
from the index catalog's `indkey`), with cosine preprocessing applied to match the
stored convention.  Rows not visible to the active snapshot are dropped.  The
training-free layouts (`plain`, `ieeefp16`, `ieeefp8`) decode inline with no heap
fetch, as originally planned; a future calibration-retrain-at-seal path could replace
the heap fetches (plugin-model §15 already plans a heap/TID stream for EXTERNAL
sources).

## 2. Immutable segment rules (as built)

* WARM segments are never appended to, never tombstones: the IVF SoA wire format has no
  state byte.  `ambulkdelete` counts live entries from the segment header; dead rows are
  filtered by heap visibility at the executor.  Structural compaction (drop dead codes,
  re-partition) is phase 10.
* `is_searchable()` = `Published | QueuedForMigration | Retiring`; `Retired` segments are
  out of every scan.
* `UPDATE` produces a new TID; the old entry stays in its immutable segment and is
  filtered by heap visibility — no TID dedup across segments is needed.

## 3. Vamana centroid router (M6)

The router is an **embedded hnswsq region** stored in the index relation (base block in
`AgentVecMetaPage.router_base`): the ported DISANN greedy-search + robust-prune machinery,
so the centroid graph is a Vamana graph.  Each owned IVF segment's centroids are its
nodes, encoded as elements whose synthetic heap TID is `(segment_id, list_id + 1)` —
nothing ever fetches a heap tuple through it.

* build: `ensure_router_region` (created lazily inside the meta RMW, `m = 16`,
  `ef_construction = 100`, `plain` precision) + `add_segment_centroids` on every
  consolidation.
* query: `route()` runs once per scan (before the segment loop): greedy search with
  `ef = 64`, segment ranking by the best visited-centroid distance, activating the top
  `router_top_m` owned IVF segments (`router_top_m = 8` by default).
* always searched regardless of routing: HOT segments, legacy FLAT segments, and
  external segments (no owned centroids → activated at segment granularity as a whole).
* fallback: no router region, or a region with no usable nodes (created by a crashed
  consolidation) → every owned IVF segment is searched.

## 4. Configuration

| reloption | default | meaning |
|---|---|---|
| `router_top_m` | 8 | max owned IVF segments the router activates per query |
| `router_group_top_m` | 2 | reserved (phase 12 grouping/recoding) |
| `ivf_lists` | 100 | centroids per converted segment (k-means caps at n) |
| `ivf_probes` | 10 | lists probed inside each activated segment |
| `rabitq_bits` | 1 | RaBitQ bits per dimension (min dims ≥ 8) |

## 5. Tests

* lifecycle + row identity of the converted segment; recall vs exact ground truth;
  no-op without sealed segments; self-healing of `Retiring` claims; vacuum over
  converted segments (phase 3).
* router: no decision without a region; decided sets contain only owned WARM ids and
  are capped by `router_top_m`; routed recall with more WARM segments than `router_top_m`.
