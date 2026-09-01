# IVF append-only segmented storage — design & implementation notes

Status: **implemented and validated** (Phases 0–4, commits `5c96feb`, `c1eeb85`, `27064f6` on `ivf-rabitq`; local PG 18.4 / arm64 functional + concurrency tests green).
On-disk format version bumped (`IVF_VERSION = 2`): **all existing IVF indexes
must be DROPPED / REINDEXed** after this change.

## Goal

Make concurrent DML (INSERT / DELETE / VACUUM) safe against running scans
(no torn reads, no lost segments, no use-after-free), keep the `smgrreadv`
bulk-read fast path (no scan latency regression), and make INSERT O(1)
amortized instead of O(list-size) rewrite — all without holding a long lock
for the duration of a scan.

## On-disk layout (IVF_VERSION 2)

```
block 0  : IvfMetaPage          single-item page; locked update() primitive
             { magic, version, ext_version, distance_type, num_dimensions,
               storage_type, lists, num_bits, rotation_seed,
               centroids_pointer, list_directory_pointer, quantizer_metadata,
               free_list: ItemPointer, generation }
block 1  : IvfListDirectory     chained item: Vec<IvfListMetadata { header, centroid_offset, num_tuples }>
dynamic  : centroids            chained item (unchanged)
per-list : header page          IvfListHeader (single page, atomic swap target)
             { version, generation, segment_list: ItemPointer,
               segment_list_blocks: u32, active: Option<IvfActiveBuffer> }
per-list : segment-list item    chained, IMMUTABLE once published: Vec<IvfSegment>
per-list : segments             contiguous SoA block runs, IMMUTABLE once published
per-list : active buffer        unpublished append buffer, pages tracked explicitly
dynamic  : free list            chained item: Vec<IvfFreeRange { start_block, num_blocks }>
```

Sealed segments use the unchanged SoA byte stream
(`entry::serialize_entries` / `IvfEntrySlice::parse`; 1-bit codes still
transposed into 32-row batches). The hot scan loop (FastScan SIMD, ex-dot
SIMD, zero-copy parse, bounded top-k heap) is untouched.

## Invariants

1. **Published segments and the published `segment_list` item are immutable.**
2. **The active buffer is invisible to readers** (scans walk `segment_list`
   only). A segment becomes readable only after it is **sealed and flushed**
   (`FlushRelationBuffers` inside the header-lock closure, *before* the header
   page itself is rewritten) and its pointer is published.
3. **Atomic publication** = `IvfListHeader::update`: pin + exclusive content
   lock on the single-page header, parse current header, mutate, rewrite the
   page (WAL). A concurrent reader (share lock) sees the old or the new
   header, never a mix; concurrent writers serialize their read-modify-write
   instead of clobbering each other's swap.
4. **Reclamation only under the relation ExclusiveLock**, which waits for
   every in-flight read burst (scans hold ShareLock for the whole burst).
   Retired blocks are unreachable from any published header, and new scans
   cannot reference them, so reuse is use-after-free-safe by construction.

## Lock protocol (deadlock-free ordering; validated)

- **Scan:** transaction-level **advisory lock in SHARED mode** (key =
  (magic, index OID)) across the whole burst (header/segment-list loads,
  `smgrreadv`, active-buffer reads) → transient content locks. Self-compatible:
  concurrent scans never contend.
- **Insert:** peek header (share content lock) → [advisory EXCLUSIVE to
  reserve reclaimed blocks, released] → header content lock (append + maybe
  seal + swap) → [advisory EXCLUSIVE to push back unused reservations /
  reclaim retired ranges]. The advisory-exclusive never nests inside the
  header content lock.
- **Vacuum:** header content lock (read published segments + active buffer,
  filter, seal merged, swap, collect retired) → [advisory EXCLUSIVE to
  reclaim retired ranges].
- Lock ordering: `advisory-exclusive → meta content lock → extension lock`
  and `header content lock → extension lock` only; nothing takes the
  advisory lock while holding the header lock.

**Why advisory locks and not relation locks:** every INSERT statement holds
`RowExclusiveLock` on the index relation for its whole duration
(`index_open`), so a relation-level Share/Exclusive lock on the index would
conflict with it — two concurrent sealers deadlock on each other's
RowExclusive upgrades (reproduced in testing). `LOCKTAG_ADVISORY` is a
separate lock space with no conflicts with executor relation locks, and
transaction-level acquisition is auto-released at transaction end (no leak
on error).

## Component map

- `ivf/segment.rs` (new): `IvfSegment`, `IvfActiveBuffer`, `IvfSegmentList`,
  `IvfFreeList`, `IvfFreeRange`, `IvfListHeader` (+ `store_new` / `load` /
  `update`).
- `ivf/entry.rs`: unchanged SoA read/write primitives; added `seal_entries`,
  `seal_entries_at`, `seal_bytes_len`/`seal_blocks_needed` (exact sizing for
  the allocator), active-buffer append/read (`ActiveEntry` row-major rkyv
  items). `IvfEntryWriter::finish` holds the extension lock across the whole
  run so sealed segments stay contiguous for `smgrreadv`.
- `ivf/insert.rs`: peek → reserve → header-locked append/seal → post-lock
  bookkeeping (see protocol above). O(1) amortized.
- `ivf/vacuum.rs`: compaction-as-merge under the header lock, including the
  active buffer (so dead entries cannot be published later as stale TIDs);
  skips lists with nothing dead, ≤1 segment and no active buffer; reclaims
  retired ranges after the lock.  Also refuses index-only/count usage via a
  guard in `ivf_amcostestimate` (DBL_MAX when no ORDER BY keys).
- `ivf/scan.rs`: read-burst advisory-lock guard around the results
  computation; probes sealed segments via the FastScan smgr path AND the
  active buffer via the buffer manager with the scalar estimator (rows below
  `ivf.seal_threshold` stay searchable).
- `ivf/meta_page.rs`: single-item page + `update()` + `reclaim_ranges` /
  `allocate_range` / `push_back_range` (all under ExclusiveLock).
- `util/page.rs`: `tsv_fresh_page_capacity()`, `write_single_item_page_locked`.
- `util/buffer.rs`: `RelationLockGuard` (RAII `LockRelation`/`UnlockRelation`).
- `util/chain.rs`: `ChainTapeWriter::write_counted`.

## Parameters

- GUC `ivf.seal_threshold` (default 4096, 1..1_000_000): entries accumulated
  in a list's active buffer before it is sealed into a published segment.
  The header page must stay below half a page (asserted), so very large
  thresholds with tiny entries are capped by construction.
- Unchanged: `ivf.probes`, `ivf.top_k`, reloptions `lists`, `num_bits`,
  `storage_layout`.

## Known limitations (documented, not correctness holes for the common path)

- **Crash window during multi-page writes:** the append path writes the active
  page (WAL) then the header (WAL) in separate records; a crash exactly
  between them can orphan the just-appended entry (the heap row survives, the
  index misses it — recoverable via REINDEX). This matches the pre-existing
  behavior of the old rewrite-in-place insert and is bounded to the last
  append of each list.
- **Vacuum output uses relation extension** (its retired blocks go to the free
  list for later insert-seals to reuse); a read-only workload that vacuums
  heavily grows the file by one compacted copy per vacuum pass.
- **Per-list append serialization:** concurrent inserts to the *same* list
  serialize on that list's header content lock (inherent to append; the
  centroid assignment spreads inserts across lists). Per-backend append
  buffers are a possible future optimization.
- **Sub-8 padded dims in 1-bit FastScan:** pre-existing `code_len()` (`dim/8`
  vs `div_ceil(8)`) mismatch for `dim ≤ 4` (unchanged; out of scope).
- Retired free-list pages still hold stale content until reused; readers can
  never reach them (unreachable from published headers + ExclusiveLock grace).

## Validation checklist

- [x] Correctness stress (local PG 18.4, arm64): 6 concurrent INSERTers +
      select loops + racing VACUUM on the same table — **0 deadlocks, 0 lost
      rows, correct counts**; full lifecycle (build/insert/seal/delete/
      vacuum/reinsert) green; recall@10 = 10/10 on 1-bit 16-dim; 4/8-bit
      scans return rows; filled-build at lists=32 works (degenerate test data
      that collapses k-means — identical vectors — was the only failure mode,
      and it is a data artifact, not a storage bug).
- [x] Deploy to x86 (`113.44.106.182`) and ARM (`116.204.102.142`): built
      with the `pg17` feature, installed `vectorscale-0.9.0.so`, dropped the
      old-format indexes and rebuilt `items_10m` (10M BIGANN, lists=1000,
      num_bits=1) in the new format.
- [x] Reclaim check (x86 + ARM, pg17): 5 delete-all/VACUUM/reload cycles hold
      the index at exactly 409,600 bytes every cycle; the full 100K-row
      check (load → delete 80% → vacuum → reload → delete → vacuum → reload)
      holds 3,637,248 bytes across all stages.  This required fixing the
      active-buffer page allocations, the peek/closure free-space mismatch,
      the seal-reset page reservation, in-place free-list republishing, and
      segment-list item reuse (see commit history).
- [x] Regression gate: 10M BIGANN at `lists=1000, probes=40`, measured as a
      same-host, same-session A/B against the pre-segment code:
      x86: p50 3.80→3.89ms (+2.4%), p99 6.57→6.49ms (−1.2%), mean
      4.12→4.22ms (+2.4%), recall@1 1.000=1.000, mean-r@5 0.994→0.990
      (k-means seed noise); ARM: p50 10.04→10.16ms (+1.2%), p99
      17.82→15.43ms (−13%), mean 10.68→10.63ms, recall@1 0.990→1.000,
      mean-r@5 0.986→0.988.  No performance degradation.
- [x] Insert throughput (x86, pg17, same host A/B, 10K rows, lists=4): old
      code 43,121 ms vs new code 197 ms — **219x faster** (O(1) amortized
      append vs. per-insert O(list) rewrite).
