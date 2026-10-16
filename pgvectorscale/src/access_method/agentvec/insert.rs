//! Segment lifecycle: creating a HOT segment, sealing it, and the foreground
//! insert path.
//!
//! The foreground contract (design §9, §10) is that `aminsert()` performs
//! bounded local work:
//!
//! ```text
//! resolve the appendable HOT pool
//!     │
//!     ├── pick the least-loaded member (cached; O(1) on the steady path)
//!     │
//!     ├── over the row threshold?  →  seal it + open the next one (metadata only)
//!     │
//!     └── append the entry to its active chain
//! ```
//!
//! No migration, no centroid retraining, no re-encoding and no posting-list
//! rewrite happens here; the sealed segment is handed to asynchronous
//! maintenance, which is what phase 4/5 add.
//!
//! # The HOT pool
//!
//! `hot_segment_count` (default 1) controls how many HOT segments accept
//! inserts at the same time.  The pool is *derived from the published
//! directory* — every segment with `level == Hot` and `state == Published` is
//! appendable — so it needs no new on-disk metadata and no format version
//! bump: an index written by an earlier build simply has one such segment.
//! The directory pointer is the invalidation key for the backend-local cache,
//! because every publication allocates a fresh chained item, so a changed
//! pointer always means the pool changed.
//!
//! Every member is an independent embedded hnswsq region with its own
//! metapage, insert page, header page and entry point, and every member stays
//! `Published` and searchable until it is sealed, so the transaction contract
//! is unchanged: the entry is written into the region and WAL-logged inside
//! the inserting transaction, exactly as it was with a single HOT segment.
//!
//! Lock protocol (see `meta_page`): the metadata path takes the meta page lock
//! and then a segment header lock inside it; the plain append path takes only
//! the header lock.  The order is never inverted.  Sealing additionally takes
//! the *region* update page lock exclusively (the same lock every insert holds
//! in share mode for the duration of its graph update) so a segment can never
//! be sealed while an insert is in flight against it.

use pgrx::*;

use crate::access_method::agentvec::directory::{
    AgentVecDirectory, AgentVecSegmentHeader, AgentVecSegmentMeta, SegmentAlgorithm, SegmentLevel,
    SegmentOwnership, SegmentState,
};
use crate::access_method::agentvec::flat;
use crate::access_method::agentvec::meta_page::AgentVecMetaPage;
use crate::access_method::agentvec::options::{
    TSVAgentVecOptions, HOT_SEGMENT_COUNT_AUTO, MAX_HOT_SEGMENT_COUNT,
};
use crate::access_method::distance::{preprocess_cosine, DistanceType};
use crate::access_method::hnswsq::quantize::{HnswPrecision, Sq8Calibration};
use crate::access_method::pg_vector::PgVectorInternal;
use crate::util::ItemPointer;

/// Insert a tuple into the index.
///
/// This is the entire foreground path: resolve the HOT pool, pick a member,
/// seal it if it has reached its row threshold, append the entry.  Nothing
/// here migrates, retrains, re-encodes, splits, merges or rewrites a posting
/// list.
#[pg_guard]
pub unsafe extern "C-unwind" fn aminsert(
    index: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    _heap: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    // Null vectors are not indexed (the index is optional for them).
    if *isnull {
        return false;
    }

    let index_rel = PgRelation::from_pg(index);
    let options = TSVAgentVecOptions::from_relation(&index_rel);
    // One meta page read serves both the dimension and the pool resolution:
    // `insert_entry` takes the already-parsed meta instead of re-fetching it.
    let meta = AgentVecMetaPage::fetch(&index_rel);
    let vector = extract_vector(*values, meta.get_num_dimensions() as usize);

    insert_entry(
        &index_rel,
        ItemPointer::with_item_pointer_data(*heap_tid),
        &vector,
        meta.get_distance_type(),
        &options,
        &meta,
        false,
    );

    false
}


/// Payload format version of a phase-1 `FLAT` segment.
pub const SEGMENT_FORMAT_FLAT_V1: u32 = 1;
/// Payload format version of an embedded hnswsq region (HOT).
pub const SEGMENT_FORMAT_HNSW_V1: u32 = 2;

/// The segment foreground inserts currently target.
#[derive(Clone, Copy, Debug)]
pub struct HotSegment {
    /// Stable segment id.
    pub segment_id: u64,
    /// Its header page (the lifecycle/counter record for every algorithm).
    pub header: ItemPointer,
    /// Entries appended so far, as of the header read.
    pub num_entries: u64,
    /// Which algorithm the segment stores.
    pub algorithm: SegmentAlgorithm,
    /// Payload root: the embedded hnswsq region base block (HNSW segments),
    /// invalid for `FLAT`.
    pub code_root: ItemPointer,
}

/// Resolve the segment foreground inserts should target.
///
/// `None` means the index has no usable HOT segment yet (an index whose
/// directory was never populated), which the caller fixes by opening one.
///
/// This is the single-segment form, kept for the paths that only need *one*
/// appendable segment.  The foreground insert path uses [`resolve_hot_pool`].
pub unsafe fn current_hot_segment(index: &PgRelation) -> Option<HotSegment> {
    let meta = AgentVecMetaPage::fetch(index);
    let hot_id = meta.get_hot_segment_id();
    if hot_id == 0 {
        return None;
    }
    let directory = meta.load_directory(index);
    let segment = directory.get(hot_id)?;
    if !segment.is_searchable() {
        return None;
    }
    let header = AgentVecSegmentHeader::load(index, segment.header);
    Some(HotSegment {
        segment_id: segment.segment_id,
        header: segment.header,
        num_entries: header.num_entries,
        algorithm: segment.algorithm(),
        code_root: segment.code_root,
    })
}

// ---------------------------------------------------------------------------
// The HOT pool
// ---------------------------------------------------------------------------

/// Whether a directory entry is a segment foreground inserts may append to.
///
/// A segment stops being appendable the moment it is sealed (its state leaves
/// `Published`), which is what lets the pool be derived from the published
/// directory with no extra metadata.
fn is_appendable_hot(segment: &AgentVecSegmentMeta) -> bool {
    segment.level() == SegmentLevel::Hot && segment.state() == SegmentState::Published
}

/// The appendable HOT segments of a published directory, in directory order.
fn pool_from_directory(directory: &AgentVecDirectory) -> Vec<HotSegment> {
    directory
        .segments
        .iter()
        .filter(|s| is_appendable_hot(s))
        .map(|s| HotSegment {
            segment_id: s.segment_id,
            header: s.header,
            // The authoritative count lives in the header, which the append
            // path reads for the segment it picks; the directory copy is only
            // used to spread load and may lag.
            num_entries: s.vector_count,
            algorithm: s.algorithm(),
            code_root: s.code_root,
        })
        .collect()
}

/// A resolved pool plus the per-backend bookkeeping used to spread inserts.
struct CachedPool {
    /// Identity of the index this pool belongs to.
    relid: pg_sys::Oid,
    /// The published directory pointer the pool was resolved from.  Every
    /// publication allocates a fresh chained item, so a different pointer
    /// always means a different pool.
    directory_pointer: ItemPointer,
    /// The appendable segments, in directory order.
    segments: Vec<HotSegment>,
    /// Inserts this backend has made into each member since resolution, so a
    /// backend does not keep choosing the same segment every time.  Indexed in
    /// lockstep with `segments`.
    local_pending: Vec<u64>,
}

thread_local! {
    /// Backend-local HOT pool cache.  Backends are single-threaded, so a
    /// thread-local is safe.
    static HOT_POOL: std::cell::RefCell<Option<CachedPool>> =
        const { std::cell::RefCell::new(None) };
}

/// How many HOT segments the writer side should keep appendable.
///
/// Returns a value in `1..=MAX_HOT_SEGMENT_COUNT`.
///
/// * an explicit `hot_segment_count` is honoured as-is (clamped to the pool
///   bounds);
/// * [`HOT_SEGMENT_COUNT_AUTO`] derives the count from the *consolidation
///   backlog* — the number of HOT segments already sealed and waiting for
///   maintenance.  That is the in-index signal for "the write side is
///   producing sealed segments faster than maintenance drains them", the one
///   condition under which a second appendable segment genuinely helps: the
///   writers of a sealing, not-yet-consolidated segment can continue in a
///   sibling instead of queuing behind the seal.  It needs no new persistent
///   state, because the backlog is already visible in the published directory.
///
/// Measured on 32 vCPU / PG 17.11 with 8, 16 and 32 concurrent inserters, K
/// independent hnswsq regions (emulated with K independent indexes holding
/// identical graphs) produced no aggregate throughput change, so the default
/// is 1 rather than auto: the foreground cost is per-row CPU and
/// buffer-manager work, not contention on one graph's pages.
fn effective_hot_segment_count(configured: usize, backlog: usize) -> usize {
    if configured != HOT_SEGMENT_COUNT_AUTO {
        return configured.clamp(1, MAX_HOT_SEGMENT_COUNT);
    }
    // Grow with the backlog, but conservatively: every extra HOT segment is
    // searched by every scan.
    const AUTO_CEILING: usize = 4;
    (1 + backlog).clamp(1, AUTO_CEILING.min(MAX_HOT_SEGMENT_COUNT))
}

/// Resolve the appendable HOT pool for `index`.
///
/// The authoritative segment list is the published directory; the backend-local
/// cache is keyed on `(index, directory pointer)`, so the steady-state path
/// only pays a pointer comparison on top of the meta page read the caller
/// already did.  A changed pointer (any seal, consolidation or build) forces a
/// reload, which is exactly when the pool can change.
unsafe fn resolve_hot_pool(
    index: &PgRelation,
    meta: &AgentVecMetaPage,
    options: &TSVAgentVecOptions,
) -> Vec<HotSegment> {
    let relid = index.oid();
    let pointer = meta
        .get_directory_pointer()
        .unwrap_or_else(ItemPointer::new_invalid);

    HOT_POOL.with(|cell| {
        let mut slot = cell.borrow_mut();
        let fresh = match slot.as_ref() {
            Some(c) => c.relid != relid || c.directory_pointer != pointer,
            None => true,
        };
        let backlog = if fresh {
            let directory = meta.load_directory(index);
            let mut segments = pool_from_directory(&directory);
            // The directory's `vector_count` is only refreshed when a segment
            // is sealed, so for appendable HOT segments it is a placeholder.
            // Read each member's header for the authoritative count; this
            // happens once per publication (K page reads), not per insert.
            for seg in segments.iter_mut() {
                seg.num_entries =
                    AgentVecSegmentHeader::load(index, seg.header).num_entries;
            }
            let backlog = directory
                .segments
                .iter()
                .filter(|s| {
                    s.level() == SegmentLevel::Hot
                        && matches!(
                            s.state(),
                            SegmentState::QueuedForMigration | SegmentState::Retiring
                        )
                })
                .count();
            let local_pending = vec![0u64; segments.len()];
            *slot = Some(CachedPool {
                relid,
                directory_pointer: pointer,
                segments,
                local_pending,
            });
            backlog
        } else {
            0
        };
        let cached = slot.as_mut().expect("pool just populated");
        let k = effective_hot_segment_count(options.get_hot_segment_count(), backlog);

        // Spread the inserts over the first `k` members, accounting for what
        // this backend has already put into each since resolution.
        let mut best = usize::MAX;
        let mut best_load = u64::MAX;
        for (i, seg) in cached.segments.iter().enumerate().take(k) {
            let load = seg.num_entries.saturating_add(cached.local_pending[i]);
            if load < best_load {
                best_load = load;
                best = i;
            }
        }
        if best == usize::MAX {
            return Vec::new();
        }
        cached.local_pending[best] += 1;
        vec![cached.segments[best]]
    })
}

/// Drop the backend-local pool cache, forcing the next insert to reload the
/// published directory.  Called after a publication that can change the pool.
fn invalidate_hot_pool_cache() {
    HOT_POOL.with(|cell| *cell.borrow_mut() = None);
}

/// A guard holding a segment's hnswsq region update page lock exclusively.
///
/// This is the *same* lock `insert_tuple_on_disk` takes in share mode for the
/// whole duration of a graph update, so acquiring it exclusively is a barrier:
/// it waits for every in-flight insert into that region to finish and keeps new
/// ones out until the guard is dropped.  Sealing under this guard is what
/// guarantees a segment cannot be demoted while an insert is in flight against
/// it — the precondition for consolidation to collect a complete snapshot.
pub struct RegionUpdateBarrierGuard {
    index: pg_sys::Relation,
    base: pg_sys::BlockNumber,
}

impl RegionUpdateBarrierGuard {
    /// Acquire the barrier.  Returns `None` for a segment with no HNSW region
    /// (a `FLAT` HOT segment), where there is no graph to protect.
    pub unsafe fn acquire_region(index: &PgRelation, base: pg_sys::BlockNumber) -> Option<Self> {
        if base == 0 || base == pg_sys::InvalidBlockNumber {
            return None;
        }
        pg_sys::LockPage(
            index.as_ptr(),
            crate::access_method::hnswsq::types::update_lock_page(base),
            pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
        );
        Some(RegionUpdateBarrierGuard {
            index: index.as_ptr(),
            base,
        })
    }
}

impl Drop for RegionUpdateBarrierGuard {
    fn drop(&mut self) {
        unsafe {
            pg_sys::UnlockPage(
                self.index,
                crate::access_method::hnswsq::types::update_lock_page(self.base),
                pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
            );
        }
    }
}


/// Create a fresh HOT segment and publish it in the directory.
///
/// Must be called with the meta page's content lock held (the caller passes
/// the meta it is mutating).  Extends the relation twice — once for the
/// segment header page, once for the new directory item — and atomically
/// republishes the directory, so a reader either sees the old directory (the
/// new segment simply does not exist for it yet) or the new one.
unsafe fn open_new_hot_segment(
    index: &PgRelation,
    directory: &mut AgentVecDirectory,
    meta: &mut AgentVecMetaPage,
    level: SegmentLevel,
    options: &TSVAgentVecOptions,
) -> u64 {
    let segment_id = meta.take_next_segment_id();
    let algorithm = SegmentAlgorithm::Hnsw;
    let header = AgentVecSegmentHeader::new(level, algorithm);
    let header_pointer = header.store_new(index);

    // The HNSW region: metapage at its base block, written under the
    // relation extension lock (create_meta_page asserts the base matches the
    // relation end).
    let base = create_hot_region(
        index,
        meta.get_num_dimensions(),
        options.get_hot_precision(),
        options.get_hot_m(),
        options.get_hot_ef_construction(),
    );
    let code_root = ItemPointer::new(base, 1);

    directory.segments.push(AgentVecSegmentMeta::new(
        segment_id,
        0,
        level,
        SegmentState::Published,
        algorithm,
        SegmentOwnership::Owned,
        meta.get_epoch(),
        header_pointer,
        meta.get_distance_type() as u16,
        meta.get_num_dimensions(),
        SEGMENT_FORMAT_HNSW_V1,
    ));
    directory
        .get_mut(segment_id)
        .expect("segment just created")
        .code_root = code_root;

    let (directory_pointer, directory_blocks) = directory.store(index);
    meta.set_directory(directory_pointer, directory_blocks);
    meta.set_hot_segment_id(segment_id);
    segment_id
}

/// Create the embedded hnswsq region of a fresh HOT segment and return its
/// base block.
///
/// The caller must hold the relation extension lock so the region's first
/// block is exactly the relation end (asserted by `create_meta_page`).  The
/// region is an ordinary hnswsq metapage + graph pages inside the AgentVec
/// relation; nothing here touches any other segment.
unsafe fn create_hot_region(
    index: &PgRelation,
    dim: u32,
    precision: HnswPrecision,
    m: usize,
    ef_construction: usize,
) -> pg_sys::BlockNumber {
    let _ext_lock = crate::util::buffer::LockRelationForExtension::new(index);
    let calibration = if precision.needs_calibration() {
        // An empty-start HOT segment has no samples: use the provisional
        // [-1, 1] calibration (the hnswsq convention for incremental builds).
        // Store it FIRST: it extends the relation, and the region base must
        // be the relation end *after* it (create_meta_page asserts this).
        let calib = Sq8Calibration::provisional(dim as usize);
        calib.store(index)
    } else {
        ItemPointer::new_invalid()
    };
    let base =
        pg_sys::RelationGetNumberOfBlocksInFork(index.as_ptr(), pg_sys::ForkNumber::MAIN_FORKNUM);
    crate::access_method::hnswsq::utils::init_region(
        index.as_ptr(),
        base,
        dim as usize,
        m,
        ef_construction,
        precision,
        calibration,
    );
    base
}

/// Create the first HOT segment of an index that has none.
pub unsafe fn open_initial_hot_segment(index: &PgRelation) -> u64 {
    let options = TSVAgentVecOptions::from_relation(index);
    let id = AgentVecMetaPage::update(index, |meta| {
        let mut directory = meta.load_directory(index);
        let already = directory.segments.iter().any(is_appendable_hot);
        if already {
            // Someone else created a live HOT while we waited for the meta
            // lock.  (A retired bulk-built segment must NOT count: nothing
            // would ever open its successor.)
            return meta.get_hot_segment_id();
        }
        open_new_hot_segment(index, &mut directory, meta, SegmentLevel::Hot, &options)
    });
    invalidate_hot_pool_cache();
    id
}

/// Top the appendable HOT pool up to `hot_segment_count` members.
///
/// Called from the build path (so a fresh index already has the requested
/// number of appendable segments, and the steady-state insert path never has to
/// take the meta lock to create one) and from the insert path when the pool is
/// short of its configured size.  Returns the number of members after the call.
pub unsafe fn ensure_hot_pool(index: &PgRelation) -> usize {
    let options = TSVAgentVecOptions::from_relation(index);
    let configured = options.get_hot_segment_count();
    if configured == HOT_SEGMENT_COUNT_AUTO {
        // A fresh index has no consolidation backlog, so the adaptive count
        // starts at 1 and grows as sealed segments accumulate.
        ensure_initial_hot_segment(index);
        return 1;
    }
    let target = configured.clamp(1, MAX_HOT_SEGMENT_COUNT);
    let mut created = 0usize;
    let members = AgentVecMetaPage::update(index, |meta| {
        let mut directory = meta.load_directory(index);
        let mut have = directory.segments.iter().filter(|s| is_appendable_hot(s)).count();
        if have == 0 {
            open_new_hot_segment(index, &mut directory, meta, SegmentLevel::Hot, &options);
            have = 1;
            created += 1;
        }
        while have < target {
            open_new_hot_segment(index, &mut directory, meta, SegmentLevel::Hot, &options);
            have += 1;
            created += 1;
        }
        have
    });
    if created > 0 {
        invalidate_hot_pool_cache();
    }
    members
}

/// Make sure at least one appendable HOT segment exists.
unsafe fn ensure_initial_hot_segment(index: &PgRelation) {
    let options = TSVAgentVecOptions::from_relation(index);
    let created = AgentVecMetaPage::update(index, |meta| {
        let mut directory = meta.load_directory(index);
        if directory.segments.iter().any(is_appendable_hot) {
            return false;
        }
        open_new_hot_segment(index, &mut directory, meta, SegmentLevel::Hot, &options);
        true
    });
    if created {
        invalidate_hot_pool_cache();
    }
}

/// Seal `expected_hot_id` and open its successor.
///
/// Returns true when this call performed the seal (false means another
/// transaction got there first).  The work is metadata only: sealing freezes
/// the segment's active chain descriptor, and the new segment is an empty
/// header plus one directory republication — the vectors themselves are not
/// touched (design §10).
///
/// The seal runs under [`RegionUpdateBarrierGuard`], so it also waits for every
/// in-flight insert into the segment's HNSW region and keeps new ones out
/// until the segment has left the appendable pool.  That is what makes the
/// subsequent consolidation able to collect a complete snapshot of the region
/// (without it, an insert that had already resolved the segment could append
/// after the collector walked the graph, and the row would be missing from both
/// the replacement segment and the retired source).
pub unsafe fn seal_hot_and_open_new(index: &PgRelation, expected_hot_id: u64) -> bool {
    let options = TSVAgentVecOptions::from_relation(index);

    // Resolve the region base *before* taking any lock, then the barrier, then
    // the meta lock.  Lock order: region page lock -> meta content lock ->
    // segment header content lock, which is consistent with the plain append
    // path (region page lock -> header content lock) and with the documented
    // meta -> header order.
    let region_base = {
        let meta = AgentVecMetaPage::fetch(index);
        meta.load_directory(index)
            .get(expected_hot_id)
            .map(|s| s.code_root.block_number)
    };
    let _barrier = region_base.and_then(|base| RegionUpdateBarrierGuard::acquire_region(index, base));

    let sealed = AgentVecMetaPage::update(index, |meta| {
        let mut directory = meta.load_directory(index);
        let Some(segment) = directory.get(expected_hot_id) else {
            return false;
        };
        if !is_appendable_hot(segment) {
            // Another transaction already sealed it (or it was never
            // appendable).  A successor may still be missing if that
            // transaction crashed between the two steps, so fall through and
            // make sure one exists.
            if !directory.segments.iter().any(is_appendable_hot) {
                open_new_hot_segment(index, &mut directory, meta, SegmentLevel::Hot, &options);
            }
            return false;
        }
        let header_pointer = segment.header;
        // Freeze the active chain.  This takes the header lock while holding
        // the meta lock, which is the documented order.
        let (vector_count, live_count, dead_count) =
            AgentVecSegmentHeader::update(index, header_pointer.block_number, |header| {
                header.seal_active();
                (
                    header.num_entries,
                    header.live_entries(),
                    header.dead_entries,
                )
            });

        let epoch = meta.bump_epoch();
        let segment = directory
            .get_mut(expected_hot_id)
            .expect("hot segment disappeared");
        // The frozen segment is fully searchable; it is only marked as an
        // input for asynchronous maintenance.
        segment.state = SegmentState::QueuedForMigration as u8;
        segment.generation += 1;
        // Directory counts are refreshed when the directory is republished
        // anyway (here); between publications the segment header is the
        // authoritative source, which is what `agentvec_index_info()` reports.
        segment.vector_count = vector_count;
        segment.live_count = live_count;
        segment.dead_count = dead_count;
        segment.epoch = epoch;

        open_new_hot_segment(index, &mut directory, meta, SegmentLevel::Hot, &options);
        true
    });

    // The directory was republished either way, so the pool must be reloaded.
    invalidate_hot_pool_cache();
    sealed
}


/// Extract and validate a vector datum.
pub unsafe fn extract_vector(datum: pg_sys::Datum, expected_dim: usize) -> Vec<f32> {
    let detoasted = pg_sys::pg_detoast_datum_copy(datum.cast_mut_ptr());
    let pg_vec = detoasted.cast::<PgVectorInternal>();
    let dim = (*pg_vec).dim as usize;
    let vector = (*pg_vec).to_slice().to_vec();
    // Some PostgreSQL builds return the original pointer for non-toasted
    // data; only free what was actually copied.
    if detoasted != datum.cast_mut_ptr() {
        pg_sys::pfree(detoasted.cast());
    }
    if dim != expected_dim {
        error!(
            "agentvec: vector has {} dimensions, index expects {}",
            dim, expected_dim
        );
    }
    vector
}

/// Append one vector to the index — the shared path for `aminsert` and the
/// heap scan of `ambuild`.
///
/// `meta` is the meta page the caller already parsed; taking it as an argument
/// rather than re-fetching it removes one page read and one rkyv parse per row
/// from the foreground path.
///
/// Returns the segment id the entry landed in.
pub unsafe fn insert_entry(
    index: &PgRelation,
    heap_tid: ItemPointer,
    vector: &[f32],
    distance_type: DistanceType,
    options: &TSVAgentVecOptions,
    meta: &AgentVecMetaPage,
    building: bool,
) -> u64 {
    // Vectors are stored in the representation the distance function expects:
    // cosine distance is computed on unit vectors (see `distance_cosine`),
    // while L2 and inner product use the vector as given.
    let mut stored = vector.to_vec();
    if distance_type == DistanceType::Cosine {
        preprocess_cosine(&mut stored);
    }
    let max_rows = options.get_hot_segment_max_rows();
    // The FLAT entry framing is only needed by the FLAT branch below; building
    // it on the HNSW path was pure per-row allocation and copying.
    let mut flat_entry: Option<Vec<u8>> = None;

    // Bounded: the first pass may have to create a segment, the second may
    // have to seal a full one, and the third appends.  The meta page the caller
    // parsed is reused on the fast path; the slow paths republish the directory
    // and therefore re-read it, so the retry sees the new pool instead of a
    // stale pointer.
    let mut refreshed: Option<AgentVecMetaPage> = None;
    for _attempt in 0..4 {
        let meta = refreshed.as_ref().unwrap_or(meta);
        let pool = resolve_hot_pool(index, meta, options);
        let Some(hot) = pool.first().copied() else {
            if options.get_hot_segment_count() != HOT_SEGMENT_COUNT_AUTO {
                ensure_hot_pool(index);
            } else {
                open_initial_hot_segment(index);
            }
            refreshed = Some(AgentVecMetaPage::fetch(index));
            continue;
        };

        if hot.algorithm == SegmentAlgorithm::IvfRaBitQ {
            error!(
                "agentvec: HOT segment {} uses {} storage, which this phase cannot append to",
                hot.segment_id,
                hot.algorithm.as_str()
            );
        }

        // The directory copy of the count may lag; the header is authoritative.
        let header = AgentVecSegmentHeader::load(index, hot.header);
        if header.num_entries >= max_rows {
            // Sealing republishes the directory and therefore refreshes the
            // pool on the next attempt, which is how the writer moves on to a
            // sibling segment instead of re-reading a stale pool.
            seal_hot_and_open_new(index, hot.segment_id);
            refreshed = Some(AgentVecMetaPage::fetch(index));
            continue;
        }

        if hot.algorithm == SegmentAlgorithm::Flat {
            let entry_bytes = flat_entry.get_or_insert_with(|| {
                flat::encode_entry(heap_tid, &stored, flat::STATE_LIVE)
            });
            AgentVecSegmentHeader::update(index, hot.header.block_number, |header| {
                let active = header.active.take();
                header.active = Some(flat::append_entry(index, active, entry_bytes));
                header.num_entries += 1;
            });
            return hot.segment_id;
        }

        // HNSW: append into the embedded hnswsq region, then bump the
        // segment's counter.  `stored` is already cosine-normalized where the
        // metric requires it, matching hnswsq's own insert convention.  The
        // whole insert runs in a per-insert memory context like hnswsq's
        // aminsert: the search pallocs one element-sized buffer per neighbor it
        // loads (the hnswsq insert path frees them with its per-insert
        // context), and without the reset they accumulate ~100 KB/row in bulk
        // builds.
        let mut insert_ctx =
            pgrx::PgMemoryContexts::new("agentvec hnsw insert temporary context");
        let segment_id = insert_ctx.switch_to(|_| {
            let base = hot.code_root.block_number;
            let support =
                crate::access_method::hnswsq::utils::init_support(index.as_ptr(), base);
            let mut encoded = vec![0u8; support.codec.vector_bytes()];
            let clamped = support.codec.encode_into(&stored, &mut encoded);
            let mut tid_data = pg_sys::ItemPointerData::default();
            heap_tid.to_item_pointer_data(&mut tid_data);
            crate::access_method::hnswsq::insert::insert_tuple_on_disk(
                index.as_ptr(),
                base,
                &support,
                &encoded,
                &tid_data,
                building,
                clamped,
            );
            AgentVecSegmentHeader::update(index, hot.header.block_number, |header| {
                header.num_entries += 1;
            });
            hot.segment_id
        });
        drop(insert_ctx);
        return segment_id;
    }
    error!("agentvec: could not resolve a HOT segment to insert into");
}

/// Fetch the exact heap vector of a live row — the conversion source for
/// calibrated HOT layouts, whose inline bytes clamp.  Returns `None` when the
/// row is not visible to the active snapshot (deleted or updated away while
/// the conversion ran).
/// Prefetch the heap pages that a rerank pass is about to read.
///
/// The candidates are scattered heap tids, so their reads are independent:
/// issuing the prefetches up front lets the storage layer work on them in
/// parallel instead of the scan stalling on one page at a time.  Prefetching
/// is advisory — a page that arrives late (or is already cached) only costs
/// the call.
///
/// The relation must be the *heap*: the index only stores tids, and
/// `PrefetchBuffer` raises when a block lies beyond the end of the relation
/// it is given.  Blocks past the current end (a tid whose row was removed by a
/// rewrite) are skipped for the same reason.
pub unsafe fn prefetch_heap_tids(index: &PgRelation, tids: impl Iterator<Item = ItemPointer>) {
    let heap_rel = pg_sys::table_open(
        (*(*index.as_ptr()).rd_index).indrelid,
        pg_sys::AccessShareLock as pg_sys::LOCKMODE,
    );
    let nblocks =
        pg_sys::RelationGetNumberOfBlocksInFork(heap_rel, pg_sys::ForkNumber::MAIN_FORKNUM);
    for tid in tids {
        if tid.is_valid() && tid.block_number < nblocks {
            pg_sys::PrefetchBuffer(heap_rel, pg_sys::ForkNumber::MAIN_FORKNUM, tid.block_number);
        }
    }
    pg_sys::table_close(heap_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
}

pub unsafe fn fetch_heap_vector(
    index: &PgRelation,
    tid: pg_sys::ItemPointerData,
    dim: usize,
    distance_type: DistanceType,
) -> Option<Vec<f32>> {
    let heap_rel = pg_sys::table_open(
        (*(*index.as_ptr()).rd_index).indrelid,
        pg_sys::AccessShareLock as pg_sys::LOCKMODE,
    );
    let result = {
        // The heap tuple's attnum of the indexed vector column, from the
        // index catalog's indkey (agentvec supports exactly one indexed
        // column).  A buffer-heap-tuple slot deforms with ITS descriptor, so
        // it must be the heap relation's descriptor, and the attnum must be
        // the vector's position in the heap tuple.
        let rd_index = (*index.as_ptr()).rd_index;
        let attnum = *(*rd_index).indkey.values.as_ptr() as i32;
        let slot =
            pg_sys::MakeSingleTupleTableSlot((*heap_rel).rd_att, &pg_sys::TTSOpsBufferHeapTuple);
        let mut htup: pg_sys::HeapTupleData = std::mem::zeroed();
        // PG18's heap_fetch reads the TID from the tuple's t_self.
        htup.t_self = tid;
        let mut buffer: pg_sys::Buffer = 0;
        let got = pg_sys::heap_fetch(
            heap_rel,
            pg_sys::GetActiveSnapshot(),
            &mut htup,
            &mut buffer,
            true,
        );
        let out = if got {
            // Transfer OUR heap_fetch pin to the slot (ExecStorePinnedBufferHeapTuple,
            // transfer_pin=true): plain ExecStoreBufferHeapTuple pins the buffer
            // AGAIN in PG17.11+/PG18, and the clear below then releases only that
            // second pin — leaking the heap_fetch pin (one per fetched candidate,
            // "resource was not closed" warnings at statement end).
            pg_sys::ExecStorePinnedBufferHeapTuple(&mut htup, slot, buffer);
            let mut isnull = false;
            let datum = pg_sys::slot_getattr(slot, attnum, &mut isnull);
            if isnull {
                None
            } else {
                let mut vector = super::insert::extract_vector(datum, dim);
                if distance_type == DistanceType::Cosine {
                    preprocess_cosine(&mut vector);
                }
                Some(vector)
            }
        } else {
            None
        };
        // Release the slot's pin (the transferred heap_fetch pin) explicitly.
        pg_sys::ExecClearTuple(slot);
        pg_sys::ExecDropSingleTupleTableSlot(slot);
        out
    };
    pg_sys::table_close(heap_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
    result
}
