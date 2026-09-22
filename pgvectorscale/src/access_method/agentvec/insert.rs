//! Segment lifecycle: creating a HOT segment, sealing it, and the foreground
//! insert path.
//!
//! The foreground contract (design §9, §10) is that `aminsert()` performs
//! bounded local work:
//!
//! ```text
//! resolve the current HOT segment
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
//! Lock protocol (see `meta_page`): the metadata path takes the meta page lock
//! and then a segment header lock inside it; the plain append path takes only
//! the header lock.  The order is never inverted.

use pgrx::*;

use crate::access_method::agentvec::directory::{
    AgentVecDirectory, AgentVecSegmentHeader, AgentVecSegmentMeta, SegmentAlgorithm, SegmentLevel,
    SegmentOwnership, SegmentState,
};
use crate::access_method::hnswsq::quantize::{HnswPrecision, Sq8Calibration};
use crate::access_method::agentvec::flat;
use crate::access_method::agentvec::meta_page::AgentVecMetaPage;
use crate::access_method::agentvec::options::TSVAgentVecOptions;
use crate::access_method::distance::{preprocess_cosine, DistanceType};
use crate::access_method::pg_vector::PgVectorInternal;
use crate::util::ItemPointer;

/// Insert a tuple into the index.
///
/// This is the entire foreground path: resolve the HOT segment, seal it if it
/// has reached its row threshold, append the entry.  Nothing here migrates,
/// retrains, re-encodes, splits, merges or rewrites a posting list.
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
    let meta = AgentVecMetaPage::fetch(&index_rel);
    let options = TSVAgentVecOptions::from_relation(&index_rel);
    let vector = extract_vector(*values, meta.get_num_dimensions() as usize);

    insert_entry(
        &index_rel,
        ItemPointer::with_item_pointer_data(*heap_tid),
        &vector,
        meta.get_distance_type(),
        &options,
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
    AgentVecMetaPage::update(index, |meta| {
        let mut directory = meta.load_directory(index);
        if directory.get(meta.get_hot_segment_id()).is_some() {
            // Someone else created it while we waited for the meta lock.
            return meta.get_hot_segment_id();
        }
        open_new_hot_segment(index, &mut directory, meta, SegmentLevel::Hot, &options)
    })
}

/// Seal the current HOT segment and open its successor.
///
/// Returns true when this call performed the seal (false means another
/// transaction got there first).  The work is metadata only: sealing freezes
/// the segment's active chain descriptor, and the new segment is an empty
/// header plus one directory republication — the vectors themselves are not
/// touched (design §10).
pub unsafe fn seal_hot_and_open_new(index: &PgRelation, expected_hot_id: u64) -> bool {
    let options = TSVAgentVecOptions::from_relation(index);
    AgentVecMetaPage::update(index, |meta| {
        if meta.get_hot_segment_id() != expected_hot_id {
            // Another transaction already sealed this segment.
            return false;
        }
        let mut directory = meta.load_directory(index);
        if let Some(segment) = directory.get(expected_hot_id) {
            let header_pointer = segment.header;
            // Freeze the active chain.  This takes the header lock while
            // holding the meta lock, which is the documented order.
            let (vector_count, live_count, dead_count) = AgentVecSegmentHeader::update(
                index,
                header_pointer.block_number,
                |header| {
                    header.seal_active();
                    (
                        header.num_entries,
                        header.live_entries(),
                        header.dead_entries,
                    )
                },
            );

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
            // authoritative source, which is what `agentvec_index_info()`
            // reports.
            segment.vector_count = vector_count;
            segment.live_count = live_count;
            segment.dead_count = dead_count;
            segment.epoch = epoch;
        }
        open_new_hot_segment(index, &mut directory, meta, SegmentLevel::Hot, &options);
        true
    })
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
/// Returns the segment id the entry landed in.
pub unsafe fn insert_entry(
    index: &PgRelation,
    heap_tid: ItemPointer,
    vector: &[f32],
    distance_type: DistanceType,
    options: &TSVAgentVecOptions,
    building: bool,
) -> u64 {
    // Vectors are stored in the representation the distance function expects:
    // cosine distance is computed on unit vectors (see `distance_cosine`),
    // while L2 and inner product use the vector as given.
    let mut stored = vector.to_vec();
    if distance_type == DistanceType::Cosine {
        preprocess_cosine(&mut stored);
    }
    let entry_bytes = flat::encode_entry(heap_tid, &stored, flat::STATE_LIVE);
    let max_rows = options.get_hot_segment_max_rows();

    // Bounded: the first pass may have to create a segment, the second may
    // have to seal the previous one, and the third appends.
    for _attempt in 0..4 {
        match current_hot_segment(index) {
            None => {
                open_initial_hot_segment(index);
            }
            Some(hot) if hot.algorithm == SegmentAlgorithm::IvfRaBitQ => {
                error!(
                    "agentvec: HOT segment {} uses {} storage, which this phase cannot append to",
                    hot.segment_id,
                    hot.algorithm.as_str()
                );
            }
            Some(hot) if hot.num_entries >= max_rows => {
                seal_hot_and_open_new(index, hot.segment_id);
            }
            Some(hot) if hot.algorithm == SegmentAlgorithm::Flat => {
                AgentVecSegmentHeader::update(index, hot.header.block_number, |header| {
                    let active = header.active.take();
                    header.active = Some(flat::append_entry(index, active, &entry_bytes));
                    header.num_entries += 1;
                });
                return hot.segment_id;
            }
            Some(hot) => {
                // HNSW: append into the embedded hnswsq region, then bump the
                // segment's counter.  `stored` is already cosine-normalized
                // where the metric requires it, matching hnswsq's own insert
                // convention.  The whole insert runs in a per-insert memory
                // context like hnswsq's aminsert: the search pallocs one
                // element-sized buffer per neighbor it loads (the hnswsq
                // insert path frees them with its per-insert context), and
                // without the reset they accumulate ~100 KB/row in bulk
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
        }
    }
    error!("agentvec: could not resolve a HOT segment to insert into");
}
