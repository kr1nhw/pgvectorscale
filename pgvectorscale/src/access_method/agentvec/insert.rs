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
    );

    false
}

/// Payload format version of a phase-1 `FLAT` segment.
pub const SEGMENT_FORMAT_FLAT_V1: u32 = 1;

/// The segment foreground inserts currently target.
#[derive(Clone, Copy, Debug)]
pub struct HotSegment {
    /// Stable segment id.
    pub segment_id: u64,
    /// Its header page.
    pub header: ItemPointer,
    /// Entries appended so far, as of the header read.
    pub num_entries: u64,
    /// Which algorithm the segment stores.
    pub algorithm: SegmentAlgorithm,
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
) -> u64 {
    let segment_id = meta.take_next_segment_id();
    let header = AgentVecSegmentHeader::new(level, SegmentAlgorithm::Flat);
    let header_pointer = header.store_new(index);
    directory.segments.push(AgentVecSegmentMeta::new(
        segment_id,
        0,
        level,
        SegmentState::Published,
        SegmentAlgorithm::Flat,
        SegmentOwnership::Owned,
        meta.get_epoch(),
        header_pointer,
        meta.get_distance_type() as u16,
        meta.get_num_dimensions(),
        SEGMENT_FORMAT_FLAT_V1,
    ));

    let (directory_pointer, directory_blocks) = directory.store(index);
    meta.set_directory(directory_pointer, directory_blocks);
    meta.set_hot_segment_id(segment_id);
    segment_id
}

/// Create the first HOT segment of an index that has none.
pub unsafe fn open_initial_hot_segment(index: &PgRelation) -> u64 {
    AgentVecMetaPage::update(index, |meta| {
        let mut directory = meta.load_directory(index);
        if directory.get(meta.get_hot_segment_id()).is_some() {
            // Someone else created it while we waited for the meta lock.
            return meta.get_hot_segment_id();
        }
        open_new_hot_segment(index, &mut directory, meta, SegmentLevel::Hot)
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
        open_new_hot_segment(index, &mut directory, meta, SegmentLevel::Hot);
        true
    })
}

/// Extract and validate a vector datum.
pub unsafe fn extract_vector(datum: pg_sys::Datum, expected_dim: usize) -> Vec<f32> {
    let detoasted = pg_sys::pg_detoast_datum_copy(datum.cast_mut_ptr());
    let pg_vec = detoasted.cast::<PgVectorInternal>();
    let dim = (*pg_vec).dim as usize;
    let vector = (*pg_vec).to_slice().to_vec();
    pg_sys::pfree(detoasted.cast());
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
            Some(hot) if hot.algorithm != SegmentAlgorithm::Flat => {
                error!(
                    "agentvec: HOT segment {} uses {} storage, which this phase cannot append to",
                    hot.segment_id,
                    hot.algorithm.as_str()
                );
            }
            Some(hot) if hot.num_entries >= max_rows => {
                seal_hot_and_open_new(index, hot.segment_id);
            }
            Some(hot) => {
                AgentVecSegmentHeader::update(index, hot.header.block_number, |header| {
                    let active = header.active.take();
                    header.active = Some(flat::append_entry(index, active, &entry_bytes));
                    header.num_entries += 1;
                });
                return hot.segment_id;
            }
        }
    }
    error!("agentvec: could not resolve a HOT segment to insert into");
}
