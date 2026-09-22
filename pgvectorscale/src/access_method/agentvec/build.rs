//! AgentVec index build.
//!
//! Phase 1 builds the index in a single streaming pass over the heap: the meta
//! page and the first HOT segment are created up front, then every heap row is
//! appended through the same path foreground inserts use
//! ([`insert::insert_entry`]), so sealing and segment creation behave
//! identically during build and during live DML.
//!
//! Memory is bounded by the batch the executor hands the callback, not by the
//! table: no sample is retained and no in-memory graph is built.  Later phases
//! replace this with the level-aware streaming build (WARM/COLD segments trained
//! from a bounded reservoir sample, as the IVF AM does today).

use pg_sys::{FunctionCall0Coll, InvalidOid};
use pgrx::pg_sys::index_getprocinfo;
use pgrx::*;

use crate::access_method::agentvec::meta_page::AgentVecMetaPage;
use crate::access_method::agentvec::options::TSVAgentVecOptions;
use crate::access_method::agentvec::{insert, AGENTVEC_DISTANCE_TYPE_PROC};
use crate::access_method::distance::DistanceType;
use crate::util::ItemPointer;

/// The distance metric this index was created for, taken from the operator
/// class's support function 1.
fn index_distance_type(indexrel: pg_sys::Relation) -> DistanceType {
    unsafe {
        let fmgr_info = index_getprocinfo(indexrel, 1, AGENTVEC_DISTANCE_TYPE_PROC);
        if fmgr_info.is_null() {
            error!("agentvec: no distance type function found for index");
        }
        let result = FunctionCall0Coll(fmgr_info, InvalidOid).value() as u16;
        DistanceType::from_u16(result)
    }
}

/// Vector dimensions of the indexed column (`vector(1536)` has atttypmod 1536).
fn index_dimensions(index_rel: &PgRelation) -> usize {
    let dimensions = index_rel
        .tuple_desc()
        .get(0)
        .map(|attr| attr.atttypmod as usize)
        .unwrap_or(0);
    if dimensions == 0 {
        panic!("agentvec: cannot determine vector dimensions from index");
    }
    dimensions
}

/// Create the on-disk skeleton of an empty index: meta page (block 0) plus one
/// empty HOT segment published in the directory.
unsafe fn write_empty_index(index_rel: &PgRelation, distance_type: DistanceType, dim: usize) {
    AgentVecMetaPage::create(index_rel, dim as u32, distance_type);
    insert::open_initial_hot_segment(index_rel);
}

/// Build a new AgentVec index over an existing heap.
///
/// The initial HOT segment is built with the hnswsq **bulk** builder
/// (`hnswsq::build::build_region`) into a region whose base is the relation
/// end — the same streaming two-pass build the standalone AM uses, so CREATE
/// INDEX keeps the hnswsq build speed instead of paying the per-row insert
/// path.  If the heap exceeds `hot_segment_max_rows` the segment is sealed
/// immediately (QueuedForMigration) and the first later insert opens a fresh
/// HOT segment; otherwise it stays the active HOT segment exactly like the
/// incremental path's.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambuild(
    heap: pg_sys::Relation,
    index: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    use crate::access_method::agentvec::directory::{
        AgentVecSegmentHeader, AgentVecSegmentMeta, SegmentAlgorithm, SegmentLevel, SegmentState,
    };
    use crate::access_method::agentvec::insert::SEGMENT_FORMAT_HNSW_V1;

    let heap_rel = PgRelation::from_pg(heap);
    let index_rel = PgRelation::from_pg(index);

    let distance_type = index_distance_type(index);
    let num_dimensions = index_dimensions(&index_rel);
    let options = TSVAgentVecOptions::from_relation(&index_rel);

    // Scaffold: meta page (block 0); the region is built FIRST so its base
    // is exactly the relation end (create_meta_page asserts it).  The
    // segment header + directory items are written after the region — their
    // placement is irrelevant, only the pointers matter.
    AgentVecMetaPage::create(&index_rel, num_dimensions as u32, distance_type);
    let base = pg_sys::RelationGetNumberOfBlocksInFork(
        index_rel.as_ptr(),
        pg_sys::ForkNumber::MAIN_FORKNUM,
    );


    // The bulk build into the embedded region (also writes the region's
    // metapage and, for calibrated layouts, the trained calibration chain).
    let (reltuples, indtuples) = crate::access_method::hnswsq::build::build_region(
        Some(heap_rel.as_ptr()),
        index_rel.as_ptr(),
        index_info,
        base,
        options.get_hot_m(),
        options.get_hot_ef_construction(),
        options.get_hot_precision(),
        30_000,
    );

    // Publish the HOT segment that owns the region.
    let segment_id = AgentVecMetaPage::update(&index_rel, |meta| {
        let mut directory = meta.load_directory(&index_rel);
        let segment_id = meta.take_next_segment_id();
        let header = AgentVecSegmentHeader::new(SegmentLevel::Hot, SegmentAlgorithm::Hnsw);
        let header_pointer = header.store_new(&index_rel);
        let mut seg = AgentVecSegmentMeta::new(
            segment_id,
            0,
            SegmentLevel::Hot,
            SegmentState::Published,
            SegmentAlgorithm::Hnsw,
            crate::access_method::agentvec::directory::SegmentOwnership::Owned,
            meta.get_epoch(),
            header_pointer,
            distance_type as u16,
            num_dimensions as u32,
            SEGMENT_FORMAT_HNSW_V1,
        );
        seg.code_root = ItemPointer::new(base, 1);
        directory.segments.push(seg);
        let (ptr, blocks) = directory.store(&index_rel);
        meta.set_directory(ptr, blocks);
        meta.set_hot_segment_id(segment_id);
        segment_id
    });

    // Record the row counts; seal immediately when the segment exceeds the
    // HOT cap (a later insert opens the successor HOT segment).
    let seal = indtuples as u64 >= options.get_hot_segment_max_rows();
    let header_block = AgentVecMetaPage::fetch(&index_rel)
        .load_directory(&index_rel)
        .get(segment_id)
        .map(|seg| seg.header.block_number)
        .expect("segment just created");
    AgentVecSegmentHeader::update(&index_rel, header_block, |header| {
        header.num_entries = indtuples as u64;
    });
    AgentVecMetaPage::update(&index_rel, |meta| {
        meta.set_num_tuples(reltuples as u64);
        if seal {
            let mut directory = meta.load_directory(&index_rel);
            if let Some(seg) = directory.get_mut(segment_id) {
                seg.state = SegmentState::QueuedForMigration as u8;
            }
            let (ptr, blocks) = directory.store(&index_rel);
            meta.set_directory(ptr, blocks);
        }
    });

    // Phase 4: make sure this database has a maintenance worker (no-op when
    // one is already running or dynamic background workers are unavailable).
    crate::access_method::agentvec::maintenance::launch_worker_for_current_database();

    let mut result = PgBox::<pg_sys::IndexBuildResult>::alloc0();
    result.heap_tuples = reltuples;
    result.index_tuples = indtuples;
    result.into_pg()
}

/// Build an empty index image.
///
/// NOTE: this writes the image to the relation's main fork, like the IVF AM on
/// this branch does.  A fully correct implementation must also write the
/// `INIT_FORKNUM` image that crash recovery copies for unlogged relations;
/// that needs a fork-aware page writer and is tracked as a known limitation.
#[pg_guard]
pub extern "C-unwind" fn ambuildempty(index: pg_sys::Relation) {
    unsafe {
        let index_rel = PgRelation::from_pg(index);
        let distance_type = index_distance_type(index);
        let num_dimensions = index_dimensions(&index_rel);
        write_empty_index(&index_rel, distance_type, num_dimensions);
    }
}
