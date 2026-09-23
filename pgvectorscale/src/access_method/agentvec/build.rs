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

/// Finish a build: record the row count and launch the maintenance worker.
unsafe fn return_finish(
    index_rel: &PgRelation,
    reltuples: f64,
    indtuples: f64,
) -> *mut pg_sys::IndexBuildResult {
    AgentVecMetaPage::update(index_rel, |meta| {
        meta.set_num_tuples(reltuples as u64);
    });
    crate::access_method::agentvec::maintenance::launch_worker_for_current_database();
    let mut result = PgBox::<pg_sys::IndexBuildResult>::alloc0();
    result.heap_tuples = reltuples;
    result.index_tuples = indtuples;
    result.into_pg()
}

/// Build a new AgentVec index over an existing heap.
///
/// The bulk load builds the payload DIRECTLY as an immutable IVF-RaBitQ
/// segment (the `ivf` AM's streaming two-pass build, Lance-style: reservoir
/// sample -> k-means -> per-list batched seal, memory bounded by
/// `num_lists x SEGMENT_ENTRIES`).  The HNSW HOT path is only for live
/// inserts (`aminsert`), whose sealed segments the maintenance worker
/// later converts into further WARM segments.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambuild(
    heap: pg_sys::Relation,
    index: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    use crate::access_method::agentvec::directory::{
        AgentVecSegmentHeader, AgentVecSegmentMeta, SegmentAlgorithm, SegmentLevel,
        SegmentOwnership, SegmentState,
    };
    use crate::access_method::agentvec::consolidate::SEGMENT_FORMAT_IVF_V1;
    use crate::access_method::agentvec::insert::SEGMENT_FORMAT_HNSW_V1;

    let heap_rel = PgRelation::from_pg(heap);
    let index_rel = PgRelation::from_pg(index);

    let distance_type = index_distance_type(index);
    let num_dimensions = index_dimensions(&index_rel);
    let options = TSVAgentVecOptions::from_relation(&index_rel);

    // Scaffold: meta page (block 0); the payload region starts at the
    // relation end right after it.
    AgentVecMetaPage::create(&index_rel, num_dimensions as u32, distance_type);
    let base = pg_sys::RelationGetNumberOfBlocksInFork(
        index_rel.as_ptr(),
        pg_sys::ForkNumber::MAIN_FORKNUM,
    );

    // RaBitQ needs >= 8 dimensions (its 1-bit rotation layout); tiny-dim
    // indexes bulk-build an HNSW HOT segment instead (the incremental
    // path's limits are irrelevant at these sizes).
    if num_dimensions < 8 {
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
                SegmentOwnership::Owned,
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
        AgentVecSegmentHeader::update(&index_rel, {
            AgentVecMetaPage::fetch(&index_rel)
                .load_directory(&index_rel)
                .get(segment_id)
                .expect("segment just created")
                .header
                .block_number
        }, |header| {
            header.num_entries = indtuples as u64;
        });
        return return_finish(&index_rel, reltuples, indtuples);
    }
    let (reltuples, indtuples, centroids, directory_ptr) =
        crate::access_method::ivf::build::build_embedded_region(
            heap_rel.as_ptr(),
            index_rel.as_ptr(),
            index_info,
            base,
            options.get_ivf_lists() as usize,
            options.get_rabitq_bits(),
            distance_type,
            30_000,
        );

    let bulk_segment_id = if indtuples > 0.0 {
        // Publish the bulk payload as a WARM IVF-RaBitQ segment.
        let segment_id = AgentVecMetaPage::update(&index_rel, |meta| {
            let mut directory = meta.load_directory(&index_rel);
            let segment_id = meta.take_next_segment_id();
            let header =
                AgentVecSegmentHeader::new(SegmentLevel::Warm, SegmentAlgorithm::IvfRaBitQ);
            let header_pointer = header.store_new(&index_rel);
            let mut seg = AgentVecSegmentMeta::new(
                segment_id,
                0,
                SegmentLevel::Warm,
                SegmentState::Published,
                SegmentAlgorithm::IvfRaBitQ,
                SegmentOwnership::Owned,
                meta.get_epoch(),
                header_pointer,
                distance_type as u16,
                num_dimensions as u32,
                SEGMENT_FORMAT_IVF_V1,
            );
            seg.code_root = ItemPointer::new(base, 1);
            seg.posting_root = directory_ptr;
            seg.vector_count = indtuples as u64;
            directory.segments.push(seg);
            let (ptr, blocks) = directory.store(&index_rel);
            meta.set_directory(ptr, blocks);
            segment_id
        });
        AgentVecSegmentHeader::update(&index_rel, {
            AgentVecMetaPage::fetch(&index_rel)
                .load_directory(&index_rel)
                .get(segment_id)
                .expect("segment just created")
                .header
                .block_number
        }, |header| {
            header.num_entries = indtuples as u64;
        });

        // Register the segment's centroids in the router Vamana graph.
        if !centroids.is_empty() {
            let router_base = crate::access_method::agentvec::router::ensure_router_region(
                &index_rel,
                num_dimensions as u32,
            );
            crate::access_method::agentvec::router::add_segment_centroids(
                &index_rel,
                router_base,
                segment_id,
                &centroids,
            );
        }
        Some(segment_id)
    } else {
        None
    };

    // Always open an empty HOT segment for live inserts (the design's
    // invariant: a HOT segment exists from day one; bulk rows never go
    // through it).
    let _hot_id = insert::open_initial_hot_segment(&index_rel);

    return_finish(&index_rel, reltuples, indtuples)
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
