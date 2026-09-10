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

/// Build state shared with the heap-scan callback.
struct BuildState {
    distance_type: DistanceType,
    num_dimensions: usize,
    nrows: u64,
}

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
#[pg_guard]
pub unsafe extern "C-unwind" fn ambuild(
    heap: pg_sys::Relation,
    index: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    let heap_rel = PgRelation::from_pg(heap);
    let index_rel = PgRelation::from_pg(index);

    let distance_type = index_distance_type(index);
    let num_dimensions = index_dimensions(&index_rel);
    write_empty_index(&index_rel, distance_type, num_dimensions);

    let mut state = BuildState {
        distance_type,
        num_dimensions,
        nrows: 0,
    };
    pg_sys::IndexBuildHeapScan(
        heap_rel.as_ptr(),
        index_rel.as_ptr(),
        index_info,
        Some(build_callback),
        &mut state as *mut BuildState as *mut std::os::raw::c_void,
    );

    // Record the row count on the meta page (approximate by construction: it
    // counts rows appended, tombstoned ones included).
    AgentVecMetaPage::update(&index_rel, |meta| {
        meta.set_num_tuples(state.nrows);
    });

    let mut result = PgBox::<pg_sys::IndexBuildResult>::alloc0();
    result.heap_tuples = state.nrows as f64;
    result.index_tuples = state.nrows as f64;
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

/// Heap-scan callback: append one row.
unsafe extern "C-unwind" fn build_callback(
    index: pg_sys::Relation,
    tid: pg_sys::ItemPointer,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut std::os::raw::c_void,
) {
    if *isnull {
        return;
    }
    let state = &mut *(state as *mut BuildState);
    let index_rel = PgRelation::from_pg(index);
    let options = TSVAgentVecOptions::from_relation(&index_rel);
    let vector = insert::extract_vector(*values, state.num_dimensions);
    insert::insert_entry(
        &index_rel,
        ItemPointer::with_item_pointer_data(*tid),
        &vector,
        state.distance_type,
        &options,
    );
    state.nrows += 1;
}
