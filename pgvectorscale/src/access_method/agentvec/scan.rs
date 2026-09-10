//! AgentVec index scan.
//!
//! Scan state is built once, on the first `amgettuple()` call, and is
//! immutable afterwards — the design's lock-avoidance rule (§19): the
//! activated segment list and the candidates derived from it never consult
//! live mutable state again during execution.
//!
//! Phase 1 search is the exact `FLAT` executor:
//!
//! ```text
//! directory (immutable item) → searchable segments
//!      → per segment: frozen runs + active chain
//!           → exact distance per live entry
//!                → bounded top-N (or exhaustive) → distance order
//! ```
//!
//! Because the distances are exact and identical to what the ORDER BY
//! operator computes, `xs_recheckorderby` stays false: the executor can trust
//! the order we return and stop as soon as the query's LIMIT is satisfied.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use pgrx::*;

use crate::access_method::agentvec::directory::{AgentVecSegmentHeader, SegmentAlgorithm};
use crate::access_method::agentvec::flat;
use crate::access_method::agentvec::meta_page::AgentVecMetaPage;
use crate::access_method::agentvec::options::TSVAgentVecOptions;
use crate::access_method::distance::{preprocess_cosine, DistanceType};
use crate::access_method::pg_vector::PgVectorInternal;
use crate::util::ItemPointer;

/// A (distance, heap tid) candidate ordered by distance, used as the element
/// type of the bounded top-N max-heap.
#[derive(PartialEq)]
struct DistTid {
    dist: f32,
    tid: ItemPointer,
}

impl Eq for DistTid {}

impl PartialOrd for DistTid {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistTid {
    fn cmp(&self, other: &Self) -> Ordering {
        // total_cmp gives a total order on f32 (including NaN); tie-break by
        // tid so Ord stays consistent with the derived PartialEq/Eq.
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.tid.cmp(&other.tid))
    }
}

/// Scan state for AgentVec index scans.
pub struct AgentVecScanState {
    /// Exactly one order-by key is supported; empty means "no query".
    query: Vec<f32>,
    /// Whether the scan was given non-order-by keys, which this AM does not
    /// evaluate itself and therefore requires the executor to recheck.
    has_keys: bool,
    /// Materialized candidates in ascending distance order.
    results: Vec<(f32, ItemPointer)>,
    /// Position in `results`.
    result_index: usize,
    /// Whether `results` has been materialized for the current scan keys.
    results_computed: bool,
}

impl AgentVecScanState {
    fn new() -> Self {
        Self {
            query: Vec::new(),
            has_keys: false,
            results: Vec::new(),
            result_index: 0,
            results_computed: false,
        }
    }

    fn reset(&mut self) {
        self.results.clear();
        self.result_index = 0;
        self.results_computed = false;
    }
}

/// Extract a query vector from a `vector` datum, normalizing it for cosine
/// (the stored vectors are normalized the same way).
unsafe fn extract_query_vector(datum: pg_sys::Datum, distance_type: DistanceType) -> Vec<f32> {
    let detoasted = pg_sys::pg_detoast_datum_copy(datum.cast_mut_ptr());
    let pg_vec = detoasted.cast::<PgVectorInternal>();
    let mut vector = (*pg_vec).to_slice().to_vec();
    pg_sys::pfree(detoasted.cast());

    if distance_type == DistanceType::Cosine {
        preprocess_cosine(&mut vector);
    }
    vector
}

/// Begin a scan.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambeginscan(
    index: pg_sys::Relation,
    nkeys: std::os::raw::c_int,
    norderbys: std::os::raw::c_int,
) -> pg_sys::IndexScanDesc {
    let scan = pg_sys::RelationGetIndexScan(index, nkeys, norderbys);

    let state = AgentVecScanState::new();
    let state_ptr = pg_sys::palloc0(std::mem::size_of::<AgentVecScanState>())
        as *mut AgentVecScanState;
    *state_ptr = state;
    (*scan).opaque = state_ptr as *mut std::os::raw::c_void;

    scan
}

/// Reset a scan with new keys.
#[pg_guard]
pub unsafe extern "C-unwind" fn amrescan(
    scan: pg_sys::IndexScanDesc,
    _keys: pg_sys::ScanKey,
    nkeys: std::os::raw::c_int,
    orderbys: pg_sys::ScanKey,
    norderbys: std::os::raw::c_int,
) {
    let state = &mut *((*scan).opaque as *mut AgentVecScanState);
    state.reset();
    state.has_keys = nkeys > 0;

    let index_rel = PgRelation::from_pg((*scan).indexRelation);
    let meta = AgentVecMetaPage::fetch(&index_rel);

    if norderbys > 0 && !orderbys.is_null() {
        let orderby = &*orderbys;
        if !orderby.sk_argument.is_null() {
            state.query = extract_query_vector(orderby.sk_argument, meta.get_distance_type());
        }
    }

    if norderbys > 1 {
        error!("agentvec: only one ORDER BY distance key is supported");
    }
}

/// Materialize the candidates for the current scan keys.
unsafe fn compute_results(scan: pg_sys::IndexScanDesc, state: &mut AgentVecScanState) {
    let index_rel = PgRelation::from_pg((*scan).indexRelation);
    let meta = AgentVecMetaPage::fetch(&index_rel);
    let options = TSVAgentVecOptions::from_relation(&index_rel);
    let distance_fn = meta.get_distance_type().get_distance_function();
    let bound = options.get_search_candidates();
    let dim = meta.get_num_dimensions() as usize;

    let directory = meta.load_directory(&index_rel);
    let mut all: Vec<(f32, ItemPointer)> = Vec::new();
    let mut heap: BinaryHeap<DistTid> = BinaryHeap::new();
    // Reused across entries so the scan performs one allocation, not one per
    // candidate.
    let mut vector_scratch: Vec<f32> = Vec::with_capacity(dim);

    for segment in directory.searchable() {
        match segment.algorithm() {
            SegmentAlgorithm::Flat => {}
            other => error!(
                "agentvec: segment {} uses {} storage, which this version cannot search",
                segment.segment_id,
                other.as_str()
            ),
        }
        let header = AgentVecSegmentHeader::load(&index_rel, segment.header);
        for chain_start in header.chain_starts() {
            flat::for_each_entry(&index_rel, chain_start, |_block, _offset, bytes| {
                if flat::decode_state(bytes) != flat::STATE_LIVE {
                    return;
                }
                let decoded = flat::decode_vector_into(bytes, &mut vector_scratch);
                if decoded != dim {
                    panic!(
                        "agentvec: segment {} has a {}-dimensional entry, expected {}",
                        segment.segment_id, decoded, dim
                    );
                }
                let dist = distance_fn(&state.query, &vector_scratch);
                let candidate = DistTid {
                    dist,
                    tid: flat::decode_tid(bytes),
                };
                match bound {
                    // Exhaustive: exact ordered scan, so query semantics never
                    // depend on a candidate bound.
                    None => all.push((candidate.dist, candidate.tid)),
                    Some(k) => {
                        if heap.len() < k {
                            heap.push(candidate);
                        } else if let Some(mut worst) = heap.peek_mut() {
                            if candidate.dist < worst.dist {
                                *worst = candidate;
                            }
                        }
                    }
                }
            });
        }
    }

    state.results = match bound {
        None => {
            all.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            all
        }
        Some(_) => heap
            .into_sorted_vec()
            .into_iter()
            .map(|c| (c.dist, c.tid))
            .collect(),
    };
    state.results_computed = true;
}

/// Return the next tuple of the scan.
#[pg_guard]
pub unsafe extern "C-unwind" fn amgettuple(
    scan: pg_sys::IndexScanDesc,
    _direction: pg_sys::ScanDirection::Type,
) -> bool {
    let state = &mut *((*scan).opaque as *mut AgentVecScanState);

    if !state.results_computed {
        if state.query.is_empty() {
            // No ORDER BY key: this AM cannot produce an ordered candidate
            // stream, and an unordered one would silently undercount.
            state.results_computed = true;
            state.results.clear();
            return false;
        }
        compute_results(scan, state);
    }

    if state.result_index >= state.results.len() {
        return false;
    }
    let (distance, heap_tid) = state.results[state.result_index];
    state.result_index += 1;

    let mut tid_data = pg_sys::ItemPointerData::default();
    heap_tid.to_item_pointer_data(&mut tid_data);
    (*scan).xs_heaptid = tid_data;
    // This AM evaluates no index quals of its own: anything the planner passed
    // as a scan key must be rechecked against the heap tuple.
    (*scan).xs_recheck = state.has_keys;
    // FLAT distances are exact and use the same formula as the ordering
    // operator, so the order is trustworthy and no recheck is needed.  (An
    // approximate segment algorithm must set this to true and supply
    // `xs_orderbyvals` as float8 datums.)
    (*scan).xs_recheckorderby = false;
    let _ = distance;

    true
}

/// End a scan.
#[pg_guard]
pub unsafe extern "C-unwind" fn amendscan(scan: pg_sys::IndexScanDesc) {
    let state = (*scan).opaque as *mut AgentVecScanState;
    if !state.is_null() {
        // Run the Rust destructor first: pfree only releases the palloc'd
        // struct, it does not drop the `Vec` buffers inside it.
        std::ptr::drop_in_place(state);
        pg_sys::pfree(state as *mut std::os::raw::c_void);
        (*scan).opaque = std::ptr::null_mut();
    }
}
