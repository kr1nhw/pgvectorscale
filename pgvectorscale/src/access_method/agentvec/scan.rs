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

use crate::access_method::agentvec::directory::{
    AgentVecSegmentHeader, SegmentAlgorithm, SegmentOwnership,
};
use crate::access_method::agentvec::flat;
use crate::access_method::agentvec::meta_page::AgentVecMetaPage;
use crate::access_method::agentvec::options::TSVAgentVecOptions;
use crate::access_method::agentvec::router;
use crate::access_method::distance::{preprocess_cosine, DistanceType};
use crate::access_method::hnswsq::options::{HNSW_EF_SEARCH, HNSW_SQ8_DISTANCE};
use crate::access_method::hnswsq::quantize::HnswPrecision;
use crate::access_method::hnswsq::types::{Element, Visited};
use crate::access_method::hnswsq::utils::SearchScratch;
use crate::access_method::hnswsq::utils;
use crate::access_method::pg_vector::PgVectorInternal;
use crate::util::ItemPointer;

/// A (distance, heap tid) candidate ordered by distance, used as the element
/// type of the bounded top-N max-heap.  `exact` is false for candidates whose
/// distance is a lower bound (HNSW quantized layouts, IVF estimates): those
/// are emitted with `xs_recheckorderby = true` and their bound as the orderby
/// value, so the executor's reorder queue restores the exact ordering.
#[derive(PartialEq)]
struct DistTid {
    dist: f64,
    tid: ItemPointer,
    exact: bool,
}

impl Eq for DistTid {}

impl PartialOrd for DistTid {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistTid {
    fn cmp(&self, other: &Self) -> Ordering {
        // total_cmp gives a total order on f64 (including NaN and -infinity,
        // which clamped HNSW elements emit); tie-break by tid so Ord stays
        // consistent with the derived PartialEq/Eq.
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
    results: Vec<(f64, ItemPointer, bool)>,
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
    if detoasted != datum.cast_mut_ptr() {
        pg_sys::pfree(detoasted.cast());
    }

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
            let expected_dim = meta.get_num_dimensions() as usize;
            if state.query.len() != expected_dim {
                error!("different vector dimensions");
            }
        }
    }

    if norderbys > 1 {
        error!("agentvec: only one ORDER BY distance key is supported");
    }
}

/// Materialize the candidates for the current scan keys.
///
/// One snapshot of the directory, then per-algorithm execution:
/// * `FLAT` segments contribute exact distances (every live entry when
///   `search_candidates` is 0);
/// * `HNSW` segments run the embedded hnswsq region's one-shot search with
///   `ef = search_candidates` (or `hnswsq.ef_search` when unbounded) and
///   contribute their per-element values: exact for the `plain` layout,
///   provable lower bounds for the quantized layouts.
unsafe fn compute_results(scan: pg_sys::IndexScanDesc, state: &mut AgentVecScanState) {
    let index_rel = PgRelation::from_pg((*scan).indexRelation);
    let meta = AgentVecMetaPage::fetch(&index_rel);
    let options = TSVAgentVecOptions::from_relation(&index_rel);
    let distance_fn = meta.get_distance_type().get_distance_function();
    let bound = options.get_search_candidates();
    let dim = meta.get_num_dimensions() as usize;
    let norm_q = state.query.iter().map(|x| x * x).sum::<f32>().sqrt();

    let directory = meta.load_directory(&index_rel);
    // The deterministic router (phase 3.5): navigate the centroid Vamana
    // graph once per scan and activate only the top `router_top_m` owned IVF
    // segments.  HOT/FLAT segments are always searched; external segments
    // have no owned centroids and are activated as a whole.
    let routed = router::route(
        &index_rel,
        &state.query,
        &directory,
        &options,
        meta.get_router_base(),
    );
    let mut all: Vec<(f64, ItemPointer, bool)> = Vec::new();
    let mut heap: BinaryHeap<DistTid> = BinaryHeap::new();
    // Reused across entries so the scan performs one allocation, not one per
    // candidate.
    let mut vector_scratch: Vec<f32> = Vec::with_capacity(dim);

    let mut push = |all: &mut Vec<(f64, ItemPointer, bool)>,
                    heap: &mut BinaryHeap<DistTid>,
                    candidate: DistTid| {
        match bound {
            None => all.push((candidate.dist, candidate.tid, candidate.exact)),
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
    };

    for segment in directory.searchable() {
        match segment.algorithm() {
            SegmentAlgorithm::Flat => {
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
                        push(
                            &mut all,
                            &mut heap,
                            DistTid {
                                dist: dist as f64,
                                tid: flat::decode_tid(bytes),
                                exact: true,
                            },
                        );
                    });
                }
            }
            SegmentAlgorithm::Hnsw => {
                let base = segment.code_root.block_number;
                let support = utils::init_support(index_rel.as_ptr(), base);
                let q = if state.query.is_empty() {
                    None
                } else {
                    Some(state.query.as_slice())
                };
                let ef = match bound {
                    Some(k) => k,
                    None => (HNSW_EF_SEARCH.get() as usize).max(1),
                };
                // SQ8: quantize the query once per segment search (the same
                // convention as the hnswsq AM scan).
                let qstate = q
                    .map(|q| utils::sq8_query_state(&support, q, HNSW_SQ8_DISTANCE.get()))
                    .flatten();

                let mut m = 0usize;
                utils::get_meta_page_info(index_rel.as_ptr(), base, Some(&mut m), None);
                let mut visited = Visited::new(1000 * m * 2);
                let mut scratch = SearchScratch::new(m);
                let mut tuples: i64 = 0;
                let (candidates, _m) = crate::access_method::hnswsq::scan::region_candidates(
                    index_rel.as_ptr(),
                    base,
                    &support,
                    q,
                    ef,
                    &mut visited,
                    &mut scratch,
                    qstate.as_ref(),
                    &mut tuples,
                );
                // The search returns furthest-first; drain from the back so
                // candidates enter the merge in ascending bound order.
                if let Some(k) = bound {
                    // Bounded scans rerank every segment exactly: keep the
                    // top-k by the quantized bound, heap-fetch their vectors,
                    // and emit exact distances — the whole stream is then
                    // exact (no mixed recheckorderby, which the executor's
                    // reorder machinery does not support).
                    let mut est_heap: BinaryHeap<DistTid> = BinaryHeap::new();
                    for sc in candidates.into_iter().rev() {
                        let element =
                            crate::access_method::hnswsq::ptr::access::<Element>(
                                std::ptr::null_mut(),
                                sc.element,
                            );
                        if (*element).deleted != 0 {
                            continue;
                        }
                        let dist = crate::access_method::hnswsq::scan::emit_candidate(
                            &support,
                            &state.query,
                            norm_q,
                            &sc,
                        );
                        let cand = DistTid {
                            dist,
                            tid: ItemPointer::with_item_pointer_data((*element).heaptid),
                            exact: false,
                        };
                        if est_heap.len() < k {
                            est_heap.push(cand);
                        } else if let Some(mut worst) = est_heap.peek_mut() {
                            if cand.dist < worst.dist {
                                *worst = cand;
                            }
                        }
                    }
                    for cand in est_heap {
                        let mut tid_data = pg_sys::ItemPointerData::default();
                        cand.tid.to_item_pointer_data(&mut tid_data);
                        if let Some(vector) = super::insert::fetch_heap_vector(
                            &index_rel,
                            tid_data,
                            dim,
                            meta.get_distance_type(),
                        ) {
                            let dist = distance_fn(&state.query, &vector);
                            push(
                                &mut all,
                                &mut heap,
                                DistTid {
                                    dist: dist as f64,
                                    tid: cand.tid,
                                    exact: true,
                                },
                            );
                        }
                    }
                } else {
                    let exact = support.precision == HnswPrecision::Plain;
                    for sc in candidates.into_iter().rev() {
                        let element =
                            crate::access_method::hnswsq::ptr::access::<Element>(
                                std::ptr::null_mut(),
                                sc.element,
                            );
                        if (*element).deleted != 0 {
                            continue;
                        }
                        let dist =
                            crate::access_method::hnswsq::scan::emit_candidate(
                                &support,
                                &state.query,
                                norm_q,
                                &sc,
                            );
                        push(
                            &mut all,
                            &mut heap,
                            DistTid {
                                dist,
                                tid: ItemPointer::with_item_pointer_data((*element).heaptid),
                                exact,
                            },
                        );
                    }
                }
            }
            SegmentAlgorithm::IvfRaBitQ => {
                // The router may have deactivated this segment: `route`
                // returns None (search all) when it cannot decide.
                if segment.ownership() == SegmentOwnership::Owned {
                    if let Some(active) = &routed {
                        if !active.contains(&segment.segment_id) {
                            continue;
                        }
                    }
                }
                // The immutable IVF payload: the same composition the ivf AM
                // scan performs, over the embedded region's meta/centroids/
                // list directory.
                let base = segment.code_root.block_number;
                let ivf_meta = crate::access_method::ivf::meta_page::IvfMetaPage::fetch(
                    &index_rel,
                    base,
                );
                let Some(centroid_pointer) = ivf_meta.get_centroids_pointer() else {
                    continue;
                };
                let centroid_page =
                    crate::access_method::ivf::centroid_page::IvfCentroidPage::load(
                        &index_rel,
                        centroid_pointer,
                    );
                if centroid_page.centroids.is_empty() {
                    continue;
                }
                let Some(list_directory_pointer) = ivf_meta.get_list_directory_pointer() else {
                    continue;
                };
                let list_directory =
                    crate::access_method::ivf::list_directory::IvfListDirectory::load_at(
                        &index_rel,
                        list_directory_pointer,
                    );

                let probes = options.get_ivf_probes() as usize;
                let nearest = crate::access_method::ivf::simd::find_nearest_centroids(
                    &state.query,
                    &centroid_page.centroids,
                    meta.get_distance_type(),
                    probes,
                );
                // The RaBitQ estimate is not a guaranteed lower bound, so it
                // cannot be the executor's orderby hint ("index returned
                // tuples in wrong order" otherwise).  With a bounded scan the
                // candidates are exact-reranked HERE: keep the top-k by
                // estimate, heap-fetch their vectors, and emit exact
                // distances (phase 8's rerank, done at emission).  The
                // unbounded scan keeps -infinity hints (the executor's
                // reorder queue restores order; phase 8 bounds it too).
                if let Some(k) = bound {
                    let mut est_heap: BinaryHeap<DistTid> = BinaryHeap::new();
                    crate::access_method::ivf::scan::for_each_candidate(
                        &index_rel,
                        &ivf_meta,
                        &centroid_page,
                        &list_directory,
                        &state.query,
                        &nearest,
                        |dist, tid| {
                            let cand = DistTid {
                                dist: dist as f64,
                                tid,
                                exact: false,
                            };
                            if est_heap.len() < k {
                                est_heap.push(cand);
                            } else if let Some(mut worst) = est_heap.peek_mut() {
                                if cand.dist < worst.dist {
                                    *worst = cand;
                                }
                            }
                        },
                    );
                    for cand in est_heap {
                        let mut tid_data = pg_sys::ItemPointerData::default();
                        cand.tid.to_item_pointer_data(&mut tid_data);
                        if let Some(vector) = super::insert::fetch_heap_vector(
                            &index_rel,
                            tid_data,
                            dim,
                            meta.get_distance_type(),
                        ) {
                            let dist = distance_fn(&state.query, &vector);
                            push(
                                &mut all,
                                &mut heap,
                                DistTid {
                                    dist: dist as f64,
                                    tid: cand.tid,
                                    exact: true,
                                },
                            );
                        }
                    }
                } else {
                    crate::access_method::ivf::scan::for_each_candidate(
                        &index_rel,
                        &ivf_meta,
                        &centroid_page,
                        &list_directory,
                        &state.query,
                        &nearest,
                        |_dist, tid| {
                            push(
                                &mut all,
                                &mut heap,
                                DistTid {
                                    dist: f64::NEG_INFINITY,
                                    tid,
                                    exact: false,
                                },
                            );
                        },
                    );
                }
            }
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
            .map(|c| (c.dist, c.tid, c.exact))
            .collect(),
    };
    // Mixed scans must not mix tuples with and without `xs_recheckorderby`:
    // the executor's `IndexNextWithReorder` calls `cmp_orderbyvals` on every
    // tuple it pulls while reordering and dereferences `xs_orderbyvals`
    // (NULL for exact tuples) — a bulk IVF segment (exact emission-time
    // rerank) plus a quantized HOT segment in one scan segfaults otherwise.
    // When any candidate is approximate, downgrade the whole stream: the
    // exact distances stay exact, they just become orderby hints.
    if state.results.iter().any(|(_, _, exact)| !exact) {
        for entry in state.results.iter_mut() {
            entry.2 = false;
        }
    }
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
    let (distance, heap_tid, exact) = state.results[state.result_index];
    state.result_index += 1;

    let mut tid_data = pg_sys::ItemPointerData::default();
    heap_tid.to_item_pointer_data(&mut tid_data);
    (*scan).xs_heaptid = tid_data;
    // This AM evaluates no index quals of its own: anything the planner passed
    // as a scan key must be rechecked against the heap tuple.
    (*scan).xs_recheck = state.has_keys;

    if exact {
        // FLAT and HNSW-plain distances are exact and use the same formula as
        // the ordering operator, so the order is trustworthy and no recheck
        // is needed.
        (*scan).xs_recheckorderby = false;
        (*scan).xs_orderbyvals = std::ptr::null_mut();
        (*scan).xs_orderbynulls = std::ptr::null_mut();
    } else {
        // Approximate: the value is a lower bound (HNSW quantized layouts, and
        // later IVF estimates).  Publish it as a float8 datum — the ordering
        // operator's type — so the executor's reorder queue can compare it
        // against the recomputed exact value.
        (*scan).xs_recheckorderby = true;
        let orderbyvals =
            pg_sys::palloc(std::mem::size_of::<pg_sys::Datum>()) as *mut pg_sys::Datum;
        let orderbynulls = pg_sys::palloc(std::mem::size_of::<bool>()) as *mut bool;
        *orderbyvals = pg_sys::Datum::from(distance.to_bits() as usize);
        *orderbynulls = false;
        (*scan).xs_orderbyvals = orderbyvals;
        (*scan).xs_orderbynulls = orderbynulls;
    }

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
