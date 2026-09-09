//! hnswsq index scan implementation.
//!
//! - `ambeginscan`: allocate the scan state (Rust `Box` hung off `opaque`).
//! - `amrescan`: extract + preprocess the query vector, reset the state.
//! - `amgettuple`: on the first call run the layered HNSW search and cache the
//!   ranked results, then emit one heap TID per call with
//!   `xs_recheckorderby = true` — the executor fetches the heap tuple (MVCC
//!   visibility) and recomputes the exact operator value, so the final
//!   ordering is exact over the candidates the graph produced.
//! - `amendscan`: drop the state (releases the last-returned-page pin).
//!
//! Reads only ever take ONE share content lock at a time (snapshot-and-
//! release inside `load_node_view`), so scans never contend with the insert
//! protocol beyond per-page atomicity, and stale node pointers into
//! vacuum-freed-and-reused pages resolve safely (see `node::load_node_view`).

use pgrx::*;

use crate::access_method::distance::{preprocess_cosine, DistanceType};
use crate::access_method::hnswsq::graph::{
    distance_encoded, greedy_descent, search_layer, DiskGraph,
};
use crate::access_method::hnswsq::insert::codec_for;
use crate::access_method::hnswsq::meta_page::HnswMetaPage;
use crate::access_method::hnswsq::node::load_node_view;
use crate::access_method::hnswsq::options::HNSWSQ_EF_SEARCH;
use crate::access_method::hnswsq::quantize;
use crate::access_method::pg_vector::PgVectorInternal;
use crate::util::buffer::PinnedBufferShare;
use crate::util::ItemPointer;

/// One cached scan result: approximate (stored-precision) distance, the heap
/// TID to emit, and the node location to pin on emit.
struct ScanResult {
    dist: f32,
    heap_tid: ItemPointer,
    node_ptr: ItemPointer,
}

/// Scan state for hnswsq index scans.
pub struct HnswScanState {
    /// Query vector (cosine-normalized when the index is cosine).
    query: Vec<f32>,
    /// Ranked results (ascending approximate distance), computed lazily.
    results: Vec<ScanResult>,
    /// Cursor into `results`.
    result_index: usize,
    /// Whether the search has run for the current query.
    results_computed: bool,
    /// Preallocated `xs_orderbyvals`/`xs_orderbynulls` slots (palloc'd once
    /// per scan in the executor's per-query context).
    orderbyvals: *mut pg_sys::Datum,
    orderbynulls: *mut bool,
    /// Pin on the page of the node backing the last-returned tuple (the
    /// amgettuple pinning contract).
    last_buffer: Option<PinnedBufferShare>,
}

impl HnswScanState {
    fn new(norderbys: i32) -> Self {
        let n = (norderbys.max(1)) as usize;
        unsafe {
            let orderbyvals = pg_sys::palloc(std::mem::size_of::<pg_sys::Datum>() * n)
                as *mut pg_sys::Datum;
            let orderbynulls =
                pg_sys::palloc(std::mem::size_of::<bool>() * n) as *mut bool;
            for i in 0..n {
                *orderbynulls.add(i) = true;
            }
            Self {
                query: Vec::new(),
                results: Vec::new(),
                result_index: 0,
                results_computed: false,
                orderbyvals,
                orderbynulls,
                last_buffer: None,
            }
        }
    }
}

/// Extract the query vector from an ORDER BY datum, normalizing for cosine.
unsafe fn extract_query_vector(datum: pg_sys::Datum, distance_type: DistanceType) -> Vec<f32> {
    let detoasted = pg_sys::pg_detoast_datum_copy(datum.cast_mut_ptr());
    let pg_vec = detoasted.cast::<PgVectorInternal>();
    let mut vec = (*pg_vec).to_slice().to_vec();
    pg_sys::pfree(detoasted.cast());

    if distance_type == DistanceType::Cosine {
        preprocess_cosine(&mut vec);
    }
    vec
}

/// Begin a scan of the hnswsq index.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambeginscan(
    index: pg_sys::Relation,
    nkeys: std::os::raw::c_int,
    norderbys: std::os::raw::c_int,
) -> pg_sys::IndexScanDesc {
    let scan = unsafe { pg_sys::RelationGetIndexScan(index, nkeys, norderbys) };
    let state = Box::new(HnswScanState::new(norderbys));
    unsafe {
        (*scan).opaque = Box::into_raw(state) as *mut std::os::raw::c_void;
    }
    scan
}

/// Rescan with a new query vector.
#[pg_guard]
pub unsafe extern "C-unwind" fn amrescan(
    scan: pg_sys::IndexScanDesc,
    _keys: pg_sys::ScanKey,
    _nkeys: std::os::raw::c_int,
    orderbys: pg_sys::ScanKey,
    norderbys: std::os::raw::c_int,
) {
    let state = unsafe { &mut *((*scan).opaque as *mut HnswScanState) };
    state.results.clear();
    state.result_index = 0;
    state.results_computed = false;
    state.last_buffer = None;

    let index_rel = unsafe { PgRelation::from_pg((*scan).indexRelation) };
    let meta = HnswMetaPage::fetch(&index_rel);
    let distance_type = meta.get_distance_type();

    if norderbys > 0 && !orderbys.is_null() {
        let orderby = unsafe { &*orderbys };
        if !orderby.sk_argument.is_null() {
            state.query = extract_query_vector(orderby.sk_argument, distance_type);
        }
    }
}

/// Run the layered search and cache the ranked live results.
///
/// Emitted distances are PROVABLE LOWER BOUNDS of the exact operator value in
/// the operator's own units — the contract `nodeIndexscan.c` enforces when
/// `xs_recheckorderby = true` (it recomputes the exact value per tuple and
/// errors with "index returned tuples in wrong order" when the index value
/// exceeds it; the reorder queue then restores the exact ordering).
///
/// With per-element quantization error `δ = v̂ − v`:
/// - L2 (`<->` = sqrt squared-L2): `‖q−v‖ ≥ ‖q−v̂‖ − ‖δ‖` → emit
///   `sqrt(max(0, d − 2√d·e − e²))` with `e ≥ ‖δ‖`;
/// - cosine (`<=>` = 1 − dot on normalized vectors) and IP (`<#>` = −dot):
///   `|dot(q,v̂) − dot(q,v)| ≤ ‖q‖·‖δ‖` → emit `d − ‖q‖·e`;
/// where `e = rel_err·‖v̂‖·margin` for the IEEE layouts and
/// `e = ‖scales‖/2` for SQ8 (plus a small absolute slack covering SIMD
/// accumulation differences against the executor's recomputation).  For
/// `plain`, `e = 0` and the emitted value is the exact distance minus slack.
unsafe fn compute_results(index_rel: &PgRelation, state: &mut HnswScanState) {
    state.results_computed = true;
    if state.query.is_empty() {
        return;
    }

    let meta = HnswMetaPage::fetch(index_rel);
    let distance_type = meta.get_distance_type();
    let codec = codec_for(index_rel, &meta);
    if codec.dim() != state.query.len() {
        // Query dimension mismatch (shouldn't happen through the executor):
        // return nothing rather than mis-decoding.
        return;
    }
    let Some(ep) = meta.get_entry_point() else {
        return; // empty index
    };
    let entry_level = meta.get_entry_level().max(0) as usize;

    let access = DiskGraph { index: index_rel };

    let Some(ep_view) = load_node_view(index_rel, ep) else {
        return; // entry vanished (crash orphan freed by vacuum)
    };
    let mut cur = (
        distance_encoded(&codec, distance_type, &state.query, &ep_view.vector),
        ep,
    );
    if entry_level > 0 {
        cur = greedy_descent(
            &codec,
            distance_type,
            &state.query,
            &access,
            cur,
            entry_level,
            1,
        );
    }

    let ef = (HNSWSQ_EF_SEARCH.get() as usize).max(1);
    let hits = search_layer(&codec, distance_type, &state.query, &access, vec![cur], ef, 0);
    let hits_len = hits.len();

    // Error-bound ingredients (see the function docs).
    let precision = meta.get_precision();
    let rel_err = quantize::relative_element_error(precision);
    // Margin covers ‖v‖ vs ‖v̂‖ (a factor (1+eps)/(1−eps)) and fp8 clamp
    // edge effects; generous but negligible against the base error.
    let rel_margin = match precision {
        quantize::HnswPrecision::IeeeFp8 => 1.15,
        quantize::HnswPrecision::IeeeFp16 => 1.01,
        _ => 1.0,
    };
    let sq8_half_norm = codec.sq8_scale_norm() / 2.0;
    let norm_q = state
        .query
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();

    // Emit only live nodes; tombstones keep routing but never surface.  The
    // node load also yields the heap TID and the decoded vector (for the
    // per-node error bound).
    state.results = hits
        .into_iter()
        .filter(|h| !h.deleted)
        .filter_map(|h| {
            let view = load_node_view(index_rel, h.id)?;
            if view.deleted || !view.heap_tid.is_valid() {
                return None;
            }
            let decoded = codec.decode(&view.vector);
            let norm_v = decoded.iter().map(|x| x * x).sum::<f32>().sqrt();
            let e = rel_err * rel_margin * norm_v + sq8_half_norm;
            let slack = 1e-4 * (1.0 + norm_q * norm_v);
            let d = h.dist;
            let dist = if view.clamped {
                // Encoding saturated components: no finite lower bound can be
                // proven from the index alone.  An ultra-conservative value
                // keeps the executor's cmp check valid; its reorder queue
                // restores the exact ordering.
                f32::NEG_INFINITY
            } else {
                match distance_type {
                    DistanceType::L2 => {
                        let sqrt_d = d.max(0.0).sqrt();
                        let lb = d - 2.0 * sqrt_d * e - e * e - slack * (1.0 + sqrt_d);
                        lb.max(0.0).sqrt()
                    }
                    DistanceType::Cosine => {
                        let lb = d - norm_q.max(1.0) * e - slack;
                        lb.max(0.0)
                    }
                    DistanceType::InnerProduct => d - norm_q * e - slack,
                }
            };
            Some(ScanResult {
                dist,
                heap_tid: view.heap_tid,
                node_ptr: h.id,
            })
        })
        .collect();
    // Per-node error bounds can make the lower bounds non-monotonic in the
    // approximate distance; re-sort so the executor's reorder queue drains
    // with as few extra pulls as possible (legality does not depend on this
    // — recheckorderby reorders — but latency does).
    state.results.sort_by(|a, b| {
        a.dist
            .total_cmp(&b.dist)
            .then_with(|| a.node_ptr.cmp(&b.node_ptr))
    });

    // Test-build diagnostics: verify the emitted set against the candidate
    // set (useful when debugging recall/membership issues).
    #[cfg(any(test, feature = "pg_test"))]
    {
        pgrx::log!(
            "hnswsq scan diag: ef={} candidates={} emitted={} entry_level={}",
            ef,
            hits_len,
            state.results.len(),
            entry_level
        );
    }
}

/// Get the next tuple from the hnswsq index scan.
#[pg_guard]
pub unsafe extern "C-unwind" fn amgettuple(
    scan: pg_sys::IndexScanDesc,
    _direction: pg_sys::ScanDirection::Type,
) -> bool {
    let state = unsafe { &mut *((*scan).opaque as *mut HnswScanState) };

    if !state.results_computed {
        let index_rel = unsafe { PgRelation::from_pg((*scan).indexRelation) };
        compute_results(&index_rel, state);
    }

    if state.result_index < state.results.len() {
        let res = &state.results[state.result_index];
        state.result_index += 1;

        unsafe {
            let mut tid_data = pg_sys::ItemPointerData::default();
            res.heap_tid.to_item_pointer_data(&mut tid_data);
            (*scan).xs_heaptid = tid_data;
            (*scan).xs_recheck = false;
            // Distances come from the stored (possibly reduced-precision)
            // vectors and are emitted as provable LOWER BOUNDS (see
            // compute_results): the executor rechecks the exact operator
            // value from the heap tuple for all layouts (also performs MVCC
            // visibility) and restores exact ordering via its reorder queue.
            (*scan).xs_recheckorderby = true;
            // pgvector's distance operators return float8: the orderbyval
            // datum MUST be a double — the executor compares it against the
            // recomputed float8 with the operator's sort support, and raw
            // f32 bits would be reinterpreted as a garbage f64.
            *state.orderbyvals =
                pg_sys::Datum::from((res.dist as f64).to_bits() as usize);
            *state.orderbynulls = false;
            (*scan).xs_orderbyvals = state.orderbyvals;
            (*scan).xs_orderbynulls = state.orderbynulls;

            // An index scan must keep a pin on the page holding the item it
            // last returned (postgres index-locking contract).
            let index_rel = PgRelation::from_pg((*scan).indexRelation);
            state.last_buffer =
                Some(PinnedBufferShare::read(&index_rel, res.node_ptr.block_number));
        }
        true
    } else {
        false
    }
}

/// End the scan: dropping the state releases the pinned buffer and the result
/// allocations.
#[pg_guard]
pub unsafe extern "C-unwind" fn amendscan(scan: pg_sys::IndexScanDesc) {
    unsafe {
        let ptr = (*scan).opaque as *mut HnswScanState;
        if !ptr.is_null() {
            drop(Box::from_raw(ptr));
            (*scan).opaque = std::ptr::null_mut();
        }
    }
}
