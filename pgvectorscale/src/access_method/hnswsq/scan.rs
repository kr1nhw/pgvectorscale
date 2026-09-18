//! hnswsq index scan — the Rust translation of pgvector's `hnswscan.c`
//! (Algorithm 5 + the iterative scan), with the old engine's order-by
//! contract:
//!
//! * `plain` (lossless): the emitted distances ARE the operator's values and
//!   `xs_recheckorderby` is false — exactly pgvector's contract for `vector`
//!   columns; the executor trusts the index order.
//! * quantized layouts: the emitted value is a provable lower bound of the
//!   exact operator value and `xs_recheckorderby` is true, so the executor
//!   recomputes the exact value per tuple and restores exact ordering.
//!
//! Divergences from the reference, and only these: the scan keeps a pin-less
//! heap-TID emission (pgvector returns heap TIDs with no index-page pin), the
//! quantized layouts materialize the encoded vectors of admitted candidates
//! (the reference never materializes values in a scan), and the query vector
//! is a decoded/normalized `Vec<f32>`.

use pgrx::pg_sys;
use pgrx::*;

use crate::access_method::distance::{preprocess_cosine, DistanceType};
use crate::access_method::hnswsq::quantize::{self, HnswPrecision};
use crate::access_method::hnswsq::options::HNSW_EF_SEARCH;
use crate::access_method::hnswsq::options::{
    HNSW_ITERATIVE_SCAN, HNSW_MAX_SCAN_TUPLES, HNSW_SCAN_MEM_MULTIPLIER,
    ITERATIVE_SCAN_OFF, ITERATIVE_SCAN_STRICT,
};
use crate::access_method::hnswsq::types::*;
use crate::access_method::hnswsq::quantize::Sq8QueryState;
use crate::access_method::hnswsq::utils::*;
use crate::access_method::pg_vector::PgVectorInternal;

/// The scan state (pgvector `HnswScanOpaqueData` + the emission contract).
pub struct ScanState {
    pub support: Support,
    /// Base block of the hnswsq region (0 for the standalone AM).
    pub base: pg_sys::BlockNumber,
    pub first: bool,
    /// The result candidates, furthest-first (pgvector drains `llast`).
    pub w: Vec<SearchCandidate>,
    pub visited: Visited,
    pub discarded: Option<CandidateHeap>,
    /// Decoded + normalized query; empty = null query (no results).
    pub q: Vec<f32>,
    /// SQ8 per-query integer distance state (see `sq8_query_state`);
    /// recomputed in `get_scan_items` from `q`.
    pub qstate: Option<quantize::Sq8QueryState>,
    pub m: usize,
    pub tuples: i64,
    pub previous_distance: f64,
    pub max_memory: usize,
    pub scratch: SearchScratch,
    pub tmp_ctx: PgMemoryContexts,
    // Emission
    pub orderbyvals: *mut pg_sys::Datum,
    pub orderbynulls: *mut bool,
    /// False for the lossless `plain` layout (exact order, no recheck).
    pub recheck_orderby: bool,
    pub norm_q: f32,
}

/// Whether the scan materializes the encoded vectors of admitted candidates
/// (the quantized layouts need them for their lower bounds; `plain` never
/// does — the pgvector reference never materializes in a scan at all).
fn scan_load_vec(precision: HnswPrecision) -> bool {
    precision != HnswPrecision::Plain
}

/// `GetScanItems` (hnswscan.c): layered descent (ef = 1) then the layer-0
/// search with `ef_search`.
unsafe fn get_scan_items(state: &mut ScanState, index: pg_sys::Relation) -> Vec<SearchCandidate> {
    let q = if state.q.is_empty() {
        None
    } else {
        Some(state.q.as_slice())
    };

    // SQ8: quantize the query once for the whole scan (see sq8_query_state).
    state.qstate = q
        .map(|q| {
            crate::access_method::hnswsq::utils::sq8_query_state(
                &state.support,
                q,
                crate::access_method::hnswsq::options::HNSW_SQ8_DISTANCE.get(),
            )
        })
        .flatten();

    let ef = (HNSW_EF_SEARCH.get() as usize).max(1);
    let (w, m) = region_candidates(
        index,
        state.base,
        &state.support,
        q,
        ef,
        &mut state.visited,
        &mut state.scratch,
        state.qstate.as_ref(),
        &mut state.tuples,
    );
    state.m = m;
    w
}

/// One-shot materialized search over an hnswsq region (the standalone AM's
/// whole index, or an embedded AgentVec HOT segment at `base`): layered
/// descent (ef = 1) then the layer-0 search with `ef`.
///
/// Returns the candidates in furthest-first order (drain from the back),
/// with their encoded values materialized in `scratch`'s arena, and the
/// region's `m`.  [`emit_candidate`] turns each result into the value the
/// scan would publish.
#[allow(clippy::too_many_arguments)]
pub unsafe fn region_candidates(
    index: pg_sys::Relation,
    base: pg_sys::BlockNumber,
    support: &Support,
    q: Option<&[f32]>,
    ef: usize,
    visited: &mut Visited,
    scratch: &mut SearchScratch,
    qstate: Option<&Sq8QueryState>,
    tuples: &mut i64,
) -> (Vec<SearchCandidate>, usize) {
    let load_vec = scan_load_vec(support.precision);

    // Get m and entry point
    let mut m = 0usize;
    let mut entry = None;
    get_meta_page_info(index, base, Some(&mut m), Some(&mut entry));

    let Some(entry) = entry else {
        return (Vec::new(), m);
    };

    // The entry element must outlive the search results (the candidates in
    // `w` reference it until the caller drains them), so it lives in the
    // caller's element arena like every other materialized element.
    let entry_ptr = scratch.elements.alloc();
    std::ptr::copy_nonoverlapping(&*entry, entry_ptr, 1);

    let mut ep = vec![entry_candidate(
        std::ptr::null_mut(),
        entry_ptr,
        q,
        Some(index),
        support,
        load_vec,
        qstate,
    )];
    let entry_level = (*entry_ptr).level as usize;

    // Descent: the scratch elements persist (admitted candidates at upper
    // layers become entry points below), like pgvector's tmpCtx.
    for lc in (1..=entry_level).rev() {
        let w = search_layer(
            std::ptr::null_mut(),
            Some(index),
            support,
            m,
            q,
            &ep,
            1,
            lc,
            load_vec,
            None,
            visited,
            None,
            true,
            None,
            scratch,
            qstate,
        );
        ep = w;
    }

    let mut discarded: CandidateHeap = CandidateHeap::with_capacity(ef + 1);
    let w = search_layer(
        std::ptr::null_mut(),
        Some(index),
        support,
        m,
        q,
        &ep,
        ef,
        0,
        load_vec,
        None,
        visited,
        Some(&mut discarded),
        true,
        Some(tuples),
        scratch,
        qstate,
    );
    drop(discarded);
    (w, m)
}

/// `ResumeScanItems` (hnswscan.c): continue the layer-0 search from the
/// discarded candidates.
unsafe fn resume_scan_items(state: &mut ScanState, index: pg_sys::Relation) -> Vec<SearchCandidate> {
    let load_vec = scan_load_vec(state.support.precision);
    let batch_size = (HNSW_EF_SEARCH.get() as usize).max(1);

    let mut ep: Vec<SearchCandidate> = Vec::new();
    for _ in 0..batch_size {
        let Some(sc) = state.discarded.as_mut().and_then(|d| d.pop()) else {
            break;
        };
        ep.push(sc.0);
    }
    if ep.is_empty() {
        return Vec::new();
    }

    let q = if state.q.is_empty() {
        None
    } else {
        Some(state.q.as_slice())
    };
    let discarded = state.discarded.as_mut().expect("iterative scan owns its heap");
    search_layer(
        std::ptr::null_mut(),
        Some(index),
        &state.support,
        state.m,
        q,
        &ep,
        batch_size,
        0,
        load_vec,
        None,
        &mut state.visited,
        Some(discarded),
        false,
        Some(&mut state.tuples),
        &mut state.scratch,
        state.qstate.as_ref(),
    )
}

/// The approximate memory an iterative scan's state occupies (pgvector reads
/// `MemoryContextMemAllocated`; ours lives in Rust scratch buffers).
fn scan_memory(state: &ScanState) -> usize {
    let vec_bytes = state.support.codec.vector_bytes();
    state.scratch.elements.len() * (std::mem::size_of::<Element>() + vec_bytes)
        + state.scratch.tids.capacity() * std::mem::size_of::<pg_sys::ItemPointerData>()
        + state.visited.capacity_bytes()
}

/// The emitted order-by value: the exact operator value for `plain`, a
/// provable lower bound for the quantized layouts (see the module docs and
/// the old engine's scan for the error-bound derivation).
unsafe fn emit_distance(state: &ScanState, sc: &SearchCandidate) -> f64 {
    emit_candidate(&state.support, &state.q, state.norm_q, sc)
}

/// The value a scan publishes for one materialized candidate: the exact
/// operator value for the `plain` layout, and a provable per-element lower
/// bound in the operator's units for the quantized layouts (`clamped`
/// elements emit -infinity).  This is the single source of the emission
/// contract for both the `hnswsq` AM scan and AgentVec's merged scan over an
/// embedded HOT segment.
pub unsafe fn emit_candidate(
    support: &Support,
    q: &[f32],
    norm_q: f32,
    sc: &SearchCandidate,
) -> f64 {
    let base = std::ptr::null_mut();
    let element = crate::access_method::hnswsq::ptr::access::<Element>(base, sc.element);
    let vec_bytes = support.codec.vector_bytes();

    if support.precision == HnswPrecision::Plain {
        // Lossless layout: the stored distance IS the operator's value (L2
        // additionally applies the sqrt the operator applies).
        return match support.dist_type {
            DistanceType::L2 => (sc.distance.max(0.0)).sqrt() as f64,
            _ => sc.distance as f64,
        };
    }

    // Quantized: per-element error bound, in the operator's units.
    let value = get_value(base, element, vec_bytes);
    let mut decoded = vec![0.0f32; support.codec.dim()];
    support.codec.decode_into(value, decoded.as_mut_slice());
    let norm_v = decoded.iter().map(|x| x * x).sum::<f32>().sqrt();

    let rel_err = quantize::relative_element_error(support.precision);
    let rel_margin = match support.precision {
        quantize::HnswPrecision::IeeeFp8 => 1.15,
        quantize::HnswPrecision::IeeeFp16 => 1.01,
        _ => 1.0,
    };
    let sq_error_norm = support.codec.quant_error_norm();
    let e = rel_err * rel_margin * norm_v + sq_error_norm;
    let slack = 1e-4 * (1.0 + norm_q * norm_v);

    // The search distance may come from an integer-distance form
    // (hnswsq.sq8_distance) whose quantity differs from the operator's
    // distance over the decoded vector; the lower-bound proof below needs
    // the true decoded distance, so recompute it per emitted tuple (a scan
    // emits only ~LIMIT tuples — cheap).
    let d = match support.dist_type {
        DistanceType::L2 => {
            let mut acc = 0.0f32;
            for i in 0..support.codec.dim() {
                let diff = q[i] - decoded[i];
                acc += diff * diff;
            }
            acc
        }
        DistanceType::Cosine => {
            let mut dot = 0.0f32;
            for i in 0..support.codec.dim() {
                dot += q[i] * decoded[i];
            }
            (1.0 - dot).max(0.0)
        }
        DistanceType::InnerProduct => {
            let mut dot = 0.0f32;
            for i in 0..support.codec.dim() {
                dot += q[i] * decoded[i];
            }
            -dot
        }
    };

    if (*element).clamped != 0 {
        // Encoding saturated components: no finite lower bound is provable.
        // An ultra-conservative value keeps the executor's cmp check valid;
        // its reorder queue restores the exact ordering.
        return f64::NEG_INFINITY;
    }

    match support.dist_type {
        DistanceType::L2 => {
            let sqrt_d = d.max(0.0).sqrt();
            let lb = d - 2.0 * sqrt_d * e - e * e - slack * (1.0 + sqrt_d);
            (lb.max(0.0).sqrt()) as f64
        }
        DistanceType::Cosine => {
            let lb = d - norm_q.max(1.0) * e - slack;
            (lb.max(0.0)) as f64
        }
        DistanceType::InnerProduct => (d - norm_q * e - slack) as f64,
    }
}

/// `hnswbeginscan` (hnswscan.c).
#[pg_guard]
pub unsafe extern "C-unwind" fn ambeginscan(
    index: pg_sys::Relation,
    nkeys: std::os::raw::c_int,
    norderbys: std::os::raw::c_int,
) -> pg_sys::IndexScanDesc {
    let scan = pg_sys::RelationGetIndexScan(index, nkeys, norderbys);
    let support = init_support(index, HNSW_STANDALONE_BASE);
    let (m, _ef_construction) = region_params(index, HNSW_STANDALONE_BASE);
    let tmp_ctx = PgMemoryContexts::new("hnswsq scan temporary context");

    let n = norderbys.max(1) as usize;
    let orderbyvals =
        pg_sys::palloc(std::mem::size_of::<pg_sys::Datum>() * n) as *mut pg_sys::Datum;
    let orderbynulls = pg_sys::palloc(std::mem::size_of::<bool>() * n) as *mut bool;
    for i in 0..n {
        *orderbynulls.add(i) = true;
    }

    // max memory: work_mem * multiplier, +256 bytes to fill the last block
    let max_memory = ((pg_sys::work_mem as f64) * HNSW_SCAN_MEM_MULTIPLIER * 1024.0 + 256.0)
        as usize;

    let state = Box::new(ScanState {
        recheck_orderby: support.precision != HnswPrecision::Plain,
        support,
        base: HNSW_STANDALONE_BASE,
        first: true,
        w: Vec::new(),
        // Sized for the worst-case ef (1000) like pgvector's
        // InitVisited(ef * m * 2): a too-small table rehashes per grow and
        // the rehash churn dominates the per-query cost.
        visited: Visited::new(1000 * m * 2),
        discarded: None,
        q: Vec::new(),
        qstate: None,
        m,
        tuples: 0,
        previous_distance: f64::NEG_INFINITY,
        max_memory,
        scratch: SearchScratch::new(m),
        tmp_ctx,
        orderbyvals,
        orderbynulls,
        norm_q: 0.0,
    });
    (*scan).opaque = Box::into_raw(state) as *mut std::os::raw::c_void;
    scan
}

/// `hnswrescan` (hnswscan.c): reset and re-extract the query.
#[pg_guard]
pub unsafe extern "C-unwind" fn amrescan(
    scan: pg_sys::IndexScanDesc,
    _keys: pg_sys::ScanKey,
    _nkeys: std::os::raw::c_int,
    orderbys: pg_sys::ScanKey,
    norderbys: std::os::raw::c_int,
) {
    let state = &mut *((*scan).opaque as *mut ScanState);

    state.first = true;
    state.w.clear();
    state.visited.clear();
    state.discarded = None;
    state.tuples = 0;
    state.previous_distance = f64::NEG_INFINITY;
    state.scratch.elements.clear();
    pg_sys::MemoryContextReset(state.tmp_ctx.value());

    // Extract the query vector (NULL/absent → empty → no results).
    state.q.clear();
    if norderbys > 0 && !orderbys.is_null() {
        let orderby = &*orderbys;
        if !orderby.sk_argument.is_null() {
            let detoasted = pg_sys::pg_detoast_datum_copy(orderby.sk_argument.cast_mut_ptr());
            let pg_vec = detoasted.cast::<PgVectorInternal>();
            state.q.extend_from_slice((*pg_vec).to_slice());
            pg_sys::pfree(detoasted.cast());
            if state.support.dist_type == DistanceType::Cosine {
                preprocess_cosine(&mut state.q);
            }
        }
    }
    state.norm_q = state.q.iter().map(|x| x * x).sum::<f32>().sqrt();
}

/// `hnswgettuple` (hnswscan.c): run the search lazily, then emit one heap
/// TID per call.
#[pg_guard]
pub unsafe extern "C-unwind" fn amgettuple(
    scan: pg_sys::IndexScanDesc,
    dir: pg_sys::ScanDirection::Type,
) -> bool {
    debug_assert_eq!(dir, pg_sys::ScanDirection::ForwardScanDirection);
    let state = &mut *((*scan).opaque as *mut ScanState);
    let index = (*scan).indexRelation;

    if state.first {
        // Safety check
        if (*scan).orderByData.is_null() {
            error!("cannot scan hnswsq index without order");
        }
        // Requires MVCC-compliant snapshot (not able to maintain a pin)
        // IsMVCCSnapshot: snapshot_type == SNAPSHOT_MVCC.
        if (*(*scan).xs_snapshot).snapshot_type != pg_sys::SnapshotType::SNAPSHOT_MVCC {
            error!("non-MVCC snapshots are not supported with hnswsq");
        }

        // A shared lock lets vacuum ensure no in-flight scans before marking
        // tuples deleted.
        pg_sys::LockPage(index, scan_lock_page(state.base), pg_sys::ShareLock as pg_sys::LOCKMODE);
        state.w = get_scan_items(state, index);
        pg_sys::UnlockPage(index, scan_lock_page(state.base), pg_sys::ShareLock as pg_sys::LOCKMODE);

        // The iterative scan owns its discarded heap from the start.
        if HNSW_ITERATIVE_SCAN.get().as_i32() != ITERATIVE_SCAN_OFF {
            if state.discarded.is_none() {
                state.discarded = Some(CandidateHeap::new());
            }
        }

        state.first = false;
    }

    let iterative = HNSW_ITERATIVE_SCAN.get().as_i32();

    loop {
        let base = std::ptr::null_mut();
        let element: *mut Element;
        let sc: SearchCandidate;

        if state.w.is_empty() {
            if iterative == ITERATIVE_SCAN_OFF {
                break;
            }
            // Empty index
            if state.discarded.is_none() {
                break;
            }

            // Reached max number of tuples or memory limit
            let mem = scan_memory(state);
            if state.tuples >= HNSW_MAX_SCAN_TUPLES.get() as i64
                && HNSW_MAX_SCAN_TUPLES.get() >= 0
                || mem > state.max_memory
            {
                let empty = state
                    .discarded
                    .as_ref()
                    .map(|d| d.is_empty())
                    .unwrap_or(true);
                if empty {
                    break;
                }
                // Return remaining tuples
                let sc = state.discarded.as_mut().unwrap().pop().unwrap().0;
                state.w.push(sc);
            } else {
                // Locking ensures when neighbors are read, the elements they
                // reference will not be deleted (and replaced) during the
                // iteration.
                pg_sys::LockPage(index, scan_lock_page(state.base), pg_sys::ShareLock as pg_sys::LOCKMODE);
                state.w = resume_scan_items(state, index);
                pg_sys::UnlockPage(index, scan_lock_page(state.base), pg_sys::ShareLock as pg_sys::LOCKMODE);
            }

            if state.w.is_empty() {
                break;
            }
        }

        let sc_ref = state.w.last().expect("w not empty");
        sc = *sc_ref;
        element = crate::access_method::hnswsq::ptr::access::<Element>(base, sc.element);

        // Move to next element if no valid heap TIDs
        if (*element).heaptid_set == 0 {
            state.w.pop();
            continue;
        }

        let heaptid = (*element).heaptid;
        (*element).heaptid_set = 0;

        if iterative == ITERATIVE_SCAN_STRICT {
            if (sc.distance as f64) < state.previous_distance {
                continue;
            }
            state.previous_distance = sc.distance as f64;
        }

        let emitted = emit_distance(state, &sc);

        (*scan).xs_heaptid = heaptid;
        (*scan).xs_recheck = false;
        (*scan).xs_recheckorderby = state.recheck_orderby;
        // pgvector's distance operators return float8: the orderbyval datum
        // MUST be a double (raw f32 bits would be a garbage f64).
        *state.orderbyvals = pg_sys::Datum::from(emitted.to_bits() as usize);
        *state.orderbynulls = false;
        (*scan).xs_orderbyvals = state.orderbyvals;
        (*scan).xs_orderbynulls = state.orderbynulls;
        return true;
    }

    false
}

/// `hnswendscan` (hnswscan.c).
#[pg_guard]
pub unsafe extern "C-unwind" fn amendscan(scan: pg_sys::IndexScanDesc) {
    let ptr = (*scan).opaque as *mut ScanState;
    if !ptr.is_null() {
        #[cfg(any(test, feature = "pg_test"))]
        {
            let state = &*ptr;
            pgrx::log!(
                "hnswsq scan stats: visited_len={} tuples={}",
                state.visited.len(),
                state.tuples
            );
        }
        drop(Box::from_raw(ptr));
        (*scan).opaque = std::ptr::null_mut();
    }
}
