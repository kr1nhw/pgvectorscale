//! IVF index scan implementation.
//!
//! Implements the scan callbacks for the IVF access method:
//! - ambeginscan: Initialize scan state
//! - amrescan: Reset scan state with new parameters
//! - amgettuple: Return next matching tuple
//! - amendscan: Clean up scan state

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use pgrx::*;

use crate::access_method::distance::DistanceType;
use crate::access_method::ivf::centroid_page::IvfCentroidPage;
use crate::access_method::ivf::entry::IvfEntryReader;
use crate::access_method::ivf::list_directory::IvfListDirectory;
use crate::access_method::ivf::meta_page::IvfMetaPage;
use crate::access_method::ivf::options::{IVF_PROBES, IVF_TOP_K};
use crate::access_method::ivf::segment::{IvfListHeader, IvfSegmentList};
use crate::access_method::ivf::simd::find_nearest_centroids;
use crate::access_method::pg_vector::PgVectorInternal;
use crate::access_method::quantization::rabitq::{RabitqFastScan, RabitqQuantizer};
use crate::util::buffer::AdvisoryLockGuard;
use crate::util::ItemPointer;

/// A (distance, heap tid) candidate pair ordered by distance, used as the
/// element type of the bounded top-k max-heap.
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
        // total_cmp gives a total order on f32 (incl. NaN); tiebreak by tid so
        // Ord stays consistent with the derived PartialEq/Eq.
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.tid.cmp(&other.tid))
    }
}

/// Scan state for IVF index scans
pub struct IvfScanState {
    /// Query vector (pre-processed, e.g. normalized for cosine)
    pub query: Vec<f32>,
    /// Number of probes (lists to scan)
    pub probes: usize,
    /// Search results as (distance, heap_tid) pairs, sorted by distance
    pub results: Vec<(f32, ItemPointer)>,
    /// Current position in results
    pub result_index: usize,
    /// Whether results have been computed
    pub results_computed: bool,
}

impl IvfScanState {
    /// Create a new scan state
    pub fn new() -> Self {
        Self {
            query: Vec::new(),
            probes: 10,
            results: Vec::new(),
            result_index: 0,
            results_computed: false,
        }
    }
}

/// Extract a query vector from a `vector` datum, normalizing for cosine.
fn extract_query_vector(datum: pg_sys::Datum, distance_type: DistanceType) -> Vec<f32> {
    unsafe {
        let detoasted = pg_sys::pg_detoast_datum_copy(datum.cast_mut_ptr());
        let pg_vec = detoasted.cast::<PgVectorInternal>();
        let dim = (*pg_vec).dim;
        let mut vec = (*pg_vec).to_slice().to_vec();
        pg_sys::pfree(detoasted.cast());


        if distance_type == DistanceType::Cosine {
            crate::access_method::distance::preprocess_cosine(&mut vec);
        }
        vec
    }
}

/// Begin a scan of the IVF index.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambeginscan(
    index: pg_sys::Relation,
    nkeys: std::os::raw::c_int,
    norderbys: std::os::raw::c_int,
) -> pg_sys::IndexScanDesc {
    let scan = unsafe { pg_sys::RelationGetIndexScan(index, nkeys, norderbys) };

    // Allocate scan state
    let scan_state = IvfScanState::new();
    let scan_state_ptr = pg_sys::palloc0(std::mem::size_of::<IvfScanState>()) as *mut IvfScanState;
    unsafe {
        *scan_state_ptr = scan_state;
        (*scan).opaque = scan_state_ptr as *mut std::os::raw::c_void;
    }

    scan
}

/// Rescan the IVF index with new parameters.
#[pg_guard]
pub unsafe extern "C-unwind" fn amrescan(
    scan: pg_sys::IndexScanDesc,
    _keys: pg_sys::ScanKey,
    _nkeys: std::os::raw::c_int,
    orderbys: pg_sys::ScanKey,
    norderbys: std::os::raw::c_int,
) {
    let scan_state = unsafe { &mut *((*scan).opaque as *mut IvfScanState) };

    // Reset results
    scan_state.results.clear();
    scan_state.result_index = 0;
    scan_state.results_computed = false;

    let index_rel = unsafe { PgRelation::from_pg((*scan).indexRelation) };
    let meta = IvfMetaPage::fetch(&index_rel);
    let distance_type = meta.get_distance_type();

    // Extract query vector from orderbys (first orderby is the query vector).
    if norderbys > 0 && !orderbys.is_null() {
        let orderby = unsafe { &*orderbys };
        if !orderby.sk_argument.is_null() {
            scan_state.query = extract_query_vector(orderby.sk_argument, distance_type);
        }
    }

    scan_state.probes = IVF_PROBES.get() as usize;
}

/// Get the next tuple from the IVF index scan.
#[pg_guard]
pub unsafe extern "C-unwind" fn amgettuple(
    scan: pg_sys::IndexScanDesc,
    _direction: pg_sys::ScanDirection::Type,
) -> bool {
    let scan_state = unsafe { &mut *((*scan).opaque as *mut IvfScanState) };

    // Compute results on first call
    if !scan_state.results_computed {
        let index_rel = unsafe { PgRelation::from_pg((*scan).indexRelation) };
        // Read-burst protocol: hold the shared advisory lock across the header
        // / segment-list reads and the smgrreadv bursts.  Reclamation takes the
        // exclusive advisory lock, so it cannot free (and reuse) blocks while
        // this scan is still reading them; the shared lock is self-compatible,
        // so concurrent scans do not contend, and it never conflicts with the
        // executor's relation locks on the index.
        let (k1, k2) = crate::access_method::ivf::meta_page::advisory_keys(&index_rel);
        let _read_guard = AdvisoryLockGuard::acquire_shared(k1, k2);
        let meta = IvfMetaPage::fetch(&index_rel);
        let distance_type = meta.get_distance_type();
        let centroid_page = match meta.get_centroids_pointer() {
            Some(p) => IvfCentroidPage::load(&index_rel, p),
            None => IvfCentroidPage::new(Vec::new()),
        };
        let list_directory = IvfListDirectory::load(&index_rel);

        // Step 1: find nearest centroids (which lists to probe)
        let nearest = find_nearest_centroids(
            &scan_state.query,
            &centroid_page.centroids,
            distance_type,
            scan_state.probes,
        );

        // Step 2: scan each probed list, estimating distances with RaBitQ.
        // Iterate zero-copy over archived entries and keep only the top-K
        // candidates (by estimate) in a bounded max-heap.
        let num_bits = meta.get_bq_num_bits_per_dimension();
        let rotation_seed = meta.get_rotation_seed();
        let quantizer = RabitqQuantizer::new(num_bits, rotation_seed, meta.get_num_dimensions() as usize);
        let reader = IvfEntryReader::new(&index_rel);
        let top_k = (IVF_TOP_K.get() as usize).max(1);
        let mut heap: BinaryHeap<DistTid> = BinaryHeap::with_capacity(top_k.min(1024));
        for list_id in nearest {
            if let Some(list_meta) = list_directory.get_list(list_id as u16) {
                if !list_meta.header.is_valid() {
                    continue;
                }
                let header = IvfListHeader::load(&index_rel, list_meta.header);
                let segment_list = IvfSegmentList::load(&index_rel, header.segment_list);
                // A list with no published segments can still have entries in
                // its (unsealed) active buffer, so only skip when both empty.
                if segment_list.segments.is_empty() && header.active.is_none() {
                    continue;
                }
                let centroid = &centroid_page.centroids[list_id as usize];
                let rq = quantizer.rotate_query_residual(centroid, &scan_state.query);
                let fastscan = RabitqFastScan::new(&rq, num_bits, quantizer.dim());
                let segments: Vec<&crate::access_method::ivf::segment::IvfSegment> =
                    segment_list
                        .segments
                        .iter()
                        .filter(|s| {
                            s.start_page != pg_sys::InvalidBlockNumber && s.num_blocks > 0
                        })
                        .collect();
                for (seg_idx, segment) in segments.iter().enumerate() {
                    // Neon: pipeline the next segment's page fetches with this
                    // segment's scoring.  `smgrprefetch` on libpagestore issues
                    // asynchronous page requests, so the next segment's
                    // `smgrreadv` finds the pages already in flight or resident
                    // instead of paying a synchronous round trip.
                    #[cfg(feature = "neon")]
                    if let Some(next) = segments.get(seg_idx + 1) {
                        reader.prefetch_blocks(next.start_page, next.num_blocks);
                    }
                    reader.for_each_slice(segment.start_page, segment.num_blocks, |view| {
                        if num_bits == 1 {
                            // 1-bit: SIMD FastScan sum + fused SIMD estimate over
                            // 32-row transposed batches.
                            let mut sums = [0u16; 32];
                            let mut dists = [0f32; 32];
                            for batch in 0..view.num_batches() {
                                fastscan.sum_batch(view.code_batch(batch), &mut sums);
                                let base = batch * 32;
                                let count = (view.len() - base).min(32);
                                fastscan.estimate_batch(
                                    &sums[..count],
                                    &view.scale_slice()[base..base + count],
                                    &view.sum_of_x2_slice()[base..base + count],
                                    &view.margin_factor_slice()[base..base + count],
                                    &mut dists[..count],
                                    count,
                                );
                                for r in 0..count {
                                    let candidate =
                                        DistTid { dist: dists[r], tid: view.tid(base + r) };
                                    if heap.len() < top_k {
                                        heap.push(candidate);
                                    } else if let Some(mut worst) = heap.peek_mut() {
                                        if candidate.dist < worst.dist {
                                            *worst = candidate;
                                        }
                                    }
                                }
                            }
                        } else if num_bits == 2 {
                            // 2-bit: SIMD FastScan over both bit-planes
                            // (sign + ex), fused into one SIMD estimate per
                            // 32-row transposed batch.
                            let mut sums0 = [0u16; 32];
                            let mut sums1 = [0u16; 32];
                            let mut dists = [0f32; 32];
                            for batch in 0..view.num_batches() {
                                fastscan.sum_batch(view.code_batch_plane0(batch), &mut sums0);
                                fastscan.sum_batch(view.code_batch_plane1(batch), &mut sums1);
                                let base = batch * 32;
                                let count = (view.len() - base).min(32);
                                fastscan.estimate_batch_2bit(
                                    &sums0[..count],
                                    &sums1[..count],
                                    &view.scale_slice()[base..base + count],
                                    &view.sum_of_x2_slice()[base..base + count],
                                    &view.margin_factor_slice()[base..base + count],
                                    &mut dists[..count],
                                    count,
                                );
                                for r in 0..count {
                                    let candidate =
                                        DistTid { dist: dists[r], tid: view.tid(base + r) };
                                    if heap.len() < top_k {
                                        heap.push(candidate);
                                    } else if let Some(mut worst) = heap.peek_mut() {
                                        if candidate.dist < worst.dist {
                                            *worst = candidate;
                                        }
                                    }
                                }
                            }
                        } else {
                            // 4/8-bit: SIMD ex-dot per entry (row-major).
                            for i in 0..view.len() {
                                let full_dot = fastscan.full_dot_multi(view.code(i));
                                let d = fastscan.estimate_from_full_dot(
                                    full_dot,
                                    view.sum_of_x2(i),
                                    view.scale(i),
                                    view.margin_factor(i),
                                );
                                let candidate = DistTid { dist: d, tid: view.tid(i) };
                                if heap.len() < top_k {
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
                // Open (append) buffer: entries not yet sealed into a
                // published segment are still visible — read them through the
                // buffer manager (append-only pages, one row-major item per
                // entry) with the scalar estimator.  This is the slow path;
                // sealed segments use the SIMD FastScan path above.
                if let Some(active) = header.active.as_ref() {
                    let entries =
                        crate::access_method::ivf::entry::read_active_entries(&index_rel, active);
                    for e in &entries {
                        let d = quantizer.estimate_l2_fields(
                            e.code.num_bits,
                            e.code.dim,
                            &e.code.packed_code,
                            e.code.sum_of_x2,
                            e.code.l1_of_rotated,
                            &rq,
                        );
                        let candidate = DistTid {
                            dist: d,
                            tid: e.heap_tid,
                        };
                        if heap.len() < top_k {
                            heap.push(candidate);
                        } else if let Some(mut worst) = heap.peek_mut() {
                            if candidate.dist < worst.dist {
                                *worst = candidate;
                            }
                        }
                    }
                }
            }
        }

        // Step 3: emit the top-K candidates in ascending estimate order.  The
        // executor rechecks exact distances (xs_recheckorderby), so `top_k`
        // must be >= the query's LIMIT (see ivf.top_k).
        let results = heap.into_sorted_vec();
        scan_state.results = results.into_iter().map(|c| (c.dist, c.tid)).collect();
        scan_state.results_computed = true;
    }

    // Return next result if available
    if scan_state.result_index < scan_state.results.len() {
        let (distance, heap_tid) = scan_state.results[scan_state.result_index];
        scan_state.result_index += 1;

        unsafe {
            let mut tid_data = pg_sys::ItemPointerData::default();
            heap_tid.to_item_pointer_data(&mut tid_data);
            (*scan).xs_heaptid = tid_data;
            (*scan).xs_recheck = false;
            // The RaBitQ estimate is a lower bound on the exact distance, so the
            // executor can recheck and reorder by the exact distance.
            (*scan).xs_recheckorderby = true;

            // Provide the approximate distance as a proper Datum (the executor
            // still compares against it to order the recheck queue).
            let orderbyvals =
                pg_sys::palloc(std::mem::size_of::<pg_sys::Datum>()) as *mut pg_sys::Datum;
            let orderbynulls = pg_sys::palloc(std::mem::size_of::<bool>()) as *mut bool;
            *orderbyvals = pg_sys::Datum::from(distance.to_bits() as usize);
            *orderbynulls = false;
            (*scan).xs_orderbyvals = orderbyvals;
            (*scan).xs_orderbynulls = orderbynulls;
        }
        true
    } else {
        false
    }
}

/// End the IVF index scan.
#[pg_guard]
pub unsafe extern "C-unwind" fn amendscan(scan: pg_sys::IndexScanDesc) {
    // Free scan state
    let scan_state = unsafe { (*scan).opaque as *mut IvfScanState };
    if !scan_state.is_null() {
        unsafe {
            pg_sys::pfree(scan_state as *mut std::os::raw::c_void);
        }
        unsafe {
            (*scan).opaque = std::ptr::null_mut();
        }
    }
}
