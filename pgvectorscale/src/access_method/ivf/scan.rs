//! IVF index scan implementation.
//!
//! Implements the scan callbacks for the IVF access method:
//! - ambeginscan: Initialize scan state
//! - amrescan: Reset scan state with new parameters
//! - amgettuple: Return next matching tuple
//! - amendscan: Clean up scan state

use std::cmp::Ordering;

use pgrx::*;

use crate::access_method::distance::DistanceType;
use crate::access_method::ivf::centroid_page::IvfCentroidPage;
use crate::access_method::ivf::entry::IvfEntryReader;
use crate::access_method::ivf::list_directory::IvfListDirectory;
use crate::access_method::ivf::meta_page::IvfMetaPage;
use crate::access_method::ivf::options::IVF_PROBES;
use crate::access_method::ivf::simd::{find_nearest_centroids, get_simd_distance_fn};
use crate::access_method::pg_vector::PgVectorInternal;
use crate::util::ItemPointer;

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

    // Load the meta page to determine the distance type for query pre-processing.
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
        let meta = IvfMetaPage::fetch(&index_rel);
        let distance_type = meta.get_distance_type();
        let centroid_page = IvfCentroidPage::load(&index_rel);
        let list_directory = IvfListDirectory::load(&index_rel);

        warning!(
            "IVF search: query_len={} n_centroids={} c0_len={}",
            scan_state.query.len(),
            centroid_page.centroids.len(),
            centroid_page.centroids.first().map(|c| c.len()).unwrap_or(0)
        );

        // Step 1: find nearest centroids (which lists to probe)
        let nearest = find_nearest_centroids(
            &scan_state.query,
            &centroid_page.centroids,
            distance_type,
            scan_state.probes,
        );

        // Step 2: scan each probed list and compute exact distances
        let reader = IvfEntryReader::new(&index_rel);
        let dist_fn = get_simd_distance_fn(distance_type);
        let mut results: Vec<(f32, ItemPointer)> = Vec::new();
        for list_id in nearest {
            if let Some(list_meta) = list_directory.get_list(list_id as u16) {
                if list_meta.start_page != pg_sys::InvalidBlockNumber {
                    for entry in reader.read_entries(list_meta.start_page) {
                        let d = dist_fn(&scan_state.query, &entry.vector);
                        results.push((d, entry.heap_tid));
                    }
                }
            }
        }

        // Step 3: sort by distance ascending
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        scan_state.results = results;
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

            // Allocate memory for distance value
            let distance_ptr = pg_sys::palloc(std::mem::size_of::<f32>()) as *mut f32;
            if !distance_ptr.is_null() {
                *distance_ptr = distance;
                // Convert f32 pointer to Datum pointer
                (*scan).xs_orderbyvals = distance_ptr as *mut pg_sys::Datum;
            } else {
                (*scan).xs_orderbyvals = std::ptr::null_mut();
            }
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
