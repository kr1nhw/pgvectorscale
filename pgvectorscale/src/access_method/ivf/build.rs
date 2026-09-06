//! IVF index build implementation.
//!
//! Implements serial and parallel build pipelines for IVF indexes.

use pgrx::*;
use rand::prelude::*;
use rand::rngs::SmallRng;

use crate::access_method::distance::DistanceType;
use crate::access_method::ivf::centroid::{kmeans_plus_plus_init, lloyds_algorithm};
use crate::access_method::ivf::centroid_page::IvfCentroidPage;
use crate::access_method::ivf::entry::{seal_entries, IvfEntry};
use crate::access_method::ivf::list_directory::IvfListDirectory;
use crate::access_method::ivf::meta_page::IvfMetaPage;
use crate::access_method::ivf::options::TSVIvfOptions;
use crate::access_method::ivf::segment::{IvfListHeader, IvfSegment, IvfSegmentList};
use crate::access_method::pg_vector::PgVectorInternal;
use crate::access_method::quantization::rabitq::{RabitqQuantizer, RabitqVector};
use crate::util::ItemPointer;
use rayon::prelude::*;

/// Default number of samples for K-means training (reservoir-sampled during
/// the first heap pass, so this bounds the build's sample memory).
const DEFAULT_SAMPLE_SIZE: usize = 30000;

/// Maximum iterations for Lloyd's algorithm
const MAX_KMEANS_ITERATIONS: usize = 100;

/// Entries sealed per segment during the streaming second pass (bounds the
/// per-list buffer memory to `num_lists × SEGMENT_ENTRIES × sizeof(IvfEntry)`).
const SEGMENT_ENTRIES: usize = 10_000;

/// Build state for the first heap pass: reservoir-sampled vectors only.
struct SampleState {
    sample: Vec<Vec<f32>>,
    sample_size: usize,
    nrows: u64,
    rng: SmallRng,
}

/// Build state for the second heap pass: per-list entry buffers + sealed
/// segments, so the full dataset is never held in memory.
struct AssignState {
    centroids: Vec<Vec<f32>>,
    quantizer: RabitqQuantizer,
    distance_type: DistanceType,
    index: pg_sys::Relation,
    list_buffers: Vec<Vec<IvfEntry>>,
    segments: Vec<Vec<IvfSegment>>,
}

/// Build a new IVF index (streaming two-pass version).
///
/// Memory is bounded independent of the table size: pass 1 keeps only a
/// reservoir sample for K-means, and pass 2 assigns/quantizes/seals rows in
/// per-list batches, so a 100M-row build needs ~`num_lists x SEGMENT_ENTRIES`
/// entries in flight instead of the full dataset.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambuild(
    heap: pg_sys::Relation,
    index: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    let heap_rel = unsafe { PgRelation::from_pg(heap) };
    let index_rel = unsafe { PgRelation::from_pg(index) };

    // Get index options
    let options = TSVIvfOptions::from_relation(&index_rel);

    // Determine distance type (default to L2 for now)
    let distance_type = DistanceType::L2;

    // Number of dimensions from the index's vector column typmod (the typmod is
    // the dimension count directly, e.g. `vector(3)` has atttypmod 3).
    let num_dimensions = index_rel
        .tuple_desc()
        .get(0)
        .map(|attr| attr.atttypmod as usize)
        .unwrap_or(0);

    if num_dimensions == 0 {
        panic!("Cannot determine vector dimensions from index");
    }

    let num_lists = options.get_lists() as usize;
    let num_bits: u8 = options.get_num_bits();

    // ---- Pass 1: reservoir-sample the heap (bounded memory). ----
    let mut sample_state = SampleState {
        sample: Vec::with_capacity(DEFAULT_SAMPLE_SIZE),
        sample_size: DEFAULT_SAMPLE_SIZE,
        nrows: 0,
        rng: SmallRng::from_entropy(),
    };
    unsafe {
        pg_sys::IndexBuildHeapScan(
            heap_rel.as_ptr(),
            index_rel.as_ptr(),
            index_info,
            Some(sample_callback),
            &mut sample_state,
        );
    }
    let reltuples = sample_state.nrows as f64;

    if sample_state.sample.is_empty() {
        // PG calls ambuild (not ambuildempty) even for empty tables.
        write_empty_index(&index_rel, &options, num_dimensions as u32);
        let mut pg_result = unsafe { PgBox::<pg_sys::IndexBuildResult>::alloc0() };
        pg_result.heap_tuples = reltuples;
        pg_result.index_tuples = 0.0;
        return pg_result.into_pg();
    }

    // ---- K-means on the sample. ----
    let mut rng = SmallRng::from_entropy();
    let rotation_seed: u64 = rng.gen();
    let quantizer = RabitqQuantizer::new(num_bits, rotation_seed, num_dimensions);
    let mut centroids = kmeans_plus_plus_init(&sample_state.sample, num_lists, distance_type);
    centroids = lloyds_algorithm(
        &sample_state.sample,
        &mut centroids,
        MAX_KMEANS_ITERATIONS,
        distance_type,
    );
    drop(sample_state);

    // ---- Write meta + empty directory + centroid page (before pass 2 so the
    // segment allocator sees the reserved ranges, as in the old flow). ----
    let storage_type = options.get_storage_type();
    let mut meta_page = unsafe {
        IvfMetaPage::create(
            &index_rel,
            num_dimensions as u32,
            distance_type,
            num_lists as u16,
            storage_type,
            num_bits,
            rotation_seed,
        )
    };
    let mut list_directory = IvfListDirectory::new(num_lists as u16);
    unsafe {
        list_directory.store(&index_rel, true);
    }
    let centroid_page = IvfCentroidPage::new(centroids.clone());
    let centroid_ptr = unsafe { centroid_page.store(&index_rel, None) };
    unsafe {
        meta_page.set_list_directory_pointer(ItemPointer::new(1, 1));
        meta_page.set_centroids_pointer(centroid_ptr);
        meta_page.store(&index_rel, false);
    }

    // ---- Pass 2: assign + quantize + seal in per-list batches. ----
    let mut assign_state = AssignState {
        centroids,
        quantizer,
        distance_type,
        index: index_rel.as_ptr(),
        list_buffers: (0..num_lists).map(|_| Vec::new()).collect(),
        segments: (0..num_lists).map(|_| Vec::new()).collect(),
    };
    unsafe {
        pg_sys::IndexBuildHeapScan(
            heap_rel.as_ptr(),
            index_rel.as_ptr(),
            index_info,
            Some(assign_callback),
            &mut assign_state,
        );
    }

    // Flush the remaining per-list buffers and publish one segment list +
    // header per list.
    let mut num_tuples = 0u64;
    for list_id in 0..num_lists {
        let entries = std::mem::take(&mut assign_state.list_buffers[list_id]);
        if !entries.is_empty() {
            let segment = seal_entries(&index_rel, entries);
            if !segment.is_empty() {
                assign_state.segments[list_id].push(segment);
            }
        }
        let count: u64 = assign_state.segments[list_id]
            .iter()
            .map(|s| s.num_entries)
            .sum();
        num_tuples += count;

        let segment_list =
            IvfSegmentList::new(std::mem::take(&mut assign_state.segments[list_id]));
        let (segment_list_ptr, segment_list_blocks) = unsafe { segment_list.store(&index_rel) };
        let header = IvfListHeader::new(segment_list_ptr, segment_list_blocks);
        let header_ptr = unsafe { header.store_new(&index_rel) };
        if let Some(list_meta) = list_directory.get_list_mut(list_id as u16) {
            list_meta.header = header_ptr;
            list_meta.num_tuples = count;
        }
    }

    unsafe {
        list_directory.store(&index_rel, false);
        // Bulk smgr scans need the built entry blocks on disk first.
        pg_sys::FlushRelationBuffers(index_rel.as_ptr());
    }

    let mut pg_result = unsafe { PgBox::<pg_sys::IndexBuildResult>::alloc0() };
    pg_result.heap_tuples = reltuples;
    pg_result.index_tuples = num_tuples as f64;
    pg_result.into_pg()
}

/// Pass-1 callback: reservoir sampling (Algorithm R) with a row counter.
#[pg_guard]
unsafe extern "C-unwind" fn sample_callback(
    _index: pg_sys::Relation,
    _tid: pg_sys::ItemPointer,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut std::os::raw::c_void,
) {
    if *isnull {
        return;
    }
    let sample_state = &mut *(state as *mut SampleState);

    let datum = *values;
    let datum_ptr = datum.cast_mut_ptr::<pg_sys::varlena>();
    let detoasted_ptr = pg_sys::pg_detoast_datum(datum_ptr);
    let detoasted_datum = pg_sys::Datum::from(detoasted_ptr);
    let pg_vec_internal = detoasted_datum.cast_mut_ptr::<PgVectorInternal>();
    let vec_slice = unsafe { (*pg_vec_internal).to_slice() };

    let i = sample_state.nrows;
    sample_state.nrows += 1;
    if sample_state.sample.len() < sample_state.sample_size {
        sample_state.sample.push(vec_slice.to_vec());
    } else {
        let j = sample_state.rng.gen_range(0..=i);
        if j < sample_state.sample_size as u64 {
            sample_state.sample[j as usize] = vec_slice.to_vec();
        }
    }
}

/// Pass-2 callback: assign to the nearest centroid, quantize the residual,
/// and buffer the entry; seals a segment whenever a list buffer fills.
#[pg_guard]
unsafe extern "C-unwind" fn assign_callback(
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
    let assign_state = &mut *(state as *mut AssignState);

    let datum = *values;
    let datum_ptr = datum.cast_mut_ptr::<pg_sys::varlena>();
    let detoasted_ptr = pg_sys::pg_detoast_datum(datum_ptr);
    let detoasted_datum = pg_sys::Datum::from(detoasted_ptr);
    let pg_vec_internal = detoasted_datum.cast_mut_ptr::<PgVectorInternal>();
    let vec_slice = unsafe { (*pg_vec_internal).to_slice() };

    let list_id =
        nearest_centroid(vec_slice, &assign_state.centroids, assign_state.distance_type) as usize;
    let code = assign_state
        .quantizer
        .quantize_residual(&assign_state.centroids[list_id], vec_slice);
    let item_ptr = ItemPointer::with_item_pointer_data(*tid);
    let entry = IvfEntry::new(item_ptr, code);

    assign_state.list_buffers[list_id].push(entry);
    if assign_state.list_buffers[list_id].len() >= SEGMENT_ENTRIES {
        let entries = std::mem::take(&mut assign_state.list_buffers[list_id]);
        let segment = seal_entries(&PgRelation::from_pg(assign_state.index), entries);
        if !segment.is_empty() {
            assign_state.segments[list_id].push(segment);
        }
    }
}

/// Build an empty IVF index (for CREATE INDEX on empty table).
#[pg_guard]
pub extern "C-unwind" fn ambuildempty(index: pg_sys::Relation) {
    unsafe {
        let index_rel = PgRelation::from_pg(index);
        let options = TSVIvfOptions::from_relation(&index_rel);
        let num_dimensions = index_rel
            .tuple_desc()
            .get(0)
            .map(|attr| attr.atttypmod as usize)
            .unwrap_or(0);
        if num_dimensions == 0 {
            panic!("Cannot determine vector dimensions from index");
        }
        write_empty_index(&index_rel, &options, num_dimensions as u32);
    }
}

/// Write the minimal IVF on-disk structure (meta + empty list directory +
/// empty centroid page + one empty header per list) with the fixed block layout.
fn write_empty_index(index: &PgRelation, options: &TSVIvfOptions, num_dimensions: u32) {
    let num_lists = options.get_lists() as usize;
    let mut rng = SmallRng::from_entropy();
    let rotation_seed: u64 = rng.gen();

    // Meta page (block 0), empty list directory (block 1), empty centroids
    // (dynamic block recorded in the meta), and one empty header per list.
    let mut meta = unsafe {
        IvfMetaPage::create(
            index,
            num_dimensions,
            DistanceType::L2,
            num_lists as u16,
            options.get_storage_type(),
            options.get_num_bits(),
            rotation_seed,
        )
    };

    let mut list_directory = IvfListDirectory::new(num_lists as u16);
    unsafe {
        list_directory.store(index, true);
    }

    let centroid_page = IvfCentroidPage::new(Vec::new());
    let centroid_ptr = unsafe { centroid_page.store(index, None) };

    // All empty lists share one immutable empty segment-list item, marked with
    // segment_list_blocks = 0: a sentinel meaning "shared, never retire" (a
    // list that later seals creates its own item, so the shared one is never
    // the target of a retire-and-reclaim).
    let empty_segment_list = IvfSegmentList::new(Vec::new());
    let (empty_segment_list_ptr, _) = unsafe { empty_segment_list.store(index) };
    for list_id in 0..num_lists {
        let header = IvfListHeader::new(empty_segment_list_ptr, 0);
        let header_ptr = unsafe { header.store_new(index) };
        if let Some(list_meta) = list_directory.get_list_mut(list_id as u16) {
            list_meta.header = header_ptr;
        }
    }

    unsafe {
        list_directory.store(index, false);
        meta.set_list_directory_pointer(ItemPointer::new(1, 1));
        meta.set_centroids_pointer(centroid_ptr);
        meta.store(index, false);
    }
}

/// Sample vectors using reservoir sampling for K-means training.
///
/// Uses Algorithm R (Vitter, 1985) for reservoir sampling.
pub fn sample_vectors(vectors: &[Vec<f32>], sample_size: usize) -> Vec<Vec<f32>> {
    if vectors.len() <= sample_size {
        return vectors.to_vec();
    }

    let mut rng = SmallRng::from_entropy();
    let mut reservoir: Vec<Vec<f32>> = vectors[..sample_size].to_vec();

    for i in sample_size..vectors.len() {
        let j = rng.gen_range(0..=i);
        if j < sample_size {
            reservoir[j] = vectors[i].clone();
        }
    }

    reservoir
}

/// Assign vectors to nearest centroid.
///
/// Returns a vector of list IDs (one per input vector).
pub fn assign_vectors_to_lists(
    vectors: &[Vec<f32>],
    centroids: &[Vec<f32>],
    distance_type: DistanceType,
) -> Vec<u16> {
    let dist_fn = distance_type.get_distance_function();
    let k = centroids.len();

    vectors
        .iter()
        .map(|vec| {
            let mut best_list = 0u16;
            let mut best_dist = dist_fn(vec, &centroids[0]);

            for (c, centroid) in centroids.iter().enumerate().skip(1) {
                let d = dist_fn(vec, centroid);
                if d < best_dist {
                    best_dist = d;
                    best_list = c as u16;
                }
            }

            best_list
        })
        .collect()
}

/// Find the nearest centroid to a single vector (index into `centroids`).
#[inline]
pub fn nearest_centroid(
    vec: &[f32],
    centroids: &[Vec<f32>],
    distance_type: DistanceType,
) -> u16 {
    let dist_fn = distance_type.get_distance_function();
    let mut best_list = 0u16;
    let mut best_dist = dist_fn(vec, &centroids[0]);

    for (c, centroid) in centroids.iter().enumerate().skip(1) {
        let d = dist_fn(vec, centroid);
        if d < best_dist {
            best_dist = d;
            best_list = c as u16;
        }
    }

    best_list
}

/// Build IVF index from vectors (serial version).
///
/// This is the main build function that:
/// 1. Samples vectors for K-means training
/// 2. Runs K-means to get centroids
/// 3. Assigns vectors to lists
/// 4. Writes centroids, list directory, and entry pages
pub fn build_ivf_index_serial(
    index: &PgRelation,
    vectors: &[Vec<f32>],
    heap_tids: &[ItemPointer],
    options: &TSVIvfOptions,
    distance_type: DistanceType,
    num_dimensions: u32,
) -> IvfBuildResult {
    if vectors.is_empty() {
        // PG calls ambuild (not ambuildempty) even for empty tables, so write
        // the minimal on-disk structure here.
        write_empty_index(index, options, num_dimensions);
        return IvfBuildResult {
            num_tuples: 0,
            centroids: Vec::new(),
            list_directory: IvfListDirectory::new(0),
        };
    }

    let num_lists = options.get_lists() as usize;
    let sample_size = DEFAULT_SAMPLE_SIZE.min(vectors.len());

    // RaBitQ quantization: bits per dim from the reloption (1/4/8), random
    // deterministic rotation seed.
    let num_bits: u8 = options.get_num_bits();
    let mut rng = SmallRng::from_entropy();
    let rotation_seed: u64 = rng.gen();
    let quantizer = RabitqQuantizer::new(num_bits, rotation_seed, num_dimensions as usize);

    // Step 1: Sample vectors for K-means
    let samples = sample_vectors(vectors, sample_size);

    // Step 2: Run K-means to get centroids
    let mut centroids = kmeans_plus_plus_init(&samples, num_lists, distance_type);
    centroids = lloyds_algorithm(
        &samples,
        &mut centroids,
        MAX_KMEANS_ITERATIONS,
        distance_type,
    );

    // Step 3: Assign vectors to their nearest centroid and quantize each
    // residual in parallel (the CPU-heavy part of the build).
    let assigned: Vec<(u16, RabitqVector)> = vectors
        .par_iter()
        .map(|v| {
            let list_id = nearest_centroid(v, &centroids, distance_type);
            let code = quantizer.quantize_residual(&centroids[list_id as usize], v);
            (list_id, code)
        })
        .collect();

    // Step 4: Write meta page first (block 0).
    let storage_type = options.get_storage_type();
    let mut meta_page = unsafe {
        crate::access_method::ivf::meta_page::IvfMetaPage::create(
            index,
            num_dimensions,
            distance_type,
            num_lists as u16,
            storage_type,
            num_bits,       // bq_num_bits_per_dimension
            rotation_seed,  // rotation seed
        )
    };

    // Step 5: Write an empty list directory at block 1.
    let mut list_directory = IvfListDirectory::new(num_lists as u16);
    unsafe {
        list_directory.store(index, true);
    }

    // Step 6: Write the centroid page (a dynamic block after the directory) and
    // record its pointer in the meta page.
    let centroid_page = IvfCentroidPage::new(centroids.clone());
    let centroid_ptr = unsafe { centroid_page.store(index, None) };
    unsafe {
        meta_page.set_list_directory_pointer(ItemPointer::new(1, 1));
        meta_page.set_centroids_pointer(centroid_ptr);
        meta_page.store(index, false);
    }

    // Step 7: Seal each list into one immutable segment and write its header
    // (which points at a fresh segment-list item).
    for list_id in 0..num_lists {
        let entries: Vec<IvfEntry> = assigned
            .iter()
            .enumerate()
            .filter(|(_, (l, _))| *l == list_id as u16)
            .map(|(i, (_, code))| IvfEntry::new(heap_tids[i], code.clone()))
            .collect();

        let count = entries.len() as u64;
        let segment = seal_entries(index, entries);
        let segments = if segment.is_empty() {
            Vec::new()
        } else {
            vec![segment]
        };
        let segment_list = IvfSegmentList::new(segments);
        let (segment_list_ptr, segment_list_blocks) =
            unsafe { segment_list.store(index) };
        let header = IvfListHeader::new(segment_list_ptr, segment_list_blocks);
        let header_ptr = unsafe { header.store_new(index) };

        if let Some(list_meta) = list_directory.get_list_mut(list_id as u16) {
            list_meta.header = header_ptr;
            list_meta.num_tuples = count;
        }
    }

    // Step 8: Rewrite list directory in place at block 1 with the header pointers.
    unsafe {
        list_directory.store(index, false);
        // Bulk smgr scans need the built entry blocks on disk first.
        pg_sys::FlushRelationBuffers(index.as_ptr());
    }

    IvfBuildResult {
        num_tuples: vectors.len() as u64,
        centroids,
        list_directory,
    }
}

/// Build IVF index from vectors (parallel version).
///
/// Falls back to serial build if parallel fails or is not beneficial.
/// TODO: Implement actual parallel build with PostgreSQL workers.
pub fn build_ivf_index_parallel(
    index: &PgRelation,
    vectors: &[Vec<f32>],
    heap_tids: &[ItemPointer],
    options: &TSVIvfOptions,
    distance_type: DistanceType,
    num_dimensions: u32,
    _num_workers: i32,
) -> IvfBuildResult {
    // The assignment + quantization phase is parallelized internally with
    // rayon (see build_ivf_index_serial), so the parallel entry point simply
    // delegates to it.  K-means runs on a small reservoir sample and is cheap.
    build_ivf_index_serial(
        index,
        vectors,
        heap_tids,
        options,
        distance_type,
        num_dimensions,
    )
}

/// Result of building an IVF index.
pub struct IvfBuildResult {
    pub num_tuples: u64,
    pub centroids: Vec<Vec<f32>>,
    pub list_directory: IvfListDirectory,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_method::distance::DistanceType;

    #[test]
    fn test_sample_vectors_small() {
        let vectors: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, 0.0]).collect();
        let samples = sample_vectors(&vectors, 5);
        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn test_sample_vectors_large() {
        let vectors: Vec<Vec<f32>> = (0..1000).map(|i| vec![i as f32, 0.0]).collect();
        let samples = sample_vectors(&vectors, 100);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_sample_vectors_exact_size() {
        let vectors: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32, 0.0]).collect();
        let samples = sample_vectors(&vectors, 100);
        assert_eq!(samples.len(), 100);
        assert_eq!(samples, vectors);
    }

    #[test]
    fn test_assign_vectors_to_lists_basic() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        let centroids = vec![vec![0.05, 0.05], vec![10.05, 10.05]];

        let assignments = assign_vectors_to_lists(&vectors, &centroids, DistanceType::L2);
        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments[0], 0);
        assert_eq!(assignments[1], 0);
        assert_eq!(assignments[2], 1);
        assert_eq!(assignments[3], 1);
    }

    #[test]
    fn test_assign_vectors_to_lists_single_centroid() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let centroids = vec![vec![0.0, 0.0]];

        let assignments = assign_vectors_to_lists(&vectors, &centroids, DistanceType::L2);
        assert_eq!(assignments.len(), 3);
        assert!(assignments.iter().all(|&a| a == 0));
    }

    #[test]
    fn test_assign_vectors_to_lists_cosine() {
        let vectors = vec![
            vec![1.0, 0.0],
            vec![0.99, 0.01],
            vec![0.0, 1.0],
            vec![0.01, 0.99],
        ];
        let centroids = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let assignments = assign_vectors_to_lists(&vectors, &centroids, DistanceType::Cosine);
        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments[0], 0);
        assert_eq!(assignments[1], 0);
        assert_eq!(assignments[2], 1);
        assert_eq!(assignments[3], 1);
    }

    #[test]
    fn test_build_ivf_index_empty() {
        // This test would need a PgRelation, so we'll skip it for now
        // In a real test, we'd create a test index and verify the build
    }

    #[test]
    fn test_build_ivf_index_single_vector() {
        // This test would need a PgRelation
        // Placeholder for integration test
    }

    #[test]
    fn test_build_ivf_index_small_dataset() {
        // This test would need a PgRelation
        // Placeholder for integration test
    }
}
