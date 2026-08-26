//! IVF index insert implementation.

use pgrx::*;

use crate::access_method::ivf::centroid_page::IvfCentroidPage;
use crate::access_method::ivf::entry::{IvfEntry, IvfEntryReader, IvfEntryWriter};
use crate::access_method::ivf::list_directory::IvfListDirectory;
use crate::access_method::ivf::meta_page::IvfMetaPage;
use crate::access_method::ivf::simd::find_nearest_centroids;
use crate::access_method::pg_vector::PgVectorInternal;
use crate::access_method::quantization::rabitq::RabitqQuantizer;
use crate::util::ItemPointer;

/// Insert a tuple into the IVF index.
///
/// Finds the nearest centroid, quantizes the vector relative to it, and
/// appends the entry to that inverted list (rewriting the list's entry chain).
#[pg_guard]
pub unsafe extern "C-unwind" fn aminsert(
    index: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    _heap: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    // Skip null vectors.
    if *isnull {
        return false;
    }

    let index_rel = unsafe { PgRelation::from_pg(index) };
    let meta = IvfMetaPage::fetch(&index_rel);
    let mut centroid_page = IvfCentroidPage::load(&index_rel);
    let mut list_directory = IvfListDirectory::load(&index_rel);

    // Extract the vector.
    let datum = *values;
    let detoasted = pg_sys::pg_detoast_datum_copy(datum.cast_mut_ptr());
    let pg_vec = detoasted.cast::<PgVectorInternal>();
    let vector = (*pg_vec).to_slice().to_vec();
    pg_sys::pfree(detoasted.cast());

    // If the index has no centroids yet (built empty), seed the first centroid
    // from this vector so subsequent inserts and scans have a list to use.
    if centroid_page.centroids.is_empty() {
        centroid_page = IvfCentroidPage::new(vec![vector.clone()]);
        unsafe {
            centroid_page.store(&index_rel, false);
        }
    }

    // Find the nearest centroid.
    let nearest = find_nearest_centroids(
        &vector,
        &centroid_page.centroids,
        meta.get_distance_type(),
        1,
    );
    if nearest.is_empty() {
        return false;
    }
    let list_id = nearest[0] as u16;

    // Quantize relative to the list centroid.
    let quantizer = RabitqQuantizer::new(
        meta.get_bq_num_bits_per_dimension(),
        meta.get_rotation_seed(),
        meta.get_num_dimensions() as usize,
    );
    let centroid = &centroid_page.centroids[list_id as usize];
    let code = quantizer.quantize_residual(centroid, &vector);
    let entry = IvfEntry::new(ItemPointer::with_item_pointer_data(*heap_tid), code);

    // Append the entry to the list: read existing entries, push the new one,
    // and rewrite the list as a contiguous entry block run.
    let reader = IvfEntryReader::new(&index_rel);
    let (start_page, num_blocks) = list_directory
        .get_list(list_id)
        .map(|m| (m.start_page, m.num_blocks))
        .unwrap_or((pg_sys::InvalidBlockNumber, 0));
    let mut entries = if start_page != pg_sys::InvalidBlockNumber && num_blocks > 0 {
        reader.read_entries(start_page, num_blocks)
    } else {
        Vec::new()
    };
    entries.push(entry);

    let mut writer = IvfEntryWriter::new(&index_rel, list_id);
    for e in &entries {
        writer.add_entry(e.clone());
    }
    let (start_page, num_blocks, count) = writer.finish();

    if let Some(list_meta) = list_directory.get_list_mut(list_id) {
        list_meta.start_page = start_page.unwrap_or(pg_sys::InvalidBlockNumber);
        list_meta.num_blocks = num_blocks;
        list_meta.insert_page = list_meta.start_page;
        list_meta.num_tuples = count as u64;
    }

    unsafe {
        list_directory.store(&index_rel, false);
        // The scan bulk-reads entry blocks via smgr (bypassing shared_buffers),
        // so flush the newly written blocks to disk before they are visible.
        pg_sys::FlushRelationBuffers(index);
    }

    false
}
