//! IVF index vacuum implementation.

use pgrx::*;

use crate::access_method::ivf::centroid_page::IvfCentroidPage;
use crate::access_method::ivf::entry::{IvfEntryReader, IvfEntryWriter};
use crate::access_method::ivf::list_directory::IvfListDirectory;
use crate::access_method::ivf::meta_page::IvfMetaPage;

/// Bulk delete tuples from the IVF index.
///
/// For each inverted list, reads the entries, drops those whose heap TID the
/// `callback` reports as dead, and rewrites the list with the survivors.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambulkdelete(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    callback: pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut std::os::raw::c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let results = if stats.is_null() {
        unsafe { PgBox::<pg_sys::IndexBulkDeleteResult>::alloc0().into_pg() }
    } else {
        stats
    };

    let index_rel = unsafe { PgRelation::from_pg((*info).index) };
    let _meta = IvfMetaPage::fetch(&index_rel);
    let _centroid_page = IvfCentroidPage::load(&index_rel);
    let mut list_directory = IvfListDirectory::load(&index_rel);

    let reader = IvfEntryReader::new(&index_rel);
    let mut total_live = 0u64;
    let mut total_dead = 0u64;

    for list_id in 0..list_directory.num_lists() {
        let (start_page, num_blocks) = list_directory
            .get_list(list_id as u16)
            .map(|m| (m.start_page, m.num_blocks))
            .unwrap_or((pg_sys::InvalidBlockNumber, 0));
        if start_page == pg_sys::InvalidBlockNumber || num_blocks == 0 {
            continue;
        }

        let entries = reader.read_entries(start_page, num_blocks);
        let mut live_entries = Vec::new();
        for entry in entries {
            let is_dead = if let Some(cb) = callback {
                let mut tid_data = pg_sys::ItemPointerData::default();
                entry.heap_tid.to_item_pointer_data(&mut tid_data);
                unsafe { cb(&mut tid_data, callback_state) }
            } else {
                false
            };
            if is_dead {
                total_dead += 1;
            } else {
                live_entries.push(entry);
                total_live += 1;
            }
        }

        // Rewrite the list with the surviving entries.
        let mut writer = IvfEntryWriter::new(&index_rel, list_id as u16);
        for e in &live_entries {
            writer.add_entry(e.clone());
        }
        let (new_start, new_blocks, count) = writer.finish();
        if let Some(list_meta) = list_directory.get_list_mut(list_id as u16) {
            list_meta.start_page = new_start.unwrap_or(pg_sys::InvalidBlockNumber);
            list_meta.num_blocks = new_blocks;
            list_meta.insert_page = list_meta.start_page;
            list_meta.num_tuples = count as u64;
        }
    }

    unsafe {
        list_directory.store(&index_rel, false);
        // Bulk smgr scans need the rewritten blocks on disk first.
        pg_sys::FlushRelationBuffers(index_rel.as_ptr());
        (*results).pages_deleted = total_dead as u32;
        (*results).num_index_tuples = total_live as f64;
    }

    results
}

/// Cleanup after vacuum.
#[pg_guard]
pub unsafe extern "C-unwind" fn amvacuumcleanup(
    _info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    stats
}
