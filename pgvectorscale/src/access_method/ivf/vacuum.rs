//! IVF index vacuum implementation.

use pgrx::*;

use crate::access_method::ivf::centroid_page::IvfCentroidPage;
use crate::access_method::ivf::entry::{seal_entries, IvfEntryReader};
use crate::access_method::ivf::list_directory::IvfListDirectory;
use crate::access_method::ivf::meta_page::IvfMetaPage;
use crate::access_method::ivf::segment::{IvfListHeader, IvfSegmentList};

/// Bulk delete tuples from the IVF index.
///
/// For each inverted list, reads the entries of every published segment, drops
/// those whose heap TID the `callback` reports as dead, seals the survivors
/// into ONE fresh merged segment (append-only, old segments stay immutable),
/// and atomically swaps the list header to point at it.  Concurrent scans keep
/// reading the old segments until reclamation (a later phase).
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
    let meta = IvfMetaPage::fetch(&index_rel);
    let _centroid_page = match meta.get_centroids_pointer() {
        Some(p) => IvfCentroidPage::load(&index_rel, p),
        None => IvfCentroidPage::new(Vec::new()),
    };
    let mut list_directory = IvfListDirectory::load(&index_rel);

    let reader = IvfEntryReader::new(&index_rel);
    let mut total_live = 0u64;
    let mut total_dead = 0u64;

    for list_id in 0..list_directory.num_lists() {
        let header_block = match list_directory.get_list(list_id as u16) {
            Some(m) if m.header.is_valid() => m.header.block_number,
            _ => continue,
        };

        // Read-modify-write under the header's exclusive content lock so a
        // concurrent insert's seal cannot be clobbered (the header is always
        // re-parsed under the lock before publishing).
        let dead_before = total_dead;
        let num_entries = unsafe {
            IvfListHeader::update(&index_rel, header_block, |header| {
                let segment_list = IvfSegmentList::load(&index_rel, header.segment_list);

                let mut live_entries = Vec::new();
                for segment in &segment_list.segments {
                    if segment.start_page == pg_sys::InvalidBlockNumber
                        || segment.num_blocks == 0
                    {
                        continue;
                    }
                    let entries = reader.read_entries(segment.start_page, segment.num_blocks);
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
                }

                // Skip the rewrite when nothing died and there is nothing to
                // merge (a single segment is already as compact as it gets).
                let current_total: u64 =
                    segment_list.segments.iter().map(|s| s.num_entries).sum();
                if total_dead == dead_before && segment_list.segments.len() <= 1 {
                    return current_total;
                }

                // Seal the survivors into one merged segment and swap the
                // header; the old segment-list item and the old segments
                // become garbage and are retired for reclamation.
                let segment = seal_entries(&index_rel, live_entries);
                let num_entries = segment.num_entries;
                let segments = if segment.is_empty() {
                    Vec::new()
                } else {
                    vec![segment]
                };
                let segment_list_new = IvfSegmentList::new(segments);
                let (new_ptr, new_blocks) = segment_list_new.store(&index_rel);
                // The merged blocks must be on disk before the header swap
                // becomes visible to smgrreadv scans: flush inside the closure,
                // i.e. BEFORE the header page itself is rewritten on exit.
                pg_sys::FlushRelationBuffers(index_rel.as_ptr());

                // Retire the old segment-list item and the old segments.
                let retired_generation = header.generation + 1;
                let mut retired = vec![crate::access_method::ivf::segment::IvfRetiredRange {
                    start_block: header.segment_list.block_number,
                    num_blocks: header.segment_list_blocks,
                    retired_generation,
                }];
                for s in &segment_list.segments {
                    if s.start_page != pg_sys::InvalidBlockNumber && s.num_blocks > 0 {
                        retired.push(crate::access_method::ivf::segment::IvfRetiredRange {
                            start_block: s.start_page,
                            num_blocks: s.num_blocks,
                            retired_generation,
                        });
                    }
                }
                IvfMetaPage::retire_ranges(&index_rel, retired);

                header.version += 1;
                header.generation += 1;
                header.segment_list = new_ptr;
                header.segment_list_blocks = new_blocks;
                header.active = None;
                num_entries
            })
        };

        if let Some(list_meta) = list_directory.get_list_mut(list_id as u16) {
            list_meta.num_tuples = num_entries;
        }
    }

    unsafe {
        list_directory.store(&index_rel, false);
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
