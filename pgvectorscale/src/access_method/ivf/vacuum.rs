//! IVF index vacuum implementation.

use pgrx::*;

use crate::access_method::ivf::centroid_page::IvfCentroidPage;
use crate::access_method::ivf::entry::{seal_entries, IvfEntryReader};
use crate::access_method::ivf::list_directory::IvfListDirectory;
use crate::access_method::ivf::meta_page::IvfMetaPage;
use crate::access_method::ivf::segment::{IvfFreeRange, IvfListHeader, IvfSegmentList};

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
        let header_ptr = match list_directory.get_list(list_id as u16) {
            Some(m) if m.header.is_valid() => m.header,
            _ => continue,
        };
        let header_block = header_ptr.block_number;

        // Optimistically reserve a reclaimed block for the new segment-list
        // item when this list looks like it will be rewritten (multiple
        // segments or an active buffer).  The advisory-exclusive lock must
        // not nest inside the header lock, so the reservation happens here.
        let peek_rewrite = {
            let peek = IvfListHeader::load(&index_rel, header_ptr);
            let segs = IvfSegmentList::load(&index_rel, peek.segment_list).segments.len();
            segs > 1 || peek.active.is_some()
        };
        let reserved_item: Option<pg_sys::BlockNumber> = if peek_rewrite {
            unsafe { IvfMetaPage::allocate_range(&index_rel, 1) }
        } else {
            None
        };

        // Read-modify-write under the header's exclusive content lock so a
        // concurrent insert's seal cannot be clobbered (the header is always
        // re-parsed under the lock before publishing).
        let dead_before = total_dead;
        let (num_entries, retired, item_used, unused_item) = unsafe {
            IvfListHeader::update(&index_rel, header_block, |header| {
                let segment_list = IvfSegmentList::load(&index_rel, header.segment_list);

                let mut live_entries = Vec::new();
                let mut process = |entries: Vec<crate::access_method::ivf::entry::IvfEntry>| {
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
                };
                for segment in &segment_list.segments {
                    if segment.start_page == pg_sys::InvalidBlockNumber
                        || segment.num_blocks == 0
                    {
                        continue;
                    }
                    let entries = reader.read_entries(segment.start_page, segment.num_blocks);
                    process(entries);
                }
                // The unpublished active buffer must be merged too: entries
                // for deleted rows left there would otherwise be published by a
                // later seal as stale TIDs (physically removed heap tuples),
                // which breaks index-only heap fetches downstream.
                let active_present = header.active.is_some();
                let mut active_pages: Vec<pg_sys::BlockNumber> = Vec::new();
                if let Some(active) = header.active.as_ref() {
                    let entries =
                        crate::access_method::ivf::entry::read_active_entries(&index_rel, active);
                    process(entries);
                    active_pages = active.pages.clone();
                }

                // Skip the rewrite when nothing died and there is nothing to
                // merge (a single published segment and no active buffer).
                let current_total: u64 =
                    segment_list.segments.iter().map(|s| s.num_entries).sum();
                if total_dead == dead_before
                    && segment_list.segments.len() <= 1
                    && !active_present
                {
                    return (current_total, Vec::new(), false, reserved_item);
                }

                // Seal the survivors into one merged segment and swap the
                // header; the old segment-list item and the old segments
                // become garbage and are retired for reclamation.
                let segment = seal_entries(&index_rel, live_entries);
                let num_entries = segment.num_entries;
                let seg_start = segment.start_page;
                let seg_blocks = segment.num_blocks;
                let seg_empty = segment.is_empty();
                let segments = if seg_empty {
                    Vec::new()
                } else {
                    vec![segment]
                };
                let segment_list_new = IvfSegmentList::new(segments);
                let mut item_used = false;
                let (new_ptr, new_blocks) = match reserved_item {
                    Some(b) if segment_list_new.fits_one_page() => {
                        item_used = true;
                        segment_list_new.store_at(&index_rel, b)
                    }
                    _ => segment_list_new.store(&index_rel),
                };
                // The merged blocks must be on disk before the header swap
                // becomes visible to smgrreadv scans: flush only the freshly
                // written blocks, inside the closure (i.e. BEFORE the header
                // page itself is rewritten on exit) — a full
                // FlushRelationBuffers here would try to flush the header page
                // we hold exclusively and self-deadlock.
                if !seg_empty {
                    crate::util::page::flush_block_range(&index_rel, seg_start, seg_blocks);
                }
                crate::util::page::flush_block_range(&index_rel, new_ptr.block_number, new_blocks);

                // Retire the old segment-list item and the old segments.
                // segment_list_blocks == 0 marks the shared empty item used by
                // never-sealed lists — never retire it.
                let mut retired = Vec::new();
                if header.segment_list_blocks > 0 {
                    retired.push(IvfFreeRange {
                        start_block: header.segment_list.block_number,
                        num_blocks: header.segment_list_blocks,
                    });
                }
                for s in &segment_list.segments {
                    if s.start_page != pg_sys::InvalidBlockNumber && s.num_blocks > 0 {
                        retired.push(IvfFreeRange {
                            start_block: s.start_page,
                            num_blocks: s.num_blocks,
                        });
                    }
                }
                // The merged active buffer's pages are garbage too.
                for &page in &active_pages {
                    retired.push(IvfFreeRange {
                        start_block: page,
                        num_blocks: 1,
                    });
                }

                header.version += 1;
                header.generation += 1;
                header.segment_list = new_ptr;
                header.segment_list_blocks = new_blocks;
                // The active buffer's entries were just merged into the new
                // segment, so clearing the pointer now is correct (they are
                // no longer unreachable — nothing is orphaned).
                header.active = None;
                (
                    num_entries,
                    retired,
                    item_used,
                    if item_used { None } else { reserved_item },
                )
            })
        };

        // Reclaim under the ExclusiveLock only AFTER the header lock was
        // released (the ExclusiveLock must never nest inside it).
        unsafe {
            if let Some(block) = unused_item {
                IvfMetaPage::push_back_range(&index_rel, block, 1);
            }
            IvfMetaPage::reclaim_ranges(&index_rel, retired);
        }

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
