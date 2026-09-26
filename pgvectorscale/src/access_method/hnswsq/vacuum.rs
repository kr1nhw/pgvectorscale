//! hnswsq vacuum — the Rust translation of pgvector's `hnswvacuum.c`.
//!
//! Four passes, exactly as in the reference:
//! 1. `remove_heap_tids`: walk the graph, drop dead heap TIDs, collect the
//!    deletion set and the highest/fallback points;
//! 2. `repair_graph`: repair the entry point, then every element whose
//!    neighbor list references a deleted element;
//! 3. `confirm_repaired`: verify no live list references the deletion set;
//! 4. `mark_deleted`: overwrite deleted elements (zeroed vector, cleared
//!    neighbor tuple, bumped version) and reset the insert-page hint.
//!
//! Divergences from the reference, and only these: the walk starts at the
//! metapage's recorded `graph_head` (the SQ8 calibration chain may occupy
//! blocks before the graph pages), and the deleting set is a Rust open-
//! addressing table.

use pgrx::pg_sys;
use pgrx::*;

use crate::access_method::hnswsq::insert::update_neighbors_on_disk;
use crate::access_method::hnswsq::types::*;
use crate::access_method::hnswsq::utils::*;
use crate::util::ports::{PageGetItem, PageGetItemId, PageGetMaxOffsetNumber};

/// The vacuum state (pgvector `HnswVacuumState`).
pub struct VacuumState {
    pub index: pg_sys::Relation,
    /// Base block of the hnswsq region (0 for the standalone AM).
    pub base: pg_sys::BlockNumber,
    pub stats: *mut pg_sys::IndexBulkDeleteResult,
    pub callback: pg_sys::IndexBulkDeleteCallback,
    pub callback_state: *mut std::os::raw::c_void,
    pub m: usize,
    pub ef_construction: usize,
    pub support: Support,
    pub deleting: Visited,
    pub bas: pg_sys::BufferAccessStrategy,
    pub ntup: Vec<u8>,
    pub highest: Box<Element>,
    pub fallback: Box<Element>,
    pub tmp_ctx: PgMemoryContexts,
    pub scratch: SearchScratch,
    pub decode: Vec<f32>,
    pub pair_scratch: Vec<f32>,
    pub visited: Visited,
}

/// `DeletingElement` (hnswvacuum.c).
unsafe fn deleting_element(deleting: &Visited, tid: pg_sys::ItemPointerData) -> bool {
    // A lookup, not an insert: use a probe against the same table semantics.
    deleting.contains(pack_tid(tid))
}

/// `RemoveHeapTids` (hnswvacuum.c): pass 1.
unsafe fn remove_heap_tids(vac: &mut VacuumState) {
    let index = vac.index;
    let highest = &mut *vac.highest;
    let fallback = &mut *vac.fallback;

    // Store separately since the element level is u8
    let mut highest_level: i32 = -1;
    let mut fallback_level: i32 = -1;

    // Initialize highest point and fallback point
    highest.blkno = pg_sys::InvalidBlockNumber;
    highest.offno = pg_sys::InvalidOffsetNumber;
    fallback.blkno = pg_sys::InvalidBlockNumber;
    fallback.offno = pg_sys::InvalidOffsetNumber;

    let mut blkno = meta_graph_head(index, vac.base);
    while blkno != pg_sys::InvalidBlockNumber {
        check_for_interrupts!();

        let buf = pg_sys::ReadBufferExtended(
            index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            blkno,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            vac.bas,
        );
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
        let state = pg_sys::GenericXLogStart(index);
        let page = pg_sys::GenericXLogRegisterBuffer(state, buf, 0);
        let maxoffno = PageGetMaxOffsetNumber(page);
        let mut updated = false;

        // Iterate over nodes
        let mut offno = pg_sys::FirstOffsetNumber;
        while offno <= maxoffno as pg_sys::OffsetNumber {
            let etup = PageGetItem(page, PageGetItemId(page, offno)).cast::<ElementTupleData>();
            let mut item_updated = false;

            // Skip neighbor tuples
            if (*etup).type_ != ELEMENT_TUPLE_TYPE {
                offno += 1;
                continue;
            }

            // Skip deleted tuples (they must not enter the deletion list, to
            // avoid false positives in NeedsUpdated and ConfirmRepaired)
            if (*etup).deleted != 0 {
                offno += 1;
                continue;
            }

            if ip_block(&(*etup).heaptid) != pg_sys::InvalidBlockNumber {
                let dead = (vac.callback).expect("bulkdelete callback")(
                    &mut (*etup).heaptid,
                    vac.callback_state,
                );
                if dead {
                    item_updated = true;
                    (*vac.stats).tuples_removed += 1.0;
                } else {
                    (*vac.stats).num_index_tuples += 1.0;
                }

                if item_updated {
                    pgrx::itemptr::item_pointer_set_all(
                        &mut (*etup).heaptid,
                        pg_sys::InvalidBlockNumber,
                        pg_sys::InvalidOffsetNumber,
                    );
                    updated = true;
                }
            }

            if ip_block(&(*etup).heaptid) == pg_sys::InvalidBlockNumber {
                // Add to deletion list
                let mut indextid = pg_sys::ItemPointerData::default();
                pgrx::itemptr::item_pointer_set_all(&mut indextid, blkno, offno);
                let found = vac.deleting.insert(pack_tid(indextid));
                debug_assert!(!found, "deletion list got a duplicate");
            } else if (*etup).level as i32 > highest_level {
                if highest.blkno != pg_sys::InvalidBlockNumber {
                    // Current highest point becomes fallback
                    fallback.blkno = highest.blkno;
                    fallback.offno = highest.offno;
                    fallback.level = highest.level;
                    fallback_level = highest_level;
                }

                // Keep track of highest point
                highest.blkno = blkno;
                highest.offno = offno;
                highest.level = (*etup).level;
                highest_level = (*etup).level as i32;
            } else if (*etup).level as i32 > fallback_level {
                // Keep track of second highest point
                fallback.blkno = blkno;
                fallback.offno = offno;
                fallback.level = (*etup).level;
                fallback_level = (*etup).level as i32;
            }

            offno += 1;
        }

        blkno = (*page_opaque(page)).nextblkno;

        if updated {
            pg_sys::GenericXLogFinish(state);
        } else {
            pg_sys::GenericXLogAbort(state);
        }

        pg_sys::UnlockReleaseBuffer(buf);
    }
}

/// `NeedsUpdated` (hnswvacuum.c): does the element's neighbor tuple reference
/// a deleted element (or has a not-full layer 0)?
unsafe fn needs_updated(vac: &VacuumState, element: *mut Element) -> bool {
    let buf = pg_sys::ReadBufferExtended(
        vac.index,
        pg_sys::ForkNumber::MAIN_FORKNUM,
        (*element).neighbor_page,
        pg_sys::ReadBufferMode::RBM_NORMAL,
        vac.bas,
    );
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = pg_sys::BufferGetPage(buf);
    let ntup = PageGetItem(page, PageGetItemId(page, (*element).neighbor_offno))
        .cast::<NeighborTupleData>();

    debug_assert_eq!((*ntup).type_, NEIGHBOR_TUPLE_TYPE);

    let mut needs = false;
    let tids = ntup
        .cast::<u8>()
        .add(NEIGHBOR_TUPLE_HEADER_SIZE)
        .cast::<pg_sys::ItemPointerData>();

    // Check neighbors
    for i in 0..(*ntup).count as usize {
        let indextid = &*tids.add(i);
        if ip_block(indextid) == pg_sys::InvalidBlockNumber {
            continue;
        }
        // Check if in deletion list
        if deleting_element(&vac.deleting, *indextid) {
            needs = true;
            break;
        }
    }

    // Also update if layer 0 is not full (this could indicate too many
    // candidates being deleted during insert; there should always be more
    // than zero indextids, but check for safety)
    if !needs && (*ntup).count > 0 {
        needs = ip_block(&*tids.add((*ntup).count as usize - 1)) == pg_sys::InvalidBlockNumber;
    }

    pg_sys::UnlockReleaseBuffer(buf);
    needs
}

/// `RepairGraphElement` (hnswvacuum.c).
unsafe fn repair_graph_element(
    vac: &mut VacuumState,
    element: *mut Element,
    entry_point: Option<*mut Element>,
) {
    let index = vac.index;
    let m = vac.m;
    let ef_construction = vac.ef_construction;
    let support = &vac.support;

    // Skip if element is entry point
    if let Some(ep) = entry_point {
        if (*element).blkno == (*ep).blkno && (*element).offno == (*ep).offno {
            return;
        }
    }

    // Init fields
    let allocator = Allocator::Palloc;
    init_neighbors(std::ptr::null_mut(), element, m, &allocator);
    (*element).heaptid_set = 0;

    // Find neighbors for element, skipping itself
    find_element_neighbors(
        std::ptr::null_mut(),
        element,
        entry_point,
        Some(index),
        support,
        m,
        ef_construction,
        true,
        &mut vac.scratch,
        &mut vac.decode,
        &mut vac.pair_scratch,
        &mut vac.visited,
        crate::access_method::hnswsq::options::HNSW_SQ8_DISTANCE.get(),
    );

    // Update neighbor tuple (before getting the page, to minimize locking)
    let ntup_size = neighbor_tuple_size((*element).level as usize, m);
    vac.ntup.fill(0);
    set_neighbor_tuple(
        std::ptr::null_mut(),
        vac.ntup.as_mut_ptr().cast::<NeighborTupleData>(),
        element,
        m,
    );

    // Get neighbor page
    let buf = pg_sys::ReadBufferExtended(
        index,
        pg_sys::ForkNumber::MAIN_FORKNUM,
        (*element).neighbor_page,
        pg_sys::ReadBufferMode::RBM_NORMAL,
        vac.bas,
    );
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
    let state = pg_sys::GenericXLogStart(index);
    let page = pg_sys::GenericXLogRegisterBuffer(state, buf, 0);

    // Overwrite tuple
    if !pg_sys::PageIndexTupleOverwrite(
        page,
        (*element).neighbor_offno,
        vac.ntup.as_mut_ptr().cast(),
        ntup_size,
    ) {
        error!("hnswsq: failed to add index item");
    }

    // Commit
    pg_sys::GenericXLogFinish(state);
    pg_sys::UnlockReleaseBuffer(buf);

    // Update neighbors (vacuum-local scratch: this is per repaired element,
    // not the per-row insert hot path)
    let mut pair_scratch = vec![0.0f32; support.codec.dim()];
    let mut decode = vec![0.0f32; support.codec.dim()];
    let mut na_array: Vec<u8> = Vec::new();
    let mut na_tids: Vec<pg_sys::ItemPointerData> = Vec::new();
    let mut na_elements = ElementArena::new();
    update_neighbors_on_disk(
        index,
        vac.base,
        support,
        element,
        m,
        false,
        &mut pair_scratch,
        &mut decode,
        &mut na_array,
        &mut na_tids,
        &mut na_elements,
    );
}

/// `RepairGraphEntryPoint` (hnswvacuum.c).
unsafe fn repair_graph_entry_point(vac: &mut VacuumState) {
    // Repair graph for highest non-entry point. Highest point may be outdated
    // due to inserts that happen during and after RemoveHeapTids.
    let mut highest: *mut Element = &mut *vac.highest;
    if (*highest).blkno == pg_sys::InvalidBlockNumber {
        highest = std::ptr::null_mut();
    }

    if !highest.is_null() {
        // Get a shared lock
        pg_sys::LockPage(
            vac.index,
            update_lock_page(vac.base),
            pg_sys::ShareLock as pg_sys::LOCKMODE,
        );

        // Get latest entry point
        let mut entry = get_entry_point(vac.index, vac.base);

        // Use fallback point if highest point is entry point
        if let Some(ep) = entry.as_deref() {
            if (*ep).blkno == (*highest).blkno && (*ep).offno == (*highest).offno {
                let mut fallback: *mut Element = &mut *vac.fallback;
                if (*fallback).blkno == pg_sys::InvalidBlockNumber {
                    highest = std::ptr::null_mut();
                } else {
                    highest = fallback;
                }
            }
        }

        if !highest.is_null() {
            // Load element
            load_element(
                highest,
                None,
                None,
                vac.index,
                &vac.support,
                true,
                None,
                None,
            );

            // Repair if needed
            if needs_updated(vac, highest) {
                let entry_ptr = entry.as_deref_mut().map(|e| e as *mut Element);
                repair_graph_element(vac, highest, entry_ptr);
            }
        }

        // Release lock
        pg_sys::UnlockPage(
            vac.index,
            update_lock_page(vac.base),
            pg_sys::ShareLock as pg_sys::LOCKMODE,
        );
    }

    // Prevent concurrent inserts when possibly updating entry point
    pg_sys::LockPage(
        vac.index,
        update_lock_page(vac.base),
        pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
    );

    // Get latest entry point
    let mut entry = get_entry_point(vac.index, vac.base);

    if let Some(entry) = entry.as_deref_mut() {
        let entry_ptr = entry as *mut Element;
        let mut ep_data = pg_sys::ItemPointerData::default();
        pgrx::itemptr::item_pointer_set_all(&mut ep_data, (*entry).blkno, (*entry).offno);

        if deleting_element(&vac.deleting, ep_data) {
            // Replace the entry point with the highest point. If highest
            // point is outdated and empty, the entry point will be empty
            // until an element is repaired.
            update_meta_page(
                vac.index,
                vac.base,
                UPDATE_ENTRY_ALWAYS,
                if highest.is_null() {
                    None
                } else {
                    Some(highest)
                },
                pg_sys::InvalidBlockNumber,
                false,
            );
        } else {
            // Repair the entry point with the highest point. If highest point
            // is outdated, this can remove connections at higher levels in
            // the graph until they are repaired, but this should be fine.
            load_element(
                entry_ptr,
                None,
                None,
                vac.index,
                &vac.support,
                true,
                None,
                None,
            );

            if needs_updated(vac, entry_ptr) {
                // Reset neighbors from previous update
                if !highest.is_null() {
                    crate::access_method::hnswsq::ptr::store(
                        std::ptr::null_mut(),
                        &mut (*highest).neighbors,
                        std::ptr::null_mut::<u8>(),
                    );
                }

                repair_graph_element(
                    vac,
                    entry_ptr,
                    if highest.is_null() {
                        None
                    } else {
                        Some(highest)
                    },
                );
            }
        }
    }

    // Release lock
    pg_sys::UnlockPage(
        vac.index,
        update_lock_page(vac.base),
        pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
    );
}

/// `RepairGraph` (hnswvacuum.c): pass 2.
unsafe fn repair_graph(vac: &mut VacuumState) {
    let index = vac.index;

    // Wait for inserts to complete. Inserts before this point may have
    // neighbors about to be deleted. Inserts after this point will not.
    pg_sys::LockPage(
        index,
        update_lock_page(vac.base),
        pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
    );
    pg_sys::UnlockPage(
        index,
        update_lock_page(vac.base),
        pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
    );

    // Repair entry point first
    repair_graph_entry_point(vac);

    let mut blkno = meta_graph_head(index, vac.base);
    while blkno != pg_sys::InvalidBlockNumber {
        check_for_interrupts!();

        // Everything per page — loading and the repair itself — runs in the
        // temporary context, reset per page (pgvector switches to tmpCtx for
        // the same scope).
        let old_ctx = pg_sys::CurrentMemoryContext;
        pg_sys::CurrentMemoryContext = vac.tmp_ctx.value();

        let mut elements: Vec<Box<Element>> = Vec::new();
        {
            let buf = pg_sys::ReadBufferExtended(
                index,
                pg_sys::ForkNumber::MAIN_FORKNUM,
                blkno,
                pg_sys::ReadBufferMode::RBM_NORMAL,
                vac.bas,
            );
            pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
            let page = pg_sys::BufferGetPage(buf);
            let maxoffno = PageGetMaxOffsetNumber(page);

            // Load items into memory to minimize locking
            let mut offno = pg_sys::FirstOffsetNumber;
            while offno <= maxoffno as pg_sys::OffsetNumber {
                let etup = PageGetItem(page, PageGetItemId(page, offno)).cast::<ElementTupleData>();

                // Skip neighbor tuples
                if (*etup).type_ != ELEMENT_TUPLE_TYPE {
                    offno += 1;
                    continue;
                }
                // Skip deleted tuples
                if (*etup).deleted != 0 {
                    offno += 1;
                    continue;
                }
                // Skip updating neighbors if being deleted
                if ip_block(&(*etup).heaptid) == pg_sys::InvalidBlockNumber {
                    offno += 1;
                    continue;
                }

                // Create an element
                let mut element = init_element_from_block(blkno, offno);
                load_element_from_tuple(
                    &mut *element,
                    etup,
                    false,
                    true,
                    vac.support.codec.vector_bytes(),
                );
                elements.push(element);

                offno += 1;
            }

            blkno = (*page_opaque(page)).nextblkno;

            pg_sys::UnlockReleaseBuffer(buf);
        }

        // Update neighbor pages
        for element in elements.iter_mut() {
            let element_ptr = element.as_mut() as *mut Element;
            let mut lockmode = pg_sys::ShareLock as pg_sys::LOCKMODE;

            // Check if any neighbors point to deleted values
            if !needs_updated(vac, element_ptr) {
                continue;
            }

            // Get a shared lock
            pg_sys::LockPage(index, update_lock_page(vac.base), lockmode);

            // Refresh entry point for each element
            let mut entry = get_entry_point(index, vac.base);
            let mut entry_ptr = entry.as_deref_mut().map(|e| e as *mut Element);

            // Prevent concurrent inserts when likely updating entry point
            if entry_ptr.is_none() || (*element_ptr).level > (*entry_ptr.unwrap()).level {
                // Release shared lock
                pg_sys::UnlockPage(index, update_lock_page(vac.base), lockmode);

                // Get exclusive lock
                lockmode = pg_sys::ExclusiveLock as pg_sys::LOCKMODE;
                pg_sys::LockPage(index, update_lock_page(vac.base), lockmode);

                // Get latest entry point after lock is acquired
                entry = get_entry_point(index, vac.base);
                entry_ptr = entry.as_deref_mut().map(|e| e as *mut Element);
            }

            // Repair connections
            repair_graph_element(vac, element_ptr, entry_ptr);

            // Update metapage if needed (only if the entry point was
            // replaced and the highest point was outdated)
            if entry_ptr.is_none() || (*element_ptr).level > (*entry_ptr.unwrap()).level {
                update_meta_page(
                    index,
                    vac.base,
                    UPDATE_ENTRY_GREATER,
                    Some(element_ptr),
                    pg_sys::InvalidBlockNumber,
                    false,
                );
            }

            // Release lock
            pg_sys::UnlockPage(index, update_lock_page(vac.base), lockmode);
        }

        pg_sys::CurrentMemoryContext = old_ctx;
        pg_sys::MemoryContextReset(vac.tmp_ctx.value());
    }
}

/// `ConfirmRepaired` (hnswvacuum.c).
unsafe fn confirm_repaired(vac: &VacuumState) {
    let index = vac.index;
    let mut blkno = meta_graph_head(index, vac.base);

    while blkno != pg_sys::InvalidBlockNumber {
        check_for_interrupts!();

        let buf = pg_sys::ReadBufferExtended(
            index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            blkno,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            vac.bas,
        );
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
        let page = pg_sys::BufferGetPage(buf);
        let maxoffno = PageGetMaxOffsetNumber(page);

        // Iterate over nodes
        let mut offno = pg_sys::FirstOffsetNumber;
        while offno <= maxoffno as pg_sys::OffsetNumber {
            let etup = PageGetItem(page, PageGetItemId(page, offno)).cast::<ElementTupleData>();

            // Skip neighbor tuples
            if (*etup).type_ != ELEMENT_TUPLE_TYPE {
                offno += 1;
                continue;
            }
            // Skip deleted tuples
            if (*etup).deleted != 0 {
                offno += 1;
                continue;
            }
            // Skip if being deleted
            if ip_block(&(*etup).heaptid) == pg_sys::InvalidBlockNumber {
                offno += 1;
                continue;
            }

            // Get neighbor page
            let neighbor_page = ip_block(&(*etup).neighbortid);
            let neighbor_offno = ip_offset(&(*etup).neighbortid);

            let nbuf: pg_sys::Buffer;
            let npage: pg_sys::Page;
            if neighbor_page == blkno {
                nbuf = buf;
                npage = page;
            } else {
                nbuf = pg_sys::ReadBufferExtended(
                    index,
                    pg_sys::ForkNumber::MAIN_FORKNUM,
                    neighbor_page,
                    pg_sys::ReadBufferMode::RBM_NORMAL,
                    vac.bas,
                );
                pg_sys::LockBuffer(nbuf, pg_sys::BUFFER_LOCK_SHARE as i32);
                npage = pg_sys::BufferGetPage(nbuf);
            }

            let ntup = PageGetItem(npage, PageGetItemId(npage, neighbor_offno))
                .cast::<NeighborTupleData>();
            let tids = ntup
                .cast::<u8>()
                .add(NEIGHBOR_TUPLE_HEADER_SIZE)
                .cast::<pg_sys::ItemPointerData>();

            // Check neighbors
            for i in 0..(*ntup).count as usize {
                let indextid = &*tids.add(i);
                if ip_block(indextid) == pg_sys::InvalidBlockNumber {
                    continue;
                }
                // Check if in deletion list
                if deleting_element(&vac.deleting, *indextid) {
                    // pgvector errors here without unlocking: the error
                    // recovery releases the buffers.
                    error!("hnswsq graph not repaired");
                }
            }

            if nbuf != buf {
                pg_sys::UnlockReleaseBuffer(nbuf);
            }

            offno += 1;
        }

        blkno = (*page_opaque(page)).nextblkno;

        pg_sys::UnlockReleaseBuffer(buf);
    }
}

/// `MarkDeleted` (hnswvacuum.c): passes 3 and 4.
unsafe fn mark_deleted(vac: &mut VacuumState) {
    let index = vac.index;
    let mut insert_page = pg_sys::InvalidBlockNumber;

    // Wait for inserts and index scans to complete. Inserts and scans before
    // this point may visit tuples about to be deleted. Inserts and scans
    // after this point will not, since the graph has been repaired.
    pg_sys::LockPage(
        index,
        update_lock_page(vac.base),
        pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
    );
    pg_sys::UnlockPage(
        index,
        update_lock_page(vac.base),
        pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
    );

    confirm_repaired(vac);

    pg_sys::LockPage(
        index,
        scan_lock_page(vac.base),
        pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
    );
    pg_sys::UnlockPage(
        index,
        scan_lock_page(vac.base),
        pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
    );

    let mut blkno = meta_graph_head(index, vac.base);
    let vec_bytes = vac.support.codec.vector_bytes();

    while blkno != pg_sys::InvalidBlockNumber {
        check_for_interrupts!();

        let buf = pg_sys::ReadBufferExtended(
            index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            blkno,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            vac.bas,
        );

        // ambulkdelete cannot delete entries from pages that are pinned by
        // other backends
        pg_sys::LockBufferForCleanup(buf);

        let mut state = pg_sys::GenericXLogStart(index);
        let mut page = pg_sys::GenericXLogRegisterBuffer(state, buf, 0);
        let maxoffno = PageGetMaxOffsetNumber(page);

        // Update element and neighbors together
        let mut offno = pg_sys::FirstOffsetNumber;
        while offno <= maxoffno as pg_sys::OffsetNumber {
            let etup = PageGetItem(page, PageGetItemId(page, offno)).cast::<ElementTupleData>();

            // Skip neighbor tuples
            if (*etup).type_ != ELEMENT_TUPLE_TYPE {
                offno += 1;
                continue;
            }

            // Skip deleted tuples
            if (*etup).deleted != 0 {
                // Set to first free page
                if insert_page == pg_sys::InvalidBlockNumber {
                    insert_page = blkno;
                }
                offno += 1;
                continue;
            }

            // Skip live tuples
            if ip_block(&(*etup).heaptid) != pg_sys::InvalidBlockNumber {
                offno += 1;
                continue;
            }

            // Get neighbor page
            let neighbor_page = ip_block(&(*etup).neighbortid);
            let neighbor_offno = ip_offset(&(*etup).neighbortid);

            let nbuf: pg_sys::Buffer;
            let npage: pg_sys::Page;
            if neighbor_page == blkno {
                nbuf = buf;
                npage = page;
            } else {
                nbuf = pg_sys::ReadBufferExtended(
                    index,
                    pg_sys::ForkNumber::MAIN_FORKNUM,
                    neighbor_page,
                    pg_sys::ReadBufferMode::RBM_NORMAL,
                    vac.bas,
                );
                pg_sys::LockBuffer(nbuf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
                npage = pg_sys::GenericXLogRegisterBuffer(state, nbuf, 0);
            }

            let ntup = PageGetItem(npage, PageGetItemId(npage, neighbor_offno))
                .cast::<NeighborTupleData>();

            // Overwrite element
            (*etup).deleted = 1;
            std::ptr::write_bytes(
                etup.cast::<u8>().add(ELEMENT_TUPLE_VECTOR_OFFSET),
                0,
                vec_bytes,
            );

            // Overwrite neighbors
            let tids = ntup
                .cast::<u8>()
                .add(NEIGHBOR_TUPLE_HEADER_SIZE)
                .cast::<pg_sys::ItemPointerData>();
            for i in 0..(*ntup).count as usize {
                pgrx::itemptr::item_pointer_set_all(
                    &mut *tids.add(i),
                    pg_sys::InvalidBlockNumber,
                    pg_sys::InvalidOffsetNumber,
                );
            }

            // Increment version (avoids incorrect reads for iterative scans;
            // reserve some bits for future use)
            (*etup).version = (*etup).version.wrapping_add(1);
            if (*etup).version > 15 {
                (*etup).version = 1;
            }
            (*ntup).version = (*etup).version;

            // We modified the tuples in place, no need to call
            // PageIndexTupleOverwrite

            // Commit
            pg_sys::GenericXLogFinish(state);
            if nbuf != buf {
                pg_sys::UnlockReleaseBuffer(nbuf);
            }

            // Set to first free page
            if insert_page == pg_sys::InvalidBlockNumber {
                insert_page = blkno;
            }

            // Prepare new xlog
            state = pg_sys::GenericXLogStart(index);
            page = pg_sys::GenericXLogRegisterBuffer(state, buf, 0);

            offno += 1;
        }

        blkno = (*page_opaque(page)).nextblkno;

        pg_sys::GenericXLogAbort(state);
        pg_sys::UnlockReleaseBuffer(buf);
    }

    // Update insert page last, after everything has been marked as deleted
    update_meta_page(index, vac.base, 0, None, insert_page, false);
}

/// The graph's head block from the metapage (pgvector's `HNSW_HEAD_BLKNO`,
/// recorded because the calibration chain may precede the graph pages).
pub unsafe fn meta_graph_head(
    index: pg_sys::Relation,
    base: pg_sys::BlockNumber,
) -> pg_sys::BlockNumber {
    let buf = pg_sys::ReadBuffer(index, metapage_block(base));
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = pg_sys::BufferGetPage(buf);
    let metap = page_get_meta(page);
    let head = (*metap).graph_head;
    pg_sys::UnlockReleaseBuffer(buf);
    head
}

// ---------------------------------------------------------------------------
// AM entry points (hnswvacuum.c: hnswbulkdelete / hnswvacuumcleanup)
// ---------------------------------------------------------------------------

/// `hnswbulkdelete` (hnswvacuum.c).
#[pg_guard]
pub unsafe extern "C-unwind" fn ambulkdelete(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    callback: pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut std::os::raw::c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let index = (*info).index;
    vacuum_region(index, HNSW_STANDALONE_BASE, callback, callback_state, stats)
}

/// Run the full vacuum pass over one hnswsq region: the standalone AM's whole
/// index, or an embedded AgentVec HOT segment at `base`.  The lock-page
/// anchors, the graph walk and the metapage updates are all region-local.
pub unsafe fn vacuum_region(
    index: pg_sys::Relation,
    base: pg_sys::BlockNumber,
    callback: pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut std::os::raw::c_void,
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let stats = if stats.is_null() {
        pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBulkDeleteResult>())
            .cast::<pg_sys::IndexBulkDeleteResult>()
    } else {
        stats
    };

    let support = init_support(index, base);
    let mut m = 0usize;
    get_meta_page_info(index, base, Some(&mut m), None);
    let dim = support.codec.dim();
    let (_m_metapage, ef) = region_params(index, base);

    let mut vac = VacuumState {
        index,
        base,
        stats,
        callback,
        callback_state,
        m,
        ef_construction: ef,
        support,
        deleting: Visited::new(256),
        bas: pg_sys::GetAccessStrategy(pg_sys::BufferAccessStrategyType::BAS_BULKREAD),
        ntup: vec![0u8; pg_sys::BLCKSZ as usize],
        highest: init_element_from_block(pg_sys::InvalidBlockNumber, pg_sys::InvalidOffsetNumber),
        fallback: init_element_from_block(pg_sys::InvalidBlockNumber, pg_sys::InvalidOffsetNumber),
        tmp_ctx: PgMemoryContexts::new("hnswsq vacuum temporary context"),
        scratch: SearchScratch::new(m),
        decode: Vec::with_capacity(dim),
        pair_scratch: Vec::with_capacity(dim),
        visited: Visited::new(ef * m * 2),
    };

    // Pass 1: Remove heap TIDs
    remove_heap_tids(&mut vac);

    // Pass 2: Repair graph
    repair_graph(&mut vac);

    // Passes 3 and 4: Confirm repaired and mark as deleted
    mark_deleted(&mut vac);

    vac.stats
}

/// `hnswvacuumcleanup` (hnswvacuum.c).
#[pg_guard]
pub unsafe extern "C-unwind" fn amvacuumcleanup(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let rel = (*info).index;

    if (*info).analyze_only {
        return stats;
    }

    // stats is NULL if ambulkdelete not called; OK to return NULL if index
    // not changed
    if stats.is_null() {
        return std::ptr::null_mut();
    }

    (*stats).num_pages =
        pg_sys::RelationGetNumberOfBlocksInFork(rel, pg_sys::ForkNumber::MAIN_FORKNUM);

    stats
}
