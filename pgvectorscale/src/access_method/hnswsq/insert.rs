//! hnswsq insert — the Rust translation of pgvector's `hnswinsert.c`.
//!
//! The transactional (on-disk) insert: element/neighbor page packing with
//! deleted-tuple reuse, the two-tuple split path, backlink updates with the
//! connection heuristic, and the insert-page/entry-point hints.  Divergences
//! from the reference, and only these:
//!
//! * no duplicate-element merging (`FindDuplicateOnDisk`/`AddDuplicateOnDisk`
//!   always miss with the single-heaptid layout, see `types.rs`);
//! * the value is the encoded vector (`dim × elem_bytes`), so tuple sizes are
//!   fixed per index;
//! * scratch buffers are caller-owned per insert instead of a reset context.

use pgrx::pg_sys;
use pgrx::*;

use crate::access_method::distance::{preprocess_cosine, DistanceType};
use crate::access_method::hnswsq::options::Hnsw2Options;
use crate::access_method::hnswsq::types::*;
use crate::access_method::hnswsq::utils::*;
use crate::access_method::pg_vector::PgVectorInternal;
use crate::util::ports::{PageGetItem, PageGetItemId, PageGetMaxOffsetNumber};

/// `GetInsertPage` (hnswinsert.c): the append hint from the metapage.
pub unsafe fn get_insert_page(index: pg_sys::Relation) -> pg_sys::BlockNumber {
    let buf = pg_sys::ReadBuffer(index, METAPAGE_BLKNO);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = pg_sys::BufferGetPage(buf);
    let metap = page_get_meta(page);
    let insert_page = (*metap).insert_page;
    pg_sys::UnlockReleaseBuffer(buf);
    insert_page
}

/// The result of [`free_offset`] (pgvector `HnswFreeOffset`).
struct FreeOffsets {
    nbuf: pg_sys::Buffer,
    free_offno: pg_sys::OffsetNumber,
    free_neighbor_offno: pg_sys::OffsetNumber,
    tuple_version: u8,
}

/// `HnswFreeOffset` (hnswinsert.c): find a deleted element on `page` whose
/// element and neighbor slots fit the new tuples.  The neighbor buffer is
/// returned pinned and exclusively locked (or `buf` itself when the neighbor
/// tuple lives on the same page).
///
/// # Safety
/// `buf`/`page` are pinned and exclusively locked by the caller; the returned
/// `nbuf` (when different) stays locked for the caller to register or unlock.
unsafe fn free_offset(
    index: pg_sys::Relation,
    buf: pg_sys::Buffer,
    page: pg_sys::Page,
    etup_size: usize,
    ntup_size: usize,
    new_insert_page: &mut pg_sys::BlockNumber,
) -> Option<FreeOffsets> {
    let maxoffno = PageGetMaxOffsetNumber(page);
    let mut offno = pg_sys::FirstOffsetNumber;

    while offno <= maxoffno as pg_sys::OffsetNumber {
        let eitemid = PageGetItemId(page, offno);
        let etup = PageGetItem(page, eitemid).cast::<ElementTupleData>();

        // Skip neighbor tuples
        if (*etup).type_ != ELEMENT_TUPLE_TYPE {
            offno += 1;
            continue;
        }

        if (*etup).deleted != 0 {
            let element_page = pg_sys::BufferGetBlockNumber(buf);
            let neighbor_page = ip_block(&(*etup).neighbortid);
            let neighbor_offno =
                ip_offset(&(*etup).neighbortid);

            if *new_insert_page == pg_sys::InvalidBlockNumber {
                *new_insert_page = element_page;
            }

            let nbuf: pg_sys::Buffer;
            let npage: pg_sys::Page;
            if neighbor_page == element_page {
                nbuf = buf;
                npage = page;
            } else {
                nbuf = pg_sys::ReadBuffer(index, neighbor_page);
                pg_sys::LockBuffer(nbuf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
                // Skip WAL for now
                npage = pg_sys::BufferGetPage(nbuf);
            }

            let nitemid = PageGetItemId(npage, neighbor_offno);

            // Calculate free space individually since tuples are overwritten
            // individually (in separate calls to PageIndexTupleOverwrite)
            let mut page_free = (*eitemid).lp_len() as usize + pg_sys::PageGetFreeSpace(page);
            let mut npage_free = (*nitemid).lp_len() as usize;
            if neighbor_page != element_page {
                npage_free += pg_sys::PageGetFreeSpace(npage);
            } else if page_free >= etup_size {
                npage_free += page_free - etup_size;
            }

            // Check for space
            if page_free >= etup_size && npage_free >= ntup_size {
                return Some(FreeOffsets {
                    nbuf,
                    free_offno: offno,
                    free_neighbor_offno: neighbor_offno,
                    tuple_version: (*etup).version,
                });
            } else if nbuf != buf {
                pg_sys::UnlockReleaseBuffer(nbuf);
            }
        }

        offno += 1;
    }

    None
}

/// `HnswInsertAppendPage` (hnswinsert.c): add a new page after `page` (whose
/// buffer the caller holds exclusively) and link it.
///
/// # Safety
/// The caller holds `buf`/`page` exclusively; `state` is their GenericXLog
/// state (or null when building).
unsafe fn insert_append_page(
    index: pg_sys::Relation,
    nbuf: &mut pg_sys::Buffer,
    npage: &mut pg_sys::Page,
    state: *mut pg_sys::GenericXLogState,
    page: pg_sys::Page,
    building: bool,
) {
    // Add a new page
    pg_sys::LockRelationForExtension(index, pg_sys::ExclusiveLock as pg_sys::LOCKMODE);
    *nbuf = new_buffer(index);
    pg_sys::UnlockRelationForExtension(index, pg_sys::ExclusiveLock as pg_sys::LOCKMODE);

    // Init new page
    if building {
        *npage = pg_sys::BufferGetPage(*nbuf);
    } else {
        *npage = pg_sys::GenericXLogRegisterBuffer(state, *nbuf, pg_sys::GENERIC_XLOG_FULL_IMAGE as i32);
    }

    init_page(*nbuf, *npage);

    // Update previous buffer
    (*page_opaque(page)).nextblkno = pg_sys::BufferGetBlockNumber(*nbuf);
}

/// `AddElementOnDisk` (hnswinsert.c): the page-packing loop.
#[allow(clippy::too_many_arguments)]
pub unsafe fn add_element_on_disk(
    index: pg_sys::Relation,
    support: &Support,
    e: *mut Element,
    m: usize,
    insert_page: pg_sys::BlockNumber,
    updated_insert_page: &mut pg_sys::BlockNumber,
    building: bool,
) {
    let vec_bytes = support.codec.vector_bytes();
    let etup_size = element_tuple_size(vec_bytes);
    let ntup_size = neighbor_tuple_size((*e).level as usize, m);
    let combined_size = etup_size + ntup_size + std::mem::size_of::<pg_sys::ItemIdData>();
    let max_size = max_page_item_size();
    let min_combined_size =
        etup_size + neighbor_tuple_size(0, m) + std::mem::size_of::<pg_sys::ItemIdData>();

    // Prepare element tuple
    let mut etup = vec![0u8; etup_size];
    set_element_tuple(
        std::ptr::null_mut(),
        etup.as_mut_ptr().cast::<ElementTupleData>(),
        e,
        support.precision as u8,
        vec_bytes,
    );

    // Prepare neighbor tuple
    let mut ntup = vec![0u8; ntup_size];
    set_neighbor_tuple(
        std::ptr::null_mut(),
        ntup.as_mut_ptr().cast::<NeighborTupleData>(),
        e,
        m,
    );

    let mut current_page = insert_page;
    let mut free_offno = pg_sys::InvalidOffsetNumber;
    let mut free_neighbor_offno = pg_sys::InvalidOffsetNumber;
    let mut new_insert_page = pg_sys::InvalidBlockNumber;
    let mut tuple_version: u8 = 0;

    let mut buf: pg_sys::Buffer = 0;
    let mut page: pg_sys::Page = std::ptr::null_mut();
    let mut nbuf: pg_sys::Buffer = 0;
    let mut npage: pg_sys::Page = std::ptr::null_mut();
    let mut state: *mut pg_sys::GenericXLogState = std::ptr::null_mut();

    // Find a page (or two if needed) to insert the tuples
    loop {
        buf = pg_sys::ReadBuffer(index, current_page);
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);

        if building {
            state = std::ptr::null_mut();
            page = pg_sys::BufferGetPage(buf);
        } else {
            state = pg_sys::GenericXLogStart(index);
            page = pg_sys::GenericXLogRegisterBuffer(state, buf, 0);
        }

        // Keep track of first page where element at level 0 can fit
        if new_insert_page == pg_sys::InvalidBlockNumber
            && pg_sys::PageGetFreeSpace(page) >= min_combined_size
        {
            new_insert_page = current_page;
        }

        // First, try the fastest path
        // Space for both tuples on the current page
        // This can split existing tuples in rare cases
        if pg_sys::PageGetFreeSpace(page) >= combined_size {
            nbuf = buf;
            npage = page;
            break;
        }

        // Next, try space from a deleted element
        if let Some(f) = free_offset(index, buf, page, etup_size, ntup_size, &mut new_insert_page)
        {
            if f.nbuf != buf {
                if building {
                    npage = pg_sys::BufferGetPage(f.nbuf);
                } else {
                    npage = pg_sys::GenericXLogRegisterBuffer(state, f.nbuf, 0);
                }
            } else {
                npage = page;
            }
            nbuf = f.nbuf;
            free_offno = f.free_offno;
            free_neighbor_offno = f.free_neighbor_offno;
            tuple_version = f.tuple_version;

            // Set tuple version
            (*etup.as_mut_ptr().cast::<ElementTupleData>()).version = tuple_version;
            (*ntup.as_mut_ptr().cast::<NeighborTupleData>()).version = tuple_version;

            break;
        }

        // Finally, try space for element only if last page
        // Skip if both tuples can fit on the same page
        if combined_size > max_size
            && pg_sys::PageGetFreeSpace(page) >= etup_size
            && (*page_opaque(page)).nextblkno == pg_sys::InvalidBlockNumber
        {
            insert_append_page(index, &mut nbuf, &mut npage, state, page, building);
            break;
        }

        current_page = (*page_opaque(page)).nextblkno;

        if current_page != pg_sys::InvalidBlockNumber {
            // Move to next page
            if !building {
                pg_sys::GenericXLogAbort(state);
            }
            pg_sys::UnlockReleaseBuffer(buf);
        } else {
            // Append a new page after the current one
            let mut newbuf: pg_sys::Buffer = 0;
            let mut newpage: pg_sys::Page = std::ptr::null_mut();
            insert_append_page(index, &mut newbuf, &mut newpage, state, page, building);

            // Commit
            if building {
                pg_sys::MarkBufferDirty(buf);
            } else {
                pg_sys::GenericXLogFinish(state);
            }

            // Unlock previous buffer
            pg_sys::UnlockReleaseBuffer(buf);

            // Prepare new buffer
            buf = newbuf;
            if building {
                state = std::ptr::null_mut();
                page = pg_sys::BufferGetPage(buf);
            } else {
                state = pg_sys::GenericXLogStart(index);
                page = pg_sys::GenericXLogRegisterBuffer(state, buf, 0);
            }

            // Create new page for neighbors if needed
            if pg_sys::PageGetFreeSpace(page) < combined_size {
                insert_append_page(index, &mut nbuf, &mut npage, state, page, building);
            } else {
                nbuf = buf;
                npage = page;
            }

            break;
        }
    }

    (*e).blkno = pg_sys::BufferGetBlockNumber(buf);
    (*e).neighbor_page = pg_sys::BufferGetBlockNumber(nbuf);

    // Added tuple to new page if newInsertPage is not set
    // So can set to neighbor page instead of element page
    if new_insert_page == pg_sys::InvalidBlockNumber {
        new_insert_page = (*e).neighbor_page;
    }

    if free_offno != pg_sys::InvalidOffsetNumber {
        (*e).offno = free_offno;
        (*e).neighbor_offno = free_neighbor_offno;
    } else {
        (*e).offno = (PageGetMaxOffsetNumber(page) + 1) as pg_sys::OffsetNumber;
        if nbuf == buf {
            (*e).neighbor_offno = (*e).offno + 1;
        } else {
            (*e).neighbor_offno = pg_sys::FirstOffsetNumber;
        }
    }

    pgrx::itemptr::item_pointer_set_all(
        &mut (*etup.as_mut_ptr().cast::<ElementTupleData>()).neighbortid,
        (*e).neighbor_page,
        (*e).neighbor_offno,
    );

    // Add element and neighbors
    if free_offno != pg_sys::InvalidOffsetNumber {
        if !pg_sys::PageIndexTupleOverwrite(
            page,
            (*e).offno,
            etup.as_mut_ptr().cast(),
            etup_size,
        ) {
            error!("hnswsq: failed to add index item");
        }

        if !pg_sys::PageIndexTupleOverwrite(
            npage,
            (*e).neighbor_offno,
            ntup.as_mut_ptr().cast(),
            ntup_size,
        ) {
            error!("hnswsq: failed to add index item");
        }
    } else {
        if pg_sys::PageAddItemExtended(
            page,
            etup.as_mut_ptr().cast(),
            etup_size,
            pg_sys::InvalidOffsetNumber,
            0,
        ) != (*e).offno
        {
            error!("hnswsq: failed to add index item");
        }

        if pg_sys::PageAddItemExtended(
            npage,
            ntup.as_mut_ptr().cast(),
            ntup_size,
            pg_sys::InvalidOffsetNumber,
            0,
        ) != (*e).neighbor_offno
        {
            error!("hnswsq: failed to add index item");
        }
    }

    // Commit
    if building {
        pg_sys::MarkBufferDirty(buf);
        if nbuf != buf {
            pg_sys::MarkBufferDirty(nbuf);
        }
    } else {
        pg_sys::GenericXLogFinish(state);
    }
    pg_sys::UnlockReleaseBuffer(buf);
    if nbuf != buf {
        pg_sys::UnlockReleaseBuffer(nbuf);
    }

    // Update the insert page
    if new_insert_page != pg_sys::InvalidBlockNumber && new_insert_page != insert_page {
        *updated_insert_page = new_insert_page;
    }
}

// ---------------------------------------------------------------------------
// Backlinks (hnswinsert.c: HnswLoadNeighbors / LoadElementsForInsert /
// GetUpdateIndex / UpdateNeighborOnDisk / HnswUpdateNeighborsOnDisk)
// ---------------------------------------------------------------------------

/// An owned neighbor array + the elements its candidates point into
/// (pgvector `HnswLoadNeighbors` pallocs both in the caller's update context;
/// the Boxes keep the pointers valid for the lifetime of this struct).
pub struct LoadedNeighbors {
    pub array: Vec<u8>,
    pub elements: Vec<Box<Element>>,
}

impl LoadedNeighbors {
    fn header(&mut self) -> *mut NeighborArray {
        self.array.as_mut_ptr().cast()
    }
}

/// `HnswLoadNeighbors` (hnswinsert.c).
pub unsafe fn load_neighbors(
    element: *mut Element,
    index: pg_sys::Relation,
    m: usize,
    lm: usize,
    lc: usize,
) -> LoadedNeighbors {
    let mut loaded = LoadedNeighbors {
        array: vec![0u8; neighbor_array_size(lm)],
        elements: Vec::with_capacity(lm),
    };
    let na = loaded.header();
    (*na).length = 0;
    (*na).closer_set = false;

    let mut tids = vec![pg_sys::ItemPointerData::default(); lm];
    if !load_neighbor_tids(element, &mut tids, index, m, lm, lc) {
        return loaded;
    }

    for tid in tids.iter().take(lm) {
        let blkno = ip_block(tid);
        if blkno == pg_sys::InvalidBlockNumber {
            break;
        }
        let offno = ip_offset(tid);

        let mut e = init_element_from_block(blkno, offno);
        let mut hp = crate::access_method::hnswsq::ptr::HnswPtr {
            ptr: std::ptr::null_mut(),
        };
        crate::access_method::hnswsq::ptr::store(
            std::ptr::null_mut(),
            &mut hp,
            e.as_mut() as *mut Element,
        );
        loaded.elements.push(e);

        let items = neighbor_items(na).add((*na).length as usize);
        *items = Candidate {
            element: hp,
            distance: 0.0,
            closer: false,
        };
        (*na).length += 1;
    }

    loaded
}

/// `LoadElementsForInsert` (hnswinsert.c): materialize each neighbor's value
/// and distance; stop at the first element being deleted (returning its
/// index).
unsafe fn load_elements_for_insert(
    loaded: &mut LoadedNeighbors,
    q: &[f32],
    index: pg_sys::Relation,
    support: &Support,
) -> i32 {
    let na = loaded.header();
    for i in 0..(*na).length as usize {
        let hc = &mut *neighbor_items(na).add(i);
        let element =
            crate::access_method::hnswsq::ptr::access::<Element>(std::ptr::null_mut(), hc.element);
        let mut distance = 0.0f32;
        load_element(
            element,
            Some(&mut distance),
            Some(q),
            index,
            support,
            true,
            None,
        );
        hc.distance = distance;

        // Prune element if being deleted
        if (*element).heaptid_set == 0 {
            return i as i32;
        }
    }
    -1
}

/// `GetUpdateIndex` (hnswinsert.c): where the new element goes in `element`'s
/// list at layer `lc` — -2 for "append to a free slot", -1 for "not selected",
/// or the replacement index.
#[allow(clippy::too_many_arguments)]
unsafe fn get_update_index(
    element: *mut Element,
    new_element: *mut Element,
    distance: f32,
    m: usize,
    lm: usize,
    lc: usize,
    index: pg_sys::Relation,
    support: &Support,
    vec_bytes: usize,
    pair_scratch: &mut Vec<f32>,
    decode: &mut Vec<f32>,
) -> i32 {
    let mut idx: i32;

    // Get latest neighbors since they may have changed. Do not lock yet since
    // selecting neighbors can take time. Could use optimistic locking to
    // retry if another update occurs before getting exclusive lock.
    let mut loaded = load_neighbors(element, index, m, lm, lc);

    if (*loaded.header()).length < lm as u32 {
        idx = -2;
    } else {
        // q = the target element's value (materialized during the search)
        let q_bytes = get_value(std::ptr::null_mut(), element, vec_bytes);
        decode.clear();
        decode.resize(support.codec.dim(), 0.0);
        support.codec.decode_into(q_bytes, decode.as_mut_slice());

        idx = load_elements_for_insert(&mut loaded, decode.as_slice(), index, support);

        if idx == -1 {
            update_connection(
                std::ptr::null_mut(),
                loaded.header(),
                new_element,
                distance,
                lm,
                Some(&mut idx),
                Some(index),
                support,
                pair_scratch,
            );
        }
    }

    idx
}

/// `ConnectionExists` (hnswinsert.c).
unsafe fn connection_exists(
    e: *mut Element,
    ntup: *mut NeighborTupleData,
    start_idx: usize,
    lm: usize,
) -> bool {
    let tids = ntup
        .cast::<u8>()
        .add(NEIGHBOR_TUPLE_HEADER_SIZE)
        .cast::<pg_sys::ItemPointerData>();
    for i in 0..lm {
        let indextid = &*tids.add(start_idx + i);
        if ip_block(indextid) == pg_sys::InvalidBlockNumber {
            break;
        }
        if ip_block(indextid) == (*e).blkno
            && ip_offset(indextid) == (*e).offno
        {
            return true;
        }
    }
    false
}

/// `UpdateNeighborOnDisk` (hnswinsert.c).
#[allow(clippy::too_many_arguments)]
unsafe fn update_neighbor_on_disk(
    element: *mut Element,
    new_element: *mut Element,
    mut idx: i32,
    m: usize,
    lm: usize,
    lc: usize,
    index: pg_sys::Relation,
    building: bool,
) {
    // Register page
    let buf = pg_sys::ReadBuffer(index, (*element).neighbor_page);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
    let state = if building {
        std::ptr::null_mut()
    } else {
        pg_sys::GenericXLogStart(index)
    };
    let page = if building {
        pg_sys::BufferGetPage(buf)
    } else {
        pg_sys::GenericXLogRegisterBuffer(state, buf, 0)
    };

    // Get tuple
    let ntup = PageGetItem(page, PageGetItemId(page, (*element).neighbor_offno))
        .cast::<NeighborTupleData>();

    // Calculate index for update
    let start_idx = ((*element).level as usize - lc) * m;

    // Check for existing connection
    if connection_exists(new_element, ntup, start_idx, lm) {
        idx = -1;
    } else if idx == -2 {
        // Find free offset if still exists
        // TODO Retry updating connections if not
        let tids = ntup
            .cast::<u8>()
            .add(NEIGHBOR_TUPLE_HEADER_SIZE)
            .cast::<pg_sys::ItemPointerData>();
        for j in 0..lm {
            if ip_block(&*tids.add(start_idx + j))
                == pg_sys::InvalidBlockNumber
            {
                idx = (start_idx + j) as i32;
                break;
            }
        }
    } else {
        idx += start_idx as i32;
    }

    // Make robust to issues
    if idx >= 0 && (idx as usize) < (*ntup).count as usize {
        let tids = ntup
            .cast::<u8>()
            .add(NEIGHBOR_TUPLE_HEADER_SIZE)
            .cast::<pg_sys::ItemPointerData>();
        let indextid = &mut *tids.add(idx as usize);

        // Update neighbor on the buffer
        pgrx::itemptr::item_pointer_set_all(indextid, (*new_element).blkno, (*new_element).offno);

        // Commit
        if building {
            pg_sys::MarkBufferDirty(buf);
        } else {
            pg_sys::GenericXLogFinish(state);
        }
    } else if !building {
        pg_sys::GenericXLogAbort(state);
    }

    pg_sys::UnlockReleaseBuffer(buf);
}

/// `HnswUpdateNeighborsOnDisk` (hnswinsert.c).
pub unsafe fn update_neighbors_on_disk(
    index: pg_sys::Relation,
    support: &Support,
    e: *mut Element,
    m: usize,
    building: bool,
) {
    let vec_bytes = support.codec.vector_bytes();
    let mut pair_scratch = vec![0.0f32; support.codec.dim()];
    let mut decode = vec![0.0f32; support.codec.dim()];

    for lc in (0..=(*e).level as usize).rev() {
        let lm = get_layer_m(m, lc);
        let neighbors = get_neighbors(std::ptr::null_mut(), e, lc);

        for i in 0..(*neighbors).length as usize {
            let hc = &*neighbor_items(neighbors).add(i);
            let neighbor =
                crate::access_method::hnswsq::ptr::access::<Element>(std::ptr::null_mut(), hc.element);
            let idx = get_update_index(
                neighbor,
                e,
                hc.distance,
                m,
                lm,
                lc,
                index,
                support,
                vec_bytes,
                &mut pair_scratch,
                &mut decode,
            );

            // New element was not selected as a neighbor
            if idx == -1 {
                continue;
            }

            update_neighbor_on_disk(neighbor, e, idx, m, lm, lc, index, building);
        }
    }
}

/// `UpdateGraphOnDisk` (hnswinsert.c), minus the duplicate search.
unsafe fn update_graph_on_disk(
    index: pg_sys::Relation,
    support: &Support,
    element: *mut Element,
    m: usize,
    entry_point: Option<&Element>,
    building: bool,
) {
    let mut new_insert_page = pg_sys::InvalidBlockNumber;

    // Add element
    add_element_on_disk(
        index,
        support,
        element,
        m,
        get_insert_page(index),
        &mut new_insert_page,
        building,
    );

    // Update insert page if needed
    if new_insert_page != pg_sys::InvalidBlockNumber {
        update_meta_page(
            index,
            0,
            None,
            new_insert_page,
            building,
        );
    }

    // Update neighbors
    update_neighbors_on_disk(index, support, element, m, building);

    // Update entry point if needed
    if entry_point.is_none() || (*element).level > entry_point.unwrap().level {
        update_meta_page(
            index,
            UPDATE_ENTRY_GREATER,
            Some(element),
            pg_sys::InvalidBlockNumber,
            building,
        );
    }
}

// ---------------------------------------------------------------------------
// Insert entry points (hnswinsert.c: HnswInsertTupleOnDisk / hnswinsert)
// ---------------------------------------------------------------------------

/// `HnswInsertTupleOnDisk` (hnswinsert.c): insert one already-encoded vector.
/// `building` skips WAL (the build's on-disk phase).  `clamped` records
/// whether the encode saturated any component (the scan needs it for its
/// lower-bound proof).
pub unsafe fn insert_tuple_on_disk(
    index: pg_sys::Relation,
    support: &Support,
    value: &[u8],
    heaptid: &pg_sys::ItemPointerData,
    building: bool,
    clamped: bool,
) -> bool {
    let mut lockmode = pg_sys::ShareLock as pg_sys::LOCKMODE;

    // Get a shared lock. This allows vacuum to ensure no in-flight inserts
    // before repairing graph. Use a page lock so it does not interfere with
    // buffer locks (or reads when vacuuming).
    pg_sys::LockPage(index, UPDATE_LOCK_PAGE, lockmode);

    // Get m and entry point
    let mut m = 0usize;
    let mut entry = None;
    get_meta_page_info(index, Some(&mut m), Some(&mut entry));

    // Create an element.  The level honors hnswsq.build_seed when pinned:
    // entropy-seeded insert levels make the incremental tests' exact-match
    // probes a random sample of the graph (the old engine pinned them for
    // the same reason).
    let level = build_level(
        crate::access_method::hnswsq::options::HNSW_BUILD_SEED.get(),
        get_ml(m),
        get_max_level(m),
        *heaptid,
    );
    let allocator = Allocator::Palloc;
    let element = init_element(std::ptr::null_mut(), heaptid, m, level, &allocator);
    (*element).clamped = clamped as u8;
    let value_ptr = pg_sys::palloc(value.len()).cast::<u8>();
    std::ptr::copy_nonoverlapping(value.as_ptr(), value_ptr, value.len());
    crate::access_method::hnswsq::ptr::store(std::ptr::null_mut(), &mut (*element).value, value_ptr);

    // Prevent concurrent inserts when likely updating entry point
    let mut entry_locked = false;
    if entry.is_none() || (*element).level > entry.as_ref().unwrap().level {
        // Release shared lock
        pg_sys::UnlockPage(index, UPDATE_LOCK_PAGE, lockmode);

        // Get exclusive lock
        lockmode = pg_sys::ExclusiveLock as pg_sys::LOCKMODE;
        pg_sys::LockPage(index, UPDATE_LOCK_PAGE, lockmode);

        // Get latest entry point after lock is acquired
        entry = get_entry_point(index);
        entry_locked = true;
    }

    // Find neighbors for element
    let ef_construction = get_ef_construction(index);
    let mut scratch = SearchScratch::new(m);
    let mut decode = vec![0.0f32; support.codec.dim()];
    let mut pair_scratch = vec![0.0f32; support.codec.dim()];
    let mut visited = Visited::new(ef_construction * m * 2);
    let entry_ptr = entry
        .as_deref_mut()
        .map(|e| e as *mut Element);
    find_element_neighbors(
        std::ptr::null_mut(),
        element,
        entry_ptr,
        Some(index),
        support,
        m,
        ef_construction,
        false,
        &mut scratch,
        &mut decode,
        &mut pair_scratch,
        &mut visited,
    );

    // Update graph on disk
    update_graph_on_disk(index, support, element, m, entry.as_deref(), building);

    // Release lock
    pg_sys::UnlockPage(index, UPDATE_LOCK_PAGE, lockmode);

    let _ = entry_locked;
    true
}

/// `hnswinsert` (hnswinsert.c): the AM insert entry point.
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

    // The insert memory context (pgvector's insertCtx): all per-insert
    // allocations die here.
    let mut insert_ctx = PgMemoryContexts::new("hnswsq insert temporary context");
    insert_ctx.switch_to(|_| {
        let support = init_support(index);

        // Extract the vector (detoast-copy pattern shared with the old paths).
        let datum = *values;
        let detoasted = pg_sys::pg_detoast_datum_copy(datum.cast_mut_ptr());
        let pg_vec = detoasted.cast::<PgVectorInternal>();
        let mut vec = (*pg_vec).to_slice().to_vec();
        pg_sys::pfree(detoasted.cast());
        if support.dist_type == DistanceType::Cosine {
            preprocess_cosine(&mut vec);
        }

        let mut encoded = Vec::with_capacity(support.codec.vector_bytes());
        let clamped = support.codec.encode_into(&vec, &mut encoded);

        insert_tuple_on_disk(index, &support, &encoded, &*heap_tid, false, clamped);
    });
    drop(insert_ctx);

    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_exists() {
        unsafe {
            let mut storage = vec![0u8; NEIGHBOR_TUPLE_HEADER_SIZE + 3 * 6];
            let ntup = storage.as_mut_ptr().cast::<NeighborTupleData>();
            let tids = ntup
                .cast::<u8>()
                .add(NEIGHBOR_TUPLE_HEADER_SIZE)
                .cast::<pg_sys::ItemPointerData>();
            pgrx::itemptr::item_pointer_set_all(&mut *tids, 10, 1);
            pgrx::itemptr::item_pointer_set_all(&mut *tids.add(1), 11, 2);
            pgrx::itemptr::item_pointer_set_all(
                &mut *tids.add(2),
                pg_sys::InvalidBlockNumber,
                pg_sys::InvalidOffsetNumber,
            );

            let mut e = init_element_from_block(11, 2);
            assert!(connection_exists(&mut *e, ntup, 0, 3));

            let mut e2 = init_element_from_block(12, 3);
            assert!(!connection_exists(&mut *e2, ntup, 0, 3));
        }
    }
}
