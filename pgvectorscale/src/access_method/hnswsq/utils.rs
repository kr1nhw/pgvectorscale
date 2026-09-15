//! hnswsq algorithm core — the Rust translation of pgvector's `hnswutils.c`.
//!
//! Function-for-function correspondence with the vendored reference
//! (`.design/reference/pgvector/hnswutils.c`): search layer (Algorithm 2),
//! neighbor selection (Algorithm 4, including the `closer` caching), the
//! connection update, element loading (in-memory and on-disk), entry-point
//! handling, and the metapage.  Deliberate divergences, and only these:
//!
//! * candidate heaps are Rust `BinaryHeap`s with a deterministic tie-break
//!   (see `types.rs`); pgvector's pairing heaps order equal distances
//!   arbitrarily — the recall gates verify the difference is unobservable;
//! * distances are computed by the `Codec` over the encoded vectors instead
//!   of by the operator function over datums (the storage-layout adaptation);
//! * scratch buffers (unvisited/tids/local-neighborhood/visited/decode/element
//!   store) are caller-owned and reused instead of palloc'd per call in a
//!   reset context;
//! * `relptr` arithmetic lives in `ptr.rs`.
//!
//! The single-content-lock rule of the old engine is inherited: this module
//! takes at most one buffer content lock at a time and never holds it while
//! acquiring another.

use pgrx::pg_sys;
use pgrx::*;

use crate::access_method::distance::{self as kernels, DistanceType};
use crate::access_method::hnswsq::quantize::{Codec, HnswPrecision, Sq8Calibration, Sq8QueryState};
use crate::access_method::hnswsq::options::Hnsw2Options;
use crate::access_method::hnswsq::ptr::HnswPtr;
use crate::access_method::hnswsq::types::*;
use crate::util::ports::{PageGetContents, PageGetItem, PageGetItemId, PageGetMaxOffsetNumber};

// ---------------------------------------------------------------------------
// Parameter helpers (hnswutils.c: HnswGetM / HnswGetEfConstruction / layers)
// ---------------------------------------------------------------------------

/// `HnswGetLayerM`: layer 0 gets 2m connections, upper layers m.
#[inline]
pub fn get_layer_m(m: usize, lc: usize) -> usize {
    if lc == 0 {
        m * 2
    } else {
        m
    }
}

/// `HnswGetMl`: the optimal level-scale from the paper.
#[inline]
pub fn get_ml(m: usize) -> f64 {
    1.0 / (m as f64).ln()
}

/// `HnswGetMaxLevel`: the largest level whose neighbor tuple still fits a
/// page, clamped to what a u8 level can express.
pub fn get_max_level(m: usize) -> usize {
    let max_size = max_page_item_size();
    let per_page =
        (max_size - NEIGHBOR_TUPLE_HEADER_SIZE) / std::mem::size_of::<pg_sys::ItemPointerData>();
    ((per_page / m).saturating_sub(2)).min(63)
}

/// The m reloption of an open index (pgvector `HnswGetM`).
pub unsafe fn get_m(index: pg_sys::Relation) -> usize {
    let opts = Hnsw2Options::from_relation(&PgRelation::from_pg(index));
    opts.m as usize
}

/// The ef_construction reloption of an open index.
pub unsafe fn get_ef_construction(index: pg_sys::Relation) -> usize {
    let opts = Hnsw2Options::from_relation(&PgRelation::from_pg(index));
    opts.ef_construction as usize
}

/// The storage-layout reloption of an open index.
pub unsafe fn get_precision(index: pg_sys::Relation) -> HnswPrecision {
    let opts = Hnsw2Options::from_relation(&PgRelation::from_pg(index));
    opts.get_precision()
}

/// The sample-size reloption (SQ8 calibration).  0 means "auto" (the
/// reloption default): the reservoir uses `DEFAULT_SAMPLE_SIZE`.
pub unsafe fn get_sample_size(index: pg_sys::Relation) -> usize {
    let opts = Hnsw2Options::from_relation(&PgRelation::from_pg(index));
    let s = opts.sample_size as usize;
    if s == 0 {
        crate::access_method::hnswsq::options::DEFAULT_SAMPLE_SIZE
    } else {
        s
    }
}

// ---------------------------------------------------------------------------
// Support (hnswutils.c: HnswInitSupport + the type support we replace)
// ---------------------------------------------------------------------------

/// Resolve the index's distance type from its opclass support proc 1 — the
/// same `distance_type_*` convention the old engine and IVF use.
pub unsafe fn resolve_distance_type(index: pg_sys::Relation) -> DistanceType {
    let fmgr_info = pg_sys::index_getprocinfo(index, 1, 1);
    if fmgr_info.is_null() {
        error!("hnswsq: no distance type function found for index");
    }
    let result = pg_sys::FunctionCall0Coll(fmgr_info, pg_sys::InvalidOid).value() as u16;
    DistanceType::from_u16(result)
}

/// Build the [`Support`] for an already-initialized index (insert/scan/vacuum):
/// distance type from the opclass, precision/dimensions/calibration from the
/// metapage.
pub unsafe fn init_support(index: pg_sys::Relation) -> Support {
    let dist_type = resolve_distance_type(index);
    let (precision, dimensions, calibration) = meta_page_layout(index);
    let codec = if precision == HnswPrecision::Sq8 {
        let calib = Sq8Calibration::load(&PgRelation::from_pg(index), calibration);
        Codec::new_sq8(&calib)
    } else {
        Codec::new(precision, dimensions)
    };
    Support {
        dist_type,
        precision,
        codec,
    }
}

// ---------------------------------------------------------------------------
// Buffers and pages (hnswutils.c: HnswNewBuffer / HnswInitPage)
// ---------------------------------------------------------------------------

/// `HnswNewBuffer`: extend the relation and return the new buffer, locked
/// exclusively (the caller holds the relation extension lock).
pub unsafe fn new_buffer(index: pg_sys::Relation) -> pg_sys::Buffer {
    let buf = pg_sys::ReadBufferExtended(
        index,
        pg_sys::ForkNumber::MAIN_FORKNUM,
        pg_sys::InvalidBlockNumber,
        pg_sys::ReadBufferMode::RBM_NORMAL,
        std::ptr::null_mut(),
    );
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
    buf
}

/// `HnswInitPage`: initialize a page's header and special area.
pub unsafe fn init_page(buf: pg_sys::Buffer, page: pg_sys::Page) {
    pg_sys::PageInit(
        page,
        pg_sys::BufferGetPageSize(buf),
        std::mem::size_of::<PageOpaqueData>(),
    );
    let opaque = page_opaque(page);
    (*opaque).nextblkno = pg_sys::InvalidBlockNumber;
    (*opaque).page_id = HNSW_PAGE_ID;
}

/// `HnswPageGetOpaque`.
#[inline]
pub unsafe fn page_opaque(page: pg_sys::Page) -> *mut PageOpaqueData {
    crate::util::ports::PageGetSpecialPointer(page).cast()
}

// ---------------------------------------------------------------------------
// Allocation (hnswutils.c: HnswAlloc + hnswbuild.c allocators)
// ---------------------------------------------------------------------------

/// The two allocation regimes of the build, matching pgvector's
/// `HnswAllocator` function pointer: backend memory (palloc from the graph
/// context, absolute pointers) or the shared area (bump allocation, relptrs).
///
/// `memory_used` is maintained in the graph: the shared variant bumps it
/// (callers hold `allocator_lock`), the private variant adds each
/// allocation's MAXALIGN'd size (single-threaded).
pub enum Allocator {
    Private {
        ctx: pg_sys::MemoryContext,
        graph: *mut Graph,
    },
    Shared {
        base: *mut u8,
        graph: *mut Graph,
    },
    /// Plain palloc (the transactional insert path; freed by the caller's
    /// memory context), pgvector's `allocator == NULL` case.
    Palloc,
}

impl Allocator {
    /// Allocate `size` bytes (MAXALIGN'd) under the allocator's regime.
    ///
    /// # Safety
    /// For [`Allocator::Shared`] the caller must hold the graph's
    /// `allocator_lock` exclusively, and `base` must be the shared area the
    /// graph's relptrs are relative to.
    pub unsafe fn alloc(&self, size: usize) -> *mut u8 {
        match self {
            Allocator::Private { ctx, graph } => {
                let aligned = pg_sys::MAXALIGN(size);
                let p = pg_sys::MemoryContextAlloc(*ctx, aligned);
                (**graph).memory_used += aligned;
                p.cast()
            }
            Allocator::Shared { base, graph } => {
                let aligned = pg_sys::MAXALIGN(size);
                if aligned > 1024 * 1024 {
                    error!("hnswsq allocation too large");
                }
                let new_used = (**graph).memory_used + aligned;
                if new_used > (**graph).memory_total {
                    error!("hnswsq allocator out of memory");
                }
                let chunk = (*base).add((**graph).memory_used);
                (**graph).memory_used = new_used;
                chunk
            }
            Allocator::Palloc => pg_sys::palloc(pg_sys::MAXALIGN(size)).cast(),
        }
    }
}

/// `HnswInitNeighborArray`: allocate a neighbor array of `lm` slots.
pub unsafe fn init_neighbor_array(lm: usize, allocator: &Allocator) -> *mut NeighborArray {
    let a = allocator.alloc(neighbor_array_size(lm)).cast::<NeighborArray>();
    (*a).length = 0;
    (*a).closer_set = false;
    a
}

/// `HnswInitNeighbors`: allocate the per-layer neighbor array pointers.
pub unsafe fn init_neighbors(base: *mut u8, element: *mut Element, m: usize, allocator: &Allocator) {
    let level = (*element).level as usize;
    let neighbor_list = allocator
        .alloc((level + 1) * std::mem::size_of::<HnswPtr>())
        .cast::<HnswPtr>();
    crate::access_method::hnswsq::ptr::store(base, &mut (*element).neighbors, neighbor_list);
    for lc in 0..=level {
        let na = init_neighbor_array(get_layer_m(m, lc), allocator);
        crate::access_method::hnswsq::ptr::store(base, &mut *neighbor_list.add(lc), na);
    }
}

/// `HnswGetNeighbors`: the layer `lc` neighbor array of an element.
///
/// # Safety
/// `element->level >= lc` must hold.
#[inline]
pub unsafe fn get_neighbors(base: *mut u8, element: *mut Element, lc: usize) -> *mut NeighborArray {
    debug_assert!((*element).level as usize >= lc);
    let neighbor_list =
        crate::access_method::hnswsq::ptr::access::<HnswPtr>(base, (*element).neighbors);
    crate::access_method::hnswsq::ptr::access::<NeighborArray>(base, *neighbor_list.add(lc))
}

/// `HnswGetValue`: the element's encoded vector bytes.
///
/// # Safety
/// `element->value` must have been materialized (or the result is an empty
/// slice).
#[inline]
pub unsafe fn get_value(base: *mut u8, element: *mut Element, vec_bytes: usize) -> &'static [u8] {
    let p = crate::access_method::hnswsq::ptr::access::<u8>(base, (*element).value);
    if p.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(p, vec_bytes)
    }
}

// ---------------------------------------------------------------------------
// Elements (hnswutils.c: HnswInitElement / AddHeapTid / InitElementFromBlock)
// ---------------------------------------------------------------------------

/// `HnswInitElement` — with the level drawn by the caller (the build pins
/// levels per row TID for determinism; inserts draw from entropy).
pub unsafe fn init_element(
    base: *mut u8,
    heaptid: &pg_sys::ItemPointerData,
    m: usize,
    level: usize,
    allocator: &Allocator,
) -> *mut Element {
    let element = allocator.alloc(std::mem::size_of::<Element>()).cast::<Element>();
    let element = &mut *element;
    crate::access_method::hnswsq::ptr::store(base, &mut element.next, std::ptr::null_mut::<u8>());
    element.heaptid_set = 0;
    add_heap_tid(element, heaptid);
    element.level = level as u8;
    element.deleted = 0;
    element.clamped = 0;
    // Start at one to make it easier to find issues (pgvector).
    element.version = 1;
    element.hash = 0;
    init_neighbors(base, element, m, allocator);
    crate::access_method::hnswsq::ptr::store(base, &mut element.value, std::ptr::null_mut::<u8>());
    element
}

/// `HnswAddHeapTid` (single-TID variant).
#[inline]
pub unsafe fn add_heap_tid(element: *mut Element, heaptid: &pg_sys::ItemPointerData) {
    (*element).heaptid = *heaptid;
    (*element).heaptid_set = 1;
}

/// `HnswInitElementFromBlock`: a backend-local element addressing an on-disk
/// tuple; the value is materialized on demand.
pub fn init_element_from_block(
    blkno: pg_sys::BlockNumber,
    offno: pg_sys::OffsetNumber,
) -> Box<Element> {
    // All fields zeroed, like pgvector's palloc + only-some-fields-set (the
    // never-read ones are zero rather than uninitialized; the lock is never
    // initialized because on-disk elements never take it).
    let mut element = Box::new(unsafe { std::mem::zeroed::<Element>() });
    element.blkno = blkno;
    element.offno = offno;
    element.neighbor_offno = pg_sys::InvalidOffsetNumber;
    element.neighbor_page = pg_sys::InvalidBlockNumber;
    element
}

/// Fill a zeroed, caller-owned element in place (`init_element_from_block`
/// without the Box).
pub unsafe fn init_element_at(
    element: *mut Element,
    blkno: pg_sys::BlockNumber,
    offno: pg_sys::OffsetNumber,
) {
    (*element).blkno = blkno;
    (*element).offno = offno;
    (*element).neighbor_offno = pg_sys::InvalidOffsetNumber;
    (*element).neighbor_page = pg_sys::InvalidBlockNumber;
}

// ---------------------------------------------------------------------------
// Metapage (hnswutils.c: HnswGetMetaPageInfo / HnswUpdateMetaPage /
// hnswbuild.c: CreateMetaPage)
// ---------------------------------------------------------------------------

/// `HnswPageGetMeta`: the metapage struct lives at the page contents.
#[inline]
pub unsafe fn page_get_meta(page: pg_sys::Page) -> *mut MetaPageData {
    PageGetContents(page).cast()
}

/// `HnswGetMetaPageInfo` — fetch m and/or the entry point from the metapage.
pub unsafe fn get_meta_page_info(
    index: pg_sys::Relation,
    m: Option<&mut usize>,
    entry: Option<&mut Option<Box<Element>>>,
) {
    let buf = pg_sys::ReadBuffer(index, METAPAGE_BLKNO);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = pg_sys::BufferGetPage(buf);
    let metap = page_get_meta(page);

    if metap.is_null() || (*metap).magic_number != HNSW_MAGIC {
        pg_sys::UnlockReleaseBuffer(buf);
        error!("hnswsq index is not valid (bad magic)");
    }

    if let Some(m) = m {
        *m = (*metap).m as usize;
    }

    if let Some(entry) = entry {
        if (*metap).entry_blkno != pg_sys::InvalidBlockNumber {
            let mut e = init_element_from_block((*metap).entry_blkno, (*metap).entry_offno);
            e.level = (*metap).entry_level as u8;
            *entry = Some(e);
        } else {
            *entry = None;
        }
    }

    pg_sys::UnlockReleaseBuffer(buf);
}


/// Read a TID's block without pgrx's validity assertion (we check invalid
/// TIDs all the time — that IS the check).
#[inline]
pub unsafe fn ip_block(tid: &pg_sys::ItemPointerData) -> pg_sys::BlockNumber {
    pgrx::itemptr::item_pointer_get_block_number_no_check(*tid)
}

/// Read a TID's offset without pgrx's validity assertion.
#[inline]
pub unsafe fn ip_offset(tid: &pg_sys::ItemPointerData) -> pg_sys::OffsetNumber {
    pgrx::itemptr::item_pointer_get_offset_number_no_check(*tid)
}

/// `HnswGetEntryPoint`.
pub unsafe fn get_entry_point(index: pg_sys::Relation) -> Option<Box<Element>> {
    let mut entry = None;
    get_meta_page_info(index, None, Some(&mut entry));
    entry
}

/// Read `(precision, dimensions, calibration pointer)` from the metapage.
pub unsafe fn meta_page_layout(
    index: pg_sys::Relation,
) -> (HnswPrecision, usize, crate::util::ItemPointer) {
    let buf = pg_sys::ReadBuffer(index, METAPAGE_BLKNO);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = pg_sys::BufferGetPage(buf);
    let metap = page_get_meta(page);
    if (*metap).magic_number != HNSW_MAGIC {
        pg_sys::UnlockReleaseBuffer(buf);
        error!("hnswsq index is not valid (bad magic)");
    }
    let precision = HnswPrecision::from_u8((*metap).precision);
    let dimensions = (*metap).dimensions as usize;
    let calibration = crate::util::ItemPointer::new(
        (*metap).calibration_blkno,
        (*metap).calibration_offno,
    );
    pg_sys::UnlockReleaseBuffer(buf);
    (precision, dimensions, calibration)
}

/// `HnswUpdateMetaPage`: RMW the metapage (WAL-logged unless building).
///
/// # Safety
/// The caller must not hold any other buffer content lock.
pub unsafe fn update_meta_page(
    index: pg_sys::Relation,
    update_entry: i32,
    entry_point: Option<*mut Element>,
    insert_page: pg_sys::BlockNumber,
    building: bool,
) {
    let buf = pg_sys::ReadBuffer(index, METAPAGE_BLKNO);
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
    let metap = page_get_meta(page);

    if update_entry != 0 {
        match entry_point {
            None => {
                (*metap).entry_blkno = pg_sys::InvalidBlockNumber;
                (*metap).entry_offno = pg_sys::InvalidOffsetNumber;
                (*metap).entry_level = -1;
            }
            Some(ep) => {
                if (*ep).level as i16 > (*metap).entry_level || update_entry == UPDATE_ENTRY_ALWAYS
                {
                    (*metap).entry_blkno = (*ep).blkno;
                    (*metap).entry_offno = (*ep).offno;
                    (*metap).entry_level = (*ep).level as i16;
                }
            }
        }
    }

    if insert_page != pg_sys::InvalidBlockNumber {
        (*metap).insert_page = insert_page;
    }

    if building {
        pg_sys::MarkBufferDirty(buf);
    } else {
        pg_sys::GenericXLogFinish(state);
    }
    pg_sys::UnlockReleaseBuffer(buf);
}

/// `CreateMetaPage` (hnswbuild.c): write block 0 during a build.
///
/// # Safety
/// Called once per build, before any graph page exists.
pub unsafe fn create_meta_page(
    index: pg_sys::Relation,
    dimensions: usize,
    m: usize,
    ef_construction: usize,
    precision: HnswPrecision,
    calibration: crate::util::ItemPointer,
) {
    let buf = new_buffer(index);
    let page = pg_sys::BufferGetPage(buf);
    init_page(buf, page);

    let metap = page_get_meta(page);
    (*metap).magic_number = HNSW_MAGIC;
    (*metap).version = HNSW_VERSION;
    (*metap).dimensions = dimensions as u32;
    (*metap).m = m as u16;
    (*metap).ef_construction = ef_construction as u16;
    (*metap).precision = precision as u8;
    (*metap).entry_blkno = pg_sys::InvalidBlockNumber;
    (*metap).entry_offno = pg_sys::InvalidOffsetNumber;
    (*metap).entry_level = -1;
    (*metap).insert_page = pg_sys::InvalidBlockNumber;
    (*metap).graph_head = pg_sys::InvalidBlockNumber;
    (*metap).calibration_blkno = calibration.block_number;
    (*metap).calibration_offno = calibration.offset;

    // pd_lower sits right past the metapage struct, as in pgvector.
    let header = page.cast::<pg_sys::PageHeaderData>();
    (*header).pd_lower = (pg_sys::MAXALIGN(std::mem::offset_of!(
        pg_sys::PageHeaderData,
        pd_linp
    )) + std::mem::size_of::<MetaPageData>()) as u16;

    pg_sys::MarkBufferDirty(buf);
    pg_sys::UnlockReleaseBuffer(buf);
}

/// Record the SQ8 calibration chain pointer in the metapage (build mode:
/// no WAL, the whole page range is logged at the end).
///
/// # Safety
/// The caller must not hold any other buffer content lock.
pub unsafe fn set_meta_calibration(index: pg_sys::Relation, ptr: crate::util::ItemPointer) {
    let buf = pg_sys::ReadBuffer(index, METAPAGE_BLKNO);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
    let page = pg_sys::BufferGetPage(buf);
    let metap = page_get_meta(page);
    (*metap).calibration_blkno = ptr.block_number;
    (*metap).calibration_offno = ptr.offset;
    pg_sys::MarkBufferDirty(buf);
    pg_sys::UnlockReleaseBuffer(buf);
}

// ---------------------------------------------------------------------------
// Tuple <-> element (hnswutils.c: HnswSetElementTuple / HnswSetNeighborTuple /
// HnswLoadElementFromTuple)
// ---------------------------------------------------------------------------

/// `HnswSetElementTuple`: fill an element tuple (header + encoded vector)
/// from an in-memory element, except the neighbor pointer.
///
/// # Safety
/// `etup` must have room for `element_tuple_size(vec_bytes)` bytes and the
/// element's value must be materialized.
pub unsafe fn set_element_tuple(
    base: *mut u8,
    etup: *mut ElementTupleData,
    element: *mut Element,
    layout: u8,
    vec_bytes: usize,
) {
    (*etup).type_ = ELEMENT_TUPLE_TYPE;
    (*etup).layout = layout;
    (*etup).level = (*element).level;
    (*etup).deleted = 0;
    (*etup).version = (*element).version;
    (*etup).clamped = (*element).clamped;
    (*etup).heaptid = (*element).heaptid;
    let value = get_value(base, element, vec_bytes);
    let dst = etup.cast::<u8>().add(ELEMENT_TUPLE_VECTOR_OFFSET);
    std::ptr::copy_nonoverlapping(value.as_ptr(), dst, vec_bytes);
}

/// `HnswSetNeighborTuple`: fill a neighbor tuple (all layers, invalid
/// padding) from an in-memory element.
///
/// # Safety
/// `ntup` must have room for `neighbor_tuple_size(level, m)` bytes.
pub unsafe fn set_neighbor_tuple(
    base: *mut u8,
    ntup: *mut NeighborTupleData,
    element: *mut Element,
    m: usize,
) {
    let mut idx = 0usize;
    let level = (*element).level as usize;

    (*ntup).type_ = NEIGHBOR_TUPLE_TYPE;


    let tids = ntup
        .cast::<u8>()
        .add(NEIGHBOR_TUPLE_HEADER_SIZE)
        .cast::<pg_sys::ItemPointerData>();

    for lc in (0..=level).rev() {
        let neighbors = get_neighbors(base, element, lc);
        let lm = get_layer_m(m, lc);
        for i in 0..lm {
            let indextid = &mut *tids.add(idx);
            idx += 1;
            if i < (*neighbors).length as usize {
                let hc = &*neighbor_items(neighbors).add(i);
                let hce = crate::access_method::hnswsq::ptr::access::<Element>(base, hc.element);
                pgrx::itemptr::item_pointer_set_all(indextid, (*hce).blkno, (*hce).offno);
            } else {
                pgrx::itemptr::item_pointer_set_all(
                    indextid,
                    pg_sys::InvalidBlockNumber,
                    pg_sys::InvalidOffsetNumber,
                );
            }
        }
    }

    (*ntup).count = idx as u16;
    (*ntup).version = (*element).version;
}

/// `HnswLoadElementFromTuple`.
///
/// # Safety
/// `etup` must point at a valid element tuple for this index's layout.
pub unsafe fn load_element_from_tuple(
    element: *mut Element,
    etup: *mut ElementTupleData,
    load_heaptid: bool,
    load_vec: bool,
    vec_bytes: usize,
) {
    (*element).level = (*etup).level;
    (*element).deleted = (*etup).deleted;
    (*element).version = (*etup).version;
    (*element).clamped = (*etup).clamped;
    (*element).neighbor_page = ip_block(&(*etup).neighbortid);
    (*element).neighbor_offno =
        ip_offset(&(*etup).neighbortid);
    (*element).heaptid_set = 0;

    if load_heaptid {
        if ip_block(&(*etup).heaptid)
            != pg_sys::InvalidBlockNumber
        {
            (*element).heaptid = (*etup).heaptid;
            (*element).heaptid_set = 1;
        }
    }

    if load_vec {
        // datumCopy equivalent: the encoded vector is copied out so the
        // caller never holds the page lock while using it.  In plain cargo
        // unit tests pgrx FFI calls panic off the main thread, so the copy
        // leaks a Box instead (pg_test runs and production use palloc, which
        // the caller's memory-context reset frees).
        // Plain cargo unit tests run on non-main threads where pgrx FFI
        // calls panic; the pg_test suite runs inside the backend where
        // palloc is correct.  Both compile with cfg(test) during `cargo
        // pgrx test`, so the unit-test path takes the leak-forever Box
        // (test processes are short-lived).
        #[cfg(test)]
        let p = Box::into_raw(vec![0u8; vec_bytes].into_boxed_slice()) as *mut u8;
        #[cfg(not(test))]
        let p = pg_sys::palloc(vec_bytes).cast::<u8>();
        std::ptr::copy_nonoverlapping(
            etup.cast::<u8>().add(ELEMENT_TUPLE_VECTOR_OFFSET),
            p,
            vec_bytes,
        );
        crate::access_method::hnswsq::ptr::store(std::ptr::null_mut(), &mut (*element).value, p);
    }
}

// ---------------------------------------------------------------------------
// Element loading with in-page distance (hnswutils.c: HnswLoadElementImpl)
// ---------------------------------------------------------------------------

/// `HnswLoadElementImpl`: read an element tuple, compute the distance to `q`
/// straight out of the pinned page (no copy), and materialize the element
/// only when it is admitted (`distance == None || max_distance == None ||
/// distance < max_distance`).  A materialized element is either filled in
/// place (`element = Some(ptr)`) or pushed into `store` (`element = None`),
/// which is how the search path avoids allocating for non-admitted
/// candidates.  Returns the materialized element pointer.
///
/// Per-search SQ8 query state: `None` unless the index is SQ8 + L2 and the
/// `hnswsq.sq8_distance` GUC selects one of the integer forms (see
/// `smoke.rs`).  The query is quantized once per search; every candidate
/// distance then uses pure integer arithmetic.
pub fn sq8_query_state(
    support: &Support,
    q: &[f32],
    mode: crate::access_method::hnswsq::options::Sq8DistanceMode,
) -> Option<Sq8QueryState> {
    use crate::access_method::hnswsq::options::Sq8DistanceMode;
    if support.precision != HnswPrecision::Sq8 || support.dist_type != DistanceType::L2 {
        return None;
    }
    match mode {
        Sq8DistanceMode::Scalar => None,
        Sq8DistanceMode::Pairwise => Some(support.codec.sq8_query_state(q, true)),
    }
}

/// Distance from `q` to the encoded `bytes`, using the per-search SQ8 state
/// when one is present (the scalar codec path otherwise).
#[inline]
pub unsafe fn encoded_distance(
    support: &Support,
    q: &[f32],
    bytes: &[u8],
    qstate: Option<&Sq8QueryState>,
) -> f32 {
    match qstate {
        Some(Sq8QueryState::Pairwise(qhat)) => support.codec.distance_l2_sq8_pairwise(qhat, bytes),
        None => support.distance(q, bytes),
    }
}

/// # Safety
/// Exactly one of `element`/`store` must be Some; the preallocated element
/// must be caller-owned.
#[allow(clippy::too_many_arguments)]
pub unsafe fn load_element_impl(
    blkno: pg_sys::BlockNumber,
    offno: pg_sys::OffsetNumber,
    mut distance: Option<&mut f32>,
    q: Option<&[f32]>,
    index: pg_sys::Relation,
    support: &Support,
    load_vec: bool,
    max_distance: Option<f32>,
    qstate: Option<&Sq8QueryState>,
    element: Option<*mut Element>,
    store: Option<&mut ElementArena>,
) -> Option<*mut Element> {
    let vec_bytes = support.codec.vector_bytes();

    // Read vector
    let buf = pg_sys::ReadBuffer(index, blkno);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = pg_sys::BufferGetPage(buf);

    let etup = PageGetItem(page, PageGetItemId(page, offno)).cast::<ElementTupleData>();

    debug_assert_eq!((*etup).type_, ELEMENT_TUPLE_TYPE);

    if (*etup).deleted != 0 {
        pg_sys::UnlockReleaseBuffer(buf);
        error!("cannot load deleted element");
    }

    // Calculate distance
    if let Some(d) = distance.as_deref_mut() {
        *d = match q {
            None => 0.0,
            Some(q) => {
                let bytes = std::slice::from_raw_parts(
                    etup.cast::<u8>().add(ELEMENT_TUPLE_VECTOR_OFFSET),
                    vec_bytes,
                );
                encoded_distance(support, q, bytes, qstate)
            }
        };
    }

    // Load element (only when admitted)
    let admitted = match distance.as_deref() {
        None => true,
        Some(d) => match max_distance {
            None => true,
            Some(m) => *d < m,
        },
    };
    let result = if admitted {
        let eptr = match (element, store) {
            (Some(e), _) => e,
            (None, Some(store)) => {
                let eptr = store.alloc();
                init_element_at(eptr, blkno, offno);
                eptr
            }
            (None, None) => unreachable!("element or store required"),
        };
        (*eptr).blkno = blkno;
        (*eptr).offno = offno;
        load_element_from_tuple(eptr, etup, true, load_vec, vec_bytes);
        Some(eptr)
    } else {
        None
    };

    pg_sys::UnlockReleaseBuffer(buf);
    result
}

/// `HnswLoadElement`: load `element` (addressed by its blkno/offno) and
/// optionally compute its distance to `q`.
pub unsafe fn load_element(
    element: *mut Element,
    distance: Option<&mut f32>,
    q: Option<&[f32]>,
    index: pg_sys::Relation,
    support: &Support,
    load_vec: bool,
    max_distance: Option<f32>,
    qstate: Option<&Sq8QueryState>,
) {
    let _ = load_element_impl(
        (*element).blkno,
        (*element).offno,
        distance,
        q,
        index,
        support,
        load_vec,
        max_distance,
        qstate,
        Some(element),
        None,
    );
}

// ---------------------------------------------------------------------------
// Search candidates (hnswutils.c: HnswInitSearchCandidate / EntryCandidate)
// ---------------------------------------------------------------------------

/// The deterministic tie-break key of a candidate: packed TID, relptr offset,
/// or pointer depending on where the element lives.
#[inline]
pub unsafe fn candidate_key(base: *mut u8, index: Option<pg_sys::Relation>, hp: HnswPtr) -> u64 {
    if index.is_some() {
        let e = crate::access_method::hnswsq::ptr::access::<Element>(base, hp);
        pack_tid(tid_of(&*e))
    } else if !base.is_null() {
        crate::access_method::hnswsq::ptr::offset(hp) as u64
    } else {
        crate::access_method::hnswsq::ptr::pointer(hp) as u64
    }
}

/// The `ItemPointerData` of an element's location (both modes have blkno/
/// offno once inserted or loaded).
#[inline]
pub unsafe fn tid_of(e: &Element) -> pg_sys::ItemPointerData {
    let mut tid = pg_sys::ItemPointerData::default();
    pgrx::itemptr::item_pointer_set_all(&mut tid, e.blkno, e.offno);
    tid
}

/// `HnswInitSearchCandidate`.
#[inline]
pub unsafe fn init_search_candidate(
    base: *mut u8,
    index: Option<pg_sys::Relation>,
    element: *mut Element,
    distance: f32,
) -> SearchCandidate {
    let mut hp = HnswPtr { ptr: std::ptr::null_mut() };
    crate::access_method::hnswsq::ptr::store(base, &mut hp, element);
    SearchCandidate {
        element: hp,
        distance,
        key: candidate_key(base, index, hp),
    }
}

/// `GetElementDistance`: distance from `q` to an in-memory element's value.
#[inline]
pub unsafe fn get_element_distance(
    base: *mut u8,
    element: *mut Element,
    q: &[f32],
    support: &Support,
    qstate: Option<&Sq8QueryState>,
) -> f32 {
    encoded_distance(
        support,
        q,
        get_value(base, element, support.codec.vector_bytes()),
        qstate,
    )
}

/// `HnswEntryCandidate`.
pub unsafe fn entry_candidate(
    base: *mut u8,
    entry_point: *mut Element,
    q: Option<&[f32]>,
    index: Option<pg_sys::Relation>,
    support: &Support,
    load_vec: bool,
    qstate: Option<&Sq8QueryState>,
) -> SearchCandidate {
    let distance = match index {
        None => get_element_distance(base, entry_point, q.unwrap_or(&[]), support, qstate),
        Some(index) => {
            let mut distance = 0.0f32;
            load_element(
                entry_point,
                Some(&mut distance),
                q,
                index,
                support,
                load_vec,
                None,
                qstate,
            );
            distance
        }
    };
    init_search_candidate(base, index, entry_point, distance)
}

// ---------------------------------------------------------------------------
// Visited tracking (hnswutils.c: InitVisited / AddToVisited / CountElement)
// ---------------------------------------------------------------------------

/// pgvector's murmur64 mixing (identical to `murmurhash64` in the reference).
#[inline]
pub fn murmur64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

/// `PrecomputeHash`: the element's hash for the in-memory visited tables.
pub unsafe fn precompute_hash(base: *mut u8, element: *mut Element) {
    if base.is_null() {
        (*element).hash = murmur64(element as u64);
    } else {
        (*element).hash = murmur64((element as *mut u8).offset_from(base) as u64);
    }
}

/// `AddToVisited`: record `hp` in the visited set; returns true when it was
/// already present.  For the in-memory modes the element's precomputed hash
/// is used (pgvector stores it for exactly this).
pub unsafe fn add_to_visited(
    base: *mut u8,
    index: Option<pg_sys::Relation>,
    v: &mut Visited,
    hp: HnswPtr,
) -> bool {
    if index.is_some() {
        let e = crate::access_method::hnswsq::ptr::access::<Element>(base, hp);
        let key = pack_tid(tid_of(&*e));
        v.insert(key)
    } else if !base.is_null() {
        let key = crate::access_method::hnswsq::ptr::offset(hp) as u64;
        let e = crate::access_method::hnswsq::ptr::access::<Element>(base, hp);
        v.insert_key_hash(key, (*e).hash)
    } else {
        let key = crate::access_method::hnswsq::ptr::pointer(hp) as u64;
        let e = crate::access_method::hnswsq::ptr::access::<Element>(base, hp);
        v.insert_key_hash(key, (*e).hash)
    }
}

/// `CountElement`: deleted elements do not count towards `ef` when the caller
/// passes a skip element (vacuum repair); otherwise everything counts.
#[inline]
pub unsafe fn count_element(skip_element: Option<*mut Element>, e: *mut Element) -> bool {
    if skip_element.is_none() {
        return true;
    }
    // Ensure we do not access heaptid_set during an in-memory parallel build
    // without ordering (pgvector's pg_memory_barrier()).
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    (*e).heaptid_set != 0
}

// ---------------------------------------------------------------------------
// Neighbor loading (hnswutils.c: LoadUnvisitedFromMemory / LoadNeighborTids /
// LoadUnvisitedFromDisk)
// ---------------------------------------------------------------------------

/// `HnswLoadUnvisitedFromMemory`: copy the layer's neighborhood out under the
/// element's lock, then collect the not-yet-visited neighbors.
pub unsafe fn load_unvisited_from_memory(
    base: *mut u8,
    element: *mut Element,
    lc: usize,
    v: &mut Visited,
    unvisited: &mut Vec<Unvisited>,
    local: &mut Vec<Candidate>,
) {
    // pgvector sets *unvisitedLength = 0 here: the buffer is per-expansion.
    unvisited.clear();
    let neighborhood = get_neighbors(base, element, lc);

    // Copy the neighborhood to local memory
    pg_sys::LWLockAcquire(std::ptr::addr_of_mut!((*element).lock), pg_sys::LWLockMode::LW_SHARED);
    local.clear();
    for i in 0..(*neighborhood).length as usize {
        local.push(*neighbor_items(neighborhood).add(i));
    }
    pg_sys::LWLockRelease(std::ptr::addr_of_mut!((*element).lock));

    for hc in local.iter() {
        if !add_to_visited(base, None, v, hc.element) {
            unvisited.push(Unvisited::Element(hc.element));
        }
    }
}

/// `HnswLoadNeighborTids`: copy the layer's TID list out of the neighbor
/// tuple.  Returns false when the tuple was deleted or replaced between scan
/// iterations (version/count mismatch).
pub unsafe fn load_neighbor_tids(
    element: *mut Element,
    indextids: &mut [pg_sys::ItemPointerData],
    index: pg_sys::Relation,
    m: usize,
    lm: usize,
    lc: usize,
) -> bool {
    let buf = pg_sys::ReadBuffer(index, (*element).neighbor_page);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = pg_sys::BufferGetPage(buf);

    let ntup = PageGetItem(page, PageGetItemId(page, (*element).neighbor_offno))
        .cast::<NeighborTupleData>();

    // Ensure the neighbor tuple has not been deleted or replaced between
    // index scan iterations
    if (*ntup).version != (*element).version
        || (*ntup).count as usize != ((*element).level as usize + 2) * m
    {
        pg_sys::UnlockReleaseBuffer(buf);
        return false;
    }

    // Copy to minimize lock time
    let start = ((*element).level as usize - lc) * m;
    let src = ntup
        .cast::<u8>()
        .add(NEIGHBOR_TUPLE_HEADER_SIZE)
        .cast::<pg_sys::ItemPointerData>();
    std::ptr::copy_nonoverlapping(src.add(start), indextids.as_mut_ptr(), lm);

    pg_sys::UnlockReleaseBuffer(buf);
    true
}

/// `HnswLoadUnvisitedFromDisk`: filter the layer's TIDs against the visited
/// set, collecting the unseen ones.
pub unsafe fn load_unvisited_from_disk(
    element: *mut Element,
    index: pg_sys::Relation,
    m: usize,
    lm: usize,
    lc: usize,
    v: &mut Visited,
    unvisited: &mut Vec<Unvisited>,
    tids: &mut Vec<pg_sys::ItemPointerData>,
) {
    // pgvector sets *unvisitedLength = 0 here: the buffer is per-expansion.
    unvisited.clear();
    tids.clear();
    tids.resize(lm, pg_sys::ItemPointerData::default());
    if !load_neighbor_tids(element, tids, index, m, lm, lc) {
        return;
    }

    for indextid in tids.iter().take(lm) {
        if ip_block(indextid) == pg_sys::InvalidBlockNumber {
            break;
        }
        let found = v.insert(pack_tid(*indextid));        if !found {
            unvisited.push(Unvisited::Tid(*indextid));
        }
    }
}

// ---------------------------------------------------------------------------
// Algorithm 2 from the paper (hnswutils.c: HnswSearchLayer)
// ---------------------------------------------------------------------------

/// A bump allocator for the elements a search materializes.  The candidates
/// hold raw pointers into it for as long as the caller uses the results, so
/// allocations must be address-stable: fixed-capacity chunks give one
/// allocation per [`ELEMENT_ARENA_CHUNK`] elements instead of one `malloc`
/// per admitted candidate (the pgvector reference pallocs per candidate, but
/// its palloc is cheaper than a general-purpose malloc + free pair).
pub struct ElementArena {
    chunks: Vec<*mut Element>,
    next: usize,
    len: usize,
}

/// Elements per arena chunk (512 × ~128 B ≈ 64 KiB).
const ELEMENT_ARENA_CHUNK: usize = 512;

impl ElementArena {
    pub fn new() -> Self {
        ElementArena {
            chunks: Vec::new(),
            next: ELEMENT_ARENA_CHUNK,
            len: 0,
        }
    }

    /// Allocate one zeroed element with a stable address; the caller fills it
    /// in (an on-disk element never takes its lock).
    pub fn alloc(&mut self) -> *mut Element {
        if self.next == ELEMENT_ARENA_CHUNK {
            let layout = std::alloc::Layout::array::<Element>(ELEMENT_ARENA_CHUNK).unwrap();
            let p = unsafe { std::alloc::alloc_zeroed(layout) } as *mut Element;
            assert!(!p.is_null(), "out of memory allocating element arena chunk");
            self.chunks.push(p);
            self.next = 0;
        }
        let ptr = unsafe { (*self.chunks.last().unwrap()).add(self.next) };
        self.next += 1;
        self.len += 1;
        ptr
    }

    /// Elements currently allocated.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
        self.next = ELEMENT_ARENA_CHUNK;
        self.len = 0;
    }

    /// Rewind for reuse: keep one chunk's worth of arena memory and start
    /// allocating from its beginning again (all previously returned pointers
    /// are dead).  The insert path calls this per insert instead of
    /// reallocating a 64 KiB chunk every row.
    pub fn reset(&mut self) {
        if self.chunks.is_empty() {
            // Never allocated: leave the bump positioned so the next alloc
            // creates the first chunk.
            self.next = ELEMENT_ARENA_CHUNK;
        } else {
            self.chunks.truncate(1);
            self.next = 0;
        }
        self.len = 0;
    }
}

impl Drop for ElementArena {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::array::<Element>(ELEMENT_ARENA_CHUNK).unwrap();
        for chunk in self.chunks.drain(..) {
            unsafe { std::alloc::dealloc(chunk.cast::<u8>(), layout) };
        }
    }
}

/// Scratch space the search layer and its callers reuse across calls
/// (pgvector pallocs the same per call in the caller's reset context).
/// `elements` owns the on-disk elements materialized by the search, keeping
/// the candidates' pointers alive for as long as the caller uses the result
/// (pgvector frees them with the same context reset that frees the result).
pub struct SearchScratch {
    pub unvisited: Vec<Unvisited>,
    pub tids: Vec<pg_sys::ItemPointerData>,
    pub local: Vec<Candidate>,
    pub elements: ElementArena,
}

impl SearchScratch {
    pub fn new(m: usize) -> Self {
        SearchScratch {
            unvisited: Vec::with_capacity(unvisited_capacity(m)),
            tids: Vec::with_capacity(unvisited_capacity(m)),
            local: Vec::with_capacity(unvisited_capacity(m)),
            elements: ElementArena::new(),
        }
    }
}

/// `HnswSearchLayer` (Algorithm 2): search one layer, returning the result
/// candidates in furthest-first order (pgvector drains W).  `v` is cleared
/// when `init_visited` (pgvector allocates a fresh set per initial search).
///
/// # Safety
/// All pointer arguments follow the base convention; the scratch buffers must
/// not alias anything reachable from the graph.
#[allow(clippy::too_many_arguments)]
pub unsafe fn search_layer(
    base: *mut u8,
    index: Option<pg_sys::Relation>,
    support: &Support,
    m: usize,
    q: Option<&[f32]>,
    ep: &[SearchCandidate],
    ef: usize,
    lc: usize,
    inserting: bool,
    skip_element: Option<*mut Element>,
    v: &mut Visited,
    mut discarded: Option<&mut CandidateHeap>,
    init_visited: bool,
    mut tuples: Option<&mut i64>,
    scratch: &mut SearchScratch,
    qstate: Option<&Sq8QueryState>,
) -> Vec<SearchCandidate> {
    let mut w: Vec<SearchCandidate> = Vec::new();
    let mut c: CandidateHeap = CandidateHeap::with_capacity(ef + 1);
    let mut furthest: FurthestHeap = FurthestHeap::with_capacity(ef + 1);
    let mut wlen = 0usize;
    #[cfg(any(test, feature = "pg_test"))]
    let mut n_probes: u64 = 0;
    #[cfg(any(test, feature = "pg_test"))]
    let mut n_expansions: u64 = 0;
    let lm = get_layer_m(m, lc);

    if init_visited {
        v.clear();
    }

    // Add entry points to v, C, and W
    for sc in ep {
        if init_visited {
            let _ = add_to_visited(base, index, v, sc.element);
            // OK to count elements instead of tuples
            if let Some(tuples) = tuples.as_deref_mut() {
                *tuples += 1;
            }
        }

        c.push(NearestItem(*sc));
        furthest.push(FurthestItem(*sc));

        // Do not count elements being deleted towards ef when vacuuming
        let e = crate::access_method::hnswsq::ptr::access::<Element>(base, sc.element);
        if count_element(skip_element, e) {
            wlen += 1;
        }
    }

    scratch.unvisited.clear();

    while let Some(c_item) = c.pop() {
        let c_sc = c_item.0;
        // W is never empty while C is not: every C entry was also pushed to
        // W, W only shrinks past ef entries (ef >= 1), and wlen is never
        // decremented on those pops.
        let f_sc = furthest.peek().expect("W never empty while C is not").0;

        if c_sc.distance > f_sc.distance {
            break;
        }

        let c_element = crate::access_method::hnswsq::ptr::access::<Element>(base, c_sc.element);
        #[cfg(any(test, feature = "pg_test"))]
        {
            n_expansions += 1;
        }

        match index {
            None => {
                load_unvisited_from_memory(
                    base,
                    c_element,
                    lc,
                    v,
                    &mut scratch.unvisited,
                    &mut scratch.local,
                );
            }
            Some(index) => {
                load_unvisited_from_disk(
                    c_element,
                    index,
                    m,
                    lm,
                    lc,
                    v,
                    &mut scratch.unvisited,
                    &mut scratch.tids,
                );
            }
        }

        // OK to count elements instead of tuples
        if let Some(tuples) = tuples.as_deref_mut() {
            *tuples += scratch.unvisited.len() as i64;
        }

        for u in scratch.unvisited.iter().copied() {
            let f = furthest.peek().expect("W never empty").0;
            let always_add = wlen < ef;

            let e_distance: f32;
            let e_element: *mut Element;
            match (u, index) {
                (Unvisited::Element(hp), None) => {
                    e_element = crate::access_method::hnswsq::ptr::access::<Element>(base, hp);
                    e_distance =
                        get_element_distance(base, e_element, q.unwrap_or(&[]), support, qstate);
                    #[cfg(any(test, feature = "pg_test"))]
                    {
                        n_probes += 1;
                    }
                }
                (Unvisited::Tid(tid), Some(index)) => {
                    let blkno = ip_block(&tid);
                    let offno = ip_offset(&tid);

                    // Avoid any allocations if not adding (pgvector passes
                    // the furthest distance as maxDistance unless we always
                    // add or track discarded).
                    let max_d = if always_add || discarded.is_some() {
                        None
                    } else {
                        Some(f.distance)
                    };
                    let mut dist = 0.0f32;
                    let Some(eptr) = load_element_impl(
                        blkno,
                        offno,
                        Some(&mut dist),
                        q,
                        index,
                        support,
                        inserting,
                        max_d,
                        qstate,
                        None,
                        Some(&mut scratch.elements),
                    ) else {
                        continue;
                    };
                    e_element = eptr;
                    e_distance = dist;
                }
                // Unreachable: the element/tid variant always matches the
                // index mode.
                _ => unreachable!("visited variant does not match search mode"),
            }

            if !(e_distance < f.distance || always_add) {
                if let Some(discarded) = discarded.as_deref_mut() {
                    let e = init_search_candidate(base, index, e_element, e_distance);
                    discarded.push(NearestItem(e));
                }
                continue;
            }

            // Make robust to issues
            if ((*e_element).level as usize) < lc {
                continue;
            }

            // Create a new candidate
            let e = init_search_candidate(base, index, e_element, e_distance);
            c.push(NearestItem(e));
            furthest.push(FurthestItem(e));

            // Do not count elements being deleted towards ef when vacuuming
            if count_element(skip_element, e_element) {
                wlen += 1;

                // No need to decrement wlen
                if wlen > ef {
                    let d = furthest.pop().expect("W has ef entries");
                    if let Some(discarded) = discarded.as_deref_mut() {
                        discarded.push(NearestItem(d.0));
                    }
                }
            }
        }
    }

    // Add each element of W to w
    while let Some(item) = furthest.pop() {
        w.push(item.0);
    }

    #[cfg(any(test, feature = "pg_test"))]
    if index.is_none() && n_probes > 0 {
        pgrx::log!(
            "hnswsq search_layer: lc={} ef={} wlen={} probes={} expansions={}",
            lc,
            ef,
            wlen,
            n_probes,
            n_expansions
        );
    }

    w
}

// ---------------------------------------------------------------------------
// Algorithm 4 from the paper (hnswutils.c: SelectNeighbors / CheckElementCloser
// / CompareCandidateDistances / AddConnections / HnswUpdateConnection)
// ---------------------------------------------------------------------------

/// `CompareCandidateDistances`: sort descending by distance, ties by pointer
/// (ascending) — and the relptr twin.  pgvector sorts with the tie-breaker to
/// make the `closer` caching deterministic.
fn compare_candidate_distances(base: *mut u8, a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    b.distance.total_cmp(&a.distance).then_with(|| {
        if base.is_null() {
            crate::access_method::hnswsq::ptr::pointer(a.element)
                .cmp(&crate::access_method::hnswsq::ptr::pointer(b.element))
        } else {
            crate::access_method::hnswsq::ptr::offset(a.element)
                .cmp(&crate::access_method::hnswsq::ptr::offset(b.element))
        }
    })
}

/// Distance between two stored (encoded) vectors — the pair distance the
/// selection heuristic needs (`HnswGetDistance` over element values).  For
/// the lossless `plain` layout both sides feed the SIMD kernels directly; the
/// quantized layouts decode one side into `scratch` (the pgvector reference
/// never pays a decode because its stored value is the datum itself).
pub unsafe fn pair_distance(
    base: *mut u8,
    support: &Support,
    a: *mut Element,
    b: *mut Element,
    scratch: &mut Vec<f32>,
) -> f32 {
    let vec_bytes = support.codec.vector_bytes();
    let a_bytes = get_value(base, a, vec_bytes);
    let b_bytes = get_value(base, b, vec_bytes);

    if support.precision == HnswPrecision::Plain
        && (a_bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>())
        && (b_bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>())
    {
        let a_f: &[f32] =
            std::slice::from_raw_parts(a_bytes.as_ptr().cast::<f32>(), support.codec.dim());
        let b_f: &[f32] =
            std::slice::from_raw_parts(b_bytes.as_ptr().cast::<f32>(), support.codec.dim());
        return match support.dist_type {
            DistanceType::L2 => kernels::distance_l2(a_f, b_f),
            DistanceType::Cosine => kernels::distance_cosine(a_f, b_f),
            DistanceType::InnerProduct => kernels::distance_inner_product(a_f, b_f),
        };
    }

    // Decode one side, then compare against the other's stored bytes.
    support.codec.decode_into(a_bytes, scratch.as_mut_slice());
    support.distance(scratch, b_bytes)
}

/// `CheckElementCloser`: is `e` closer to `q` (its own distance) than to every
/// element already in `r`?
unsafe fn check_element_closer(
    base: *mut u8,
    e: *mut Candidate,
    r: &[*mut Candidate],
    support: &Support,
    scratch: &mut Vec<f32>,
) -> bool {
    let e_element = crate::access_method::hnswsq::ptr::access::<Element>(base, (*e).element);

    for &ri in r {
        let ri_element =
            crate::access_method::hnswsq::ptr::access::<Element>(base, (*ri).element);
        let distance = pair_distance(base, support, e_element, ri_element, scratch);
        if distance <= (*e).distance {
            return false;
        }
    }

    true
}

/// `SelectNeighbors` (Algorithm 4): the diversity heuristic with pgvector's
/// `closer` caching.  `c` holds pointers to live candidates (neighbor-array
/// items and/or the caller's `new_candidate`); the returned vector holds the
/// selected ones in nearest-first order (or `c`'s order when nothing is
/// pruned).
///
/// # Safety
/// The candidates must outlive the call; `scratch` is a decode buffer of at
/// least `dim` f32s.
pub unsafe fn select_neighbors(
    base: *mut u8,
    c: &[*mut Candidate],
    lm: usize,
    support: &Support,
    closer_set: &mut bool,
    new_candidate: Option<*mut Candidate>,
    pruned: Option<&mut Option<*mut Candidate>>,
    sort_candidates: bool,
    scratch: &mut Vec<f32>,
) -> Vec<*mut Candidate> {
    let mut r: Vec<*mut Candidate> = Vec::new();
    let mut w: Vec<*mut Candidate> = c.to_vec();
    let mut wd: Vec<*mut Candidate> = Vec::new();
    let mut wdoff = 0usize;
    let mut must_calculate = !*closer_set;
    let mut added: Vec<*mut Candidate> = Vec::new();
    let mut removed_any = false;

    if w.len() <= lm {
        return w;
    }

    // Ensure order of candidates is deterministic for closer caching
    if sort_candidates {
        w.sort_by(|a, b| compare_candidate_distances(base, unsafe { &**a }, unsafe { &**b }));
    }

    while !w.is_empty() && r.len() < lm {
        // Assumes w is already ordered desc
        let e = w.pop().unwrap();

        // Use previous state of r and wd to skip work when possible
        if must_calculate {
            (*e).closer = check_element_closer(base, e, &r, support, scratch);
        } else if !added.is_empty() {
            // If the current candidate was closer, we only need to compare it
            // with the other candidates that we have added.
            if (*e).closer {
                (*e).closer = check_element_closer(base, e, &added, support, scratch);
                if !(*e).closer {
                    removed_any = true;
                }
            } else if removed_any {
                // If we have removed any candidates from closer, a candidate
                // that was not closer earlier might now be.
                (*e).closer = check_element_closer(base, e, &r, support, scratch);
                if (*e).closer {
                    added.push(e);
                }
            }
        } else if Some(e) == new_candidate {
            (*e).closer = check_element_closer(base, e, &r, support, scratch);
            if (*e).closer {
                added.push(e);
            }
        }

        if (*e).closer {
            r.push(e);
        } else {
            wd.push(e);
        }
    }

    // Cached value can only be used in future if sorted deterministically
    *closer_set = sort_candidates;

    // Keep pruned connections
    while wdoff < wd.len() && r.len() < lm {
        r.push(wd[wdoff]);
        wdoff += 1;
    }

    // Return pruned for update connections
    if let Some(pruned) = pruned {
        *pruned = if wdoff < wd.len() {
            Some(wd[wdoff])
        } else {
            w.first().copied()
        };
    }

    r
}

/// `AddConnections`.
unsafe fn add_connections(
    base: *mut u8,
    element: *mut Element,
    neighbors: &[*mut Candidate],
    lc: usize,
) {
    let a = get_neighbors(base, element, lc);
    for &hc in neighbors {
        *neighbor_items(a).add((*a).length as usize) = *hc;
        (*a).length += 1;
    }
}

/// `HnswUpdateConnection`: append the new element to the neighbor array, or —
/// when the list is full — re-run the selection and replace the pruned entry.
///
/// # Safety
/// `neighbors` is the target element's array (under its lock in a parallel
/// build); `new_element` must be a live element with a materialized value.
pub unsafe fn update_connection(
    base: *mut u8,
    neighbors: *mut NeighborArray,
    new_element: *mut Element,
    distance: f32,
    lm: usize,
    update_idx: Option<&mut i32>,
    index: Option<pg_sys::Relation>,
    support: &Support,
    scratch: &mut Vec<f32>,
) {
    let mut new_hp = HnswPtr { ptr: std::ptr::null_mut() };
    crate::access_method::hnswsq::ptr::store(base, &mut new_hp, new_element);
    let mut new_hc = Candidate {
        element: new_hp,
        distance,
        closer: false,
    };

    if (*neighbors).length < lm as u32 {
        *neighbor_items(neighbors).add((*neighbors).length as usize) = new_hc;
        (*neighbors).length += 1;

        // Track update
        if let Some(update_idx) = update_idx {
            *update_idx = -2;
        }
    } else {
        // Shrink connections
        let mut c: Vec<*mut Candidate> = Vec::with_capacity(lm + 1);
        for i in 0..(*neighbors).length as usize {
            c.push(&mut *neighbor_items(neighbors).add(i));
        }
        c.push(&mut new_hc);

        let mut closer_set = (*neighbors).closer_set;
        let mut pruned: Option<*mut Candidate> = None;
        let _r = select_neighbors(
            base,
            &c,
            lm,
            support,
            &mut closer_set,
            Some(&mut new_hc as *mut Candidate),
            Some(&mut pruned),
            true,
            scratch,
        );
        (*neighbors).closer_set = closer_set;

        // Should not happen
        let Some(pruned) = pruned else { return };

        // Find and replace the pruned element
        for i in 0..(*neighbors).length as usize {
            if crate::access_method::hnswsq::ptr::equal(
                base,
                (*neighbor_items(neighbors).add(i)).element,
                (*pruned).element,
            ) {
                *neighbor_items(neighbors).add(i) = new_hc;

                // Track update
                if let Some(update_idx) = update_idx {
                    *update_idx = i as i32;
                }
                break;
            }
        }
    }
}

/// `RemoveElements`: filter out the skip element and elements being deleted.
pub unsafe fn remove_elements(
    base: *mut u8,
    w: Vec<Candidate>,
    skip_element: Option<*mut Element>,
) -> Vec<Candidate> {
    // Ensure does not access heaptid_set during an in-memory build
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

    w.into_iter()
        .filter(|hc| {
            let hce = crate::access_method::hnswsq::ptr::access::<Element>(base, hc.element);
            if let Some(skip) = skip_element {
                if (*hce).blkno == (*skip).blkno && (*hce).offno == (*skip).offno {
                    return false;
                }
            }
            (*hce).heaptid_set != 0
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Algorithm 1 from the paper (hnswutils.c: HnswFindElementNeighbors)
// ---------------------------------------------------------------------------

/// `HnswFindElementNeighbors`: find and set `element`'s neighbors at every
/// layer, starting from the entry point.  `existing` is the vacuum-repair
/// mode (skip the element itself, +1 to ef).
///
/// # Safety
/// `element` has a materialized value; `entry_point` is the current entry (or
/// none for an empty graph); the scratch buffers are caller-owned.
#[allow(clippy::too_many_arguments)]
pub unsafe fn find_element_neighbors(
    base: *mut u8,
    element: *mut Element,
    entry_point: Option<*mut Element>,
    index: Option<pg_sys::Relation>,
    support: &Support,
    m: usize,
    ef_construction: usize,
    existing: bool,
    scratch: &mut SearchScratch,
    decode: &mut Vec<f32>,
    pair_scratch: &mut Vec<f32>,
    visited: &mut Visited,
    sq8_mode: crate::access_method::hnswsq::options::Sq8DistanceMode,
) {
    let mut level = (*element).level as usize;
    let skip_element = if existing { Some(element) } else { None };
    let in_memory = index.is_none();

    // q = the element's value, decoded once (the search compares the stored
    // vector against the other elements' stored vectors).
    let vec_bytes = support.codec.vector_bytes();
    let value = get_value(base, element, vec_bytes);
    decode.clear();
    decode.resize(support.codec.dim(), 0.0);
    pair_scratch.clear();
    pair_scratch.resize(support.codec.dim(), 0.0);
    support.codec.decode_into(value, decode.as_mut_slice());

    // SQ8: quantize the decoded query once; every search distance below is
    // then pure integer arithmetic (see sq8_query_state).  The mode comes
    // from the caller: the parallel build passes the leader's GUC value
    // through shared memory (workers do not see the leader's session GUCs).
    let qstate = sq8_query_state(support, decode.as_slice(), sq8_mode);

    // Precompute hash
    if in_memory {
        precompute_hash(base, element);
    }

    // No neighbors if no entry point
    let Some(entry_point) = entry_point else {
        return;
    };

    // Get entry point and level
    let mut ep = vec![entry_candidate(
        base,
        entry_point,
        Some(decode.as_slice()),
        index,
        support,
        true,
        qstate.as_ref(),
    )];
    let mut entry_level = (*entry_point).level as usize;

    // 1st phase: greedy search to insert level
    for lc in ((level + 1)..=entry_level).rev() {
        let w = search_layer(
            base,
            index,
            support,
            m,
            Some(decode.as_slice()),
            &ep,
            1,
            lc,
            true,
            skip_element,
            visited,
            None,
            true,
            None,
            scratch,
            qstate.as_ref(),
        );
        ep = w;
    }

    if level > entry_level {
        level = entry_level;
    }

    // Add one for existing element
    let ef_construction = if existing {
        ef_construction + 1
    } else {
        ef_construction
    };

    // 2nd phase
    for lc in (0..=level).rev() {
        let lm = get_layer_m(m, lc);

        let w = search_layer(
            base,
            index,
            support,
            m,
            Some(decode.as_slice()),
            &ep,
            ef_construction,
            lc,
            true,
            skip_element,
            visited,
            None,
            true,
            None,
            scratch,
            qstate.as_ref(),
        );

        // Convert search candidates to candidates
        let mut lw: Vec<Candidate> = w
            .iter()
            .map(|sc| Candidate {
                element: sc.element,
                distance: sc.distance,
                closer: false,
            })
            .collect();

        // Elements being deleted or skipped can help with search
        // but should be removed before selecting neighbors
        if !in_memory {
            lw = remove_elements(base, lw, skip_element);
        }

        let c: Vec<*mut Candidate> = lw.iter_mut().map(|hc| hc as *mut Candidate).collect();

        // Candidates are sorted, but not deterministically. Could set
        // sortCandidates to true for in-memory builds to enable closer
        // caching, but there does not seem to be a difference in performance.
        let mut closer_set = (*get_neighbors(base, element, lc)).closer_set;
        let neighbors = select_neighbors(
            base,
            &c,
            lm,
            support,
            &mut closer_set,
            None,
            None,
            false,
            pair_scratch,
        );
        (*get_neighbors(base, element, lc)).closer_set = closer_set;

        add_connections(base, element, &neighbors, lc);

        ep = w;
    }
}

// ---------------------------------------------------------------------------
// Level drawing (hnswutils.c: HnswInitElement's RandomDouble draw)
// ---------------------------------------------------------------------------

/// One geometric level draw: `floor(-ln(u) * ml)`, clamped to `max_level`
/// (pgvector `HnswInitElement`; the caller supplies the uniform draw so the
/// build can pin levels per row).
pub fn random_level(ml: f64, max_level: usize, u: f64) -> usize {
    let level = (-u.ln() * ml) as usize;
    level.min(max_level)
}

/// An entropy draw for the insert path (pgvector uses the global PRNG; we
/// use the rand crate like the old engine).
pub fn entropy_level(ml: f64, max_level: usize) -> usize {
    use rand::Rng;
    let u: f64 = rand::thread_rng().gen_range(0.0..1.0);
    random_level(ml, max_level, u)
}

/// A build row's level.  With a pinned `hnswsq.build_seed` the level is a
/// pure function of `(seed, tid)` so a parallel build assigns the same level
/// no matter which worker processes the row (the old engine's determinism
/// machinery; pgvector draws from the process PRNG instead).  `seed < 0`
/// draws from entropy, like pgvector.
pub fn build_level(seed: i32, ml: f64, max_level: usize, tid: pg_sys::ItemPointerData) -> usize {
    if seed < 0 {
        return entropy_level(ml, max_level);
    }
    let block = ((tid.ip_blkid.bi_hi as u32) << 16) | tid.ip_blkid.bi_lo as u32;
    let offset = tid.ip_posid;
    // Same fnv1a mixing as the old levels module, so levels match for the
    // same (seed, tid) when comparing builds.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in block
        .to_le_bytes()
        .into_iter()
        .chain(offset.to_le_bytes())
    {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    // Fold the 64-bit hash to a uniform draw in [0, 1).
    let u = (h >> 11) as f64 / (1u64 << 53) as f64;
    random_level(ml, max_level, u)
}

// ---------------------------------------------------------------------------
// Tests (lock-free pieces; the locked search path is exercised by pg_test)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_method::hnswsq::ptr::HnswPtr;

    /// An allocator for plain Rust tests: Box-allocated, absolute pointers,
    /// no backend needed (no LWLocks are touched by the tested functions).
    struct TestAlloc;
    impl TestAlloc {
        unsafe fn alloc<T>(&self) -> *mut T {
            Box::into_raw(Box::<T>::new_uninit()).cast()
        }
        unsafe fn alloc_bytes(&self, size: usize) -> *mut u8 {
            let mut v = vec![0u8; size];
            let p = v.as_mut_ptr();
            std::mem::forget(v);
            p
        }
    }

    unsafe fn test_element(
        alloc: &TestAlloc,
        m: usize,
        level: usize,
        value: Vec<f32>,
    ) -> (*mut Element, Codec) {
        let dim = value.len();
        let codec = Codec::new(HnswPrecision::Plain, dim);
        let mut tid = pg_sys::ItemPointerData::default();
        pgrx::itemptr::item_pointer_set_all(&mut tid, 1, 1);
        // Allocate the element and its value by hand (init_element needs an
        // Allocator + a backend; the unit tests build the same shape).
        let element = alloc.alloc::<Element>();
        (*element).next = HnswPtr { ptr: std::ptr::null_mut() };
        (*element).heaptid = tid;
        (*element).heaptid_set = 1;
        (*element).level = level as u8;
        (*element).deleted = 0;
        (*element).clamped = 0;
        (*element).version = 1;
        (*element).hash = 0;
        (*element).blkno = 0;
        (*element).offno = 0;
        (*element).neighbor_offno = pg_sys::InvalidOffsetNumber;
        (*element).neighbor_page = pg_sys::InvalidBlockNumber;
        let neighbor_list = alloc
            .alloc_bytes((level + 1) * std::mem::size_of::<HnswPtr>())
            .cast::<HnswPtr>();
        crate::access_method::hnswsq::ptr::store(
            std::ptr::null_mut(),
            &mut (*element).neighbors,
            neighbor_list,
        );
        for lc in 0..=level {
            let lm = get_layer_m(m, lc);
            let na = alloc
                .alloc_bytes(neighbor_array_size(lm))
                .cast::<NeighborArray>();
            (*na).length = 0;
            (*na).closer_set = false;
            crate::access_method::hnswsq::ptr::store(
                std::ptr::null_mut(),
                &mut *neighbor_list.add(lc),
                na,
            );
        }
        let bytes = codec.encode(&value);
        let vp = alloc.alloc_bytes(bytes.len());
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), vp, bytes.len());
        crate::access_method::hnswsq::ptr::store(std::ptr::null_mut(), &mut (*element).value, vp);
        (element, codec)
    }

    #[test]
    fn test_layer_m_and_ml() {
        assert_eq!(get_layer_m(16, 0), 32);
        assert_eq!(get_layer_m(16, 3), 16);
        assert!((get_ml(16) - 1.0 / 16f64.ln()).abs() < 1e-12);
        assert!(get_max_level(16) >= 16, "typical levels fit");
        assert!(get_max_level(100) <= 63);
    }

    #[test]
    fn test_random_level_distribution() {
        let ml = get_ml(16);
        let levels: Vec<usize> = (0..20_000)
            .map(|_| {
                use rand::Rng;
                let u: f64 = rand::thread_rng().gen_range(0.0..1.0);
                random_level(ml, 63, u)
            })
            .collect();
        let zero = levels.iter().filter(|&&l| l == 0).count();
        assert!(
            zero > 20_000 * 14 / 16 - 2000,
            "most nodes are level 0: {}",
            zero
        );
        assert!(levels.iter().any(|&l| l > 0), "some are not");
        assert_eq!(random_level(ml, 0, 1e-12), 0, "the cap holds");
    }

    #[test]
    fn test_select_neighbors_prefers_diverse_close_candidates() {
        unsafe {
            let alloc = TestAlloc;
            let support = Support {
                dist_type: DistanceType::L2,
                precision: HnswPrecision::Plain,
                codec: Codec::new(HnswPrecision::Plain, 1),
            };
            // Four candidates on a line, values 0.1..0.4, distances to q the
            // same.  The algorithm pops from the END of w, so candidates are
            // passed furthest-first, like search_layer's result list.  With
            // lm = 2, the heuristic keeps the closest (0.1) and the next
            // closest that is farther from 0.1 than from q (0.2); 0.3 is 0.2
            // away from 0.1, which is not > 0.3, so it loses.
            let mk = |v: f32| -> (*mut Element, Candidate) {
                let (e, _codec) = test_element(&alloc, 4, 0, vec![v]);
                let mut hp = HnswPtr { ptr: std::ptr::null_mut() };
                crate::access_method::hnswsq::ptr::store(std::ptr::null_mut(), &mut hp, e);
                (
                    e,
                    Candidate {
                        element: hp,
                        distance: v,
                        closer: false,
                    },
                )
            };
            let (e1, c1) = mk(0.1);
            let (e2, c2) = mk(0.2);
            let (e3, c3) = mk(0.3);
            let (e4, c4) = mk(0.4);
            let mut arr = [c4, c3, c2, c1]; // furthest-first
            let c: Vec<*mut Candidate> = arr.iter_mut().map(|c| c as *mut Candidate).collect();
            let mut closer_set = false;
            let mut scratch = vec![0.0f32; 1];
            let r = select_neighbors(
                std::ptr::null_mut(),
                &c,
                2,
                &support,
                &mut closer_set,
                None,
                None,
                false,
                &mut scratch,
            );
            assert_eq!(r.len(), 2);
            let dists: Vec<f32> = r.iter().map(|c| (**c).distance).collect();
            assert_eq!(dists, vec![0.1, 0.2]);
            let _ = (e1, e2, e3, e4);
        }
    }

    #[test]
    fn test_update_connection_appends_then_replaces() {
        unsafe {
            let alloc = TestAlloc;
            let support = Support {
                dist_type: DistanceType::L2,
                precision: HnswPrecision::Plain,
                codec: Codec::new(HnswPrecision::Plain, 1),
            };
            let na = alloc
                .alloc_bytes(neighbor_array_size(2))
                .cast::<NeighborArray>();
            (*na).length = 0;
            (*na).closer_set = false;
            let mut scratch = vec![0.0f32; 1];

            // Append when there is room: idx = -2.
            let (a, _) = test_element(&alloc, 4, 0, vec![1.0]);
            let mut idx = 0i32;
            update_connection(
                std::ptr::null_mut(),
                na,
                a,
                1.0,
                2,
                Some(&mut idx),
                None,
                &support,
                &mut scratch,
            );
            assert_eq!(idx, -2);
            assert_eq!((*na).length, 1);

            // Fill the second slot.
            let (b, _) = test_element(&alloc, 4, 0, vec![2.0]);
            let mut idx = 0i32;
            update_connection(
                std::ptr::null_mut(),
                na,
                b,
                2.0,
                2,
                Some(&mut idx),
                None,
                &support,
                &mut scratch,
            );
            assert_eq!(idx, -2);
            assert_eq!((*na).length, 2);

            // The list is full: the new element (value 0.5, squared distance
            // 0.25) enters; the diversity rule keeps b (2.25 away from the
            // new element vs 2.0 from q) and prunes a (0.25 away from the new
            // element vs 1.0 from q) — L2 distances are squared, exactly as
            // the operator computes them.
            let (c, _) = test_element(&alloc, 4, 0, vec![0.5]);
            let mut idx = 0i32;
            update_connection(
                std::ptr::null_mut(),
                na,
                c,
                0.5,
                2,
                Some(&mut idx),
                None,
                &support,
                &mut scratch,
            );
            assert_eq!((*na).length, 2);
            let mut vals: Vec<f32> = (0..2)
                .map(|i| {
                    let hp = (*neighbor_items(na).add(i)).element;
                    let e =
                        crate::access_method::hnswsq::ptr::access::<Element>(
                            std::ptr::null_mut(),
                            hp,
                        );
                    let v = get_value(std::ptr::null_mut(), e, 4);
                    Codec::new(HnswPrecision::Plain, 1).decode(v)[0]
                })
                .collect();
            vals.sort_by(|a, b| a.total_cmp(b));
            assert_eq!(vals, vec![0.5, 2.0]);
        }
    }

    #[test]
    fn test_visited_and_pack_tid() {
        let mut v = Visited::new(4);
        let mut tid = pg_sys::ItemPointerData::default();
        pgrx::itemptr::item_pointer_set_all(&mut tid, 7, 3);
        let key = pack_tid(tid);
        assert!(!v.insert(key));
        assert!(v.insert(key));
        assert_eq!(murmur64(42), murmur64(42));
        assert_ne!(murmur64(42), murmur64(43));
    }

    #[test]
    fn test_element_tuple_roundtrip() {
        unsafe {
            let alloc = TestAlloc;
            let (element, codec) = test_element(&alloc, 4, 2, vec![1.0, -2.0, 3.5, 0.0]);
            (*element).blkno = 5;
            (*element).offno = 9;
            let vec_bytes = codec.vector_bytes();
            let etup_size = element_tuple_size(vec_bytes);
            let etup = alloc.alloc_bytes(etup_size).cast::<ElementTupleData>();
            std::ptr::write_bytes(etup.cast::<u8>(), 0, etup_size);
            set_element_tuple(
                std::ptr::null_mut(),
                etup,
                element,
                HnswPrecision::Plain as u8,
                vec_bytes,
            );
            assert_eq!((*etup).type_, ELEMENT_TUPLE_TYPE);
            assert_eq!((*etup).level, 2);
            assert_eq!((*etup).version, 1);
            assert_eq!((*etup).deleted, 0);

            // Load back into a fresh element.
            let mut out = init_element_from_block(5, 9);
            load_element_from_tuple(&mut *out, etup, true, true, vec_bytes);
            assert_eq!(out.level, 2);
            assert_eq!(out.heaptid_set, 1);
            assert_eq!(
                get_value(std::ptr::null_mut(), &mut *out, vec_bytes),
                codec.encode(&[1.0, -2.0, 3.5, 0.0])
            );

            // Neighbor tuple roundtrip: set with two candidate neighbors.
            let (n1, _) = test_element(&alloc, 4, 0, vec![1.0, 0.0, 0.0, 0.0]);
            (*n1).blkno = 10;
            (*n1).offno = 1;
            let (n2, _) = test_element(&alloc, 4, 0, vec![0.0, 1.0, 0.0, 0.0]);
            (*n2).blkno = 11;
            (*n2).offno = 2;
            let na = get_neighbors(std::ptr::null_mut(), element, 0);
            let mut hp1 = HnswPtr { ptr: std::ptr::null_mut() };
            crate::access_method::hnswsq::ptr::store(std::ptr::null_mut(), &mut hp1, n1);
            let mut hp2 = HnswPtr { ptr: std::ptr::null_mut() };
            crate::access_method::hnswsq::ptr::store(std::ptr::null_mut(), &mut hp2, n2);
            *neighbor_items(na) = Candidate {
                element: hp1,
                distance: 0.1,
                closer: false,
            };
            *neighbor_items(na).add(1) = Candidate {
                element: hp2,
                distance: 0.2,
                closer: false,
            };
            (*na).length = 2;

            let ntup_size = neighbor_tuple_size(2, 4);
            let ntup = alloc.alloc_bytes(ntup_size).cast::<NeighborTupleData>();
            std::ptr::write_bytes(ntup.cast::<u8>(), 0, ntup_size);
            set_neighbor_tuple(std::ptr::null_mut(), ntup, element, 4);
            assert_eq!((*ntup).type_, NEIGHBOR_TUPLE_TYPE);
            assert_eq!((*ntup).count as usize, (2 + 2) * 4);
            let tids = ntup
                .cast::<u8>()
                .add(NEIGHBOR_TUPLE_HEADER_SIZE)
                .cast::<pg_sys::ItemPointerData>();
            // Highest layer first: layer 2 at index 0.
            assert_eq!(
                ip_block(&*tids.add(2 * 4)),
                10
            );
            assert_eq!(
                ip_block(&*tids.add(2 * 4 + 1)),
                11
            );
            // Padding beyond the valid prefix is invalid.
            assert_eq!(
                ip_block(&*tids.add(2 * 4 + 2)),
                pg_sys::InvalidBlockNumber
            );
        }
    }
}
