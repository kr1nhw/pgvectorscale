//! hnswsq2 build — the Rust translation of pgvector's `hnswbuild.c`.
//!
//! Two phases, exactly as in the reference:
//!
//! 1. **In-memory phase**: the graph is held completely in memory (backend
//!    memory, or a shared area for a parallel build).  When the graph is
//!    fully built — or `maintenance_work_mem` runs out — the pages are
//!    materialized (`flush_pages`) and the build switches to the on-disk
//!    path (`insert_tuple_on_disk`), which inserts row by row without WAL.
//! 2. **On-disk phase**: same insert code as `INSERT`, minus WAL.  After the
//!    build, the whole page range is WAL-logged at once (`log_newpage_range`).
//!
//! Deliberate divergences from the reference, and only these:
//!
//! * the SQ8 calibration is reservoir-sampled in a first scan, its chain is
//!   written right after the metapage (so parallel workers can read the
//!   codec), and the graph head block is recorded in the metapage because
//!   the chain may occupy blocks before the graph pages;
//! * levels are pinned per row `(seed, tid)` when `hnswsq2.build_seed` is
//!   set (pgvector draws from the process PRNG), so a parallel build assigns
//!   the same levels regardless of scheduling;
//! * no duplicate-element merging (single-heaptid divergence, see
//!   `types.rs`);
//! * scratch buffers are caller-owned instead of reset per row.

use pgrx::pg_sys;
use pgrx::*;
use rand::SeedableRng;

use crate::access_method::distance::{preprocess_cosine, DistanceType};
use crate::access_method::hnswsq::quantize::{Codec, HnswPrecision, Sq8Calibration};
use crate::access_method::hnswsq2::options::{Hnsw2Options, HNSW2_BUILD_SEED};
use crate::access_method::hnswsq2::types::*;
use crate::access_method::hnswsq2::utils::*;
use crate::access_method::pg_vector::PgVectorInternal;
use crate::util::ports::PageGetMaxOffsetNumber;
use crate::util::ItemPointer;

/// shm_toc keys of a parallel build (pgvector's `PARALLEL_KEY_HNSW_*`).
const PARALLEL_KEY_SHARED: u64 = 0xA000_0000_0000_0001;
const PARALLEL_KEY_AREA: u64 = 0xA000_0000_0000_0002;

/// The library PostgreSQL loads in a parallel worker to find the entry point
/// (versioned, as pgrx installs it).
const LIBRARY: &str = concat!("vectorscale-", env!("CARGO_PKG_VERSION"), "\0");

/// Per-allocation margin for shm_toc chunk estimates (the old engine's
/// driver measured `shm_toc_allocate` charging more than BUFFERALIGN).
const CHUNK_MARGIN: usize = 64;

// ---------------------------------------------------------------------------
// Build state
// ---------------------------------------------------------------------------

/// The build's backend-local state (pgvector `HnswBuildState`).
pub struct BuildState {
    pub heap: Option<pg_sys::Relation>,
    pub index: pg_sys::Relation,
    pub index_info: *mut pg_sys::IndexInfo,
    pub m: usize,
    pub ef_construction: usize,
    pub dimensions: usize,
    pub support: Support,
    pub precision: HnswPrecision,
    pub reltuples: f64,
    pub indtuples: f64,
    /// The private-path graph, by value; `graph_ptr` addresses either this or
    /// the shared one.
    pub graph: Graph,
    pub graph_ptr: *mut Graph,
    /// Shared-area base; null for the private path (absolute pointers).
    pub base: *mut u8,
    pub ml: f64,
    pub max_level: usize,
    pub graph_ctx: PgMemoryContexts,
    pub tmp_ctx: PgMemoryContexts,
    pub tranche: i32,
    pub seed: i32,
    pub leader: Option<Leader>,
    // Scratch (caller-owned, reused per row)
    pub scratch: SearchScratch,
    pub decode: Vec<f32>,
    pub pair_scratch: Vec<f32>,
    pub visited: Visited,
    pub encoded: Vec<u8>,
    pub vector: Vec<f32>,
}

/// pgvector `HnswLeader`.
pub struct Leader {
    pub pcxt: *mut pg_sys::ParallelContext,
    pub nparticipanttuplesorts: i32,
    pub shared: *mut Shared,
    pub snapshot: pg_sys::Snapshot,
    /// Whether `snapshot` was registered (concurrent builds); `SnapshotAny`
    /// (non-concurrent) must not be unregistered.
    pub snapshot_registered: bool,
    pub area: *mut u8,
}

/// Register an LWLock tranche for a build (the old engine's runtime route:
/// `RequestNamedLWLockTranche` only works from shared_preload_libraries).
pub fn register_tranche(name: &'static std::ffi::CStr) -> i32 {
    unsafe {
        let id = pg_sys::LWLockNewTrancheId();
        pg_sys::LWLockRegisterTranche(id, name.as_ptr());
        id
    }
}

/// `InitGraph` (hnswbuild.c): initialize a graph's pointers, memory budget
/// and locks.
///
/// # Safety
/// `graph` must be zeroed storage the caller owns (backend or shared), and
/// `base` the relptr base (null for backend memory).
pub unsafe fn init_graph(graph: *mut Graph, base: *mut u8, memory_total: usize, tranche: i32) {
    crate::access_method::hnswsq2::ptr::store(base, &mut (*graph).head, std::ptr::null_mut::<u8>());
    crate::access_method::hnswsq2::ptr::store(
        base,
        &mut (*graph).entry_point,
        std::ptr::null_mut::<u8>(),
    );
    // Avoid the base address for relptrs: offset 0 is the NULL encoding in
    // this port's relptr (ptr.rs stores raw offsets), so the first shared
    // allocation must not land at offset 0 — pgvector's pre-14.5 workaround
    // (`memoryUsed += MAXALIGN(1)`), kept unconditionally.
    (*graph).memory_used = if base.is_null() {
        0
    } else {
        pg_sys::MAXALIGN(1)
    };
    (*graph).memory_total = memory_total.min(MAX_GRAPH_MEMORY);
    (*graph).flushed = false;
    (*graph).indtuples = 0.0;
    pg_sys::SpinLockInit(std::ptr::addr_of_mut!((*graph).lock));
    pg_sys::LWLockInitialize(std::ptr::addr_of_mut!((*graph).entry_lock), tranche);
    pg_sys::LWLockInitialize(std::ptr::addr_of_mut!((*graph).entry_wait_lock), tranche);
    pg_sys::LWLockInitialize(std::ptr::addr_of_mut!((*graph).allocator_lock), tranche);
    pg_sys::LWLockInitialize(std::ptr::addr_of_mut!((*graph).flush_lock), tranche);
}

/// The allocator for `build`'s current regime.
unsafe fn allocator(build: &BuildState) -> Allocator {
    if build.base.is_null() {
        Allocator::Private {
            ctx: build.graph_ctx.value(),
            graph: build.graph_ptr,
        }
    } else {
        Allocator::Shared {
            base: build.base,
            graph: build.graph_ptr,
        }
    }
}

// ---------------------------------------------------------------------------
// In-memory insert (hnswbuild.c: InsertTuple / InsertTupleInMemory /
// UpdateGraphInMemory)
// ---------------------------------------------------------------------------

/// `AddElementInMemory`: push an element onto the graph's head list.
unsafe fn add_element_in_memory(base: *mut u8, graph: *mut Graph, element: *mut Element) {
    pg_sys::SpinLockAcquire(std::ptr::addr_of_mut!((*graph).lock));
    (*element).next = (*graph).head;
    crate::access_method::hnswsq2::ptr::store(base, &mut (*graph).head, element);
    pg_sys::SpinLockRelease(std::ptr::addr_of_mut!((*graph).lock));
}

/// `UpdateNeighborsInMemory`: add `e` to its neighbors' lists (backlinks),
/// one element lock at a time.
unsafe fn update_neighbors_in_memory(
    base: *mut u8,
    support: &Support,
    e: *mut Element,
    m: usize,
    local: &mut Vec<Candidate>,
    pair_scratch: &mut Vec<f32>,
) {
    for lc in (0..=(*e).level as usize).rev() {
        let lm = get_layer_m(m, lc);

        // Copy neighbors to local memory
        pg_sys::LWLockAcquire(std::ptr::addr_of_mut!((*e).lock), pg_sys::LWLockMode::LW_SHARED);
        let neighbors = get_neighbors(base, e, lc);
        local.clear();
        for i in 0..(*neighbors).length as usize {
            local.push(*neighbor_items(neighbors).add(i));
        }
        pg_sys::LWLockRelease(std::ptr::addr_of_mut!((*e).lock));

        for hc in local.iter() {
            let neighbor = crate::access_method::hnswsq2::ptr::access::<Element>(base, hc.element);
            pg_sys::LWLockAcquire(std::ptr::addr_of_mut!((*neighbor).lock), pg_sys::LWLockMode::LW_EXCLUSIVE);
            update_connection(
                base,
                get_neighbors(base, neighbor, lc),
                e,
                hc.distance,
                lm,
                None,
                None,
                support,
                pair_scratch,
            );
            pg_sys::LWLockRelease(std::ptr::addr_of_mut!((*neighbor).lock));
        }
    }
}

/// `UpdateGraphInMemory` (minus the duplicate search — the single-heaptid
/// divergence makes it always miss, so it is skipped entirely).
unsafe fn update_graph_in_memory(
    base: *mut u8,
    support: &Support,
    element: *mut Element,
    m: usize,
    entry_point: Option<*mut Element>,
    graph: *mut Graph,
    local: &mut Vec<Candidate>,
    pair_scratch: &mut Vec<f32>,
) {
    // Add element
    add_element_in_memory(base, graph, element);

    // Update neighbors
    update_neighbors_in_memory(base, support, element, m, local, pair_scratch);

    // Update entry point if needed (already have lock)
    if entry_point.is_none() || (*element).level > (*entry_point.unwrap()).level {
        crate::access_method::hnswsq2::ptr::store(base, &mut (*graph).entry_point, element);
    }
}

/// `InsertTupleInMemory`: the entry-lock handshake, then find neighbors and
/// update the graph.
unsafe fn insert_tuple_in_memory(build: &mut BuildState, element: *mut Element) {
    let graph = build.graph_ptr;
    let base = build.base;
    let support = &build.support;
    let m = build.m;
    let ef_construction = build.ef_construction;

    // Wait if another process needs exclusive lock on entry lock
    pg_sys::LWLockAcquire(std::ptr::addr_of_mut!((*graph).entry_wait_lock), pg_sys::LWLockMode::LW_EXCLUSIVE);
    pg_sys::LWLockRelease(std::ptr::addr_of_mut!((*graph).entry_wait_lock));

    // Get entry point
    pg_sys::LWLockAcquire(std::ptr::addr_of_mut!((*graph).entry_lock), pg_sys::LWLockMode::LW_SHARED);
    let mut entry_hp = (*graph).entry_point;
    let mut entry_point = if crate::access_method::hnswsq2::ptr::is_null(base, entry_hp) {
        None
    } else {
        Some(crate::access_method::hnswsq2::ptr::access::<Element>(base, entry_hp))
    };

    // Prevent concurrent inserts when likely updating entry point
    if entry_point.is_none() || (*element).level > (*entry_point.unwrap()).level {
        // Release shared lock
        pg_sys::LWLockRelease(std::ptr::addr_of_mut!((*graph).entry_lock));

        // Tell other processes to wait and get exclusive lock
        pg_sys::LWLockAcquire(std::ptr::addr_of_mut!((*graph).entry_wait_lock), pg_sys::LWLockMode::LW_EXCLUSIVE);
        pg_sys::LWLockAcquire(std::ptr::addr_of_mut!((*graph).entry_lock), pg_sys::LWLockMode::LW_EXCLUSIVE);
        pg_sys::LWLockRelease(std::ptr::addr_of_mut!((*graph).entry_wait_lock));

        // Get latest entry point after lock is acquired
        entry_hp = (*graph).entry_point;
        entry_point = if crate::access_method::hnswsq2::ptr::is_null(base, entry_hp) {
            None
        } else {
            Some(crate::access_method::hnswsq2::ptr::access::<Element>(base, entry_hp))
        };
    }

    // Find neighbors for element
    find_element_neighbors(
        base,
        element,
        entry_point,
        None,
        support,
        m,
        ef_construction,
        false,
        &mut build.scratch,
        &mut build.decode,
        &mut build.pair_scratch,
        &mut build.visited,
    );

    // Update graph in memory
    update_graph_in_memory(
        base,
        support,
        element,
        m,
        entry_point,
        graph,
        &mut build.scratch.local,
        &mut build.pair_scratch,
    );

    // Release entry lock
    pg_sys::LWLockRelease(std::ptr::addr_of_mut!((*graph).entry_lock));
}

/// `InsertTuple` (hnswbuild.c): form the encoded value, respect the memory
/// budget (flushing to the on-disk path when it runs out), allocate the
/// element and insert it.
unsafe fn insert_tuple(
    build: &mut BuildState,
    heaptid: &pg_sys::ItemPointerData,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
) -> bool {
    // Skip nulls
    if *isnull {
        return false;
    }

    // Form index value: detoast, normalize for cosine, encode.
    build.vector.clear();
    {
        let detoasted = pg_sys::pg_detoast_datum_copy((*values).cast_mut_ptr());
        let pg_vec = detoasted.cast::<PgVectorInternal>();
        build.vector.extend_from_slice((*pg_vec).to_slice());
        pg_sys::pfree(detoasted.cast());
    }
    if build.support.dist_type == DistanceType::Cosine {
        preprocess_cosine(&mut build.vector);
    }
    build.encoded.clear();
    let clamped = build.support.codec.encode_into(&build.vector, &mut build.encoded);

    let value_size = build.encoded.len();
    let graph = build.graph_ptr;
    let base = build.base;

    // In a parallel build, add a margin so allocations never fail
    let memory_margin = if base.is_null() { 0 } else { MEMORY_MARGIN };

    // Ensure graph not flushed when inserting
    pg_sys::LWLockAcquire(std::ptr::addr_of_mut!((*graph).flush_lock), pg_sys::LWLockMode::LW_SHARED);

    // Are we in the on-disk phase?
    if (*graph).flushed {
        pg_sys::LWLockRelease(std::ptr::addr_of_mut!((*graph).flush_lock));
        return crate::access_method::hnswsq2::insert::insert_tuple_on_disk(
            build.index,
            &build.support,
            &build.encoded,
            heaptid,
            true,
            clamped,
        );
    }

    pg_sys::LWLockAcquire(std::ptr::addr_of_mut!((*graph).allocator_lock), pg_sys::LWLockMode::LW_EXCLUSIVE);

    // Check that we have enough memory available for the new element now that
    // we have the allocator lock, and flush pages if needed.
    if (*graph).memory_used + memory_margin >= (*graph).memory_total {
        pg_sys::LWLockRelease(std::ptr::addr_of_mut!((*graph).allocator_lock));

        pg_sys::LWLockRelease(std::ptr::addr_of_mut!((*graph).flush_lock));
        pg_sys::LWLockAcquire(std::ptr::addr_of_mut!((*graph).flush_lock), pg_sys::LWLockMode::LW_EXCLUSIVE);

        if !(*graph).flushed {
            let indtuples = (*graph).indtuples;
            pgrx::notice!(
                "hnswsq2 graph no longer fits into maintenance_work_mem after {} tuples",
                indtuples as i64
            );
            flush_pages(build);
        }

        pg_sys::LWLockRelease(std::ptr::addr_of_mut!((*graph).flush_lock));

        return crate::access_method::hnswsq2::insert::insert_tuple_on_disk(
            build.index,
            &build.support,
            &build.encoded,
            heaptid,
            true,
            clamped,
        );
    }

    // Ok, we can proceed to allocate the element
    let level = build_level(build.seed, build.ml, build.max_level, *heaptid);
    let allocator = allocator(build);
    let element = init_element(base, heaptid, build.m, level, &allocator);
    let value_ptr = allocator.alloc(value_size);

    // We have now allocated the space needed for the element, so we don't
    // need the allocator lock anymore.
    pg_sys::LWLockRelease(std::ptr::addr_of_mut!((*graph).allocator_lock));

    // Copy the encoded value
    std::ptr::copy_nonoverlapping(build.encoded.as_ptr(), value_ptr, value_size);
    crate::access_method::hnswsq2::ptr::store(base, &mut (*element).value, value_ptr);

    // Create a lock for the element
    pg_sys::LWLockInitialize(std::ptr::addr_of_mut!((*element).lock), build.tranche);

    // Insert tuple
    insert_tuple_in_memory(build, element);

    // Release flush lock
    pg_sys::LWLockRelease(std::ptr::addr_of_mut!((*graph).flush_lock));

    true
}

/// `BuildCallback` (hnswbuild.c): one row of the heap scan.
unsafe extern "C-unwind" fn build_callback(
    _index: pg_sys::Relation,
    ctid: pg_sys::ItemPointer,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut std::os::raw::c_void,
) {
    let build = &mut *(state as *mut BuildState);
    let graph = build.graph_ptr;

    let old_ctx = pg_sys::CurrentMemoryContext;
    pg_sys::CurrentMemoryContext = build.tmp_ctx.value();
    if insert_tuple(build, &*ctid, values, isnull) {
        // Update progress
        pg_sys::SpinLockAcquire(std::ptr::addr_of_mut!((*graph).lock));
        (*graph).indtuples += 1.0;
        pg_sys::SpinLockRelease(std::ptr::addr_of_mut!((*graph).lock));
    }
    pg_sys::CurrentMemoryContext = old_ctx;
    pg_sys::MemoryContextReset(build.tmp_ctx.value());
}

// ---------------------------------------------------------------------------
// Flush (hnswbuild.c: CreateGraphPages / WriteNeighborTuples / FlushPages)
// ---------------------------------------------------------------------------

/// `HnswBuildAppendPage`: chain a fresh page onto the graph page list.
unsafe fn build_append_page(
    index: pg_sys::Relation,
    buf: &mut pg_sys::Buffer,
    page: &mut pg_sys::Page,
) {
    // Add a new page
    let newbuf = new_buffer(index);

    // Update previous page
    (*page_opaque(*page)).nextblkno = pg_sys::BufferGetBlockNumber(newbuf);

    // Commit
    pg_sys::MarkBufferDirty(*buf);
    pg_sys::UnlockReleaseBuffer(*buf);

    // Can take a while, so ensure we can interrupt (needs no locks held)
    pg_sys::LockBuffer(newbuf, pg_sys::BUFFER_LOCK_UNLOCK as i32);
    check_for_interrupts!();
    pg_sys::LockBuffer(newbuf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);

    // Prepare new page
    *buf = newbuf;
    *page = pg_sys::BufferGetPage(*buf);
    init_page(*buf, *page);
}

/// `CreateGraphPages` (hnswbuild.c): write element tuples and neighbor-tuple
/// placeholders for the whole in-memory graph, recording each element's
/// on-disk location.
unsafe fn create_graph_pages(build: &mut BuildState) {
    let index = build.index;
    let base = build.base;
    let graph = build.graph_ptr;
    let m = build.m;
    let vec_bytes = build.support.codec.vector_bytes();

    let max_size = max_page_item_size();

    // Allocate once (BLCKSZ each, pgvector's HNSW_TUPLE_ALLOC_SIZE)
    let mut etup = vec![0u8; pg_sys::BLCKSZ as usize];
    let mut ntup = vec![0u8; pg_sys::BLCKSZ as usize];

    // Prepare first page
    let mut buf = new_buffer(index);
    let mut page = pg_sys::BufferGetPage(buf);
    init_page(buf, page);
    let first_block = pg_sys::BufferGetBlockNumber(buf);

    let mut iter = (*graph).head;
    let mut chain_count = 0usize;
    while !crate::access_method::hnswsq2::ptr::is_null(base, iter) {
        let element = crate::access_method::hnswsq2::ptr::access::<Element>(base, iter);

        // Update iterator
        iter = (*element).next;
        chain_count += 1;

        let etup_size = element_tuple_size(vec_bytes);
        let ntup_size = neighbor_tuple_size((*element).level as usize, m);
        let combined_size = etup_size + ntup_size + std::mem::size_of::<pg_sys::ItemIdData>();

        // Initial size check
        if etup_size > pg_sys::BLCKSZ as usize {
            error!("hnswsq2: index tuple too large");
        }

        etup.fill(0);
        set_element_tuple(
            base,
            etup.as_mut_ptr().cast::<ElementTupleData>(),
            element,
            build.precision as u8,
            vec_bytes,
        );

        // Keep element and neighbors on the same page if possible
        if pg_sys::PageGetFreeSpace(page) < etup_size
            || (combined_size <= max_size && pg_sys::PageGetFreeSpace(page) < combined_size)
        {
            build_append_page(index, &mut buf, &mut page);
        }

        // Calculate offsets
        (*element).blkno = pg_sys::BufferGetBlockNumber(buf);
        (*element).offno =
            (PageGetMaxOffsetNumber(page) + 1) as pg_sys::OffsetNumber;
        if combined_size <= max_size {
            (*element).neighbor_page = (*element).blkno;
            (*element).neighbor_offno = (*element).offno + 1;
        } else {
            (*element).neighbor_page = (*element).blkno + 1;
            (*element).neighbor_offno = pg_sys::FirstOffsetNumber;
        }

        let etup_ptr = etup.as_mut_ptr().cast::<ElementTupleData>();
        pgrx::itemptr::item_pointer_set_all(
            &mut (*etup_ptr).neighbortid,
            (*element).neighbor_page,
            (*element).neighbor_offno,
        );

        // Add element
        if pg_sys::PageAddItemExtended(
            page,
            etup.as_mut_ptr().cast(),
            etup_size,
            pg_sys::InvalidOffsetNumber,
            0,
        ) != (*element).offno
        {
            error!("hnswsq2: failed to add index item");
        }

        // Add new page if needed
        if pg_sys::PageGetFreeSpace(page) < ntup_size {
            build_append_page(index, &mut buf, &mut page);
        }

        // Add placeholder for neighbors
        ntup.fill(0);
        if pg_sys::PageAddItemExtended(
            page,
            ntup.as_mut_ptr().cast(),
            ntup_size,
            pg_sys::InvalidOffsetNumber,
            0,
        ) != (*element).neighbor_offno
        {
            error!("hnswsq2: failed to add index item");
        }
    }

    let insert_page = pg_sys::BufferGetBlockNumber(buf);

    // Commit
    pg_sys::MarkBufferDirty(buf);
    pg_sys::UnlockReleaseBuffer(buf);

    let _ = chain_count; // debug aid: chain vs indtuples at flush
    let entry_point =
        crate::access_method::hnswsq2::ptr::access::<Element>(base, (*graph).entry_point);
    let entry_point_opt = if entry_point.is_null() {
        None
    } else {
        Some(entry_point)
    };
    update_meta_page(index, UPDATE_ENTRY_ALWAYS, entry_point_opt, insert_page, true);

    // Record where the graph chain starts (see the module docs).
    let buf = pg_sys::ReadBuffer(index, METAPAGE_BLKNO);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
    let mpage = pg_sys::BufferGetPage(buf);
    let metap = page_get_meta(mpage);
    (*metap).graph_head = first_block;
    pg_sys::MarkBufferDirty(buf);
    pg_sys::UnlockReleaseBuffer(buf);
}

/// `WriteNeighborTuples` (hnswbuild.c): overwrite the placeholders with the
/// real neighbor lists.
unsafe fn write_neighbor_tuples(build: &BuildState) {
    let index = build.index;
    let base = build.base;
    let graph = build.graph_ptr;
    let m = build.m;

    let mut ntup = vec![0u8; pg_sys::BLCKSZ as usize];

    let mut iter = (*graph).head;
    while !crate::access_method::hnswsq2::ptr::is_null(base, iter) {
        let element = crate::access_method::hnswsq2::ptr::access::<Element>(base, iter);
        let ntup_size = neighbor_tuple_size((*element).level as usize, m);

        // Update iterator
        iter = (*element).next;

        // Can take a while, so ensure we can interrupt (no locks held)
        check_for_interrupts!();

        let buf = pg_sys::ReadBuffer(index, (*element).neighbor_page);
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
        let page = pg_sys::BufferGetPage(buf);

        ntup.fill(0);
        set_neighbor_tuple(base, ntup.as_mut_ptr().cast::<NeighborTupleData>(), element, m);

        if !pg_sys::PageIndexTupleOverwrite(
            page,
            (*element).neighbor_offno,
            ntup.as_mut_ptr().cast(),
            ntup_size,
        ) {
            error!("hnswsq2: failed to overwrite neighbor tuple");
        }

        // Commit
        pg_sys::MarkBufferDirty(buf);
        pg_sys::UnlockReleaseBuffer(buf);
    }
}

/// `FlushPages` (hnswbuild.c): materialize the in-memory graph.  The metapage
/// (and the SQ8 calibration chain) are written by the caller before this —
/// see `ambuild` — so this only writes graph pages, the neighbor tuples, and
/// the entry/insert hints.
unsafe fn flush_pages(build: &mut BuildState) {
    create_graph_pages(build);
    write_neighbor_tuples(build);

    (*build.graph_ptr).flushed = true;
    pg_sys::MemoryContextReset(build.graph_ctx.value());
    (*build.graph_ptr).memory_used = 0;
}

// ---------------------------------------------------------------------------
// Parallel build (hnswbuild.c: HnswBeginParallel / HnswParallelBuildMain /
// HnswEndParallel / ParallelHeapScan)
// ---------------------------------------------------------------------------

/// `shm_toc_estimate_chunk` (macro): per-allocation rounding + margin.
unsafe fn estimate_chunk(pcxt: *mut pg_sys::ParallelContext, bytes: usize) {
    let e = &mut (*pcxt).estimator;
    e.space_for_chunks += bytes.div_ceil(8) * 8 + CHUNK_MARGIN;
}

/// `shm_toc_estimate_keys` (macro).
unsafe fn estimate_keys(pcxt: *mut pg_sys::ParallelContext, keys: usize) {
    (*pcxt).estimator.number_of_keys += keys;
}

/// `ParallelTableScanFromHnswShared`.
unsafe fn parallel_scan_from_shared(shared: *mut Shared) -> pg_sys::ParallelTableScanDesc {
    (shared as *mut u8)
        .add(pg_sys::MAXALIGN(std::mem::size_of::<Shared>()))
        .cast()
}

/// `ParallelHeapScan` (hnswbuild.c): the leader waits for all participants.
unsafe fn parallel_heap_scan(build: &mut BuildState) -> f64 {
    let leader = build.leader.as_ref().expect("parallel build has a leader");
    let shared = leader.shared;
    let nparticipanttuplesorts = leader.nparticipanttuplesorts;
    let reltuples;
    loop {
        pg_sys::SpinLockAcquire(std::ptr::addr_of_mut!((*shared).mutex));
        if (*shared).nparticipantsdone == nparticipanttuplesorts {
            build.graph_ptr = &mut (*shared).graph;
            build.base = leader.area;
            reltuples = (*shared).reltuples;
            pg_sys::SpinLockRelease(std::ptr::addr_of_mut!((*shared).mutex));
            break;
        }
        pg_sys::SpinLockRelease(std::ptr::addr_of_mut!((*shared).mutex));

        pg_sys::ConditionVariableSleep(
            std::ptr::addr_of_mut!((*shared).workersdonecv),
            pg_sys::WaitEventIPC::WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN as u32,
        );
    }

    pg_sys::ConditionVariableCancelSleep();

    reltuples
}

/// `HnswParallelScanAndInsert` (hnswbuild.c): a participant's scan of its
/// share of the heap into the shared graph.
unsafe fn parallel_scan_and_insert(
    heap: pg_sys::Relation,
    index: pg_sys::Relation,
    shared: *mut Shared,
    progress: bool,
    build: &mut BuildState,
) -> f64 {
    let index_info = pg_sys::BuildIndexInfo(index);
    let scan = pg_sys::table_beginscan_parallel(heap, parallel_scan_from_shared(shared));

    let ctx = ParallelInsertCtx { build };
    let am = (*heap).rd_tableam;
    let build_range = (*am)
        .index_build_range_scan
        .expect("the table AM has no index_build_range_scan");
    let reltuples = build_range(
        heap,
        index,
        index_info,
        true,   // allow_sync
        progress,
        progress,
        0,
        pg_sys::InvalidBlockNumber,
        Some(parallel_build_callback),
        &ctx as *const ParallelInsertCtx as *mut std::os::raw::c_void,
        scan,
    );

    // Record statistics
    pg_sys::SpinLockAcquire(std::ptr::addr_of_mut!((*shared).mutex));
    (*shared).nparticipantsdone += 1;
    (*shared).reltuples += reltuples;
    pg_sys::SpinLockRelease(std::ptr::addr_of_mut!((*shared).mutex));

    // Notify leader
    pg_sys::ConditionVariableSignal(std::ptr::addr_of_mut!((*shared).workersdonecv));

    reltuples
}

struct ParallelInsertCtx<'a> {
    build: &'a mut BuildState,
}

unsafe extern "C-unwind" fn parallel_build_callback(
    _index: pg_sys::Relation,
    ctid: pg_sys::ItemPointer,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut std::os::raw::c_void,
) {
    let ctx = &mut *(state as *mut ParallelInsertCtx);
    if *isnull {
        return;
    }
    let build = &mut *ctx.build;
    let old_ctx = pg_sys::CurrentMemoryContext;
    pg_sys::CurrentMemoryContext = build.tmp_ctx.value();
    if insert_tuple(build, &*ctid, values, isnull) {
        let graph = build.graph_ptr;
        pg_sys::SpinLockAcquire(std::ptr::addr_of_mut!((*graph).lock));
        (*graph).indtuples += 1.0;
        pg_sys::SpinLockRelease(std::ptr::addr_of_mut!((*graph).lock));
    }
    pg_sys::CurrentMemoryContext = old_ctx;
    pg_sys::MemoryContextReset(build.tmp_ctx.value());
}

/// The entry point PostgreSQL calls in each parallel worker.
///
/// # Safety
/// Called by PostgreSQL only, with the segment it already attached and the
/// toc it built.
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn hnswsq2_parallel_build_main(
    _seg: *mut pg_sys::dsm_segment,
    toc: *mut pg_sys::shm_toc,
) {
    // Look up shared state
    let shared = pg_sys::shm_toc_lookup(toc, PARALLEL_KEY_SHARED, false).cast::<Shared>();
    assert!(!shared.is_null(), "the leader published no shared state");

    // Open relations using lock modes known to be obtained by index.c
    let (heap_lockmode, index_lockmode) = if !(*shared).isconcurrent {
        (
            pg_sys::ShareLock as pg_sys::LOCKMODE,
            pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
        )
    } else {
        (
            pg_sys::ShareUpdateExclusiveLock as pg_sys::LOCKMODE,
            pg_sys::RowExclusiveLock as pg_sys::LOCKMODE,
        )
    };

    let heap = pg_sys::table_open((*shared).heaprelid, heap_lockmode);
    let index = pg_sys::index_open((*shared).indexrelid, index_lockmode);

    let area = pg_sys::shm_toc_lookup(toc, PARALLEL_KEY_AREA, false).cast::<u8>();

    // Worker state: same shape as the leader's, over the shared graph.  The
    // codec comes from the (already written) metapage and calibration chain.
    let support = init_support(index);
    debug_assert_eq!(get_precision(index), support.precision);
    let mut build = init_build_state(
        Some(heap),
        index,
        pg_sys::BuildIndexInfo(index),
        support.codec.clone(),
    );
    build.graph_ptr = &mut (*shared).graph;
    build.base = area;

    // Perform inserts
    parallel_scan_and_insert(heap, index, shared, false, &mut build);

    // Close relations within worker
    pg_sys::index_close(index, index_lockmode);
    pg_sys::table_close(heap, heap_lockmode);

    // Free the worker's state (contexts; the graph lives in the shared area)
    drop(build);
}

/// `HnswEndParallel` (hnswbuild.c).
unsafe fn end_parallel(leader: &Leader) {
    pg_sys::WaitForParallelWorkersToFinish(leader.pcxt);

    // Free last reference to MVCC snapshot, if one was used
    if leader.snapshot_registered {
        pg_sys::UnregisterSnapshot(leader.snapshot);
    }
    pg_sys::DestroyParallelContext(leader.pcxt);
    pg_sys::ExitParallelMode();
}

/// `HnswBeginParallel` (hnswbuild.c): estimate, allocate and launch.
unsafe fn begin_parallel(build: &mut BuildState, isconcurrent: bool, request: i32) {
    // Enter parallel mode and create context
    pg_sys::EnterParallelMode();
    debug_assert!(request > 0);
    let pcxt = pg_sys::CreateParallelContext(
        LIBRARY.as_ptr().cast_mut().cast::<std::os::raw::c_char>(),
        c"hnswsq2_parallel_build_main".as_ptr().cast_mut(),
        request,
    );

    // Get snapshot for table scan
    let snapshot = if !isconcurrent {
        std::ptr::addr_of_mut!(pg_sys::SnapshotAnyData)
    } else {
        pg_sys::RegisterSnapshot(pg_sys::GetTransactionSnapshot())
    };
    let estshared = pg_sys::MAXALIGN(std::mem::size_of::<Shared>())
        + pg_sys::table_parallelscan_estimate(build.heap.unwrap(), snapshot);
    estimate_chunk(pcxt, estshared);

    // Leave space for other objects in shared memory (Docker has a default
    // limit of 64 MB for shm_size, which happens to be the default value of
    // maintenance_work_mem)
    let mut estarea = (pg_sys::maintenance_work_mem as usize).saturating_mul(1024);
    let estother = 3 * 1024 * 1024;
    if estarea > estother {
        estarea -= estother;
    }
    estarea = estarea.min(MAX_GRAPH_MEMORY);
    estimate_chunk(pcxt, estarea);
    estimate_keys(pcxt, 2);

    // Everyone's had a chance to ask for space, so now create the DSM
    pg_sys::InitializeParallelDSM(pcxt);

    // If no DSM segment was available, back out (do serial build)
    if (*pcxt).seg.is_null() {
        if isconcurrent {
            pg_sys::UnregisterSnapshot(snapshot);
        }
        pg_sys::DestroyParallelContext(pcxt);
        pg_sys::ExitParallelMode();
        return;
    }

    // Store shared build state, for which we reserved space
    let shared = pg_sys::shm_toc_allocate((*pcxt).toc, estshared).cast::<Shared>();
    // Initialize immutable state
    (*shared).heaprelid = PgRelation::from_pg(build.heap.unwrap()).oid();
    (*shared).indexrelid = PgRelation::from_pg(build.index).oid();
    (*shared).isconcurrent = isconcurrent;
    pg_sys::ConditionVariableInit(std::ptr::addr_of_mut!((*shared).workersdonecv));
    pg_sys::SpinLockInit(std::ptr::addr_of_mut!((*shared).mutex));
    // Initialize mutable state
    (*shared).nparticipantsdone = 0;
    (*shared).reltuples = 0.0;
    pg_sys::table_parallelscan_initialize(
        build.heap.unwrap(),
        parallel_scan_from_shared(shared),
        snapshot,
    );

    let area = pg_sys::shm_toc_allocate((*pcxt).toc, estarea).cast::<u8>();
    init_graph(&mut (*shared).graph, area, estarea, build.tranche);

    pg_sys::shm_toc_insert((*pcxt).toc, PARALLEL_KEY_SHARED, shared.cast::<std::os::raw::c_void>());
    pg_sys::shm_toc_insert((*pcxt).toc, PARALLEL_KEY_AREA, area.cast::<std::os::raw::c_void>());

    // Launch workers, saving status for leader/caller
    pg_sys::LaunchParallelWorkers(pcxt);
    let mut nparticipanttuplesorts = (*pcxt).nworkers_launched;
    // The leader participates (pgvector does not define
    // DISABLE_LEADER_PARTICIPATION)
    nparticipanttuplesorts += 1;

    // If no workers were successfully launched, back out (do serial build)
    if (*pcxt).nworkers_launched == 0 {
        end_parallel(&Leader {
            pcxt,
            nparticipanttuplesorts,
            shared,
            snapshot,
            snapshot_registered: isconcurrent,
            area,
        });
        return;
    }

    // Log participants (pgvector logs DEBUG1 here; LOG makes the cross-
    // process scaffold verifiable from the server log).
    pgrx::log!(
        "hnswsq2 using {} parallel workers for index build",
        (*pcxt).nworkers_launched
    );

    // Save leader state now that it's clear build will be parallel
    build.leader = Some(Leader {
        pcxt,
        nparticipanttuplesorts,
        shared,
        snapshot,
        snapshot_registered: isconcurrent,
        area,
    });

    // From here on, the leader inserts into the SHARED graph (pgvector's
    // HnswParallelScanAndInsert switches the buildstate over before the
    // leader participates).
    build.graph_ptr = &mut (*shared).graph;
    build.base = area;

    // Join heap scan ourselves
    parallel_scan_and_insert(build.heap.unwrap(), build.index, shared, true, build);

    // Wait for all launched workers
    pg_sys::WaitForParallelWorkersToAttach(pcxt);
}

/// `ComputeParallelWorkers` (hnswbuild.c).  Divergence: the table's
/// `parallel_workers` storage parameter is not consulted (pgrx does not bind
/// `RelationGetParallelWorkers`); `plan_create_index_workers` already caps by
/// `max_parallel_maintenance_workers`, which is what pgvector returns when no
/// table reloption is set.
unsafe fn compute_parallel_workers(heap: pg_sys::Relation, index: pg_sys::Relation) -> i32 {
    // Make sure it's safe to use parallel workers
    let parallel_workers = pg_sys::plan_create_index_workers(
        PgRelation::from_pg(heap).oid(),
        PgRelation::from_pg(index).oid(),
    );
    if parallel_workers == 0 {
        return 0;
    }

    pg_sys::max_parallel_maintenance_workers
}

/// `BuildGraph` (hnswbuild.c): parallel if possible, then scan + flush.
unsafe fn build_graph(build: &mut BuildState) {
    let mut parallel_workers = 0i32;

    // Calculate parallel workers
    if build.heap.is_some() {
        parallel_workers = compute_parallel_workers(build.heap.unwrap(), build.index);
    }

    // Attempt to launch parallel worker scan when required
    if parallel_workers > 0 {
        let isconcurrent = (*build.index_info).ii_Concurrent;
        begin_parallel(build, isconcurrent, parallel_workers);
    }

    // Add tuples to graph
    if build.heap.is_some() {
        if build.leader.is_some() {
            build.reltuples = parallel_heap_scan(build);
        } else {
            pg_sys::IndexBuildHeapScan(
                build.heap.unwrap(),
                build.index,
                build.index_info,
                Some(build_callback),
                build as *mut BuildState as *mut std::os::raw::c_void,
            );
        }

        build.indtuples = (*build.graph_ptr).indtuples;
    }

    // Flush pages
    if !(*build.graph_ptr).flushed {
        flush_pages(build);
    }

    // End parallel build
    if let Some(leader) = build.leader.take() {
        end_parallel(&leader);
    }
}

// ---------------------------------------------------------------------------
// Build state lifecycle (hnswbuild.c: InitBuildState / FreeBuildState)
// ---------------------------------------------------------------------------

/// `InitBuildState` (hnswbuild.c): everything the build needs, from the
/// relation itself (workers run this too — the reloptions/tupdesc/opclass
/// are all readable by them).  The codec is supplied by the caller: the
/// leader builds it from the in-memory SQ8 calibration, workers from the
/// metapage (which exists before they launch).
pub unsafe fn init_build_state(
    heap: Option<pg_sys::Relation>,
    index: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
    codec: Codec,
) -> Box<BuildState> {
    let index_rel = PgRelation::from_pg(index);
    let options = Hnsw2Options::from_relation(&index_rel);
    let precision = options.get_precision();
    let m = options.get_m() as usize;
    let ef_construction = options.get_ef_construction() as usize;

    // Dimensions from the indexed column's typmod (vector(N) → N).
    let atttypmod = index_rel
        .tuple_desc()
        .get(0)
        .map(|attr| attr.atttypmod)
        .unwrap_or(-1);
    if atttypmod < 1 {
        error!(
            "hnswsq2: the indexed column must have a fixed dimension (e.g. vector(128)); \
             got atttypmod {}",
            atttypmod
        );
    }
    let dimensions = atttypmod as usize;
    if dimensions > MAX_DIM {
        error!(
            "hnswsq2: column cannot have more than {} dimensions",
            MAX_DIM
        );
    }
    if ef_construction / 2 < m {
        error!("hnswsq2: ef_construction must be greater than or equal to 2 * m");
    }

    let dist_type = resolve_distance_type(index);

    let graph_ctx = PgMemoryContexts::new("hnswsq2 build graph context");
    let tmp_ctx = PgMemoryContexts::new("hnswsq2 build temporary context");
    let tranche = register_tranche(c"hnswsq2 build");

    let graph = unsafe { std::mem::zeroed::<Graph>() };
    let mut build = Box::new(BuildState {
        heap,
        index,
        index_info,
        m,
        ef_construction,
        dimensions,
        support: Support {
            dist_type,
            precision,
            codec,
        },
        precision,
        reltuples: 0.0,
        indtuples: 0.0,
        graph,
        graph_ptr: std::ptr::null_mut(),
        base: std::ptr::null_mut(),
        ml: get_ml(m),
        max_level: get_max_level(m),
        graph_ctx,
        tmp_ctx,
        tranche,
        seed: HNSW2_BUILD_SEED.get(),
        leader: None,
        scratch: SearchScratch::new(m),
        decode: Vec::with_capacity(dimensions),
        pair_scratch: Vec::with_capacity(dimensions),
        visited: Visited::new(ef_construction * m * 2),
        encoded: Vec::with_capacity(dimensions * precision.elem_bytes()),
        vector: Vec::with_capacity(dimensions),
    });
    let graph_ptr: *mut Graph = &mut build.graph;
    build.graph_ptr = graph_ptr;
    init_graph(
        graph_ptr,
        std::ptr::null_mut(),
        (pg_sys::maintenance_work_mem as usize).saturating_mul(1024),
        build.tranche,
    );

    build
}

/// The SQ8 calibration reservoir pass (the old engine's sampler, reused: the
/// sample is what the codec's min/max is trained on).
struct SampleState {
    sample: Vec<Vec<f32>>,
    sample_size: usize,
    nrows: usize,
    distance_type: DistanceType,
    rng: rand::rngs::SmallRng,
}

unsafe extern "C-unwind" fn sample_callback(
    _index: pg_sys::Relation,
    _ctid: pg_sys::ItemPointer,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut std::os::raw::c_void,
) {
    use rand::Rng;
    let state = &mut *(state as *mut SampleState);
    if *isnull {
        return;
    }
    let detoasted = pg_sys::pg_detoast_datum_copy((*values).cast_mut_ptr());
    let pg_vec = detoasted.cast::<PgVectorInternal>();
    let mut vec = (*pg_vec).to_slice().to_vec();
    pg_sys::pfree(detoasted.cast());
    if state.distance_type == DistanceType::Cosine {
        preprocess_cosine(&mut vec);
    }

    // Reservoir sampling of size sample_size.
    if state.sample.len() < state.sample_size {
        state.sample.push(vec);
    } else {
        let j: usize = state.rng.gen_range(0..state.nrows + 1);
        if j < state.sample_size {
            state.sample[j] = vec;
        }
    }
    state.nrows += 1;
}

// ---------------------------------------------------------------------------
// ambuild / ambuildempty (hnswbuild.c: BuildIndex / hnswbuild / hnswbuildempty)
// ---------------------------------------------------------------------------


/// pgvector's `RelationNeedsWAL`: permanent relations need WAL.
unsafe fn relation_needs_wal(index: pg_sys::Relation) -> bool {
    (*(*index).rd_rel).relpersistence == pg_sys::RELPERSISTENCE_PERMANENT as std::os::raw::c_char
}

/// `BuildIndex` (hnswbuild.c).
unsafe fn build_index(
    heap: Option<pg_sys::Relation>,
    index: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> (f64, f64) {
    let index_rel = PgRelation::from_pg(index);
    let options = Hnsw2Options::from_relation(&index_rel);
    let precision = options.get_precision();
    let m = options.get_m() as usize;
    let ef_construction = options.get_ef_construction() as usize;
    let dimensions = index_rel
        .tuple_desc()
        .get(0)
        .map(|attr| attr.atttypmod)
        .unwrap_or(-1) as usize;

    // SQ8: reservoir-sample + train BEFORE the build.  The metapage must
    // claim block 0 before the calibration chain writes (the old engine's
    // ordering lesson), and the chain must exist before parallel workers
    // start (they read the codec from it) — so: train in memory, write the
    // metapage, write the chain, record its pointer.
    let codec = if precision.needs_calibration() {
        let sample_size = get_sample_size(index);
        let seed = HNSW2_BUILD_SEED.get();
        let rng = if seed < 0 {
            rand::rngs::SmallRng::from_entropy()
        } else {
            rand::rngs::SmallRng::seed_from_u64(seed as u64)
        };
        let mut sample_state = SampleState {
            sample: Vec::with_capacity(sample_size),
            sample_size,
            nrows: 0,
            distance_type: resolve_distance_type(index),
            rng,
        };
        if let Some(heap) = heap {
            pg_sys::IndexBuildHeapScan(
                heap,
                index,
                index_info,
                Some(sample_callback),
                &mut sample_state as *mut SampleState as *mut std::os::raw::c_void,
            );
        }
        let calib = Sq8Calibration::train(&sample_state.sample, dimensions);

        create_meta_page(
            index,
            dimensions,
            m,
            ef_construction,
            precision,
            ItemPointer::new_invalid(),
        );
        let ptr = calib.store(&index_rel);
        set_meta_calibration(index, ptr);
        Codec::new_sq8(&calib)
    } else {
        create_meta_page(
            index,
            dimensions,
            m,
            ef_construction,
            precision,
            ItemPointer::new_invalid(),
        );
        Codec::new(precision, dimensions)
    };

    let mut build = init_build_state(heap, index, index_info, codec);

    build_graph(&mut build);

    if relation_needs_wal(index) {
        pg_sys::log_newpage_range(
            index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            0,
            pg_sys::RelationGetNumberOfBlocksInFork(index, pg_sys::ForkNumber::MAIN_FORKNUM),
            true,
        );
    }

    (build.reltuples, build.indtuples)
}

/// `hnswbuild` (hnswbuild.c).
#[pg_guard]
pub unsafe extern "C-unwind" fn ambuild(
    heap: pg_sys::Relation,
    index: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    let (reltuples, indtuples) = build_index(Some(heap), index, index_info);

    let result = pg_sys::palloc(std::mem::size_of::<pg_sys::IndexBuildResult>())
        .cast::<pg_sys::IndexBuildResult>();
    (*result).heap_tuples = reltuples;
    (*result).index_tuples = indtuples;
    result
}

/// `hnswbuildempty` (hnswbuild.c): an unlogged/empty index gets just the
/// metapage (and a provisional SQ8 calibration).
#[pg_guard]
pub unsafe extern "C-unwind" fn ambuildempty(index: pg_sys::Relation) {
    let index_info = pg_sys::BuildIndexInfo(index);
    let _ = build_index(None, index, index_info);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levels_and_sizes() {
        assert!(get_max_level(16) >= 16);
        assert!((get_ml(16) - 1.0 / 16f64.ln()).abs() < 1e-12);
        let et = element_tuple_size(512);
        let nt = neighbor_tuple_size(2, 16);
        assert!(et + nt + 4 <= max_page_item_size());
    }
}
