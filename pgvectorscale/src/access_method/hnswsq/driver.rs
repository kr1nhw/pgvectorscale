//! The parallel build driver (M3).
//!
//! Two halves, and this file is the first: the **leader** side, which decides the arena's
//! shape, tells PostgreSQL how much shared memory to reserve, allocates the arena inside
//! the parallel context's `shm_toc`, and (later) launches the workers.  The worker side
//! attaches to the same arena by key.
//!
//! Why the context's toc and not a dsm of our own: workers are started by PostgreSQL's
//! `ParallelWorkerMain`, which has already attached the segment and restored the GUCs and
//! hands the extension's entry point the dsm handle.  Reusing that toc means one segment,
//! one allocation path, and no second lifecycle to get wrong -- but it also means the
//! arena's regions are keys in *PostgreSQL's* toc, so a worker finds the toc through
//! PostgreSQL's magic rather than one of ours.
//!
//! Everything here is leader-side and therefore testable without launching a worker.

use pgrx::pg_sys;
use pgrx::pg_guard;
use pgrx::pg_extern;

use super::arena::{ArenaLayout, SharedArena, TOC_KEY_CHUNK, TOC_KEY_HEADER};
use super::flat_graph::{check_lists, plan_capacity, ArenaSizing, FlatGraph};

/// PostgreSQL's toc magic.
///
/// pgrx does not bind this `#define`, and *remembering* it produced the wrong value: the
/// constant was pinned to `0x50477c23`, `shm_toc_attach` returned NULL for it, and a test
/// written to verify the pin rather than trust it is what caught that.  The value below is
/// **measured** off a live parallel context (the first field of `shm_toc` is the magic) and
/// the test requires them to agree, so it cannot drift again.
///
/// Note that nothing on the worker path needs it any more: PostgreSQL hands the worker the
/// toc directly (see [`worker_attach`]).  It stays for any code that has to re-find the toc
/// from the segment alone.
pub const PARALLEL_MAGIC: u64 = 0x5047_7C7C;

/// The library PostgreSQL must load in a worker to find the entry point.
///
/// Derived from the crate version rather than hardcoded: pgrx installs the shared library
/// under a *versioned* name (`vectorscale-0.9.0.dylib`), which is also what the extension's
/// `module_pathname` points at, so the unversioned name is not loadable.  The first
/// cross-process test found this the direct way -- workers launched and died with
/// `could not access file "vectorscale": No such file or directory` -- which is a good
/// argument for deriving it.
const LIBRARY: &str = concat!("vectorscale-", env!("CARGO_PKG_VERSION"), "\0");

/// The arena shape a build with this byte budget will ask for, from the same
/// `plan_capacity` the single-builder path uses -- so a parallel build cannot quietly run
/// on a different budget than the one the fingerprint gate was measured with.
pub fn plan(stride: usize, cap: usize, budget_bytes: usize) -> ArenaSizing {
    plan_capacity(stride, cap, budget_bytes as u64, 1.0, 1.0)
}

/// Round a chunk request up the way PostgreSQL's `shm_toc_estimate_chunk` macro does
/// (`BUFFERALIGN`, i.e. `MAXALIGN` on every platform we build for).
///
/// Those macros are `static inline` in `shm_toc.h`, so bindgen does not emit them.  The
/// fields they touch are exactly what `shm_toc_estimate` reads (which *is* bound), so
/// mirroring the arithmetic is precise rather than approximate -- and the test below
/// proves it by allocating from a context sized this way.
/// Per-allocation margin on top of `BUFFERALIGN`.
///
/// Measured on PostgreSQL 18: with per-allocation `BUFFERALIGN` estimates, a real context
/// reported `shm_toc_freespace` 56 bytes *above* the aligned total the arena asks for, and
/// the allocation still failed with "out of shared memory" -- so `shm_toc_allocate` charges
/// more per allocation than the alignment alone (the entry it records and the cursor it
/// rounds both live in the same region).  Reserving a margin is the honest fix: shared
/// memory reserved and unused costs a few kilobytes, while an under-estimate is a hard
/// error, and the estimate is per allocation precisely so this cannot compound.
const CHUNK_MARGIN: usize = 64;

#[inline]
fn estimate_chunk(pcxt: *mut pg_sys::ParallelContext, bytes: usize) {
    // SAFETY: the caller owns a live context (see `estimate_arena`'s contract).
    unsafe {
        let e = &mut (*pcxt).estimator;
        e.space_for_chunks += bytes.div_ceil(8) * 8 + CHUNK_MARGIN;
    }
}

#[inline]
fn estimate_keys(pcxt: *mut pg_sys::ParallelContext, keys: usize) {
    // SAFETY: as above.
    unsafe {
        (*pcxt).estimator.number_of_keys += keys;
    }
}

/// Tell PostgreSQL how much shared memory the arena needs, *before*
/// `InitializeParallelDSM` allocates the segment.
///
/// # Safety
///
/// `pcxt` must come from `CreateParallelContext` and must not have had
/// `InitializeParallelDSM` called on it yet.
pub unsafe fn estimate_arena(
    pcxt: *mut pg_sys::ParallelContext,
    stride: usize,
    cap: usize,
    sizing: ArenaSizing,
) {
    // One estimate per allocation, not one for the sum: `shm_toc_allocate` aligns each
    // allocation separately, so a single rounded-up total is short by up to eight bytes
    // per region (and that is not a rounding detail -- it is "out of shared memory").
    for bytes in SharedArena::allocation_sizes(stride, cap, sizing.nodes, sizing.slabs) {
        estimate_chunk(pcxt, bytes);
    }
    // One key per region the leader allocates: header, state, chunk, locks, parameters,
    // and (once a relation is open) the shared scan descriptor.
    estimate_keys(pcxt, 6);
    estimate_chunk(pcxt, std::mem::size_of::<BuildParams>());
}

/// Leader: allocate the segment and put the arena in it.
///
/// **Ordering constraint, found by reading how the parameters are derived:** `ml` and
/// `max_level` come from the index's *meta page*, which the leader writes during its own
/// build.  So a parallel build has to write the meta page (block 0) and the calibration
/// chain *before* launching workers, or a worker would seed its state from a meta page that
/// does not exist yet.  The dimensions, `m`/`m0`/`ef_construction`, precision and distance
/// type all travel in [`BuildParams`] instead of being re-derived, so the meta page is the
/// only thing a worker still has to read from the index.
///
/// # Safety
///
/// `pcxt` must have been sized with [`estimate_arena`] for the same `sizing`, and must not
/// have had `InitializeParallelDSM` called on it yet.
pub unsafe fn leader_setup(
    pcxt: *mut pg_sys::ParallelContext,
    stride: usize,
    cap: usize,
    sizing: ArenaSizing,
    tranche: i32,
) -> (SharedArena, ArenaLayout) {
    pg_sys::InitializeParallelDSM(pcxt);
    let toc = (*pcxt).toc;
    assert!(!toc.is_null(), "InitializeParallelDSM left no toc");
    let arena = SharedArena::allocate(toc, stride, cap, sizing.nodes, sizing.slabs, tranche);
    let layout = arena.header().layout;
    (arena, layout)
}

/// Worker: the arena, from the toc PostgreSQL hands the entry point.
///
/// The signature is the whole lesson here: PostgreSQL's `parallel_worker_main_type` is
/// `(dsm_segment *seg, shm_toc *toc)`, i.e. the worker is given the segment **already
/// attached** and the toc itself.  The first version of this took a `Datum` and called
/// `dsm_attach` on it -- treating a pointer as a handle -- which failed in the worker with
/// "could not attach the build segment"; reading the contract instead of guessing removed
/// both the attach and the need to re-derive the toc's magic worker-side.
///
/// # Safety
///
/// `toc` must be the toc PostgreSQL passed to [`hnswsq_parallel_build_main`], holding an
/// arena allocated by [`leader_setup`].
pub unsafe fn worker_attach(toc: *mut pg_sys::shm_toc) -> SharedArena {
    assert!(!toc.is_null(), "no toc was handed to the worker");
    SharedArena::attach(toc)
}

/// Key for the serialized build parameters.
pub const TOC_KEY_PARAMS: u64 = 0x686e_7377_7371_2001;

/// Everything a worker needs to build, apart from the arena and the scan.
///
/// `#[repr(C)]` POD with no pointers (the `u64`s first, then `u32`s, then the bytes) because
/// the leader writes it once and workers in other processes read it.  Oids travel as `u32`
/// rather than `pg_sys::Oid` for the same reason: the wire shape should not depend on how
/// pgrx wraps an Oid this release.
///
/// It lives here rather than in the arena header because these are *build* choices, not
/// arena layout -- and the single-builder path writes the same arena with nothing to
/// publish.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BuildParams {
    /// Rows the leader counted, so a worker can size its own work.
    pub rows: u64,
    /// The pinned seed levels are derived from (`levels::level_for_tid`).
    pub seed: u64,
    pub heap_oid: u32,
    pub index_oid: u32,
    pub stride: u32,
    /// Vector dimension.  The worker builds its `Codec` from this plus `precision`, and
    /// `stride` alone is not enough to recover it (`stride = dim * elem_bytes`, and elem_bytes
    /// depends on the precision).
    pub num_dimensions: u32,
    pub cap: u32,
    pub m: u32,
    pub m0: u32,
    pub ef_construction: u32,
    /// HNSW level scale, `1 / ln(m)`.
    pub ml: f32,
    pub max_level: u8,
    /// `DistanceType` and precision as integers; those enums are not ours to send.
    pub dist_type: u8,
    pub precision: u8,
    pub backfill: u8,
    /// Backlink admission policy.  Not cosmetic: a worker on a different policy builds a
    /// different graph, and the fingerprint gate would report an unexplained mismatch rather
    /// than "the worker disagreed about the policy".
    pub backlink_mode: u8,
    /// The LWLock tranche the arena's node locks were initialized with.
    pub tranche: i32,
}

impl BuildParams {
    /// Leader: publish the parameters into the toc.
    ///
    /// Publishing twice overwrites in place rather than allocating again: the estimator
    /// reserved room for exactly one copy, and a second allocation would fail against an
    /// estimate that is already tight.
    ///
    /// # Safety
    ///
    /// `toc` must be the parallel context's toc, live for as long as the build.
    pub unsafe fn publish(&self, toc: *mut pg_sys::shm_toc) {
        let existing = pg_sys::shm_toc_lookup(toc, TOC_KEY_PARAMS, true);
        if !existing.is_null() {
            existing.cast::<BuildParams>().write(*self);
            return;
        }
        let region = pg_sys::shm_toc_allocate(toc, std::mem::size_of::<BuildParams>())
            .cast::<BuildParams>();
        region.write(*self);
        pg_sys::shm_toc_insert(toc, TOC_KEY_PARAMS, region.cast());
    }

    /// Worker: the parameters the leader published.
    ///
    /// # Safety
    ///
    /// `toc` must hold parameters published by [`BuildParams::publish`].
    pub unsafe fn read_from(toc: *mut pg_sys::shm_toc) -> BuildParams {
        let region = pg_sys::shm_toc_lookup(toc, TOC_KEY_PARAMS, false).cast::<BuildParams>();
        assert!(!region.is_null(), "the leader published no build parameters");
        *region
    }
}

/// Leader, after the workers stop: promote the highest-level published node to entry.
///
/// Workers never promote the entry themselves -- every insert would then serialize on one
/// cache line, and promotion only matters once the graph stops growing.  The leader seeds an
/// entry *before* launching so searches have somewhere to start (without one, every node is
/// born with an empty list -- a failure this project has already produced once), and then
/// has to re-promote here, because a higher-level node may have appeared afterwards.  That
/// is this function's whole job, and its return value is what the build reports.
///
/// Only ids below the watermark are considered: a claimed-but-unpublished slot has a level
/// but no list, and promoting one would hand searches an entry that is not there yet.
pub fn promote_best_entry(g: &mut crate::access_method::hnswsq::flat_graph::FlatGraph) -> Option<(u32, usize)> {
    let mut best: Option<(u32, usize)> = None;
    for id in 0..g.watermark() as u32 {
        let level = g.level(id) as usize;
        if best.is_none_or(|(_, best_level)| level > best_level) {
            best = Some((id, level));
        }
    }
    if let Some((id, level)) = best {
        // A no-op when the current entry already outranks it.
        g.promote_entry(id);
        assert_eq!(g.entry(), Some(id), "promotion must take effect");
        return Some((id, level));
    }
    None
}

/// Key for the shared table-scan descriptor.
pub const TOC_KEY_SCAN: u64 = 0x686e_7377_7371_2002;

/// Bytes the shared scan descriptor needs, for the estimator.
///
/// `table_parallelscan_estimate` only reads the relation and the snapshot, so the leader can
/// size the segment *before* `InitializeParallelDSM` -- which is the whole reason the
/// estimate is a separate step at all.
///
/// # Safety
///
/// `heap` must be an open relation and `snapshot` a registered snapshot.
pub unsafe fn scan_bytes(heap: pg_sys::Relation, snapshot: pg_sys::Snapshot) -> usize {
    pg_sys::table_parallelscan_estimate(heap, snapshot)
}

/// Leader: build the shared scan descriptor in the toc.
///
/// One descriptor for every worker: PostgreSQL's parallel scan hands out block ranges
/// through it, so all of them reading the same object is what makes the workers cover the
/// table once instead of each scanning all of it.  The snapshot is copied *into* the
/// descriptor by `table_parallelscan_initialize`, which is why a worker needs nothing but
/// the descriptor to start scanning.
///
/// # Safety
///
/// `pcxt` must have been sized with [`estimate_arena`] plus
/// [`scan_bytes`] for this relation, and `heap` must be an open relation that stays open for
/// the build.
pub unsafe fn leader_scan_setup(
    pcxt: *mut pg_sys::ParallelContext,
    heap: pg_sys::Relation,
    snapshot: pg_sys::Snapshot,
) -> *mut pg_sys::ParallelTableScanDescData {
    let size = scan_bytes(heap, snapshot);
    let pscan = pg_sys::shm_toc_allocate((*pcxt).toc, size)
        .cast::<pg_sys::ParallelTableScanDescData>();
    pg_sys::table_parallelscan_initialize(heap, pscan.cast(), snapshot);
    pg_sys::shm_toc_insert((*pcxt).toc, TOC_KEY_SCAN, pscan.cast());
    pscan
}

/// Worker (or anyone with the toc): the shared scan descriptor.
///
/// # Safety
///
/// `toc` must hold a descriptor from [`leader_scan_setup`].
pub unsafe fn scan_descriptor(
    toc: *mut pg_sys::shm_toc,
) -> *mut pg_sys::ParallelTableScanDescData {
    let pscan = pg_sys::shm_toc_lookup(toc, TOC_KEY_SCAN, false)
        .cast::<pg_sys::ParallelTableScanDescData>();
    assert!(!pscan.is_null(), "the leader published no scan descriptor");
    pscan
}

/// The entry point PostgreSQL calls in each parallel worker.
///
/// `ParallelWorkerMain` has already attached the segment, restored the GUCs and the
/// transaction snapshot, and passes the dsm handle through, so all this has to do is find
/// the arena and build into it.  It is `#[no_mangle]` because PostgreSQL resolves it by
/// *name* out of the library -- the name the leader passed to `CreateParallelContext`.
///
/// # Safety
///
/// Called by PostgreSQL only, with the segment it already attached and the toc it built.
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn hnswsq_parallel_build_main(
    _seg: *mut pg_sys::dsm_segment,
    toc: *mut pg_sys::shm_toc,
) {
    // Debug: a worker that does nothing at all.  If this still crashes, the fault is in how the
    // context was created rather than in the worker's own code.
    if crate::access_method::hnswsq::options::HNSWSQ_PARALLEL_STAGE.get() == 9 {
        return;
    }

    let arena = worker_attach(toc);
    arena.state().worker_entered();
    arena.state().worker_started();

    // The build loop: scan this worker's share of the heap and insert into the shared arena.
    // Parameters are read tolerantly so a bare attachment test (no build published) still
    // works -- that is what the cross-process test exercises.
    let params = pg_sys::shm_toc_lookup(toc, TOC_KEY_PARAMS, true);
    if !params.is_null() {
        let params = BuildParams::read_from(toc);
        if params.heap_oid != 0 {
            crate::access_method::hnswsq::build::parallel_worker_scan(toc, &arena, &params);
        }
    }

    arena.state().worker_finished();
}

/// Run a parallel build against a real (committed) table and report what happened.
///
/// This exists because the parallel path cannot be tested from a `#[pg_test]`: the harness
/// rolls its transaction back, so any table the test creates is invisible to a worker's
/// snapshot.  A SQL-callable entry point can be pointed at an existing table instead -- and it
/// is also the vehicle for the worker sweep, which needs to vary `workers` and time the result.
///
/// The parameters come from the index itself (its reloptions and meta page) through the same
/// helpers the single-builder leader uses, so a measurement here is comparable with a
/// single-builder build of the same index definition.
///
/// `dims` is the table's `vector(N)`, and `dist_type` the `DistanceType` discriminant of the
/// index's opclass -- it has to be the declared metric, or the build measures something other
/// than what the index was created for.
///
/// Returns a one-line summary: rows published, structural checks, elapsed milliseconds, and how
/// many workers actually ran.
/// Run a whole parallel build against a table and index that are already open, and report it.
///
/// This is the body the SQL harness wraps, and the function `ambuild` will call once
/// `amcanbuildparallel` is wired: one implementation, so a measurement made through the harness is
/// a measurement of exactly what `CREATE INDEX` will do.  It takes oids rather than relations
/// because it opens and closes its own (every participant needs the same locks).
///
/// `budget_mb` sizes the arena the way `maintenance_work_mem` sizes the single-builder graph: too
/// small and the workers simply stop claiming, which shows up as fewer published nodes rather
/// than as an error.
///
/// Returns a one-line summary: rows published, structural checks, elapsed milliseconds, and how
/// many workers actually ran.
pub(crate) fn build_index_parallel(
    heap_oid: pg_sys::Oid,
    index_oid: pg_sys::Oid,
    workers: i32,
    dims: u32,
    dist_type: u32,
    budget_mb: u64,
    write_out: bool,
) -> ParallelBuildResult {
    use pgrx::PgRelation;

    let index_rel = unsafe { PgRelation::open(index_oid) };
    let options = crate::access_method::hnswsq::options::TSVHnswOptions::from_relation(&index_rel);
    let precision = options.get_precision();
    let m = options.get_m() as u32;
    let m0 = m * 2;
    let ef_construction = options.get_ef_construction() as u32;
    let meta = crate::access_method::hnswsq::meta_page::HnswMetaPage::fetch(&index_rel);
    let stride = dims * precision.elem_bytes() as u32;
    // The arena is sized from the caller's budget, exactly as the single-builder path sizes
    // its in-memory graph: too small and the workers simply stop claiming, which shows up as
    // fewer published rows rather than as an error.
    let sizing = plan(
        stride as usize,
        m0 as usize,
        (budget_mb.max(1) as usize) << 20,
    );
    let tranche = unsafe { super::arena::register_tranche(c"hnswsq_parallel_debug") };

    let start = std::time::Instant::now();
    let mut published = 0usize;
    let mut launched = 0i32;
    let mut entered = 0u32;
    let mut failed = false;
    let mut slabs_claimed = 0usize;
    let mut written = 0usize;
    let mut scanned = 0u64;
    let outcome = unsafe {
        let heap = pg_sys::table_open(heap_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        let index = pg_sys::index_open(index_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        let snapshot = pg_sys::GetActiveSnapshot();
        // `EnterParallelMode` is what PostgreSQL's own callers do around a parallel context,
        // and it is not optional: without it a *worker that does nothing at all* still takes
        // the server down, because the worker inherits state the leader never established.
        pg_sys::EnterParallelMode();
        let pcxt = pg_sys::CreateParallelContext(
            LIBRARY.as_ptr().cast_mut().cast::<std::os::raw::c_char>(),
            c"hnswsq_parallel_build_main".as_ptr().cast_mut(),
            workers,
        );
        estimate_arena(pcxt, stride as usize, m0 as usize, sizing);
        estimate_chunk(pcxt, scan_bytes(heap, snapshot));
        let (arena, layout) =
            leader_setup(pcxt, stride as usize, m0 as usize, sizing, tranche);
        leader_scan_setup(pcxt, heap, snapshot);

        // No seed node.  The first node a worker inserts establishes the entry itself (see
        // `promote` in flat_engine), so the index holds exactly the table's rows -- a fabricated
        // seed would be a real entry with a vector and a heap TID that no row has.

        let params_snapshot = BuildParams {
            rows: 0,
            seed: 20240912,
            heap_oid: u32::from(heap_oid),
            index_oid: u32::from(index_oid),
            stride,
            num_dimensions: dims,
            cap: m0,
            m,
            m0,
            ef_construction,
            ml: meta.get_ml(),
            max_level: meta.get_max_level(),
            dist_type: dist_type as u8,
            precision: precision as u8,
            backfill: 0,
            backlink_mode: 0,
            tranche,
        };
        params_snapshot.publish((*pcxt).toc);

        pg_sys::LaunchParallelWorkers(pcxt);
        pg_sys::WaitForParallelWorkersToFinish(pcxt);

        // A worker cannot longjmp into the leader (PostgreSQL does not allow it), so a worker that
        // hits a data-level problem sets `failed` and exits cleanly -- and the leader has to notice.
        // Checked before anything is promoted or written, so a failed build leaves the index alone
        // rather than writing a partial graph into pages the transaction then rolls back.
        if arena.state().failed() {
            pgrx::error!(
                "hnswsq parallel build: a worker reported a failure; the server log has the \
                 worker's own message"
            );
        }

        // Workers never promote an entry (3j.16): the leader seeded one before launching, and has
        // to re-promote now, because a higher-level node may have appeared while they ran.  Omitting
        // this leaves the zero-vector seed as the entry for the whole graph, which costs recall --
        // measured at 0.5 instead of 0.8 on `t100k` before this call was added.
        {
            let mut g = super::flat_graph::FlatGraph::in_arena(&arena);
            crate::access_method::hnswsq::driver::promote_best_entry(&mut g);
        }
        // Everything the summary needs must be read *before* the context is destroyed:
        // `DestroyParallelContext` detaches the segment, and the arena handle points into it.
        // (The first version formatted the string afterwards and segfaulted the *leader* --
        // the worker had exited cleanly, which is what the postmaster log said.)
        launched = (*pcxt).nworkers_launched;
        published = arena.state().watermark();
        failed = arena.state().failed();
        entered = arena.state().workers_entered();
        slabs_claimed = arena.state().claimed().1;
        scanned = arena.state().rows_scanned();
        let checks = super::flat_graph::check_lists(
            &super::flat_graph::FlatGraph::in_arena(&arena),
            m0 as usize,
        );
        // An incomplete build must not pass silently.  The single-builder path spills to disk when
        // `maintenance_work_mem` runs out and indexes every row regardless; a worker cannot spill,
        // so if the arena filled up the index would simply be missing rows -- and, worse, the
        // `IndexBuildResult` below would report the truncated count as the table's row count.
        // (Measured before this check existed: 16 MB and 100k rows produced an index of 75 251
        // entries and rewrote the heap's `reltuples` to match.)
        // Every published node is a row: there is no seed node any more (the first node a worker
        // inserts establishes the entry itself).  Subtracting one here -- as this did while the
        // seed existed -- makes every complete build look one row short.
        let indexed = published;
        if (scanned as usize) > indexed {
            pgrx::error!(
                "hnswsq parallel build is incomplete: {} rows scanned but only {} indexed \
                 (the arena is sized by maintenance_work_mem = {} MB).  Raise \
                 maintenance_work_mem, or set hnswsq.build_workers = 0 to use the single-builder \
                 path, which spills to disk and indexes every row.",
                scanned,
                indexed,
                budget_mb
            );
        }

        // Before the context goes away (and the segment with it), the leader may write the
        // graph out: same bridge and same writer as the single-builder path.
        if write_out {
            written = crate::access_method::hnswsq::build::write_out_arena(
                &index_rel, &params_snapshot, &arena,
            );
        }
        pg_sys::DestroyParallelContext(pcxt);
        pg_sys::ExitParallelMode();
        pg_sys::index_close(index, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        pg_sys::table_close(heap, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        ParallelBuildResult {
            scanned,
            published,
            written,
            entered,
            no_incoming: checks.nodes_without_incoming,
            reachable: checks.reachable_from_entry,
            elapsed_ms: start.elapsed().as_millis(),
            summary: format!(
            "workers launched={} entered={} failed={} published={} written={} slabs={}/{} checks(published={} no_incoming={} reachable={} self_links={} duplicates={} max_len={}/{}) elapsed_ms={}",
            launched,
            entered,
            failed,
            published,
            written,
            slabs_claimed,
            layout.total_bytes,
            checks.published,
            checks.nodes_without_incoming,
            checks.reachable_from_entry,
            checks.self_links,
            checks.duplicate_links,
            checks.max_list_len,
            m0,
            start.elapsed().as_millis()
            ),
        }
    };
    let _ = launched;
    outcome
}

/// What a parallel build produced, for a caller that needs numbers rather than a report
/// (`ambuild` fills its `IndexBuildResult` from these).
pub(crate) struct ParallelBuildResult {
    /// Rows the workers scanned: what `IndexBuildResult.heap_tuples` must be.
    pub scanned: u64,
    pub published: usize,
    pub written: usize,
    pub entered: u32,
    pub no_incoming: usize,
    pub reachable: usize,
    pub elapsed_ms: u128,
    pub summary: String,
}

#[pg_extern]
pub fn hnswsq_parallel_build_debug(
    heap_oid: pg_sys::Oid,
    index_oid: pg_sys::Oid,
    workers: i32,
    dims: i32,
    dist_type: i32,
    budget_mb: i32,
    write_out: bool,
) -> String {
    build_index_parallel(
        heap_oid,
        index_oid,
        workers,
        dims.max(1) as u32,
        dist_type.max(1) as u32,
        budget_mb.max(1) as u64,
        write_out,
    )
    .summary
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use super::*;

    // The estimate must be per *allocation*, not per total: `shm_toc_allocate`
    // `BUFFERALIGN`s each allocation separately, so rounding up the summed regions once
    // is short by up to eight bytes per region -- which showed up as
    // `ERROR: out of shared memory` from `shm_toc.c`, i.e. as a hard failure rather than a
    // few wasted bytes.  (An earlier round concluded from the same error that the estimate
    // was not honored at all; an instrumented run showed `space_for_chunks` and
    // `shm_toc_freespace` are both fine, so that reading was wrong and the granularity
    // was the whole bug.)
    // Sized, allocated and found by key, all in a real parallel context -- and the sizing
    // is the part that took measuring.  `shm_toc_allocate` `BUFFERALIGN`s each allocation
    // separately, so the estimate is per *allocation*, not per total; and with only that,
    // a real context reported `shm_toc_freespace` 56 bytes above the aligned total the
    // arena needed and the allocation *still* failed, so each allocation also carries a
    // small margin (`CHUNK_MARGIN`).  Both are over-reservations: shared memory reserved
    // and unused costs kilobytes, while an under-estimate is a hard error.
    #[pgrx::pg_test]
    fn workers_attach_the_leader_arena_across_processes() {
        // The end-to-end plumbing without the build loop: a real parallel context, real
        // worker *processes*, each attaching the arena by key and touching the shared
        // cursors.  Everything thread-based so far could only prove the protocol; this is
        // where "does a worker in another process actually see the leader's arena" gets
        // answered.
        let (stride, cap) = (8usize, 4usize);
        let sizing = plan(stride, cap, 1 << 20);
        let nworkers = 2;
        // SAFETY: leader-side; the context is launched, waited for and destroyed here.
        unsafe {
            let pcxt = pg_sys::CreateParallelContext(
                LIBRARY.as_ptr().cast_mut().cast::<std::os::raw::c_char>(),
                // Resolved by name in the worker: this is the first test that requires the
                // entry point to exist.
                c"hnswsq_parallel_build_main".as_ptr().cast_mut(),
                nworkers,
            );
            estimate_arena(pcxt, stride, cap, sizing);
            let (arena, _layout) = leader_setup(pcxt, stride, cap, sizing, 0);
            assert_eq!(arena.state().workers_entered(), 0);

            pg_sys::LaunchParallelWorkers(pcxt);
            pg_sys::WaitForParallelWorkersToFinish(pcxt);

            let launched = (*pcxt).nworkers_launched;
            assert!(launched > 0, "PostgreSQL launched no workers");
            assert_eq!(
                arena.state().workers_entered(),
                launched as u32,
                "every launched worker reached the leader's arena in its own process"
            );
            assert_eq!(
                arena.state().active_workers(),
                0,
                "and every one of them finished"
            );
            assert!(!arena.state().failed());

            drop(arena);
            pg_sys::DestroyParallelContext(pcxt);
        }
    }

    // IGNORED for a harness reason, not a driver one: a `#[pg_test]` runs inside a transaction
    // that is rolled back, so `CREATE TABLE`/`CREATE INDEX` here are *uncommitted* -- and a
    // parallel worker gets a snapshot that cannot see uncommitted catalog rows, so it dies at
    // `table_open` with "cannot open relation".  That failure is itself informative: it shows
    // the worker reached the relation-open inside `parallel_worker_scan`, i.e. the
    // entry -> parameters -> scan wiring is live.  To finish this test the fixtures have to be
    // committed before the workers start (create them outside the test transaction, e.g. from
    // the harness/setup SQL, or drive the parallel build from a script against a real cluster
    // where such a table exists).
    #[pgrx::pg_test]
    #[ignore = "fixtures are uncommitted, so workers cannot open them; see the comment above"]
    fn workers_build_a_shared_graph_in_parallel() {
        // The first real parallel build: two worker *processes* scanning one table into one
        // shared arena.  The leader seeds an entry first, because workers never promote one and
        // a graph without an entry gets born with no edges at all (a failure this project has
        // produced before).
        //
        // The index is a real hnswsq index so the scan's key attribute is the embedding -- the
        // callback receives the vector through `BuildIndexInfo` -- but it is *not* the index
        // being built: the AM does not drive this yet, so the workers insert into a fresh arena.
        pgrx::Spi::run("CREATE TABLE par_build_test(id int, embedding vector(3))").unwrap();
        pgrx::Spi::run(
            "INSERT INTO par_build_test
             SELECT g, ('[' || g || ',' || (g % 7) || ',' || (g % 3) || ']')::vector
             FROM generate_series(1, 200) g",
        )
        .unwrap();
        pgrx::Spi::run(
            "CREATE INDEX par_build_test_idx ON par_build_test USING hnswsq (embedding vector_l2_ops)
             WITH (storage_layout='plain', m=8, ef_construction=32)",
        )
        .unwrap();
        let heap_oid: pg_sys::Oid =
            pgrx::Spi::get_one("SELECT 'par_build_test'::regclass::oid").unwrap().unwrap();
        let index_oid: pg_sys::Oid =
            pgrx::Spi::get_one("SELECT 'par_build_test_idx'::regclass::oid").unwrap().unwrap();

        let (stride, dims, m) = (12u32, 3u32, 8u32);
        let cap = m * 2;
        let sizing = plan(stride as usize, cap as usize, 1 << 20);
        assert!(sizing.nodes > 300, "room for the table: {:?}", sizing);
        let tranche = super::super::arena::register_tranche(c"hnswsq_parallel_build_test");

        // SAFETY: leader-side; relations are opened here and closed with the context.
        unsafe {
            let heap = pg_sys::table_open(heap_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
            let snapshot = pg_sys::GetActiveSnapshot();
            let pcxt = pg_sys::CreateParallelContext(
                LIBRARY.as_ptr().cast_mut().cast::<std::os::raw::c_char>(),
                c"hnswsq_parallel_build_main".as_ptr().cast_mut(),
                2,
            );
            estimate_arena(pcxt, stride as usize, cap as usize, sizing);
            estimate_chunk(pcxt, scan_bytes(heap, snapshot));
            let (arena, _layout) = leader_setup(pcxt, stride as usize, cap as usize, sizing, tranche);
            leader_scan_setup(pcxt, heap, snapshot);

            // The leader's seed: without an entry no search finds anything and every node is
            // born with an empty list.
            {
                let mut seed = FlatGraph::in_arena(&arena);
                let id = seed.claim_slot(0).expect("room for the seed");
                seed.publish(id, crate::util::ItemPointer::new(1, 1), false, &[1u8; 12]);
                assert!(seed.promote_entry(id), "the seed owns the entry");
            }
            arena.state().set_start_nodes(0);

            BuildParams {
                rows: 200,
                seed: 20240912,
                heap_oid: u32::from(heap_oid),
                index_oid: u32::from(index_oid),
                stride,
                num_dimensions: dims,
                cap,
                m,
                m0: cap,
                ef_construction: 32,
                ml: 1.0 / 8f32.ln(),
                max_level: 7,
                dist_type: 0,
                precision: 0, // Plain, matching the index's storage_layout
                backfill: 0,
                backlink_mode: 0,
                tranche,
            }
            .publish((*pcxt).toc);

            pg_sys::LaunchParallelWorkers(pcxt);
            pg_sys::WaitForParallelWorkersToFinish(pcxt);

            let launched = (*pcxt).nworkers_launched;
            assert!(launched > 0, "no workers were launched");
            assert_eq!(
                arena.state().workers_entered(),
                launched as u32,
                "every worker entered and ran its scan"
            );
            assert!(!arena.state().failed(), "no worker failed");

            // Every row became a node: the seed plus the table.
            let watermark = arena.state().watermark();
            assert_eq!(
                watermark,
                1 + 200,
                "the parallel build published every row (plus the seed)"
            );

            // ... and the graph two processes co-built is structurally sound: no self-links,
            // no repeated ids, capacity respected, and the entry reaches it.
            let g = FlatGraph::in_arena(&arena);
            let checks = check_lists(&g, cap as usize);
            assert_eq!(checks.self_links, 0, "{}", checks.summary(cap as usize));
            assert_eq!(checks.duplicate_links, 0, "{}", checks.summary(cap as usize));
            assert!(checks.max_list_len > 0, "backlinks landed: {}", checks.summary(cap as usize));
            assert!(checks.max_list_len <= cap as usize);
            assert!(
                checks.reachable_from_entry > 100,
                "the seeded entry reaches most of the graph: {}",
                checks.summary(cap as usize)
            );

            drop(g);
            pg_sys::DestroyParallelContext(pcxt);
            pg_sys::table_close(heap, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        }
    }

    #[pgrx::pg_test]
    fn a_worker_state_is_built_over_the_shared_arena() {
        // `worker_build_state` against a real arena.  The state's own fields are private to
        // `build.rs`, so what is asserted here is what this side can see: the construction
        // succeeds against parameters shaped like a real index, and it leaves the arena
        // alone -- a state that claimed or published anything while being built would be
        // corrupting a graph the leader may already have seeded.  The finer assertions
        // (budget, disk mode, the graph being the arena's) want accessors on `BuildState`
        // that the driver will need anyway for the writeout; they come with that.
        use super::super::build;
        use super::super::quantize::HnswPrecision;

        let size = 1usize << 20;
        // SAFETY: a fresh backend-owned segment, detached before the test returns.
        let seg = unsafe { pgrx::pg_sys::dsm_create(size, 0) };
        let toc = unsafe {
            pgrx::pg_sys::shm_toc_create(
                super::super::arena::HNSWSQ_TOC_MAGIC,
                pgrx::pg_sys::dsm_segment_address(seg),
                size,
            )
        };
        let tranche = super::super::arena::register_tranche(c"hnswsq_worker_state_test");
        let (stride, cap, nodes, slabs) = (12u32, 4u32, 16u32, 32u32);
        let arena = unsafe {
            super::super::arena::SharedArena::allocate(
                toc,
                stride as usize,
                cap as usize,
                nodes as usize,
                slabs as usize,
                tranche,
            )
        };

        let params = BuildParams {
            rows: 0,
            seed: 20240912,
            heap_oid: 0,
            index_oid: 0,
            stride,
            num_dimensions: 3,
            cap,
            m: 8,
            m0: 16,
            ef_construction: 32,
            ml: 0.5,
            max_level: 7,
            dist_type: 0,
            precision: HnswPrecision::Plain as u8,
            backfill: 0,
            backlink_mode: 0,
            tranche,
        };

        let state = build::worker_build_state(&params, &arena);
        assert_eq!(
            arena.state().claimed(),
            (0, 0),
            "building a worker's state must not claim anything"
        );
        assert_eq!(arena.state().watermark(), 0);
        assert!(!arena.state().failed());
        drop(state);
        drop(arena);
        // SAFETY: no handle references the mapping any more.
        unsafe { pgrx::pg_sys::dsm_detach(seg) };
    }

    #[pgrx::pg_test]
    fn the_leader_publishes_a_shared_scan_descriptor() {
        // A real table, a real snapshot and a descriptor every worker can read: this is the
        // object PostgreSQL's parallel scan hands block ranges out through, so all workers
        // reading the *same* one is what makes them cover the table once rather than each
        // scanning all of it.  The snapshot is copied into the descriptor by
        // `table_parallelscan_initialize`, which is why a worker needs nothing else.
        pgrx::Spi::run("CREATE TABLE driver_scan_test(id int)").unwrap();
        pgrx::Spi::run(
            "INSERT INTO driver_scan_test SELECT g FROM generate_series(1, 500) g",
        )
        .unwrap();
        let relid: pg_sys::Oid = pgrx::Spi::get_one("SELECT 'driver_scan_test'::regclass::oid")
            .unwrap()
            .unwrap();

        let (stride, cap) = (8usize, 4usize);
        let sizing = plan(stride, cap, 1 << 20);
        // SAFETY: one leader-side context; the relation is opened and closed here.
        unsafe {
            let heap = pg_sys::table_open(relid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
            let snapshot = pg_sys::GetActiveSnapshot();
            assert!(!snapshot.is_null(), "a scan needs a snapshot");

            let bytes = scan_bytes(heap, snapshot);
            assert!(
                bytes >= std::mem::size_of::<pg_sys::ParallelTableScanDescData>(),
                "the estimate must cover the descriptor itself: {} bytes",
                bytes
            );

            let pcxt = pg_sys::CreateParallelContext(
                LIBRARY.as_ptr().cast_mut().cast::<std::os::raw::c_char>(),
                c"hnswsq_parallel_build_main".as_ptr().cast_mut(),
                2,
            );
            estimate_arena(pcxt, stride, cap, sizing);
            // The descriptor is one more allocation, and the estimator is per allocation.
            estimate_chunk(pcxt, bytes);
            let (_arena, _layout) = leader_setup(pcxt, stride, cap, sizing, 0);

            let pscan = leader_scan_setup(pcxt, heap, snapshot);
            assert!(!pscan.is_null(), "the descriptor was allocated");
            assert_eq!(
                scan_descriptor((*pcxt).toc),
                pscan,
                "a worker finds the same descriptor by key"
            );

            pg_sys::DestroyParallelContext(pcxt);
            pg_sys::table_close(heap, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        }
    }

    #[pgrx::pg_test]
    fn the_leader_can_size_and_allocate_the_arena() {
        // The leader half of the driver without launching a worker:
        // estimate -> InitializeParallelDSM -> allocate the arena in the context's toc
        // -> find its regions again by key, the way a worker would.
        let (stride, cap) = (8usize, 4usize);
        let sizing = plan(stride, cap, 1 << 20);
        assert!(sizing.nodes > 10, "sanity: {:?}", sizing);

        // SAFETY: leader-side and single-threaded; the context is created and destroyed
        // inside this test.
        unsafe {
            let library = LIBRARY.as_ptr().cast_mut().cast::<std::os::raw::c_char>();
            // The same entry point the cross-process test launches; no worker is started
            // here, which is what keeps this test leader-only.
            let entry = c"hnswsq_parallel_build_main".as_ptr().cast_mut();
            let pcxt = pg_sys::CreateParallelContext(library, entry, 2);
            assert!(!pcxt.is_null(), "CreateParallelContext failed");

            estimate_arena(pcxt, stride, cap, sizing);
            let (arena, layout) = leader_setup(pcxt, stride, cap, sizing, 0);

            // The arena got the shape we planned: the estimator really did reserve enough
            // for it (a short estimate fails inside InitializeParallelDSM or the first
            // allocate, not silently).
            assert_eq!(arena.header().max_nodes, sizing.nodes);
            assert_eq!(arena.header().max_slabs, sizing.slabs);
            assert_eq!(arena.chunk().total_bytes(), layout.total_bytes);
            assert!(layout.total_bytes > 0);

            // pgx does not bind `PARALLEL_MAGIC`, and memory of it was wrong (the pinned
            // value made `shm_toc_attach` return NULL, which is how this assertion earned
            // its place).  So read it off the toc PostgreSQL actually built -- the first
            // field of `shm_toc` is the magic -- and require the pinned constant to match
            // it.  From here on the constant is verified, not remembered.
            let measured = *((*pcxt).toc as *const u64);
            assert_eq!(
                measured, PARALLEL_MAGIC,
                "the pinned toc magic must match the one PostgreSQL uses"
            );
            let attached =
                pg_sys::shm_toc_attach(measured, pg_sys::dsm_segment_address((*pcxt).seg));
            assert_eq!(attached, (*pcxt).toc, "the magic re-finds the toc");

            // Every region is reachable by key through it, which is a worker's whole
            // attachment story.
            assert!(!pg_sys::shm_toc_lookup(attached, TOC_KEY_HEADER, false).is_null());
            assert!(!pg_sys::shm_toc_lookup(attached, TOC_KEY_CHUNK, false).is_null());
            let worker_view = SharedArena::attach(attached);
            assert_eq!(worker_view.header().max_nodes, sizing.nodes);
            assert_eq!(worker_view.chunk().total_bytes(), layout.total_bytes);

            // ... and the build parameters round-trip, which is the contract the worker
            // loop will read.
            let params = BuildParams {
                rows: 1234,
                seed: 20240912,
                heap_oid: 1,
                index_oid: 2,
                stride: stride as u32,
                num_dimensions: 2,
                cap: cap as u32,
                m: 8,
                m0: 16,
                ef_construction: 64,
                ml: 1.0 / 8f32.ln(),
                max_level: 7,
                dist_type: 0,
                precision: 0,
                backfill: 0,
                backlink_mode: 0,
                tranche: 0,
            };
            params.publish((*pcxt).toc);
            assert_eq!(BuildParams::read_from(attached), params, "parameters survive");
            params.publish((*pcxt).toc);
            assert_eq!(
                BuildParams::read_from(attached),
                params,
                "republishing overwrites instead of allocating again"
            );

            drop(worker_view);
            drop(arena);
            pg_sys::DestroyParallelContext(pcxt);
        }
    }

}
