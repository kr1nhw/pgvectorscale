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

use super::arena::{ArenaLayout, SharedArena, TOC_KEY_CHUNK, TOC_KEY_HEADER};
use super::flat_graph::{plan_capacity, ArenaSizing};

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
    let arena = worker_attach(toc);
    arena.state().worker_entered();
    arena.state().worker_started();

    // The build loop goes here: a shared scan, `claim_slot`, search/plan/apply with
    // `Locking::Locks`, then `publish`.  Until it lands, a worker's whole job is to prove
    // it can reach the leader's arena from its own process -- which is the thing no
    // thread-based test could establish.

    arena.state().worker_finished();
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
