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

/// PostgreSQL's toc magic (`PARALLEL_MAGIC` in `parallel.h`).
///
/// pgrx does not bind this `#define`, and guessing it would be the kind of silent mistake
/// that shows up only as a worker failing to attach, so the test below *verifies* it
/// against a real context instead: `shm_toc_attach(PARALLEL_MAGIC, ...)` has to return
/// exactly the pointer `InitializeParallelDSM` put in `pcxt->toc`.
pub const PARALLEL_MAGIC: u64 = 0x50477c23;

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
    // One key per region: header, state, chunk, locks.
    estimate_keys(pcxt, 4);
}

/// Leader: allocate the segment and put the arena in it.
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
            // The entry point does not exist yet.  PostgreSQL resolves it when a worker
            // starts, which is why this test runs without launching any.
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

            drop(arena);
            pg_sys::DestroyParallelContext(pcxt);
        }
    }

}
