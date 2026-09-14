//! Node-level locking for the parallel flat engine (M3).
//!
//! The shared-memory arena will hand every worker a `&HnswArena`, so mutation has
//! to go through interior locks — one per node, which is also the granularity the
//! backlink step needs (one target list at a time).  In this prototype the locks
//! are `std::sync::RwLock`s; in the arena they become LWLock tranches behind the
//! same read/write API, so the algorithm code does not change.
//!
//! The one rule the protocol depends on: **at most one node write lock may be held
//! at a time**.  Two would make lock-order deadlocks possible (backlink updates
//! touch different targets in different orders), and PostgreSQL does not detect
//! deadlocks between locks taken inside an extension's own structures.  Rather
//! than trusting reviewers, [`NodeLocks::write`] counts the write guards held by
//! the current thread and panics when a second one is taken; read guards may nest
//! freely (searches read many nodes) and a read guard never blocks another read.

use std::cell::Cell;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

thread_local! {
    /// Write guards currently held by this thread (see [`NodeLocks::write`]).
    static WRITE_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Read guard: one node's data may be read while other threads read it too.
pub struct NodeReadGuard<'a> {
    _guard: RwLockReadGuard<'a, ()>,
}

/// Write guard: exclusive access to one node's list.
pub struct NodeWriteGuard<'a> {
    _guard: RwLockWriteGuard<'a, ()>,
}

impl Drop for NodeWriteGuard<'_> {
    fn drop(&mut self) {
        WRITE_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// One lock per node slot.
pub struct NodeLocks {
    locks: Vec<RwLock<()>>,
}

impl NodeLocks {
    pub fn new(nodes: usize) -> Self {
        Self {
            locks: (0..nodes).map(|_| RwLock::new(())).collect(),
        }
    }

    /// Add locks for newly published nodes (the arena grows by claiming node ids
    /// from a shared counter; the lock array is extended under the caller's
    /// serialization, never while a worker holds a guard).
    pub fn grow_to(&mut self, nodes: usize) {
        while self.locks.len() < nodes {
            self.locks.push(RwLock::new(()));
        }
    }

    pub fn len(&self) -> usize {
        self.locks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.locks.is_empty()
    }

    /// Shared access to `id`'s node.
    ///
    /// Panics if the id has no lock slot, which is a programming error: ids come
    /// from the graph's own id space, and claiming an id publishes its slot first.
    pub fn read(&self, id: u32) -> NodeReadGuard<'_> {
        NodeReadGuard {
            _guard: self.locks[id as usize]
                .read()
                .unwrap_or_else(|e| e.into_inner()),
        }
    }

    /// Exclusive access to `id`'s node.
    ///
    /// Panics when the current thread already holds a node write guard: the
    /// backlink protocol takes one target lock at a time precisely so the lock
    /// order can never form a cycle.
    pub fn write(&self, id: u32) -> NodeWriteGuard<'_> {
        let guard = self.locks[id as usize]
            .write()
            .unwrap_or_else(|e| e.into_inner());
        WRITE_DEPTH.with(|d| {
            let depth = d.get();
            assert_eq!(
                depth, 0,
                "at most one node write lock may be held at a time (already {} deep) — \
                 the backlink protocol takes one target lock at a time to keep the \
                 lock order acyclic",
                depth
            );
            d.set(depth + 1);
        });
        NodeWriteGuard { _guard: guard }
    }
}

// ---------------------------------------------------------------------------
// Shared-chunk layout
// ---------------------------------------------------------------------------

/// One region of the arena chunk: where it starts and how big it is.  Offsets, not
/// pointers — the chunk is mapped at a different address in every process, so the
/// graph may only ever refer to its own storage by offset (which the flat arrays
/// already do: every access is index arithmetic).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub offset: usize,
    pub len: usize,
    /// Alignment the region's element type requires.
    pub align: usize,
}

impl Region {
    pub fn end(&self) -> usize {
        self.offset + self.len
    }
}

/// Byte layout of the arena for `nodes` node slots and `slabs` `(node, layer)`
/// lists.  Mirrors the flat graph's arrays one-for-one, so the storage swap is
/// "point each array at its region" rather than a reshape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaLayout {
    pub total_bytes: usize,
    pub vectors: Region,
    pub ids: Region,
    pub lens: Region,
    pub levels: Region,
    pub tids: Region,
    pub clamped: Region,
    pub published: Region,
    pub slab_off: Region,
}

#[inline]
fn align_up(x: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (x + align - 1) & !(align - 1)
}

/// Plan the chunk.  Regions are laid out in descending alignment need and never
/// overlap; `total_bytes` is what the shared segment has to reserve (before the
/// allocator's own margin).
pub fn arena_layout(stride: usize, cap: usize, nodes: usize, slabs: usize) -> ArenaLayout {
    let mut off = 0usize;
    let mut take = |len: usize, align: usize, off: &mut usize| -> Region {
        let start = align_up(*off, align);
        *off = start + len;
        Region {
            offset: start,
            len,
            align,
        }
    };
    let vectors = take(nodes * stride, 1, &mut off);
    let tids = take(
        nodes * std::mem::size_of::<crate::util::ItemPointer>(),
        std::mem::align_of::<crate::util::ItemPointer>(),
        &mut off,
    );
    let slab_off = take(nodes * std::mem::size_of::<u32>(), 4, &mut off);
    let ids = take(slabs * cap * std::mem::size_of::<u32>(), 4, &mut off);
    let lens = take(slabs * std::mem::size_of::<u16>(), 2, &mut off);
    let levels = take(nodes, 1, &mut off);
    let clamped = take(nodes, 1, &mut off);
    let published = take(nodes, 1, &mut off);
    ArenaLayout {
        total_bytes: off,
        vectors,
        ids,
        lens,
        levels,
        tids,
        clamped,
        published,
        slab_off,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_regions_are_aligned_and_disjoint() {
        let stride = 512;
        let cap = 32;
        let (nodes, slabs) = (1000usize, 1100usize);
        let l = arena_layout(stride, cap, nodes, slabs);

        let regions = [
            l.vectors, l.tids, l.slab_off, l.ids, l.lens, l.levels, l.clamped, l.published,
        ];
        for r in regions {
            assert_eq!(r.offset % r.align, 0, "region at {} needs align {}", r.offset, r.align);
            assert!(r.end() <= l.total_bytes, "region runs past the chunk");
        }
        // Sizes match the array shapes the graph uses.
        assert_eq!(l.vectors.len, nodes * stride);
        assert_eq!(l.ids.len, slabs * cap * 4);
        assert_eq!(l.lens.len, slabs * 2);
        assert_eq!(l.levels.len, nodes);
        assert_eq!(l.slab_off.len, nodes * 4);

        // Ordered layout: every region starts at or after the previous one's end.
        let mut sorted = regions;
        sorted.sort_by_key(|r| r.offset);
        for pair in sorted.windows(2) {
            assert!(pair[0].end() <= pair[1].offset, "regions overlap");
        }
    }

    #[test]
    fn layout_fits_the_budget_it_was_planned_for() {
        use super::super::flat_graph::plan_capacity;
        let (stride, cap) = (512, 32);
        let budget = 1u64 << 30;
        let margin = 0.7;
        let sizing = plan_capacity(stride, cap, budget, 1.07, margin);
        assert!(sizing.nodes > 0);

        let l = arena_layout(stride, cap, sizing.nodes, sizing.slabs);
        let allowed = (budget as f64 * margin) as usize;
        assert!(
            l.total_bytes <= allowed,
            "layout {} exceeds the {} bytes the plan allowed",
            l.total_bytes,
            allowed
        );
        // ... and the layout is not wildly smaller than planned either, which would
        // mean the two formulas had drifted apart.
        assert!(l.total_bytes as f64 > 0.9 * allowed as f64);
        // A node's vectors alone are the floor.
        assert!(l.total_bytes >= sizing.nodes * stride);
    }

    #[test]
    fn read_guards_nest_freely() {
        let locks = NodeLocks::new(4);
        let a = locks.read(0);
        let b = locks.read(1);
        let c = locks.read(0); // same node twice: shared locks do not conflict
        drop((a, b, c));
    }

    #[test]
    fn one_write_guard_at_a_time() {
        let locks = NodeLocks::new(4);
        {
            let _w = locks.write(0);
            // A read on a *different* node is still allowed while a write is held
            // (the search holds read guards; the backlink holds one write guard).
            let _r = locks.read(1);
        }
        // and the counter unwinds, so the next write is fine
        let _w = locks.write(1);
    }

    #[test]
    #[should_panic(expected = "at most one node write lock")]
    fn nested_write_guard_is_rejected() {
        let locks = NodeLocks::new(4);
        let _outer = locks.write(0);
        let _inner = locks.write(1); // would be an A→B / B→A deadlock waiting to happen
    }

    #[test]
    fn grow_to_adds_slots_without_touching_existing_ones() {
        let mut locks = NodeLocks::new(2);
        assert_eq!(locks.len(), 2);
        locks.grow_to(5);
        assert_eq!(locks.len(), 5);
        locks.grow_to(3); // no-op
        assert_eq!(locks.len(), 5);
        let _w = locks.write(4);
    }
}
