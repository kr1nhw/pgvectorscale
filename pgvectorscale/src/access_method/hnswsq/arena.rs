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

#[cfg(test)]
mod tests {
    use super::*;

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
