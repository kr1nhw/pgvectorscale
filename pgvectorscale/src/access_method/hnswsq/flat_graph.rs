//! Flat, policy-free memory graph for the build — successor to `build.rs`'s
//! `MemGraph`, written alongside it (see
//! `.design/hnswsq_parallel_build_todos.md`, "Approach change").
//!
//! Shape of the storage, per node:
//!
//! ```text
//! levels[i]            u8            node i's top layer
//! tids[i]              ItemPointer   heap TID (Invalid for a placeholder)
//! clamped[i]           bool          encoding saturated a component
//! vectors[i*stride..]  [u8; stride]  encoded vector (dim × elem_bytes)
//! slab(slab_off[i]+l)  [u32; cap]    layer l's neighbour ids, `lens` valid
//! lens[slab]           u16           valid prefix length of that slab
//! ```
//!
//! Three differences from `MemGraph`, all deliberate:
//!
//! * **one flat slab per (node, layer)** instead of `Vec<Vec<Vec<u32>>>` — no
//!   per-node allocation, no three-level indirection, and a fixed slot size, which
//!   is exactly what the shared-memory arena will need (contiguous, fixed-size,
//!   no process-local pointers);
//! * **ids only** — `list_dists` / `list_masks` are gone, matching the decided
//!   backlink policy (pgvector's append/shrink, which needs no per-entry state);
//! * **policy-free** — this module stores lists; who may join a list, and what
//!   happens on overflow, stays with the driver.  The legacy engine keeps its
//!   exact incremental re-prune; the new engine will use append/shrink.
//!
//! Capacity per slab is `m0` (layer 0) and `m` (upper layers), the same bounds the
//! on-disk node format uses, so a slab can never hold more than a node page item
//! would.

use crate::util::ItemPointer;

/// Flat memory graph: parallel slabs, fixed capacity, no per-node allocation.
pub struct FlatGraph {
    /// Bytes per encoded vector (`Codec::vector_bytes`).
    stride: usize,
    /// Ids per (node, layer) slab (`max(m, m0)`).
    cap: usize,
    levels: Vec<u8>,
    tids: Vec<ItemPointer>,
    clamped: Vec<bool>,
    /// `len * stride` bytes: node `i`'s encoded vector at `i * stride`.
    vectors: Vec<u8>,
    /// `slab_off[node]` = index of the node's layer-0 slab; a node with
    /// `level + 1` layers owns `slab_off[node] .. slab_off[node] + level + 1`.
    slab_off: Vec<u32>,
    /// `slabs * cap` ids.
    ids: Vec<u32>,
    /// One length per slab.
    lens: Vec<u16>,
    entry: Option<u32>,
    entry_level: usize,
    /// Fixed budgets for the arena: `usize::MAX` means "grow like a `Vec`" (the
    /// prototype and the legacy single-backend path), a finite value means the
    /// backing storage is preallocated and cannot grow — the shape the shared
    /// chunk needs, where running out is a normal outcome rather than an error.
    max_nodes: usize,
    max_slabs: usize,
}

impl FlatGraph {
    /// `stride` = encoded vector bytes, `cap` = ids per slab (`m0`; layer-0
    /// capacity bounds every layer because `m0 >= m`).
    pub fn new(stride: usize, cap: usize) -> Self {
        assert!(cap > 0 && cap <= u16::MAX as usize);
        Self {
            stride,
            cap,
            levels: Vec::new(),
            tids: Vec::new(),
            clamped: Vec::new(),
            vectors: Vec::new(),
            slab_off: Vec::new(),
            ids: Vec::new(),
            lens: Vec::new(),
            entry: None,
            entry_level: 0,
            max_nodes: usize::MAX,
            max_slabs: usize::MAX,
        }
    }

    /// Fixed-capacity graph: at most `max_nodes` nodes and `max_slabs`
    /// `(node, layer)` lists, with the backing storage preallocated so it can be
    /// laid out in one shared chunk.  `push_node` then has to go through
    /// [`FlatGraph::try_push_node`], and `false` from it means "full".
    pub fn with_limits(stride: usize, cap: usize, max_nodes: usize, max_slabs: usize) -> Self {
        let mut g = Self::new(stride, cap);
        g.max_nodes = max_nodes;
        g.max_slabs = max_slabs;
        g.levels.reserve_exact(max_nodes);
        g.tids.reserve_exact(max_nodes);
        g.clamped.reserve_exact(max_nodes);
        g.vectors.reserve_exact(max_nodes * stride);
        g.slab_off.reserve_exact(max_nodes);
        g.lens.reserve_exact(max_slabs);
        g.ids.reserve_exact(max_slabs * cap);
        g
    }

    /// Register a node if the budgets allow it; `false` leaves the graph
    /// untouched.  The arena cannot grow, so the driver answers `false` by
    /// writing out what exists and continuing on the disk path (see
    /// `spill_to_disk`), exactly like the `maintenance_work_mem` transition.
    pub fn try_push_node(
        &mut self,
        level: u8,
        tid: ItemPointer,
        clamped: bool,
        encoded: &[u8],
    ) -> bool {
        let layers = level as usize + 1;
        if self.len() >= self.max_nodes || self.lens.len() + layers > self.max_slabs {
            return false;
        }
        self.push_node(level, tid, clamped, encoded);
        true
    }

    /// Whether either budget is exhausted (always `false` in grow mode).
    pub fn is_full(&self) -> bool {
        self.len() >= self.max_nodes || self.lens.len() >= self.max_slabs
    }

    /// `(slabs used, slab budget)`.
    pub fn slab_usage(&self) -> (usize, usize) {
        (self.lens.len(), self.max_slabs)
    }

    /// `(nodes used, node budget)`.
    pub fn node_usage(&self) -> (usize, usize) {
        (self.len(), self.max_nodes)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn cap(&self) -> usize {
        self.cap
    }

    #[inline]
    pub fn entry(&self) -> Option<u32> {
        self.entry
    }

    #[inline]
    pub fn entry_level(&self) -> usize {
        self.entry_level
    }

    /// Promote `id` to entry point when its level is higher than the current
    /// one (the caller does this after the node's own list is published, so a
    /// searcher never lands on an unlinked entry).
    pub fn promote_entry(&mut self, id: u32) -> bool {
        let level = self.level(id) as usize;
        if level > self.entry_level || self.entry.is_none() {
            self.entry = Some(id);
            self.entry_level = level;
            true
        } else {
            false
        }
    }

    /// Register a node with `level + 1` empty layers.  The caller assigns ids in
    /// order (`id == self.len()` before the call), which keeps the writeout's
    /// id → page/offset mapping and the level stream deterministic.
    pub fn push_node(&mut self, level: u8, tid: ItemPointer, clamped: bool, encoded: &[u8]) {
        assert_eq!(
            encoded.len(),
            self.stride,
            "encoded vector length must equal the codec's stride"
        );
        let layers = level as usize + 1;
        let node = self.len() as u32;
        self.levels.push(level);
        self.tids.push(tid);
        self.clamped.push(clamped);
        self.vectors.extend_from_slice(encoded);
        self.slab_off
            .push(self.lens.len() as u32); // next slab index == this node's base
        self.lens.resize(self.lens.len() + layers, 0);
        self.ids.resize(self.lens.len() * self.cap, 0);
        debug_assert_eq!(self.slab_off.len(), self.levels.len());
        debug_assert!(self.slab_base(node).is_some());
    }

    /// Replace a node's layer-`layer` list with `ids` (ids only: the policy lives
    /// with the driver).  Panics on a bad layer or an over-capacity list, which
    /// is a programming error, not a data condition.
    pub fn set_list(&mut self, id: u32, layer: usize, ids: &[u32]) {
        assert!(
            ids.len() <= self.cap,
            "neighbour list of {} exceeds slab capacity {}",
            ids.len(),
            self.cap
        );
        let slab = self.slab(id, layer).expect("layer beyond the node's level");
        let base = slab * self.cap;
        self.ids[base..base + ids.len()].copy_from_slice(ids);
        self.lens[slab] = ids.len() as u16;
    }

    /// Borrowed view of a node's layer-`layer` list (valid prefix only).
    #[inline]
    pub fn neighbors(&self, id: u32, layer: usize) -> &[u32] {
        match self.slab(id, layer) {
            Some(slab) => {
                let base = slab * self.cap;
                &self.ids[base..base + self.lens[slab] as usize]
            }
            None => &[],
        }
    }

    /// Copy a node's layer-`layer` list into `out` (cleared first).  This is the
    /// lock-friendly shape the arena will need, since a borrow cannot outlive the
    /// guard that protects the slab.
    pub fn copy_neighbors(&self, id: u32, layer: usize, out: &mut Vec<u32>) {
        out.clear();
        out.extend_from_slice(self.neighbors(id, layer));
    }

    #[inline]
    pub fn level(&self, id: u32) -> u8 {
        self.levels[id as usize]
    }

    #[inline]
    pub fn tid(&self, id: u32) -> ItemPointer {
        self.tids[id as usize]
    }

    #[inline]
    pub fn clamped(&self, id: u32) -> bool {
        self.clamped[id as usize]
    }

    /// `id`'s encoded vector.
    #[inline]
    pub fn vector(&self, id: u32) -> &[u8] {
        let i = id as usize;
        &self.vectors[i * self.stride..(i + 1) * self.stride]
    }

    /// Bytes of heap storage the graph's own structures occupy (used by the
    /// maintenance_work_mem budget accounting).
    pub fn heap_bytes(&self) -> usize {
        self.levels.capacity()
            + self.tids.capacity() * std::mem::size_of::<ItemPointer>()
            + self.clamped.capacity()
            + self.vectors.capacity()
            + self.slab_off.capacity() * std::mem::size_of::<u32>()
            + self.ids.capacity() * std::mem::size_of::<u32>()
            + self.lens.capacity() * std::mem::size_of::<u16>()
    }

    /// Slab index of `(id, layer)`, or `None` when the node has no such layer.
    #[inline]
    fn slab(&self, id: u32, layer: usize) -> Option<usize> {
        let base = self.slab_base(id)?;
        (layer <= self.levels[id as usize] as usize).then_some(base + layer)
    }

    #[inline]
    fn slab_base(&self, id: u32) -> Option<usize> {
        let base = *self.slab_off.get(id as usize)? as usize;
        // A node with `level + 1` layers must own that many slabs.
        let layers = self.levels[id as usize] as usize + 1;
        (base + layers <= self.lens.len()).then_some(base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tid(n: u32) -> ItemPointer {
        ItemPointer::new(n, 1)
    }

    #[test]
    fn push_and_read_back_nodes() {
        let stride = 8;
        let cap = 4;
        let mut g = FlatGraph::new(stride, cap);
        assert!(g.is_empty() && g.entry().is_none());

        g.push_node(0, tid(10), false, &[1u8; 8]);
        g.push_node(2, tid(11), true, &[2u8; 8]);
        assert_eq!(g.len(), 2);
        assert_eq!(g.level(0), 0);
        assert_eq!(g.level(1), 2);
        assert_eq!(g.tid(1), tid(11));
        assert!(g.clamped(1));
        assert_eq!(g.vector(1), &[2u8; 8]);

        // layer 0 exists on both nodes, layer 1/2 only on the second
        assert!(g.neighbors(0, 0).is_empty());
        assert!(g.neighbors(0, 1).is_empty());
        assert!(g.neighbors(1, 2).is_empty());
    }

    #[test]
    fn set_list_round_trips_and_is_capacity_bounded() {
        let mut g = FlatGraph::new(4, 4);
        g.push_node(1, tid(1), false, &[0u8; 4]);
        g.set_list(0, 0, &[7, 8, 9]);
        g.set_list(0, 1, &[5]);
        assert_eq!(g.neighbors(0, 0), &[7, 8, 9]);
        assert_eq!(g.neighbors(0, 1), &[5]);

        let mut out = vec![99u8 as u32];
        g.copy_neighbors(0, 0, &mut out);
        assert_eq!(out, vec![7, 8, 9]);
        g.copy_neighbors(0, 1, &mut out);
        assert_eq!(out, vec![5]);

        // shortest list wins: rewriting shrinks the valid prefix
        g.set_list(0, 0, &[1]);
        assert_eq!(g.neighbors(0, 0), &[1]);

        // two nodes never share slab storage
        g.push_node(0, tid(2), false, &[0u8; 4]);
        g.set_list(1, 0, &[42]);
        assert_eq!(g.neighbors(1, 0), &[42]);
        assert_eq!(g.neighbors(0, 0), &[1]);
    }

    #[test]
    #[should_panic(expected = "exceeds slab capacity")]
    fn over_capacity_list_is_rejected() {
        let mut g = FlatGraph::new(4, 2);
        g.push_node(0, tid(1), false, &[0u8; 4]);
        g.set_list(0, 0, &[1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "layer beyond the node's level")]
    fn layer_beyond_level_is_rejected() {
        let mut g = FlatGraph::new(4, 2);
        g.push_node(0, tid(1), false, &[0u8; 4]);
        g.set_list(0, 1, &[1]);
    }

    #[test]
    fn fixed_capacity_reports_full_instead_of_growing() {
        // Node budget of one: the second node is refused, the graph is unchanged.
        let mut g = FlatGraph::with_limits(4, 2, 1, 8);
        assert!(g.try_push_node(0, tid(1), false, &[0u8; 4]));
        assert_eq!(g.node_usage(), (1, 1));
        assert!(g.is_full(), "node budget of one is used up");
        assert!(!g.try_push_node(0, tid(2), false, &[0u8; 4]));
        assert_eq!(g.len(), 1);

        // Slab budget of one: a level-1 node needs two slabs, so it is refused
        // before anything is written; a level-0 node then fits exactly.
        let mut g = FlatGraph::with_limits(4, 2, 8, 1);
        assert!(
            !g.try_push_node(1, tid(1), false, &[0u8; 4]),
            "a level-1 node needs two slabs"
        );
        assert_eq!(g.len(), 0, "a refused push leaves the graph untouched");
        assert!(g.try_push_node(0, tid(1), false, &[0u8; 4]));
        assert_eq!(g.slab_usage(), (1, 1));
        assert!(g.is_full(), "the slab budget is exhausted");
    }

    #[test]
    fn grow_mode_never_reports_full() {
        let mut g = FlatGraph::new(4, 2);
        for i in 0..10 {
            assert!(g.try_push_node(0, tid(i + 1), false, &[0u8; 4]));
        }
        assert!(!g.is_full());
        assert_eq!(g.node_usage().1, usize::MAX);
    }

    #[test]
    fn entry_promotion_keeps_the_highest_level() {
        let mut g = FlatGraph::new(4, 2);
        g.push_node(0, tid(1), false, &[0u8; 4]);
        assert!(g.promote_entry(0));
        assert_eq!((g.entry(), g.entry_level()), (Some(0), 0));

        g.push_node(3, tid(2), false, &[0u8; 4]);
        assert!(g.promote_entry(1));
        assert_eq!((g.entry(), g.entry_level()), (Some(1), 3));

        g.push_node(1, tid(3), false, &[0u8; 4]);
        assert!(!g.promote_entry(2), "a lower level must not take the entry");
        assert_eq!(g.entry(), Some(1));
    }
}
