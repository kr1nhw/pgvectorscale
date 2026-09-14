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

use crate::access_method::hnswsq::arena::{arena_layout, ArenaLayout, Chunk};
use crate::util::ItemPointer;

/// Flat memory graph: parallel slabs, fixed capacity, no per-node allocation.
pub struct FlatGraph {
    /// Bytes per encoded vector (`Codec::vector_bytes`).
    stride: usize,
    /// Ids per (node, layer) slab (`max(m, m0)`).
    cap: usize,
    levels: Vec<u8>,
    /// Nodes handed out so far -- the authoritative node count, in both modes.
    /// It is not `levels.len()`: with the level region in the chunk the `Vec` is
    /// empty.  Also the id the next claim gets, so it must advance before anything
    /// else can observe the new slot.
    nodes_used: usize,
    tids: Vec<ItemPointer>,
    clamped: Vec<bool>,
    /// `len * stride` bytes: node `i`'s encoded vector at `i * stride`.  Empty when
    /// the graph is chunk-backed (see `chunk`).
    vectors: Vec<u8>,
    /// When set, node storage lives in this chunk (the arena shape) instead of the
    /// `Vec`s, and `layout` describes its regions.  Being converted region by region:
    /// `vectors`, `ids`, `lens` and `levels` are in the chunk, the rest follow, at
    /// which point the `Vec` fields disappear.
    chunk: Option<Chunk>,
    layout: ArenaLayout,
    /// `slab_off[node]` = index of the node's layer-0 slab; a node with
    /// `level + 1` layers owns `slab_off[node] .. slab_off[node] + level + 1`.
    slab_off: Vec<u32>,
    /// `slabs * cap` ids.
    ids: Vec<u32>,
    /// One length per slab.
    lens: Vec<u16>,
    /// Slabs handed out so far -- the authoritative cursor, in both modes.  It is
    /// not `lens.len()`: with the lens region in the chunk the `Vec` is empty, and
    /// even in grow mode the length store must not double as a cursor.
    slabs_used: usize,
    entry: Option<u32>,
    entry_level: usize,
    /// Fixed budgets for the arena: `usize::MAX` means "grow like a `Vec`" (the
    /// prototype and the legacy single-backend path), a finite value means the
    /// backing storage is preallocated and cannot grow — the shape the shared
    /// chunk needs, where running out is a normal outcome rather than an error.
    max_nodes: usize,
    max_slabs: usize,
    /// Per node: has its level/tid/clamped/vector been written yet?
    published_flags: Vec<bool>,
    /// Every id below this watermark is published.  In the parallel build a worker
    /// claims an id from the shared counter and only then writes the node, so a
    /// peer that observes the id must skip it: readers bound-check against this
    /// watermark, never against the claim counter (`len`).
    watermark: usize,
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
            nodes_used: 0,
            tids: Vec::new(),
            clamped: Vec::new(),
            vectors: Vec::new(),
            chunk: None,
            layout: arena_layout(0, 0, 0, 0),
            slab_off: Vec::new(),
            ids: Vec::new(),
            lens: Vec::new(),
            slabs_used: 0,
            entry: None,
            entry_level: 0,
            max_nodes: usize::MAX,
            max_slabs: usize::MAX,
            published_flags: Vec::new(),
            watermark: 0,
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
        // Chunk-backed from here on: the vector arena lives in the chunk's `vectors`
        // region, so the `Vec` copy stays empty (and `heap_bytes` reports the layout).
        g.layout = arena_layout(stride, cap, max_nodes, max_slabs);
        g.chunk = Some(Chunk::new(g.layout));
        // `levels` is a chunk region now too, so no `Vec` capacity for it.
        g.tids.reserve_exact(max_nodes);
        g.clamped.reserve_exact(max_nodes);
        g.slab_off.reserve_exact(max_nodes);
        // `ids` and `lens` are chunk regions now, so no `Vec` capacity is reserved
        // for them -- that was up to `max_slabs * cap * 4` bytes of dead allocation.
        g.slabs_used = 0;
        g.published_flags.reserve_exact(max_nodes);
        g
    }

    /// Number of nodes every part of whose data is written.  Equal to `len()` in
    /// the single-threaded prototype and the legacy path; smaller while workers
    /// hold claimed-but-unwritten slots in the arena.
    #[inline]
    pub fn watermark(&self) -> usize {
        self.watermark
    }

    /// Reserve a node id and its slabs without writing any data.  The slot stays
    /// invisible to readers (see [`FlatGraph::watermark`]) until
    /// [`FlatGraph::publish`] completes it.  `None` when the budgets are
    /// exhausted — the driver then spills, exactly as for `try_push_node`.
    pub fn claim_slot(&mut self, level: u8) -> Option<u32> {
        let layers = level as usize + 1;
        if self.len() >= self.max_nodes || self.slabs_used + layers > self.max_slabs {
            return None;
        }
        let id = self.nodes_used as u32;
        self.nodes_used += 1;
        match self.chunk.as_mut() {
            Some(chunk) => chunk.region_bytes_mut(self.layout.levels)[id as usize] = level,
            None => self.levels.push(level),
        }
        debug_assert!(self.chunk.is_some() || self.levels.len() == self.nodes_used);
        self.tids.push(ItemPointer::new_invalid());
        self.clamped.push(false);
        if self.chunk.is_none() {
            self.vectors.resize(self.vectors.len() + self.stride, 0);
        }
        self.slab_off.push(self.slabs_used as u32);
        self.slabs_used += layers;
        if self.chunk.is_none() {
            // Grow mode only: `lens`/`ids` are the storage.  They are kept in
            // lockstep with the cursor, filled with zero lengths exactly like the
            // chunk's zeroed region, so a claimed-but-unwritten slab reads empty.
            self.lens.resize(self.slabs_used, 0);
            self.ids.resize(self.slabs_used * self.cap, 0);
        }
        self.published_flags.push(false);
        Some(id)
    }

    /// Write a claimed node's data and publish it.  Publication is what makes the
    /// node observable; the watermark advances only through the contiguous
    /// published prefix, so several workers completing out of order is fine.
    pub fn publish(&mut self, id: u32, tid: ItemPointer, clamped: bool, encoded: &[u8]) {
        assert_eq!(encoded.len(), self.stride, "encoded vector must match stride");
        let i = id as usize;
        assert!(i < self.len(), "publish of an unclaimed id");
        assert!(!self.published_flags[i], "node {} published twice", id);
        self.tids[i] = tid;
        self.clamped[i] = clamped;
        let start = i * self.stride;
        match self.chunk.as_mut() {
            Some(chunk) => {
                chunk.region_bytes_mut(self.layout.vectors)[start..start + self.stride]
                    .copy_from_slice(encoded);
            }
            None => self.vectors[start..start + self.stride].copy_from_slice(encoded),
        }
        self.published_flags[i] = true;
        while self.watermark < self.published_flags.len() && self.published_flags[self.watermark] {
            self.watermark += 1;
        }
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
        if self.len() >= self.max_nodes || self.slabs_used + layers > self.max_slabs {
            return false;
        }
        self.push_node(level, tid, clamped, encoded);
        true
    }

    /// Whether either budget is exhausted (always `false` in grow mode).
    pub fn is_full(&self) -> bool {
        self.len() >= self.max_nodes || self.slabs_used >= self.max_slabs
    }

    /// `(slabs used, slab budget)`.
    pub fn slab_usage(&self) -> (usize, usize) {
        (self.slabs_used, self.max_slabs)
    }

    /// `(nodes used, node budget)`.
    pub fn node_usage(&self) -> (usize, usize) {
        (self.len(), self.max_nodes)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nodes_used
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
        let id = self
            .claim_slot(level)
            .expect("graph capacity exhausted (use try_push_node, or claim/publish)");
        self.publish(id, tid, clamped, encoded);
        debug_assert_eq!(self.slab_off.len(), self.nodes_used);
        debug_assert!(self.slab_base(id).is_some());
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
        let n = ids.len();
        match self.chunk.as_mut() {
            Some(chunk) => {
                chunk.region_u32_mut(self.layout.ids)[base..base + n].copy_from_slice(ids);
            }
            None => self.ids[base..base + n].copy_from_slice(ids),
        }
        match self.chunk.as_mut() {
            Some(chunk) => chunk.region_u16_mut(self.layout.lens)[slab] = n as u16,
            None => self.lens[slab] = n as u16,
        }
    }

    /// Borrowed view of a node's layer-`layer` list (valid prefix only).
    #[inline]
    pub fn neighbors(&self, id: u32, layer: usize) -> &[u32] {
        match self.slab(id, layer) {
            Some(slab) => {
                let base = slab * self.cap;
                let (ids, n) = match &self.chunk {
                    Some(chunk) => (
                        chunk.region_u32(self.layout.ids),
                        chunk.region_u16(self.layout.lens)[slab] as usize,
                    ),
                    None => (&self.ids[..], self.lens[slab] as usize),
                };
                &ids[base..base + n]
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
        match &self.chunk {
            Some(chunk) => chunk.region_bytes(self.layout.levels)[id as usize],
            None => self.levels[id as usize],
        }
    }

    #[inline]
    pub fn tid(&self, id: u32) -> ItemPointer {
        self.tids[id as usize]
    }

    #[inline]
    pub fn clamped(&self, id: u32) -> bool {
        self.clamped[id as usize]
    }

    /// `id`'s encoded vector (from the chunk when the graph is chunk-backed, which
    /// is the shape the arena uses: the same bytes, addressed by offset).
    #[inline]
    pub fn vector(&self, id: u32) -> &[u8] {
        let i = id as usize;
        let start = i * self.stride;
        match &self.chunk {
            Some(chunk) => {
                &chunk.region_bytes(self.layout.vectors)[start..start + self.stride]
            }
            None => &self.vectors[start..start + self.stride],
        }
    }

    /// Bytes of heap storage the graph's own structures occupy (used by the
    /// maintenance_work_mem budget accounting).
    pub fn heap_bytes(&self) -> usize {
        if let Some(chunk) = &self.chunk {
            return chunk.total_bytes();
        }
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
        (layer <= self.level(id) as usize).then_some(base + layer)
    }

    #[inline]
    fn slab_base(&self, id: u32) -> Option<usize> {
        let base = *self.slab_off.get(id as usize)? as usize;
        // A node with `level + 1` layers must own that many slabs.
        let layers = self.level(id) as usize + 1;
        (base + layers <= self.slabs_used).then_some(base)
    }
}

/// Capacity chosen for a byte budget, before any worker starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaSizing {
    /// Node slots the budget affords.
    pub nodes: usize,
    /// `(node, layer)` slabs those nodes need at the assumed layer count.
    pub slabs: usize,
    /// Bytes one node costs (its vector, its layer-0 slab, its bookkeeping).
    pub bytes_per_node: usize,
}

/// Per-node bytes in the flat layout: vector bytes, `level`/`clamped` (1 each),
/// the heap TID (8), `slab_off` (4), the published flag (1), and for every layer a
/// slab of `cap` ids (4 each) plus its `u16` length (2).
#[inline]
pub fn bytes_per_node(stride: usize, cap: usize, layers: f64) -> usize {
    let per_layer = cap * std::mem::size_of::<u32>() + std::mem::size_of::<u16>();
    let fixed = stride + 1 + 8 + 1 + 4 + 1;
    fixed + (per_layer as f64 * layers).ceil() as usize
}

/// Size the arena from a byte budget.
///
/// `layers_per_node` is the average number of layers a node carries (1.0 plus the
/// fraction of nodes above layer 0, ≈1.07 for a 1M build at `m = 16`), and `margin`
/// is the fraction of the budget the graph may use — the arena cannot grow, so the
/// rest stays headroom for the transient copies the writeout makes.
///
/// A zero result means "this budget cannot host a parallel build"; the caller then
/// keeps the single-backend path rather than starting workers that would spill
/// immediately.
pub fn plan_capacity(
    stride: usize,
    cap: usize,
    budget_bytes: u64,
    layers_per_node: f64,
    margin: f64,
) -> ArenaSizing {
    let layers = layers_per_node.max(1.0);
    let per_node = bytes_per_node(stride, cap, layers).max(1);
    let usable = (budget_bytes as f64 * margin.clamp(0.0, 1.0)) as u64;
    let nodes = (usable / per_node as u64) as usize;
    let slabs = (nodes as f64 * layers).ceil() as usize;
    ArenaSizing {
        nodes,
        slabs,
        bytes_per_node: per_node,
    }
}

/// Structural checks on a build graph.  This is the gate the parallel plan applies
/// at every worker count: a concurrent insert may not leave a node unreachable or a
/// list malformed, and the connectivity number is exactly what exposed the rejected
/// batched design (up to 45% of nodes with no incoming edge, recall 0.955 -> 0.58).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ListChecks {
    /// Nodes claimed (including any still unpublished).
    pub nodes: usize,
    /// Nodes readable by a search (the watermark).
    pub published: usize,
    pub max_list_len: usize,
    pub self_links: usize,
    pub duplicate_links: usize,
    /// Published nodes that no list points at: invisible to every search.
    pub nodes_without_incoming: usize,
    /// Published nodes reachable from the entry point over layer-0 lists.
    pub reachable_from_entry: usize,
}

impl ListChecks {
    /// `cap` is the slab capacity the graph was built with.
    pub fn is_healthy(&self, cap: usize) -> bool {
        self.self_links == 0
            && self.duplicate_links == 0
            && self.max_list_len <= cap
            && self.nodes_without_incoming == 0
            && self.reachable_from_entry == self.published
    }

    /// One-line summary for a test failure message.
    pub fn summary(&self, cap: usize) -> String {
        format!(
            "nodes={} published={} max_list_len={}/{} self_links={} duplicates={} \
             no_incoming={} reachable={} healthy={}",
            self.nodes,
            self.published,
            self.max_list_len,
            cap,
            self.self_links,
            self.duplicate_links,
            self.nodes_without_incoming,
            self.reachable_from_entry,
            self.is_healthy(cap),
        )
    }
}

/// Walk every layer-0 list of every published node and report the structural checks.
pub fn check_lists(g: &FlatGraph, cap: usize) -> ListChecks {
    let published = g.watermark();
    let mut incoming = vec![0usize; published];
    let mut checks = ListChecks {
        nodes: g.len(),
        published,
        ..Default::default()
    };

    for node in 0..published as u32 {
        let list = g.neighbors(node, 0);
        checks.max_list_len = checks.max_list_len.max(list.len());
        for (i, &nb) in list.iter().enumerate() {
            if nb == node {
                checks.self_links += 1;
            }
            if list[..i].contains(&nb) {
                checks.duplicate_links += 1;
            }
            if (nb as usize) < published {
                incoming[nb as usize] += 1;
            }
        }
    }
    checks.nodes_without_incoming = incoming.iter().filter(|&&c| c == 0).count();

    // Reachability over layer-0 lists from the entry point (a BFS, bounded by the
    // published prefix so a claimed-but-unwritten node can never be walked into).
    if let Some(entry) = g.entry().filter(|&e| (e as usize) < published) {
        let mut seen = vec![false; published];
        let mut stack = vec![entry];
        seen[entry as usize] = true;
        let mut reached = 1usize;
        while let Some(node) = stack.pop() {
            for &nb in g.neighbors(node, 0) {
                let ni = nb as usize;
                if ni < published && !seen[ni] {
                    seen[ni] = true;
                    reached += 1;
                    stack.push(nb);
                }
            }
        }
        checks.reachable_from_entry = reached;
    }
    checks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_method::distance::{distance_l2, DistanceType};
    use crate::access_method::hnswsq::quantize::Codec;

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
    fn capacity_arithmetic_and_margin() {
        // 512 B vector + 15 B bookkeeping + (32 ids * 4 + 2) B slab = 657 B/node.
        assert_eq!(bytes_per_node(512, 32, 1.0), 512 + 15 + 130);
        // Extra layers add their slabs.
        assert_eq!(bytes_per_node(512, 32, 2.0), 512 + 15 + 260);

        let one_gb = 1u64 << 30;
        let full = plan_capacity(512, 32, one_gb, 1.0, 1.0);
        assert_eq!(full.bytes_per_node, 657);
        assert_eq!(full.nodes as u64, one_gb / 657);
        assert_eq!(full.slabs, full.nodes);

        // A margin only ever reduces the capacity, monotonically.
        let tight = plan_capacity(512, 32, one_gb, 1.0, 0.5);
        assert!(tight.nodes < full.nodes);
        assert!((tight.nodes as f64 / full.nodes as f64 - 0.5).abs() < 0.01);

        // More layers per node -> the same bytes buy fewer nodes, and slabs scale.
        let layered = plan_capacity(512, 32, one_gb, 1.5, 1.0);
        assert!(layered.nodes < full.nodes);
        assert_eq!(layered.slabs, (layered.nodes as f64 * 1.5).ceil() as usize);

        // A budget that cannot host even one node reports zero, so the caller
        // falls back to the single-backend path instead of spilling immediately.
        assert_eq!(plan_capacity(512, 32, 100, 1.0, 0.9).nodes, 0);
        // ... and layers below 1.0 are clamped rather than shrinking the maths.
        assert_eq!(plan_capacity(512, 32, one_gb, 0.0, 1.0), full);
    }

    #[test]
    fn planned_capacity_admits_more_nodes_than_its_estimate() {
        // The sizing must not under-count: a graph filled to the planned node
        // count still accepts nodes.
        let planned = plan_capacity(8, 4, 1 << 20, 1.0, 1.0);
        assert!(planned.nodes > 100, "sanity: {}", planned.nodes);
        let mut g = FlatGraph::with_limits(8, 4, planned.nodes, planned.slabs);
        for i in 0..planned.nodes as u32 {
            assert!(g.try_push_node(0, tid(i + 1), false, &[0u8; 8]), "node {}", i);
        }
        assert!(g.is_full());
        assert!(!g.try_push_node(0, tid(1), false, &[0u8; 8]));
    }

    #[test]
    fn checks_accept_a_well_formed_graph() {
        // Mutual links: every node has an incoming edge and the entry reaches all.
        let mut g = FlatGraph::new(4, 4);
        for (i, p) in [[0.0f32, 0.0], [1.0, 0.0], [2.0, 0.0]].iter().enumerate() {
            g.push_node(0, tid(i as u32 + 1), false, &[0u8; 4]);
            let _ = p;
        }
        g.set_list(0, 0, &[1]);
        g.set_list(1, 0, &[0, 2]);
        g.set_list(2, 0, &[1]);
        assert!(g.promote_entry(0));

        let c = check_lists(&g, 4);
        assert_eq!(c.nodes_without_incoming, 0);
        assert_eq!(c.reachable_from_entry, 3);
        assert!(c.is_healthy(4), "{}", c.summary(4));
    }

    #[test]
    fn checks_catch_a_disconnected_node_and_bad_links() {
        let mut g = FlatGraph::new(4, 4);
        for i in 0..4u32 {
            g.push_node(0, tid(i + 1), false, &[0u8; 4]);
        }
        g.set_list(0, 0, &[0, 1, 1]); // self-link plus a duplicate
        g.set_list(1, 0, &[0]);
        // node 2 is isolated; node 3 is reachable from nothing either
        assert!(g.promote_entry(0));

        let c = check_lists(&g, 4);
        assert_eq!(c.self_links, 1, "{}", c.summary(4));
        assert_eq!(c.duplicate_links, 1, "{}", c.summary(4));
        assert_eq!(c.nodes_without_incoming, 2, "{}", c.summary(4));
        assert_eq!(c.reachable_from_entry, 2, "{}", c.summary(4));
        assert!(!c.is_healthy(4), "an unhealthy graph must be reported");
    }

    #[test]
    fn checks_bound_reachability_by_the_watermark() {
        // A claimed-but-unwritten node must not be walked into, and must not count
        // as reachable.
        let mut g = FlatGraph::new(4, 4);
        g.push_node(0, tid(1), false, &[0u8; 4]);
        let claimed = g.claim_slot(0).expect("room");
        g.set_list(0, 0, &[claimed]);
        assert!(g.promote_entry(0));

        let c = check_lists(&g, 4);
        assert_eq!(c.nodes, 2);
        assert_eq!(c.published, 1);
        assert_eq!(c.reachable_from_entry, 1, "{}", c.summary(4));
        assert!(!c.is_healthy(4), "an unpublished reachable id is not healthy");
    }

    #[test]
    fn chunk_backed_graph_stores_vectors_in_the_chunk() {
        let (stride, cap) = (8usize, 4usize);
        let mut g = FlatGraph::with_limits(stride, cap, 8, 9);
        assert!(g.chunk.is_some(), "with_limits is the arena shape");
        assert_eq!(g.heap_bytes(), g.layout.total_bytes);

        let a = [1u8; 8];
        let b = [2u8; 8];
        g.push_node(0, tid(1), false, &a);
        g.push_node(0, tid(2), false, &b);
        assert_eq!(g.vector(0), &a, "vectors come back from the chunk");
        assert_eq!(g.vector(1), &b);
        assert!(g.vectors.is_empty(), "no Vec copy alongside the chunk");

        // Slab storage: written and read back through the chunk's ids region.
        g.set_list(0, 0, &[7, 8, 9]);
        assert_eq!(g.neighbors(0, 0), &[7, 8, 9]);
        assert!(
            g.ids.is_empty() && g.lens.is_empty(),
            "no Vec copy of the ids or the lens either"
        );
        g.set_list(0, 0, &[5]);
        assert_eq!(g.neighbors(0, 0), &[5], "a shorter list still reads back");
        {
            let chunk = g.chunk.as_ref().unwrap();
            let lens = chunk.region_u16(g.layout.lens);
            assert_eq!(lens[0], 1, "the length lives in the chunk's lens region");
            assert_eq!(lens[1], 0, "a slab nothing was written into reads empty");
        }
        assert_eq!(g.slab_usage(), (2, 9), "one slab per level-0 push");

        // The cursor is `slabs_used`, not `lens.len()` -- which is now 0 here.
        let claimed = g.claim_slot(2).expect("room for a 3-layer node");
        assert_eq!(g.slab_usage(), (5, 9), "a 3-layer claim reserves 3 slabs");
        assert_eq!(g.slab_base(claimed), Some(2));
        assert_eq!(g.len(), 3, "the node cursor advanced with the claim");
        assert_eq!(g.level(claimed), 2, "the level comes from the chunk's region");
        assert_eq!(g.level(0), 0, "and earlier nodes keep theirs");
        assert_eq!(
            g.chunk.as_ref().unwrap().region_bytes(g.layout.levels)[claimed as usize],
            2
        );
        assert!(g.levels.is_empty(), "no Vec copy of the levels either");
        // Layers beyond the node's level are not slabs of its own.
        assert_eq!(g.slab_base(0), Some(0));
        assert_eq!(g.neighbors(0, 1), &[] as &[u32], "level-0 node has one layer");
        // Claimed-but-unwritten storage reads as zeros from the zeroed chunk.
        assert_eq!(g.vector(claimed), &[0u8; 8]);
        g.publish(claimed, tid(3), false, &[3u8; 8]);
        assert_eq!(g.vector(claimed), &[3u8; 8]);
        assert_eq!(g.watermark(), 3);

        // The grow-mode graph still uses its Vecs, with the same cursor semantics.
        let mut g = FlatGraph::new(stride, cap);
        assert!(g.chunk.is_none());
        g.push_node(1, tid(1), false, &a);
        assert_eq!(g.vector(0), &a);
        assert!(!g.vectors.is_empty());
        assert_eq!(g.slab_usage(), (2, usize::MAX), "grow mode counts slabs too");
        assert_eq!(g.lens.len(), 2, "and keeps the length store in lockstep");
        assert_eq!(g.len(), 1, "the node cursor counts nodes, not slabs");
        assert_eq!(g.level(0), 1);
        assert_eq!(g.levels.len(), 1, "and keeps the level store in lockstep");
        assert_eq!(g.slab_base(0), Some(0));
    }

    #[test]
    fn push_node_publishes_immediately() {
        let mut g = FlatGraph::new(4, 2);
        g.push_node(0, tid(1), false, &[0u8; 4]);
        g.push_node(0, tid(2), false, &[0u8; 4]);
        assert_eq!(g.watermark(), g.len(), "single-threaded pushes publish at once");
    }

    #[test]
    fn watermark_advances_only_through_the_contiguous_prefix() {
        let mut g = FlatGraph::new(4, 2);
        for i in 0..3 {
            assert_eq!(g.claim_slot(0), Some(i));
        }
        assert_eq!((g.len(), g.watermark()), (3, 0), "nothing written yet");
        g.publish(1, tid(2), false, &[1u8; 4]);
        assert_eq!(g.watermark(), 0, "id 0 is still unwritten");
        g.publish(2, tid(3), false, &[2u8; 4]);
        assert_eq!(g.watermark(), 0, "the hole at id 0 still blocks the watermark");
        g.publish(0, tid(1), false, &[3u8; 4]);
        assert_eq!(g.watermark(), 3, "the prefix is complete now");
    }

    #[test]
    fn claimed_but_unpublished_nodes_are_invisible_to_the_search() {
        use crate::access_method::hnswsq::build::SearchScratch;
        use crate::access_method::hnswsq::flat_engine::search_layer_flat;
        use crate::access_method::hnswsq::quantize::HnswPrecision;

        let codec = Codec::new(HnswPrecision::Plain, 2);
        let mut g = FlatGraph::new(codec.vector_bytes(), 4);
        for (i, p) in [[0.0f32, 0.0], [1.0, 0.0]].iter().enumerate() {
            g.push_node(0, tid(i as u32 + 1), false, &codec.encode(&p.to_vec()));
        }
        // A third node is claimed (so it has an id and slabs) but not written:
        // it must not be reachable, and addressing it must not panic.
        let claimed = g.claim_slot(0).expect("room for a third node");
        assert_eq!(claimed, 2);
        assert_eq!(g.watermark(), 2);
        g.set_list(0, 0, &[1, claimed]); // even linked from a live node
        g.set_list(1, 0, &[0]);

        let mut scratch = SearchScratch::new();
        let hits = search_layer_flat(
            &codec,
            DistanceType::L2,
            &[0.0, 0.0],
            &g,
            &[(distance_l2(&[0.0, 0.0], &[0.0, 0.0]), 0)],
            8,
            0,
            &mut scratch,
        );
        let ids: Vec<u32> = hits.iter().map(|h| h.id).collect();
        assert_eq!(ids, vec![0, 1], "the claimed node is skipped, not indexed");
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
