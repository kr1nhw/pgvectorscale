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

use crate::access_method::hnswsq::arena::{arena_layout, ArenaLayout, Chunk, NodeLocks};
use super::arena::ArenaState;
use super::arena::SharedArena;
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
    /// Every array below now lives in this chunk (vectors, ids, lens, levels, tids,
    /// clamped, published, slab_off), so the `Vec` fields are empty in this mode and
    /// only `nodes_used`/`slabs_used` are kept on the side.  The `Vec` arms are the
    /// grow-mode backing, kept until the arena is the only mode.
    chunk: Option<Chunk>,
    layout: ArenaLayout,
    /// `slab_off[node]` = index of the node's layer-0 slab; a node with
    /// `level + 1` layers owns `slab_off[node] .. slab_off[node] + level + 1`.
    /// Empty when the graph is chunk-backed, like every other `Vec` here.
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
    /// When set, the cursors (claim, slabs, watermark, entry) live in this shared
    /// state instead of the fields below -- a parallel build shares them across
    /// workers, so `len`/`claim_slot`/`publish` must not keep private copies that
    /// would drift.  The `Vec`s stay the single-builder path.
    state: Option<*mut ArenaState>,
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
            state: None,
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
        // Every per-node array is a chunk region now, so no `Vec` capacity is
        // reserved for any of them -- `ids`/`lens` alone would have been up to
        // `max_slabs * cap * 4` bytes of dead allocation, and `nodes_used`/
        // `slabs_used` are the only cursors the graph keeps on the side.
        g.nodes_used = 0;
        g.slabs_used = 0;
        g
    }

    /// A graph over a shared arena: the same storage discipline as
    /// [`FlatGraph::with_limits`], but the cursors come from the segment's
    /// [`ArenaState`], so every worker that attaches sees one claim counter, one
    /// watermark and one entry point.  The caller guarantees single-threaded access
    /// to *this handle* -- concurrent workers each build their own, and the arena's
    /// node locks are what make that safe.
    pub fn in_arena(arena: &SharedArena) -> Self {
        let header = arena.header();
        let mut g = Self::new(header.stride, header.cap);
        g.max_nodes = header.max_nodes;
        g.max_slabs = header.max_slabs;
        g.layout = header.layout;
        // SAFETY: the arena's chunk lives in the segment for as long as the arena
        // handle; this graph is a second view of the same bytes.
        g.chunk = Some(unsafe { Chunk::attach(arena.chunk().base_ptr(), header.layout) });
        g.state = Some(arena.state() as *const ArenaState as *mut ArenaState);
        g
    }

    /// The cursors, whichever backing holds them.
    #[inline]
    fn nodes_used(&self) -> usize {
        match self.state {
            // SAFETY: the state lives in the segment for the arena's lifetime.
            Some(state) => unsafe { &*state }.claimed().0,
            None => self.nodes_used,
        }
    }

    #[inline]
    fn slabs_used(&self) -> usize {
        match self.state {
            // SAFETY: as above.
            Some(state) => unsafe { &*state }.claimed().1,
            None => self.slabs_used,
        }
    }

    #[inline]
    fn shared_state(&self) -> Option<&ArenaState> {
        // SAFETY: the state lives in the segment for the arena's lifetime.
        self.state.map(|s| unsafe { &*s })
    }

    /// Number of nodes every part of whose data is written.  Equal to `len()` in
    /// the single-threaded prototype and the legacy path; smaller while workers
    /// hold claimed-but-unwritten slots in the arena.
    #[inline]
    pub fn watermark(&self) -> usize {
        match self.shared_state() {
            Some(state) => state.watermark(),
            None => self.watermark,
        }
    }

    /// Reserve a node id and its slabs without writing any data.  The slot stays
    /// invisible to readers (see [`FlatGraph::watermark`]) until
    /// [`FlatGraph::publish`] completes it.  `None` when the budgets are
    /// exhausted — the driver then spills, exactly as for `try_push_node`.
    pub fn claim_slot(&mut self, level: u8) -> Option<u32> {
        let layers = level as usize + 1;
        let (id, base) = match self.shared_state() {
            // Shared: one CAS reserves the id *and* its slabs, so two workers can
            // never be handed the same slab range.
            Some(state) => {
                let (id, slab) = state.claim(layers, self.max_nodes, self.max_slabs)?;
                (id, slab)
            }
            None => {
                if self.len() >= self.max_nodes || self.slabs_used + layers > self.max_slabs {
                    return None;
                }
                let id = self.nodes_used as u32;
                self.nodes_used += 1;
                (id, self.slabs_used as u32)
            }
        };
        match self.chunk.as_mut() {
            Some(chunk) => chunk.region_bytes_mut(self.layout.levels)[id as usize] = level,
            None => self.levels.push(level),
        }
        debug_assert!(self.chunk.is_some() || self.levels.len() == self.nodes_used);
        let i = id as usize;
        match self.chunk.as_mut() {
            // Chunk mode: every per-node region is claimed here.  The flags are zero
            // with the chunk and a slot is never reused, but the tid *must* be
            // written: an all-zero `ItemPointer` is block 0 / offset 0, not the
            // invalid one, so a reader that slipped past the watermark would find a
            // real-looking pointer instead of "no row yet".
            Some(chunk) => {
                chunk.region_slice_mut::<ItemPointer>(self.layout.tids)[i] =
                    ItemPointer::new_invalid();
                chunk.region_bytes_mut(self.layout.clamped)[i] = 0;
                chunk.region_bytes_mut(self.layout.published)[i] = 0;
            }
            None => {
                self.tids.push(ItemPointer::new_invalid());
                self.clamped.push(false);
                self.published_flags.push(false);
            }
        }
        if self.chunk.is_none() {
            self.vectors.resize(self.vectors.len() + self.stride, 0);
        }
        match self.chunk.as_mut() {
            Some(chunk) => chunk.region_u32_mut(self.layout.slab_off)[i] = base,
            None => self.slab_off.push(base),
        }
        if self.state.is_none() {
            self.slabs_used += layers;
        }
        if self.chunk.is_none() {
            // Grow mode only: `lens`/`ids` are the storage.  They are kept in
            // lockstep with the cursor, filled with zero lengths exactly like the
            // chunk's zeroed region, so a claimed-but-unwritten slab reads empty.
            self.lens.resize(self.slabs_used(), 0);
            self.ids.resize(self.slabs_used() * self.cap, 0);
        }
        Some(id)
    }

    /// Write a claimed node's data and publish it.  Publication is what makes the
    /// node observable; the watermark advances only through the contiguous
    /// published prefix, so several workers completing out of order is fine.
    pub fn publish(&mut self, id: u32, tid: ItemPointer, clamped: bool, encoded: &[u8]) {
        assert_eq!(encoded.len(), self.stride, "encoded vector must match stride");
        let i = id as usize;
        assert!(i < self.len(), "publish of an unclaimed id");
        assert!(!self.is_published(i), "node {} published twice", id);
        match self.chunk.as_mut() {
            Some(chunk) => chunk.region_slice_mut::<ItemPointer>(self.layout.tids)[i] = tid,
            None => self.tids[i] = tid,
        }
        match self.chunk.as_mut() {
            Some(chunk) => chunk.region_bytes_mut(self.layout.clamped)[i] = clamped as u8,
            None => self.clamped[i] = clamped,
        }
        let start = i * self.stride;
        match self.chunk.as_mut() {
            Some(chunk) => {
                chunk.region_bytes_mut(self.layout.vectors)[start..start + self.stride]
                    .copy_from_slice(encoded);
            }
            None => self.vectors[start..start + self.stride].copy_from_slice(encoded),
        }
        match self.chunk.as_mut() {
            Some(chunk) => chunk.region_bytes_mut(self.layout.published)[i] = 1,
            None => self.published_flags[i] = true,
        }
        match self.shared_state() {
            // Shared: whoever closes the gap moves the watermark, and the contiguity
            // rule is the state's, not this handle's.
            Some(state) => {
                state.advance_watermark(|i| self.is_published(i), self.nodes_used());
            }
            None => {
                while self.watermark < self.nodes_used && self.is_published(self.watermark) {
                    self.watermark += 1;
                }
            }
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
        // Through the cursor methods: with a shared state the fields are 0, and a
        // pre-check that reads them would report "room" right up to the point where
        // `push_node`'s claim panics on exhaustion -- `false` is the driver's spill
        // signal, so it has to be the real answer.
        if self.len() >= self.max_nodes || self.slabs_used() + layers > self.max_slabs {
            return false;
        }
        self.push_node(level, tid, clamped, encoded);
        true
    }

    /// Whether either budget is exhausted (always `false` in grow mode).
    pub fn is_full(&self) -> bool {
        self.len() >= self.max_nodes || self.slabs_used() >= self.max_slabs
    }

    /// `(slabs used, slab budget)`.
    pub fn slab_usage(&self) -> (usize, usize) {
        (self.slabs_used(), self.max_slabs)
    }

    /// `(nodes used, node budget)`.
    pub fn node_usage(&self) -> (usize, usize) {
        (self.len(), self.max_nodes)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nodes_used()
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
    /// The graph's entry point -- the third cursor, and the third place a private copy
    /// would go wrong: with a shared state the field is `None` forever, so every worker
    /// would see an empty graph, plan an empty list and produce a graph with no edges
    /// at all (which is exactly what the structure gate caught).
    pub fn entry(&self) -> Option<u32> {
        match self.shared_state() {
            Some(state) => match state.entry() {
                super::arena::NO_ENTRY => None,
                id => Some(id),
            },
            None => self.entry,
        }
    }

    #[inline]
    pub fn entry_level(&self) -> usize {
        match self.shared_state() {
            Some(state) => state.entry_level(),
            None => self.entry_level,
        }
    }

    /// Promote `id` to entry point when its level is higher than the current
    /// one (the caller does this after the node's own list is published, so a
    /// searcher never lands on an unlinked entry).
    pub fn promote_entry(&mut self, id: u32) -> bool {
        let level = self.level(id) as usize;
        if level > self.entry_level() || self.entry().is_none() {
            match self.shared_state() {
                // One store, level first: a reader that sees the id must see the level
                // that belongs with it.  Callers serialize promotion (the driver
                // promotes after the workers stop; a worker never promotes at all).
                Some(state) => state.set_entry(id, level),
                None => {
                    self.entry = Some(id);
                    self.entry_level = level;
                }
            }
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
        debug_assert!(self.chunk.is_some() || self.slab_off.len() == self.nodes_used);
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

    /// Write a node's list without a `&mut` borrow -- the parallel write path.
    ///
    /// # Safety
    ///
    /// The caller must be the only writer of `id`'s layer-`layer` slab: either the
    /// worker that claimed `id` (its own list, before publishing), or a holder of
    /// `id`'s node write lock.  Readers of the same node must hold its read lock.
    /// Only valid for a chunk-backed graph -- a grow-mode `Vec` would have to
    /// reallocate, which no reader could survive.
    pub unsafe fn set_list_concurrent(&self, id: u32, layer: usize, ids: &[u32]) {
        assert!(
            ids.len() <= self.cap,
            "neighbour list of {} exceeds slab capacity {}",
            ids.len(),
            self.cap
        );
        let chunk = self
            .chunk
            .as_ref()
            .expect("concurrent writes need the chunk backing");
        let slab = self.slab(id, layer).expect("layer beyond the node's level");
        let base = slab * self.cap;
        let n = ids.len();
        {
            // SAFETY: this caller is the only writer of this slab's ids (see above),
            // and the region's alignment and length are checked by the accessor.
            let view = unsafe { chunk.region_u32_concurrent(self.layout.ids) };
            view[base..base + n].copy_from_slice(ids);
        }
        // SAFETY: as above, for the length that belongs to the same slab.  The write
        // is last, so a concurrent reader can never see a length pointing at ids that
        // have not been written yet.
        unsafe { chunk.region_u16_concurrent(self.layout.lens)[slab] = n as u16 };
    }

    /// Write a *target* node's list under its node lock -- the backlink step, where
    /// the writer does not own the node.  This is the only way a worker may touch a
    /// list other than its own.
    pub fn set_list_locked(&self, locks: &NodeLocks, id: u32, layer: usize, ids: &[u32]) {
        // The node write lock excludes every other writer of this node's storage, and
        // every reader of it holds the read lock.
        let _guard = locks.write(id);
        // SAFETY: as above -- the write lock is what makes this the only writer.
        unsafe { self.set_list_concurrent(id, layer, ids) };
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
        match &self.chunk {
            Some(chunk) => chunk.region_slice::<ItemPointer>(self.layout.tids)[id as usize],
            None => self.tids[id as usize],
        }
    }

    #[inline]
    pub fn clamped(&self, id: u32) -> bool {
        match &self.chunk {
            Some(chunk) => chunk.region_bytes(self.layout.clamped)[id as usize] != 0,
            None => self.clamped[id as usize],
        }
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
    /// Whether node `i`'s data is written and observable (see `watermark`).
    #[inline]
    fn is_published(&self, i: usize) -> bool {
        match &self.chunk {
            Some(chunk) => chunk.region_bytes(self.layout.published)[i] != 0,
            None => self.published_flags[i],
        }
    }

    #[inline]
    fn slab(&self, id: u32, layer: usize) -> Option<usize> {
        let base = self.slab_base(id)?;
        (layer <= self.level(id) as usize).then_some(base + layer)
    }

    #[inline]
    fn slab_base(&self, id: u32) -> Option<usize> {
        // Bound by `nodes_used`, not by the array length: in chunk mode the region is
        // `max_nodes` long, so a length check would accept unclaimed ids and hand out
        // slabs for nodes that do not exist yet.
        if id as usize >= self.nodes_used() {
            return None;
        }
        let base = match &self.chunk {
            Some(chunk) => chunk.region_u32(self.layout.slab_off)[id as usize],
            None => self.slab_off[id as usize],
        } as usize;
        // A node with `level + 1` layers must own that many slabs.
        let layers = self.level(id) as usize + 1;
        (base + layers <= self.slabs_used()).then_some(base)
    }
}

/// A graph handle that may be handed to another worker.
///
/// `Send`/`Sync` are asserted **here**, on the shared construction, and not on
/// `FlatGraph`: only a chunk-backed graph over an arena segment can be shared -- the
/// growth, the cursors and the locking discipline are all the arena's -- while a
/// grow-mode graph owns `Vec`s that a second writer would race.  Putting the impls on
/// `FlatGraph` would silently permit exactly that, and nothing in the type would say
/// which one a caller had.
pub struct SharedGraph(FlatGraph);

// SAFETY: the inner graph was built by `FlatGraph::in_arena`, so its storage is the
// arena's segment and its cursors are the arena's `ArenaState`.  Every mutation of a
// node the handle does not own goes through that node's lock (`set_list_locked`), the
// claim/publish protocol is atomic, and no `Vec` field is populated in this mode.
unsafe impl Send for SharedGraph {}
unsafe impl Sync for SharedGraph {}

impl SharedGraph {
    /// A worker's handle on a shared arena.  Cheap: it is pointers and offsets.
    pub fn new(arena: &SharedArena) -> Self {
        Self(FlatGraph::in_arena(arena))
    }

    pub fn graph(&self) -> &FlatGraph {
        &self.0
    }

    /// Mutable access for what only the owner of a node may do: claiming it, writing
    /// its own lists, and publishing it.  Nothing here touches another worker's node.
    pub fn graph_mut(&mut self) -> &mut FlatGraph {
        &mut self.0
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

// `any(test, feature = "pg_test")` + `#[pgrx::pg_schema]`, not just `cfg(test)`: the
// build script emits the SQL for `#[pg_test]` functions and compiles the crate
// *without* `cfg(test)`, so a `cfg(test)`-only module would register the Rust test
// but never create its `tests.<name>()` function.
#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
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

        // Per-node flags: chunk bytes, not `Vec<bool>`s.
        assert!(g.clamped.is_empty() && g.published_flags.is_empty());
        assert!(!g.clamped(0) && !g.clamped(claimed), "unclamped by default");
        g.push_node(0, tid(4), true, &[4u8; 8]);
        assert!(g.clamped(3), "the saturation flag lands in the clamped region");
        assert_eq!(g.watermark(), 4);
        {
            let chunk = g.chunk.as_ref().unwrap();
            assert_eq!(chunk.region_bytes(g.layout.clamped)[3], 1);
            assert_eq!(chunk.region_bytes(g.layout.published)[3], 1);
            assert_eq!(chunk.region_bytes(g.layout.clamped)[0], 0, "not neighbours");
        }
        assert!(
            g.tids.is_empty() && g.slab_off.is_empty(),
            "the last two arrays are in the chunk as well"
        );
        assert!(g.tid(3).is_valid(), "a published node's tid comes back");
        // The id bounds the lookup, not the (max_nodes-long) region: an id that was
        // never claimed must not resolve to slabs, and its tid reads invalid rather
        // than as the zeroed block 0 / offset 0 that an untouched region holds.
        assert_eq!(g.slab_base(7), None, "unclaimed id has no slabs");
        assert!(!g.tid(7).is_valid(), "unclaimed slot was written as invalid");
        assert_eq!(g.slab_base(0), Some(0), "and claimed ids still resolve");
        assert_eq!(g.slab_base(3), Some(5), "after the 3-layer claim's slabs");

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
        assert!(!g.clamped.is_empty() && !g.published_flags.is_empty());
        assert!(!g.clamped(0) && g.len() == 1);
        assert!(!g.tids.is_empty() && !g.slab_off.is_empty());
        assert!(g.tid(0).is_valid());
        assert_eq!(g.slab_base(5), None, "grow mode rejects unclaimed ids too");
        assert_eq!(g.slab_base(0), Some(0));
    }

    #[test]
    fn the_arena_is_valid_at_any_mapping_address() {
        // The property the shared-memory design rests on: nothing in the chunk is
        // addressed absolutely, so the same bytes work when mapped somewhere else.
        // A dsm segment is mapped at a different address in every participant, and
        // the leader's copy is not the workers' copy -- so move the bytes and require
        // every accessor to answer identically.
        let (stride, cap) = (8usize, 4usize);
        let mut g = FlatGraph::with_limits(stride, cap, 8, 9);
        g.push_node(0, tid(1), false, &[1u8; 8]);
        g.push_node(2, tid(2), true, &[2u8; 8]);
        g.set_list(0, 0, &[1, 2]);
        g.set_list(1, 0, &[0]);
        g.entry = Some(1);

        let layout = g.layout;
        let source = g.chunk.as_ref().unwrap().as_bytes().to_vec();
        let mut moved: Vec<u64> = source
            .chunks_exact(8)
            .map(|w| u64::from_ne_bytes(w.try_into().unwrap()))
            .collect();
        assert_ne!(
            moved.as_ptr() as *const u8,
            g.chunk.as_ref().unwrap().as_bytes().as_ptr(),
            "the test is vacuous if the copy landed at the same address"
        );

        // SAFETY: `moved` is a live, 8-byte-aligned, byte-identical copy of the chunk
        // that outlives `g`, and nothing else writes it.
        g.chunk = Some(unsafe { Chunk::attach(moved.as_mut_ptr(), layout) });
        assert!(!g.chunk.as_ref().unwrap().is_owned());

        assert_eq!(g.vector(0), &[1u8; 8], "vectors survive relocation");
        assert_eq!(g.vector(1), &[2u8; 8]);
        assert_eq!(g.neighbors(0, 0), &[1, 2], "ids and lens do too");
        assert_eq!(g.neighbors(1, 0), &[0]);
        assert_eq!(g.level(1), 2);
        assert_eq!(g.slab_base(1), Some(1));
        assert_eq!(g.len(), 2);
        assert_eq!(g.slab_usage(), (4, 9), "a level-2 node owns three slabs");
        assert!(g.tid(0).is_valid() && g.tid(1).is_valid(), "tids survive");
        assert!(g.clamped(1) && !g.clamped(0), "flags survive");

        // It is still a live arena: writes go to the moved bytes.
        g.set_list(1, 0, &[0, 0]);
        assert_eq!(g.neighbors(1, 0), &[0, 0]);
        let owner_byte = moved[layout.levels.offset / 8];
        assert!(owner_byte != 0, "and the owner sees them in the moved segment");
    }

    #[pgrx::pg_test]
    fn a_graph_over_a_shared_arena_shares_its_cursors() {
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
        let tranche = super::super::arena::register_tranche(c"hnswsq_shared_graph_test");
        let (stride, cap, nodes, slabs) = (8usize, 4usize, 8usize, 9usize);
        let arena = unsafe { super::super::arena::SharedArena::allocate(toc, stride, cap, nodes, slabs, tranche) };

        // Two handles, the way two workers would each build their own.
        let mut a = FlatGraph::in_arena(&arena);
        let peer_arena = unsafe { super::super::arena::SharedArena::attach(toc) };
        let mut b = FlatGraph::in_arena(&peer_arena);

        assert_eq!(a.len(), 0);
        assert_eq!(a.slab_usage(), (0, slabs), "capacities come from the header");
        assert!(a.chunk.is_some() && a.levels.is_empty(), "storage is the segment's");

        // Claim and publish through one handle, observe through the other.
        let id0 = a.claim_slot(0).expect("room");
        assert_eq!(id0, 0);
        assert_eq!(b.len(), 1, "the claim cursor is shared");
        assert_eq!(a.watermark(), 0, "claimed but unpublished is not visible");
        a.publish(id0, crate::util::ItemPointer::new_invalid(), false, &[9u8; 8]);
        assert_eq!(b.watermark(), 1, "publishing advances the shared watermark");
        assert_eq!(b.vector(0), &[9u8; 8], "and the peer sees the data");

        // A gap left by an out-of-order publisher is the same for both handles.
        let id1 = a.claim_slot(1).expect("room for a 2-layer node");
        let id2 = b.claim_slot(0).expect("room");
        assert_eq!((id1, id2), (1, 2), "ids stay unique across handles");
        b.publish(id2, crate::util::ItemPointer::new_invalid(), false, &[7u8; 8]);
        assert_eq!(a.watermark(), 1, "id 1 is still unwritten");
        a.publish(id1, crate::util::ItemPointer::new_invalid(), true, &[8u8; 8]);
        assert_eq!(a.watermark(), 3, "closing the gap moves it past both");

        assert_eq!(a.level(1), 1);
        assert_eq!(a.clamped(1), true);
        assert_eq!(a.slab_base(2), Some(3), "slab ranges come from the shared CAS");
        assert_eq!(a.slab_usage(), (4, slabs), "1 + 2 + 1 slabs, no double booking");

        drop(b);
        drop(peer_arena);
        drop(a);
        drop(arena);
        // SAFETY: no handle references the mapping any more.
        unsafe { pgrx::pg_sys::dsm_detach(seg) };
    }

    #[pgrx::pg_test]
    fn two_threads_run_the_engine_over_one_arena() {
        // The design's central claim: two writers can build one graph, each searching
        // the shared arena and backlinking under the target's lock.  Threads again, so
        // the *cross-process* LWLock case remains the build's to prove -- but the
        // protocol, the claim/publish watermark, the search's watermark bound and the
        // lock discipline all run here for real.
        use super::super::build::SearchScratch;
        use super::super::flat_engine::{apply_flat, plan_flat, FlatPairBuf, Locking};
        use super::super::quantize::HnswPrecision;
        use crate::access_method::distance::{distance_l2, DistanceType};

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
        let codec = Codec::new(HnswPrecision::Plain, 2);
        let (m, m0, efc) = (2usize, 4usize, 8usize);
        let (stride, cap, nodes, slabs) = (codec.vector_bytes(), m0, 64usize, 128usize);
        let tranche = super::super::arena::register_tranche(c"hnswsq_two_threads_test");
        let arena =
            unsafe { super::super::arena::SharedArena::allocate(toc, stride, cap, nodes, slabs, tranche) };
        let per = 20usize;
        // The workers' rows; the seed above is node 0, so the graph ends up with one
        // more node than this.
        let total = (2 * per) as u32;

        // The leader seeds the entry point *before* any worker starts.  Without one,
        // every search finds nothing and every node is born with an empty list -- this
        // test asserted exactly that first (40 published nodes, `max_list_len=0`,
        // `reachable=0`), which is the failure mode the rendezvous exists to prevent.
        {
            let mut seed = SharedGraph::new(&arena);
            let mut scratch = SearchScratch::new();
            // The dimension, not `m0`: the buffer holds one vector per pair.
            let mut buf = FlatPairBuf::new(2);
            let v = [0.0f32, 0.0];
            let mut encoded = Vec::with_capacity(stride);
            let clamped = codec.encode_into(&v, &mut encoded);
            let subject = codec.decode(&encoded);
            let id = seed.graph_mut().claim_slot(0).expect("room for the seed");
            let plan = plan_flat(
                &codec, DistanceType::L2, distance_l2, seed.graph(), &mut scratch, &mut buf,
                id, 0, &subject, m, m0, efc, false,
            );
            // `SoleWriter`: the seed is what makes an entry exist, so it must be
            // promoted here rather than left to the driver's post-join promotion.
            apply_flat(
                &codec, distance_l2, seed.graph_mut(), &Locking::SoleWriter, &mut buf,
                id, 0, plan, m, m0,
            );
            seed.graph_mut()
                .publish(id, crate::util::ItemPointer::new(1, 1), clamped, &encoded);
            assert_eq!(seed.graph().entry(), Some(id), "the seed owns the entry");
        }
        // Locks for the workers.  NOT the arena's: those are LWLocks, PostgreSQL FFI
        // may only be called from the backend's main thread (pgrx enforces it:
        // "postgres FFI may not be called from multiple threads"), and a spawned
        // thread is no substitute for a worker *process*.  In the real build each
        // worker is its own process with its own main thread, and the arena's LWLocks
        // are what exclude them; here the storage is still the shared segment, and
        // `RwLock`s give the threads the same exclusion.  The cross-process LWLock
        // case is therefore the worker sweep's to prove, not this test's.
        let locks = NodeLocks::new(nodes);

        // Rendezvous: the nodes that exist when the workers start.
        arena.state().set_start_nodes(1);
        assert_eq!(arena.state().start_nodes(), 1);

        // Panics inside a scoped thread lose their message (the hook writes to the
        // backend's stderr), so each worker catches its own and hands the payload back.
        fn payload(e: Box<dyn std::any::Any + Send>) -> String {
            if let Some(s) = e.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic>".to_string()
            }
        }
        let built: Vec<Result<Vec<u32>, String>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..2u32)
                .map(|w| {
                    let arena = &arena;
                    let codec = &codec;
                    let locks = &locks;
                    scope.spawn(move || {
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                        let mut handle = SharedGraph::new(arena);
                        let mut scratch = SearchScratch::new();
                        // The dimension, not `m0`: the buffer holds one vector per pair.
            let mut buf = FlatPairBuf::new(2);
                        let mut mine = Vec::new();
                        for k in 0..per as u32 {
                            let x = (w * per as u32 + k) as f32;
                            let v = [x, (k % 3) as f32];
                            let mut encoded = Vec::with_capacity(stride);
                            let clamped = codec.encode_into(&v, &mut encoded);
                            let subject = codec.decode(&encoded);
                            let Some(id) = handle.graph_mut().claim_slot(0) else {
                                break;
                            };
                            let plan = plan_flat(
                                codec,
                                DistanceType::L2,
                                distance_l2,
                                handle.graph(),
                                &mut scratch,
                                &mut buf,
                                id,
                                0,
                                &subject,
                                m,
                                m0,
                                efc,
                                false,
                            );
                            apply_flat(
                                codec,
                                distance_l2,
                                handle.graph_mut(),
                                &Locking::Locks(locks),
                                &mut buf,
                                id,
                                0,
                                plan,
                                m,
                                m0,
                            );
                            // Publish last: the node becomes searchable only once its
                            // vector and its own lists are all written.
                            handle.graph_mut().publish(
                                id,
                                crate::util::ItemPointer::new(1, 1),
                                clamped,
                                &encoded,
                            );
                            mine.push(id);
                        }
                        mine
                        }))
                        .map_err(payload)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        let built: Vec<Vec<u32>> = built
            .into_iter()
            .map(|r| r.unwrap_or_else(|e| panic!("a worker panicked: {}", e)))
            .collect();

        let mut all: Vec<u32> = built.iter().flatten().copied().collect();
        let claimed = all.len();
        assert_eq!(claimed, total as usize, "every row became a node: {:?}", built);
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), claimed, "no id was handed out twice");
        assert_eq!(
            arena.state().watermark(),
            claimed + 1,
            "every node was published (the workers' plus the seed), so the watermark \
             reached the end"
        );

        // The arena's own structural gate, over the graph two threads built.
        let g = FlatGraph::in_arena(&arena);
        let checks = check_lists(&g, cap);
        assert_eq!(checks.nodes, claimed + 1, "the workers' nodes plus the seed");
        assert_eq!(checks.self_links, 0, "no node links to itself: {}", checks.summary(cap));
        assert_eq!(
            checks.duplicate_links, 0,
            "no list repeats an id: {}",
            checks.summary(cap)
        );
        assert!(
            checks.max_list_len > 0,
            "the backlinks actually landed: {}",
            checks.summary(cap)
        );
        assert!(
            checks.max_list_len <= cap,
            "capacity respected: {}",
            checks.summary(cap)
        );
        assert!(
            checks.reachable_from_entry as usize >= claimed / 2,
            "the seeded entry reaches the graph both threads built: {}",
            checks.summary(cap)
        );

        drop(g);
        drop(arena);
        // SAFETY: no handle references the mapping any more.
        unsafe { pgrx::pg_sys::dsm_detach(seg) };
    }

    #[pgrx::pg_test]
    fn the_leader_promotes_the_best_entry_after_the_workers_stop() {
        // Workers never promote the entry (every insert would serialize on one cache line),
        // so the leader seeds one before launching and re-promotes after the join -- because
        // a higher-level node may have appeared in between.  This pins that contract,
        // including the part that is easy to get wrong: a claimed-but-unpublished slot has a
        // level but no list yet, and must not become the entry.
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
        let tranche = super::super::arena::register_tranche(c"hnswsq_entry_promotion_test");
        let (stride, cap, nodes, slabs) = (8usize, 4usize, 8usize, 24usize);
        let arena =
            unsafe { super::super::arena::SharedArena::allocate(toc, stride, cap, nodes, slabs, tranche) };
        let mut g = FlatGraph::in_arena(&arena);

        // The leader's seed: a search needs an entry before any worker starts.
        let a = g.claim_slot(0).expect("room");
        g.publish(a, tid(1), false, &[1u8; 8]);
        assert!(g.promote_entry(a), "the seed becomes the entry");

        // What workers do: claim, publish, never promote.
        let b = g.claim_slot(1).expect("room");
        g.publish(b, tid(2), false, &[2u8; 8]);
        let c = g.claim_slot(2).expect("room");
        g.publish(c, tid(3), false, &[3u8; 8]);
        assert_eq!(g.entry(), Some(a), "no worker promoted anything");

        // The leader's post-join promotion picks the highest level.
        assert_eq!(
            super::super::driver::promote_best_entry(&mut g),
            Some((c, 2)),
            "the level-2 node wins"
        );
        assert_eq!(g.entry(), Some(c));
        assert_eq!(g.entry_level(), 2);

        // A claimed slot with a higher level is *not* entry material: it has no list yet.
        let d = g.claim_slot(2).expect("room");
        assert!(d > c);
        assert_eq!(
            super::super::driver::promote_best_entry(&mut g),
            Some((c, 2)),
            "only published nodes are considered"
        );

        // A peer handle sees the promoted entry, because it lives in the arena.
        let peer_arena = unsafe { super::super::arena::SharedArena::attach(toc) };
        let peer = FlatGraph::in_arena(&peer_arena);
        assert_eq!(peer.entry(), Some(c));
        assert_eq!(peer.entry_level(), 2);

        drop(peer);
        drop(peer_arena);
        drop(g);
        drop(arena);
        // SAFETY: no handle references the mapping any more.
        unsafe { pgrx::pg_sys::dsm_detach(seg) };
    }

    #[pgrx::pg_test]
    fn a_locked_write_reaches_a_peer_handle() {
        // `set_list_locked` is the backlink path: writing a node the caller does not
        // own, under that node's lock, through the shared chunk rather than a `&mut`.
        // Contention itself is covered at the arena level (see arena.rs), where the
        // send/sync argument is already made.
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
        let tranche = super::super::arena::register_tranche(c"hnswsq_locked_write_test");
        let (stride, cap, nodes, slabs) = (8usize, 4usize, 8usize, 16usize);
        let arena =
            unsafe { super::super::arena::SharedArena::allocate(toc, stride, cap, nodes, slabs, tranche) };
        let mut writer = FlatGraph::in_arena(&arena);
        for i in 0..3u32 {
            let id = writer.claim_slot(0).expect("room");
            writer.publish(id, tid(i + 1), false, &[i as u8; 8]);
        }
        let peer_arena = unsafe { super::super::arena::SharedArena::attach(toc) };
        let peer = FlatGraph::in_arena(&peer_arena);

        writer.set_list_locked(arena.locks(), 1, 0, &[2, 0]);
        assert_eq!(peer.neighbors(1, 0), &[2, 0], "the peer sees the locked write");
        writer.set_list(1, 0, &[]);
        assert_eq!(peer.neighbors(1, 0), &[] as &[u32], "and an empty list too");

        // The write path checks capacity rather than overrunning the slab.
        let too_long = [0u32; 5];
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            writer.set_list_locked(arena.locks(), 1, 0, &too_long);
        }));
        assert!(caught.is_err(), "over-capacity lists are a programming error");

        drop(peer);
        drop(peer_arena);
        drop(writer);
        drop(arena);
        // SAFETY: no handle references the mapping any more.
        unsafe { pgrx::pg_sys::dsm_detach(seg) };
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
