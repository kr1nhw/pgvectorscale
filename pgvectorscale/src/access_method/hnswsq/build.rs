//! hnswsq index build.
//!
//! Shape of the build (mirrors the IVF streaming design and pgvector's hybrid
//! build):
//!
//! - `f8` (SQ8) only: **pass 1** reservoir-samples the heap to calibrate the
//!   per-dimension min/max ranges.  The training-free layouts (`plain`,
//!   `ieeefp16`, `ieeefp8`) skip straight to the row stream.
//! - **pass 2** streams rows into an in-memory HNSW graph while the estimated
//!   footprint fits `maintenance_work_mem`.  When the budget is exhausted (or
//!   for `CREATE INDEX CONCURRENTLY`, where a bulk writeout cannot race with
//!   live inserters), the memory graph is flushed to node pages in one
//!   sequential writeout and the remaining rows flow through the regular disk
//!   insert path (`insert::insert_vector`).
//!
//! The bulk writeout precomputes every node's final `ItemPointer` (pages
//! extend contiguously during a non-concurrent build — the heap ShareLock
//! excludes DML — and offsets are assigned in add order), so each node is
//! written exactly once with its final neighbor lists: append-only, no
//! rewrite pass.

use pgrx::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::access_method::distance::{preprocess_cosine, DistanceFn, DistanceType};
use crate::access_method::hnswsq::graph::{
    distance_encoded, greedy_descent, random_level, search_layer, ExpandResult, GraphAccess,
    HeapItem, ProbeResult, SearchHit, VisitData,
};
use crate::access_method::hnswsq::flat_engine::{apply_flat, plan_flat, FlatPairBuf, Locking};
use crate::access_method::hnswsq::flat_graph::FlatGraph;
use crate::access_method::hnswsq::insert::{codec_for, insert_vector, InsertCtx};
use crate::access_method::hnswsq::meta_page::HnswMetaPage;
use crate::access_method::hnswsq::node::{
    compute_max_level, item_fits, max_dim_for_page, probe_serialized_len, HnswNode,
};
use crate::access_method::hnswsq::options::{TSVHnswOptions, DEFAULT_SAMPLE_SIZE};
use crate::access_method::hnswsq::quantize::{Codec, Sq8Calibration};
use crate::access_method::hnswsq::HNSWSQ_DISTANCE_TYPE_PROC;
use crate::access_method::node::WriteableNode;
use crate::access_method::pg_vector::PgVectorInternal;
use crate::util::page::{tsv_fresh_page_capacity, PageType, WritablePage};
use crate::util::ItemPointer;

/// Per-node in-memory overhead beyond the encoded vector and neighbor ids
/// (Vec headers, tid, level, slack).  Used for the maintenance_work_mem budget.
const MEM_OVERHEAD_PER_NODE: u64 = 224;

/// Build state for the SQ8 calibration pass (reservoir sampling).
struct SampleState {
    sample: Vec<Vec<f32>>,
    sample_size: usize,
    nrows: u64,
    distance_type: DistanceType,
    rng: SmallRng,
}

/// In-memory HNSW graph (local u32 ids).
struct MemGraph {
    levels: Vec<u8>,
    tids: Vec<ItemPointer>,
    clamped: Vec<bool>,
    /// Encoded vectors (dim × elem_bytes each).
    vectors: Vec<Vec<u8>>,
    /// `[node][layer]` → valid neighbor ids (no padding in RAM).
    neighbors: Vec<Vec<Vec<u32>>>,
    /// `[node][layer][i]` → distance from the list owner to neighbor `i`
    /// (same order as `neighbors`).  Build-only: lets the backlink re-prune
    /// reuse the distances it computed when the list was produced, instead of
    /// re-deriving them for every revision.
    list_dists: Vec<Vec<Vec<f32>>>,
    /// `[node][layer]` → bitmask over the stored neighbors: bit `i` set means
    /// neighbor `i` was **accepted by the occlusion heuristic**; clear means it
    /// was pruned and only re-added by the closest-pruned backfill.  The
    /// incremental backlink re-prune needs this split to reproduce the
    /// heuristic's result exactly (see [`backlink_prune_mem`]).  Build-only,
    /// never written to disk.
    list_masks: Vec<Vec<[u64; 2]>>,
    entry: Option<u32>,
    entry_level: usize,
}

/// Maximum supported neighbor-list length for the build-time masks
/// (`m0 <= 100` per the reloption bounds, so two u64 words always suffice).
const LIST_MASK_WORDS: usize = 2;

impl MemGraph {
    fn new() -> Self {
        Self {
            levels: Vec::new(),
            tids: Vec::new(),
            clamped: Vec::new(),
            vectors: Vec::new(),
            neighbors: Vec::new(),
            list_dists: Vec::new(),
            list_masks: Vec::new(),
            entry: None,
            entry_level: 0,
        }
    }

    fn len(&self) -> usize {
        self.tids.len()
    }

    /// Register a new node with `level + 1` layers.
    fn push_node(&mut self, level: u8, tid: ItemPointer, clamped: bool, encoded: Vec<u8>) {
        self.levels.push(level);
        self.tids.push(tid);
        self.clamped.push(clamped);
        self.vectors.push(encoded);
        let layers = level as usize + 1;
        self.neighbors.push(vec![Vec::new(); layers]);
        self.list_dists.push(vec![Vec::new(); layers]);
        self.list_masks.push(vec![[0u64; LIST_MASK_WORDS]; layers]);
    }

    /// Structural gate on the in-memory graph, mirroring `flat_graph::check_lists`
    /// so both engines report the same numbers: `(nodes, no_incoming, reachable,
    /// self_links, duplicates, max_len)`.  This is the control that says whether the
    /// flat engine's connectivity is normal for HNSW at this scale or a defect of
    /// the append/shrink policy.
    fn list_checks(&self, cap: usize) -> (usize, usize, usize, usize, usize, usize) {
        let n = self.len();
        let mut incoming = vec![0usize; n];
        let (mut self_links, mut duplicates, mut max_len) = (0usize, 0usize, 0usize);
        for node in 0..n {
            let list = &self.neighbors[node][0];
            max_len = max_len.max(list.len());
            for (i, &nb) in list.iter().enumerate() {
                let ni = nb as usize;
                if ni == node {
                    self_links += 1;
                }
                if list[..i].contains(&nb) {
                    duplicates += 1;
                }
                if ni < n {
                    incoming[ni] += 1;
                }
            }
        }
        let _ = cap; // capacity is reported by the caller's checks
        let no_incoming = incoming.iter().filter(|&&c| c == 0).count();
        let mut reachable = 0usize;
        if let Some(entry) = self.entry.filter(|&e| (e as usize) < n) {
            let mut seen = vec![false; n];
            let mut stack = vec![entry];
            seen[entry as usize] = true;
            reachable = 1;
            while let Some(node) = stack.pop() {
                for &nb in &self.neighbors[node as usize][0] {
                    let ni = nb as usize;
                    if ni < n && !seen[ni] {
                        seen[ni] = true;
                        reachable += 1;
                        stack.push(nb);
                    }
                }
            }
        }
        (n, no_incoming, reachable, self_links, duplicates, max_len)
    }

    /// Stable fingerprint of the graph's neighbour lists: nodes in id order,
    /// layers in ascending order, ids in list order, with a terminator between
    /// layers.  Iteration order only, so it never depends on hashing.  Used to
    /// (a) prove a build is deterministic for a given seed independently of
    /// timing and (b) compare the two engines' graphs directly.
    fn fingerprint(&self) -> u64 {
        fn mix(h: &mut u64, x: u64) {
            for b in x.to_le_bytes() {
                *h ^= b as u64;
                *h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
        mix(&mut h, self.len() as u64);
        for node in 0..self.len() {
            mix(&mut h, self.levels[node] as u64);
            for layer in 0..(self.levels[node] as usize + 1) {
                for &id in &self.neighbors[node][layer] {
                    mix(&mut h, id as u64);
                }
                mix(&mut h, u32::MAX as u64); // layer terminator
            }
        }
        mix(&mut h, self.entry.unwrap_or(u32::MAX) as u64);
        h
    }

    /// Store a computed neighbor list (ids + their distances + heuristic mask).
    fn set_list(&mut self, id: u32, layer: usize, ids: Vec<u32>, dists: Vec<f32>, mask: [u64; 2]) {
        let n = id as usize;
        self.neighbors[n][layer] = ids;
        self.list_dists[n][layer] = dists;
        self.list_masks[n][layer] = mask;
    }
}

impl GraphAccess for MemGraph {
    type Id = u32;

    fn visit(&self, id: u32, layer: usize) -> Option<VisitData<u32>> {
        let i = id as usize;
        Some(VisitData {
            level: *self.levels.get(i)?,
            deleted: false,
            encoded: self.vectors.get(i)?.clone(),
            neighbors: self
                .neighbors
                .get(i)
                .and_then(|n| n.get(layer))
                .cloned()
                .unwrap_or_default(),
            heap_tid: *self.tids.get(i)?,
            clamped: *self.clamped.get(i)?,
        })
    }

    fn vector(&self, id: u32) -> Option<Vec<u8>> {
        self.vectors.get(id as usize).cloned()
    }

    /// Borrowed probe: no copy, no allocation (the graph lives in RAM).
    fn probe(
        &self,
        id: u32,
        query: &[f32],
        codec: &Codec,
        dist_type: DistanceType,
    ) -> Option<ProbeResult> {
        let i = id as usize;
        Some(ProbeResult {
            dist: distance_encoded(codec, dist_type, query, self.vectors.get(i)?),
            deleted: false,
            heap_tid: *self.tids.get(i)?,
            clamped: *self.clamped.get(i)?,
        })
    }

    /// Borrowed expansion into the caller's buffer.
    fn expand(&self, id: u32, layer: usize, out: &mut Vec<u32>) -> Option<ExpandResult> {
        let i = id as usize;
        out.clear();
        if let Some(list) = self.neighbors.get(i).and_then(|n| n.get(layer)) {
            out.extend_from_slice(list);
        }
        Some(ExpandResult {
            level: *self.levels.get(i)?,
            deleted: false,
        })
    }
}

/// Everything the flat engine needs that is not the codec/parameters: the graph,
/// its beam-search scratch, and the decode-once pair buffer.
struct FlatEngineState {
    graph: FlatGraph,
    scratch: SearchScratch,
    buf: FlatPairBuf,
}

impl FlatEngineState {
    /// `budget_bytes` sizes the graph up front (`maintenance_work_mem`): the arena
    /// this is a prototype for cannot grow, so the capacity is planned before the
    /// first row and exhaustion is answered by spilling.  A budget that affords no
    /// node at all falls back to grow mode, which the driver's own per-row budget
    /// check still spills from.
    fn new(stride: usize, cap: usize, dim: usize, budget_bytes: u64) -> Self {
        // ≈1.07 layers per node for m = 16 at scale (layer 0 plus the 1/16 of nodes
        // that also get layer 1); 0.7 margin leaves headroom for the writeout's
        // transient copy.
        let sizing = crate::access_method::hnswsq::flat_graph::plan_capacity(
            stride, cap, budget_bytes, 1.07, 0.7,
        );
        let graph = if sizing.nodes > 0 {
            FlatGraph::with_limits(stride, cap, sizing.nodes, sizing.slabs)
        } else {
            FlatGraph::new(stride, cap)
        };
        Self {
            graph,
            scratch: SearchScratch::new(),
            buf: FlatPairBuf::new(dim),
        }
    }
}

/// One row through the flat engine: encode, register the node, plan (beam search
/// + selection per layer), then apply (publish the node's lists, backlink, entry
/// promotion).  Mirrors `mem_insert`'s structure so the engines are comparable
/// step for step.
/// Returns `false` when the flat graph refused the node because its fixed
/// capacity is exhausted — the driver then writes out what exists and continues
/// on the disk path, exactly like the `maintenance_work_mem` transition.
fn flat_insert(state: &mut BuildState, heap_tid: ItemPointer, vector: &[f32]) -> bool {
    let level = random_level(state.ml, state.max_level, &mut state.rng);
    flat_insert_at_level(state, heap_tid, vector, level)
}

/// The same insert with the level already decided.
///
/// A parallel worker cannot draw its level from an RNG stream -- which worker draws next
/// depends on scheduling, so the level stream, the graph and its fingerprint would all
/// change run to run.  It derives the level from the row instead
/// (`levels::level_for_tid`) and comes in through here, which keeps *one* insert body: the
/// two paths differ only in where the level came from.
fn flat_insert_at_level(
    state: &mut BuildState,
    heap_tid: ItemPointer,
    vector: &[f32],
    level: u8,
) -> bool {
    if state.stats.enabled {
        state.stats.nodes += 1;
    }
    let mut encoded = Vec::with_capacity(state.codec.vector_bytes());
    let clamped = state.codec.encode_into(vector, &mut encoded);
    let subject = state.codec.decode(&encoded);

    let (m, m0, efc, dist_type, dist_fn) = (
        state.m,
        state.m0,
        state.ef_construction,
        state.distance_type,
        state.dist_fn,
    );
    let codec = &state.codec;
    // Policy read once per insert, here rather than inside the engine: a GUC read
    // needs a backend, and the engine is also exercised by unit tests.
    let backfill =
        unsafe { crate::access_method::hnswsq::options::HNSWSQ_BUILD_BACKFILL.get() } != 0;
    let FlatEngineState {
        graph,
        scratch,
        buf,
    } = state.flat.as_mut().expect("flat engine state");
    let id = graph.len() as u32;

    if !graph.try_push_node(level, heap_tid, clamped, &encoded) {
        return false;
    }
    // Coarse attribution, matching the legacy engine's two halves: the planning
    // half is beam search + neighbour selection, the apply half is publishing the
    // node's own lists, the backlinks and entry promotion -- i.e. where the
    // append/shrink policy's cost lands.
    let t_plan = state.stats.enabled.then(std::time::Instant::now);
    let plan = plan_flat(
        codec, dist_type, dist_fn, graph, scratch, buf, id, level, &subject, m, m0, efc, backfill,
    );
    if let Some(t) = t_plan {
        state.stats.search_ns += t.elapsed().as_nanos() as u64;
    }
    let t_apply = state.stats.enabled.then(std::time::Instant::now);
    // Single-builder build: this thread is the graph's only writer, so the engine
    // takes no node locks.  A parallel driver passes `Locking::Locks(arena.locks())`
    // here instead, and the heuristic runs identically.
    apply_flat(
        codec,
        dist_fn,
        graph,
        &Locking::SoleWriter,
        buf,
        id,
        level,
        plan,
        m,
        m0,
    );
    if let Some(t) = t_apply {
        state.stats.backlink_select_ns += t.elapsed().as_nanos() as u64;
    }
    true
}

/// Materialise the flat graph as a legacy `MemGraph` so the existing writeout and
/// spill paths run unchanged.
///
/// TEMPORARY BRIDGE (measurement aid for M2): the writeout only reads ids, tids,
/// levels, vectors, clamps and the entry point, all of which exist in both
/// layouts, but `flush_mem_graph` is written against `MemGraph`.  The arena step
/// brings a writeout that reads the flat/arena layout directly; until then this
/// costs one transient copy at writeout time (never in a hot loop) and lets the
/// two engines be A/B'd end to end.  `list_dists`/`list_masks` get dummies:
/// nothing reads them after the build's own lists are written, and the flat engine
/// never produced them.
fn flat_to_mem(flat: &FlatGraph) -> MemGraph {
    let mut g = MemGraph::new();
    for id in 0..flat.len() as u32 {
        g.push_node(
            flat.level(id),
            flat.tid(id),
            flat.clamped(id),
            flat.vector(id).to_vec(),
        );
    }
    for id in 0..flat.len() as u32 {
        for layer in 0..=(flat.level(id) as usize) {
            let ids: Vec<u32> = flat.neighbors(id, layer).to_vec();
            let dists = vec![0.0f32; ids.len()];
            let mut mask = [0u64; LIST_MASK_WORDS];
            for i in 0..ids.len().min(LIST_MASK_WORDS * 64) {
                mask[i / 64] |= 1u64 << (i % 64);
            }
            g.set_list(id, layer, ids, dists, mask);
        }
    }
    if let Some(entry) = flat.entry() {
        g.entry = Some(entry);
        g.entry_level = flat.entry_level();
    }
    g
}

/// Make `state.graph` hold the build that has to be written out, converting from
/// the flat engine when that is the active one.
fn writeout_graph(state: &mut BuildState) {
    if let Some(flat) = state.flat.take() {
        if state.stats.enabled {
            // Structural gate on the graph the flat engine just produced, before it
            // is converted for the writeout: `workers > 0` will be judged on exactly
            // these numbers (zero nodes without an incoming edge, everything
            // reachable from the entry) rather than on recall alone.
            let checks =
                crate::access_method::hnswsq::flat_graph::check_lists(&flat.graph, flat.graph.cap());
            state.stats.check_published = checks.published;
            state.stats.check_no_incoming = checks.nodes_without_incoming;
            state.stats.check_reachable = checks.reachable_from_entry;
            state.stats.check_self_links = checks.self_links;
            state.stats.check_duplicates = checks.duplicate_links;
            state.stats.check_max_len = checks.max_list_len;
        }
        state.graph = flat_to_mem(&flat.graph);
        return;
    }
    if state.stats.enabled {
        // Legacy engine: the same structural gate, as the control for the flat
        // engine's connectivity numbers.
        let (published, no_incoming, reachable, self_links, duplicates, max_len) =
            state.graph.list_checks(state.m0);
        state.stats.check_published = published;
        state.stats.check_no_incoming = no_incoming;
        state.stats.check_reachable = reachable;
        state.stats.check_self_links = self_links;
        state.stats.check_duplicates = duplicates;
        state.stats.check_max_len = max_len;
    }
}

/// Pass-2 build state.
struct BuildState {
    codec: Codec,
    dist_fn: DistanceFn,
    distance_type: DistanceType,
    m: usize,
    m0: usize,
    ef_construction: usize,
    ml: f32,
    max_level: u8,
    /// maintenance_work_mem budget in bytes for the in-memory graph
    /// (0 = build directly on disk).
    budget_bytes: u64,
    mem_used: u64,
    graph: MemGraph,
    /// Reusable decode-once distance buffer for the pair-heavy build loops
    /// (own-list selection and backlink admission) — see [`DistBuf`].
    pair_buf: DistBuf,
    /// Per-phase timing counters (`hnswsq.build_stats`).
    stats: BuildStats,
    /// Reusable search state (epoch marks + heaps) for the memory beam search.
    search_scratch: SearchScratch,
    /// New flat engine state (`hnswsq.build_engine = 1`), `None` for the legacy
    /// engine.  Owns its own scratch and pair buffer so the two engines never
    /// share mutable state, and is taken (converted into the legacy graph) when
    /// the build has to be written out — see `writeout_graph`.
    flat: Option<FlatEngineState>,
    /// Test-only reference path: re-run the *full* neighbor-selection
    /// heuristic for every backlink instead of the incremental
    /// [`backlink_prune_mem`].  Used by
    /// `test_backlink_prune_matches_full_heuristic` to prove the incremental
    /// result is identical; always false in production builds.
    reference_backlinks: bool,
    /// Backlink admission policy (`hnswsq.build_backlink_mode`).
    backlink_mode: BacklinkMode,
    /// True after the memory graph was flushed (or for concurrent builds);
    /// remaining rows go through the disk insert path.
    disk_mode: bool,
    nrows: u64,
    rng: SmallRng,
    /// The seed levels are derived from when a level is keyed on the row rather than on
    /// draw order (`levels::level_for_tid`) -- which is what a parallel build needs, since
    /// which worker draws next is not deterministic.  The single-builder path still draws
    /// from `rng`, so this value does not affect it.
    level_seed: u64,
}

/// Per-phase build timing counters, enabled by `hnswsq.build_stats`.
/// Disabled counters cost nothing (the timers are only taken when enabled).
#[derive(Clone, Copy, Default, Debug)]
pub struct BuildStats {
    pub enabled: bool,
    pub nodes: u64,
    pub search_ns: u64,
    pub select_ns: u64,
    pub backlink_pairs_ns: u64,
    pub backlink_select_ns: u64,
    pub backlink_lists: u64,
    /// Pair distances evaluated (SIMD, from the decode-once buffer) and how
    /// many vector decodes were needed to feed them.
    pub pair_dists: u64,
    pub vector_decodes: u64,
    pub flush_ns: u64,
    /// Backlink entries resolved by the O(1) "only the new node can occlude"
    /// path vs entries that needed a full re-check (previously backfilled or
    /// never occlusion-evaluated), plus how many extras were tracked.
    pub fast_entries: u64,
    pub full_entries: u64,
    pub extras_seen: u64,
    /// Pair-distance traffic split by caller, so the cost can be attributed
    /// (own-list selection vs backlink admission vs extras).
    pub pair_dists_select: u64,
    pub pair_dists_backlink: u64,
    pub pair_dists_extras: u64,
    /// Beam-search calls and the neighbour hits they returned (proxy for the
    /// work the search path does per insert).
    pub search_calls: u64,
    pub search_hits: u64,
    /// Flat-engine structural gate, taken at writeout (`check_lists`): how many
    /// nodes were published, how many of them had no incoming edge, how many were
    /// reachable from the entry over layer-0 lists, self-links, duplicate links and
    /// the longest list.  Zero everywhere means the graph is well formed and fully
    /// connected -- the property a concurrent build is most likely to break, and
    /// the one a recall number cannot show.
    pub check_published: usize,
    pub check_no_incoming: usize,
    pub check_reachable: usize,
    pub check_self_links: usize,
    pub check_duplicates: usize,
    pub check_max_len: usize,
    /// Stable fingerprint of the graph that was written out (see
    /// `MemGraph::fingerprint`).  Two builds with the same seed and engine must
    /// produce the same value; the two engines are expected to differ, because
    /// their backlink policies differ.
    pub graph_fingerprint: u64,
    /// Backlink admission outcome in ranked mode: edges skipped by the cutoff
    /// test, edges admitted, and how many of those needed an overflow prune.
    pub cutoff_skips: u64,
    pub ranked_admits: u64,
    pub ranked_prunes: u64,
}

/// Backlink admission policy for the in-memory build.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BacklinkMode {
    /// Lance-style ranked list: append the edge if it beats the target's current
    /// worst neighbour (`cutoff`), prune only when the list overflows.
    Ranked,
    /// Exact incremental re-prune (bit-identical to a full heuristic re-run).
    Exact,
}


impl BuildStats {
    fn new() -> Self {
        Self {
            enabled: unsafe { crate::access_method::hnswsq::options::HNSWSQ_BUILD_STATS.get() },
            ..Default::default()
        }
    }

    /// Human-readable one-line summary (benchmark/diagnostic output).
    pub fn summary(&self) -> String {
        let ms = |ns: u64| ns as f64 / 1e6;
        let total = ms(self.search_ns + self.select_ns + self.backlink_pairs_ns + self.backlink_select_ns + self.flush_ns);
        format!(
            "hnswsq build stats: nodes={} backlink_lists={} pair_dists={} vector_decodes={} \
             search={:.1}ms select={:.1}ms backlink_pairs={:.1}ms backlink_select={:.1}ms \
             flush={:.1}ms accounted_total={:.1}ms fast_entries={} full_entries={} extras_seen={} \
             pair(select)={} pair(backlink)={} pair(extras)={} search_calls={} search_hits={} \
             cutoff_skips={} ranked_admits={} ranked_prunes={} fingerprint={:016x} \
             checks(published={} no_incoming={} reachable={} self_links={} duplicates={} max_len={})",
            self.nodes,
            self.backlink_lists,
            self.pair_dists,
            self.vector_decodes,
            ms(self.search_ns),
            ms(self.select_ns),
            ms(self.backlink_pairs_ns),
            ms(self.backlink_select_ns),
            ms(self.flush_ns),
            total,
            self.fast_entries,
            self.full_entries,
            self.extras_seen,
            self.pair_dists_select,
            self.pair_dists_backlink,
            self.pair_dists_extras,
            self.search_calls,
            self.search_hits,
            self.cutoff_skips,
            self.ranked_admits,
            self.ranked_prunes,
            self.graph_fingerprint,
            self.check_published,
            self.check_no_incoming,
            self.check_reachable,
            self.check_self_links,
            self.check_duplicates,
            self.check_max_len,
        )
    }
}

/// Which hot loop is asking for a pair distance (statistics attribution).
#[derive(Clone, Copy, PartialEq, Eq)]
enum PairCaller {
    /// Neighbour selection for the inserted node's own lists.
    Select,
    /// Backlink admission/re-prune distances.
    Backlink,
    /// Checks against entries newly accepted in this pass ("extras").
    Extras,
}

/// Decode-once distance buffer for the build's pair-heavy loops.
///
/// Both hot loops — neighbour selection for the inserted node's own lists, and
/// backlink admission/pruning — work on a *small* set of ids per call: the
/// candidates of one selection, or one neighbour list plus the incoming node.
/// Pushing each of those vectors into a flat `f32` buffer exactly once and
/// evaluating pairs straight through the SIMD `dist_fn` replaces the
/// build-scoped `HashMap<(u32, u32), f32>` oracle this used to be (and returns
/// bit-identical distances: the map stored `dist_fn` over the same two decoded
/// vectors).
///
/// Why the map had to go (100k-node build, dim 16, m=16, efc=64): its probes
/// are random accesses into a table that grows to ~100MB, so a single lookup
/// measured ~2.5us — more than decoding and comparing both vectors costs — and
/// 99.4% of the backlink pairs were cold anyway.  The buffer version keeps a
/// whole selection's vectors in L1/L2 and does no hashing at all.
struct DistBuf {
    dim: usize,
    /// Ids of the decoded vectors, in slot order (slot `i` is `ids[i]`).
    ids: Vec<u32>,
    /// `ids.len() * dim` decoded values, one slot after another.
    data: Vec<f32>,
}

impl DistBuf {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            ids: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Drop all slots, keeping the allocation (one selection's worth of f32).
    fn clear(&mut self) {
        self.ids.clear();
        self.data.clear();
    }

    /// Decode `id`'s stored vector into the next slot; returns the slot index.
    fn push(&mut self, codec: &Codec, g: &MemGraph, id: u32, stats: &mut BuildStats) -> usize {
        if stats.enabled {
            stats.vector_decodes += 1;
        }
        let start = self.data.len();
        self.data.resize(start + self.dim, 0.0);
        codec.decode_into(&g.vectors[id as usize], &mut self.data[start..]);
        self.ids.push(id);
        self.ids.len() - 1
    }

    #[inline]
    fn slice(&self, slot: usize) -> &[f32] {
        &self.data[slot * self.dim..(slot + 1) * self.dim]
    }

    /// Distance between two slots.
    #[inline]
    fn dist(
        &self,
        dist_fn: DistanceFn,
        stats: &mut BuildStats,
        caller: PairCaller,
        a: usize,
        b: usize,
    ) -> f32 {
        debug_assert!(a != b, "pair distance between a slot and itself");
        note_pair(stats, caller);
        dist_fn(self.slice(a), self.slice(b))
    }

    /// Distance from an already-decoded query vector to `id`'s stored vector.
    /// One decode of `id`, no cache: `g` is not consulted for the query side
    /// because the caller (an insert) holds it decoded already.
    fn dist_to_query(
        &mut self,
        codec: &Codec,
        dist_fn: DistanceFn,
        stats: &mut BuildStats,
        caller: PairCaller,
        query: &[f32],
        g: &MemGraph,
        id: u32,
    ) -> f32 {
        self.clear();
        self.push(codec, g, id, stats);
        note_pair(stats, caller);
        dist_fn(query, self.slice(0))
    }
}

/// Count one evaluated pair distance, attributed to its caller.
#[inline]
fn note_pair(stats: &mut BuildStats, caller: PairCaller) {
    if stats.enabled {
        stats.pair_dists += 1;
        match caller {
            PairCaller::Select => stats.pair_dists_select += 1,
            PairCaller::Backlink => stats.pair_dists_backlink += 1,
            PairCaller::Extras => stats.pair_dists_extras += 1,
        }
    }
}

/// Memory-graph SELECT-NEIGHBORS-HEURISTIC.
///
/// Same semantics as the disk-graph [`select_neighbors_heuristic`]; the
/// pairwise distances come from a decode-once [`DistBuf`] over the candidate
/// set, where slot `i` is the candidate at sorted position `i`, so every
/// occlusion check is one SIMD kernel call on two slices of a small hot buffer.
pub(crate) fn select_neighbors_heuristic_mem(
    codec: &Codec,
    dist_fn: DistanceFn,
    g: &MemGraph,
    buf: &mut DistBuf,
    stats: &mut BuildStats,
    candidates: Vec<(f32, u32)>,
    cap: usize,
) -> (Vec<u32>, Vec<f32>, [u64; LIST_MASK_WORDS]) {
    let mut sorted = candidates;
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    buf.clear();
    for &(_, id) in sorted.iter() {
        buf.push(codec, g, id, stats);
    }
    if sorted.len() <= cap {
        // Room for everyone: the returned list is every candidate in ascending
        // order.  The mask still records, for each entry, whether the occlusion
        // rule would have accepted it — the incremental backlink re-prune
        // relies on that bit, and it must not claim "accepted" for an entry
        // that was never evaluated.
        let mut accepted: Vec<usize> = Vec::with_capacity(sorted.len());
        let mut mask = [0u64; LIST_MASK_WORDS];
        for (i, &(d, _)) in sorted.iter().enumerate() {
            let mut keep = true;
            for &sel in &accepted {
                if buf.dist(dist_fn, stats, PairCaller::Select, i, sel) < d {
                    keep = false;
                    break;
                }
            }
            if keep {
                accepted.push(i);
                if i < LIST_MASK_WORDS * 64 {
                    mask[i / 64] |= 1u64 << (i % 64);
                }
            }
        }
        let ids: Vec<u32> = sorted.iter().map(|c| c.1).collect();
        let dists: Vec<f32> = sorted.iter().map(|c| c.0).collect();
        return (ids, dists, mask);
    }
    // Selected entries carry their buffer slot so the occlusion checks index
    // the decoded buffer directly; only the backfill below drops it (it is
    // appended after every check is done).
    let mut selected: Vec<(f32, u32, usize)> = Vec::with_capacity(cap);
    let mut pruned: Vec<(f32, u32)> = Vec::new();
    for (i, cand) in sorted.iter().enumerate() {
        if selected.len() >= cap {
            pruned.push(*cand);
            continue;
        }
        let mut keep = true;
        for sel in &selected {
            if buf.dist(dist_fn, stats, PairCaller::Select, i, sel.2) < cand.0 {
                keep = false;
                break;
            }
        }
        if keep {
            selected.push((cand.0, cand.1, i));
        } else {
            pruned.push(*cand);
        }
    }
    // Backfill with the closest pruned candidates (ascending order).
    let mut entries: Vec<(f32, u32)> = selected.iter().map(|&(d, id, _)| (d, id)).collect();
    let main_loop_len = entries.len();
    for p in pruned {
        if entries.len() >= cap {
            break;
        }
        entries.push(p);
    }
    let mut mask = [0u64; LIST_MASK_WORDS];
    let mut ids = Vec::with_capacity(entries.len());
    let mut dists = Vec::with_capacity(entries.len());
    for (i, (d, id)) in entries.into_iter().enumerate() {
        if i < main_loop_len && i < LIST_MASK_WORDS * 64 {
            mask[i / 64] |= 1u64 << (i % 64);
        }
        ids.push(id);
        dists.push(d);
    }
    (ids, dists, mask)
}

/// Sort a ranked neighbour list by `(distance, id)` — the invariant `cutoff`
/// and the ranked insertion position both depend on — carrying each entry's
/// mask bit along with it.
fn sort_ranked_list(ids: &mut Vec<u32>, dists: &mut Vec<f32>, mask: &mut [u64; LIST_MASK_WORDS]) {
    let mut entries: Vec<(f32, u32, bool)> = ids
        .iter()
        .copied()
        .zip(dists.iter().copied())
        .enumerate()
        .map(|(i, (id, d))| (d, id, (i < LIST_MASK_WORDS * 64) && (mask[i / 64] >> (i % 64)) & 1 == 1))
        .collect();
    entries.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut new_mask = [0u64; LIST_MASK_WORDS];
    for (i, (d, id, was_selected)) in entries.into_iter().enumerate() {
        ids[i] = id;
        dists[i] = d;
        if was_selected && i < LIST_MASK_WORDS * 64 {
            new_mask[i / 64] |= 1u64 << (i % 64);
        }
    }
    *mask = new_mask;
}

/// Reusable per-search state for the memory-graph beam search.
///
/// Shared with the flat engine (`flat_engine.rs`), which runs the same beam
/// search over its slabs: the epoch-mark discipline (no hashing, O(visited)
/// reset) and the reused heaps are worth keeping in exactly one place.
///
/// Lance's builder passes a `VisitedGenerator` (bitmap + recently-visited list)
/// into every insert; the equivalent here is an epoch-stamped mark array plus
/// reusable heaps, so a search allocates nothing per visited node and does no
/// hashing at all.
pub(crate) struct SearchScratch {
    /// Per node id: the epoch in which it was visited.
    pub(crate) visited_epoch: Vec<u32>,
    /// Per node id: the epoch in which its neighbour list was expanded.
    pub(crate) expanded_epoch: Vec<u32>,
    pub(crate) epoch: u32,
    pub(crate) candidates: std::collections::BinaryHeap<std::cmp::Reverse<HeapItem<u32>>>,
    pub(crate) results: std::collections::BinaryHeap<HeapItem<u32>>,
}

impl SearchScratch {
    pub(crate) fn new() -> Self {
        Self {
            visited_epoch: Vec::new(),
            expanded_epoch: Vec::new(),
            epoch: 0,
            candidates: std::collections::BinaryHeap::new(),
            results: std::collections::BinaryHeap::new(),
        }
    }

    /// Start a new search epoch, growing the mark arrays as the graph grows.
    pub(crate) fn begin(&mut self, nodes: usize) {
        if self.visited_epoch.len() < nodes {
            self.visited_epoch.resize(nodes, 0);
            self.expanded_epoch.resize(nodes, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            // Wrapped: clear both arrays so epoch 0 stays "never visited".
            self.visited_epoch.iter_mut().for_each(|e| *e = 0);
            self.expanded_epoch.iter_mut().for_each(|e| *e = 0);
            self.epoch = 1;
        }
        self.candidates.clear();
        self.results.clear();
    }
}

/// Memory-graph beam search: same semantics as the generic
/// [`crate::access_method::hnswsq::graph::search_layer`], but with borrowed
/// neighbour slices, epoch-stamped visited marks and reused heaps instead of
/// per-visit clones plus `HashSet`/`HashMap` bookkeeping.  The memory graph has
/// no tombstones and no vanished ids, which is what makes the simplification
/// safe.  `test_search_layer_mem_matches_generic` asserts the two agree.
fn search_layer_mem(
    codec: &Codec,
    dist_type: DistanceType,
    query: &[f32],
    g: &MemGraph,
    entries: &[(f32, u32)],
    ef: usize,
    layer: usize,
    scratch: &mut SearchScratch,
) -> Vec<SearchHit<u32>> {
    let ef = ef.max(1);
    scratch.begin(g.len());
    let SearchScratch {
        visited_epoch,
        expanded_epoch,
        epoch,
        candidates,
        results,
    } = scratch;
    let epoch = *epoch;

    for &(d, id) in entries {
        let i = id as usize;
        if i >= visited_epoch.len() || visited_epoch[i] == epoch {
            continue;
        }
        visited_epoch[i] = epoch;
        // The build graph never emits heap TIDs (nothing to scan there), so the
        // emission metadata is a placeholder: keeping it out of this loop's
        // memory traffic matters more than filling it in.
        candidates.push(std::cmp::Reverse(HeapItem {
            dist: d,
            id,
            deleted: false,
            heap_tid: ItemPointer::new_invalid(),
            clamped: false,
        }));
        results.push(HeapItem {
            dist: d,
            id,
            deleted: false,
            heap_tid: ItemPointer::new_invalid(),
            clamped: false,
        });
    }

    while let Some(std::cmp::Reverse(cur)) = candidates.pop() {
        // Same termination rule as the generic version: stop once the closest
        // unexpanded candidate is worse than the worst kept result.
        if results.len() >= ef {
            if let Some(worst) = results.peek() {
                if cur.dist > worst.dist {
                    break;
                }
            }
        }

        let ci = cur.id as usize;
        if ci >= expanded_epoch.len() || expanded_epoch[ci] == epoch {
            continue;
        }
        expanded_epoch[ci] = epoch;
        let Some(level_neighbors) = g.neighbors.get(ci).and_then(|l| l.get(layer)) else {
            continue;
        };
        for &nb in level_neighbors {
            let ni = nb as usize;
            if ni >= visited_epoch.len() || visited_epoch[ni] == epoch {
                continue;
            }
            visited_epoch[ni] = epoch;
            let Some(enc) = g.vectors.get(ni) else {
                continue;
            };
            let d = distance_encoded(codec, dist_type, query, enc);
            let item = HeapItem {
                dist: d,
                id: nb,
                deleted: false,
                heap_tid: ItemPointer::new_invalid(),
                clamped: false,
            };
            candidates.push(std::cmp::Reverse(item.clone()));
            if results.len() < ef {
                results.push(item);
            } else if let Some(mut worst) = results.peek_mut() {
                if item.dist < worst.dist {
                    *worst = item;
                }
            }
        }
    }

    results
        .clone()
        .into_sorted_vec()
        .into_iter()
        .map(|i| SearchHit {
            dist: i.dist,
            id: i.id,
            deleted: false,
            heap_tid: i.heap_tid,
            clamped: i.clamped,
        })
        .collect()
}

/// Lance-style backlink admission: append the edge to the target's ranked
/// neighbour list and prune only when the list overflows.
///
/// The cheap part is the *cutoff* test — the edge is dropped outright when the
/// new node is not closer to the target than the target's current worst
/// neighbour and the list is already full.  In a saturated graph that skips the
/// great majority of backlink work (no list copy, no sort, no heuristic), which
/// is precisely what `lance-index` does in
/// `GraphBuilderNode::cutoff` + `Builder::insert`.
///
/// Returns `None` when the edge is skipped (the target's list is unchanged).
/// The mask is not maintained in this mode: only [`BacklinkMode::Exact`]
/// consumes [`MemGraph::list_masks`].
#[allow(clippy::too_many_arguments)]
fn backlink_add_ranked_mem(
    codec: &Codec,
    dist_fn: DistanceFn,
    g: &MemGraph,
    buf: &mut DistBuf,
    stats: &mut BuildStats,
    existing_ids: &[u32],
    existing_dists: &[f32],
    d_new: f32,
    new_id: u32,
    cap: usize,
    always_admit: bool,
) -> Option<(Vec<u32>, Vec<f32>, [u64; LIST_MASK_WORDS])> {
    debug_assert_eq!(existing_ids.len(), existing_dists.len());

    // 1. cutoff admission (O(1)): only a full list can reject an edge, and then
    //    only when the new node is no better than the current worst neighbour.
    //    The ranked list is kept ascending, so the worst is the last entry.
    //
    //    `always_admit` bypasses the test for the new node's own nearest
    //    neighbour: without it, a node inserted into an already-saturated dense
    //    cluster can end up with no incoming edge at all and become unreachable
    //    from the entry point (measured: recall 0.63-0.76 vs 1.00 exact).
    if !always_admit && existing_ids.len() >= cap {
        if let Some(&worst) = existing_dists.last() {
            if d_new >= worst {
                if stats.enabled {
                    stats.cutoff_skips += 1;
                }
                return None;
            }
        }
    }
    if stats.enabled {
        stats.ranked_admits += 1;
    }

    // 2. insert into the ranked list (ascending, ties broken by id — the same
    //    order the neighbor-selection heuristic would produce).
    let pos = existing_ids
        .iter()
        .zip(existing_dists.iter())
        .position(|(&id, &d)| d > d_new || (d == d_new && id > new_id))
        .unwrap_or(existing_ids.len());
    let mut ids = existing_ids.to_vec();
    let mut dists = existing_dists.to_vec();
    ids.insert(pos, new_id);
    dists.insert(pos, d_new);

    let mut mask = [0u64; LIST_MASK_WORDS];
    for i in 0..ids.len().min(LIST_MASK_WORDS * 64) {
        mask[i / 64] |= 1u64 << (i % 64);
    }

    // 3. prune only on overflow, with the same heuristic as everywhere else.
    if ids.len() <= cap {
        return Some((ids, dists, mask));
    }
    if stats.enabled {
        stats.ranked_prunes += 1;
    }
    let cands: Vec<(f32, u32)> = dists.iter().copied().zip(ids.iter().copied()).collect();
    let (mut ids, mut dists, mut mask) =
        select_neighbors_heuristic_mem(codec, dist_fn, g, buf, stats, cands, cap);
    // The heuristic emits accepted entries followed by backfilled ones, which
    // is not globally ascending; the ranked-list invariant (`cutoff`, insertion
    // position) needs a true order, so re-sort.  The SET is unchanged, so
    // recall is unaffected.
    sort_ranked_list(&mut ids, &mut dists, &mut mask);
    Some((ids, dists, mask))
}

/// Exact incremental backlink re-prune.
///
/// The backlink step re-runs the neighbor-selection heuristic over
/// `existing ∪ {new}`.  The existing list is itself the output of that same
/// heuristic, and the new node only ever *adds* occlusion relations — it can
/// never remove one that existed among the old members (those checks are
/// unchanged, and every old member's position in distance order is fixed).
/// Two consequences make the re-run O(len) instead of O(len · cap):
///
/// * old members **before** the new node keep their previous status verbatim
///   (the `list_masks` bit records whether the heuristic accepted them or they
///   were re-added by the closest-pruned backfill);
/// * old members **after** the new node need exactly one new check — against
///   the new node, and only if it was accepted.
///
/// The result (ids, distances, mask) is bit-identical to running
/// [`select_neighbors_heuristic_mem`] over the merged candidate set; the
/// `test_backlink_prune_matches_full_heuristic` test asserts that equality on
/// randomized graphs.
///
/// Distances come from a decode-once [`DistBuf`] holding the existing list plus
/// the incoming node (slots `0..len` in list order, then the new node), so the
/// per-entry checks are SIMD calls on one small hot buffer rather than hash-map
/// probes.
#[allow(clippy::too_many_arguments)]
fn backlink_prune_mem(
    codec: &Codec,
    dist_fn: DistanceFn,
    g: &MemGraph,
    buf: &mut DistBuf,
    stats: &mut BuildStats,
    existing_ids: &[u32],
    existing_dists: &[f32],
    existing_mask: [u64; LIST_MASK_WORDS],
    d_new: f32,
    new_id: u32,
    cap: usize,
) -> (Vec<u32>, Vec<f32>, [u64; LIST_MASK_WORDS]) {
    debug_assert_eq!(existing_ids.len(), existing_dists.len());

    buf.clear();
    for &id in existing_ids.iter() {
        buf.push(codec, g, id, stats);
    }
    let new_slot = buf.push(codec, g, new_id, stats);

    // The old list is ascending by distance (heuristic output order); the new
    // node slots in at the first position where (dist, id) wins.
    let mut merged: Vec<(f32, u32, bool, usize)> = Vec::with_capacity(existing_ids.len() + 1);
    for (i, (&id, &d)) in existing_ids.iter().zip(existing_dists.iter()).enumerate() {
        let was_selected = (existing_mask[i / 64] >> (i % 64)) & 1 == 1;
        merged.push((d, id, was_selected, i));
    }
    // Old entries may not be perfectly sorted if a list ever arrived from a
    // different path; sorting keeps this exact rather than assumed.  `(dist, id)`
    // is a total order over distinct ids, so an unstable sort is equivalent.
    merged.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    // Insertion position under the heuristic's (dist, id) ordering.
    let p = merged
        .iter()
        .position(|c| {
            d_new.total_cmp(&c.0).then_with(|| new_id.cmp(&c.1)) == std::cmp::Ordering::Less
        })
        .unwrap_or(merged.len());
    merged.insert(p, (d_new, new_id, false, new_slot));

    // Room for everyone: the list is every candidate in ascending order (the
    // heuristic's fast path).  The mask must still record which entries the
    // occlusion rule would accept — the same simulation the full heuristic's
    // fast path does — otherwise later incremental updates would trust an
    // acceptance that was never evaluated.
    if merged.len() <= cap {
        let mut accepted: Vec<usize> = Vec::with_capacity(merged.len());
        let mut mask = [0u64; LIST_MASK_WORDS];
        for (i, &(d, _, _, slot)) in merged.iter().enumerate() {
            let mut keep = true;
            for &sel in &accepted {
                if buf.dist(dist_fn, stats, PairCaller::Select, slot, sel) < d {
                    keep = false;
                    break;
                }
            }
            if keep {
                accepted.push(slot);
                if i < LIST_MASK_WORDS * 64 {
                    mask[i / 64] |= 1u64 << (i % 64);
                }
            }
        }
        let ids: Vec<u32> = merged.iter().map(|c| c.1).collect();
        let dists: Vec<f32> = merged.iter().map(|c| c.0).collect();
        return (ids, dists, mask);
    }

    // Accepted entries carry their buffer slot (used as the occluder index).
    let mut selected: Vec<(f32, u32, usize)> = Vec::with_capacity(cap);
    let mut pruned: Vec<(f32, u32)> = Vec::new();
    let mut new_selected = false;
    // Entries the occlusion rule had *not* accepted before but accepts now
    // (their previous occluder is gone, or they were never evaluated).  They
    // are the only occluders a previously-accepted entry has not already been
    // checked against, so mask-set entries only need to be tested against
    // these plus the new node — instead of rescanning `selected`.
    let mut extras: Vec<usize> = Vec::new();
    for (i, &(d, id, was_selected, slot)) in merged.iter().enumerate() {
        if selected.len() >= cap {
            pruned.push((d, id));
            continue;
        }
        let keep = if i == p {
            // The new node: full occlusion check against the accepted prefix.
            let mut keep = true;
            for sel in &selected {
                if buf.dist(dist_fn, stats, PairCaller::Backlink, slot, sel.2) < d {
                    keep = false;
                    break;
                }
            }
            new_selected = keep;
            keep
        } else if was_selected {
            if stats.enabled {
                stats.fast_entries += 1;
            }
            // Previously accepted by the occlusion rule: every check it passed
            // then still holds, so only the *new* occluders matter — the new
            // node (if accepted and positioned before this entry) and the
            // `extras` accepted in this run that were not accepted before.
            let mut keep = true;
            if i > p && new_selected {
                if buf.dist(dist_fn, stats, PairCaller::Backlink, slot, new_slot) < d {
                    keep = false;
                }
            }
            if keep {
                for &extra in &extras {
                    if buf.dist(dist_fn, stats, PairCaller::Extras, slot, extra) < d {
                        keep = false;
                        break;
                    }
                }
            }
            keep
        } else {
            if stats.enabled {
                stats.full_entries += 1;
            }
            // Never occlusion-evaluated (fast path) or re-added by the
            // previous backfill: re-check against `selected`, which now also
            // contains the new node when it was accepted.
            let mut keep = true;
            for sel in &selected {
                if buf.dist(dist_fn, stats, PairCaller::Backlink, slot, sel.2) < d {
                    keep = false;
                    break;
                }
            }
            keep
        };
        if keep {
            selected.push((d, id, slot));
            if !was_selected && i != p {
                extras.push(slot);
                if stats.enabled {
                    stats.extras_seen += 1;
                }
            }
        } else {
            pruned.push((d, id));
        }
    }

    // Backfill with the closest pruned candidates (ascending order); those keep
    // a clear mask bit, exactly like the full heuristic.
    let mut entries: Vec<(f32, u32)> = selected.iter().map(|&(d, id, _)| (d, id)).collect();
    let main_loop_len = entries.len();
    for p_item in pruned {
        if entries.len() >= cap {
            break;
        }
        entries.push(p_item);
    }

    let mut mask = [0u64; LIST_MASK_WORDS];
    let mut ids = Vec::with_capacity(entries.len());
    let mut dists = Vec::with_capacity(entries.len());
    for (i, (d, id)) in entries.into_iter().enumerate() {
        if i < main_loop_len && i < LIST_MASK_WORDS * 64 {
            mask[i / 64] |= 1u64 << (i % 64);
        }
        ids.push(id);
        dists.push(d);
    }
    (ids, dists, mask)
}

/// Resolve the distance type from the operator class support function
/// (amsupport proc 1), like the diskann build.
unsafe fn resolve_distance_type(index: pg_sys::Relation) -> DistanceType {
    unsafe {
        let fmgr_info = pg_sys::index_getprocinfo(index, 1, HNSWSQ_DISTANCE_TYPE_PROC);
        if fmgr_info.is_null() {
            error!("hnswsq: no distance type function found for index");
        }
        let result = pg_sys::FunctionCall0Coll(fmgr_info, pg_sys::InvalidOid).value() as u16;
        DistanceType::from_u16(result)
    }
}

/// Build a new hnswsq index.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambuild(
    heap: pg_sys::Relation,
    index: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    let heap_rel = unsafe { PgRelation::from_pg(heap) };
    let index_rel = unsafe { PgRelation::from_pg(index) };

    let options = TSVHnswOptions::from_relation(&index_rel);
    let precision = options.get_precision();
    let m = options.get_m() as usize;
    let m0 = m * 2;
    let ef_construction = options.get_ef_construction() as usize;

    // Dimensions come from the indexed column's typmod (vector(N) → N).
    let atttypmod = index_rel
        .tuple_desc()
        .get(0)
        .map(|attr| attr.atttypmod)
        .unwrap_or(-1);
    if atttypmod < 1 {
        error!(
            "hnswsq: the indexed column must have a fixed dimension (e.g. vector(128)); \
             got atttypmod {}",
            atttypmod
        );
    }
    let num_dimensions = atttypmod as usize;

    let distance_type = unsafe { resolve_distance_type(index) };

    // Level/dimension limits: the largest node (max_level) must fit one page
    // item; a level-0 node that does not fit means the dimension is too large
    // for this precision.
    let max_level = match compute_max_level(num_dimensions, precision.elem_bytes(), m, m0) {
        Some(ml) => ml,
        None => {
            let max_dim = max_dim_for_page(precision.elem_bytes(), m0);
            error!(
                "hnswsq: {} dimensions do not fit a page item with storage_layout={} (max ~{} \
                 for this layout and m={}); use a reduced-precision layout (ieeefp16/ieeefp8/f8) \
                 or a smaller m",
                num_dimensions,
                precision.as_str(),
                max_dim,
                m
            );
        }
    };

    // ---- Pass 1 (SQ8 only): reservoir-sample for calibration. ----
    let sample_size = options.get_sample_size().unwrap_or(DEFAULT_SAMPLE_SIZE);
    let calibration = if precision.needs_calibration() {
        let mut sample_state = SampleState {
            sample: Vec::with_capacity(sample_size.min(DEFAULT_SAMPLE_SIZE)),
            sample_size,
            nrows: 0,
            distance_type,
            rng: crate::access_method::hnswsq::build_rng(),
        };
        unsafe {
            pg_sys::IndexBuildHeapScan(
                heap_rel.as_ptr(),
                index_rel.as_ptr(),
                index_info,
                Some(sample_callback),
                &mut sample_state,
            );
        }
        Some(Sq8Calibration::train(
            &sample_state.sample,
            num_dimensions,
        ))
    } else {
        None
    };

    // ---- Write the meta page (block 0) FIRST, then the calibration chain
    // (the chain writer extends the relation, so it must not run before the
    // meta page claims block 0), then record the pointer in the meta. ----
    let mut meta = unsafe {
        HnswMetaPage::create(
            &index_rel,
            num_dimensions as u32,
            distance_type,
            precision,
            m as u16,
            ef_construction as u32,
            max_level,
            ItemPointer::new_invalid(),
        )
    };
    if let Some(calib) = &calibration {
        let calib_ptr = unsafe { calib.store(&index_rel) };
        unsafe {
            HnswMetaPage::update(&index_rel, |mm| mm.set_calibration_pointer(calib_ptr));
        }
        meta = HnswMetaPage::fetch(&index_rel);
    }
    let codec = codec_for(&index_rel, &meta);
    // Captured before `codec` is moved into the state: the flat engine needs the
    // encoded-vector stride to size its slab arena.
    let flat_stride = codec.vector_bytes();

    // Concurrent builds cannot use the bulk writeout (live inserters would
    // race the page plan); pgvector likewise forces its disk path for CIC.
    let is_concurrent = unsafe { (*index_info).ii_Concurrent };
    let budget_bytes = if is_concurrent {
        0 // force disk mode from the first row
    } else {
        (unsafe { pg_sys::maintenance_work_mem } as u64).saturating_mul(1024)
    };

    let mut state = BuildState {
        codec,
        dist_fn: distance_type.get_distance_function(),
        distance_type,
        m,
        m0,
        ef_construction,
        ml: meta.get_ml(),
        max_level,
        budget_bytes,
        mem_used: 0,
        graph: MemGraph::new(),
        pair_buf: DistBuf::new(num_dimensions),
        stats: BuildStats::new(),
        search_scratch: SearchScratch::new(),
        flat: if unsafe { crate::access_method::hnswsq::options::HNSWSQ_BUILD_ENGINE.get() } == 1 {
            Some(FlatEngineState::new(
                flat_stride,
                m0,
                num_dimensions,
                budget_bytes,
            ))
        } else {
            None
        },
        reference_backlinks: false,
        backlink_mode: if unsafe {
            crate::access_method::hnswsq::options::HNSWSQ_BACKLINK_MODE.get()
        } == 1
        {
            BacklinkMode::Exact
        } else {
            BacklinkMode::Ranked
        },
        disk_mode: is_concurrent || budget_bytes == 0,
        nrows: 0,
        rng: crate::access_method::hnswsq::build_rng(),
        level_seed: crate::access_method::hnswsq::build_seed_value(),
    };

    // ---- Pass 2 (or the only pass): stream rows into the builder. ----
    unsafe {
        pg_sys::IndexBuildHeapScan(
            heap_rel.as_ptr(),
            index_rel.as_ptr(),
            index_info,
            Some(build_callback),
            &mut state,
        );
    }

    // ---- Flush the residual in-memory graph (single sequential writeout). ----
    writeout_graph(&mut state);
    if state.stats.enabled {
        state.stats.graph_fingerprint = state.graph.fingerprint();
    }
    let mem_tuples = state.graph.len() as u64;
    if mem_tuples > 0 {
        let t_flush = state.stats.enabled.then(std::time::Instant::now);
        let (entry_ptr, entry_level, insert_page) = unsafe {
            flush_mem_graph(&index_rel, &state.graph, &state.codec, state.m, state.m0)
        };
        if let Some(t) = t_flush {
            state.stats.flush_ns += t.elapsed().as_nanos() as u64;
        }
        unsafe {
            HnswMetaPage::update(&index_rel, |meta| {
                meta.set_build_result(entry_ptr, entry_level, mem_tuples, insert_page);
            });
        }
    }

    if state.stats.enabled {
        pgrx::warning!(
            "{} rows_total={} disk_mode={}",
            state.stats.summary(),
            state.nrows,
            state.disk_mode
        );
    }

    let mut pg_result = unsafe { PgBox::<pg_sys::IndexBuildResult>::alloc0() };
    pg_result.heap_tuples = state.nrows as f64;
    pg_result.index_tuples = state.nrows as f64;
    pg_result.into_pg()
}

/// Pass-1 callback: reservoir sampling (Algorithm R) with cosine
/// normalization, feeding the SQ8 calibration.
#[pg_guard]
unsafe extern "C-unwind" fn sample_callback(
    _index: pg_sys::Relation,
    _tid: pg_sys::ItemPointer,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut std::os::raw::c_void,
) {
    if *isnull {
        return;
    }
    let sample_state = unsafe { &mut *(state as *mut SampleState) };

    let mut vec = unsafe { extract_vector(*values) };
    if sample_state.distance_type == DistanceType::Cosine {
        preprocess_cosine(&mut vec);
    }

    let i = sample_state.nrows;
    sample_state.nrows += 1;
    if sample_state.sample.len() < sample_state.sample_size {
        sample_state.sample.push(vec);
    } else {
        let j = sample_state.rng.gen_range(0..=i);
        if j < sample_state.sample_size as u64 {
            sample_state.sample[j as usize] = vec;
        }
    }
}

/// Pass-2 callback: quantize + insert into the memory graph (or the disk
/// graph after a spill).
#[pg_guard]
unsafe extern "C-unwind" fn build_callback(
    index: pg_sys::Relation,
    tid: pg_sys::ItemPointer,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut std::os::raw::c_void,
) {
    if *isnull {
        return;
    }
    let state = unsafe { &mut *(state as *mut BuildState) };

    let mut vec = unsafe { extract_vector(*values) };
    if state.distance_type == DistanceType::Cosine {
        preprocess_cosine(&mut vec);
    }
    let heap_tid = ItemPointer::with_item_pointer_data(*tid);
    state.nrows += 1;

    if state.disk_mode {
        let index_rel = unsafe { PgRelation::from_pg(index) };
        let meta = HnswMetaPage::fetch(&index_rel);
        let ctx = InsertCtx::from_meta(&index_rel, &meta, &state.codec);
        unsafe {
            insert_vector(&ctx, heap_tid, &vec, &mut state.rng);
        }
        return;
    }

    // In-memory insert; spill to disk when the budget is exhausted.
    let node_cost = (state.codec.vector_bytes() as u64)
        + ((state.m0 + state.m) * 4) as u64
        + MEM_OVERHEAD_PER_NODE;
    if !state.graph.levels.is_empty() && state.mem_used + node_cost > state.budget_bytes {
        let index_rel = unsafe { PgRelation::from_pg(index) };
        unsafe {
            spill_to_disk(&index_rel, state);
        }
        let meta = HnswMetaPage::fetch(&index_rel);
        let ctx = InsertCtx::from_meta(&index_rel, &meta, &state.codec);
        unsafe {
            insert_vector(&ctx, heap_tid, &vec, &mut state.rng);
        }
        return;
    }

    state.mem_used += node_cost;
    if state.flat.is_some() {
        if !flat_insert(state, heap_tid, &vec) {
            // The flat graph's fixed capacity is exhausted: write out what it
            // holds and put this row (and every later one) through the disk
            // insert path.
            let index_rel = unsafe { PgRelation::from_pg(index) };
            unsafe {
                spill_to_disk(&index_rel, state);
            }
            let meta = HnswMetaPage::fetch(&index_rel);
            let ctx = InsertCtx::from_meta(&index_rel, &meta, &state.codec);
            unsafe {
                insert_vector(&ctx, heap_tid, &vec, &mut state.rng);
            }
        }
    } else {
        mem_insert(state, heap_tid, &vec);
    }
}

/// Detoast + copy one vector datum.
unsafe fn extract_vector(datum: pg_sys::Datum) -> Vec<f32> {
    unsafe {
        let detoasted = pg_sys::pg_detoast_datum_copy(datum.cast_mut_ptr());
        let pg_vec = detoasted.cast::<PgVectorInternal>();
        let vec = (*pg_vec).to_slice().to_vec();
        pg_sys::pfree(detoasted.cast());
        vec
    }
}

/// Insert one row into the in-memory graph (same algorithm as the disk path,
/// with direct RAM mutation instead of two-phase page updates).
fn mem_insert(state: &mut BuildState, heap_tid: ItemPointer, vector: &[f32]) {
    if state.stats.enabled {
        state.stats.nodes += 1;
    }
    let level = random_level(state.ml, state.max_level, &mut state.rng);
    let mut encoded = Vec::with_capacity(state.codec.vector_bytes());
    let clamped = state.codec.encode_into(vector, &mut encoded);

    let g = &mut state.graph;
    let id = g.len() as u32;
    g.push_node(level, heap_tid, clamped, encoded.clone());

    // Empty graph → this node is the entry.
    let Some(ep) = g.entry else {
        g.entry = Some(id);
        g.entry_level = level as usize;
        return;
    };

    let subject = state.codec.decode(&encoded);
    let entry_level = g.entry_level;
    let top = (level as usize).min(entry_level);

    let mut cur = (
        distance_encoded(
            &state.codec,
            state.distance_type,
            &subject,
            &g.vectors[ep as usize],
        ),
        ep,
    );
    if entry_level > top {
        cur = greedy_descent(
            &state.codec,
            state.distance_type,
            &subject,
            &*g,
            cur,
            entry_level,
            top + 1,
        );
    }

    let (m, m0) = (state.m, state.m0);
    let cap_for = |l: usize| if l == 0 { m0 } else { m };
    let timed = state.stats.enabled;
    for l in (0..=top).rev() {
        let t_phase = timed.then(std::time::Instant::now);
        let hits = search_layer_mem(
            &state.codec,
            state.distance_type,
            &subject,
            &*g,
            &[cur],
            state.ef_construction,
            l,
            &mut state.search_scratch,
        );
        if let Some(best) = hits.first() {
            cur = (best.dist, best.id);
        }
        if timed {
            state.stats.search_calls += 1;
            state.stats.search_hits += hits.len() as u64;
        }
        let cands: Vec<(f32, u32)> = hits
            .iter()
            .filter(|h| h.id != id)
            .map(|h| (h.dist, h.id))
            .collect();
        if let Some(t) = t_phase {
            state.stats.search_ns += t.elapsed().as_nanos() as u64;
        }
        let t_phase = timed.then(std::time::Instant::now);
        let (selected, sel_dists, sel_mask) = select_neighbors_heuristic_mem(
            &state.codec,
            state.dist_fn,
            &*g,
            &mut state.pair_buf,
            &mut state.stats,
            cands,
            cap_for(l),
        );
        if let Some(t) = t_phase {
            state.stats.select_ns += t.elapsed().as_nanos() as u64;
        }
        if state.backlink_mode == BacklinkMode::Ranked {
            // Ranked mode appends into these lists and reads their last entry as
            // the cutoff, so they must be globally ascending from the start.
            let (mut ids, mut dists, mut mask) = (selected.clone(), sel_dists, sel_mask);
            sort_ranked_list(&mut ids, &mut dists, &mut mask);
            g.set_list(id, l, ids, dists, mask);
        } else {
            g.set_list(id, l, selected.clone(), sel_dists, sel_mask);
        }

        // Backlinks with heuristic re-pruning (RAM: no two-phase needed).
        // Pair distances come from the decode-once `pair_buf`; each check is a
        // SIMD kernel call on a small hot buffer (see [`DistBuf`]).
        let cap = cap_for(l);
        for (sel_idx, &n) in selected.iter().enumerate() {
            if timed {
                state.stats.backlink_lists += 1;
            }
            let t_phase = timed.then(std::time::Instant::now);
            let d_self = state.pair_buf.dist_to_query(
                &state.codec,
                state.dist_fn,
                &mut state.stats,
                PairCaller::Backlink,
                &subject,
                &*g,
                n,
            );
            if let Some(t) = t_phase {
                state.stats.backlink_pairs_ns += t.elapsed().as_nanos() as u64;
            }
            let t_phase = timed.then(std::time::Instant::now);
            let n_i = n as usize;
            // Move the list out instead of cloning it: it is overwritten at the
            // end of this iteration anyway, and `mem::take` keeps the same
            // allocation alive for the (much more common) write-back path.
            let existing_ids: Vec<u32> = std::mem::take(&mut g.neighbors[n_i][l]);
            let existing_dists: Vec<f32> = std::mem::take(&mut g.list_dists[n_i][l]);
            let existing_mask = g.list_masks[n_i][l];
            let ranked = if state.reference_backlinks || state.backlink_mode == BacklinkMode::Exact {
                None
            } else {
                Some(backlink_add_ranked_mem(
                    &state.codec,
                    state.dist_fn,
                    &*g,
                    &mut state.pair_buf,
                    &mut state.stats,
                    &existing_ids,
                    &existing_dists,
                    d_self,
                    id,
                    cap,
                    sel_idx == 0,
                ))
            };
            if let Some(ranked) = ranked {
                // Ranked mode: the edge may have been skipped by the cutoff test,
                // in which case the list taken above goes straight back.
                match ranked {
                    Some((ids, dists, mask)) => g.set_list(n, l, ids, dists, mask),
                    None => g.set_list(n, l, existing_ids, existing_dists, existing_mask),
                }
                if let Some(t) = t_phase {
                    state.stats.backlink_select_ns += t.elapsed().as_nanos() as u64;
                }
                continue;
            }
            let (new_ids, new_dists, new_mask) = if state.reference_backlinks {
                // Reference: full heuristic over existing ∪ {new}.
                let mut cands: Vec<(f32, u32)> = Vec::with_capacity(existing_ids.len() + 1);
                for (&mm, &d) in existing_ids.iter().zip(existing_dists.iter()) {
                    cands.push((d, mm));
                }
                cands.push((d_self, id));
                select_neighbors_heuristic_mem(
                    &state.codec,
                    state.dist_fn,
                    &*g,
                    &mut state.pair_buf,
                    &mut state.stats,
                    cands,
                    cap,
                )
            } else {
                backlink_prune_mem(
                    &state.codec,
                    state.dist_fn,
                    &*g,
                    &mut state.pair_buf,
                    &mut state.stats,
                    &existing_ids,
                    &existing_dists,
                    existing_mask,
                    d_self,
                    id,
                    cap,
                )
            };
            if let Some(t) = t_phase {
                state.stats.backlink_select_ns += t.elapsed().as_nanos() as u64;
            }
            g.set_list(n, l, new_ids, new_dists, new_mask);
        }
    }

    if (level as usize) > g.entry_level {
        g.entry = Some(id);
        g.entry_level = level as usize;
    }
}

/// Flush the in-memory graph to disk and switch the build to disk mode.
unsafe fn spill_to_disk(index: &PgRelation, state: &mut BuildState) {
    writeout_graph(state);
    if !state.graph.levels.is_empty() {
        let (entry_ptr, entry_level, insert_page) =
            flush_mem_graph(index, &state.graph, &state.codec, state.m, state.m0);
        let count = state.graph.len() as u64;
        HnswMetaPage::update(index, |meta| {
            meta.set_build_result(entry_ptr, entry_level, count, insert_page);
        });
    }
    state.graph = MemGraph::new();
    state.mem_used = 0;
    state.disk_mode = true;
}

/// Serialized length of a node at `level` for this index shape.
fn level_len(codec: &Codec, level: usize, m: usize, m0: usize) -> usize {
    probe_serialized_len(
        codec.dim(),
        codec.precision().elem_bytes(),
        level,
        m,
        m0,
    )
}

/// Compute the full (block, offset) mapping for the writeout.  Pages extend
/// contiguously from the current relation end (the build holds ShareLock on
/// the heap, so nothing else extends this index) and offsets follow
/// PageAddItemExtended's sequential assignment.
unsafe fn page_plan(
    index: &PgRelation,
    g: &MemGraph,
    codec: &Codec,
    m: usize,
    m0: usize,
) -> Vec<ItemPointer> {
    let capacity = tsv_fresh_page_capacity();
    let mut len_by_level: Vec<usize> = Vec::new();

    let start_block = unsafe {
        pg_sys::RelationGetNumberOfBlocksInFork(
            index.as_ptr(),
            pg_sys::ForkNumber::MAIN_FORKNUM,
        )
    };
    let mut mapping = Vec::with_capacity(g.len());
    // free = 0 forces a fresh page for the first node: block becomes
    // (start_block - 1) + 1 = start_block.
    let mut block = start_block - 1;
    let mut free: i64 = 0;
    let mut next_off: pg_sys::OffsetNumber = 0;
    for id in 0..g.len() {
        let level = g.levels[id] as usize;
        while len_by_level.len() <= level {
            let l = len_by_level.len();
            len_by_level.push(level_len(codec, l, m, m0));
        }
        let len = len_by_level[level];
        if !item_fits(free.max(0) as usize, len) {
            block += 1;
            free = capacity as i64;
            next_off = 0;
        }
        next_off += 1;
        mapping.push(ItemPointer::new(block, next_off));
        free -= (((len + 7) & !7) + std::mem::size_of::<pg_sys::ItemIdData>()) as i64;
    }
    mapping
}

/// Write the memory graph to node pages in one sequential pass, returning
/// `(entry_pointer, entry_level, last_page_block)`.  Each node is serialized
/// exactly once with its final neighbor pointers (append-only).
unsafe fn flush_mem_graph(
    index: &PgRelation,
    g: &MemGraph,
    codec: &Codec,
    m: usize,
    m0: usize,
) -> (ItemPointer, u8, pg_sys::BlockNumber) {
    let mapping = page_plan(index, g, codec, m, m0);
    let n = g.len();
    let mut id = 0usize;
    let mut last_block = pg_sys::InvalidBlockNumber;
    while id < n {
        let block_expected = mapping[id].block_number;
        let mut page = WritablePage::new(index, PageType::HnswNode);
        assert_eq!(
            page.get_block_number(),
            block_expected,
            "hnswsq build: relation extended unexpectedly (concurrent writer?)"
        );
        // Fill this page with every node mapped onto it.
        while id < n && mapping[id].block_number == block_expected {
            let mapped: Vec<Vec<ItemPointer>> = g.neighbors[id]
                .iter()
                .map(|list| list.iter().map(|nid| mapping[*nid as usize]).collect())
                .collect();
            let node = HnswNode::new_clamped(
                g.tids[id],
                g.levels[id],
                g.vectors[id].clone(),
                mapped,
                m,
                m0,
                g.clamped[id],
            );
            let bytes = node.serialize_to_vec();
            let off = page.add_item(&bytes);
            assert_eq!(
                off, mapping[id].offset,
                "hnswsq build: offset plan mismatch"
            );
            id += 1;
        }
        page.commit();
        last_block = block_expected;
    }
    let entry_id = g.entry.unwrap_or(0) as usize;
    (
        mapping[entry_id],
        g.levels[entry_id],
        last_block,
    )
}

/// Build an empty hnswsq index (unlogged-table init fork; PG also routes
/// empty tables through `ambuild`, whose zero-row stream takes the same
/// minimal path).
#[pg_guard]
pub extern "C-unwind" fn ambuildempty(index: pg_sys::Relation) {
    unsafe {
        let index_rel = PgRelation::from_pg(index);
        write_empty_index(&index_rel, index);
    }
}

/// Write the minimal on-disk structure: meta page (+ provisional SQ8
/// calibration) with an Invalid entry point.
unsafe fn write_empty_index(index: &PgRelation, index_ptr: pg_sys::Relation) {    let options = TSVHnswOptions::from_relation(index);
    let precision = options.get_precision();
    let m = options.get_m() as usize;
    let m0 = m * 2;
    let ef_construction = options.get_ef_construction() as u32;

    let atttypmod = index
        .tuple_desc()
        .get(0)
        .map(|attr| attr.atttypmod)
        .unwrap_or(-1);
    if atttypmod < 1 {
        error!(
            "hnswsq: the indexed column must have a fixed dimension (e.g. vector(128)); \
             got atttypmod {}",
            atttypmod
        );
    }
    let num_dimensions = atttypmod as usize;
    let max_level = match compute_max_level(num_dimensions, precision.elem_bytes(), m, m0) {
        Some(ml) => ml,
        None => error!(
            "hnswsq: {} dimensions do not fit a page item with storage_layout={}",
            num_dimensions,
            precision.as_str()
        ),
    };

    // Resolve the real distance type (the init fork must carry it: inserts
    // after an unlogged-table reset read it from the meta page).
    let distance_type = resolve_distance_type(index_ptr);

    // Meta page first (block 0), then the provisional SQ8 calibration chain,
    // then record its pointer (same ordering rule as ambuild).
    unsafe {
        HnswMetaPage::create(
            index,
            num_dimensions as u32,
            distance_type,
            precision,
            m as u16,
            ef_construction,
            max_level,
            ItemPointer::new_invalid(),
        );
    }
    if precision.needs_calibration() {
        let calib = Sq8Calibration::provisional(num_dimensions);
        let ptr = unsafe { calib.store(index) };
        unsafe {
            HnswMetaPage::update(index, |mm| mm.set_calibration_pointer(ptr));
        }
    }
}

#[cfg(test)]
mod mem_tests {
    //! Pure in-memory build + search recall simulation (no PostgreSQL):
    //! isolates graph quality from the disk/writeout path.

    use super::*;
    use crate::access_method::distance::distance_l2;
    use crate::access_method::hnswsq::graph::{greedy_descent, search_layer};
    use crate::access_method::hnswsq::quantize::HnswPrecision;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn clustered(
        n_clusters: usize,
        per_cluster: usize,
        dim: usize,
        noise: f32,
        seed: u64,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let centers: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect();
        let mut rows = Vec::new();
        for c in &centers {
            for _ in 0..per_cluster {
                rows.push(
                    c.iter()
                        .map(|x| x + (rng.gen::<f32>() - 0.5) * noise)
                        .collect(),
                );
            }
        }
        let queries: Vec<Vec<f32>> = centers
            .iter()
            .map(|c| {
                c.iter()
                    .map(|x| x + (rng.gen::<f32>() - 0.5) * noise * 0.2)
                    .collect()
            })
            .collect();
        (rows, queries)
    }

    fn mem_search(
        state: &BuildState,
        query: &[f32],
        ef: usize,
    ) -> Vec<u32> {
        let g = &state.graph;
        let Some(ep) = g.entry else { return Vec::new() };
        let d0 = distance_encoded(
            &state.codec,
            state.distance_type,
            query,
            &g.vectors[ep as usize],
        );
        let mut cur = (d0, ep);
        if g.entry_level > 0 {
            cur = greedy_descent(
                &state.codec,
                state.distance_type,
                query,
                &*g,
                cur,
                g.entry_level,
                1,
            );
        }
        search_layer(&state.codec, state.distance_type, query, &*g, vec![cur], ef, 0)
            .into_iter()
            .map(|h| h.id)
            .collect()
    }

    /// Randomized function-level check: for random list states, the
    /// incremental backlink prune must equal the full heuristic over
    /// `existing ∪ {new}`.
    #[test]
    fn test_backlink_prune_fuzz() {
        use rand::Rng;
        let dim = 6usize;
        let cap = 8usize;
        let n_nodes = 40u32;
        let codec = Codec::new(HnswPrecision::Plain, dim);
        let mut rng = SmallRng::seed_from_u64(99);

        let mut g = MemGraph::new();
        let mut vecs: Vec<Vec<f32>> = Vec::new();
        for i in 0..n_nodes {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            g.push_node(0, ItemPointer::new(i + 1, 1), false, codec.encode(&v));
            vecs.push(v);
        }
        let owner = 0u32;

        for case in 0..2000 {
            let mut buf = DistBuf::new(dim);
            let mut stats = BuildStats::default();

            // Random existing list, produced by the full heuristic so the mask
            // has the same provenance as a real backlink list.
            let n_cands = rng.gen_range(1..(cap + 3));
            let mut cands: Vec<(f32, u32)> = Vec::new();
            let mut used: Vec<u32> = vec![owner];
            for _ in 0..n_cands {
                let mut id = rng.gen_range(1..n_nodes);
                while used.contains(&id) {
                    id = rng.gen_range(1..n_nodes);
                }
                used.push(id);
                cands.push((distance_l2(&vecs[owner as usize], &vecs[id as usize]), id));
            }
            let (existing_ids, existing_dists, existing_mask) = select_neighbors_heuristic_mem(
                &codec, distance_l2, &g, &mut buf, &mut stats, cands, cap,
            );

            // A fresh node not in the list.
            let mut new_id = rng.gen_range(1..n_nodes);
            while existing_ids.contains(&new_id) {
                new_id = rng.gen_range(1..n_nodes);
            }
            let d_new = distance_l2(&vecs[owner as usize], &vecs[new_id as usize]);

            // Reference: full heuristic over existing ∪ {new}.
            let mut merged: Vec<(f32, u32)> = existing_ids
                .iter()
                .copied()
                .zip(existing_dists.iter().copied())
                .map(|(id, d)| (d, id))
                .collect();
            merged.push((d_new, new_id));
            let (ref_ids, _, _) = select_neighbors_heuristic_mem(
                &codec, distance_l2, &g, &mut buf, &mut stats, merged, cap,
            );

            let (inc_ids, _, _) = backlink_prune_mem(
                &codec,
                distance_l2,
                &g,
                &mut buf,
                &mut stats,
                &existing_ids,
                &existing_dists,
                existing_mask,
                d_new,
                new_id,
                cap,
            );

            let mut dbg: Vec<(f32, u32)> = existing_ids
                .iter()
                .copied()
                .zip(existing_dists.iter().copied())
                .map(|(id, d)| (d, id))
                .collect();
            dbg.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            assert_eq!(
                ref_ids, inc_ids,
                "case {}: existing_sorted={:?} mask={:?} new={} d_new={} -> reference {:?} vs incremental {:?}",
                case, dbg, existing_mask, new_id, d_new, ref_ids, inc_ids
            );
        }
    }

    /// Same exactness check as above, but over the dataset/thresholds used by
    /// the pg-level structure test and many build seeds — the production build
    /// seeds its level RNG from entropy, so equality must hold for every seed,
    /// not just the ones a single fixed-seed test happens to cover.
    #[test]
    fn test_backlink_prune_matches_full_heuristic_many_seeds() {
        let dim = 16;
        let (rows, _) = clustered(20, 50, dim, 0.05, 12345);
        let build = |reference: bool, seed: u64| -> BuildState {
            let codec = Codec::new(HnswPrecision::Plain, dim);
            let m = 16usize;
            let m0 = 32usize;
            let mut state = BuildState {
                codec,
                dist_fn: distance_l2,
                distance_type: DistanceType::L2,
                m,
                m0,
                ef_construction: 64,
                ml: 1.0 / (m as f64).ln() as f32,
                max_level: compute_max_level(dim, 4, m, m0).unwrap(),
                budget_bytes: u64::MAX,
                mem_used: 0,
                graph: MemGraph::new(),
                pair_buf: DistBuf::new(dim),
                flat: None,
                stats: BuildStats::default(),
                search_scratch: SearchScratch::new(),
                reference_backlinks: reference,
                backlink_mode: BacklinkMode::Exact,
                disk_mode: false,
                nrows: 0,
                rng: SmallRng::seed_from_u64(seed),
                level_seed: seed,
            };
            for (i, v) in rows.iter().enumerate() {
                mem_insert(&mut state, ItemPointer::new(i as u32 + 1, 1), v);
            }
            state
        };
        for seed in 0..40u64 {
            let reference = build(true, seed);
            let incremental = build(false, seed);
            for node in 0..reference.graph.len() {
                for layer in 0..reference.graph.neighbors[node].len() {
                    assert_eq!(
                        reference.graph.neighbors[node][layer],
                        incremental.graph.neighbors[node][layer],
                        "seed {} node {} layer {}",
                        seed,
                        node,
                        layer
                    );
                    assert_eq!(
                        reference.graph.list_dists[node][layer],
                        incremental.graph.list_dists[node][layer],
                        "distances seed {} node {} layer {}",
                        seed,
                        node,
                        layer
                    );
                    assert_eq!(
                        reference.graph.list_masks[node][layer],
                        incremental.graph.list_masks[node][layer],
                        "mask seed {} node {} layer {}",
                        seed,
                        node,
                        layer
                    );
                }
            }
        }
    }

    /// Interleaves the reference and incremental builds insert by insert and
    /// asserts that every neighbor list, its distances and its heuristic mask
    /// stay identical after each one (the end-state test above would not catch
    /// a divergence that later washed out).
    #[test]
    fn test_backlink_incremental_matches_every_insert() {
        let dim = 12;
        let (rows, _) = clustered(16, 40, dim, 0.08, 4242);
        let mk = |reference: bool| -> BuildState {
            let codec = Codec::new(HnswPrecision::Plain, dim);
            let m = 16usize;
            let m0 = 32usize;
            BuildState {
                codec,
                dist_fn: distance_l2,
                distance_type: DistanceType::L2,
                m,
                m0,
                ef_construction: 64,
                ml: 1.0 / (m as f64).ln() as f32,
                max_level: compute_max_level(dim, 4, m, m0).unwrap(),
                budget_bytes: u64::MAX,
                mem_used: 0,
                graph: MemGraph::new(),
                pair_buf: DistBuf::new(dim),
                flat: None,
                stats: BuildStats::default(),
                search_scratch: SearchScratch::new(),
                reference_backlinks: reference,
                backlink_mode: BacklinkMode::Exact,
                disk_mode: false,
                nrows: 0,
                rng: SmallRng::seed_from_u64(7),
                level_seed: 7,
            }
        };
        let mut reference = mk(true);
        let mut incremental = mk(false);
        for (i, v) in rows.iter().enumerate() {
            // snapshot incremental lists before this insert
            let before_lists: Vec<Vec<Vec<u32>>> = incremental.graph.neighbors.clone();
            let before_dists: Vec<Vec<Vec<f32>>> = incremental.graph.list_dists.clone();
            let before_masks: Vec<Vec<[u64; 2]>> = incremental.graph.list_masks.clone();
            mem_insert(&mut reference, ItemPointer::new(i as u32 + 1, 1), v);
            mem_insert(&mut incremental, ItemPointer::new(i as u32 + 1, 1), v);
            for node in 0..incremental.graph.len() {
                for layer in 0..incremental.graph.neighbors[node].len() {
                    let r = &reference.graph.neighbors[node][layer];
                    let c = &incremental.graph.neighbors[node][layer];
                    let rd = &reference.graph.list_dists[node][layer];
                    let cd = &incremental.graph.list_dists[node][layer];
                    let rm = reference.graph.list_masks[node][layer];
                    let cm = incremental.graph.list_masks[node][layer];
                    if r == c && (rd != cd || rm != cm) {
                        panic!(
                            "STATE diverged (ids equal) after insert {} node {} layer {}: ref_dists={:?} inc_dists={:?} ref_mask={:?} inc_mask={:?}",
                            i, node, layer, rd, cd, rm, cm
                        );
                    }
                    if r != c {
                        let owner_v: Vec<f32> = (0..dim)
                            .map(|d| f32::from_le_bytes([
                                incremental.graph.vectors[node][d * 4],
                                incremental.graph.vectors[node][d * 4 + 1],
                                incremental.graph.vectors[node][d * 4 + 2],
                                incremental.graph.vectors[node][d * 4 + 3],
                            ]))
                            .collect();
                        let new_id = (i as u32).max(0);
                        let d_new = distance_l2(&owner_v, v);
                        panic!(
                            "insert {} (new_id={}) node {} layer {}: reference {:?} incremental {:?}\nPRE existing_ids={:?}\nPRE existing_dists={:?}\nPRE mask={:?}\nd_new={}\n",
                            i,
                            new_id,
                            node,
                            layer,
                            r,
                            c,
                            before_lists.get(node).and_then(|l| l.get(layer)).cloned().unwrap_or_default(),
                            before_dists.get(node).and_then(|l| l.get(layer)).cloned().unwrap_or_default(),
                            before_masks.get(node).and_then(|l| l.get(layer)).copied().unwrap_or([0, 0]),
                            d_new
                        );
                    }
                }
            }
        }
    }

    /// Temporary diagnostic: ranked-mode counters + recall on one graph.
    #[test]
    fn test_backlink_ranked_mode_diag() {
        let dim = 16;
        let (rows, queries) = clustered(20, 50, dim, 0.05, 12345);
        let codec = Codec::new(HnswPrecision::Plain, dim);
        let (m, m0) = (16usize, 32usize);
        let mut state = BuildState {
            codec,
            dist_fn: distance_l2,
            distance_type: DistanceType::L2,
            m,
            m0,
            ef_construction: 64,
            ml: 1.0 / (m as f64).ln() as f32,
            max_level: compute_max_level(dim, 4, m, m0).unwrap(),
            budget_bytes: u64::MAX,
            mem_used: 0,
            graph: MemGraph::new(),
            pair_buf: DistBuf::new(dim),
            flat: None,
            stats: BuildStats {
                enabled: true,
                ..Default::default()
            },
            search_scratch: SearchScratch::new(),
            reference_backlinks: false,
            backlink_mode: BacklinkMode::Ranked,
            disk_mode: false,
            nrows: 0,
            rng: SmallRng::seed_from_u64(7),
            level_seed: 7,
        };
        for (i, v) in rows.iter().enumerate() {
            mem_insert(&mut state, ItemPointer::new(i as u32 + 1, 1), v);
        }
        println!("{}", state.stats.summary());
        // recall of the ranked graph
        let mut hit = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let cand = mem_search(&state, q, 100);
            let mut exact: Vec<(f32, u32)> = rows
                .iter()
                .enumerate()
                .map(|(i, v)| (distance_l2(q, v), i as u32))
                .collect();
            exact.sort_by(|a, b| a.0.total_cmp(&b.0));
            let top: Vec<u32> = exact.iter().take(10).map(|(_, i)| *i).collect();
            for t in &top {
                if cand.contains(t) {
                    hit += 1;
                }
                total += 1;
            }
        }
        println!("ranked recall = {}", hit as f64 / total as f64);
        // Also dump the degree distribution of layer 0 to spot missing backlinks.
        let mut degs: Vec<usize> = state
            .graph
            .neighbors
            .iter()
            .map(|layers| layers[0].len())
            .collect();
        degs.sort_unstable();
        println!(
            "layer0 degree: min={} p50={} max={} avg={:.1}",
            degs[0],
            degs[degs.len() / 2],
            degs[degs.len() - 1],
            degs.iter().sum::<usize>() as f64 / degs.len() as f64
        );
    }

    /// Ranked (Lance-style) admission must not cost recall    /// The memory search must agree exactly with the generic `search_layer`
    /// (which the disk path uses): same hits, same order.
    #[test]
    fn test_search_layer_mem_matches_generic() {
        use crate::access_method::hnswsq::graph::search_layer;
        let dim = 12;
        let (rows, queries) = clustered(16, 40, dim, 0.08, 4242);
        let codec = Codec::new(HnswPrecision::Plain, dim);
        let (m, m0) = (16usize, 32usize);
        let mut state = BuildState {
            codec,
            dist_fn: distance_l2,
            distance_type: DistanceType::L2,
            m,
            m0,
            ef_construction: 64,
            ml: 1.0 / (m as f64).ln() as f32,
            max_level: compute_max_level(dim, 4, m, m0).unwrap(),
            budget_bytes: u64::MAX,
            mem_used: 0,
            graph: MemGraph::new(),
            pair_buf: DistBuf::new(dim),
            flat: None,
            stats: BuildStats::default(),
            search_scratch: SearchScratch::new(),
            reference_backlinks: false,
            backlink_mode: BacklinkMode::Exact,
            disk_mode: false,
            nrows: 0,
            rng: SmallRng::seed_from_u64(3),
            level_seed: 3,
        };
        for (i, v) in rows.iter().enumerate() {
            mem_insert(&mut state, ItemPointer::new(i as u32 + 1, 1), v);
        }

        let g = &state.graph;
        let ep = g.entry.expect("entry");
        for q in queries.iter().take(40) {
            for ef in [1usize, 4, 17, 64] {
                for layer in 0..=(g.entry_level.min(2)) {
                    let entry_dist = distance_encoded(&state.codec, DistanceType::L2, q, &g.vectors[ep as usize]);
                    let entries = vec![(entry_dist, ep)];
                    let generic = search_layer(
                        &state.codec,
                        DistanceType::L2,
                        q,
                        g,
                        entries.clone(),
                        ef,
                        layer,
                    );
                    let mem = search_layer_mem(
                        &state.codec,
                        DistanceType::L2,
                        q,
                        g,
                        &entries,
                        ef,
                        layer,
                        &mut state.search_scratch,
                    );
                    let gids: Vec<(u32, u32)> = generic
                        .iter()
                        .map(|h| (h.id, h.dist.to_bits()))
                        .collect();
                    let mids: Vec<(u32, u32)> = mem.iter().map(|h| (h.id, h.dist.to_bits())).collect();
                    assert_eq!(
                        gids, mids,
                        "ef={ef} layer={layer}: memory search diverged from the generic path"
                    );
                }
            }
        }
    }

    /// The ranked (Lance-style `cutoff`) admission mode is kept as an    /// The ranked (Lance-style `cutoff`) admission mode is kept as an
    /// experimental knob, not as a default: measured against the exact mode on
    /// the same data and build seed it is both slower (a full heuristic per
    /// admitted edge) and lower recall.  This test pins the *observed* deficit
    /// so a future change that makes the mode worse is caught, and documents
    /// why the exact mode is the default.
    #[test]
    fn test_backlink_ranked_mode_recall_floor() {
        let mut worst_shortfall = 0.0f64;
        for seed in 0..16u64 {
            let exact = run_sim_mode(12345, seed, BacklinkMode::Exact);
            let ranked = run_sim_mode(12345, seed, BacklinkMode::Ranked);
            assert!(
                exact >= 0.90,
                "seed {seed}: exact mode recall regressed to {exact:.4}"
            );
            assert!(
                ranked >= 0.85,
                "seed {seed}: ranked mode recall {ranked:.4} below the documented floor"
            );
            worst_shortfall = worst_shortfall.max(exact - ranked);
        }
        println!("worst ranked-vs-exact recall shortfall = {worst_shortfall:.4}");
        assert!(
            worst_shortfall <= 0.06,
            "ranked mode deficit grew to {worst_shortfall:.4} (was <= 0.03 when measured)"
        );
    }

    /// Builds the same random graph twice — once with the incremental backlink    /// Builds the same random graph twice — once with the incremental backlink
    /// re-prune, once with the full neighbor-selection heuristic over the
    /// merged candidate set — and asserts every neighbor list is identical
    /// (ids, order and length).  Proves `backlink_prune_mem` is exact.
    #[test]
    fn test_backlink_prune_matches_full_heuristic() {
        let dim = 12;
        let (rows, _) = clustered(16, 40, dim, 0.08, 4242);

        let build = |reference: bool| -> BuildState {
            let codec = Codec::new(HnswPrecision::Plain, dim);
            let m = 16usize;
            let m0 = 32usize;
            let mut state = BuildState {
                codec,
                dist_fn: distance_l2,
                distance_type: DistanceType::L2,
                m,
                m0,
                ef_construction: 64,
                ml: 1.0 / (m as f64).ln() as f32,
                max_level: compute_max_level(dim, 4, m, m0).unwrap(),
                budget_bytes: u64::MAX,
                mem_used: 0,
                graph: MemGraph::new(),
                pair_buf: DistBuf::new(dim),
                flat: None,
                stats: BuildStats::default(),
                search_scratch: SearchScratch::new(),
                reference_backlinks: reference,
                backlink_mode: BacklinkMode::Exact,
                disk_mode: false,
                nrows: 0,
                rng: SmallRng::seed_from_u64(7),
                level_seed: 7,
            };
            for (i, v) in rows.iter().enumerate() {
                mem_insert(&mut state, ItemPointer::new(i as u32 + 1, 1), v);
            }
            state
        };

        let reference = build(true);
        let incremental = build(false);

        assert_eq!(reference.graph.len(), incremental.graph.len());
        let mut lists_compared = 0usize;
        let mut lists_nonempty = 0usize;
        for (node, layers) in reference.graph.neighbors.iter().enumerate() {
            for (layer, ref_ids) in layers.iter().enumerate() {
                let inc_ids = &incremental.graph.neighbors[node][layer];
                assert_eq!(
                    ref_ids, inc_ids,
                    "neighbor list differs at node {} layer {} (reference {:?} vs incremental {:?})",
                    node, layer, ref_ids, inc_ids
                );
                // The stored distance vector must stay consistent too.
                let inc_dists = &incremental.graph.list_dists[node][layer];
                assert_eq!(inc_ids.len(), inc_dists.len(), "ids/dists length mismatch");
                for (&id, &d) in inc_ids.iter().zip(inc_dists.iter()) {
                    let expected = distance_l2(
                        &reference.graph.vectors[node as usize]
                            .iter()
                            .enumerate()
                            .step_by(4)
                            .map(|(i, _)| f32::from_le_bytes([
                                reference.graph.vectors[node as usize][i],
                                reference.graph.vectors[node as usize][i + 1],
                                reference.graph.vectors[node as usize][i + 2],
                                reference.graph.vectors[node as usize][i + 3],
                            ]))
                            .collect::<Vec<f32>>(),
                        &reference.graph.vectors[id as usize]
                            .iter()
                            .enumerate()
                            .step_by(4)
                            .map(|(i, _)| f32::from_le_bytes([
                                reference.graph.vectors[id as usize][i],
                                reference.graph.vectors[id as usize][i + 1],
                                reference.graph.vectors[id as usize][i + 2],
                                reference.graph.vectors[id as usize][i + 3],
                            ]))
                            .collect::<Vec<f32>>(),
                    );
                    assert!(
                        (d - expected).abs() <= 1e-5 * (1.0 + expected.abs()),
                        "stored distance {} != recomputed {} (node {} -> {})",
                        d,
                        expected,
                        node,
                        id
                    );
                }
                lists_compared += 1;
                if !ref_ids.is_empty() {
                    lists_nonempty += 1;
                }
            }
        }
        assert!(lists_compared > rows.len(), "expected many lists compared");
        assert!(lists_nonempty > rows.len(), "expected populated lists");
    }

    fn run_sim(data_seed: u64, build_seed: u64) -> f64 {
        run_sim_mode(data_seed, build_seed, BacklinkMode::Exact)
    }

    fn run_sim_mode(data_seed: u64, build_seed: u64, mode: BacklinkMode) -> f64 {
        let dim = 16;
        let (rows, queries) = clustered(20, 50, dim, 0.05, data_seed);
        let codec = Codec::new(HnswPrecision::Plain, dim);
        let m = 16usize;
        let m0 = 32usize;
        let mut state = BuildState {
            codec,
            dist_fn: distance_l2,
            distance_type: DistanceType::L2,
            m,
            m0,
            ef_construction: 64,
            ml: 1.0 / (m as f64).ln() as f32,
            max_level: compute_max_level(dim, 4, m, m0).unwrap(),
            budget_bytes: u64::MAX,
            mem_used: 0,
            graph: MemGraph::new(),
            pair_buf: DistBuf::new(dim),
            flat: None,
            stats: BuildStats::default(),
            search_scratch: SearchScratch::new(),
            reference_backlinks: false,
            backlink_mode: mode,
            disk_mode: false,
            nrows: 0,
            rng: SmallRng::seed_from_u64(build_seed),
            level_seed: build_seed,
        };
        for (i, v) in rows.iter().enumerate() {
            mem_insert(&mut state, ItemPointer::new(i as u32 + 1, 1), v);
        }
        assert_eq!(state.graph.len(), rows.len());

        let mut hit = 0usize;
        let mut total = 0usize;
        for q in &queries {
            // exact top-10 by brute force
            let mut exact: Vec<(f32, u32)> = rows
                .iter()
                .enumerate()
                .map(|(i, v)| (distance_l2(q, v), i as u32))
                .collect();
            exact.sort_by(|a, b| a.0.total_cmp(&b.0));
            let exact_top: Vec<u32> = exact.iter().take(10).map(|(_, i)| *i).collect();

            let cand = mem_search(&state, q, 100);
            // exact re-rank of candidates (executor behavior), take 10
            let mut reranked: Vec<(f32, u32)> = cand
                .iter()
                .map(|id| (distance_l2(q, &rows[*id as usize]), *id))
                .collect();
            reranked.sort_by(|a, b| a.0.total_cmp(&b.0));
            let ann_top: Vec<u32> = reranked.iter().take(10).map(|(_, i)| *i).collect();

            for id in &exact_top {
                if ann_top.contains(id) {
                    hit += 1;
                }
                total += 1;
            }
        }
        hit as f64 / total as f64
    }

    #[test]
    fn test_mem_build_recall_clustered() {
        let recall = run_sim(12345, 777);
        assert!(
            recall >= 0.95,
            "in-memory HNSW recall@10 = {} (ef_search=100, ef_c=64, m=16)",
            recall
        );
    }

    #[test]
    fn test_mem_build_recall_seed_variance() {
        let mut recalls = Vec::new();
        for build_seed in 1u64..=24 {
            recalls.push(run_sim(12345, build_seed));
        }
        for data_seed in [999u64, 31337, 55555] {
            recalls.push(run_sim(data_seed, 42));
        }
        let min = recalls.iter().cloned().fold(f64::INFINITY, f64::min);
        let mean = recalls.iter().sum::<f64>() / recalls.len() as f64;
        println!("recall min={} mean={:.4} all={:?}", min, mean, recalls);
        assert!(
            min >= 0.90,
            "worst recall across seeds = {:?} (all: {:?})",
            min,
            recalls
        );
    }
}
