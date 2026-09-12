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
    distance_encoded, greedy_descent, random_level, search_layer, GraphAccess, VisitData,
};
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
    /// re-fetching them from the pair cache for every revision.
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
        })
    }

    fn vector(&self, id: u32) -> Option<Vec<u8>> {
        self.vectors.get(id as usize).cloned()
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
    /// Build-scoped cache of pairwise (decoded-vector) distances, keyed by
    /// the unordered id pair.  The backlink re-pruning recomputes each
    /// neighbor-list pair distance on every revision; the cache turns that
    /// repeated work into one computation per pair.  Bounded: cleared when
    /// it grows past `PAIR_CACHE_MAX` (~250MB), so long builds stay within
    /// the maintenance_work_mem budget.
    pair_dist_cache: std::collections::HashMap<(u32, u32), f32>,
    /// Per-phase timing counters (`hnswsq.build_stats`).
    stats: BuildStats,
    /// Test-only reference path: re-run the *full* neighbor-selection
    /// heuristic for every backlink instead of the incremental
    /// [`backlink_prune_mem`].  Used by
    /// `test_backlink_prune_matches_full_heuristic` to prove the incremental
    /// result is identical; always false in production builds.
    reference_backlinks: bool,
    /// True after the memory graph was flushed (or for concurrent builds);
    /// remaining rows go through the disk insert path.
    disk_mode: bool,
    nrows: u64,
    rng: SmallRng,
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
    pub pair_lookups: u64,
    pub pair_misses: u64,
    pub flush_ns: u64,
    /// Backlink entries resolved by the O(1) "only the new node can occlude"
    /// path vs entries that needed a full re-check (previously backfilled or
    /// never occlusion-evaluated), plus how many extras were tracked.
    pub fast_entries: u64,
    pub full_entries: u64,
    pub extras_seen: u64,
    /// Pair-cache traffic split by caller, so a high miss rate can be
    /// attributed (own-list selection vs backlink admission vs extras).
    pub pair_lookups_select: u64,
    pub pair_misses_select: u64,
    pub pair_lookups_backlink: u64,
    pub pair_misses_backlink: u64,
    pub pair_lookups_extras: u64,
    pub pair_misses_extras: u64,
    /// Beam-search calls and the neighbour hits they returned (proxy for the
    /// work the search path does per insert).
    pub search_calls: u64,
    pub search_hits: u64,
}

/// Which hot loop is asking the pair cache for a distance.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PairCaller {
    /// Neighbour selection for the inserted node's own lists.
    Select,
    /// Backlink admission/re-prune distances.
    Backlink,
    /// Checks against entries newly accepted in this pass ("extras").
    Extras,
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
            "hnswsq build stats: nodes={} backlink_lists={} pair_lookups={} pair_misses={} \
             search={:.1}ms select={:.1}ms backlink_pairs={:.1}ms backlink_select={:.1}ms \
             flush={:.1}ms accounted_total={:.1}ms fast_entries={} full_entries={} extras_seen={} \
             pair(select)={}/{} pair(backlink)={}/{} pair(extras)={}/{} search_calls={} search_hits={}",
            self.nodes,
            self.backlink_lists,
            self.pair_lookups,
            self.pair_misses,
            ms(self.search_ns),
            ms(self.select_ns),
            ms(self.backlink_pairs_ns),
            ms(self.backlink_select_ns),
            ms(self.flush_ns),
            total,
            self.fast_entries,
            self.full_entries,
            self.extras_seen,
            self.pair_lookups_select,
            self.pair_misses_select,
            self.pair_lookups_backlink,
            self.pair_misses_backlink,
            self.pair_lookups_extras,
            self.pair_misses_extras,
            self.search_calls,
            self.search_hits,
        )
    }
}

/// Pair-distance cache cap (~4M entries ≈ 250MB).
const PAIR_CACHE_MAX: usize = 4 << 20;

/// Distance between two memory-graph nodes, computed once and cached.
#[inline]
fn cached_pair_dist(
    codec: &Codec,
    dist_fn: DistanceFn,
    g: &MemGraph,
    cache: &mut std::collections::HashMap<(u32, u32), f32>,
    stats: &mut BuildStats,
    caller: PairCaller,
    a: u32,
    b: u32,
) -> f32 {
    if a == b {
        return 0.0;
    }
    let key = if a < b { (a, b) } else { (b, a) };
    if stats.enabled {
        stats.pair_lookups += 1;
        match caller {
            PairCaller::Select => stats.pair_lookups_select += 1,
            PairCaller::Backlink => stats.pair_lookups_backlink += 1,
            PairCaller::Extras => stats.pair_lookups_extras += 1,
        }
    }
    if let Some(&d) = cache.get(&key) {
        return d;
    }
    if stats.enabled {
        stats.pair_misses += 1;
        match caller {
            PairCaller::Select => stats.pair_misses_select += 1,
            PairCaller::Backlink => stats.pair_misses_backlink += 1,
            PairCaller::Extras => stats.pair_misses_extras += 1,
        }
    }
    let va = codec.decode(&g.vectors[a as usize]);
    let d = dist_fn(&va, &codec.decode(&g.vectors[b as usize]));
    if cache.len() >= PAIR_CACHE_MAX {
        cache.clear();
    }
    cache.insert(key, d);
    d
}

/// Memory-graph SELECT-NEIGHBORS-HEURISTIC with a cached pair distance
/// oracle (see [`cached_pair_dist`]); same semantics as the disk-graph
/// [`select_neighbors_heuristic`].
pub(crate) fn select_neighbors_heuristic_mem(
    codec: &Codec,
    dist_fn: DistanceFn,
    g: &MemGraph,
    cache: &mut std::collections::HashMap<(u32, u32), f32>,
    stats: &mut BuildStats,
    candidates: Vec<(f32, u32)>,
    cap: usize,
) -> (Vec<u32>, Vec<f32>, [u64; LIST_MASK_WORDS]) {
    if candidates.len() <= cap {
        // Room for everyone: the returned list is every candidate in ascending
        // order.  The mask still records, for each entry, whether the occlusion
        // rule would have accepted it — the incremental backlink re-prune
        // relies on that bit, and it must not claim "accepted" for an entry
        // that was never evaluated.
        let mut sorted = candidates;
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        sorted.truncate(cap);
        let mut accepted: Vec<(f32, u32)> = Vec::with_capacity(sorted.len());
        let mut mask = [0u64; LIST_MASK_WORDS];
        for (i, &(d, id)) in sorted.iter().enumerate() {
            let mut keep = true;
            for sel in &accepted {
                if cached_pair_dist(codec, dist_fn, g, cache, stats, PairCaller::Select, id, sel.1) < d
                {
                    keep = false;
                    break;
                }
            }
            if keep {
                accepted.push((d, id));
                if i < LIST_MASK_WORDS * 64 {
                    mask[i / 64] |= 1u64 << (i % 64);
                }
            }
        }
        let ids: Vec<u32> = sorted.iter().map(|c| c.1).collect();
        let dists: Vec<f32> = sorted.iter().map(|c| c.0).collect();
        return (ids, dists, mask);
    }
    let mut sorted = candidates;
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut selected: Vec<(f32, u32)> = Vec::with_capacity(cap);
    let mut pruned: Vec<(f32, u32)> = Vec::new();
    for cand in sorted {
        if selected.len() >= cap {
            pruned.push(cand);
            continue;
        }
        let mut keep = true;
        for sel in &selected {
            if cached_pair_dist(
                codec,
                dist_fn,
                g,
                cache,
                stats,
                PairCaller::Select,
                cand.1,
                sel.1,
            ) < cand.0
            {
                keep = false;
                break;
            }
        }
        if keep {
            selected.push(cand);
        } else {
            pruned.push(cand);
        }
    }
    // Backfill with the closest pruned candidates (ascending order).
    let main_loop_len = selected.len();
    for p in pruned {
        if selected.len() >= cap {
            break;
        }
        selected.push(p);
    }
    let mut mask = [0u64; LIST_MASK_WORDS];
    let mut ids = Vec::with_capacity(selected.len());
    let mut dists = Vec::with_capacity(selected.len());
    for (i, (d, id)) in selected.into_iter().enumerate() {
        // Entries the main loop accepted keep their bit; entries re-added by
        // the backfill stay clear (they were pruned by the occlusion rule).
        if i < main_loop_len && i < LIST_MASK_WORDS * 64 {
            mask[i / 64] |= 1u64 << (i % 64);
        }
        ids.push(id);
        dists.push(d);
    }
    (ids, dists, mask)
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
#[allow(clippy::too_many_arguments)]
fn backlink_prune_mem(
    codec: &Codec,
    dist_fn: DistanceFn,
    g: &MemGraph,
    cache: &mut std::collections::HashMap<(u32, u32), f32>,
    stats: &mut BuildStats,
    existing_ids: &[u32],
    existing_dists: &[f32],
    existing_mask: [u64; LIST_MASK_WORDS],
    d_new: f32,
    new_id: u32,
    cap: usize,
) -> (Vec<u32>, Vec<f32>, [u64; LIST_MASK_WORDS]) {
    debug_assert_eq!(existing_ids.len(), existing_dists.len());

    // The old list is ascending by distance (heuristic output order); the new
    // node slots in at the first position where (dist, id) wins.
    let mut merged: Vec<(f32, u32, bool)> = Vec::with_capacity(existing_ids.len() + 1);
    for (i, (&id, &d)) in existing_ids.iter().zip(existing_dists.iter()).enumerate() {
        let was_selected = (existing_mask[i / 64] >> (i % 64)) & 1 == 1;
        merged.push((d, id, was_selected));
    }
    // Old entries may not be perfectly sorted if a list ever arrived from a
    // different path; sorting keeps this exact rather than assumed.
    merged.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    // Insertion position under the heuristic's (dist, id) ordering.
    let p = merged
        .iter()
        .position(|c| {
            d_new.total_cmp(&c.0).then_with(|| new_id.cmp(&c.1)) == std::cmp::Ordering::Less
        })
        .unwrap_or(merged.len());
    merged.insert(p, (d_new, new_id, false));


    // Room for everyone: the list is every candidate in ascending order (the
    // heuristic's fast path).  The mask must still record which entries the
    // occlusion rule would accept — the same simulation the full heuristic's
    // fast path does — otherwise later incremental updates would trust an
    // acceptance that was never evaluated.
    if merged.len() <= cap {
        let mut accepted: Vec<(f32, u32)> = Vec::with_capacity(merged.len());
        let mut mask = [0u64; LIST_MASK_WORDS];
        for (i, &(d, id, _)) in merged.iter().enumerate() {
            let mut keep = true;
            for sel in &accepted {
                if cached_pair_dist(codec, dist_fn, g, cache, stats, PairCaller::Select, id, sel.1) < d
                {
                    keep = false;
                    break;
                }
            }
            if keep {
                accepted.push((d, id));
                if i < LIST_MASK_WORDS * 64 {
                    mask[i / 64] |= 1u64 << (i % 64);
                }
            }
        }
        let ids: Vec<u32> = merged.iter().map(|c| c.1).collect();
        let dists: Vec<f32> = merged.iter().map(|c| c.0).collect();
        return (ids, dists, mask);
    }

    let mut selected: Vec<(f32, u32)> = Vec::with_capacity(cap);
    let mut pruned: Vec<(f32, u32)> = Vec::new();
    let mut new_selected = false;
    // Entries the occlusion rule had *not* accepted before but accepts now
    // (their previous occluder is gone, or they were never evaluated).  They
    // are the only occluders a previously-accepted entry has not already been
    // checked against, so mask-set entries only need to be tested against
    // these plus the new node — instead of rescanning `selected`.
    let mut extras: Vec<u32> = Vec::new();
    for (i, &(d, id, was_selected)) in merged.iter().enumerate() {
        if selected.len() >= cap {
            pruned.push((d, id));
            continue;
        }
        let keep = if i == p {
            // The new node: full occlusion check against the accepted prefix.
            let mut keep = true;
            for sel in &selected {
                if cached_pair_dist(
                    codec,
                    dist_fn,
                    g,
                    cache,
                    stats,
                    PairCaller::Backlink,
                    id,
                    sel.1,
                ) < d
                {
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
                if cached_pair_dist(
                    codec,
                    dist_fn,
                    g,
                    cache,
                    stats,
                    PairCaller::Backlink,
                    id,
                    new_id,
                ) < d
                {
                    keep = false;
                }
            }
            if keep {
                for &extra in &extras {
                    if cached_pair_dist(
                        codec,
                        dist_fn,
                        g,
                        cache,
                        stats,
                        PairCaller::Extras,
                        id,
                        extra,
                    ) < d
                    {
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
                if cached_pair_dist(
                    codec,
                    dist_fn,
                    g,
                    cache,
                    stats,
                    PairCaller::Backlink,
                    id,
                    sel.1,
                ) < d
                {
                    keep = false;
                    break;
                }
            }
            keep
        };
        if keep {
            selected.push((d, id));
            if !was_selected && i != p {
                extras.push(id);
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
    let main_loop_len = selected.len();
    for p_item in pruned {
        if selected.len() >= cap {
            break;
        }
        selected.push(p_item);
    }

    let mut mask = [0u64; LIST_MASK_WORDS];
    let mut ids = Vec::with_capacity(selected.len());
    let mut dists = Vec::with_capacity(selected.len());
    for (i, (d, id)) in selected.into_iter().enumerate() {
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
        pair_dist_cache: std::collections::HashMap::new(),
        stats: BuildStats::new(),
        reference_backlinks: false,
        disk_mode: is_concurrent || budget_bytes == 0,
        nrows: 0,
        rng: crate::access_method::hnswsq::build_rng(),
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
    mem_insert(state, heap_tid, &vec);
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
        let hits = search_layer(
            &state.codec,
            state.distance_type,
            &subject,
            &*g,
            vec![cur],
            state.ef_construction,
            l,
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
            &mut state.pair_dist_cache,
            &mut state.stats,
            cands,
            cap_for(l),
        );
        if let Some(t) = t_phase {
            state.stats.select_ns += t.elapsed().as_nanos() as u64;
        }
        g.set_list(id, l, selected.clone(), sel_dists, sel_mask);

        // Backlinks with heuristic re-pruning (RAM: no two-phase needed).
        // Pair distances come from the build-scoped cache: each (n, mm)
        // pair is decoded and evaluated once per build, not once per
        // neighbor-list revision.
        let cap = cap_for(l);
        for &n in &selected {
            if timed {
                state.stats.backlink_lists += 1;
            }
            let t_phase = timed.then(std::time::Instant::now);
            let d_self = cached_pair_dist(
                &state.codec,
                state.dist_fn,
                &*g,
                &mut state.pair_dist_cache,
                &mut state.stats,
                PairCaller::Backlink,
                n,
                id,
            );
            if let Some(t) = t_phase {
                state.stats.backlink_pairs_ns += t.elapsed().as_nanos() as u64;
            }
            let t_phase = timed.then(std::time::Instant::now);
            let n_i = n as usize;
            let existing_ids: Vec<u32> = g.neighbors[n_i][l].clone();
            let existing_dists: Vec<f32> = g.list_dists[n_i][l].clone();
            let existing_mask = g.list_masks[n_i][l];
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
                    &mut state.pair_dist_cache,
                    &mut state.stats,
                    cands,
                    cap,
                )
            } else {
                backlink_prune_mem(
                    &state.codec,
                    state.dist_fn,
                    &*g,
                    &mut state.pair_dist_cache,
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
            let mut cache = std::collections::HashMap::new();
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
                &codec, distance_l2, &g, &mut cache, &mut stats, cands, cap,
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
                &codec, distance_l2, &g, &mut cache, &mut stats, merged, cap,
            );

            let (inc_ids, _, _) = backlink_prune_mem(
                &codec,
                distance_l2,
                &g,
                &mut cache,
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
                pair_dist_cache: std::collections::HashMap::new(),
                stats: BuildStats::default(),
                reference_backlinks: reference,
                disk_mode: false,
                nrows: 0,
                rng: SmallRng::seed_from_u64(seed),
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
                pair_dist_cache: std::collections::HashMap::new(),
                stats: BuildStats::default(),
                reference_backlinks: reference,
                disk_mode: false,
                nrows: 0,
                rng: SmallRng::seed_from_u64(7),
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

    /// Builds the same random graph twice — once with the incremental backlink
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
                pair_dist_cache: std::collections::HashMap::new(),
                stats: BuildStats::default(),
                reference_backlinks: reference,
                disk_mode: false,
                nrows: 0,
                rng: SmallRng::seed_from_u64(7),
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
            pair_dist_cache: std::collections::HashMap::new(),
            stats: BuildStats::default(),
            reference_backlinks: false,
            disk_mode: false,
            nrows: 0,
            rng: SmallRng::seed_from_u64(build_seed),
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
