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
    distance_encoded, greedy_descent, random_level, search_layer, select_neighbors_heuristic,
    GraphAccess, VisitData,
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
    entry: Option<u32>,
    entry_level: usize,
}

impl MemGraph {
    fn new() -> Self {
        Self {
            levels: Vec::new(),
            tids: Vec::new(),
            clamped: Vec::new(),
            vectors: Vec::new(),
            neighbors: Vec::new(),
            entry: None,
            entry_level: 0,
        }
    }

    fn len(&self) -> usize {
        self.tids.len()
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
    /// True after the memory graph was flushed (or for concurrent builds);
    /// remaining rows go through the disk insert path.
    disk_mode: bool,
    nrows: u64,
    rng: SmallRng,
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
    a: u32,
    b: u32,
) -> f32 {
    if a == b {
        return 0.0;
    }
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&d) = cache.get(&key) {
        return d;
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
fn select_neighbors_heuristic_mem(
    codec: &Codec,
    dist_fn: DistanceFn,
    g: &MemGraph,
    cache: &mut std::collections::HashMap<(u32, u32), f32>,
    candidates: Vec<(f32, u32)>,
    cap: usize,
) -> Vec<u32> {
    if candidates.len() <= cap {
        let mut sorted = candidates;
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        sorted.truncate(cap);
        return sorted.into_iter().map(|(_, id)| id).collect();
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
            if cached_pair_dist(codec, dist_fn, g, cache, cand.1, sel.1) < cand.0 {
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
    for p in pruned {
        if selected.len() >= cap {
            break;
        }
        selected.push(p);
    }
    selected.into_iter().map(|(_, id)| id).collect()
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
            rng: SmallRng::from_entropy(),
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
        disk_mode: is_concurrent || budget_bytes == 0,
        nrows: 0,
        rng: SmallRng::from_entropy(),
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
        let (entry_ptr, entry_level, insert_page) = unsafe {
            flush_mem_graph(&index_rel, &state.graph, &state.codec, state.m, state.m0)
        };
        unsafe {
            HnswMetaPage::update(&index_rel, |meta| {
                meta.set_build_result(entry_ptr, entry_level, mem_tuples, insert_page);
            });
        }
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
    let level = random_level(state.ml, state.max_level, &mut state.rng);
    let mut encoded = Vec::with_capacity(state.codec.vector_bytes());
    let clamped = state.codec.encode_into(vector, &mut encoded);

    let g = &mut state.graph;
    let id = g.len() as u32;
    g.levels.push(level);
    g.tids.push(heap_tid);
    g.clamped.push(clamped);
    g.vectors.push(encoded.clone());
    g.neighbors.push(vec![Vec::new(); level as usize + 1]);

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
    for l in (0..=top).rev() {
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
        let cands: Vec<(f32, u32)> = hits
            .iter()
            .filter(|h| h.id != id)
            .map(|h| (h.dist, h.id))
            .collect();
        let selected = select_neighbors_heuristic_mem(
            &state.codec,
            state.dist_fn,
            &*g,
            &mut state.pair_dist_cache,
            cands,
            cap_for(l),
        );
        g.neighbors[id as usize][l] = selected.clone();

        // Backlinks with heuristic re-pruning (RAM: no two-phase needed).
        // Pair distances come from the build-scoped cache: each (n, mm)
        // pair is decoded and evaluated once per build, not once per
        // neighbor-list revision.
        let cap = cap_for(l);
        for &n in &selected {
            let d_self = cached_pair_dist(
                &state.codec,
                state.dist_fn,
                &*g,
                &mut state.pair_dist_cache,
                n,
                id,
            );
            let mut cands: Vec<(f32, u32)> = vec![(d_self, id)];
            for &mm in &g.neighbors[n as usize][l] {
                let d = cached_pair_dist(
                    &state.codec,
                    state.dist_fn,
                    &*g,
                    &mut state.pair_dist_cache,
                    n,
                    mm,
                );
                cands.push((d, mm));
            }
            let new_list = select_neighbors_heuristic_mem(
                &state.codec,
                state.dist_fn,
                &*g,
                &mut state.pair_dist_cache,
                cands,
                cap,
            );
            g.neighbors[n as usize][l] = new_list;
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
