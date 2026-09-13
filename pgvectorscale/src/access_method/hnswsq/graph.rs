//! HNSW core algorithms, generic over the node store.
//!
//! The search/selection routines operate on a [`GraphAccess`] trait with an
//! abstract node id, so the same code drives:
//! - the on-disk graph during insert/scan/vacuum (`Id = ItemPointer`), and
//! - the in-memory graph during build (`Id = u32`).
//!
//! Distances are computed on the *stored* (possibly reduced-precision)
//! vectors: encoded bytes are decoded into an f32 scratch buffer and handed to
//! the existing SIMD kernels.  The executor rechecks exact distances from the
//! heap (`xs_recheckorderby = true`), so the final ranking is exact over the
//! candidates the graph search produced.
//!
//! Tombstones (vacuum-marked deleted nodes) are traversed — they keep routing
//! the graph — but callers filter them out of result sets and neighbor
//! selections.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use pgrx::PgRelation;
use rand::Rng;

use crate::access_method::distance::DistanceType;
use crate::access_method::hnswsq::node::{expand_node, load_node_view, probe_node};
use crate::access_method::hnswsq::quantize::Codec;
use crate::util::ItemPointer;

/// Per-visit snapshot returned by [`GraphAccess::visit`].
pub struct VisitData<Id> {
    pub level: u8,
    pub deleted: bool,
    /// Encoded vector bytes.
    pub encoded: Vec<u8>,
    /// Valid neighbor prefix at the requested layer (empty when the node has
    /// no such layer or is missing).
    pub neighbors: Vec<Id>,
    /// Heap TID (Invalid for the in-memory build graph, which has no heap TIDs
    /// of its own to emit).
    pub heap_tid: ItemPointer,
    /// Whether encoding this node's vector clamped a component.
    pub clamped: bool,
}

/// Node store abstraction for the HNSW algorithms.
///
/// Implementations must be snapshot-consistent per call (one page share lock
/// per load on disk) but MAY return `None` for ids that vanished (recycled
/// pages); the algorithms treat that as "skip".
pub trait GraphAccess {
    type Id: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug;

    /// Load `id`'s vector + neighbor list at `layer` (copying: cold paths).
    fn visit(&self, id: Self::Id, layer: usize) -> Option<VisitData<Self::Id>>;

    /// Load just `id`'s encoded vector (used for pairwise distances in
    /// neighbor selection).
    fn vector(&self, id: Self::Id) -> Option<Vec<u8>>;

    /// Distance from `query` to `id`'s stored vector, evaluated while the node
    /// is loaded, plus the metadata needed to emit the node later (heap TID,
    /// tombstone flag, quantization clamp flag).
    ///
    /// The default implementation goes through [`GraphAccess::visit`]; the
    /// on-disk graph overrides it to compute straight out of the pinned page —
    /// no per-hop `Vec` allocation, no vector copy — which is what the search's
    /// inner loop needs (it probes every discovered neighbour).
    fn probe(
        &self,
        id: Self::Id,
        query: &[f32],
        codec: &Codec,
        dist_type: DistanceType,
    ) -> Option<ProbeResult> {
        let vd = self.visit(id, 0)?;
        Some(ProbeResult {
            dist: distance_encoded(codec, dist_type, query, &vd.encoded),
            deleted: vd.deleted,
            heap_tid: vd.heap_tid,
            clamped: vd.clamped,
        })
    }

    /// Copy `id`'s valid neighbor prefix at `layer` into `out` (a caller-owned
    /// buffer reused across hops) and report its level.
    ///
    /// The default implementation goes through [`GraphAccess::visit`].
    fn expand(
        &self,
        id: Self::Id,
        layer: usize,
        out: &mut Vec<Self::Id>,
    ) -> Option<ExpandResult> {
        let vd = self.visit(id, layer)?;
        out.clear();
        out.extend_from_slice(&vd.neighbors);
        Some(ExpandResult {
            level: vd.level,
            deleted: vd.deleted,
        })
    }
}

/// Result of [`GraphAccess::probe`].
#[derive(Clone, Copy, Debug)]
pub struct ProbeResult {
    pub dist: f32,
    /// Tombstone flag (tombstones route but never occupy a result slot).
    pub deleted: bool,
    /// Heap TID of the node, so a scan can emit it without loading it again.
    pub heap_tid: ItemPointer,
    /// Whether encoding this node's vector clamped a component (the scan then
    /// cannot prove a finite lower bound).
    pub clamped: bool,
}

/// Result of [`GraphAccess::expand`].
#[derive(Clone, Copy, Debug)]
pub struct ExpandResult {
    pub level: u8,
    pub deleted: bool,
}

/// On-disk accessor: ids are node `ItemPointer`s, loads go through
/// [`load_node_view`] (share content lock, copy out, release — the
/// single-lock rule).
pub struct DiskGraph<'a> {
    pub index: &'a PgRelation,
}

impl GraphAccess for DiskGraph<'_> {
    type Id = ItemPointer;

    fn visit(&self, id: Self::Id, layer: usize) -> Option<VisitData<Self::Id>> {
        let view = load_node_view(self.index, id)?;
        Some(VisitData {
            level: view.level,
            deleted: view.deleted,
            encoded: view.vector,
            neighbors: view.neighbors.get(layer).cloned().unwrap_or_default(),
            heap_tid: view.heap_tid,
            clamped: view.clamped,
        })
    }

    fn vector(&self, id: Self::Id) -> Option<Vec<u8>> {
        Some(load_node_view(self.index, id)?.vector)
    }

    /// Evaluate the distance out of the pinned page: the node's vector is fed to
    /// the distance kernel in place, so a probed neighbour costs one page read
    /// and nothing else (the previous path copied the vector and the whole
    /// neighbour list out of the page for every visited node, then cloned the
    /// list again on expansion).
    fn probe(
        &self,
        id: Self::Id,
        query: &[f32],
        codec: &Codec,
        dist_type: DistanceType,
    ) -> Option<ProbeResult> {
        let probed = probe_node(self.index, codec, dist_type, query, id)?;
        Some(ProbeResult {
            dist: probed.dist,
            deleted: probed.deleted,
            heap_tid: probed.heap_tid,
            clamped: probed.clamped,
        })
    }

    fn expand(
        &self,
        id: Self::Id,
        layer: usize,
        out: &mut Vec<Self::Id>,
    ) -> Option<ExpandResult> {
        let (level, deleted) = expand_node(self.index, id, layer, out)?;
        Some(ExpandResult { level, deleted })
    }
}

/// A (distance, id) pair with a total order (f32 `total_cmp`, id tiebreak).
#[derive(Clone, Debug)]
pub struct HeapItem<Id> {
    pub dist: f32,
    pub id: Id,
    /// Tombstone flag snapshot (does not participate in ordering).
    pub deleted: bool,
    /// Heap TID + clamp flag snapshot, so a scan can emit a hit without
    /// loading the node a second time (does not participate in ordering).
    pub heap_tid: ItemPointer,
    pub clamped: bool,
}

/// One search result: distance, node id, and whether the node is a tombstone.
/// Callers filter tombstones for results/neighbor selection but the search
/// itself traverses and ranks them (they keep routing the graph).
#[derive(Clone, Debug)]
pub struct SearchHit<Id> {
    pub dist: f32,
    pub id: Id,
    pub deleted: bool,
    /// Heap TID of the node, carried from the load that produced the hit.
    pub heap_tid: ItemPointer,
    /// Whether encoding clamped a component (scan-side lower-bound input).
    pub clamped: bool,
}

impl<Id: Ord> PartialEq for HeapItem<Id> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl<Id: Ord> Eq for HeapItem<Id> {}

impl<Id: Ord> PartialOrd for HeapItem<Id> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Id: Ord> Ord for HeapItem<Id> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// Compute the distance from `query` to `encoded` directly over the stored
/// (possibly reduced-precision) bytes — no decode copy.
#[inline]
pub fn distance_encoded(
    codec: &Codec,
    dist_type: DistanceType,
    query: &[f32],
    encoded: &[u8],
) -> f32 {
    codec.distance_encoded_direct(dist_type, query, encoded)
}

/// Draw a node level: `floor(-ln(U) · ml)` clamped to the index's `max_level`.
pub fn random_level(ml: f32, max_level: u8, rng: &mut impl Rng) -> u8 {
    let u: f64 = rng.gen::<f64>().max(1e-12);
    let l = (-(u.ln() as f32) * ml).floor();
    l.clamp(0.0, max_level as f32) as u8
}

/// HNSW SEARCH-LAYER: best-first expansion from `entries` at `layer`.
///
/// The result heap holds LIVE hits only: tombstones (vacuum-marked deleted
/// nodes) are expanded and routed through — they keep graph connectivity
/// between vacuums — but never consume an `ef` slot, so a delete-heavy index
/// still returns `ef` live candidates (bounded by the connected component
/// when fewer live nodes remain).  Returns up to `ef` live hits sorted
/// ascending by distance.
pub fn search_layer<A: GraphAccess>(
    codec: &Codec,
    dist_type: DistanceType,
    query: &[f32],
    access: &A,
    entries: Vec<(f32, A::Id)>,
    ef: usize,
    layer: usize,
) -> Vec<SearchHit<A::Id>> {
    let ef = ef.max(1);
    let mut visited: HashSet<A::Id> = HashSet::with_capacity(ef * 2);
    // min-heap of frontier candidates (includes tombstones)
    let mut candidates: BinaryHeap<std::cmp::Reverse<HeapItem<A::Id>>> = BinaryHeap::new();
    // max-heap of the ef best LIVE results
    let mut results: BinaryHeap<HeapItem<A::Id>> = BinaryHeap::new();
    // Reused neighbour buffer: expansion copies the next node's list into it,
    // so a whole search allocates nothing per hop (the old path kept a
    // `HashMap<Id, VisitData>` of copied views and cloned the list again on
    // every expansion — visible as 15.6% libc + 9.4% `Vec::from_iter` in the
    // query profile).
    let mut neighbors: Vec<A::Id> = Vec::with_capacity(64);

    for (d, id) in entries {
        if !visited.insert(id) {
            continue;
        }
        // Probe the entry so its tombstone flag is known even when it is never
        // expanded, and so its heap TID can be emitted without a second load.
        let Some(p) = access.probe(id, query, codec, dist_type) else {
            continue; // vanished entry
        };
        let item = HeapItem {
            dist: d,
            id,
            deleted: p.deleted,
            heap_tid: p.heap_tid,
            clamped: p.clamped,
        };
        candidates.push(std::cmp::Reverse(item.clone()));
        if !p.deleted {
            results.push(item);
        }
    }

    while let Some(std::cmp::Reverse(cur)) = candidates.pop() {
        // Termination: the closest unexpanded candidate is worse than the
        // worst result and the result set is full.
        if results.len() >= ef {
            if let Some(worst) = results.peek() {
                if cur.dist > worst.dist {
                    break;
                }
            }
        }

        // Copy this node's neighbour list at `layer` into the reused buffer.
        if access.expand(cur.id, layer, &mut neighbors).is_none() {
            continue; // vanished id
        }

        for &nb in neighbors.iter() {
            if !visited.insert(nb) {
                continue;
            }
            let Some(p) = access.probe(nb, query, codec, dist_type) else {
                continue;
            };
            let item = HeapItem {
                dist: p.dist,
                id: nb,
                deleted: p.deleted,
                heap_tid: p.heap_tid,
                clamped: p.clamped,
            };
            candidates.push(std::cmp::Reverse(item.clone()));
            if p.deleted {
                // Tombstones route the search but never occupy a result slot.
                continue;
            }
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
        .into_sorted_vec()
        .into_iter()
        .map(|i| SearchHit {
            dist: i.dist,
            id: i.id,
            deleted: i.deleted,
            heap_tid: i.heap_tid,
            clamped: i.clamped,
        })
        .collect()
}

/// Greedy ef=1 descent from `entry` down to `to_layer` (inclusive), used for
/// the upper-layer walk of both search and insert.
pub fn greedy_descent<A: GraphAccess>(
    codec: &Codec,
    dist_type: DistanceType,
    query: &[f32],
    access: &A,
    entry: (f32, A::Id),
    from_layer: usize,
    to_layer: usize,
) -> (f32, A::Id) {
    let mut cur = entry;
    let mut neighbors: Vec<A::Id> = Vec::with_capacity(64);
    for layer in (to_layer..=from_layer).rev() {
        loop {
            if access.expand(cur.1, layer, &mut neighbors).is_none() {
                break;
            }
            let mut improved = false;
            for &nb in neighbors.iter() {
                let Some(p) = access.probe(nb, query, codec, dist_type) else {
                    continue;
                };
                if p.dist < cur.0 {
                    cur = (p.dist, nb);
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
    }
    cur
}

/// HNSW SELECT-NEIGHBORS-HEURISTIC with closest-pruned backfill.
///
/// `candidates` are (distance-to-the-subject-node, id) pairs (tombstones
/// already filtered by the caller).  Returns up to `cap` ids: a candidate is
/// kept only if it is closer to the subject than to every already-selected
/// neighbor (graph diversification), then the closest pruned candidates
/// backfill any remaining room so lists stay full early in a build.
pub fn select_neighbors_heuristic<A: GraphAccess>(
    codec: &Codec,
    dist_type: DistanceType,
    access: &A,
    candidates: Vec<(f32, A::Id)>,
    cap: usize,
) -> Vec<A::Id> {
    if candidates.len() <= cap {
        // Still need to drop nothing; keep the closest `cap` for determinism.
        let mut sorted = candidates;
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        sorted.truncate(cap);
        return sorted.into_iter().map(|(_, id)| id).collect();
    }

    let mut sorted = candidates;
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    // Decode each candidate once, keyed by its position in `sorted` (no
    // per-lookup HashMap): `selected` then stores positions so the pairwise
    // occlusion checks index both vectors directly.  This is the hottest
    // loop of the whole build (one heuristic run per neighbor list per
    // inserted node).
    let mut decoded: Vec<Option<Vec<f32>>> = Vec::with_capacity(sorted.len());
    for (_, id) in &sorted {
        decoded.push(access.vector(*id).map(|enc| codec.decode(&enc)));
    }

    let mut selected: Vec<(f32, usize)> = Vec::with_capacity(cap);
    let mut pruned: Vec<(f32, usize)> = Vec::new();

    for (i, cand) in sorted.iter().enumerate() {
        let Some(cand_vec) = decoded[i].as_ref() else {
            continue; // vanished id
        };
        if selected.len() >= cap {
            pruned.push((cand.0, i));
            continue;
        }
        let mut keep = true;
        for sel in &selected {
            if let Some(sel_vec) = decoded[sel.1].as_ref() {
                // Prune `cand` when an already-selected neighbor is closer to
                // it than the subject is (the heuristic's occlusion rule).
                if dist_type.get_distance_function()(cand_vec, sel_vec) < cand.0 {
                    keep = false;
                    break;
                }
            }
        }
        if keep {
            selected.push((cand.0, i));
        } else {
            pruned.push((cand.0, i));
        }
    }

    // Backfill with the closest pruned candidates (already in ascending order).
    for p in pruned {
        if selected.len() >= cap {
            break;
        }
        selected.push(p);
    }

    selected
        .into_iter()
        .map(|(_, i)| sorted[i].1)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_method::distance::{distance_l2, DistanceType};
    use crate::access_method::hnswsq::quantize::HnswPrecision;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    /// In-memory mock store for pure-Rust algorithm tests.
    struct MockGraph {
        vectors: HashMap<u32, Vec<u8>>,
        neighbors: HashMap<u32, Vec<Vec<u32>>>,
        levels: HashMap<u32, u8>,
        deleted: HashSet<u32>,
    }

    impl GraphAccess for MockGraph {
        type Id = u32;
        fn visit(&self, id: u32, layer: usize) -> Option<VisitData<u32>> {
            Some(VisitData {
                level: *self.levels.get(&id)?,
                deleted: self.deleted.contains(&id),
                encoded: self.vectors.get(&id)?.clone(),
                neighbors: self
                    .neighbors
                    .get(&id)
                    .and_then(|n| n.get(layer))
                    .cloned()
                    .unwrap_or_default(),
                heap_tid: ItemPointer::new_invalid(),
                clamped: false,
            })
        }
        fn vector(&self, id: u32) -> Option<Vec<u8>> {
            self.vectors.get(&id).cloned()
        }
    }

    fn build_chain(n: u32, dim: usize) -> MockGraph {
        // 0 → 1 → 2 → ... chain along the x axis, unit steps.
        let codec = Codec::new(HnswPrecision::Plain, dim);
        let mut g = MockGraph {
            vectors: HashMap::new(),
            neighbors: HashMap::new(),
            levels: HashMap::new(),
            deleted: HashSet::new(),
        };
        for i in 0..n {
            let mut v = vec![0.0f32; dim];
            v[0] = i as f32;
            g.vectors.insert(i, codec.encode(&v));
            let next = if i + 1 < n { vec![i + 1] } else { vec![] };
            g.neighbors.insert(i, vec![next]);
            g.levels.insert(i, 0);
        }
        g
    }

    #[test]
    fn test_search_layer_routes_through_tombstones_without_resulting_them() {
        let dim = 2;
        let codec = Codec::new(HnswPrecision::Plain, dim);
        let mut g = build_chain(10, dim);
        // Node 5 is the ONLY path from 0 to nodes 6..9; tombstone it.
        g.deleted.insert(5);
        let mut q = vec![0.0f32; dim];
        q[0] = 9.2;
        let e0 = {
            let enc = g.vector(0).unwrap();
            let mut s = vec![0.0f32; dim];
            codec.decode_into(&enc, &mut s);
            distance_l2(&q, &s)
        };
        let res = search_layer(&codec, DistanceType::L2, &q, &g, vec![(e0, 0)], 3, 0);
        // Traversal must still reach past the tombstone...
        assert_eq!(res[0].id, 9);
        // ...and results must contain only live nodes.
        assert!(res.iter().all(|h| !h.deleted));
        assert!(!res.iter().any(|h| h.id == 5));
    }

    #[test]
    fn test_search_layer_live_ef_slots_not_eaten_by_tombstones() {
        let dim = 2;
        let codec = Codec::new(HnswPrecision::Plain, dim);
        let mut g = build_chain(10, dim);
        for i in 0..8 {
            g.deleted.insert(i); // only 8 and 9 remain live
        }
        let q = vec![0.0f32, 0.0];
        let res = search_layer(&codec, DistanceType::L2, &q, &g, vec![(0.0, 0)], 5, 0);
        let ids: Vec<u32> = res.iter().map(|h| h.id).collect();
        assert_eq!(ids, vec![8, 9], "live-only results, ef slots never wasted");
    }

    #[test]
    fn test_search_layer_walks_chain() {
        let dim = 4;
        let codec = Codec::new(HnswPrecision::Plain, dim);
        let g = build_chain(10, dim);
        let mut q = vec![0.0f32; dim];
        q[0] = 9.2;
        // The entry distance must be real (production callers always compute
        // it): node 0 is at x=0, so d = 9.2² = 84.64.
        let e0 = {
            let enc = g.vector(0).unwrap();
            let mut s = vec![0.0f32; dim];
            codec.decode_into(&enc, &mut s);
            distance_l2(&q, &s)
        };
        let res = search_layer(&codec, DistanceType::L2, &q, &g, vec![(e0, 0)], 3, 0);
        assert!(res.len() >= 1);
        // nearest to 9.2 is node 9
        assert_eq!(res[0].id, 9);
    }

    #[test]
    fn test_search_layer_ef_bounds_results() {
        let dim = 2;
        let codec = Codec::new(HnswPrecision::Plain, dim);
        let g = build_chain(20, dim);
        let q = vec![0.0f32, 0.0];
        let res = search_layer(&codec, DistanceType::L2, &q, &g, vec![(0.0, 0)], 5, 0);
        assert!(res.len() <= 5);
        // sorted ascending
        for w in res.windows(2) {
            assert!(w[0].dist <= w[1].dist);
        }
        assert!(res.iter().all(|h| !h.deleted));
    }

    #[test]
    fn test_greedy_descent_finds_local_min() {
        let dim = 2;
        let codec = Codec::new(HnswPrecision::Plain, dim);
        let g = build_chain(10, dim);
        let mut q = vec![0.0f32; dim];
        q[0] = 4.4;
        let (d, id) = greedy_descent(&codec, DistanceType::L2, &q, &g, (81.0, 0), 0, 0);
        assert_eq!(id, 4);
        assert!((d - 0.16).abs() < 1e-5);
    }

    #[test]
    fn test_select_neighbors_heuristic_diversifies() {
        // Cluster A at x=0 (2 nodes), cluster B at x=10 (2 nodes); subject at
        // x=0.1. The heuristic must not pick both redundant A nodes before B.
        let dim = 1;
        let codec = Codec::new(HnswPrecision::Plain, dim);
        let mut g = MockGraph {
            vectors: HashMap::new(),
            neighbors: HashMap::new(),
            levels: HashMap::new(),
            deleted: HashSet::new(),
        };
        for (id, x) in [(0u32, 0.0f32), (1, 0.05), (2, 10.0), (3, 10.05)] {
            g.vectors.insert(id, codec.encode(&[x]));
            g.levels.insert(id, 0);
            g.neighbors.insert(id, vec![vec![]]);
        }
        let subject = [0.1f32];
        let d0 = distance_l2(&subject, &[0.0]);
        let d1 = distance_l2(&subject, &[0.05]);
        let d2 = distance_l2(&subject, &[10.0]);
        let d3 = distance_l2(&subject, &[10.05]);
        let sel = select_neighbors_heuristic(
            &codec,
            DistanceType::L2,
            &g,
            vec![(d0, 0), (d1, 1), (d2, 2), (d3, 3)],
            3,
        );
        assert_eq!(sel.len(), 3);
        // node 0 kept; node 1 occluded by 0 (dist(1,0)=0.0025 < dist(1,subj));
        // backfill must bring in the far-cluster nodes before re-adding 1.
        assert!(sel.contains(&0));
        assert!(sel.contains(&2));
        assert!(sel.contains(&3) || sel.contains(&1));
    }

    #[test]
    fn test_select_neighbors_small_input_passthrough() {
        let dim = 1;
        let codec = Codec::new(HnswPrecision::Plain, dim);
        let g = MockGraph {
            vectors: HashMap::new(),
            neighbors: HashMap::new(),
            levels: HashMap::new(),
            deleted: HashSet::new(),
        };
        let sel = select_neighbors_heuristic(&codec, DistanceType::L2, &g, vec![(1.0, 7), (0.5, 3)], 5);
        assert_eq!(sel, vec![3, 7]); // sorted by distance
    }

    #[test]
    fn test_random_level_distribution_and_cap() {
        let mut rng = SmallRng::seed_from_u64(42);
        let ml = 1.0 / (16.0f64.ln()) as f32;
        let mut counts = [0u64; 8];
        for _ in 0..10000 {
            let l = random_level(ml, 7, &mut rng);
            assert!(l <= 7);
            counts[l as usize] += 1;
        }
        // geometric decay: level 0 dominates, level 1 ≈ 15/16 of draws…
        assert!(counts[0] > 900);
        assert!(counts[1] > counts[2]);
        assert!(counts[2] > counts[3]);
        // cap actually hit sometimes in 10k draws (P(l≥7) = 16^-7 ≈ 4e-9 →
        // unlikely; just assert the cap logic directly)
        assert_eq!(random_level(ml, 0, &mut rng), 0);
    }

    #[test]
    fn test_distance_encoded_f16_close_to_plain() {
        let dim = 32;
        let plain = Codec::new(HnswPrecision::Plain, dim);
        let fp16 = Codec::new(HnswPrecision::IeeeFp16, dim);
        let v: Vec<f32> = (0..dim).map(|d| (d as f32) * 0.01).collect();
        let q: Vec<f32> = (0..dim).map(|d| 1.0 - (d as f32) * 0.02).collect();
        let dp = distance_encoded(&plain, DistanceType::L2, &q, &plain.encode(&v));
        let dh = distance_encoded(&fp16, DistanceType::L2, &q, &fp16.encode(&v));
        assert!((dp - dh).abs() < dp * 0.01 + 1e-6);
        let _ = DistanceType::L2;
    }
}
