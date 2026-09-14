//! Flat engine: the build algorithms for the new storage (`flat_graph.rs`).
//!
//! Written alongside the legacy in-memory build rather than replacing it: the
//! legacy `MemGraph` + `mem_plan`/`mem_apply` pair keeps being the default, and
//! this module grows until it passes the gates in
//! `.design/hnswsq_parallel_build_todos.md`, at which point the legacy engine is
//! deleted (or kept for its exact backlink policy, depending on the M7
//! measurement).
//!
//! This step provides the **search** half:
//!
//! * [`search_layer_flat`] — the same beam search as the legacy
//!   `search_layer_mem`, over flat slabs instead of `Vec<Vec<Vec<u32>>>`, and
//!   reusing the same [`SearchScratch`] (epoch marks + heaps, so a search
//!   allocates nothing per visited node and resets in O(visited)).
//!
//! Differences from the legacy search that matter:
//!
//! * every node is live — the build graph has no tombstones and no vanished
//!   ids, so every visited node occupies a result slot and `deleted` is always
//!   false.  Ids are still bounds-checked against the graph on every access
//!   (never against the scratch's grown mark arrays), because the parallel
//!   engine will hand out ids from a shared counter before their node data is
//!   written, and an observer must skip such an id rather than index it;
//! * hits carry the node's heap TID and clamp flag (read from the flat arrays),
//!   which is what lets the *scan* emit a candidate without loading it again —
//!   the same contract the disk path got in P2/P3;
//! * reads go through `FlatGraph`'s accessors, so the same code runs unchanged
//!   once the arrays live in shared memory behind per-node locks.

use crate::access_method::distance::DistanceType;
use crate::access_method::hnswsq::build::SearchScratch;
use crate::access_method::hnswsq::flat_graph::FlatGraph;
use crate::access_method::hnswsq::graph::{distance_encoded, HeapItem, SearchHit};
use crate::access_method::hnswsq::quantize::Codec;

/// Beam search at `layer`, starting from `entries` (distance, id).
///
/// Returns up to `ef` hits sorted ascending by distance — the same contract and
/// the same termination rule as the legacy memory search: stop as soon as the
/// closest unexpanded candidate is worse than the worst kept result and the
/// result set is full.
pub fn search_layer_flat(
    codec: &Codec,
    dist_type: DistanceType,
    query: &[f32],
    g: &FlatGraph,
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
        // Bound by the *graph*, not by the scratch: the scratch only ever grows,
        // so a stale entry id (or, later, an id a parallel worker has claimed
        // from the shared counter but not yet initialised) would otherwise index
        // past the graph.  Same guard as in the neighbour loop below.
        if i >= g.watermark() || i >= visited_epoch.len() || visited_epoch[i] == epoch {
            continue;
        }
        visited_epoch[i] = epoch;
        let item = HeapItem {
            dist: d,
            id,
            deleted: false,
            heap_tid: g.tid(id),
            clamped: g.clamped(id),
        };
        candidates.push(std::cmp::Reverse(item.clone()));
        results.push(item);
    }

    while let Some(std::cmp::Reverse(cur)) = candidates.pop() {
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

        // Borrowed neighbour prefix: no copy, no allocation per expansion.
        let list = g.neighbors(cur.id, layer);
        for k in 0..list.len() {
            let nb = list[k];
            let ni = nb as usize;
            if ni >= g.watermark() || ni >= visited_epoch.len() || visited_epoch[ni] == epoch {
                continue;
            }
            visited_epoch[ni] = epoch;
            let d = distance_encoded(codec, dist_type, query, g.vector(nb));
            let item = HeapItem {
                dist: d,
                id: nb,
                deleted: false,
                heap_tid: g.tid(nb),
                clamped: g.clamped(nb),
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

    // `results` is a `&mut` from the scratch destructuring: clone the (≤ ef
    // element) heap to sort it, exactly as the legacy memory search does.
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

// ---------------------------------------------------------------------------
// Selection and application (M1, second half)
// ---------------------------------------------------------------------------

/// Decode-once pairwise-distance buffer for the flat engine (the same idea as
/// the legacy build's `DistBuf`): candidates are decoded into one flat `f32`
/// buffer, so the occlusion checks index slices instead of hashing a map.  The
/// allocation is reused across calls.
pub(crate) struct FlatPairBuf {
    dim: usize,
    data: Vec<f32>,
}

impl FlatPairBuf {
    pub(crate) fn new(dim: usize) -> Self {
        Self {
            dim,
            data: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.data.clear();
    }

    /// Decode `id`'s stored vector into the next slot; returns the slot index.
    fn push(&mut self, codec: &Codec, g: &FlatGraph, id: u32) -> usize {
        let start = self.data.len();
        self.data.resize(start + self.dim, 0.0);
        codec.decode_into(g.vector(id), &mut self.data[start..]);
        self.data.len() / self.dim - 1
    }

    #[inline]
    fn slice(&self, slot: usize) -> &[f32] {
        &self.data[slot * self.dim..(slot + 1) * self.dim]
    }
}

/// Occlusion rule: walking candidates in ascending distance order, keep a
/// candidate unless an already-kept one is closer to it than the subject is.
/// Returns indices into the ascending arrays, capped at `cap`.
fn occlusion_accepted(
    dist_fn: crate::access_method::distance::DistanceFn,
    buf: &FlatPairBuf,
    dists: &[f32],
    slots: &[usize],
    cap: usize,
) -> Vec<usize> {
    let mut accepted: Vec<usize> = Vec::with_capacity(cap.min(dists.len()));
    for i in 0..dists.len() {
        if accepted.len() >= cap {
            break;
        }
        let mut keep = true;
        for &sel in &accepted {
            if dist_fn(buf.slice(slots[i]), buf.slice(slots[sel])) < dists[i] {
                keep = false;
                break;
            }
        }
        if keep {
            accepted.push(i);
        }
    }
    accepted
}

/// Neighbour selection for a node's **own** list: the occlusion heuristic over
/// the beam-search candidates, in ascending distance order, up to `cap`.
///
/// No closest-pruned backfill here (pgvector's forward list is the heuristic's
/// output alone), so a list can be shorter than `cap` while the graph is young.
pub fn select_neighbors_flat(
    dist_fn: crate::access_method::distance::DistanceFn,
    codec: &Codec,
    g: &FlatGraph,
    buf: &mut FlatPairBuf,
    mut candidates: Vec<(f32, u32)>,
    cap: usize,
) -> Vec<u32> {
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    buf.clear();
    for &(_, id) in candidates.iter() {
        buf.push(codec, g, id);
    }
    let dists: Vec<f32> = candidates.iter().map(|c| c.0).collect();
    let slots: Vec<usize> = (0..candidates.len()).collect();
    occlusion_accepted(dist_fn, buf, &dists, &slots, cap)
        .into_iter()
        .map(|i| candidates[i].1)
        .collect()
}

/// Selected neighbours for one layer, produced by [`plan_flat`] and consumed by
/// [`apply_flat`].  Ids only — the flat graph stores no per-entry distances, so
/// the backlink step measures whatever it needs on demand.
pub struct LayerPlanFlat {
    pub layer: usize,
    pub ids: Vec<u32>,
}

/// Greedy ef=1 descent over the flat graph, used for the upper-layer walk.
pub fn greedy_descent_flat(
    codec: &Codec,
    dist_type: DistanceType,
    query: &[f32],
    g: &FlatGraph,
    entry: (f32, u32),
    from_layer: usize,
    to_layer: usize,
) -> (f32, u32) {
    let mut cur = entry;
    for layer in (to_layer..=from_layer).rev() {
        loop {
            let mut improved = false;
            let list = g.neighbors(cur.1, layer);
            for k in 0..list.len() {
                let nb = list[k];
                if nb as usize >= g.watermark() {
                    continue;
                }
                let d = distance_encoded(codec, dist_type, query, g.vector(nb));
                if d < cur.0 {
                    cur = (d, nb);
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

/// Read-only planning half of one insert: per layer from the node's top layer
/// down to 0, run the beam search and select the node's own neighbours.
#[allow(clippy::too_many_arguments)]
pub fn plan_flat(
    codec: &Codec,
    dist_type: DistanceType,
    dist_fn: crate::access_method::distance::DistanceFn,
    g: &FlatGraph,
    scratch: &mut SearchScratch,
    buf: &mut FlatPairBuf,
    new_id: u32,
    level: u8,
    subject: &[f32],
    m: usize,
    m0: usize,
    ef_construction: usize,
) -> Vec<LayerPlanFlat> {
    let mut plan: Vec<LayerPlanFlat> = Vec::new();
    let Some(ep) = g.entry() else {
        return plan; // empty graph: this node becomes the entry, with no lists yet
    };
    let entry_level = g.entry_level();
    let top = (level as usize).min(entry_level);

    let mut cur = (
        distance_encoded(codec, dist_type, subject, g.vector(ep)),
        ep,
    );
    if entry_level > top {
        cur = greedy_descent_flat(codec, dist_type, subject, g, cur, entry_level, top + 1);
    }

    for layer in (0..=top).rev() {
        let hits = search_layer_flat(
            codec,
            dist_type,
            subject,
            g,
            &[cur],
            ef_construction,
            layer,
            scratch,
        );
        if let Some(best) = hits.first() {
            cur = (best.dist, best.id);
        }
        let candidates: Vec<(f32, u32)> = hits
            .iter()
            .filter(|h| h.id != new_id)
            .map(|h| (h.dist, h.id))
            .collect();
        let cap = if layer == 0 { m0 } else { m };
        plan.push(LayerPlanFlat {
            layer,
            ids: select_neighbors_flat(dist_fn, codec, g, buf, candidates, cap),
        });
    }
    plan
}

/// One backlink: make `target` link back to `new_id` at `layer`.
///
/// The decided policy (pgvector's), expressed without any per-list metadata:
///
/// * **append while there is room** — no distance work at all beyond `d(target,
///   new)`, which is measured once;
/// * **on overflow, measure and decide**: the merged set `list ∪ {new}` is
///   ordered by distances measured on demand and the occlusion heuristic decides
///   who stays.  The list keeps its capacity (`len == cap`) by filling any room
///   the heuristic leaves with the *existing* members it pruned; the newcomer is
///   never re-admitted by that fill, so an evicted entry cannot come back through
///   this path.
///
/// Both branches are pure functions of the current lists and the stored vectors,
/// which is what makes the update deterministic and lock-friendly: no version,
/// no mask, nothing to keep consistent across revisions.
fn backlink_flat(
    codec: &Codec,
    dist_fn: crate::access_method::distance::DistanceFn,
    g: &mut FlatGraph,
    buf: &mut FlatPairBuf,
    target: u32,
    new_id: u32,
    layer: usize,
    cap: usize,
) {
    if target == new_id {
        return; // never self-link
    }
    let existing: Vec<u32> = g.neighbors(target, layer).to_vec();
    if existing.contains(&new_id) {
        return; // already linked
    }

    buf.clear();
    let t_slot = buf.push(codec, g, target);
    let n_slot = buf.push(codec, g, new_id);
    let d_self = dist_fn(buf.slice(t_slot), buf.slice(n_slot));

    if existing.len() < cap {
        let mut list = existing;
        list.push(new_id);
        g.set_list(target, layer, &list);
        return;
    }

    // Saturated: order the merged set by distance to the target, measuring each
    // member on demand (slot 0 is the target, kept for the comparisons).
    let mut entries: Vec<(f32, u32, usize)> = Vec::with_capacity(existing.len() + 1);
    entries.push((d_self, new_id, n_slot));
    for &member in &existing {
        let slot = buf.push(codec, g, member);
        entries.push((dist_fn(buf.slice(t_slot), buf.slice(slot)), member, slot));
    }
    entries.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let dists: Vec<f32> = entries.iter().map(|e| e.0).collect();
    let slots: Vec<usize> = entries.iter().map(|e| e.2).collect();
    let accepted = occlusion_accepted(dist_fn, buf, &dists, &slots, cap);

    let mut is_accepted = vec![false; entries.len()];
    for &i in &accepted {
        is_accepted[i] = true;
    }
    let mut list: Vec<u32> = accepted.iter().map(|&i| entries[i].1).collect();
    for (i, entry) in entries.iter().enumerate() {
        if list.len() >= cap {
            break;
        }
        if is_accepted[i] || entry.1 == new_id {
            continue; // only *existing* members backfill, never the newcomer
        }
        list.push(entry.1);
    }
    g.set_list(target, layer, &list);
}

/// Mutating half of one insert: publish the node's own lists, then apply the
/// backlinks one target at a time (append/shrink), and finally promote the entry
/// point — last, so a searcher never lands on an entry without its own list.
pub fn apply_flat(
    codec: &Codec,
    dist_fn: crate::access_method::distance::DistanceFn,
    g: &mut FlatGraph,
    buf: &mut FlatPairBuf,
    new_id: u32,
    level: u8,
    plan: Vec<LayerPlanFlat>,
    m: usize,
    m0: usize,
) {
    if g.entry().is_none() {
        g.promote_entry(new_id);
        return;
    }
    for LayerPlanFlat { layer, ids } in plan {
        g.set_list(new_id, layer, &ids);
        let cap = if layer == 0 { m0 } else { m };
        for &neighbour in &ids {
            backlink_flat(codec, dist_fn, g, buf, neighbour, new_id, layer, cap);
        }
    }
    g.promote_entry(new_id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_method::distance::distance_l2;
    use crate::access_method::hnswsq::quantize::HnswPrecision;
    use crate::util::ItemPointer;

    /// Line graph: node i sits at `(i as f32, 0)`, linked to its neighbours, so
    /// the true nearest neighbours are known by construction.
    fn line_graph(n: u32, cap: usize) -> (FlatGraph, Codec, Vec<Vec<f32>>) {
        let dim = 2;
        let codec = Codec::new(HnswPrecision::Plain, dim);
        let mut g = FlatGraph::new(codec.vector_bytes(), cap);
        let vecs: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32, 0.0]).collect();
        for (i, v) in vecs.iter().enumerate() {
            g.push_node(0, ItemPointer::new(i as u32 + 1, 1), false, &codec.encode(v));
        }
        for i in 0..n {
            let mut list = Vec::new();
            if i > 0 {
                list.push(i - 1);
            }
            if i + 1 < n {
                list.push(i + 1);
            }
            g.set_list(i, 0, &list);
        }
        (g, codec, vecs)
    }

    #[test]
    fn finds_the_true_nearest_on_a_line_graph() {
        let n = 32u32;
        let (g, codec, vecs) = line_graph(n, 4);
        let mut scratch = SearchScratch::new();
        let query = vec![7.4f32, 0.0];
        let hits = search_layer_flat(
            &codec,
            DistanceType::L2,
            &query,
            &g,
            &[(distance_l2(&query, &vecs[0]), 0)],
            8,
            0,
            &mut scratch,
        );

        let mut brute: Vec<(f32, u32)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (distance_l2(&query, v), i as u32))
            .collect();
        brute.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let want: Vec<u32> = brute.iter().take(8).map(|(_, i)| *i).collect();
        let got: Vec<u32> = hits.iter().map(|h| h.id).collect();
        assert_eq!(got, want, "beam search must match brute force here");

        // ascending and complete metadata
        for w in hits.windows(2) {
            assert!(w[0].dist <= w[1].dist);
        }
        for h in &hits {
            assert!(!h.deleted);
            assert!(h.heap_tid.is_valid());
            assert!(!h.clamped);
        }
    }

    #[test]
    fn ef_bounds_the_result_set_and_is_deterministic() {
        let (g, codec, vecs) = line_graph(64, 4);
        let mut scratch = SearchScratch::new();
        let query = vec![31.5f32, 0.0];
        let run = |scratch: &mut SearchScratch| -> Vec<u32> {
            search_layer_flat(
                &codec,
                DistanceType::L2,
                &query,
                &g,
                &[(distance_l2(&query, &vecs[0]), 0)],
                5,
                0,
                scratch,
            )
            .into_iter()
            .map(|h| h.id)
            .collect()
        };
        let first = run(&mut scratch);
        assert_eq!(first.len(), 5, "ef caps the returned set");
        assert_eq!(first, run(&mut scratch), "same inputs, same output");
        assert!(first.contains(&31) || first.contains(&32), "nearest kept");
    }

    #[test]
    fn empty_entries_and_empty_graph_are_safe() {
        let (g, codec, _) = line_graph(4, 4);
        let mut scratch = SearchScratch::new();
        let hits = search_layer_flat(
            &codec,
            DistanceType::L2,
            &[0.0, 0.0],
            &g,
            &[],
            4,
            0,
            &mut scratch,
        );
        assert!(hits.is_empty());

        let empty = FlatGraph::new(codec.vector_bytes(), 4);
        let hits = search_layer_flat(
            &codec,
            DistanceType::L2,
            &[0.0, 0.0],
            &empty,
            &[(0.0, 0)],
            4,
            0,
            &mut scratch,
        );
        assert!(hits.is_empty(), "an id beyond the graph is skipped");
    }

    #[test]
    fn upper_layer_search_uses_that_layers_list() {
        // Two nodes at layer 1 with no layer-0 link between them: a layer-1
        // search must still reach the second node, a layer-0 search must not.
        let codec = Codec::new(HnswPrecision::Plain, 2);
        let mut g = FlatGraph::new(codec.vector_bytes(), 4);
        g.push_node(1, ItemPointer::new(1, 1), false, &codec.encode(&[0.0, 0.0]));
        g.push_node(1, ItemPointer::new(2, 1), false, &codec.encode(&[1.0, 0.0]));
        g.set_list(0, 1, &[1]);
        g.set_list(1, 1, &[0]);

        let mut scratch = SearchScratch::new();
        let at_l1 = search_layer_flat(
            &codec,
            DistanceType::L2,
            &[0.9, 0.0],
            &g,
            &[(distance_l2(&[0.9, 0.0], &[0.0, 0.0]), 0)],
            2,
            1,
            &mut scratch,
        );
        assert_eq!(at_l1.len(), 2);

        let at_l0 = search_layer_flat(
            &codec,
            DistanceType::L2,
            &[0.9, 0.0],
            &g,
            &[(distance_l2(&[0.9, 0.0], &[0.0, 0.0]), 0)],
            2,
            0,
            &mut scratch,
        );
        assert_eq!(at_l0.len(), 1, "layer 0 has no edges in this graph");
    }

    /// Small graph with explicit positions and lists, for the selection/backlink
    /// tests (dim 2 so the geometry is easy to reason about).
    fn graph_with(positions: &[[f32; 2]], cap: usize, lists: &[&[u32]]) -> (FlatGraph, Codec) {
        let codec = Codec::new(HnswPrecision::Plain, 2);
        let mut g = FlatGraph::new(codec.vector_bytes(), cap);
        for (i, p) in positions.iter().enumerate() {
            g.push_node(
                0,
                ItemPointer::new(i as u32 + 1, 1),
                false,
                &codec.encode(&p.to_vec()),
            );
        }
        for (i, list) in lists.iter().enumerate() {
            g.set_list(i as u32, 0, list);
        }
        (g, codec)
    }

    fn all_lists(g: &FlatGraph) -> Vec<Vec<u32>> {
        (0..g.len() as u32)
            .map(|i| g.neighbors(i, 0).to_vec())
            .collect()
    }

    #[test]
    fn own_list_is_heuristic_output_without_backfill() {
        // Subject at the origin; node 1 is close to node 2 and node 3 is farther
        // but sits near node 1, so both are occluded by node 1.  Distances are the
        // real squared-L2 values the search produces (the occlusion rule compares
        // in those units — feeding fabricated distances here silently changes who
        // is occluded, which is how this test first failed).
        let (g, codec) = graph_with(
            &[[0.0, 0.0], [1.0, 0.0], [1.2, 0.0], [3.0, 0.0]],
            4,
            &[&[], &[], &[], &[]],
        );
        let subject = [0.0f32, 0.0];
        let cands = vec![
            (distance_l2(&subject, &[1.0, 0.0]), 1),
            (distance_l2(&subject, &[1.2, 0.0]), 2),
            (distance_l2(&subject, &[3.0, 0.0]), 3),
        ];
        let mut buf = FlatPairBuf::new(2);
        let selected = select_neighbors_flat(distance_l2, &codec, &g, &mut buf, cands, 4);
        assert_eq!(
            selected,
            vec![1],
            "occluded candidates are dropped and nothing backfills them"
        );
    }

    #[test]
    fn own_list_keeps_diverse_candidates() {
        // Candidates that are mutually far apart relative to their distance from
        // the subject are all kept: diversity is what the heuristic is for.  (Note
        // (1,±1) would NOT qualify: from a subject at the origin they are occluded
        // by (1,0), which is how this test first failed.)
        let (g, codec) = graph_with(
            &[[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
            4,
            &[&[], &[], &[], &[]],
        );
        let subject = [0.0f32, 0.0];
        let cands = vec![
            (distance_l2(&subject, &[1.0, 0.0]), 1),
            (distance_l2(&subject, &[-1.0, 0.0]), 2),
            (distance_l2(&subject, &[0.0, 1.0]), 3),
        ];
        let mut buf = FlatPairBuf::new(2);
        let selected = select_neighbors_flat(distance_l2, &codec, &g, &mut buf, cands, 4);
        assert_eq!(selected.len(), 3, "mutually distant candidates all survive");
        assert_eq!(selected[0], 1, "ascending by (distance, id)");
    }

    #[test]
    fn backlink_appends_while_there_is_room() {
        // entry at the origin, newcomer selects it: node 0 must gain it as an
        // incoming edge, which is what makes the newcomer reachable at all.
        let (mut g, codec) = graph_with(&[[0.0, 0.0], [1.0, 0.0]], 4, &[&[], &[]]);
        assert!(g.promote_entry(0));
        let mut buf = FlatPairBuf::new(2);
        apply_flat(
            &codec,
            distance_l2,
            &mut g,
            &mut buf,
            1,
            0,
            vec![LayerPlanFlat { layer: 0, ids: vec![0] }],
            2,
            4,
        );
        assert_eq!(g.neighbors(1, 0), &[0], "own list published");
        assert_eq!(g.neighbors(0, 0), &[1], "backlink appended (room was left)");
    }

    #[test]
    fn saturated_backlink_keeps_capacity_and_evicts_by_distance() {
        // node 0's list is full with node 1 (close) and node 2 (far); the newcomer
        // at 0.5 is closer than both, so one of them must go.
        let (mut g, codec) = graph_with(
            &[[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [0.5, 0.0]],
            2,
            &[&[1, 2], &[], &[], &[]],
        );
        assert!(g.promote_entry(0));
        let mut buf = FlatPairBuf::new(2);
        apply_flat(
            &codec,
            distance_l2,
            &mut g,
            &mut buf,
            3,
            0,
            vec![LayerPlanFlat { layer: 0, ids: vec![0] }],
            1,
            2,
        );
        let list = g.neighbors(0, 0);
        assert_eq!(list.len(), 2, "capacity is kept (no backfill by the newcomer)");
        assert!(list.contains(&3), "the closer newcomer was admitted");
        assert!(!list.contains(&2), "the farthest member lost its slot");
        assert!(list.contains(&1));
    }

    #[test]
    fn occluded_newcomer_does_not_join_a_saturated_list() {
        // node 3 sits between node 1 and node 2, both closer to the target than
        // node 3 and mutually occluding, so node 3 is occluded from every
        // accepted entry and the list keeps its membership.
        let (mut g, codec) = graph_with(
            &[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.2, 0.0]],
            2,
            &[&[1, 2], &[], &[], &[]],
        );
        assert!(g.promote_entry(0));
        let mut buf = FlatPairBuf::new(2);
        apply_flat(
            &codec,
            distance_l2,
            &mut g,
            &mut buf,
            3,
            0,
            vec![LayerPlanFlat { layer: 0, ids: vec![0] }],
            1,
            2,
        );
        assert_eq!(g.neighbors(0, 0), &[1, 2], "membership unchanged");
    }

    #[test]
    fn apply_is_deterministic_for_the_same_inputs() {
        let build = || -> (FlatGraph, Codec) {
            let (mut g, codec) = graph_with(
                &[[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [0.5, 0.0], [2.5, 0.3]],
                2,
                &[&[1, 2], &[], &[], &[], &[]],
            );
            assert!(g.promote_entry(0));
            let mut buf = FlatPairBuf::new(2);
            for (new_id, target) in [(3u32, 0u32), (4, 0), (4, 1)] {
                apply_flat(
                    &codec,
                    distance_l2,
                    &mut g,
                    &mut buf,
                    new_id,
                    0,
                    vec![LayerPlanFlat { layer: 0, ids: vec![target] }],
                    1,
                    2,
                );
            }
            (g, codec)
        };
        let (a, _) = build();
        let (b, _) = build();
        assert_eq!(all_lists(&a), all_lists(&b), "same inputs, same graph");
        for i in 0..a.len() as u32 {
            let list = a.neighbors(i, 0);
            assert!(list.len() <= 2, "no list exceeds capacity");
            assert!(!list.contains(&i), "no self-link");
            let mut sorted = list.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), list.len(), "no duplicate neighbour");
        }
    }
}
