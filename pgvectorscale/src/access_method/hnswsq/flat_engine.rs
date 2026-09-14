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
        if i >= g.len() || i >= visited_epoch.len() || visited_epoch[i] == epoch {
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
            if ni >= g.len() || ni >= visited_epoch.len() || visited_epoch[ni] == epoch {
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
}
