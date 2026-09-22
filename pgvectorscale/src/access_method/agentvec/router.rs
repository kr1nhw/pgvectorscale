//! Deterministic segment router: a Vamana graph over the owned IVF segments'
//! centroids (phase 3.5 / design §18, M6).
//!
//! Every consolidation inserts the new segment's centroids as nodes of an
//! embedded hnswsq region — the ported DISANN greedy-search + robust-prune
//! machinery, so the centroid graph *is* a Vamana graph.  A query navigates
//! it from the region's entry points, and the segments owning the visited
//! centroids are ranked by their best centroid distance; the top
//! `router_top_m` are activated alongside the always-searched HOT (and
//! legacy FLAT) segments.  External segments have no owned centroids, so
//! they are routed at segment granularity: always activated as a whole.
//!
//! Crash safety: centroid nodes are inserted *before* the segment is
//! published.  A crash in that window leaves nodes whose segment id never
//! appears in the directory; `route` drops unknown ids, and a later rebuild
//! reclaims the garbage.  A segment can therefore never be published without
//! its centroids being routable.
//!
//! The region stores the element's heap-TID slot as a synthetic pointer
//! `(segment_id, list_id + 1)` — nothing in the router path ever fetches a
//! heap tuple through it.

use std::collections::HashSet;

use pgrx::*;

use crate::access_method::agentvec::directory::{
    AgentVecDirectory, SegmentAlgorithm, SegmentOwnership,
};
use crate::access_method::agentvec::meta_page::AgentVecMetaPage;
use crate::access_method::agentvec::options::TSVAgentVecOptions;
use crate::access_method::distance::{preprocess_cosine, DistanceType};
use crate::access_method::hnswsq::quantize::HnswPrecision;
use crate::access_method::hnswsq::types::{Element, Visited};
use crate::access_method::hnswsq::utils::{self, SearchScratch};
use crate::util::ItemPointer;

/// Graph degree of the centroid Vamana graph.
pub const ROUTER_M: usize = 16;
/// `ef_construction` used when inserting centroid nodes.
pub const ROUTER_EF_CONSTRUCTION: usize = 100;
/// `ef` of the router search (how many centroids the query visits).
pub const ROUTER_EF_SEARCH: usize = 64;

/// Create the router region if it does not exist and return its base block.
///
/// The creation happens inside the meta-page update, so concurrent
/// consolidations serialize on the meta lock and exactly one region is
/// created.
pub unsafe fn ensure_router_region(index: &PgRelation, dim: u32) -> pg_sys::BlockNumber {
    AgentVecMetaPage::update(index, |meta| {
        if meta.get_router_base() != 0 {
            return meta.get_router_base();
        }
        let base = create_region(index, dim);
        meta.set_router_base(base);
        base
    })
}

/// Allocate the region at the relation end: metapage at `base`, empty head
/// page at `base + 1` (the empty-graph convention the HOT regions use).
unsafe fn create_region(index: &PgRelation, dim: u32) -> pg_sys::BlockNumber {
    let _ext_lock = crate::util::buffer::LockRelationForExtension::new(index);
    let base = pg_sys::RelationGetNumberOfBlocksInFork(
        index.as_ptr(),
        pg_sys::ForkNumber::MAIN_FORKNUM,
    );
    utils::init_region(
        index.as_ptr(),
        base,
        dim as usize,
        ROUTER_M,
        ROUTER_EF_CONSTRUCTION,
        HnswPrecision::Plain,
        ItemPointer::new_invalid(),
    );
    base
}

/// Insert one owned segment's centroids as router nodes, encoded as the
/// synthetic TID `(segment_id, list_id + 1)`.
pub unsafe fn add_segment_centroids(
    index: &PgRelation,
    router_base: pg_sys::BlockNumber,
    segment_id: u64,
    centroids: &[Vec<f32>],
) {
    assert!(
        (1..=u32::MAX as u64).contains(&segment_id),
        "agentvec: segment id {segment_id} does not fit the router TID encoding"
    );
    let support = utils::init_support(index.as_ptr(), router_base);
    for (list_id, centroid) in centroids.iter().enumerate() {
        let mut stored = centroid.clone();
        if support.dist_type == DistanceType::Cosine {
            preprocess_cosine(&mut stored);
        }
        let encoded = support.codec.encode(&stored);
        let mut tid = pg_sys::ItemPointerData::default();
        pgrx::itemptr::item_pointer_set_all(
            &mut tid,
            segment_id as pg_sys::BlockNumber,
            (list_id + 1) as pg_sys::OffsetNumber,
        );
        crate::access_method::hnswsq::insert::insert_tuple_on_disk(
            index.as_ptr(),
            router_base,
            &support,
            &encoded,
            &tid,
            false,
            false,
        );
    }
}

/// Route the query against the centroid Vamana graph.
///
/// Returns `Some(segment ids to search)` — the top `router_top_m` owned IVF
/// segments by best visited-centroid distance — or `None` when every owned
/// IVF segment must be searched (no router region yet, or the region has no
/// usable nodes).
pub unsafe fn route(
    index: &PgRelation,
    query: &[f32],
    directory: &AgentVecDirectory,
    options: &TSVAgentVecOptions,
    router_base: pg_sys::BlockNumber,
) -> Option<HashSet<u64>> {
    if router_base == 0 {
        return None;
    }
    let support = utils::init_support(index.as_ptr(), router_base);

    let mut m = 0usize;
    utils::get_meta_page_info(index.as_ptr(), router_base, Some(&mut m), None);
    let mut visited = Visited::new(1000 * m * 2);
    let mut scratch = SearchScratch::new(m);
    let mut tuples: i64 = 0;
    let norm_q = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    let (candidates, _m) = crate::access_method::hnswsq::scan::region_candidates(
        index.as_ptr(),
        router_base,
        &support,
        Some(query),
        ROUTER_EF_SEARCH,
        &mut visited,
        &mut scratch,
        None,
        &mut tuples,
    );

    // Rank segments by their best centroid distance.  The candidates come
    // back furthest-first, so the reversed drain is ascending: the first
    // time a segment is seen is its best distance.
    let mut ranked: Vec<(u64, f32)> = Vec::new();
    for sc in candidates.into_iter().rev() {
        let element =
            crate::access_method::hnswsq::ptr::access::<Element>(std::ptr::null_mut(), sc.element);
        if (*element).deleted != 0 {
            continue;
        }
        let segment_id = pgrx::itemptr::item_pointer_get_block_number(&(*element).heaptid) as u64;
        // Drop orphaned nodes (segment never published) and nodes of
        // segments this scan would not search anyway.
        let Some(seg) = directory.get(segment_id) else {
            continue;
        };
        if seg.algorithm() != SegmentAlgorithm::IvfRaBitQ
            || seg.ownership() != SegmentOwnership::Owned
        {
            continue;
        }
        if ranked.iter().any(|(id, _)| *id == segment_id) {
            continue;
        }
        let dist =
            crate::access_method::hnswsq::scan::emit_candidate(&support, query, norm_q, &sc) as f32;
        ranked.push((segment_id, dist));
    }

    if ranked.is_empty() {
        // The region exists but holds no usable nodes (created by a crashed
        // consolidation, or nothing converted yet): search everything.
        return None;
    }

    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    let top_m = options.get_router_top_m() as usize;
    Some(ranked.into_iter().take(top_m).map(|(id, _)| id).collect())
}
