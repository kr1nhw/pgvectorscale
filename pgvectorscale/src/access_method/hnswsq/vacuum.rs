//! hnswsq vacuum implementation (autovacuum-ready).
//!
//! `ambulkdelete` runs in four phases over a LINEAR page walk (never a graph
//! traversal, so crash-orphaned unreachable nodes are cleaned too):
//!
//! 1. **Mark** — peek each node page under a share lock; items whose heap TID
//!    the callback reports dead are tombstoned in place (deleted flag +
//!    invalidated TID) under one exclusive page lock (GenericXLog WAL).  No
//!    cleanup locks are required: tombstones keep their items (routing stays
//!    intact) and every read path re-validates locations atomically.
//! 2. **Repair** — for each fresh tombstone, its live neighbors get the
//!    tombstone removed from their lists and the tombstone's other live
//!    neighbors spliced in (heuristic-pruned), preserving connectivity.
//!    Updates use the same two-phase optimistic protocol as insert backlinks
//!    (snapshot under share lock → validate + write under exclusive lock →
//!    retry → remove-only fallback), so vacuum never holds two content locks.
//! 3. **Entry fix** — a tombstoned entry point is replaced by the highest-
//!    level live node seen during the walk (or invalidated when none).
//! 4. **Free** — pages whose items are ALL tombstoned go onto the meta free
//!    list for insert reuse.  Safe without pin quiescence: after repair no
//!    live list references them, and stale scan pointers resolve through
//!    `load_node_view`'s guards to "gone" or to the valid node that replaced
//!    them (approximate-search semantics, identical to pgvector's
//!    deleted-page reuse).
//!
//! `amvacuumcleanup` refreshes the tuple estimate from the meta counters.

use pgrx::*;

use crate::access_method::hnswsq::graph::{
    distance_encoded, select_neighbors_heuristic, DiskGraph, GraphAccess,
};
use crate::access_method::hnswsq::insert::codec_for;
use crate::access_method::hnswsq::meta_page::HnswMetaPage;
use crate::access_method::hnswsq::node::{load_node_view, modify_node, HnswNode};
use crate::util::page::{PageType, ReadablePage, WritablePage};
use crate::util::ports::{PageGetItem, PageGetItemId, PageGetMaxOffsetNumber};
use crate::util::ItemPointer;

/// LP_NORMAL line-pointer flag (bufpage.h).
const LP_NORMAL_FLAG: u32 = 1;

/// A fresh tombstone recorded during the mark pass, with its neighbor lists
/// copied out for the repair pass.
struct DeadNode {
    ptr: ItemPointer,
    level: u8,
    neighbors: Vec<Vec<ItemPointer>>,
}

/// Per-page accounting from the mark pass.
struct PageStat {
    block: pg_sys::BlockNumber,
    total: u32,
    dead: u32,
}

/// Bulk delete tuples from the hnswsq index.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambulkdelete(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    callback: pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut std::os::raw::c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let results = if stats.is_null() {
        unsafe { PgBox::<pg_sys::IndexBulkDeleteResult>::alloc0().into_pg() }
    } else {
        stats
    };

    let index_rel = unsafe { PgRelation::from_pg((*info).index) };
    let meta = HnswMetaPage::fetch(&index_rel);
    let codec = codec_for(&index_rel, &meta);
    let distance_type = meta.get_distance_type();
    let entry_point = meta.get_entry_point();

    let nblocks = unsafe {
        pg_sys::RelationGetNumberOfBlocksInFork(
            index_rel.as_ptr(),
            pg_sys::ForkNumber::MAIN_FORKNUM,
        )
    };

    let mut dead_nodes: Vec<DeadNode> = Vec::new();
    let mut page_stats: Vec<PageStat> = Vec::new();
    let mut best_live: Option<(ItemPointer, u8)> = None;
    let mut total_nodes: u64 = 0;
    let mut total_dead: u64 = 0;

    // ---- Phase 1: mark (linear walk; block 0 is the meta page). ----
    for block in 1..nblocks {
        #[cfg(feature = "pg18")]
        unsafe {
            pg_sys::vacuum_delay_point(false)
        }
        #[cfg(not(feature = "pg18"))]
        unsafe {
            pg_sys::vacuum_delay_point()
        }

        // Peek under a share lock: classify items without dirtying clean
        // pages.  The callback's dead set is stable for this vacuum run, so
        // peek-then-mark is race-free with respect to deadness.
        struct Peeked {
            off: pg_sys::OffsetNumber,
            level: u8,
            neighbors: Vec<Vec<ItemPointer>>,
        }
        let mut new_dead: Vec<Peeked> = Vec::new();
        let mut page_total: u32 = 0;
        let mut page_dead: u32 = 0;
        {
            let page = unsafe { ReadablePage::read(&index_rel, block) };
            if page.get_type() != PageType::HnswNode {
                continue; // meta / calibration / free page
            }
            let max_off = unsafe { PageGetMaxOffsetNumber(*page) };
            for off in 1..=max_off {
                let off = off as pg_sys::OffsetNumber;
                let item_id = unsafe { PageGetItemId(*page, off) };
                if unsafe { (*item_id).lp_flags() } != LP_NORMAL_FLAG
                    || unsafe { (*item_id).lp_len() } == 0
                {
                    continue;
                }
                let item = unsafe { PageGetItem(*page, item_id) } as *const u8;
                let len = unsafe { (*item_id).lp_len() } as usize;
                let node = unsafe {
                    rkyv::archived_root::<HnswNode>(std::slice::from_raw_parts(item, len))
                };
                page_total += 1;
                let ptr = ItemPointer::new(block, off);
                if node.is_deleted() {
                    // Tombstone from an earlier vacuum.
                    page_dead += 1;
                    continue;
                }
                let heap_tid = node.heap_tid.deserialize_item_pointer();
                let is_dead = if let Some(cb) = callback {
                    let mut tid_data = pg_sys::ItemPointerData::default();
                    heap_tid.to_item_pointer_data(&mut tid_data);
                    unsafe { cb(&mut tid_data, callback_state) }
                } else {
                    false
                };
                if is_dead {
                    page_dead += 1;
                    let level = node.level;
                    let neighbors = (0..=level as usize)
                        .map(|l| node.iter_valid_neighbors(l).collect())
                        .collect();
                    new_dead.push(Peeked {
                        off,
                        level,
                        neighbors,
                    });
                } else if best_live.map(|(_, bl)| node.level > bl).unwrap_or(true) {
                    best_live = Some((ptr, node.level));
                }
            }
        }

        if new_dead.is_empty() {
            if page_total > 0 {
                page_stats.push(PageStat {
                    block,
                    total: page_total,
                    dead: page_dead,
                });
                total_nodes += page_total as u64;
                total_dead += page_dead as u64;
            }
            continue;
        }

        // Mark under one exclusive page lock (all tombstones on this page in
        // a single GenericXLog commit).
        {
            let page = WritablePage::modify(&index_rel, block);
            for d in &new_dead {
                let item_id = unsafe { PageGetItemId(*page, d.off) };
                let item = unsafe { PageGetItem(*page, item_id) } as *mut u8;
                let len = unsafe { (*item_id).lp_len() } as usize;
                let archived = unsafe {
                    rkyv::archived_root_mut::<HnswNode>(std::pin::Pin::new(
                        std::slice::from_raw_parts_mut(item, len),
                    ))
                };
                unsafe {
                    archived.mark_deleted();
                }
            }
            page.commit();
        }

        for d in new_dead {
            dead_nodes.push(DeadNode {
                ptr: ItemPointer::new(block, d.off),
                level: d.level,
                neighbors: d.neighbors,
            });
        }
        page_stats.push(PageStat {
            block,
            total: page_total,
            dead: page_dead,
        });
        total_nodes += page_total as u64;
        total_dead += page_dead as u64;
    }

    // ---- Phase 2: repair (splice around fresh tombstones). ----
    for d in &dead_nodes {
        for l in 0..=d.level as usize {
            // Live neighbors of d at layer l (loaded fresh: skips old
            // tombstones and recycled pages).
            let additions: Vec<ItemPointer> = d.neighbors[l]
                .iter()
                .copied()
                .filter(|p| *p != d.ptr)
                .filter(|p| {
                    load_node_view(&index_rel, *p)
                        .map(|v| !v.deleted)
                        .unwrap_or(false)
                })
                .collect();
            for &n in &d.neighbors[l] {
                if n == d.ptr {
                    continue;
                }
                unsafe {
                    repair_neighbor(
                        &index_rel, &codec, distance_type, n, d.ptr, &additions, l,
                        meta.cap_for_layer(l),
                    );
                }
            }
        }
    }

    // ---- Phase 3: entry-point fix. ----
    let entry_dead = match entry_point {
        Some(ep) => load_node_view(&index_rel, ep)
            .map(|v| v.deleted)
            .unwrap_or(true),
        None => false,
    };

    // ---- Phase 4: free fully-dead pages (after repair, so no live list
    // references them) and publish counters.  push_free_pages re-verifies
    // each page under its exclusive lock (a concurrent insert with a stale
    // hint may have revived it) and returns what was actually freed. ----
    let candidates: Vec<pg_sys::BlockNumber> = page_stats
        .iter()
        .filter(|ps| ps.total > 0 && ps.dead == ps.total)
        .map(|ps| ps.block)
        .collect();
    let freed = if candidates.is_empty() {
        Vec::new()
    } else {
        unsafe { HnswMetaPage::push_free_pages(&index_rel, &candidates) }
    };
    for ps in page_stats.iter().filter(|ps| freed.contains(&ps.block)) {
        total_nodes -= ps.total as u64;
        total_dead -= ps.dead as u64;
    }

    // Entry fix: install the best live node when the entry was tombstoned, or
    // promote when a strictly higher live level exists (race-checked inside
    // the meta RMW against concurrent insert promotions).
    unsafe {
        HnswMetaPage::set_counts_and_entry(
            &index_rel,
            total_nodes,
            total_dead,
            best_live,
            entry_point,
            entry_dead,
        );
    }

    unsafe {
        (*results).num_index_tuples = (total_nodes - total_dead) as f64;
        (*results).pages_deleted = freed.len() as u32;
        // pg_class.relpages is updated from this field after VACUUM; leaving
        // it zero would make the planner think the index shrank to nothing.
        (*results).num_pages = pg_sys::RelationGetNumberOfBlocksInFork(
            index_rel.as_ptr(),
            pg_sys::ForkNumber::MAIN_FORKNUM,
        );
    }

    // Test-build diagnostics: walk the finished index and report live/dead
    // counts plus directed layer-0 reachability from the entry point (the
    // invariant a search relies on: every live node is reachable).
    #[cfg(any(test, feature = "pg_test"))]
    unsafe {
        use std::collections::{HashSet, VecDeque};
        let nblocks2 = pg_sys::RelationGetNumberOfBlocksInFork(
            index_rel.as_ptr(),
            pg_sys::ForkNumber::MAIN_FORKNUM,
        );
        let mut walk_total = 0u64;
        let mut walk_live = 0u64;
        for block in 1..nblocks2 {
            let page = ReadablePage::read(&index_rel, block);
            if page.get_type() != PageType::HnswNode {
                continue;
            }
            let max_off = PageGetMaxOffsetNumber(*page);
            for off in 1..=max_off {
                let item_id = PageGetItemId(*page, off as pg_sys::OffsetNumber);
                if (*item_id).lp_flags() != LP_NORMAL_FLAG || (*item_id).lp_len() == 0 {
                    continue;
                }
                let item = PageGetItem(*page, item_id) as *const u8;
                let len = (*item_id).lp_len() as usize;
                let node =
                    rkyv::archived_root::<HnswNode>(std::slice::from_raw_parts(item, len));
                walk_total += 1;
                if !node.is_deleted() {
                    walk_live += 1;
                }
            }
        }
        let meta2 = HnswMetaPage::fetch(&index_rel);
        let mut reach_live = 0u64;
        let mut reach_total = 0u64;
        if let Some(ep) = meta2.get_entry_point() {
            let mut seen: HashSet<ItemPointer> = HashSet::new();
            let mut q: VecDeque<ItemPointer> = VecDeque::new();
            seen.insert(ep);
            q.push_back(ep);
            while let Some(p) = q.pop_front() {
                if let Some(v) = load_node_view(&index_rel, p) {
                    reach_total += 1;
                    if !v.deleted {
                        reach_live += 1;
                    }
                    for n in v.neighbors.first().cloned().unwrap_or_default() {
                        if seen.insert(n) {
                            q.push_back(n);
                        }
                    }
                }
            }
        }
        pgrx::log!(
            "hnswsq vacuum diag: walk_total={} walk_live={} tomb={} counted(total={},dead={}) \
             reach_total={} reach_live={} freed={} entry={:?} entry_level={}",
            walk_total,
            walk_live,
            walk_total - walk_live,
            total_nodes,
            total_dead,
            reach_total,
            reach_live,
            freed.len(),
            meta2.get_entry_point().map(|p| (p.block_number, p.offset)),
            meta2.get_entry_level()
        );
    }

    results
}

/// Two-phase repair of one live neighbor list: remove `remove` and splice in
/// `additions` (heuristic-pruned to `cap`).  Falls back to remove-only after
/// repeated racing changes — removal is the correctness-critical part; the
/// splice is connectivity preservation.
#[allow(clippy::too_many_arguments)]
unsafe fn repair_neighbor(
    index: &PgRelation,
    codec: &crate::access_method::hnswsq::quantize::Codec,
    dist_type: crate::access_method::distance::DistanceType,
    target: ItemPointer,
    remove: ItemPointer,
    additions: &[ItemPointer],
    layer: usize,
    cap: usize,
) {
    let access = DiskGraph { index };

    enum Outcome {
        Wrote,
        Changed,
        Gone,
    }

    for _attempt in 0..3 {
        // ---- Phase 1: snapshot ----
        let Some(view) = load_node_view(index, target) else {
            return;
        };
        if view.deleted || (view.level as usize) < layer {
            return;
        }
        let current: Vec<ItemPointer> = view.neighbors.get(layer).cloned().unwrap_or_default();
        let has_remove = current.contains(&remove);
        let missing_additions: Vec<ItemPointer> = additions
            .iter()
            .copied()
            .filter(|a| !current.contains(a))
            .collect();
        if !has_remove && missing_additions.is_empty() {
            return; // nothing to do
        }

        let subject = codec.decode(&view.vector);
        let mut cands: Vec<(f32, ItemPointer)> = Vec::with_capacity(current.len() + additions.len());
        let mut seen: Vec<ItemPointer> = Vec::with_capacity(current.len() + additions.len());
        for &c in current.iter().chain(missing_additions.iter()) {
            if c == remove || seen.contains(&c) {
                continue;
            }
            let Some(enc) = access.vector(c) else {
                continue; // vanished member: dropped (implicit repair)
            };
            seen.push(c);
            let d = distance_encoded(codec, dist_type, &subject, &enc);
            cands.push((d, c));
        }
        let new_list =
            select_neighbors_heuristic(codec, dist_type, &access, cands, cap);

        // ---- Phase 2: validate + write ----
        let identity_tid = view.heap_tid;
        let identity_level = view.level;
        let outcome = modify_node(index, target, |mut archived| {
            if archived.is_deleted() {
                return Outcome::Gone;
            }
            let tid = archived.heap_tid.deserialize_item_pointer();
            if tid != identity_tid || archived.level != identity_level {
                return Outcome::Gone;
            }
            let cur: Vec<ItemPointer> = archived.iter_valid_neighbors(layer).collect();
            if cur != current {
                return Outcome::Changed;
            }
            archived.as_mut().set_neighbors(layer, &new_list, cap);
            Outcome::Wrote
        });
        match outcome {
            Some(Outcome::Wrote) | Some(Outcome::Gone) | None => return,
            Some(Outcome::Changed) => continue,
        }
    }

    // Fallback: remove-only (single page, no external loads).
    let _ = modify_node(index, target, |mut archived| {
        if archived.is_deleted() {
            return;
        }
        let cur: Vec<ItemPointer> = archived.iter_valid_neighbors(layer).collect();
        if cur.contains(&remove) {
            let list: Vec<ItemPointer> = cur.into_iter().filter(|x| *x != remove).collect();
            archived.as_mut().set_neighbors(layer, &list, cap);
        }
    });
}

/// Cleanup after vacuum: refresh the tuple estimate from the meta counters on
/// cleanup-only calls (no `ambulkdelete` phase ran).
#[pg_guard]
pub unsafe extern "C-unwind" fn amvacuumcleanup(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    if !stats.is_null() || (*info).analyze_only {
        return stats;
    }
    let results = unsafe { PgBox::<pg_sys::IndexBulkDeleteResult>::alloc0().into_pg() };
    let index_rel = unsafe { PgRelation::from_pg((*info).index) };
    let meta = HnswMetaPage::fetch(&index_rel);
    unsafe {
        (*results).num_index_tuples =
            (meta.get_node_count().saturating_sub(meta.get_deleted_count())) as f64;
        (*results).num_pages = pg_sys::RelationGetNumberOfBlocksInFork(
            index_rel.as_ptr(),
            pg_sys::ForkNumber::MAIN_FORKNUM,
        );
    }
    results
}
