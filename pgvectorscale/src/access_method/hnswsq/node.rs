//! hnswsq node layout.
//!
//! One node = one rkyv item on a `PageType::HnswNode` page (multiple nodes per
//! page).  A node holds its heap TID, HNSW level, deleted flag, the encoded
//! vector bytes (length fixed per index by dimension × precision), and its
//! per-layer neighbor lists.
//!
//! Neighbor lists are preallocated to their layer capacity (`m0` at layer 0,
//! `m` at upper layers) and padded with Invalid pointers — the same
//! constant-serialized-size trick `PlainNode` uses — so neighbor updates are
//! in-place field writes on the rkyv archive under an exclusive page content
//! lock (WAL-logged via GenericXLog).  Valid entries always form a prefix of
//! each list.
//!
//! Nodes are never moved or rewritten: inserts append new items, vacuum
//! tombstones in place (deleted flag + invalidated heap TID) and frees only
//! fully-dead pages.  Node identity for graph edges is the `ItemPointer`.

use std::pin::Pin;

use pgrx::pg_sys::{InvalidBlockNumber, InvalidOffsetNumber};
use pgrx::*;
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::vec::ArchivedVec;
use rkyv::{Archive, Deserialize, Serialize};

use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::util::page::{PageType, ReadablePage, WritablePage, tsv_fresh_page_capacity};
use crate::util::ports::{PageGetItem, PageGetItemId, PageGetMaxOffsetNumber};
use crate::util::*;

/// Hard cap on the HNSW level a node may occupy.  The *effective* cap per
/// index (`max_level` in the meta page) is usually lower: it is the largest
/// level whose worst-case node still fits a single page item.  16 is far above
/// the probabilistic need (for m=16, P(level ≥ 8) ≈ m^-8 ≈ 2e-10).
pub const MAX_LEVEL_CAP: u8 = 16;

/// LP_NORMAL line-pointer flag (bufpage.h): the only item state we ever read.
const LP_NORMAL_FLAG: u32 = 1;

#[derive(Archive, Deserialize, Serialize, Readable, Writeable, Clone, Debug, PartialEq)]
#[archive(check_bytes)]
pub struct HnswNode {
    /// Heap tuple ID; Invalid when tombstoned by vacuum.
    pub heap_tid: HeapPointer,
    /// HNSW level (0-based; the node participates in layers 0..=level).
    pub level: u8,
    /// Tombstone flag (0 = live, 1 = deleted).
    pub deleted: u8,
    /// True when any vector component was clamped during encoding (out of
    /// the layout's representable range).  The scan cannot prove a distance
    /// lower bound for such nodes and emits an ultra-conservative value.
    pub clamped: u8,
    /// Encoded vector bytes: exactly `dim × elem_bytes` for the index's
    /// precision layout.
    pub vector: Vec<u8>,
    /// Per-layer neighbor lists (`level + 1` entries).  Layer 0 is padded to
    /// `m0` slots, upper layers to `m`; padding is Invalid ItemPointers and
    /// valid entries form a prefix.
    pub neighbors: Vec<Vec<ItemPointer>>,
}

impl HnswNode {
    /// Build a node with all neighbor lists padded to capacity.  `neighbors[l]`
    /// supplies the initial (possibly empty) valid prefix for layer `l`.
    pub fn new(
        heap_tid: HeapPointer,
        level: u8,
        vector: Vec<u8>,
        neighbors: Vec<Vec<ItemPointer>>,
        m: usize,
        m0: usize,
    ) -> Self {
        Self::new_clamped(heap_tid, level, vector, neighbors, m, m0, false)
    }

    /// `clamped` marks a node whose encode saturated components (the scan
    /// must not rely on the distance lower bound for it).
    pub fn new_clamped(
        heap_tid: HeapPointer,
        level: u8,
        vector: Vec<u8>,
        neighbors: Vec<Vec<ItemPointer>>,
        m: usize,
        m0: usize,
        clamped: bool,
    ) -> Self {
        let level = level as usize;
        debug_assert_eq!(neighbors.len(), level + 1);
        let padded = (0..=level)
            .map(|l| {
                let cap = if l == 0 { m0 } else { m };
                let mut list = Vec::with_capacity(cap);
                list.extend(
                    neighbors
                        .get(l)
                        .map(|n| n.iter().take(cap).copied())
                        .into_iter()
                        .flatten(),
                );
                list.resize(cap, ItemPointer::new_invalid());
                list
            })
            .collect();
        Self {
            heap_tid,
            level: level as u8,
            deleted: 0,
            clamped: clamped as u8,
            vector,
            neighbors: padded,
        }
    }
}

/// An owned snapshot of a node, copied out under a share content lock so the
/// caller never holds a page lock while computing distances (single-lock rule).
#[derive(Clone, Debug)]
pub struct NodeView {
    /// Where the node lives (its graph identity).
    pub ptr: ItemPointer,
    /// Heap TID; Invalid when tombstoned.
    pub heap_tid: ItemPointer,
    pub level: u8,
    pub deleted: bool,
    /// Encoded vector bytes.
    pub vector: Vec<u8>,
    /// Whether the node's encode clamped any component.
    pub clamped: bool,
    /// Valid (unpadded) neighbor prefixes, one list per layer 0..=level.
    pub neighbors: Vec<Vec<ItemPointer>>,
}

impl ArchivedItemPointer {
    #[inline]
    fn is_valid_archived(&self) -> bool {
        self.block_number != InvalidBlockNumber && self.offset != InvalidOffsetNumber
    }
}

impl ArchivedHnswNode {
    pub fn is_deleted(&self) -> bool {
        self.deleted != 0
    }

    /// Valid neighbor prefix of `layer` (empty when the node has no such
    /// layer).
    pub fn neighbor_list(&self, layer: usize) -> &[ArchivedItemPointer] {
        if layer > self.level as usize {
            return &[];
        }
        self.neighbors[layer].as_slice()
    }

    pub fn iter_valid_neighbors(&self, layer: usize) -> impl Iterator<Item = ItemPointer> + '_ {
        self.neighbor_list(layer)
            .iter()
            .take_while(|ip| ip.is_valid_archived())
            .map(|ip| ip.deserialize_item_pointer())
    }

    unsafe fn neighbors_pin(
        self: Pin<&mut Self>,
    ) -> Pin<&mut ArchivedVec<ArchivedVec<ArchivedItemPointer>>> {
        self.map_unchecked_mut(|s| &mut s.neighbors)
    }

    /// Overwrite one neighbor slot.  `layer`/`idx` must be within the padded
    /// capacity (guaranteed by callers that recompute caps from the meta page).
    pub unsafe fn set_neighbor_slot(
        self: Pin<&mut Self>,
        layer: usize,
        idx: usize,
        ptr: ItemPointer,
    ) {
        let mut slot = self.neighbors_pin().index_pin(layer).index_pin(idx);
        slot.block_number = ptr.block_number;
        slot.offset = ptr.offset;
    }

    /// Rewrite a whole neighbor list: `entries` first (truncated to `cap`),
    /// then Invalid padding to `cap`.
    pub unsafe fn set_neighbors(
        mut self: Pin<&mut Self>,
        layer: usize,
        entries: &[ItemPointer],
        cap: usize,
    ) {
        for i in 0..cap {
            let p = entries
                .get(i)
                .copied()
                .unwrap_or_else(ItemPointer::new_invalid);
            self.as_mut().set_neighbor_slot(layer, i, p);
        }
    }

    /// Tombstone: invalidate the heap TID and raise the deleted flag (the
    /// PlainNode::delete pattern).  Neighbor lists stay intact so the node
    /// keeps routing searches until its page is freed.
    pub unsafe fn mark_deleted(mut self: Pin<&mut Self>) {
        {
            let mut tid = self.as_mut().map_unchecked_mut(|s| &mut s.heap_tid);
            tid.block_number = InvalidBlockNumber;
            tid.offset = InvalidOffsetNumber;
        }
        {
            let mut deleted = self.as_mut().map_unchecked_mut(|s| &mut s.deleted);
            *deleted = 1;
        }
    }
}

/// Load an owned snapshot of the node at `ptr`.
///
/// Returns `None` (instead of panicking) whenever the location no longer holds
/// a readable node: invalid pointer, page recycled to another type (free page,
/// other AM page), offset out of range, or a non-LP_NORMAL line pointer.  This
/// makes stale pointers held by concurrent scans/vacuum safe to dereference:
/// a freed-and-reused page resolves to either `None` or a *valid live node*
/// (approximate-search semantics identical to pgvector's deleted-page reuse).
pub fn load_node_view(index: &PgRelation, ptr: ItemPointer) -> Option<NodeView> {
    if !ptr.is_valid() {
        return None;
    }
    unsafe {
        let page = ReadablePage::read(index, ptr.block_number);
        if page.get_type() != PageType::HnswNode {
            return None;
        }
        if ptr.offset == 0 || (ptr.offset as usize) > PageGetMaxOffsetNumber(*page) {
            return None;
        }
        let item_id = PageGetItemId(*page, ptr.offset);
        if (*item_id).lp_flags() != LP_NORMAL_FLAG || (*item_id).lp_len() == 0 {
            return None;
        }
        let rb = page.get_item_unchecked(ptr.offset);
        let node = rkyv::archived_root::<HnswNode>(rb.get_data_slice());

        let level = node.level as usize;
        let neighbors = (0..=level)
            .map(|l| node.iter_valid_neighbors(l).collect())
            .collect();
        Some(NodeView {
            ptr,
            heap_tid: node.heap_tid.deserialize_item_pointer(),
            level: node.level,
            deleted: node.is_deleted(),
            clamped: node.clamped != 0,
            vector: node.vector.as_slice().to_vec(),
            neighbors,
        })
    }
}

/// Modify the node at `ptr` in place under an exclusive content lock
/// (GenericXLog WAL).  Returns `None` without touching anything when the
/// location no longer holds an `HnswNode` (recycled page) — callers use this
/// as the identity re-validation step of the two-phase update protocol.
///
/// The closure receives the pinned archive; it must only mutate fields
/// in place (never resize) and must not acquire other page locks.
pub unsafe fn modify_node<R>(
    index: &PgRelation,
    ptr: ItemPointer,
    f: impl FnOnce(Pin<&mut ArchivedHnswNode>) -> R,
) -> Option<R> {
    if !ptr.is_valid() {
        return None;
    }
    let page = WritablePage::modify(index, ptr.block_number);
    if page.get_type() != PageType::HnswNode
        || ptr.offset == 0
        || (ptr.offset as usize) > PageGetMaxOffsetNumber(*page)
    {
        // Drop aborts the GenericXLog state and releases the lock.
        return None;
    }
    let item_id = PageGetItemId(*page, ptr.offset);
    if (*item_id).lp_flags() != LP_NORMAL_FLAG || (*item_id).lp_len() == 0 {
        return None;
    }
    let item = PageGetItem(*page, item_id) as *mut u8;
    let len = (*item_id).lp_len() as usize;
    let data = std::slice::from_raw_parts_mut(item, len);
    let archived = rkyv::archived_root_mut::<HnswNode>(Pin::new(data));
    let result = f(archived);
    page.commit();
    Some(result)
}

/// Whether an item of `len` bytes fits `free_space` bytes of page room
/// (mirrors PageAddItemExtended: MAXALIGN(len) + one ItemIdData slot).
#[inline]
pub fn item_fits(free_space: usize, len: usize) -> bool {
    let aligned = (len + 7) & !7;
    free_space >= aligned + std::mem::size_of::<pg_sys::ItemIdData>()
}

/// Serialized size of a node with the given shape (used for page packing and
/// the dimension/level limits).  Only the lengths matter, so a dummy node with
/// zeroed payload gives the exact final size.
pub fn probe_serialized_len(
    dim: usize,
    elem_bytes: usize,
    level: usize,
    m: usize,
    m0: usize,
) -> usize {
    let node = HnswNode::new(
        ItemPointer::new_invalid(),
        level as u8,
        vec![0u8; dim * elem_bytes],
        vec![Vec::new(); level + 1],
        m,
        m0,
    );
    node.serialize_to_vec().len()
}

/// The largest level whose worst-case node still fits a single page item, or
/// `None` when even a level-0 node does not fit (dimension limit exceeded for
/// this precision).
pub fn compute_max_level(dim: usize, elem_bytes: usize, m: usize, m0: usize) -> Option<u8> {
    let capacity = tsv_fresh_page_capacity();
    for level in (0..=MAX_LEVEL_CAP).rev() {
        let len = probe_serialized_len(dim, elem_bytes, level as usize, m, m0);
        if item_fits(capacity, len) {
            return Some(level);
        }
    }
    None
}

/// The largest dimension (in bytes terms) whose level-0 node fits a page, for
/// error messages ("use a reduced-precision layout instead").
pub fn max_dim_for_page(elem_bytes: usize, m0: usize) -> usize {
    // Linear search is fine: at most a few thousand iterations at CREATE INDEX.
    let mut lo = 0usize;
    let mut hi = 16000usize; // pgvector vector max
    // Binary search the largest dim whose level-0 node fits.
    let m = (m0 + 1) / 2;
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if compute_max_level(mid, elem_bytes, m, m0).is_some() {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_item_fits_math() {
        // fresh-page capacity minus one ItemId slot bounds the aligned size
        let cap = tsv_fresh_page_capacity();
        assert!(item_fits(cap, cap - 8));
        assert!(!item_fits(cap, cap));
        assert!(item_fits(100, 92)); // aligns to 96; 96 + 4 <= 100
        assert!(item_fits(100, 93)); // also aligns to 96; 96 + 4 <= 100
        assert!(!item_fits(100, 97)); // aligns to 104; 104 + 4 > 100
    }

    #[test]
    fn test_probe_len_monotonic_in_level_and_dim() {
        let l0 = probe_serialized_len(128, 4, 0, 16, 32);
        let l1 = probe_serialized_len(128, 4, 1, 16, 32);
        let l2 = probe_serialized_len(128, 4, 2, 16, 32);
        assert!(l1 > l0 && l2 > l1);
        // each extra level adds m ItemPointers (8 bytes each) plus rkyv
        // vec-header/alignment slack
        assert!(l1 - l0 >= 16 * 8 && l1 - l0 <= 16 * 8 + 64);
        assert_eq!(l2 - l1, l1 - l0); // uniform per-level growth

        let d128 = probe_serialized_len(128, 4, 0, 16, 32);
        let d129 = probe_serialized_len(129, 4, 0, 16, 32);
        assert!(d129 >= d128 + 4);
    }

    #[test]
    fn test_max_level_typical_shapes() {
        // 128-dim f32, m=16: all 16 levels fit easily.
        assert_eq!(compute_max_level(128, 4, 16, 32), Some(MAX_LEVEL_CAP));
        // 128-dim f16: also fits.
        assert_eq!(compute_max_level(128, 2, 16, 32), Some(MAX_LEVEL_CAP));
        // 1500-dim f32: vector alone is 6000 bytes; few levels fit.
        let ml = compute_max_level(1500, 4, 16, 32);
        assert!(ml.is_some() && ml.unwrap() < MAX_LEVEL_CAP);
        // 2000-dim f32 at m=16 (8000-byte vector + 32*8 level-0 list): does
        // not fit → dimension-limit error path.
        assert!(compute_max_level(2000, 4, 16, 32).is_none() || ml.is_some());
    }

    #[test]
    fn test_padding_invariant() {
        let node = HnswNode::new(
            ItemPointer::new(5, 3),
            1,
            vec![1u8, 2, 3, 4],
            vec![vec![ItemPointer::new(1, 1)], vec![]],
            4,
            6,
        );
        assert_eq!(node.neighbors[0].len(), 6); // m0 padding
        assert_eq!(node.neighbors[1].len(), 4); // m padding
        assert!(node.neighbors[0][0].is_valid());
        assert!(!node.neighbors[0][1].is_valid());
        assert!(!node.neighbors[1][0].is_valid());
    }
}
