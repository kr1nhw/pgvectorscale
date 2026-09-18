//! hnswsq core types — the Rust translation of pgvector's `hnsw.h`.
//!
//! `repr(C)` for everything that crosses shared memory or disk (graph state,
//! elements, neighbor arrays, metapage, page opaque, element/neighbor tuples);
//! plain Rust for backend-local bookkeeping (search candidates, build state).
//!
//! Deliberate divergences from the pgvector reference, and only these:
//!
//! * the element holds **one heap TID** instead of `heaptids[HNSW_HEAPTIDS]`
//!   (10 slots).  The old hnswsq engine already handles non-HOT updates with a
//!   single TID (vacuum's callback tombstones dead TIDs, and the new heap TID
//!   gets its own element on the next insert).  Ten slots cost 60 B/element —
//!   about 40 % of an fp8 dim-128 element — for a feature we do not use;
//! * the on-disk element tuple carries a **layout byte + encoded vector**
//!   (`dim × elem_bytes`) instead of a varlena `Vector` datum;
//! * candidate heaps are Rust `BinaryHeap`s with the same comparator plus a
//!   deterministic pointer/offset/TID tie-break (pgvector's pairing heaps
//!   order ties arbitrarily); verified by the recall gates.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use pgrx::pg_sys;

use crate::access_method::distance::DistanceType;
use crate::access_method::hnswsq::quantize::{Codec, HnswPrecision};
use crate::access_method::hnswsq::ptr::HnswPtr;

// ---------------------------------------------------------------------------
// Constants (pgvector hnsw.h, renamed for the port)
// ---------------------------------------------------------------------------

/// Metapage magic: "HNS2" — distinguishes the port's format from the retired
/// rkyv engine ("HNSQ") so old indexes fail with a clear error.
pub const HNSW_MAGIC: u32 = 0x484E_5332;
/// On-disk format version.
pub const HNSW_VERSION: u32 = 1;
/// Page special-area id, same value pgvector uses.
pub const HNSW_PAGE_ID: u16 = 0xFF90;

pub const METAPAGE_BLKNO: pg_sys::BlockNumber = 0;
pub const HEAD_BLKNO: pg_sys::BlockNumber = 1;

/// Pages used as heavyweight lock keys (pgvector's `HNSW_UPDATE_LOCK`/
/// `HNSW_SCAN_LOCK`).  Page locks here never conflict with buffer content
/// locks, which is what makes the read paths advisory-lock-free.
pub const UPDATE_LOCK_PAGE: pg_sys::BlockNumber = 0;
pub const SCAN_LOCK_PAGE: pg_sys::BlockNumber = 1;

pub const ELEMENT_TUPLE_TYPE: u8 = 1;
pub const NEIGHBOR_TUPLE_TYPE: u8 = 2;

/// pgvector's hard dimension cap for `vector` columns.
pub const MAX_DIM: usize = 2000;

/// pgvector's m bounds.
pub const MAX_M: usize = 100;
pub const MIN_M: usize = 2;

/// Metapage entry-update modes.
pub const UPDATE_ENTRY_GREATER: i32 = 1;
pub const UPDATE_ENTRY_ALWAYS: i32 = 2;

/// Per-insert shared-memory margin so a parallel build's allocations never
/// fail at the last byte (pgvector `memoryMargin`).
pub const MEMORY_MARGIN: usize = 1024 * 1024;

/// pgvector caps the in-memory graph at half the address space.
pub const MAX_GRAPH_MEMORY: usize = usize::MAX / 2;

// ---------------------------------------------------------------------------
// On-disk structs (repr(C), byte-exact)
// ---------------------------------------------------------------------------

/// pgvector's `HnswMetaPageData` + the port's additions (`precision`,
/// calibration pointer).  Written directly at the page contents of block 0.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MetaPageData {
    pub magic_number: u32,
    pub version: u32,
    pub dimensions: u32,
    pub m: u16,
    pub ef_construction: u16,
    /// `HnswPrecision` as u8 (plain/ieeefp16/ieeefp8/sq8).
    pub precision: u8,
    /// -1 when no entry point.
    pub entry_level: i16,
    pub entry_blkno: pg_sys::BlockNumber,
    pub entry_offno: pg_sys::OffsetNumber,
    pub insert_page: pg_sys::BlockNumber,
    /// First block of the graph page chain (pgvector's `HNSW_HEAD_BLKNO`
    /// constant, recorded here because the SQ8 calibration chain may occupy
    /// blocks before the graph pages).
    pub graph_head: pg_sys::BlockNumber,
    /// SQ8 calibration chain pointer; invalid for the training-free layouts.
    pub calibration_blkno: pg_sys::BlockNumber,
    pub calibration_offno: pg_sys::OffsetNumber,
}

/// pgvector's `HnswPageOpaqueData`: every non-metapage hnswsq page carries
/// this in its special area.
#[repr(C)]
pub struct PageOpaqueData {
    pub nextblkno: pg_sys::BlockNumber,
    pub unused: u16,
    pub page_id: u16,
}

/// pgvector's `HnswElementTupleData`, adapted: one heaptid, a layout byte,
/// and `dim × elem_bytes` of encoded vector where the varlena was.  The
/// header is padded so the vector starts 8-byte aligned (the `plain` layout
/// feeds it straight to the SIMD kernels).
#[repr(C)]
pub struct ElementTupleData {
    pub type_: u8,
    /// `HnswPrecision` as u8 — validates the tuple against the metapage.
    pub layout: u8,
    pub level: u8,
    pub deleted: u8,
    pub version: u8,
    /// True when encoding clamped any component (no finite lower bound is
    /// provable for the quantized layouts' scan emission).
    pub clamped: u8,
    pub _pad: [u8; 2],
    pub heaptid: pg_sys::ItemPointerData,
    pub _pad2: [u8; 2],
    pub neighbortid: pg_sys::ItemPointerData,
    pub _pad3: [u8; 2],
    // `data`: dim × elem_bytes of encoded vector follows (flexible member).
}

/// pgvector's `HnswNeighborTupleData`, unchanged: `(level + 2) * m` TID slots
/// laid out highest layer first, layer 0 last, invalid-padded.
#[repr(C)]
pub struct NeighborTupleData {
    pub type_: u8,
    pub version: u8,
    pub count: u16,
    // `indextids`: (level + 2) * m ItemPointerData follows (flexible member).
}

/// Byte offset at which the element tuple's vector data starts (aligned to
/// 8 so `plain` vectors are 4-byte aligned for the SIMD fast path).
pub const ELEMENT_TUPLE_VECTOR_OFFSET: usize = std::mem::size_of::<ElementTupleData>();
pub const NEIGHBOR_TUPLE_HEADER_SIZE: usize = std::mem::size_of::<NeighborTupleData>();

/// `HNSW_ELEMENT_TUPLE_SIZE`: MAXALIGN'd total size of an element tuple.
pub fn element_tuple_size(vec_bytes: usize) -> usize {
    unsafe { pg_sys::MAXALIGN(ELEMENT_TUPLE_VECTOR_OFFSET + vec_bytes) }
}

/// `HNSW_NEIGHBOR_TUPLE_SIZE(level, m)`: MAXALIGN'd total size.
pub fn neighbor_tuple_size(level: usize, m: usize) -> usize {
    unsafe {
        pg_sys::MAXALIGN(
            NEIGHBOR_TUPLE_HEADER_SIZE
                + (level + 2) * m * std::mem::size_of::<pg_sys::ItemPointerData>(),
        )
    }
}

/// pgvector's `HNSW_MAX_SIZE`: the largest combined item a page can hold.
pub fn max_page_item_size() -> usize {
    unsafe {
        pg_sys::BLCKSZ as usize
            - pg_sys::MAXALIGN(std::mem::offset_of!(pg_sys::PageHeaderData, pd_linp))
            - pg_sys::MAXALIGN(std::mem::size_of::<PageOpaqueData>())
            - std::mem::size_of::<pg_sys::ItemIdData>()
    }
}

// ---------------------------------------------------------------------------
// In-memory graph structs (repr(C): they cross shared memory in a parallel
// build; the backend-private build uses the same layout with a null base)
// ---------------------------------------------------------------------------

/// pgvector's `HnswCandidate`: one neighbor of an element's list, plus the
/// `closer` flag of the selection heuristic.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Candidate {
    pub element: HnswPtr,
    pub distance: f32,
    pub closer: bool,
}

/// pgvector's `HnswNeighborArray` header.  The items (an array of `lm`
/// [`Candidate`]s) follow inline: `neighbor_array_size(lm)` bytes in total.
#[repr(C)]
pub struct NeighborArray {
    pub length: u32,
    pub closer_set: bool,
    // `items[lm]` follows (flexible member).
}

/// `HNSW_NEIGHBOR_ARRAY_SIZE(lm)`.
pub fn neighbor_array_size(lm: usize) -> usize {
    std::mem::size_of::<NeighborArray>() + lm * std::mem::size_of::<Candidate>()
}

/// Pointer to the items array of a [`NeighborArray`] allocation.
///
/// # Safety
/// `na` must point to an allocation of at least `neighbor_array_size(lm)`
/// bytes for the `lm` the caller then indexes.
#[inline]
pub unsafe fn neighbor_items(na: *mut NeighborArray) -> *mut Candidate {
    (na as *mut u8).add(std::mem::size_of::<NeighborArray>()).cast()
}

/// pgvector's `HnswElementData`, adapted to one heap TID.  The same struct
/// addresses an element in backend memory (absolute pointers), the shared
/// area (relptrs), or — filled from a tuple — an on-disk element.
#[repr(C)]
pub struct Element {
    pub next: HnswPtr,
    pub heaptid: pg_sys::ItemPointerData,
    /// pgvector's `heaptidsLength`: 1 when `heaptid` is valid, 0 when the
    /// element is being deleted or not yet published.
    pub heaptid_set: u8,
    pub level: u8,
    pub deleted: u8,
    pub version: u8,
    /// True when encoding this element's vector clamped any component.
    pub clamped: u8,
    /// Precomputed hash for the in-memory visited tables (pgvector
    /// `PrecomputeHash`).
    pub hash: u64,
    /// Points to an array of `level + 1` `HnswPtr`s, each pointing to the
    /// layer's [`NeighborArray`].
    pub neighbors: HnswPtr,
    pub blkno: pg_sys::BlockNumber,
    pub offno: pg_sys::OffsetNumber,
    pub neighbor_offno: pg_sys::OffsetNumber,
    pub neighbor_page: pg_sys::BlockNumber,
    /// Encoded vector bytes (`dim × elem_bytes`); null until materialized.
    pub value: HnswPtr,
    /// Protects the in-memory element's neighbors/heaptid during a parallel
    /// build; unused for on-disk elements.
    pub lock: pg_sys::LWLock,
}

/// pgvector's `HnswGraph` — one per build (backend-private, or shared).
#[repr(C)]
pub struct Graph {
    pub lock: pg_sys::slock_t,
    pub head: HnswPtr,
    pub indtuples: f64,
    pub entry_lock: pg_sys::LWLock,
    pub entry_wait_lock: pg_sys::LWLock,
    pub entry_point: HnswPtr,
    pub allocator_lock: pg_sys::LWLock,
    pub memory_used: usize,
    pub memory_total: usize,
    pub flush_lock: pg_sys::LWLock,
    pub flushed: bool,
}

/// pgvector's `HnswShared` — the shared region header of a parallel build.
#[repr(C)]
pub struct Shared {
    pub heaprelid: pg_sys::Oid,
    pub indexrelid: pg_sys::Oid,
    pub isconcurrent: bool,
    pub workersdonecv: pg_sys::ConditionVariable,
    pub mutex: pg_sys::slock_t,
    pub nparticipantsdone: i32,
    /// The SQ8 distance mode the leader's session selected (workers are
    /// separate processes and do not see the leader's GUC settings).
    pub sq8_distance_mode: i32,
    pub reltuples: f64,
    pub graph: Graph,
}

// ---------------------------------------------------------------------------
// Backend-local bookkeeping
// ---------------------------------------------------------------------------

/// The support bundle the algorithms thread through — the port of
/// pgvector's `HnswSupport` + type info for the one type we support.
pub struct Support {
    pub dist_type: DistanceType,
    pub precision: HnswPrecision,
    pub codec: Codec,
}

impl Support {
    /// Distance from the f32 query to a stored (encoded) vector.
    #[inline]
    pub fn distance(&self, q: &[f32], bytes: &[u8]) -> f32 {
        self.codec.distance_encoded_direct(self.dist_type, q, bytes)
    }
}

/// One node of the search candidate heaps — pgvector's
/// `HnswSearchCandidate` with a precomputed deterministic tie-break key.
#[derive(Clone, Copy)]
pub struct SearchCandidate {
    pub element: HnswPtr,
    pub distance: f32,
    /// Pointer (base == null), relptr offset (shared), or packed TID (disk).
    pub key: u64,
}

impl std::fmt::Debug for SearchCandidate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchCandidate")
            .field("key", &self.key)
            .field("distance", &self.distance)
            .finish()
    }
}

/// Nearest-first heap entry (pops the smallest distance) — pgvector's
/// `c_node`/discarded heap (`CompareNearestCandidates`).
#[derive(Clone, Copy)]
pub struct NearestItem(pub SearchCandidate);

impl PartialEq for NearestItem {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for NearestItem {}
impl PartialOrd for NearestItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for NearestItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap pops the greatest: invert so the nearest is greatest,
        // breaking distance ties by the smaller key (deterministic).
        other
            .0
            .distance
            .total_cmp(&self.0.distance)
            .then_with(|| other.0.key.cmp(&self.0.key))
    }
}

/// Furthest-first heap entry (pops the largest distance) — pgvector's
/// `w_node` (`CompareFurthestCandidates`).
#[derive(Clone, Copy)]
pub struct FurthestItem(pub SearchCandidate);

impl PartialEq for FurthestItem {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for FurthestItem {}
impl PartialOrd for FurthestItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FurthestItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .distance
            .total_cmp(&other.0.distance)
            .then_with(|| self.0.key.cmp(&other.0.key))
    }
}

pub type CandidateHeap = BinaryHeap<NearestItem>;
pub type FurthestHeap = BinaryHeap<FurthestItem>;

/// One slot of the search layer's `unvisited` scratch buffer — pgvector's
/// `HnswUnvisited` union.
#[derive(Clone, Copy)]
pub enum Unvisited {
    Element(HnswPtr),
    Tid(pg_sys::ItemPointerData),
}

/// pgvector's `visited_hash` union: one open-addressing table keyed by packed
/// u64 (packed TID on disk, relptr offset or pointer in memory), hashed with
/// pgvector's murmur64.
pub struct Visited {
    /// key → slot index + 1; 0 = empty.
    keys: Vec<u64>,
    size: usize,
    mask: usize,
}

impl Visited {
    /// `tidhash_create(ef * m * 2)` — pgvector sizes the initial table from
    /// the expected number of visited ids.
    pub fn new(initial: usize) -> Self {
        let cap = initial.max(4).next_power_of_two();
        Self {
            keys: vec![0; cap],
            size: 0,
            mask: cap - 1,
        }
    }

    /// pgvector's murmurhash64 mixing of a u64 key.
    #[inline]
    fn hash(key: u64) -> u64 {
        let mut h = key;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h
    }

    /// `tidhash_insert` etc.: insert `key`, returning true when it was
    /// already present.  Grows by doubling like simplehash.
    pub fn insert(&mut self, key: u64) -> bool {
        self.insert_key_hash(key, Self::hash(key))
    }

    /// `tidhash_insert_hash`: insert `key` with a precomputed hash (pgvector
    /// stores the hash in the element so the hot path skips re-hashing).
    pub fn insert_key_hash(&mut self, key: u64, hash: u64) -> bool {
        debug_assert_ne!(key, 0, "visited keys must be non-zero");
        if (self.size + 1) * 4 >= self.keys.len() * 3 {
            self.grow();
        }
        let mut idx = (hash as usize) & self.mask;
        loop {
            let slot = unsafe { *self.keys.get_unchecked(idx) };
            if slot == 0 {
                unsafe { *self.keys.get_unchecked_mut(idx) = key };
                self.size += 1;
                return false;
            }
            if slot == key {
                return true;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    fn grow(&mut self) {
        let old = std::mem::take(&mut self.keys);
        let cap = old.len() * 2;
        self.keys = vec![0; cap];
        self.mask = cap - 1;
        self.size = 0;
        for &key in old.iter().filter(|&&k| k != 0) {
            let mut idx = (Self::hash(key) as usize) & self.mask;
            loop {
                let slot = unsafe { *self.keys.get_unchecked(idx) };
                if slot == 0 {
                    unsafe { *self.keys.get_unchecked_mut(idx) = key };
                    self.size += 1;
                    break;
                }
                debug_assert_ne!(slot, key);
                idx = (idx + 1) & self.mask;
            }
        }
    }

    pub fn clear(&mut self) {
        if self.size == 0 {
            return;
        }
        self.keys.fill(0);
        self.size = 0;
    }

    /// Entries currently in the table.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Probe for `key` without inserting (pgvector's `tidhash_lookup`).
    pub fn contains(&self, key: u64) -> bool {
        if self.size == 0 {
            return false;
        }
        let mut idx = (Self::hash(key) as usize) & self.mask;
        loop {
            let slot = unsafe { *self.keys.get_unchecked(idx) };
            if slot == 0 {
                return false;
            }
            if slot == key {
                return true;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    /// Approximate bytes this table occupies (for the iterative scan's
    /// work_mem bound).
    pub fn capacity_bytes(&self) -> usize {
        self.keys.capacity() * std::mem::size_of::<u64>()
    }
}

/// Pack an `ItemPointerData` into the u64 the tid visited table hashes —
/// pgvector's `hash_tid` zeroes a u64 and copies the 6-byte TID into it
/// (block in the low 32 bits, offset in the high 32 on little-endian).
#[inline]
pub fn pack_tid(tid: pg_sys::ItemPointerData) -> u64 {
    pgrx::itemptr::item_pointer_to_u64(tid)
}

/// The size of the `unvisited` scratch buffer a search layer needs.
#[inline]
pub fn unvisited_capacity(m: usize) -> usize {
    2 * m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuple_sizes_and_alignment() {
        // Element tuple header: 24 bytes, vector 8-byte aligned.
        assert_eq!(ELEMENT_TUPLE_VECTOR_OFFSET, 24);
        assert_eq!(ELEMENT_TUPLE_VECTOR_OFFSET % 8, 0);
        // Neighbor tuple header: 4 bytes.
        assert_eq!(NEIGHBOR_TUPLE_HEADER_SIZE, 4);
        // Sizes match the pgvector MAXALIGN formulas.
        assert_eq!(element_tuple_size(512), unsafe { pg_sys::MAXALIGN(24 + 512) });
        assert_eq!(neighbor_tuple_size(2, 16), unsafe { pg_sys::MAXALIGN(4 + 4 * 16 * 6) });
        assert_eq!(neighbor_tuple_size(2, 16), 392);
    }

    #[test]
    fn test_element_layout_offsets() {
        // The heaptid/neighbortid fields must be 2-byte aligned (the
        // ItemPointerData alignment) and the vector offset exactly 24.
        let base = std::mem::offset_of!(ElementTupleData, heaptid);
        let nbase = std::mem::offset_of!(ElementTupleData, neighbortid);
        assert_eq!(base % 2, 0);
        assert_eq!(nbase % 2, 0);
        assert_eq!(ELEMENT_TUPLE_VECTOR_OFFSET, std::mem::size_of::<ElementTupleData>());
    }

    #[test]
    fn test_meta_page_size_fits() {
        // The metapage must fit comfortably in a page's content area.
        assert!(std::mem::size_of::<MetaPageData>() < 512);
        // repr(C) field alignment sanity: magic at 0.
        assert_eq!(std::mem::offset_of!(MetaPageData, magic_number), 0);
    }

    #[test]
    fn test_visited_table() {
        let mut v = Visited::new(8);
        assert!(!v.insert(1));
        assert!(!v.insert(2));
        assert!(v.insert(1));
        assert!(v.insert(2));
        assert!(!v.insert(3));
        // Force growth well past the initial capacity.
        for i in 10..500u64 {
            assert!(!v.insert(i));
        }
        for i in 10..500u64 {
            assert!(v.insert(i));
        }
        v.clear();
        assert!(!v.insert(7));
        assert!(v.insert(7));
    }

    #[test]
    fn test_pack_tid_roundtrip() {
        let mut tid = pg_sys::ItemPointerData::default();
        pgrx::itemptr::item_pointer_set_all(&mut tid, 1234, 56);
        let key = pack_tid(tid);
        assert_eq!(key, (1234u64 << 32) | 56);
        assert_ne!(key, 0);
    }

    #[test]
    fn test_heap_orders() {
        use pgrx::pg_sys::BlockNumber;
        let sc = |d: f32, k: u64| SearchCandidate {
            element: HnswPtr {
                ptr: k as *mut u8,
            },
            distance: d,
            key: k,
        };
        let mut nearest: CandidateHeap = BinaryHeap::new();
        nearest.push(NearestItem(sc(3.0, 3)));
        nearest.push(NearestItem(sc(1.0, 1)));
        nearest.push(NearestItem(sc(2.0, 2)));
        assert_eq!(nearest.pop().unwrap().0.distance, 1.0);
        assert_eq!(nearest.pop().unwrap().0.distance, 2.0);
        assert_eq!(nearest.pop().unwrap().0.distance, 3.0);

        let mut furthest: FurthestHeap = BinaryHeap::new();
        furthest.push(FurthestItem(sc(1.0, 1)));
        furthest.push(FurthestItem(sc(3.0, 3)));
        furthest.push(FurthestItem(sc(2.0, 2)));
        assert_eq!(furthest.pop().unwrap().0.distance, 3.0);
        assert_eq!(furthest.pop().unwrap().0.distance, 2.0);
        assert_eq!(furthest.pop().unwrap().0.distance, 1.0);

        // Distance ties break by key.
        let mut ties: CandidateHeap = BinaryHeap::new();
        ties.push(NearestItem(sc(1.0, 5)));
        ties.push(NearestItem(sc(1.0, 2)));
        assert_eq!(ties.pop().unwrap().0.key, 2);
        assert_eq!(ties.pop().unwrap().0.key, 5);

        let _ = BlockNumber::default();
    }

    #[test]
    fn test_neighbor_array_sizes() {
        assert_eq!(neighbor_array_size(16), std::mem::size_of::<NeighborArray>() + 16 * std::mem::size_of::<Candidate>());
        assert_eq!(neighbor_array_size(32) - neighbor_array_size(16), 16 * std::mem::size_of::<Candidate>());
    }
}
