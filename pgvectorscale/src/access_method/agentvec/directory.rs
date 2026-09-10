//! The AgentVec segment directory and per-segment state.
//!
//! Two levels of metadata, with different write frequencies:
//!
//! ```text
//! AgentVecDirectory  (block 1, chained item, COPY-ON-WRITE)
//!     one AgentVecSegmentMeta per physical segment
//!         │
//!         └── header: ItemPointer ──> AgentVecSegmentHeader (one page, in-place RMW)
//!                                        the segment's published runs + active chain
//! ```
//!
//! * The **directory** is republished copy-on-write whenever the set of
//!   segments changes (a segment is created, sealed, queued, or retired).
//!   Once published, an item is immutable, so a reader that captured it can
//!   keep walking it while writers publish a new one.  Retiring the previous
//!   item is a reclamation concern (phase 10); until then a superceded
//!   directory item's blocks simply leak, which is bounded by one item per
//!   segment-lifecycle event.
//! * A **segment header** is the single page that owns everything a writer
//!   mutates on the hot path (the active chain pointer and, on seal, the
//!   published run list).  It is rewritten in place under its buffer content
//!   lock, so concurrent appenders serialize their read-modify-write instead
//!   of clobbering each other, and a concurrent reader (share lock) sees
//!   either the old or the new page, never a mix.

use pgrx::pg_sys::BlockNumber;
use pgrx::*;
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{Archive, Deserialize, Serialize};

use crate::access_method::agentvec::flat::{FlatActive, FlatRun};
use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::util::buffer::LockedBufferExclusive;
use crate::util::chain::{ChainItemReader, ChainTapeWriter};
use crate::util::page::{self, PageType, ReadablePage, WritablePage};
use crate::util::ports::{PageGetItem, PageGetItemId};
use crate::util::*;

/// Stable identity of a segment within one logical index.
pub type SegmentId = u64;

/// The level a segment belongs to (design §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentLevel {
    /// Mutable write cache; absorbs inserts and is immediately searchable.
    Hot = 0,
    /// Recently consolidated, medium sized, still relatively active.
    Warm = 1,
    /// Large, stable, maximally compressed knowledge-base data.
    Cold = 2,
}

impl SegmentLevel {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => SegmentLevel::Hot,
            1 => SegmentLevel::Warm,
            2 => SegmentLevel::Cold,
            _ => panic!("Unknown SegmentLevel {}", value),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SegmentLevel::Hot => "hot",
            SegmentLevel::Warm => "warm",
            SegmentLevel::Cold => "cold",
        }
    }
}

/// Lifecycle state of a segment.
///
/// Only `Published` segments are visible to the router and to scans; a
/// partially built segment is never published (design §32.2).  `Building` is
/// carried inside the directory entry created for a segment under
/// construction; because the directory item is immutable once published, a
/// reader that captured an earlier item never observes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentState {
    /// Being built; not yet searchable.
    Building = 0,
    /// Fully built and searchable.
    Published = 1,
    /// Sealed and handed to asynchronous maintenance (design §10).
    QueuedForMigration = 2,
    /// Consolidation is building a replacement generation.
    Retiring = 3,
    /// No longer referenced by any reader; blocks are reclaimable.
    Retired = 4,
}

impl SegmentState {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => SegmentState::Building,
            1 => SegmentState::Published,
            2 => SegmentState::QueuedForMigration,
            3 => SegmentState::Retiring,
            4 => SegmentState::Retired,
            _ => panic!("Unknown SegmentState {}", value),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SegmentState::Building => "building",
            SegmentState::Published => "published",
            SegmentState::QueuedForMigration => "queued_for_migration",
            SegmentState::Retiring => "retiring",
            SegmentState::Retired => "retired",
        }
    }

    /// Whether scans may read this segment.
    pub fn is_searchable(self) -> bool {
        matches!(
            self,
            SegmentState::Published | SegmentState::QueuedForMigration | SegmentState::Retiring
        )
    }
}

/// Which ANN implementation backs the segment's payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentAlgorithm {
    /// Exact exhaustive scan over stored f32 vectors: the correctness
    /// baseline, and phase 1's only executor.
    Flat = 0,
    /// Graph index (pgvector-compatible HNSW) — phase 2.
    Hnsw = 1,
    /// IVF with centroid-aware RaBitQ codes — phase 3.
    IvfRaBitQ = 2,
}

impl SegmentAlgorithm {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => SegmentAlgorithm::Flat,
            1 => SegmentAlgorithm::Hnsw,
            2 => SegmentAlgorithm::IvfRaBitQ,
            _ => panic!("Unknown SegmentAlgorithm {}", value),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SegmentAlgorithm::Flat => "flat",
            SegmentAlgorithm::Hnsw => "hnsw",
            SegmentAlgorithm::IvfRaBitQ => "ivf_rabitq",
        }
    }
}

/// Who owns (and therefore may rewrite) the segment's physical storage.
///
/// `External` segments are physical indexes owned by another access method
/// that AgentVec adopts rather than builds; the field exists from the first
/// version of the format so adoption needs no on-disk migration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentOwnership {
    /// Blocks inside the AgentVec relation, written by this AM.
    Owned = 0,
    /// A separate physical index relation managed by another AM.
    External = 1,
}

impl SegmentOwnership {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => SegmentOwnership::Owned,
            1 => SegmentOwnership::External,
            _ => panic!("Unknown SegmentOwnership {}", value),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SegmentOwnership::Owned => "owned",
            SegmentOwnership::External => "external",
        }
    }
}

/// One directory entry: the stable identity of a segment plus the metadata a
/// router needs without touching the segment's payload.
///
/// The physical representation behind `header` may be replaced by a new
/// generation at any time; the `segment_id` is what stays stable (design §6).
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct AgentVecSegmentMeta {
    /// Stable segment identity.
    pub segment_id: SegmentId,
    /// Logical group/topic this segment belongs to (design §5).
    pub group_id: u32,
    /// `SegmentLevel` as u8.
    pub level: u8,
    /// `SegmentState` as u8.
    pub state: u8,
    /// `SegmentAlgorithm` as u8.
    pub algorithm: u8,
    /// `SegmentOwnership` as u8.
    pub ownership: u8,
    /// Bumped whenever the segment's physical representation is replaced.
    pub generation: u64,
    /// Monotonic epoch the segment was published at (design §7.1).
    pub epoch: u64,
    /// The segment's header page: the atomic publication point for its runs.
    pub header: ItemPointer,
    /// Root of the segment's code/vector payload (WARM/COLD, phase 3).
    pub code_root: ItemPointer,
    /// Root of the segment's posting lists (WARM/COLD, phase 3).
    pub posting_root: ItemPointer,
    /// Entries physically present, including tombstoned ones.
    pub vector_count: u64,
    /// Entries not tombstoned.
    pub live_count: u64,
    /// Tombstoned entries.
    pub dead_count: u64,
    /// Distance metric (`DistanceType` as u16).
    pub metric: u16,
    /// Vector dimensions of this segment's payload.
    pub dimension: u32,
    /// Format version of the payload, so a future format can coexist.
    pub format_version: u32,
    /// OID of the physical index relation (0 = this AgentVec relation).
    /// Non-zero only for adopted `External` segments.
    pub physical_index_oid: u32,
    /// OID of the access method backing the payload (0 = agentvec itself).
    pub access_method_oid: u32,
    /// OID of the operator class the segment was built for.
    pub opclass_oid: u32,
}

impl AgentVecSegmentMeta {
    /// Create a directory entry for a freshly allocated segment.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        segment_id: SegmentId,
        group_id: u32,
        level: SegmentLevel,
        state: SegmentState,
        algorithm: SegmentAlgorithm,
        ownership: SegmentOwnership,
        epoch: u64,
        header: ItemPointer,
        metric: u16,
        dimension: u32,
        format_version: u32,
    ) -> Self {
        Self {
            segment_id,
            group_id,
            level: level as u8,
            state: state as u8,
            algorithm: algorithm as u8,
            ownership: ownership as u8,
            generation: 0,
            epoch,
            header,
            code_root: ItemPointer::new_invalid(),
            posting_root: ItemPointer::new_invalid(),
            vector_count: 0,
            live_count: 0,
            dead_count: 0,
            metric,
            dimension,
            format_version,
            physical_index_oid: 0,
            access_method_oid: 0,
            opclass_oid: 0,
        }
    }

    pub fn level(&self) -> SegmentLevel {
        SegmentLevel::from_u8(self.level)
    }

    pub fn state(&self) -> SegmentState {
        SegmentState::from_u8(self.state)
    }

    pub fn algorithm(&self) -> SegmentAlgorithm {
        SegmentAlgorithm::from_u8(self.algorithm)
    }

    pub fn ownership(&self) -> SegmentOwnership {
        SegmentOwnership::from_u8(self.ownership)
    }

    /// Whether a scan may read this segment.
    pub fn is_searchable(&self) -> bool {
        self.state().is_searchable()
    }
}

/// The segment directory: an immutable-once-published chained item listing
/// every physical segment of the logical index.
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct AgentVecDirectory {
    pub segments: Vec<AgentVecSegmentMeta>,
}

impl AgentVecDirectory {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    /// Number of directory entries.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Look up a segment by its stable id.
    pub fn get(&self, segment_id: SegmentId) -> Option<&AgentVecSegmentMeta> {
        self.segments.iter().find(|s| s.segment_id == segment_id)
    }

    /// Look up a segment by its stable id, mutably.
    pub fn get_mut(&mut self, segment_id: SegmentId) -> Option<&mut AgentVecSegmentMeta> {
        self.segments.iter_mut().find(|s| s.segment_id == segment_id)
    }

    /// The segments a scan should read, in directory order.
    pub fn searchable(&self) -> impl Iterator<Item = &AgentVecSegmentMeta> {
        self.segments.iter().filter(|s| s.is_searchable())
    }

    /// Publish this directory as a fresh immutable chained item, returning its
    /// pointer and the number of contiguous blocks it occupies.
    ///
    /// The relation extension lock is held across the write so the item's
    /// pages are contiguous, which is what makes the block count meaningful
    /// for a later reclamation pass.
    pub unsafe fn store(&self, index: &PgRelation) -> (ItemPointer, u32) {
        let _ext_lock = crate::util::buffer::LockRelationForExtension::new(index);
        let mut stats = crate::access_method::stats::WriteStats::default();
        let mut tape = ChainTapeWriter::new(index, PageType::AgentVecDirectory, &mut stats);
        tape.write_counted(&self.serialize_to_vec())
    }

    /// Read a published directory item.
    pub fn load(index: &PgRelation, pointer: ItemPointer) -> AgentVecDirectory {
        unsafe {
            let mut stats = crate::access_method::stats::WriteStats::default();
            let mut tape = ChainItemReader::new(index, PageType::AgentVecDirectory, &mut stats);
            let mut buf: Vec<u8> = Vec::new();
            for item in tape.read(pointer) {
                buf.extend_from_slice(item.get_data_slice());
            }
            rkyv::from_bytes::<AgentVecDirectory>(&buf).unwrap_or_else(|e| {
                panic!(
                    "agentvec: directory parse failed ({} bytes): {:?}",
                    buf.len(),
                    e
                )
            })
        }
    }
}

impl Default for AgentVecDirectory {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-segment mutable state, stored as the only item of the segment's header
/// page.
///
/// This is where the segment's *published* content lives: the runs whose chain
/// has been frozen (`sealed`) and the chain still being appended to
/// (`active`).  Sealing is therefore O(1) — the chain is already durable, so
/// sealing only moves its descriptor from `active` to `sealed` — which is
/// exactly the "small metadata change" the HOT lifecycle requires (design §10).
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct AgentVecSegmentHeader {
    /// Bumped on every publication (seal).
    pub version: u64,
    /// Bumped when the segment's physical representation is replaced.
    pub generation: u64,
    /// `SegmentLevel` as u8.
    pub level: u8,
    /// `SegmentState` as u8.
    pub state: u8,
    /// `SegmentAlgorithm` as u8.
    pub algorithm: u8,
    /// Frozen chains, oldest first.
    pub sealed: Vec<FlatRun>,
    /// The chain appenders are still writing to (writer-visible only; scans
    /// read it too, since its entries are committed row by row).
    pub active: Option<FlatActive>,
    /// Entries appended, including tombstoned ones.
    pub num_entries: u64,
    /// Tombstoned entries.
    pub dead_entries: u64,
}

impl AgentVecSegmentHeader {
    /// A header for a brand-new, empty segment.
    pub fn new(level: SegmentLevel, algorithm: SegmentAlgorithm) -> Self {
        Self {
            version: 0,
            generation: 0,
            level: level as u8,
            state: SegmentState::Published as u8,
            algorithm: algorithm as u8,
            sealed: Vec::new(),
            active: None,
            num_entries: 0,
            dead_entries: 0,
        }
    }

    pub fn level(&self) -> SegmentLevel {
        SegmentLevel::from_u8(self.level)
    }

    pub fn state(&self) -> SegmentState {
        SegmentState::from_u8(self.state)
    }

    pub fn algorithm(&self) -> SegmentAlgorithm {
        SegmentAlgorithm::from_u8(self.algorithm)
    }

    /// Entries that a scan should evaluate.
    pub fn live_entries(&self) -> u64 {
        self.num_entries.saturating_sub(self.dead_entries)
    }

    /// Every chain a scan must read: frozen runs first, then the active one.
    pub fn chain_starts(&self) -> Vec<BlockNumber> {
        let mut out: Vec<BlockNumber> = self.sealed.iter().map(|r| r.first_page).collect();
        if let Some(active) = self.active.as_ref() {
            out.push(active.first_page);
        }
        out
    }

    /// Freeze the active chain, making its entries part of the published run
    /// list.  Returns true when a chain was sealed.
    ///
    /// This performs no data movement: the entries are already durable on
    /// their pages, and the pages are already immutable (nothing appends to a
    /// chain once it is frozen).
    pub fn seal_active(&mut self) -> bool {
        match self.active.take() {
            Some(active) => {
                self.sealed.push(FlatRun::new(active.first_page, active.num_entries));
                self.version += 1;
                true
            }
            None => false,
        }
    }

    /// Write a fresh header page (build path) and return its pointer.
    pub unsafe fn store_new(&self, index: &PgRelation) -> ItemPointer {
        let mut page = WritablePage::new(index, PageType::AgentVecSegmentHeader);
        let block = page.get_block_number();
        let off = page.add_item(&self.serialize_to_vec());
        page.commit();
        ItemPointer::new(block, off)
    }

    /// Load a header from its pointer (share content lock; the single-page
    /// read is atomic against a concurrent writer's rewrite).
    pub fn load(index: &PgRelation, pointer: ItemPointer) -> AgentVecSegmentHeader {
        unsafe {
            let page = ReadablePage::read(index, pointer.block_number);
            assert!(
                page.get_type() == PageType::AgentVecSegmentHeader,
                "agentvec: block {} is not a segment header",
                pointer.block_number
            );
            let item = page.get_item_unchecked(pointer.offset);
            rkyv::from_bytes::<AgentVecSegmentHeader>(item.get_data_slice())
                .unwrap_or_else(|e| panic!("agentvec: segment header parse failed: {:?}", e))
        }
    }

    /// Parse a header out of a buffer the caller already holds exclusively.
    unsafe fn parse_buffer(buffer: &LockedBufferExclusive) -> AgentVecSegmentHeader {
        let page = pg_sys::BufferGetPage(**buffer);
        let item_id = PageGetItemId(page, 1);
        let item = PageGetItem(page, item_id);
        let len = (*item_id).lp_len() as usize;
        rkyv::from_bytes::<AgentVecSegmentHeader>(std::slice::from_raw_parts(
            item as *const u8,
            len,
        ))
        .unwrap_or_else(|e| {
            panic!(
                "agentvec: segment header buffer parse failed ({} bytes): {:?}",
                len, e
            )
        })
    }

    /// Read-modify-write the header page atomically.
    ///
    /// The buffer is pinned and exclusively content-locked for the duration of
    /// `f`, which receives the *current* header parsed under that lock.  The
    /// closure may append to the active chain or seal it; the header is
    /// rewritten (WAL-logged) before the lock is released, so a concurrent
    /// reader sees either the old or the new state.
    pub unsafe fn update<R, F: FnOnce(&mut AgentVecSegmentHeader) -> R>(
        index: &PgRelation,
        block: BlockNumber,
        f: F,
    ) -> R {
        let buffer = LockedBufferExclusive::read(index, block);
        let mut header = Self::parse_buffer(&buffer);
        let result = f(&mut header);
        let bytes = header.serialize_to_vec();
        // The header must stay a single page: the run list is the only
        // unbounded field, and it grows by one entry per seal (i.e. per
        // `hot_segment_max_rows` inserts).
        assert!(
            bytes.len() < pg_sys::BLCKSZ as usize / 2,
            "agentvec: segment header exceeded half a page ({} bytes)",
            bytes.len()
        );
        page::write_single_item_page_locked(
            index,
            &buffer,
            PageType::AgentVecSegmentHeader,
            &bytes,
        );
        result
    }
}
