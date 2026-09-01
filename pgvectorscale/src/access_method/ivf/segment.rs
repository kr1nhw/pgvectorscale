//! IVF segment and per-list header management.
//!
//! A list's entries are stored as one or more **immutable segments**, each a
//! contiguous run of blocks holding a complete struct-of-arrays (SoA) byte
//! stream (see `entry::serialize_entries`).  Once a segment's pointer is
//! published, its blocks are never modified.
//!
//! The per-list **header page** is the single-page atomic publication target:
//! it holds a version + generation and a pointer to an immutable segment-list
//! item.  All writers update the header via [`IvfListHeader::update`], which
//! parses the current header, applies a mutation, and rewrites the page while
//! holding its buffer content lock exclusively — so a concurrent reader (share
//! lock) sees either the old or the new header, never a mix, and concurrent
//! writers serialize their read-modify-write instead of clobbering each other.
//!
//! The `active` field points at the list's unpublished append buffer (writer
//! only); readers ignore it.

use pgrx::pg_sys::BlockNumber;
use pgrx::*;
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{Archive, Deserialize, Serialize};

use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::util::buffer::LockedBufferExclusive;
use crate::util::chain::{ChainItemReader, ChainTapeWriter};
use crate::util::page::{self, PageType, ReadablePage};
use crate::util::ports::{PageGetItem, PageGetItemId};
use crate::util::*;

/// One immutable SoA segment of a list.
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct IvfSegment {
    /// First block of the contiguous entry run.
    pub start_page: BlockNumber,
    /// Number of contiguous blocks in the run.
    pub num_blocks: u32,
    /// Number of entries in the segment.
    pub num_entries: u64,
}

impl IvfSegment {
    pub fn new(start_page: BlockNumber, num_blocks: u32, num_entries: u64) -> Self {
        Self {
            start_page,
            num_blocks,
            num_entries,
        }
    }

    /// Whether this segment has any entries.
    pub fn is_empty(&self) -> bool {
        self.num_entries == 0
    }
}

/// The list's unpublished append buffer (writer-only, invisible to scans).
///
/// Pages are tracked individually because concurrent relation extension by
/// other lists means the buffer's blocks are not guaranteed contiguous —
/// unlike sealed segments, which are written under a held extension lock and
/// bulk-read with `smgrreadv`.
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct IvfActiveBuffer {
    /// Blocks of the buffer in allocation order.
    pub pages: Vec<BlockNumber>,
    /// Number of entries appended so far.
    pub num_entries: u64,
}

impl IvfActiveBuffer {
    pub fn new(first_page: BlockNumber) -> Self {
        Self {
            pages: vec![first_page],
            num_entries: 1,
        }
    }
}

/// Immutable segment-list item (chained, referenced from the header page).
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct IvfSegmentList {
    pub segments: Vec<IvfSegment>,
}

impl IvfSegmentList {
    pub fn new(segments: Vec<IvfSegment>) -> Self {
        Self { segments }
    }

    /// Write a fresh immutable segment-list item, returning its `ItemPointer`
    /// and the number of contiguous blocks the item occupies (so reclamation
    /// can free it later).  The extension lock is held across the write so the
    /// chain's pages are contiguous.
    pub unsafe fn store(&self, index: &PgRelation) -> (ItemPointer, u32) {
        let _ext_lock = crate::util::buffer::LockRelationForExtension::new(index);
        let mut stats = crate::access_method::stats::WriteStats::default();
        let mut tape = ChainTapeWriter::new(index, PageType::IvfSegmentList, &mut stats);
        tape.write_counted(&self.serialize_to_vec())
    }

    /// Write the item into a single reserved block (reclaimed block),
    /// returning its pointer and block count (always 1).
    pub unsafe fn store_at(&self, index: &PgRelation, block: BlockNumber) -> (ItemPointer, u32) {
        let mut stats = crate::access_method::stats::WriteStats::default();
        let mut tape = ChainTapeWriter::reinit(index, PageType::IvfSegmentList, &mut stats, block);
        let (ptr, blocks) = tape.write_counted(&self.serialize_to_vec());
        assert_eq!(blocks, 1, "segment-list item must fit one page");
        (ptr, blocks)
    }

    /// Whether the serialized item fits a single page (for the reserved-block
    /// path; callers fall back to `store` — relation extension — otherwise).
    pub fn fits_one_page(&self) -> bool {
        let bytes = self.serialize_to_vec();
        bytes.len() < crate::util::page::tsv_fresh_page_capacity()
    }

    /// Load a segment-list item from the given pointer.
    pub fn load(index: &PgRelation, pointer: ItemPointer) -> IvfSegmentList {
        unsafe {
            let mut stats = crate::access_method::stats::WriteStats::default();
            let mut tape = ChainItemReader::new(index, PageType::IvfSegmentList, &mut stats);

            let mut buf: Vec<u8> = Vec::new();
            for item in tape.read(pointer) {
                buf.extend_from_slice(item.get_data_slice());
            }
            rkyv::from_bytes::<IvfSegmentList>(&buf)
                .unwrap_or_else(|e| panic!("IVF: segment-list parse failed ({} bytes): {:?}", buf.len(), e))
        }
    }
}

/// A contiguous block range made available for reuse by reclamation.
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct IvfFreeRange {
    pub start_block: BlockNumber,
    pub num_blocks: u32,
}

/// The free block list (chained, referenced from the meta page).  Writers push
/// retired ranges; the allocator pops them for reuse.  All access happens
/// under the relation ExclusiveLock (see `IvfMetaPage::reclaim_ranges` /
/// `IvfMetaPage::allocate_range`), so it is safe against in-flight smgr reads.
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct IvfFreeList {
    pub ranges: Vec<IvfFreeRange>,
}

impl IvfFreeList {
    pub fn new() -> Self {
        Self { ranges: Vec::new() }
    }

    /// Write a fresh free-list item, returning its pointer and the number of
    /// contiguous blocks it occupies (so the previous item's blocks can be
    /// retired into the new item).  The extension lock is held across the
    /// write so the chain's pages are contiguous.
    pub unsafe fn store_counted(&self, index: &PgRelation) -> (ItemPointer, u32) {
        let _ext_lock = crate::util::buffer::LockRelationForExtension::new(index);
        let mut stats = crate::access_method::stats::WriteStats::default();
        let mut tape = ChainTapeWriter::new(index, PageType::IvfFreeList, &mut stats);
        tape.write_counted(&self.serialize_to_vec())
    }

    pub fn load(index: &PgRelation, pointer: ItemPointer) -> IvfFreeList {
        unsafe {
            let mut stats = crate::access_method::stats::WriteStats::default();
            let mut tape = ChainItemReader::new(index, PageType::IvfFreeList, &mut stats);

            let mut buf: Vec<u8> = Vec::new();
            for item in tape.read(pointer) {
                buf.extend_from_slice(item.get_data_slice());
            }
            rkyv::from_bytes::<IvfFreeList>(&buf)
                .unwrap_or_else(|e| panic!("IVF: free-list parse failed ({} bytes): {:?}", buf.len(), e))
        }
    }
}

/// Per-list header page: the single-page atomic publication target.
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct IvfListHeader {
    /// Monotonic version bumped on every header swap.
    pub version: u64,
    /// Generation (bumped on compaction) for future reclamation.
    pub generation: u64,
    /// Pointer to the immutable segment-list item readers walk.
    pub segment_list: ItemPointer,
    /// Number of contiguous blocks the segment-list item occupies (for
    /// reclamation when the item is retired by a later swap).
    pub segment_list_blocks: u32,
    /// Writer-only pointer to the unpublished append buffer.
    pub active: Option<IvfActiveBuffer>,
}

impl IvfListHeader {
    pub fn new(segment_list: ItemPointer, segment_list_blocks: u32) -> Self {
        Self {
            version: 0,
            generation: 0,
            segment_list,
            segment_list_blocks,
            active: None,
        }
    }

    /// Allocate a fresh header page and write it (build path), returning the
    /// `ItemPointer` to the header page itself.  The header is the page's only
    /// item (offset 1) so it always fits a single page.
    pub unsafe fn store_new(&self, index: &PgRelation) -> ItemPointer {
        let mut page = crate::util::page::WritablePage::new(index, PageType::IvfListHeader);
        let block = page.get_block_number();
        let off = page.add_item(&self.serialize_to_vec());
        page.commit();
        ItemPointer::new(block, off)
    }

    /// Load a header page from the given pointer (share content lock; the
    /// single-page read is atomic against a concurrent writer).
    pub fn load(index: &PgRelation, pointer: ItemPointer) -> IvfListHeader {
        unsafe {
            let page = ReadablePage::read(index, pointer.block_number);
            assert!(page.get_type() == PageType::IvfListHeader);
            let item = page.get_item_unchecked(pointer.offset);
            rkyv::from_bytes::<IvfListHeader>(item.get_data_slice())
                .unwrap_or_else(|e| panic!("IVF: list-header parse failed: {:?}", e))
        }
    }

    /// Parse the header from a buffer already pinned and exclusively locked by
    /// the caller (read-modify-write path).
    unsafe fn parse_buffer(buffer: &LockedBufferExclusive) -> IvfListHeader {
        let page = pg_sys::BufferGetPage(**buffer);
        let item_id = PageGetItemId(page, 1);
        let item = PageGetItem(page, item_id);
        let len = (*item_id).lp_len() as usize;
        rkyv::from_bytes::<IvfListHeader>(std::slice::from_raw_parts(item as *const u8, len))
            .unwrap_or_else(|e| panic!("IVF: list-header buffer parse failed ({} bytes): {:?}", len, e))
    }

    /// Read-modify-write the header page of list `block` atomically.
    ///
    /// The header buffer is pinned and exclusively content-locked for the
    /// duration of `f`, which receives the *current* header (parsed under the
    /// lock).  `f` may perform arbitrary work (e.g. append to the active
    /// buffer, seal a segment) while the lock excludes concurrent writers of
    /// this list; after `f` returns, the header is rewritten in place
    /// (WAL-logged) and the lock is released.  Returns `f`'s result.
    pub unsafe fn update<R, F: FnOnce(&mut IvfListHeader) -> R>(
        index: &PgRelation,
        block: BlockNumber,
        f: F,
    ) -> R {
        let buffer = LockedBufferExclusive::read(index, block);
        let mut header = Self::parse_buffer(&buffer);
        let result = f(&mut header);
        let bytes = header.serialize_to_vec();
        // The header must remain a single page for atomic publication.  The
        // active-buffer pages list is the only unbounded field; with the
        // default ivf.seal_threshold it is ~20-80 entries (well under a page).
        assert!(
            bytes.len() < pg_sys::BLCKSZ as usize / 2,
            "IVF list header exceeded half a page ({} bytes); reduce ivf.seal_threshold",
            bytes.len()
        );
        page::write_single_item_page_locked(index, &buffer, PageType::IvfListHeader, &bytes);
        result
    }
}
