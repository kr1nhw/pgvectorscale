//! The `FLAT` segment payload: an append-only chain of entry pages.
//!
//! Every page in a chain has the same shape:
//!
//! ```text
//! offset 1 : page link      u32  block number of the next page (0xFFFFFFFF = end)
//! offset 2+: entries        [heap block u32][heap offset u16][state u8][vector f32 x dim]
//! ```
//!
//! Properties this layout is built on:
//!
//! * **Append-only.**  Items are only ever added, never moved or resized.  The
//!   single exception is the `state` byte, flipped from `STATE_LIVE` to
//!   `STATE_DEAD` by `ambulkdelete` under the page's content lock; a scan that
//!   holds the same page's share lock therefore observes one value or the
//!   other, never a torn entry.
//! * **Chains link strictly forward.**  A new page is always produced by
//!   extending the relation, so a chain can be walked to its end without any
//!   metadata beyond its first block.
//! * **Sealing is free.**  Freezing a chain changes no page: the descriptor
//!   moves from the segment header's `active` field to its `sealed` list, so
//!   the HOT lifecycle's "seal + start a new HOT segment" costs one metadata
//!   write (design §10).
//!
//! `FLAT` is deliberately not an ANN structure: it is the exact baseline that
//! later algorithms are measured against, and it is what makes a newly
//! committed row immediately searchable with no background step.

use pgrx::pg_sys::{BlockNumber, ForkNumber, InvalidBlockNumber, OffsetNumber};
use pgrx::*;

use crate::util::page::{PageType, ReadablePage, WritablePage};
use crate::util::ports::{PageGetItem, PageGetItemId, PageGetMaxOffsetNumber};
use crate::util::*;

/// Offset of the page-link item.
pub const LINK_OFFSET: OffsetNumber = 1;
/// First offset an entry can occupy.
pub const FIRST_ENTRY_OFFSET: OffsetNumber = 2;
/// Width of the page-link item.
const LINK_LEN: usize = 4;
/// Width of the packed heap TID (block u32 + offset u16).
const TID_LEN: usize = 6;

/// Offset of the entry state byte.
const STATE_OFFSET: usize = TID_LEN;
/// Offset of the entry vector payload.
const VECTOR_OFFSET: usize = TID_LEN + 1;

/// Entry state: the row is (or may be) live in the heap.
pub const STATE_LIVE: u8 = 0;
/// Entry state: `ambulkdelete` reported the row dead.
pub const STATE_DEAD: u8 = 1;

/// Bytes one entry occupies for a given number of dimensions.
pub fn entry_len(dim: usize) -> usize {
    VECTOR_OFFSET + dim * std::mem::size_of::<f32>()
}

/// A frozen chain of entry pages.
#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Deserialize, rkyv::Serialize)]
#[archive(check_bytes)]
pub struct FlatRun {
    /// First block of the chain; the rest are reached through page links.
    pub first_page: BlockNumber,
    /// Entries in the chain.
    pub num_entries: u64,
}

impl FlatRun {
    pub fn new(first_page: BlockNumber, num_entries: u64) -> Self {
        Self {
            first_page,
            num_entries,
        }
    }
}

/// The chain appenders are still writing to.
#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Deserialize, rkyv::Serialize)]
#[archive(check_bytes)]
pub struct FlatActive {
    /// First block of the chain.
    pub first_page: BlockNumber,
    /// Block entries are appended to.
    pub last_page: BlockNumber,
    /// Entries appended so far.
    pub num_entries: u64,
}

/// Serialize one entry.
///
/// The layout is fixed-width and endian-explicit (little-endian, the same
/// encoding PostgreSQL uses on every supported platform for `PageAddItem`d
/// payloads) so it can be written straight into a page and validated on read.
pub fn encode_entry(tid: ItemPointer, vector: &[f32], state: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(entry_len(vector.len()));
    out.extend_from_slice(&tid.block_number.to_le_bytes());
    out.extend_from_slice(&tid.offset.to_le_bytes());
    out.push(state);
    for v in vector {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Decode the heap TID of an entry.
pub fn decode_tid(bytes: &[u8]) -> ItemPointer {
    debug_assert!(bytes.len() >= VECTOR_OFFSET);
    let block = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let offset = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
    ItemPointer::new(block, offset)
}

/// Decode the state byte of an entry.
pub fn decode_state(bytes: &[u8]) -> u8 {
    bytes[STATE_OFFSET]
}

/// Decode an entry's vector into a caller-owned buffer (reused across
/// entries), returning the number of dimensions decoded.
///
/// The vector bytes are copied through `from_le_bytes` rather than aliased as
/// `&[f32]`: page items are not guaranteed 4-byte aligned, so a cast would be
/// undefined behaviour.
pub fn decode_vector_into(bytes: &[u8], out: &mut Vec<f32>) -> usize {
    let vector_bytes = &bytes[VECTOR_OFFSET..];
    debug_assert_eq!(vector_bytes.len() % 4, 0);
    out.clear();
    out.reserve(vector_bytes.len() / 4);
    for c in vector_bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    out.len()
}

/// Read the forward link of a page (the block holding the next page, or
/// `InvalidBlockNumber` at the end of a chain).
unsafe fn read_link(page: &ReadablePage) -> BlockNumber {
    let page_ptr = **page;
    let item_id = PageGetItemId(page_ptr, LINK_OFFSET);
    assert!(
        (*item_id).lp_len() as usize == LINK_LEN,
        "agentvec: flat page {} has a malformed link item",
        pg_sys::BufferGetBlockNumber(**page.get_buffer())
    );
    let item = PageGetItem(page_ptr, item_id) as *const u8;
    let bytes = std::slice::from_raw_parts(item, LINK_LEN);
    u32::from_le_bytes(bytes.try_into().unwrap())
}

/// Point a page's link at the next page in its chain.
unsafe fn write_link(index: &PgRelation, block: BlockNumber, next: BlockNumber) {
    let mut buffer = ItemPointer::new(block, LINK_OFFSET).modify_bytes(index);
    {
        let data = buffer.get_data_slice();
        assert!(data.len() >= LINK_LEN);
        data[..LINK_LEN].copy_from_slice(&next.to_le_bytes());
    }
    buffer.commit();
}

/// Initialize a fresh page as a chain page carrying one entry, returning its
/// block number.
unsafe fn write_new_page(index: &PgRelation, entry: &[u8]) -> BlockNumber {
    let mut page = WritablePage::new(index, PageType::AgentVecFlatPage);
    let block = page.get_block_number();
    page.add_item(&InvalidBlockNumber.to_le_bytes());
    page.add_item(entry);
    page.commit();
    block
}

/// Append one encoded entry to a chain, creating the chain when `active` is
/// `None`.
///
/// Called from inside the segment header's read-modify-write closure: the
/// header's content lock is held for the whole append, which is what
/// serializes appenders to the same segment.  A new page is obtained by
/// extending the relation and is linked from the previous tail *after* it is
/// durable, so a walker that reads the tail early simply stops early instead
/// of following a half-written link.
pub unsafe fn append_entry(
    index: &PgRelation,
    active: Option<FlatActive>,
    entry: &[u8],
) -> FlatActive {
    match active {
        Some(mut active) => {
            let tail_block = active.last_page;
            let mut tail = WritablePage::modify(index, tail_block);
            if tail.get_aligned_free_space() >= entry.len() {
                tail.add_item(entry);
                tail.commit();
                active.num_entries += 1;
                active
            } else {
                // Abort the (unmodified) tail page before extending.
                drop(tail);
                let new_block = write_new_page(index, entry);
                write_link(index, tail_block, new_block);
                active.last_page = new_block;
                active.num_entries += 1;
                active
            }
        }
        None => {
            let block = write_new_page(index, entry);
            FlatActive {
                first_page: block,
                last_page: block,
                num_entries: 1,
            }
        }
    }
}

/// Visit every entry of a chain, oldest first, in the order the entries were
/// appended.
///
/// The callback receives the entry's page block and offset (so the entry can
/// be tombstoned later) and its raw bytes.  Pages are read under a share
/// content lock, so a concurrent appender cannot expose a partially written
/// item.
pub unsafe fn for_each_entry<F>(index: &PgRelation, chain_start: BlockNumber, mut f: F)
where
    F: FnMut(BlockNumber, OffsetNumber, &[u8]),
{
    let rel_blocks =
        pg_sys::RelationGetNumberOfBlocksInFork(index.as_ptr(), ForkNumber::MAIN_FORKNUM) as u64;
    let mut block = chain_start;
    let mut visited: u64 = 0;
    while block != InvalidBlockNumber {
        visited += 1;
        assert!(
            visited <= rel_blocks + 1,
            "agentvec: flat chain at block {} is corrupt (link cycle?)",
            chain_start
        );
        let page = ReadablePage::read(index, block);
        let next = read_link(&page);
        let max_offset = PageGetMaxOffsetNumber(*page);
        for offset in FIRST_ENTRY_OFFSET..=max_offset as OffsetNumber {
            let item_id = PageGetItemId(*page, offset);
            if (*item_id).lp_len() == 0 {
                continue; // unused line pointer
            }
            let item = PageGetItem(*page, item_id) as *const u8;
            let len = (*item_id).lp_len() as usize;
            let bytes = std::slice::from_raw_parts(item, len);
            f(block, offset, bytes);
        }
        block = next;
    }
}

/// Tombstone one entry in place.
///
/// The page's content lock is taken exclusively, so this cannot interleave
/// with a scan reading the same page, and the change is WAL-logged.
pub unsafe fn mark_dead(index: &PgRelation, block: BlockNumber, offset: OffsetNumber) {
    let mut buffer = ItemPointer::new(block, offset).modify_bytes(index);
    {
        let data = buffer.get_data_slice();
        assert_eq!(
            data[STATE_OFFSET], STATE_LIVE,
            "agentvec: entry {}/{} tombstoned twice",
            block, offset
        );
        data[STATE_OFFSET] = STATE_DEAD;
    }
    buffer.commit();
}
