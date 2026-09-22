//! IVF list directory management.
//!
//! The list directory (Page 1) stores metadata for each inverted list including
//! pointers to entry pages and tuple counts.

use pgrx::pg_sys::BlockNumber;
use pgrx::*;
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{Archive, Deserialize, Serialize};

use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::util::chain::{ChainItemReader, ChainTapeWriter};
use crate::util::page::PageType;
use crate::util::*;

const LIST_DIRECTORY_BLOCK_NUMBER: BlockNumber = 1;
const LIST_DIRECTORY_OFFSET: pgrx::pg_sys::OffsetNumber = 1;

/// Metadata for a single inverted list.
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct IvfListMetadata {
    /// Pointer to this list's header page (atomic publication target).
    pub header: ItemPointer,
    /// Offset into centroids page for this list's centroid
    pub centroid_offset: u32,
    /// Number of tuples in this list
    pub num_tuples: u64,
}

impl IvfListMetadata {
    /// Create new list metadata with default values.
    pub fn new(centroid_offset: u32) -> Self {
        Self {
            header: ItemPointer::new_invalid(),
            centroid_offset,
            num_tuples: 0,
        }
    }

    /// Check if this list has any entries.
    pub fn is_empty(&self) -> bool {
        self.num_tuples == 0
    }
}

/// IVF list directory containing metadata for all inverted lists.
/// Stored on Page 1.
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct IvfListDirectory {
    /// Metadata for each inverted list
    pub lists: Vec<IvfListMetadata>,
}

impl IvfListDirectory {
    /// Create a new list directory with the specified number of lists.
    pub fn new(num_lists: u16) -> Self {
        let lists = (0..num_lists)
            .map(|i| IvfListMetadata::new(i as u32))
            .collect();
        Self { lists }
    }

    /// Get the number of lists.
    pub fn num_lists(&self) -> usize {
        self.lists.len()
    }

    /// Get metadata for a specific list.
    pub fn get_list(&self, list_id: u16) -> Option<&IvfListMetadata> {
        self.lists.get(list_id as usize)
    }

    /// Get mutable metadata for a specific list.
    pub fn get_list_mut(&mut self, list_id: u16) -> Option<&mut IvfListMetadata> {
        self.lists.get_mut(list_id as usize)
    }

    /// Store the list directory at the AM's fixed slot (`base + 1`).
    pub unsafe fn store(&self, index: &PgRelation, base: pg_sys::BlockNumber, first_time: bool) {
        let mut stats = crate::access_method::stats::WriteStats::default();
        let mut tape = if first_time {
            ChainTapeWriter::new(index, PageType::IvfListDirectory, &mut stats)
        } else {
            ChainTapeWriter::reinit(
                index,
                PageType::IvfListDirectory,
                &mut stats,
                base + 1,
            )
        };

        let bytes = self.serialize_to_vec();
        let off = tape.write(&bytes);
        assert_eq!(
            off,
            ItemPointer::new(base + 1, LIST_DIRECTORY_OFFSET)
        );
    }

    /// Store the list directory anywhere (embedded segments), returning its
    /// pointer and block count.
    pub unsafe fn store_new(&self, index: &PgRelation) -> (crate::util::ItemPointer, u32) {
        let _ext_lock = crate::util::buffer::LockRelationForExtension::new(index);
        let mut stats = crate::access_method::stats::WriteStats::default();
        let mut tape = ChainTapeWriter::new(index, PageType::IvfListDirectory, &mut stats);
        tape.write_counted(&self.serialize_to_vec())
    }

    /// Load the list directory from the AM's fixed slot (`base + 1`).
    pub fn load(index: &PgRelation, base: pg_sys::BlockNumber) -> IvfListDirectory {
        Self::load_at(index, crate::util::ItemPointer::new(base + 1, LIST_DIRECTORY_OFFSET))
    }

    /// Load a list directory from an arbitrary pointer (embedded segments).
    pub fn load_at(index: &PgRelation, pointer: crate::util::ItemPointer) -> IvfListDirectory {
        unsafe {
            let mut stats = crate::access_method::stats::WriteStats::default();
            let mut tape = ChainItemReader::new(index, PageType::IvfListDirectory, &mut stats);

            let mut buf: Vec<u8> = Vec::new();
            for item in tape.read(pointer) {
                buf.extend_from_slice(item.get_data_slice());
            }
            rkyv::from_bytes::<IvfListDirectory>(&buf)
                .unwrap_or_else(|e| panic!("IVF: list-directory parse failed ({} bytes): {:?}", buf.len(), e))
        }
    }
}
