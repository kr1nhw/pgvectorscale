//! IVF entry page management.
//!
//! Entry pages (Page 4+) store vectors assigned to each inverted list.

use pgrx::pg_sys::BlockNumber;
use pgrx::*;
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{Archive, Deserialize, Serialize};

use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::access_method::quantization::rabitq::RabitqVector;
use crate::util::chain::{ChainItemReader, ChainTapeWriter};
use crate::util::page::PageType;
use crate::util::*;

/// A single entry in an IVF inverted list.
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct IvfEntry {
    /// Heap tuple ID (ctid) pointing to the original row
    pub heap_tid: ItemPointer,
    /// Centroid-relative RaBitQ code.
    pub code: RabitqVector,
}

impl IvfEntry {
    /// Create a new entry.
    pub fn new(heap_tid: ItemPointer, code: RabitqVector) -> Self {
        Self { heap_tid, code }
    }
}

/// IVF entry page containing entries for a single inverted list.
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct IvfEntryPage {
    /// List ID this page belongs to
    pub list_id: u16,
    /// Entries in this page
    pub entries: Vec<IvfEntry>,
    /// Next page in the chain (for this list)
    pub next_page: BlockNumber,
}

impl IvfEntryPage {
    /// Create a new entry page for the given list.
    pub fn new(list_id: u16) -> Self {
        Self {
            list_id,
            entries: Vec::new(),
            next_page: pgrx::pg_sys::InvalidBlockNumber,
        }
    }

    /// Add an entry to this page.
    pub fn add_entry(&mut self, entry: IvfEntry) {
        self.entries.push(entry);
    }

    /// Get the number of entries.
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Check if the page is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Writer for IVF entry pages.
///
/// Collects all entries for a single inverted list and writes them as one
/// chained item (a single serialized `IvfEntryPage`).  The chain handles items
/// larger than a page, so there is no arbitrary per-page entry-count limit.
pub struct IvfEntryWriter<'a> {
    index: &'a PgRelation,
    list_id: u16,
    entries: Vec<IvfEntry>,
}

impl<'a> IvfEntryWriter<'a> {
    /// Create a new entry writer for the given list.
    pub fn new(index: &'a PgRelation, list_id: u16) -> Self {
        Self {
            index,
            list_id,
            entries: Vec::new(),
        }
    }

    /// Add an entry to this list.
    pub fn add_entry(&mut self, entry: IvfEntry) {
        self.entries.push(entry);
    }

    /// Finish writing and return the first page number and total entries.
    pub fn finish(self) -> (Option<BlockNumber>, usize) {
        let total_entries = self.entries.len();
        if total_entries == 0 {
            return (None, 0);
        }

        let page = IvfEntryPage {
            list_id: self.list_id,
            entries: self.entries,
            next_page: pgrx::pg_sys::InvalidBlockNumber,
        };

        unsafe {
            let mut stats = crate::access_method::stats::WriteStats::default();
            let mut tape = ChainTapeWriter::new(self.index, PageType::IvfEntry, &mut stats);
            let bytes = page.serialize_to_vec();
            let off = tape.write(&bytes);
            (Some(off.block_number), total_entries)
        }
    }
}

/// Reader for IVF entry pages.
pub struct IvfEntryReader<'a> {
    index: &'a PgRelation,
}

impl<'a> IvfEntryReader<'a> {
    /// Create a new entry reader.
    pub fn new(index: &'a PgRelation) -> Self {
        Self { index }
    }

    /// Read all entries for a given list starting from the given page.
    pub fn read_entries(&self, start_page: BlockNumber) -> Vec<IvfEntry> {
        unsafe {
            let mut stats = crate::access_method::stats::WriteStats::default();
            let mut reader = ChainItemReader::new(self.index, PageType::IvfEntry, &mut stats);

            let mut buf: Vec<u8> = Vec::new();
            for item in reader.read(ItemPointer::new(start_page, 1)) {
                buf.extend_from_slice(item.get_data_slice());
            }

            if buf.is_empty() {
                return Vec::new();
            }

            let page = rkyv::from_bytes::<IvfEntryPage>(&buf).unwrap();
            page.entries
        }
    }
}
