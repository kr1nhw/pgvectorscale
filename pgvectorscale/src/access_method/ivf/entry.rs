//! IVF entry page management.
//!
//! Entry pages (Page 4+) store vectors assigned to each inverted list.

use pgrx::pg_sys::BlockNumber;
use pgrx::*;
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{Archive, Deserialize, Serialize};

use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::util::chain::{ChainItemReader, ChainTapeWriter};
use crate::util::page::PageType;
use crate::util::*;

/// A single entry in an IVF inverted list.
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct IvfEntry {
    /// Heap tuple ID (ctid) pointing to the original row
    pub heap_tid: ItemPointer,
    /// The vector data
    pub vector: Vec<f32>,
}

impl IvfEntry {
    /// Create a new entry.
    pub fn new(heap_tid: ItemPointer, vector: Vec<f32>) -> Self {
        Self { heap_tid, vector }
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
pub struct IvfEntryWriter<'a> {
    index: &'a PgRelation,
    current_page: IvfEntryPage,
    pages_written: Vec<BlockNumber>,
}

impl<'a> IvfEntryWriter<'a> {
    /// Create a new entry writer for the given list.
    pub fn new(index: &'a PgRelation, list_id: u16) -> Self {
        Self {
            index,
            current_page: IvfEntryPage::new(list_id),
            pages_written: Vec::new(),
        }
    }

    /// Add an entry to the current page, flushing if needed.
    pub fn add_entry(&mut self, entry: IvfEntry) {
        self.current_page.add_entry(entry);

        // Flush if page is getting full (arbitrary threshold for now)
        if self.current_page.num_entries() >= 100 {
            self.flush_page();
        }
    }

    /// Flush the current page to disk.
    fn flush_page(&mut self) {
        if self.current_page.is_empty() {
            return;
        }

        unsafe {
            let mut stats = crate::access_method::stats::WriteStats::default();
            let mut tape = ChainTapeWriter::new(self.index, PageType::IvfEntry, &mut stats);

            let bytes = self.current_page.serialize_to_vec();
            let off = tape.write(&bytes);

            let block_number = off.block_number;
            self.pages_written.push(block_number);
        }

        // Start a new page
        self.current_page = IvfEntryPage::new(self.current_page.list_id);
    }

    /// Finish writing and return the first page number and total entries.
    pub fn finish(mut self) -> (Option<BlockNumber>, usize) {
        self.flush_page();

        let first_page = self.pages_written.first().copied();
        let total_entries = self.pages_written.len() * 100; // Approximate
        (first_page, total_entries)
    }
}

/// Reader for IVF entry pages.
pub struct IvfEntryReader {
    index: PgRelation,
}

impl IvfEntryReader {
    /// Create a new entry reader.
    pub fn new(index: PgRelation) -> Self {
        Self { index }
    }

    /// Read all entries for a given list starting from the given page.
    pub fn read_entries(&self, start_page: BlockNumber) -> Vec<IvfEntry> {
        let mut entries = Vec::new();
        let mut current_page = start_page;

        loop {
            let page_data = unsafe {
                let mut stats = crate::access_method::stats::WriteStats::default();
                let mut tape = ChainItemReader::new(&self.index, PageType::IvfEntry, &mut stats);

                let mut buf: Vec<u8> = Vec::new();
                for item in tape.read(ItemPointer::new(current_page, 1)) {
                    buf.extend_from_slice(item.get_data_slice());
                }
                buf
            };

            if page_data.is_empty() {
                break;
            }

            let page = rkyv::from_bytes::<IvfEntryPage>(&page_data).unwrap();
            entries.extend(page.entries);

            if page.next_page == pgrx::pg_sys::InvalidBlockNumber {
                break;
            }
            current_page = page.next_page;
        }

        entries
    }
}
