pub mod buffer;
pub mod chain;
pub mod lru;
pub mod page;
pub mod ports;
pub mod table_slot;
pub mod tape;

use pgrx::PgRelation;
use rkyv::{Archive, Deserialize, Serialize};

use self::{
    page::{ReadablePage, WritablePage},
    ports::PageGetItemChecked,
};

/// Report an on-disk metadata parse failure as data corruption.
///
/// The bytes read back from the index do not match the layout this build
/// expects: a damaged page, a torn write, or an index written by an
/// incompatible version.  Reported through `ereport!` rather than a bare
/// `panic!` so the failure carries `ERRCODE_DATA_CORRUPTED` instead of
/// `INTERNAL_ERROR` — which is what lets a client or a monitoring system tell
/// corruption from an internal bug — while still unwinding the Rust frames on
/// the way out, so every RAII buffer lock is released before the transaction
/// aborts.
pub fn corrupt_data(message: String) -> ! {
    // `ereport!` expands to a statement, so the diverging return needs its own
    // terminator (the macro's own `unreachable!()` runs first).
    pgrx::ereport!(
        ERROR,
        pgrx::pg_sys::errcodes::PgSqlErrorCode::ERRCODE_DATA_CORRUPTED,
        message
    );
    unreachable!("ereport raised")
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Eq, Hash, Clone, Copy)]
#[archive(check_bytes)]
#[repr(C)] // Added this so we can compute size via sizeof
pub struct ItemPointer {
    pub block_number: pgrx::pg_sys::BlockNumber,
    pub offset: pgrx::pg_sys::OffsetNumber,
}

impl PartialOrd for ItemPointer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ItemPointer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.block_number
            .cmp(&other.block_number)
            .then_with(|| self.offset.cmp(&other.offset))
    }
}

impl ArchivedItemPointer {
    pub fn deserialize_item_pointer(&self) -> ItemPointer {
        self.deserialize(&mut rkyv::Infallible).unwrap()
    }
}

impl PartialEq for ArchivedItemPointer {
    fn eq(&self, other: &Self) -> bool {
        self.block_number == other.block_number && self.offset == other.offset
    }
}

impl Eq for ArchivedItemPointer {}

impl PartialOrd for ArchivedItemPointer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ArchivedItemPointer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.block_number
            .cmp(&other.block_number)
            .then_with(|| self.offset.cmp(&other.offset))
    }
}

pub struct ReadableBuffer<'a> {
    _page: ReadablePage<'a>,
    len: usize,
    ptr: *const u8,
}

impl<'a> ReadableBuffer<'a> {
    pub fn get_data_slice(&self) -> &'a [u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn get_owned_page(self) -> ReadablePage<'a> {
        self._page
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn advance(&mut self, len: usize) {
        assert!(self.len >= len);
        self.ptr = unsafe { self.ptr.add(len) };
        self.len -= len;
    }
}

pub struct WritableBuffer<'a> {
    _page: WritablePage<'a>,
    len: usize,
    ptr: *mut u8,
}

impl<'a> WritableBuffer<'a> {
    pub fn get_data_slice(&mut self) -> &'a mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn commit(self) {
        self._page.commit();
    }
}

impl ItemPointer {
    pub fn new(
        block_number: pgrx::pg_sys::BlockNumber,
        offset: pgrx::pg_sys::OffsetNumber,
    ) -> Self {
        Self {
            block_number,
            offset,
        }
    }

    pub fn new_invalid() -> Self {
        Self {
            block_number: pgrx::pg_sys::InvalidBlockNumber,
            offset: pgrx::pg_sys::InvalidOffsetNumber,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.block_number != pgrx::pg_sys::InvalidBlockNumber
            && self.offset != pgrx::pg_sys::InvalidOffsetNumber
    }

    pub unsafe fn with_page(page: &page::WritablePage, offset: pgrx::pg_sys::OffsetNumber) -> Self {
        Self {
            block_number: pgrx::pg_sys::BufferGetBlockNumber(**(page.get_buffer())),
            offset,
        }
    }

    pub unsafe fn with_item_pointer_data(ctid: pgrx::pg_sys::ItemPointerData) -> Self {
        let ip = pgrx::itemptr::item_pointer_get_block_number(&ctid);
        let off = pgrx::itemptr::item_pointer_get_offset_number(&ctid);
        Self::new(ip, off)
    }

    pub fn to_item_pointer_data(self, ctid: &mut pgrx::pg_sys::ItemPointerData) {
        pgrx::itemptr::item_pointer_set_all(ctid, self.block_number, self.offset)
    }

    pub unsafe fn read_bytes(self, index: &PgRelation) -> ReadableBuffer<'_> {
        let page = ReadablePage::read(index, self.block_number);
        page.get_item_unchecked(self.offset)
    }

    pub unsafe fn modify_bytes(self, index: &PgRelation) -> WritableBuffer<'_> {
        let page = WritablePage::modify(index, self.block_number);
        // The item's start and length both come off the page, so bound them
        // before handing the payload to a writer (`PageGetItem` alone does not).
        let (ptr, len) = PageGetItemChecked(*page, self.offset);
        WritableBuffer {
            _page: page,
            ptr,
            len,
        }
    }

    pub fn ip_distance(self, other: Self) -> usize {
        let block_diff = self.block_number as isize - other.block_number as isize;
        let offset_diff = self.offset as isize - other.offset as isize;
        debug_assert!(offset_diff < pgrx::pg_sys::MaxOffsetNumber as _);
        (block_diff * (pgrx::pg_sys::MaxOffsetNumber as isize) + offset_diff).unsigned_abs()
    }
}

pub type IndexPointer = ItemPointer;
pub type HeapPointer = ItemPointer;
