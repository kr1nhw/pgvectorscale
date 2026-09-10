//! A Page is a Postgres abstraction for a slice of memory you can write to
//! It is usually 8kb and has a special layout. See https://www.postgresql.org/docs/current/storage-page-layout.html

use pg_sys::Page;
use pgrx::{
    pg_sys::{BlockNumber, BufferGetPage, OffsetNumber, ReadBufferMode, BLCKSZ},
    *,
};
use std::ops::Deref;

use super::{
    buffer::{LockedBufferExclusive, LockedBufferShare},
    ports::{PageGetItem, PageGetItemId},
    ReadableBuffer,
};
pub struct WritablePage<'a> {
    buffer: LockedBufferExclusive<'a>,
    page: Page,
    state: *mut pg_sys::GenericXLogState,
    committed: bool,
}

pub const TSV_PAGE_ID: u16 = 0xAE24; /* magic number, generated randomly */

/// PageType identifies different types of pages in our index.
/// The layout of any one type should be consistent
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PageType {
    MetaV1 = 0,
    Node = 1,
    PqQuantizerDef = 2,
    PqQuantizerVector = 3,
    SbqMeansV1 = 4,
    SbqNode = 5,
    MetaV2 = 6,
    SbqMeans = 7,
    Meta = 8,
    RabitqNode = 9,
    RabitqMetadata = 10,
    IvfMeta = 11,
    IvfListDirectory = 12,
    IvfCentroids = 13,
    IvfQuantizerMetadata = 14,
    IvfEntry = 15,
    IvfListHeader = 16,
    IvfSegmentList = 17,
    IvfActiveBuffer = 18,
    IvfFreeList = 19,
    AgentVecMeta = 20,
    AgentVecDirectory = 21,
    AgentVecSegmentHeader = 22,
    AgentVecFlatPage = 23,
}

impl PageType {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => PageType::MetaV1,
            1 => PageType::Node,
            2 => PageType::PqQuantizerDef,
            3 => PageType::PqQuantizerVector,
            4 => PageType::SbqMeansV1,
            5 => PageType::SbqNode,
            6 => PageType::MetaV2,
            7 => PageType::SbqMeans,
            8 => PageType::Meta,
            9 => PageType::RabitqNode,
            10 => PageType::RabitqMetadata,
            11 => PageType::IvfMeta,
            12 => PageType::IvfListDirectory,
            13 => PageType::IvfCentroids,
            14 => PageType::IvfQuantizerMetadata,
            15 => PageType::IvfEntry,
            16 => PageType::IvfListHeader,
            17 => PageType::IvfSegmentList,
            18 => PageType::IvfActiveBuffer,
            19 => PageType::IvfFreeList,
            20 => PageType::AgentVecMeta,
            21 => PageType::AgentVecDirectory,
            22 => PageType::AgentVecSegmentHeader,
            23 => PageType::AgentVecFlatPage,
            _ => panic!("Unknown PageType number {}", value),
        }
    }

    /// `ChainTape` supports chaining of pages that might contain large data.
    /// This is not supported for all page types.  Note that `Tape` requires
    /// that the page type not be chained.
    pub fn is_chained(self) -> bool {
        matches!(self, PageType::SbqMeans)
            || matches!(self, PageType::Meta)
            || matches!(self, PageType::IvfMeta)
            || matches!(self, PageType::IvfListDirectory)
            || matches!(self, PageType::IvfCentroids)
            || matches!(self, PageType::IvfQuantizerMetadata)
            || matches!(self, PageType::IvfEntry)
            || matches!(self, PageType::IvfListHeader)
            || matches!(self, PageType::IvfSegmentList)
            || matches!(self, PageType::IvfFreeList)
            || matches!(self, PageType::AgentVecDirectory)
    }
}

/// This is the Tsv-specific data that goes on every "diskann-owned" page
/// It is placed at the end of a page in the "special" area
#[repr(C)]
struct TsvPageOpaqueData {
    page_type: u8, // stores the PageType enum as an integer (u8 because we doubt we'll have more than 256 types).
    _reserved: u8, // don't waste bytes, may be able to reuse later. For now: 0
    page_id: u16, //  A magic ID for debuging to identify the page as a "diskann-owned". Should be last.
}

impl TsvPageOpaqueData {
    fn new(page_type: PageType) -> Self {
        Self {
            page_type: page_type as u8,
            _reserved: 0,
            page_id: TSV_PAGE_ID,
        }
    }

    /// Safety: unsafe because no verification done. Blind cast.
    #[inline(always)]
    unsafe fn with_page(page: Page) -> *mut TsvPageOpaqueData {
        let sp = super::ports::PageGetSpecialPointer(page);
        sp.cast::<TsvPageOpaqueData>()
    }

    /// Safety: Safe because of the verify call that checks a magic number
    fn read_from_page(page: &Page) -> &TsvPageOpaqueData {
        unsafe {
            let ptr = Self::with_page(*page);
            (*ptr).verify();
            ptr.as_ref().unwrap()
        }
    }

    fn verify(&self) {
        assert_eq!(self.page_id, TSV_PAGE_ID);
        PageType::from_u8(self.page_type);
    }
}

/// WritablePage implements and RAII-guarded Page that you can write to.
/// All writes will be WAL-logged.
///
/// It is probably not a good idea to hold on to a WritablePage for a long time.
impl<'a> WritablePage<'a> {
    /// new creates a totally new page on a relation by extending the relation
    pub fn new(index: &'a PgRelation, page_type: PageType) -> Self {
        let buffer = LockedBufferExclusive::new(index);

        unsafe {
            let state = pg_sys::GenericXLogStart(index.as_ptr());
            //TODO do we need a GENERIC_XLOG_FULL_IMAGE option?
            let page = pg_sys::GenericXLogRegisterBuffer(state, *buffer, 0);
            let mut new = Self {
                buffer,
                page,
                state,
                committed: false,
            };
            new.reinit(page_type);
            new
        }
    }

    pub fn reinit(&mut self, page_type: PageType) {
        unsafe {
            pg_sys::PageInit(
                self.page,
                pg_sys::BLCKSZ as usize,
                std::mem::size_of::<TsvPageOpaqueData>(),
            );
            *TsvPageOpaqueData::with_page(self.page) = TsvPageOpaqueData::new(page_type);
        }
    }

    pub fn modify(index: &'a PgRelation, block: BlockNumber) -> Self {
        let buffer = LockedBufferExclusive::read(index, block);
        Self::modify_with_buffer(index, buffer)
    }

    pub fn add_item(&mut self, data: &[u8]) -> OffsetNumber {
        let size = data.len();
        assert!(self.get_free_space() >= size);
        unsafe { self.add_item_unchecked(data) }
    }

    pub unsafe fn add_item_unchecked(&mut self, data: &[u8]) -> OffsetNumber {
        let size = data.len();
        assert!(size < BLCKSZ as usize);

        let offset_number = pg_sys::PageAddItemExtended(
            self.page,
            data.as_ptr() as _,
            size,
            pg_sys::InvalidOffsetNumber,
            0,
        );

        assert!(offset_number != pg_sys::InvalidOffsetNumber);
        offset_number
    }

    /// get a writable page for cleanup(vacuum) operations.
    pub unsafe fn cleanup(index: &'a PgRelation, block: BlockNumber) -> Self {
        let buffer = LockedBufferExclusive::read_for_cleanup(index, block);
        Self::modify_with_buffer(index, buffer)
    }

    // Safety: Safe because it verifies the page
    fn modify_with_buffer(index: &'a PgRelation, buffer: LockedBufferExclusive<'a>) -> Self {
        unsafe {
            let state = pg_sys::GenericXLogStart(index.as_ptr());
            let page = pg_sys::GenericXLogRegisterBuffer(state, *buffer, 0);
            //this check the page
            _ = TsvPageOpaqueData::read_from_page(&page);
            Self {
                buffer,
                page,
                state,
                committed: false,
            }
        }
    }

    pub fn get_buffer(&self) -> &LockedBufferExclusive<'_> {
        &self.buffer
    }

    pub fn get_block_number(&self) -> BlockNumber {
        self.buffer.get_block_number()
    }

    fn get_free_space(&self) -> usize {
        unsafe { pg_sys::PageGetFreeSpace(self.page) }
    }

    /// The actual free space that can be used to store data.
    /// See https://github.com/postgres/postgres/blob/0164a0f9ee12e0eff9e4c661358a272ecd65c2d4/src/backend/storage/page/bufpage.c#L304
    pub fn get_aligned_free_space(&self) -> usize {
        let free_space = self.get_free_space();
        free_space - (free_space % 8)
    }

    pub fn get_type(&self) -> PageType {
        unsafe {
            let opaque_data =
            //safe to do because self.page was already verified during construction
            TsvPageOpaqueData::with_page(self.page);

            PageType::from_u8((*opaque_data).page_type)
        }
    }

    pub fn set_types(&self, new: PageType) {
        unsafe {
            let opaque_data =
            //safe to do because self.page was already verified during construction
            TsvPageOpaqueData::with_page(self.page);

            (*opaque_data).page_type = new as u8;
        }
    }
    /// commit saves all the changes to the page.
    /// Note that this will consume the page and make it unusable after the call.
    pub fn commit(mut self) {
        unsafe {
            pg_sys::MarkBufferDirty(*self.buffer);
            pg_sys::GenericXLogFinish(self.state);
        }
        self.committed = true;
    }
}

impl Drop for WritablePage<'_> {
    // drop aborts the xlog if it has not been committed.
    fn drop(&mut self) {
        if !self.committed {
            unsafe {
                pg_sys::GenericXLogAbort(self.state);
            };
        }
    }
}

/// Flush a contiguous block range of the relation's main fork to the kernel
/// (only dirty pages are written).  Unlike `FlushRelationBuffers`, this never
/// touches buffers the caller holds content locks on, so it is safe inside a
/// header/meta update closure (where `FlushRelationBuffers` would try to flush
/// the locked page and self-deadlock on its content lock).
pub unsafe fn flush_block_range(
    index: &PgRelation,
    start: pg_sys::BlockNumber,
    count: u32,
) {
    for i in 0..count {
        let block = start + i as pg_sys::BlockNumber;
        let buf = pg_sys::ReadBufferExtended(
            index.as_ptr(),
            pg_sys::ForkNumber::MAIN_FORKNUM,
            block,
            ReadBufferMode::RBM_NORMAL,
            std::ptr::null_mut(),
        );
        pg_sys::FlushOneBuffer(buf);
        pg_sys::ReleaseBuffer(buf);
    }
}

/// The usable bytes for one item on a freshly initialized TSV page — exactly
/// what `WritablePage::get_aligned_free_space` reports on a fresh page.  Used
/// by callers that must size a block allocation for a multi-page item write
/// before writing it (e.g. the IVF reclamation allocator).
pub fn tsv_fresh_page_capacity() -> usize {
    let header = std::mem::offset_of!(pg_sys::PageHeaderData, pd_linp);
    let free = pg_sys::BLCKSZ as usize - header - std::mem::size_of::<TsvPageOpaqueData>();
    free - free % 8
}

/// Rewrite a page whose buffer is ALREADY pinned and exclusively locked by the
/// caller, replacing its content with `bytes` as its only item (offset 1).
///
/// This is for callers that must hold the buffer lock across a multi-step
/// operation (e.g. the IVF list-header read-modify-write under the header's
/// content lock).  WAL-logged via GenericXLog; the caller keeps the lock.
pub unsafe fn write_single_item_page_locked(
    index: &PgRelation,
    buffer: &LockedBufferExclusive,
    page_type: PageType,
    bytes: &[u8],
) {
    let state = pg_sys::GenericXLogStart(index.as_ptr());
    let page = pg_sys::GenericXLogRegisterBuffer(state, **buffer, 0);
    pg_sys::PageInit(
        page,
        pg_sys::BLCKSZ as usize,
        std::mem::size_of::<TsvPageOpaqueData>(),
    );
    *TsvPageOpaqueData::with_page(page) = TsvPageOpaqueData::new(page_type);
    let off = pg_sys::PageAddItemExtended(
        page,
        bytes.as_ptr() as _,
        bytes.len(),
        pg_sys::InvalidOffsetNumber,
        0,
    );
    assert!(off != pg_sys::InvalidOffsetNumber);
    pg_sys::MarkBufferDirty(**buffer);
    pg_sys::GenericXLogFinish(state);
}

impl Deref for WritablePage<'_> {
    type Target = Page;
    fn deref(&self) -> &Self::Target {
        &self.page
    }
}

pub struct ReadablePage<'a> {
    buffer: LockedBufferShare<'a>,
    page: Page,
}

impl<'a> ReadablePage<'a> {
    /// new creates a totally new page on a relation by extending the relation
    pub unsafe fn read(index: &'a PgRelation, block: BlockNumber) -> Self {
        let buffer = LockedBufferShare::read(index, block);
        let page = BufferGetPage(*buffer);
        Self { buffer, page }
    }

    pub fn get_type(&self) -> PageType {
        let opaque_data = TsvPageOpaqueData::read_from_page(&self.page);
        PageType::from_u8(opaque_data.page_type)
    }

    pub fn get_buffer(&self) -> &LockedBufferShare<'_> {
        &self.buffer
    }

    // Safety: unsafe because no verification of the offset is done.
    pub unsafe fn get_item_unchecked(
        self,
        offset: pgrx::pg_sys::OffsetNumber,
    ) -> ReadableBuffer<'a> {
        let item_id = PageGetItemId(self.page, offset);
        let item = PageGetItem(self.page, item_id) as *mut u8;
        let len = (*item_id).lp_len();
        ReadableBuffer {
            _page: self,
            ptr: item,
            len: len as _,
        }
    }
}

impl Deref for ReadablePage<'_> {
    type Target = Page;
    fn deref(&self) -> &Self::Target {
        &self.page
    }
}
