//! IVF entry page management.
//!
//! Entry pages (Page 3+) store vectors assigned to each inverted list.
//!
//! A list's entries are serialized as a tightly-packed struct-of-arrays (SoA)
//! byte stream (see `serialize_entries` / `IvfEntrySlice`), then written as a
//! single chained item via `ChainTapeWriter`.  The SoA layout avoids the
//! per-entry `Vec` header and alignment padding that the rkyv `Vec<IvfEntry>`
//! representation used to incur.

use pgrx::pg_sys::BlockNumber;
use pgrx::*;
use rkyv::{Archive, Deserialize, Serialize};

use crate::access_method::quantization::rabitq::RabitqVector;
use crate::access_method::ivf::segment::{IvfActiveBuffer, IvfSegment};
use crate::util::page::{PageType, ReadablePage, WritablePage};
use crate::util::ports::{PageGetItem, PageGetItemId, PageGetMaxOffsetNumber};
use crate::util::*;

/// Magic for the SoA entry byte stream.
const IVF_ENTRY_MAGIC: u32 = 0x49564631; // "IVF1"

/// Header size: magic(4) + num_entries(4) + code_len(2) + num_bits(1) +
/// dim(2) + padding(3) = 16 bytes.
const ENTRY_HEADER_SIZE: usize = 16;

/// Packed heap-tid size: block number (u32) + offset (u16).
const TID_SIZE: usize = 6;

/// A single entry in an IVF inverted list (in-memory representation).
#[derive(Clone, Debug, PartialEq)]
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

/// Serialize a list's entries to a packed struct-of-arrays byte stream.
///
/// Layout (all little-endian):
/// ```text
/// [16-byte header][tids: n*6][codes: n*code_len][sum_of_x2: n*4]
/// [scale: n*4][margin_factor: n*4]
/// ```
///
/// `scale = -2*sum_of_x2/l1` and `margin_factor = 2*sqrt(sum_of_x2)/sqrt(dim)`
/// are precomputed at build time so the query-time estimate needs no division
/// or sqrt per entry.
pub fn serialize_entries(entries: &[IvfEntry]) -> Vec<u8> {
    let (num_bits, dim, code_len) = match entries.first() {
        Some(e) => (
            e.code.num_bits,
            e.code.dim as u16,
            e.code.packed_code.len(),
        ),
        None => (1u8, 0u16, 0usize),
    };
    let n = entries.len();
    let mut buf = Vec::with_capacity(ENTRY_HEADER_SIZE + n * (TID_SIZE + code_len + 12));
    buf.extend_from_slice(&IVF_ENTRY_MAGIC.to_le_bytes());
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&(code_len as u16).to_le_bytes());
    buf.push(num_bits);
    buf.extend_from_slice(&dim.to_le_bytes());
    buf.extend_from_slice(&[0u8; 3]); // pad header to 16 bytes
    debug_assert_eq!(buf.len(), ENTRY_HEADER_SIZE);

    let inv_sqrt_dim = 1.0 / (dim as f32).sqrt().max(1.0);

    for e in entries {
        buf.extend_from_slice(&e.heap_tid.block_number.to_le_bytes());
        buf.extend_from_slice(&e.heap_tid.offset.to_le_bytes());
    }
    if num_bits == 1 {
        // 1-bit codes are stored transposed (32-row batches) for SIMD FastScan.
        let mut row_major = Vec::with_capacity(n * code_len);
        for e in entries {
            row_major.extend_from_slice(&e.code.packed_code);
        }
        let transposed =
            crate::access_method::quantization::rabitq_fastscan::transpose_1bit(&row_major, n, code_len);
        buf.extend_from_slice(&transposed);
    } else {
        for e in entries {
            buf.extend_from_slice(&e.code.packed_code);
        }
    }
    for e in entries {
        buf.extend_from_slice(&e.code.sum_of_x2.to_le_bytes());
    }
    for e in entries {
        let scale = -2.0 * e.code.sum_of_x2 / e.code.l1_of_rotated.max(1e-9);
        buf.extend_from_slice(&scale.to_le_bytes());
    }
    for e in entries {
        let margin_factor = 2.0 * e.code.sum_of_x2.max(0.0).sqrt() * inv_sqrt_dim;
        buf.extend_from_slice(&margin_factor.to_le_bytes());
    }
    buf
}

/// A borrowed, zero-copy view over a serialized SoA entry byte stream.
pub struct IvfEntrySlice<'a> {
    num_entries: usize,
    code_len: usize,
    num_bits: u8,
    dim: u16,
    tids: &'a [u8],
    codes: &'a [u8],
    sums: &'a [u8],
    scales: &'a [u8],
    margin_factors: &'a [u8],
}

impl<'a> IvfEntrySlice<'a> {
    /// Parse an SoA byte stream, returning slices into `bytes` (zero-copy).
    pub fn parse(bytes: &'a [u8]) -> Self {
        assert!(bytes.len() >= ENTRY_HEADER_SIZE, "entry buffer too short");
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(magic, IVF_ENTRY_MAGIC, "bad entry magic");
        let num_entries = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let code_len = u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize;
        let num_bits = bytes[10];
        let dim = u16::from_le_bytes(bytes[11..13].try_into().unwrap());

        let tid_off = ENTRY_HEADER_SIZE;
        let code_off = tid_off + num_entries * TID_SIZE;
        // 1-bit codes are stored transposed (batched by 32, zero-padded).
        let codes_len = if num_bits == 1 {
            num_entries.div_ceil(32) * 32 * code_len
        } else {
            num_entries * code_len
        };
        let sum_off = code_off + codes_len;
        let scale_off = sum_off + num_entries * 4;
        let margin_off = scale_off + num_entries * 4;
        let end = margin_off + num_entries * 4;
        assert!(bytes.len() >= end, "entry buffer truncated");

        Self {
            num_entries,
            code_len,
            num_bits,
            dim,
            tids: &bytes[tid_off..code_off],
            codes: &bytes[code_off..sum_off],
            sums: &bytes[sum_off..scale_off],
            scales: &bytes[scale_off..margin_off],
            margin_factors: &bytes[margin_off..end],
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.num_entries
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_entries == 0
    }

    #[inline]
    pub fn num_bits(&self) -> u8 {
        self.num_bits
    }

    #[inline]
    pub fn dim(&self) -> u16 {
        self.dim
    }

    #[inline]
    pub fn code_len(&self) -> usize {
        self.code_len
    }

    /// Heap tid of entry `i`.
    #[inline]
    pub fn tid(&self, i: usize) -> ItemPointer {
        let b = &self.tids[i * TID_SIZE..(i + 1) * TID_SIZE];
        ItemPointer::new(
            u32::from_le_bytes(b[0..4].try_into().unwrap()),
            u16::from_le_bytes(b[4..6].try_into().unwrap()),
        )
    }

    /// Packed RaBitQ code of entry `i` (4/8-bit, row-major).
    #[inline]
    pub fn code(&self, i: usize) -> &'a [u8] {
        &self.codes[i * self.code_len..(i + 1) * self.code_len]
    }

    /// Raw code bytes (transposed for 1-bit, row-major for 4/8-bit).
    #[inline]
    pub fn codes(&self) -> &'a [u8] {
        self.codes
    }

    /// Number of 32-row transposed batches (1-bit only).
    #[inline]
    pub fn num_batches(&self) -> usize {
        self.num_entries.div_ceil(32)
    }

    /// Transposed 32-row code batch `batch` (1-bit only).
    #[inline]
    pub fn code_batch(&self, batch: usize) -> &'a [u8] {
        let b = 32 * self.code_len;
        &self.codes[batch * b..(batch + 1) * b]
    }

    /// `sum_of_x2` (residual norm squared) of entry `i`.
    #[inline]
    pub fn sum_of_x2(&self, i: usize) -> f32 {
        f32::from_le_bytes(self.sums[i * 4..(i + 1) * 4].try_into().unwrap())
    }

    /// The `sum_of_x2` array as `f32` (little-endian; x86/ARM only).
    #[inline]
    pub fn sum_of_x2_slice(&self) -> &'a [f32] {
        unsafe { std::slice::from_raw_parts(self.sums.as_ptr() as *const f32, self.num_entries) }
    }

    /// The precomputed `scale` array as `f32`.
    #[inline]
    pub fn scale_slice(&self) -> &'a [f32] {
        unsafe { std::slice::from_raw_parts(self.scales.as_ptr() as *const f32, self.num_entries) }
    }

    /// The precomputed `margin_factor` array as `f32`.
    #[inline]
    pub fn margin_factor_slice(&self) -> &'a [f32] {
        unsafe {
            std::slice::from_raw_parts(
                self.margin_factors.as_ptr() as *const f32,
                self.num_entries,
            )
        }
    }

    /// Precomputed `scale = -2*sum_of_x2/l1` of entry `i`.
    #[inline]
    pub fn scale(&self, i: usize) -> f32 {
        f32::from_le_bytes(self.scales[i * 4..(i + 1) * 4].try_into().unwrap())
    }

    /// Precomputed `margin_factor = 2*sqrt(sum_of_x2)/sqrt(dim)` of entry `i`.
    #[inline]
    pub fn margin_factor(&self, i: usize) -> f32 {
        f32::from_le_bytes(self.margin_factors[i * 4..(i + 1) * 4].try_into().unwrap())
    }

    /// Reconstruct `l1_of_rotated` from the precomputed `scale` (for the
    /// insert/vacuum rewrite round-trip).  `l1` is not stored; it is only
    /// needed to recompute `scale` when a list is rewritten.
    #[inline]
    pub fn l1_reconstructed(&self, i: usize) -> f32 {
        let sx2 = self.sum_of_x2(i);
        if sx2.abs() > f32::EPSILON {
            -2.0 * sx2 / self.scale(i)
        } else {
            0.0
        }
    }
}

/// Writer for IVF entry pages.
///
/// Collects all entries for a single inverted list and writes them as one
/// chained item (a single serialized SoA byte stream).
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

    /// Finish writing and return `(first block, block count, total entries)`.
    ///
    /// The serialized SoA bytes are written across a contiguous run of pages
    /// (one item per page), so the scan can bulk-read them with `smgrreadv`.
    /// The relation extension lock is held across the whole run, so concurrent
    /// extension by other lists cannot interleave blocks into the middle of
    /// the segment (reentrant within a backend, so the per-page
    /// `WritablePage::new` calls below just bump the count).
    pub fn finish(self) -> (Option<BlockNumber>, u32, usize) {
        let total_entries = self.entries.len();
        if total_entries == 0 {
            return (None, 0, 0);
        }

        let _ = self.list_id;
        let bytes = serialize_entries(&self.entries);

        let _ext_lock = crate::util::buffer::LockRelationForExtension::new(self.index);

        let mut page = WritablePage::new(self.index, PageType::IvfEntry);
        let first_block = page.get_block_number();
        let mut num_blocks = 0u32;
        let mut remaining: &[u8] = &bytes;
        loop {
            let cap = page.get_aligned_free_space();
            let chunk_len = remaining.len().min(cap);
            page.add_item(&remaining[..chunk_len]);
            page.commit();
            num_blocks += 1;
            remaining = &remaining[chunk_len..];
            if remaining.is_empty() {
                break;
            }
            page = WritablePage::new(self.index, PageType::IvfEntry);
        }
        (Some(first_block), num_blocks, total_entries)
    }
}

/// Seal `entries` into one immutable SoA segment (a contiguous run of blocks
/// holding a single serialized byte stream).  `entries` is consumed.
pub fn seal_entries(index: &PgRelation, entries: Vec<IvfEntry>) -> IvfSegment {
    let mut writer = IvfEntryWriter::new(index, 0);
    for e in entries {
        writer.add_entry(e);
    }
    let (start_page, num_blocks, count) = writer.finish();
    IvfSegment::new(
        start_page.unwrap_or(pg_sys::InvalidBlockNumber),
        num_blocks,
        count as u64,
    )
}

/// Exact serialized byte length of a seal of `n` entries (mirrors the
/// `serialize_entries` arithmetic), so the reclamation allocator can reserve
/// exactly the right number of blocks before sealing.
pub fn seal_bytes_len(n: usize, num_bits: u8, dim_padded: u32) -> usize {
    let code_len = match num_bits {
        1 => dim_padded as usize / 8,
        4 => dim_padded as usize / 2,
        _ => dim_padded as usize,
    };
    let codes = if num_bits == 1 {
        n.div_ceil(32) * 32 * code_len
    } else {
        n * code_len
    };
    ENTRY_HEADER_SIZE + n * TID_SIZE + codes + n * 12
}

/// Number of blocks a seal of `n` entries occupies (exact for a fresh page).
pub fn seal_blocks_needed(n: usize, num_bits: u8, dim_padded: u32) -> u32 {
    let bytes = seal_bytes_len(n, num_bits, dim_padded);
    let cap = crate::util::page::tsv_fresh_page_capacity().max(1);
    bytes.div_ceil(cap) as u32
}

/// Seal `entries` into an immutable SoA segment written into blocks
/// `[start_block, start_block + used)` (reclaimed blocks).  Returns the number
/// of blocks used; the caller must have reserved at least `seal_blocks_needed`
/// blocks and pushes back any unused tail.
pub fn seal_entries_at(
    index: &PgRelation,
    entries: Vec<IvfEntry>,
    start_block: BlockNumber,
) -> u32 {
    if entries.is_empty() {
        return 0;
    }
    let bytes = serialize_entries(&entries);
    let mut remaining: &[u8] = &bytes;
    let mut used = 0u32;
    while !remaining.is_empty() {
        let mut page = WritablePage::modify(index, start_block + used as BlockNumber);
        page.reinit(PageType::IvfEntry);
        let cap = page.get_aligned_free_space();
        let chunk_len = remaining.len().min(cap);
        page.add_item(&remaining[..chunk_len]);
        page.commit();
        used += 1;
        remaining = &remaining[chunk_len..];
    }
    used
}

// ---------------------------------------------------------------------------
// Active (append) buffer: the unpublished, writer-only staging area.
//
// Entries are stored row-major, one rkyv `ActiveEntry` item per `PageAddItem`,
// so appends never touch previously written bytes.  1-bit codes stay
// row-major here; `seal_entries` transposes them when the buffer is sealed
// into an immutable SoA segment.  Readers (scans) never look at these pages.
// ---------------------------------------------------------------------------

/// A single entry in the unpublished active buffer (row-major rkyv item).
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
pub struct ActiveEntry {
    pub heap_tid: ItemPointer,
    pub code: RabitqVector,
}

/// Serialize one entry into its active-buffer item bytes.  The entry is
/// consumed.  (Requires each entry to fit a page; 8-bit codes for dims near
/// the vector-type maximum would not — same limit as the sealed path's items.)
pub fn serialize_active_entry(entry: IvfEntry) -> Vec<u8> {
    let active = ActiveEntry {
        heap_tid: entry.heap_tid,
        code: entry.code,
    };
    rkyv::to_bytes::<_, 256>(&active).unwrap().to_vec()
}

/// Append one entry's serialized bytes to the active buffer's tail page,
/// reusing `reserved_page` (a reclaimed block) when the tail is full or no
/// buffer exists yet.  Returns the updated active buffer and whether the
/// reserved page was consumed.
pub fn append_active_entry_bytes(
    index: &PgRelation,
    active: Option<IvfActiveBuffer>,
    bytes: &[u8],
    reserved_page: Option<BlockNumber>,
) -> (IvfActiveBuffer, bool) {
    let commit_new = |block: BlockNumber, bytes: &[u8]| {
        let mut new_page = WritablePage::modify(index, block);
        new_page.reinit(PageType::IvfActiveBuffer);
        new_page.add_item(bytes);
        new_page.commit();
    };
    match active {
        Some(mut a) => {
            let tail_block = *a.pages.last().expect("active buffer has pages");
            let mut tail = WritablePage::modify(index, tail_block);
            if tail.get_aligned_free_space() >= bytes.len() {
                tail.add_item(bytes);
                tail.commit();
                a.num_entries += 1;
                (a, false)
            } else {
                drop(tail); // abort: no changes to the full tail page
                match reserved_page {
                    Some(block) => {
                        commit_new(block, bytes);
                        a.pages.push(block);
                        a.num_entries += 1;
                        (a, true)
                    }
                    None => {
                        let mut new_page = WritablePage::new(index, PageType::IvfActiveBuffer);
                        let block = new_page.get_block_number();
                        new_page.add_item(bytes);
                        new_page.commit();
                        a.pages.push(block);
                        a.num_entries += 1;
                        (a, false)
                    }
                }
            }
        }
        None => match reserved_page {
            Some(block) => {
                commit_new(block, bytes);
                (IvfActiveBuffer::new(block), true)
            }
            None => {
                let mut new_page = WritablePage::new(index, PageType::IvfActiveBuffer);
                let block = new_page.get_block_number();
                new_page.add_item(bytes);
                new_page.commit();
                (IvfActiveBuffer::new(block), false)
            }
        },
    }
}

/// Read all entries of the active buffer back (used at seal time, while the
/// list header's exclusive lock excludes concurrent appenders).  Pages are
/// followed explicitly: unlike sealed segments, the active buffer's blocks are
/// not guaranteed contiguous.
pub fn read_active_entries(index: &PgRelation, active: &IvfActiveBuffer) -> Vec<IvfEntry> {
    let mut out = Vec::with_capacity(active.num_entries as usize);
    unsafe {
        for &block in &active.pages {
            let page = ReadablePage::read(index, block);
            let page_ptr = *page;
            let max = PageGetMaxOffsetNumber(page_ptr);
            for off in 1..=max as pg_sys::OffsetNumber {
                let item_id = PageGetItemId(page_ptr, off);
                if (*item_id).lp_len() == 0 {
                    continue; // unused line pointer
                }
                let item = PageGetItem(page_ptr, item_id);
                let len = (*item_id).lp_len() as usize;
                let bytes = std::slice::from_raw_parts(item as *const u8, len);
                let entry: ActiveEntry = rkyv::from_bytes(bytes).unwrap();
                out.push(IvfEntry {
                    heap_tid: entry.heap_tid,
                    code: entry.code,
                });
            }
        }
    }
    out
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

    /// Bulk-read a list's contiguous entry blocks via `smgrreadv` and return the
    /// reassembled SoA byte stream.
    ///
    /// This bypasses the buffer manager (no per-block pin/LWLock); it reads the
    /// on-disk image directly, so the writer must flush the relation (see
    /// `FlushRelationBuffers` in build/insert/vacuum) before a scan runs.
    fn read_bytes(&self, start_page: BlockNumber, num_blocks: u32) -> Vec<u8> {
        if num_blocks == 0 {
            return Vec::new();
        }
        unsafe {
            let rel = self.index.as_ptr();
            // The relation's smgr handle is normally already open (the scan
            // reads the meta/centroid/directory pages via the buffer manager
            // first).  Fall back to opening it if not.
            let reln = (*rel).rd_smgr;
            let reln = if reln.is_null() {
                pg_sys::smgropen((*rel).rd_locator, (*rel).rd_backend)
            } else {
                reln
            };

            let n = num_blocks as usize;
            let blksz = pg_sys::BLCKSZ as usize;
            // One contiguous raw buffer, *uninitialized*: `smgrreadv` overwrites
            // every byte, so zero-filling it would be pure wasted memset.
            let mut raw: Vec<u8> = Vec::with_capacity(n * blksz);
            raw.set_len(n * blksz);
            let mut ptrs: Vec<*mut std::os::raw::c_void> = (0..n)
                .map(|i| raw.as_mut_ptr().add(i * blksz) as *mut std::os::raw::c_void)
                .collect();
            pg_sys::smgrreadv(
                reln,
                pg_sys::ForkNumber::MAIN_FORKNUM,
                start_page,
                ptrs.as_mut_ptr(),
                num_blocks as BlockNumber,
            );

            let mut buf: Vec<u8> = Vec::with_capacity(n * blksz);
            for i in 0..n {
                let block = &raw[i * blksz..(i + 1) * blksz];
                let page = block.as_ptr() as pg_sys::Page;
                let item_id = PageGetItemId(page, 1);
                let item = PageGetItem(page, item_id);
                let len = (*item_id).lp_len() as usize;
                buf.extend_from_slice(std::slice::from_raw_parts(item as *const u8, len));
            }
            buf
        }
    }

    /// Read all entries for a given list.
    pub fn read_entries(&self, start_page: BlockNumber, num_blocks: u32) -> Vec<IvfEntry> {
        let buf = self.read_bytes(start_page, num_blocks);
        if buf.is_empty() {
            return Vec::new();
        }
        let view = IvfEntrySlice::parse(&buf);
        // 1-bit codes are stored transposed; recover row-major for the
        // in-memory IvfEntry (the rewrite path re-transposes on serialize).
        let row_major: Vec<u8> = if view.num_bits() == 1 {
            crate::access_method::quantization::rabitq_fastscan::untranspose_1bit(
                view.codes(),
                view.len(),
                view.code_len(),
            )
        } else {
            Vec::new()
        };
        (0..view.len())
            .map(|i| IvfEntry {
                heap_tid: view.tid(i),
                code: RabitqVector {
                    dim: view.dim() as u32,
                    sum_of_x2: view.sum_of_x2(i),
                    l1_of_rotated: view.l1_reconstructed(i),
                    packed_code: if view.num_bits() == 1 {
                        row_major[i * view.code_len()..(i + 1) * view.code_len()].to_vec()
                    } else {
                        view.code(i).to_vec()
                    },
                    num_bits: view.num_bits(),
                    cent_dot: 0.0,
                },
            })
            .collect()
    }

    /// Read a list's entries zero-copy and invoke `f` with the parsed slice.
    ///
    /// The byte buffer is kept alive for the duration of `f`, and the returned
    /// slice borrows from it, so no owned `IvfEntry` (or per-entry `Vec<u8>`
    /// code) is allocated.
    pub fn for_each_slice<F: FnMut(&IvfEntrySlice)>(
        &self,
        start_page: BlockNumber,
        num_blocks: u32,
        mut f: F,
    ) {
        let buf = self.read_bytes(start_page, num_blocks);
        if buf.is_empty() {
            return;
        }
        let view = IvfEntrySlice::parse(&buf);
        f(&view);
    }
}
