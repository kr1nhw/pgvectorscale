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

use crate::access_method::quantization::rabitq::RabitqVector;
use crate::util::page::{PageType, WritablePage};
use crate::util::ports::{PageGetItem, PageGetItemId};
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
    for e in entries {
        buf.extend_from_slice(&e.code.packed_code);
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
        let sum_off = code_off + num_entries * code_len;
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

    /// Packed RaBitQ code of entry `i`.
    #[inline]
    pub fn code(&self, i: usize) -> &'a [u8] {
        &self.codes[i * self.code_len..(i + 1) * self.code_len]
    }

    /// `sum_of_x2` (residual norm squared) of entry `i`.
    #[inline]
    pub fn sum_of_x2(&self, i: usize) -> f32 {
        f32::from_le_bytes(self.sums[i * 4..(i + 1) * 4].try_into().unwrap())
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
    pub fn finish(self) -> (Option<BlockNumber>, u32, usize) {
        let total_entries = self.entries.len();
        if total_entries == 0 {
            return (None, 0, 0);
        }

        let _ = self.list_id;
        let bytes = serialize_entries(&self.entries);

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
            // One contiguous raw buffer (avoids n per-block allocations), with
            // `smgrreadv` filling it block-by-block.
            let mut raw = vec![0u8; n * blksz];
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
        (0..view.len())
            .map(|i| IvfEntry {
                heap_tid: view.tid(i),
                code: RabitqVector {
                    dim: view.dim() as u32,
                    sum_of_x2: view.sum_of_x2(i),
                    l1_of_rotated: view.l1_reconstructed(i),
                    packed_code: view.code(i).to_vec(),
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
