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
    if num_bits == 1 || num_bits == 2 {
        // 1-bit codes are stored transposed (32-row batches) for SIMD
        // FastScan.  2-bit codes are split into their sign and ex bit-planes
        // (each a plain 1-bit-style code of `code_len/2` bytes per vector)
        // and both planes are transposed: plane0 then plane1.
        let plane_len = if num_bits == 1 { code_len } else { code_len / 2 };
        if num_bits == 1 {
            let mut row_major = Vec::with_capacity(n * code_len);
            for e in entries {
                row_major.extend_from_slice(&e.code.packed_code);
            }
            let transposed = crate::access_method::quantization::rabitq_fastscan::transpose_1bit(
                &row_major, n, code_len,
            );
            buf.extend_from_slice(&transposed);
        } else {
            // Pre-size both planes once and write each entry directly into its
            // slot instead of allocating a fresh sp/ep pair per entry.
            let mut sign_plane = vec![0u8; n * plane_len];
            let mut ex_plane = vec![0u8; n * plane_len];
            for (i, e) in entries.iter().enumerate() {
                let sp = &mut sign_plane[i * plane_len..(i + 1) * plane_len];
                let ep = &mut ex_plane[i * plane_len..(i + 1) * plane_len];
                for (byte_idx, &b) in e.code.packed_code.iter().enumerate() {
                    for j in 0..4usize {
                        let d = byte_idx * 4 + j;
                        if b & (1 << (2 * j)) != 0 {
                            sp[d / 8] |= 1 << (d % 8);
                        }
                        if b & (1 << (2 * j + 1)) != 0 {
                            ep[d / 8] |= 1 << (d % 8);
                        }
                    }
                }
            }
            buf.extend_from_slice(&crate::access_method::quantization::rabitq_fastscan::transpose_1bit(
                &sign_plane, n, plane_len,
            ));
            buf.extend_from_slice(&crate::access_method::quantization::rabitq_fastscan::transpose_1bit(
                &ex_plane, n, plane_len,
            ));
        }
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
        // 1-bit: transposed 32-row batches.  2-bit: the sign and ex planes
        // are each transposed, so the combined region is also
        // ceil(n/32)·32·code_len (plane_len = code_len/2, two planes).
        let codes_len = if num_bits == 1 || num_bits == 2 {
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

    /// Transposed sign-plane of 2-bit batch `batch` (32 × code_len/2 bytes).
    /// Plane 0 occupies the FIRST half of the codes region.
    #[inline]
    pub fn code_batch_plane0(&self, batch: usize) -> &'a [u8] {
        let half = self.code_len / 2;
        let start = batch * 32 * half;
        &self.codes[start..start + 32 * half]
    }

    /// Transposed ex-plane of 2-bit batch `batch` (32 × code_len/2 bytes).
    /// Plane 1 occupies the SECOND half of the codes region.
    #[inline]
    pub fn code_batch_plane1(&self, batch: usize) -> &'a [u8] {
        let half = self.code_len / 2;
        let plane_bytes = self.num_batches() * 32 * half;
        let start = plane_bytes + batch * 32 * half;
        &self.codes[start..start + 32 * half]
    }

    /// `sum_of_x2` (residual norm squared) of entry `i`.
    #[inline]
    pub fn sum_of_x2(&self, i: usize) -> f32 {
        f32::from_le_bytes(self.sums[i * 4..(i + 1) * 4].try_into().unwrap())
    }

    /// Copy the packed little-endian f32 region starting at entry `base` for
    /// `out.len()` entries into the caller-provided (properly aligned) `out`.
    ///
    /// The SoA stream is tightly packed with no alignment padding, so the
    /// f32 regions can start at any byte offset (e.g. an odd entry count
    /// misaligns them); casting them to `&[f32]` in place would violate
    /// f32's 4-byte alignment.  Copying the raw bytes into the aligned
    /// destination is sound: every bit pattern is a valid f32, and the write
    /// goes through a u8 pointer into a fully initialized `&mut [f32]`.
    #[inline]
    fn copy_f32_region(src: &'a [u8], base: usize, out: &mut [f32]) {
        assert!(base + out.len() <= src.len() / 4, "f32 region out of bounds");
        // SAFETY: `src[base*4 ..]` holds `out.len()*4` initialized bytes
        // (asserted above); writing them into `out` through a u8 pointer is
        // valid because all f32 bit patterns are valid values.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr().add(base * 4),
                out.as_mut_ptr() as *mut u8,
                out.len() * 4,
            );
        }
    }

    /// Copy `out.len()` `sum_of_x2` values starting at entry `base` (see
    /// `copy_f32_region` for the alignment rationale).
    #[inline]
    pub fn copy_sums(&self, base: usize, out: &mut [f32]) {
        Self::copy_f32_region(self.sums, base, out);
    }

    /// Copy `out.len()` precomputed `scale` values starting at entry `base`.
    #[inline]
    pub fn copy_scales(&self, base: usize, out: &mut [f32]) {
        Self::copy_f32_region(self.scales, base, out);
    }

    /// Copy `out.len()` precomputed `margin_factor` values starting at entry
    /// `base`.
    #[inline]
    pub fn copy_margin_factors(&self, base: usize, out: &mut [f32]) {
        Self::copy_f32_region(self.margin_factors, base, out);
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
        2 => dim_padded as usize / 4,
        4 => dim_padded as usize / 2,
        _ => dim_padded as usize,
    };
    // 1-bit and 2-bit codes are stored transposed (2-bit as two planes, which
    // sums to the same ceil(n/32)·32·code_len total).
    let codes = if num_bits == 1 || num_bits == 2 {
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
                // Neon's fork extends `smgropen` with a `relpersistence`
                // argument; vanilla PostgreSQL does not.
                #[cfg(feature = "neon")]
                {
                    pg_sys::smgropen(
                        (*rel).rd_locator,
                        (*rel).rd_backend,
                        (*(*rel).rd_rel).relpersistence,
                    )
                }
                #[cfg(not(feature = "neon"))]
                {
                    pg_sys::smgropen((*rel).rd_locator, (*rel).rd_backend)
                }
            } else {
                reln
            };

            let n = num_blocks as usize;
            let blksz = pg_sys::BLCKSZ as usize;
            // One contiguous raw buffer, *uninitialized*: `smgrreadv` overwrites
            // every byte, so zero-filling it would be pure wasted memset.
            let mut raw: Vec<u8> = Vec::with_capacity(n * blksz);
            raw.set_len(n * blksz);

            // The smgr vtable caps one vectored read.  Vanilla md allows
            // PG_IOV_MAX (=IOV_MAX, 1024 on Linux), but Neon's fork defines
            // PG_IOV_MAX as Min(IOV_MAX, 32) because a single pagestore
            // request is bounded; its `neon_readv` errors above that.  Chunk
            // the burst so the same code works under both smgr backends.
            #[cfg(feature = "neon")]
            let max_burst: usize = 32; // Neon fork: PG_IOV_MAX = Min(IOV_MAX, 32)
            #[cfg(not(feature = "neon"))]
            let max_burst: usize = 1024; // vanilla: IOV_MAX on Linux/macOS

            let mut off = 0usize;
            while off < n {
                let chunk = (n - off).min(max_burst);
                let mut ptrs: Vec<*mut std::os::raw::c_void> = (0..chunk)
                    .map(|i| {
                        raw.as_mut_ptr().add((off + i) * blksz) as *mut std::os::raw::c_void
                    })
                    .collect();
                pg_sys::smgrreadv(
                    reln,
                    pg_sys::ForkNumber::MAIN_FORKNUM,
                    start_page + off as BlockNumber,
                    ptrs.as_mut_ptr(),
                    chunk as BlockNumber,
                );
                off += chunk;
            }

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
        // Transposed codes are recovered to row-major for the in-memory
        // IvfEntry (the rewrite path re-transposes on serialize): 1-bit is a
        // single plane; 2-bit re-interleaves its two planes into the packed
        // 4-dims-per-byte form.
        let row_major: Vec<u8> = if view.num_bits() == 1 {
            crate::access_method::quantization::rabitq_fastscan::untranspose_1bit(
                view.codes(),
                view.len(),
                view.code_len(),
            )
        } else if view.num_bits() == 2 {
            let half = view.code_len() / 2;
            let n_batches = view.num_batches();
            let plane_bytes = n_batches * 32 * half;
            let p0 = crate::access_method::quantization::rabitq_fastscan::untranspose_1bit(
                &view.codes()[..plane_bytes],
                view.len(),
                half,
            );
            let p1 = crate::access_method::quantization::rabitq_fastscan::untranspose_1bit(
                &view.codes()[plane_bytes..plane_bytes * 2],
                view.len(),
                half,
            );
            let mut packed = vec![0u8; view.len() * view.code_len()];
            for i in 0..view.len() {
                for d in 0..half * 8 {
                    let sign = (p0[i * half + d / 8] >> (d % 8)) & 1;
                    let ex = (p1[i * half + d / 8] >> (d % 8)) & 1;
                    packed[i * view.code_len() + d / 4] |=
                        (sign << (2 * (d % 4))) | (ex << (2 * (d % 4) + 1));
                }
            }
            packed
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
                    packed_code: if view.num_bits() == 1 || view.num_bits() == 2 {
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

#[cfg(test)]
mod two_bit_soa_tests {
    use super::*;
    use crate::access_method::quantization::rabitq::{padded_dim, RabitqQuantizer};

    #[test]
    fn two_bit_soa_roundtrip_and_sizing() {
        let dim = 128usize;
        let q = RabitqQuantizer::new(2, 5, dim);
        let mut entries = Vec::new();
        for i in 0..40usize {
            let v: Vec<f32> = (0..dim).map(|d| ((i * 13 + d * 7) % 23) as f32 - 11.0).collect();
            entries.push(IvfEntry {
                heap_tid: ItemPointer::new(i as u32, 1),
                code: q.quantize(&v),
            });
        }

        let bytes = serialize_entries(&entries);
        assert_eq!(
            bytes.len(),
            seal_bytes_len(entries.len(), 2, padded_dim(dim) as u32),
            "seal sizing must match the serialized length"
        );

        let view = IvfEntrySlice::parse(&bytes);
        assert_eq!(view.num_bits(), 2);
        assert_eq!(view.len(), 40);
        assert_eq!(view.num_batches(), 40usize.div_ceil(32));

        // Round-trip through the zero-copy parse → read path.
        let parsed = {
            let mut out = Vec::new();
            for i in 0..view.len() {
                let (tid, code) = (view.tid(i), view.code(i));
                let _ = (tid, code);
            }
            for i in 0..view.len() {
                out.push(view.tid(i));
            }
            out
        };
        assert_eq!(parsed.len(), 40);
        assert_eq!(parsed[7].block_number, 7);
    }
}

#[cfg(test)]
mod two_bit_plane_slice_tests {
    use super::*;
    use crate::access_method::quantization::rabitq::{padded_dim, RabitqQuantizer};
    use crate::access_method::quantization::rabitq_fastscan::transpose_1bit;

    #[test]
    fn two_bit_plane_slices_match_manual_transpose() {
        let dim = 128usize;
        let q = RabitqQuantizer::new(2, 5, dim);
        let n = 100usize; // > 32 → multiple batches
        let mut entries = Vec::new();
        let mut sign_planes = vec![0u8; n * (dim / 8)];
        let mut ex_planes = vec![0u8; n * (dim / 8)];
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|d| ((i * 13 + d * 7) % 23) as f32 - 11.0).collect();
            let code = q.quantize(&v);
            let sp = &mut sign_planes[i * (dim / 8)..(i + 1) * (dim / 8)];
            let ep = &mut ex_planes[i * (dim / 8)..(i + 1) * (dim / 8)];
            for (byte_idx, &b) in code.packed_code.iter().enumerate() {
                for j in 0..4usize {
                    let d = byte_idx * 4 + j;
                    if b & (1 << (2 * j)) != 0 {
                        sp[d / 8] |= 1 << (d % 8);
                    }
                    if b & (1 << (2 * j + 1)) != 0 {
                        ep[d / 8] |= 1 << (d % 8);
                    }
                }
            }
            entries.push(IvfEntry { heap_tid: ItemPointer::new(i as u32, 1), code });
        }
        let bytes = serialize_entries(&entries);
        let view = IvfEntrySlice::parse(&bytes);
        let half = view.code_len() / 2;

        // Manually transpose plane0 and compare with the parse slices per batch.
        let p0 = transpose_1bit(&sign_planes, n, half);
        let p1 = transpose_1bit(&ex_planes, n, half);
        for batch in 0..view.num_batches() {
            assert_eq!(
                view.code_batch_plane0(batch),
                &p0[batch * 32 * half..(batch + 1) * 32 * half],
                "plane0 batch {} mismatch",
                batch
            );
            assert_eq!(
                view.code_batch_plane1(batch),
                &p1[batch * 32 * half..(batch + 1) * 32 * half],
                "plane1 batch {} mismatch",
                batch
            );
        }
        // Full round-trip through the read_entries interleave logic.
        let row_major: Vec<u8> = {
            let plane_bytes = view.num_batches() * 32 * half;
            let p0 = crate::access_method::quantization::rabitq_fastscan::untranspose_1bit(
                &view.codes()[..plane_bytes], n, half,
            );
            let p1 = crate::access_method::quantization::rabitq_fastscan::untranspose_1bit(
                &view.codes()[plane_bytes..plane_bytes * 2], n, half,
            );
            let mut packed = vec![0u8; n * view.code_len()];
            for i in 0..n {
                for d in 0..half * 8 {
                    let sign = (p0[i * half + d / 8] >> (d % 8)) & 1;
                    let ex = (p1[i * half + d / 8] >> (d % 8)) & 1;
                    packed[i * view.code_len() + d / 4] |=
                        (sign << (2 * (d % 4))) | (ex << (2 * (d % 4) + 1));
                }
            }
            packed
        };
        for i in 0..n {
            assert_eq!(
                &row_major[i * view.code_len()..(i + 1) * view.code_len()],
                &entries[i].code.packed_code[..],
                "entry {} code round-trip mismatch",
                i
            );
        }
    }
}

#[cfg(test)]
mod f32_region_alignment_tests {
    use super::*;
    use crate::access_method::quantization::rabitq::{padded_dim, RabitqQuantizer};

    /// Odd entry counts leave the packed f32 regions (sums/scales/margin
    /// factors) at 2-mod-4 byte offsets — the case the old `from_raw_parts`
    /// casts got wrong.  The byte-copy accessors must agree with the scalar
    /// per-entry readers for every entry, for every num_bits width.
    #[test]
    fn copy_accessors_match_scalar_for_misaligned_layouts() {
        for num_bits in [1u8, 2u8, 4u8, 8u8] {
            for n in [1usize, 3usize, 33usize, 100usize] {
                let dim = 128usize;
                let q = RabitqQuantizer::new(num_bits, 5, dim);
                let entries: Vec<IvfEntry> = (0..n)
                    .map(|i| {
                        let v: Vec<f32> =
                            (0..dim).map(|d| ((i * 13 + d * 7) % 23) as f32 - 11.0).collect();
                        IvfEntry {
                            heap_tid: ItemPointer::new(i as u32, 1),
                            code: q.quantize(&v),
                        }
                    })
                    .collect();

                let bytes = serialize_entries(&entries);
                let view = IvfEntrySlice::parse(&bytes);
                assert_eq!(view.len(), n);

                // Assert the misalignment the test is about: the sums region
                // offset (16 + 6n + codes_len) mod 4 — the old cast was UB
                // whenever this is non-zero.
                let codes_len = if num_bits == 1 || num_bits == 2 {
                    n.div_ceil(32) * 32 * view.code_len()
                } else {
                    n * view.code_len()
                };
                let sums_off = ENTRY_HEADER_SIZE + n * TID_SIZE + codes_len;
                if num_bits == 1 {
                    // 1-bit: 32-row transposed batches force codes_len to a
                    // multiple of 4; 6n misaligns when n is odd.
                    assert_eq!(sums_off % 4, if n % 2 == 0 { 0 } else { 2 });
                }

                for base in (0..n).step_by(7) {
                    let count = (n - base).min(5);
                    let mut scales = [0f32; 5];
                    let mut sx2s = [0f32; 5];
                    let mut mfs = [0f32; 5];
                    view.copy_scales(base, &mut scales[..count]);
                    view.copy_sums(base, &mut sx2s[..count]);
                    view.copy_margin_factors(base, &mut mfs[..count]);
                    for r in 0..count {
                        assert_eq!(scales[r], view.scale(base + r));
                        assert_eq!(sx2s[r], view.sum_of_x2(base + r));
                        assert_eq!(mfs[r], view.margin_factor(base + r));
                    }
                }
            }
        }
    }
}
