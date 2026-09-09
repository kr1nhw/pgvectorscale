//! IVF index metadata management.
//!
//! The IVF meta page (Page 0) stores global index metadata including pointers
//! to centroids, list directory, and quantizer metadata.  It is a single-item
//! page (item at offset 1) so it can be updated atomically under its buffer
//! content lock via [`IvfMetaPage::update`].

use pgrx::*;
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{Archive, Deserialize, Serialize};
use semver::Version;

use crate::access_method::distance::DistanceType;
use crate::access_method::ivf::segment::{IvfFreeList, IvfFreeRange};
use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::access_method::storage::StorageType;
use crate::util::buffer::{AdvisoryLockGuard, LockedBufferExclusive};
use crate::util::chain::ChainTapeWriter;
use crate::util::page::{self, PageType, ReadablePage, WritablePage};
use crate::util::*;

/// Advisory-lock keys for the IVF read-burst (shared) / reclamation
/// (exclusive) protocol.  key1 is a fixed magic so we never collide with
/// other advisory-lock users; key2 is the index OID.
pub fn advisory_keys(index: &PgRelation) -> (i64, i64) {
    (0x4956_4630_i64, u32::from(index.oid()) as i64)
}

const IVF_MAGIC_NUMBER: u32 = 0x49564600; // "IVF\0"
const IVF_VERSION: u32 = 2;

const META_BLOCK_NUMBER: pg_sys::BlockNumber = 0;
const META_OFFSET: pgrx::pg_sys::OffsetNumber = 1;

/// IVF metadata about the entire index.
/// Stored as the only item of the meta page (Page 0).
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct IvfMetaPage {
    /// Magic number for sanity check
    magic_number: u32,
    /// Version number for future-proofing
    version: u32,
    /// Version of the extension when the index was built
    extension_version_when_built: String,
    /// Distance type (L2, Cosine, InnerProduct)
    distance_type: u16,
    /// Number of vector dimensions
    num_dimensions: u32,
    /// Storage type (Plain, SbqCompression, RabitqCompression)
    storage_type: u8,
    /// Number of inverted lists (centroids)
    lists: u16,
    /// Number of bits per dimension for quantization (SBQ/RaBitQ)
    bq_num_bits_per_dimension: u8,
    /// Rotation seed for RaBitQ (deterministic random rotation).
    rotation_seed: u64,
    /// Pointer to centroids page
    centroids_pointer: ItemPointer,
    /// Pointer to list directory page (Page 1)
    list_directory_pointer: ItemPointer,
    /// Pointer to quantizer metadata page
    quantizer_metadata: ItemPointer,
    /// Pointer to the free block list (chained item; writer-only, accessed
    /// under the relation ExclusiveLock).
    free_list: ItemPointer,
    /// Number of contiguous blocks the free-list item occupies (0 = unknown /
    /// not retireable).
    free_list_blocks: u32,
    /// Monotonic generation counter (bumped on compaction; used by reclamation).
    generation: u64,
}

impl IvfMetaPage {
    /// Get the number of dimensions in the vectors.
    pub fn get_num_dimensions(&self) -> u32 {
        self.num_dimensions
    }

    /// Get the distance type.
    pub fn get_distance_type(&self) -> DistanceType {
        DistanceType::from_u16(self.distance_type)
    }

    /// Get the storage type.
    pub fn get_storage_type(&self) -> StorageType {
        StorageType::from_u8(self.storage_type)
    }

    /// Get the number of inverted lists (centroids).
    pub fn get_lists(&self) -> u16 {
        self.lists
    }

    /// Get the number of bits per dimension for quantization.
    pub fn get_bq_num_bits_per_dimension(&self) -> u8 {
        self.bq_num_bits_per_dimension
    }

    /// Get the RaBitQ rotation seed.
    pub fn get_rotation_seed(&self) -> u64 {
        self.rotation_seed
    }

    /// Get pointer to centroids page.
    pub fn get_centroids_pointer(&self) -> Option<ItemPointer> {
        if self.centroids_pointer.is_valid() {
            Some(self.centroids_pointer)
        } else {
            None
        }
    }

    /// Get pointer to list directory page.
    pub fn get_list_directory_pointer(&self) -> Option<ItemPointer> {
        if self.list_directory_pointer.is_valid() {
            Some(self.list_directory_pointer)
        } else {
            None
        }
    }

    /// Get pointer to quantizer metadata page.
    pub fn get_quantizer_metadata_pointer(&self) -> Option<ItemPointer> {
        if self.quantizer_metadata.is_valid() {
            Some(self.quantizer_metadata)
        } else {
            None
        }
    }

    /// Get the reclamation generation counter.
    pub fn get_generation(&self) -> u64 {
        self.generation
    }

    /// Set pointer to centroids page.
    pub fn set_centroids_pointer(&mut self, pointer: ItemPointer) {
        self.centroids_pointer = pointer;
    }

    /// Set pointer to list directory page.
    pub fn set_list_directory_pointer(&mut self, pointer: ItemPointer) {
        self.list_directory_pointer = pointer;
    }

    /// Set pointer to quantizer metadata page.
    pub fn set_quantizer_metadata_pointer(&mut self, pointer: ItemPointer) {
        self.quantizer_metadata = pointer;
    }

    /// Get the free-list pointer.
    pub fn get_free_list_pointer(&self) -> Option<ItemPointer> {
        if self.free_list.is_valid() {
            Some(self.free_list)
        } else {
            None
        }
    }

    /// Get the free-list item's block count (0 = unknown).
    pub fn get_free_list_blocks(&self) -> u32 {
        self.free_list_blocks
    }

    /// Create a new IVF meta page and write it to block 0 of the index.
    pub unsafe fn create(
        index: &PgRelation,
        num_dimensions: u32,
        distance_type: DistanceType,
        lists: u16,
        storage_type: StorageType,
        bq_num_bits_per_dimension: u8,
        rotation_seed: u64,
    ) -> IvfMetaPage {
        let version = Version::parse(env!("CARGO_PKG_VERSION")).unwrap();

        let meta = IvfMetaPage {
            magic_number: IVF_MAGIC_NUMBER,
            version: IVF_VERSION,
            extension_version_when_built: version.to_string(),
            distance_type: distance_type as u16,
            num_dimensions,
            storage_type: storage_type as u8,
            lists,
            bq_num_bits_per_dimension,
            rotation_seed,
            centroids_pointer: ItemPointer::new_invalid(),
            list_directory_pointer: ItemPointer::new_invalid(),
            quantizer_metadata: ItemPointer::new_invalid(),
            free_list: ItemPointer::new_invalid(),
            free_list_blocks: 0,
            generation: 0,
        };

        meta.store(index, true);
        meta
    }

    /// Write the meta page to the index.  `first_time` writes a fresh page at
    /// block 0; otherwise the existing page is rewritten in place.
    pub unsafe fn store(&self, index: &PgRelation, first_time: bool) {
        assert_eq!(self.magic_number, IVF_MAGIC_NUMBER);
        assert_eq!(self.version, IVF_VERSION);

        let bytes = self.serialize_to_vec();
        if first_time {
            let mut page = WritablePage::new(index, PageType::IvfMeta);
            let block = page.get_block_number();
            let off = page.add_item(&bytes);
            page.commit();
            assert_eq!(block, META_BLOCK_NUMBER, "meta page must be block 0");
            assert_eq!(off, META_OFFSET);
        } else {
            let mut page = WritablePage::modify(index, META_BLOCK_NUMBER);
            page.reinit(PageType::IvfMeta);
            page.add_item(&bytes);
            page.commit();
        }
    }

    /// Read the meta page from the index.
    pub fn fetch(index: &PgRelation) -> IvfMetaPage {
        unsafe {
            let page = ReadablePage::read(index, META_BLOCK_NUMBER);
            assert!(page.get_type() == PageType::IvfMeta);
            let item = page.get_item_unchecked(META_OFFSET);
            let result = rkyv::from_bytes::<IvfMetaPage>(item.get_data_slice())
                .unwrap_or_else(|e| panic!("IVF: meta fetch parse failed: {:?}", e));

            // Verify magic number and version
            assert_eq!(result.magic_number, IVF_MAGIC_NUMBER);
            assert_eq!(result.version, IVF_VERSION);

            result
        }
    }

    /// Read-modify-write the meta page under its exclusive content lock.  The
    /// closure receives the current meta (parsed under the lock) and may do
    /// arbitrary work (e.g. appending to the retired list); the page is
    /// rewritten in place (WAL-logged) after the closure returns.
    pub unsafe fn update<R, F: FnOnce(&mut IvfMetaPage) -> R>(index: &PgRelation, f: F) -> R {
        let buffer = LockedBufferExclusive::read(index, META_BLOCK_NUMBER);
        let page = pg_sys::BufferGetPage(**&buffer);
        let item_id = crate::util::ports::PageGetItemId(page, META_OFFSET);
        let item = crate::util::ports::PageGetItem(page, item_id);
        let len = (*item_id).lp_len() as usize;
        let mut meta =
            rkyv::from_bytes::<IvfMetaPage>(std::slice::from_raw_parts(item as *const u8, len))
                .unwrap_or_else(|e| panic!("IVF: meta update parse failed: {:?}", e));
        assert_eq!(meta.magic_number, IVF_MAGIC_NUMBER);
        assert_eq!(meta.version, IVF_VERSION);

        let result = f(&mut meta);
        let bytes = meta.serialize_to_vec();
        page::write_single_item_page_locked(index, &buffer, PageType::IvfMeta, &bytes);
        result
    }

    /// Republish the free list, rewriting the previous item IN PLACE.
    ///
    /// The free list is only ever read under the advisory-exclusive lock (no
    /// scan walks it), so rewriting the previous item's first block in place
    /// cannot tear a reader, and the per-update block churn disappears.  If
    /// the item outgrows the previous chain, the extra pages extend the
    /// relation; a shrunk chain leaks its leftover tail pages (rare, bounded).
    unsafe fn republish_free_list(index: &PgRelation, list: &IvfFreeList, meta: &mut IvfMetaPage) {
        // Coalesce adjacent ranges so the list does not fragment into
        // 1-block runs that can never satisfy multi-block reservations.
        let mut sorted: Vec<IvfFreeRange> = list.ranges.clone();
        sorted.sort_unstable_by_key(|r| r.start_block);
        let mut coalesced: Vec<IvfFreeRange> = Vec::with_capacity(sorted.len());
        for r in sorted {
            match coalesced.last_mut() {
                Some(last)
                    if last.start_block + last.num_blocks as pg_sys::BlockNumber
                        == r.start_block =>
                {
                    last.num_blocks += r.num_blocks;
                }
                _ => coalesced.push(r),
            }
        }
        let list = IvfFreeList { ranges: coalesced };
        let bytes = list.serialize_to_vec();
        let mut stats = crate::access_method::stats::WriteStats::default();
        let (ptr, blocks) = match meta.get_free_list_pointer() {
            Some(old_ptr) => {
                let mut tape =
                    ChainTapeWriter::reinit(index, PageType::IvfFreeList, &mut stats, old_ptr.block_number);
                tape.write_counted(&bytes)
            }
            None => {
                let mut tape = ChainTapeWriter::new(index, PageType::IvfFreeList, &mut stats);
                tape.write_counted(&bytes)
            }
        };
        meta.free_list = ptr;
        meta.free_list_blocks = blocks;
    }

    /// Append retired block ranges to the free list.
    ///
    /// Takes the relation ExclusiveLock: scans hold ShareLock for the duration
    /// of their `smgrreadv` burst, so by the time the ExclusiveLock is granted
    /// no scan is still reading the retired blocks, and they can safely become
    /// reusable.  (New scans can never reference them: they were swapped out of
    /// the published header before being retired.)
    pub unsafe fn reclaim_ranges(index: &PgRelation, ranges: Vec<IvfFreeRange>) {
        if ranges.is_empty() {
            return;
        }
        let (k1, k2) = advisory_keys(index);
        let _guard = AdvisoryLockGuard::acquire_exclusive(k1, k2);
        Self::update(index, |meta| {
            let mut list = match meta.get_free_list_pointer() {
                Some(p) => IvfFreeList::load(index, p),
                None => IvfFreeList::new(),
            };
            list.ranges.extend(ranges);
            Self::republish_free_list(index, &list, meta);
        });
    }

    /// Reserve a free range of at least `need` blocks for reuse, returning its
    /// start block (the range is split if larger than `need`).  Returns `None`
    /// if no range is large enough (callers fall back to relation extension).
    ///
    /// Serialized against scans the same way as [`Self::reclaim_ranges`]: the
    /// ExclusiveLock guarantees no scan is mid-read of the returned blocks,
    /// and scans started later never reference them (they are unreachable from
    /// every published header).
    pub unsafe fn allocate_range(index: &PgRelation, need: u32) -> Option<pg_sys::BlockNumber> {
        if need == 0 {
            return None;
        }
        // Fast path: no free list yet → fall back to extension without taking
        // the ExclusiveLock (which would otherwise stall behind every
        // concurrent inserter's RowExclusiveLock on the index relation).
        if Self::fetch(index).get_free_list_pointer().is_none() {
            return None;
        }
        let (k1, k2) = advisory_keys(index);
        let _guard = AdvisoryLockGuard::acquire_exclusive(k1, k2);
        Self::update(index, |meta| {
            let mut list = match meta.get_free_list_pointer() {
                Some(p) => IvfFreeList::load(index, p),
                None => IvfFreeList::new(),
            };
            let mut found: Option<pg_sys::BlockNumber> = None;
            if let Some(pos) = list.ranges.iter().position(|r| r.num_blocks >= need) {
                let range = list.ranges.remove(pos);
                if range.num_blocks > need {
                    // Split: push the tail back.
                    list.ranges.push(IvfFreeRange {
                        start_block: range.start_block + need as pg_sys::BlockNumber,
                        num_blocks: range.num_blocks - need,
                    });
                }
                found = Some(range.start_block);
                Self::republish_free_list(index, &list, meta);
            }
            found
        })
    }

    /// Return an unused reserved range to the free list.
    pub unsafe fn push_back_range(
        index: &PgRelation,
        start_block: pg_sys::BlockNumber,
        num_blocks: u32,
    ) {
        if num_blocks == 0 {
            return;
        }
        let (k1, k2) = advisory_keys(index);
        let _guard = AdvisoryLockGuard::acquire_exclusive(k1, k2);
        Self::update(index, |meta| {
            let mut list = match meta.get_free_list_pointer() {
                Some(p) => IvfFreeList::load(index, p),
                None => IvfFreeList::new(),
            };
            list.ranges.push(IvfFreeRange {
                start_block,
                num_blocks,
            });
            Self::republish_free_list(index, &list, meta);
        });
    }
}
