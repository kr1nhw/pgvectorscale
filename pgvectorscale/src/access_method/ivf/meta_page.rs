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
use crate::access_method::ivf::segment::{IvfRetiredList, IvfRetiredRange};
use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::access_method::storage::StorageType;
use crate::util::buffer::LockedBufferExclusive;
use crate::util::page::{self, PageType, ReadablePage, WritablePage};
use crate::util::*;

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
    /// Storage type (Plain, SbqCompression, RabbitqCompression)
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
    /// Pointer to the pending-reclamation list (chained item; writer-only).
    retired_list: ItemPointer,
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

    /// Get the pending-reclamation list pointer.
    pub fn get_retired_list_pointer(&self) -> Option<ItemPointer> {
        if self.retired_list.is_valid() {
            Some(self.retired_list)
        } else {
            None
        }
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
            retired_list: ItemPointer::new_invalid(),
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
            let result = rkyv::from_bytes::<IvfMetaPage>(item.get_data_slice()).unwrap();

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
                .unwrap();
        assert_eq!(meta.magic_number, IVF_MAGIC_NUMBER);
        assert_eq!(meta.version, IVF_VERSION);

        let result = f(&mut meta);
        let bytes = meta.serialize_to_vec();
        page::write_single_item_page_locked(index, &buffer, PageType::IvfMeta, &bytes);
        result
    }

    /// Append block ranges to the pending-reclamation list (serialized by the
    /// meta page's content lock, so concurrent seals/compactions cannot lose
    /// each other's ranges).
    pub unsafe fn retire_ranges(index: &PgRelation, ranges: Vec<IvfRetiredRange>) {
        if ranges.is_empty() {
            return;
        }
        Self::update(index, |meta| {
            let mut list = match meta.get_retired_list_pointer() {
                Some(p) => IvfRetiredList::load(index, p),
                None => IvfRetiredList::new(),
            };
            list.ranges.extend(ranges);
            meta.retired_list = list.store(index);
        });
    }
}
