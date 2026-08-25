//! IVF index metadata management.
//!
//! The IVF meta page (Page 0) stores global index metadata including pointers
//! to centroids, list directory, and quantizer metadata.

use pgrx::pg_sys::{InvalidBlockNumber, InvalidOffsetNumber};
use pgrx::*;
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{Archive, Deserialize, Serialize};
use semver::Version;

use crate::access_method::distance::DistanceType;
use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::access_method::storage::StorageType;
use crate::util::chain::{ChainItemReader, ChainTapeWriter};
use crate::util::page::{self, PageType};
use crate::util::*;

const IVF_MAGIC_NUMBER: u32 = 0x49564600; // "IVF\0"
const IVF_VERSION: u32 = 1;

const META_BLOCK_NUMBER: pg_sys::BlockNumber = 0;
const META_HEADER_OFFSET: pgrx::pg_sys::OffsetNumber = 1;
const META_OFFSET: pgrx::pg_sys::OffsetNumber = 2;

/// IVF metadata header. Contains magic number and version for sanity checks.
/// Stored at offset 1 in the meta page.
#[derive(Clone, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct IvfMetaPageHeader {
    /// Magic number for identifying IVF index
    magic_number: u32,
    /// Version number for future-proofing
    version: u32,
}

/// IVF metadata about the entire index.
/// Stored at offset 2 in the meta page (Page 0).
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct IvfMetaPage {
    /// Magic number from header for sanity check
    magic_number: u32,
    /// Version number from header for sanity check
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
    /// Pointer to centroids page (Page 2)
    centroids_pointer: ItemPointer,
    /// Pointer to list directory page (Page 1)
    list_directory_pointer: ItemPointer,
    /// Pointer to quantizer metadata page (Page 3)
    quantizer_metadata: ItemPointer,
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

    /// Create a new IVF meta page and write it to the index.
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
            centroids_pointer: ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber),
            list_directory_pointer: ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber),
            quantizer_metadata: ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber),
        };

        meta.store(index, true);
        meta
    }

    /// Write the meta page to the index.
    pub unsafe fn store(&self, index: &PgRelation, first_time: bool) {
        let header = IvfMetaPageHeader {
            magic_number: self.magic_number,
            version: self.version,
        };

        assert_eq!(header.magic_number, IVF_MAGIC_NUMBER);
        assert_eq!(header.version, IVF_VERSION);

        let mut stats = crate::access_method::stats::WriteStats::default();
        let mut tape = if first_time {
            ChainTapeWriter::new(index, PageType::IvfMeta, &mut stats)
        } else {
            ChainTapeWriter::reinit(index, PageType::IvfMeta, &mut stats, META_BLOCK_NUMBER)
        };

        // Serialize the header
        let bytes = header.serialize_to_vec();
        let off = tape.write(&bytes);
        assert_eq!(off, ItemPointer::new(META_BLOCK_NUMBER, META_HEADER_OFFSET));

        // Serialize the meta
        let bytes = self.serialize_to_vec();
        let off = tape.write(&bytes);
        assert_eq!(off, ItemPointer::new(META_BLOCK_NUMBER, META_OFFSET));
    }

    /// Read the meta page from the index.
    pub fn fetch(index: &PgRelation) -> IvfMetaPage {
        unsafe {
            let mut stats = crate::access_method::stats::WriteStats::default();
            let mut tape = ChainItemReader::new(index, PageType::IvfMeta, &mut stats);

            let mut buf: Vec<u8> = Vec::new();
            for item in tape.read(ItemPointer::new(META_BLOCK_NUMBER, META_OFFSET)) {
                buf.extend_from_slice(item.get_data_slice());
            }
            let result = rkyv::from_bytes::<IvfMetaPage>(&buf).unwrap();

            // Verify magic number and version
            assert_eq!(result.magic_number, IVF_MAGIC_NUMBER);
            assert_eq!(result.version, IVF_VERSION);

            result
        }
    }
}
