//! The AgentVec meta page (block 0).
//!
//! It holds the index-wide state that changes only when the *set* of segments
//! changes — never on a plain insert:
//!
//! ```text
//! magic, version, extension version
//! distance type, dimensions
//! directory pointer (+ block count)   -> block 1, republished copy-on-write
//! next_segment_id, hot_segment_id
//! epoch, generation, num_tuples
//! ```
//!
//! The page is a single-item page (item at offset 1) so it can be rewritten
//! atomically under its buffer content lock.  The lock protocol is strictly
//! **meta → segment header**: a plain INSERT only ever takes a segment
//! header's lock (so inserts do not serialize on the meta page), while the
//! rare paths that create or seal a segment take the meta lock first and the
//! header lock inside it.  Nothing ever takes the meta lock while holding a
//! header lock, which is what keeps the order deadlock-free.

use pgrx::*;
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{Archive, Deserialize, Serialize};
use semver::Version;

use crate::access_method::distance::DistanceType;
use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::util::buffer::LockedBufferExclusive;
use crate::util::page::{self, PageType, ReadablePage, WritablePage};
use crate::util::*;

/// Block holding the meta page.
pub const META_BLOCK_NUMBER: pg_sys::BlockNumber = 0;
/// Offset of the meta item within its page.
pub const META_OFFSET: pg_sys::OffsetNumber = 1;

const AGENTVEC_MAGIC_NUMBER: u32 = 0x4156_4543; // "AVEC"
const AGENTVEC_FORMAT_VERSION: u32 = 1;

/// Index-wide AgentVec metadata.
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct AgentVecMetaPage {
    /// Magic number for sanity checks.
    magic_number: u32,
    /// On-disk format version.
    version: u32,
    /// Extension version that built the index.
    extension_version_when_built: String,
    /// Distance metric (`DistanceType` as u16).
    distance_type: u16,
    /// Number of vector dimensions.
    num_dimensions: u32,
    /// Pointer to the current published directory item.
    directory: ItemPointer,
    /// Blocks the current directory item occupies (0 = unknown).
    directory_blocks: u32,
    /// Id to hand to the next segment created.
    next_segment_id: u64,
    /// Id of the segment foreground inserts currently target.
    hot_segment_id: u64,
    /// Monotonic publication epoch (design §7.1).
    epoch: u64,
    /// Monotonic generation of the whole index.
    generation: u64,
    /// Rows the index counts as indexed (maintained approximately).
    num_tuples: u64,
}

impl AgentVecMetaPage {
    /// Create the meta page for a new index and write it to block 0.
    pub unsafe fn create(
        index: &PgRelation,
        num_dimensions: u32,
        distance_type: DistanceType,
    ) -> AgentVecMetaPage {
        let version = Version::parse(env!("CARGO_PKG_VERSION")).unwrap();
        let meta = AgentVecMetaPage {
            magic_number: AGENTVEC_MAGIC_NUMBER,
            version: AGENTVEC_FORMAT_VERSION,
            extension_version_when_built: version.to_string(),
            distance_type: distance_type as u16,
            num_dimensions,
            directory: ItemPointer::new_invalid(),
            directory_blocks: 0,
            next_segment_id: 1,
            hot_segment_id: 0,
            epoch: 1,
            generation: 0,
            num_tuples: 0,
        };
        meta.store(index, true);
        meta
    }

    /// Write the meta page.  `first_time` writes a fresh page at block 0;
    /// otherwise the existing page is rewritten in place.
    pub unsafe fn store(&self, index: &PgRelation, first_time: bool) {
        assert_eq!(self.magic_number, AGENTVEC_MAGIC_NUMBER);
        assert_eq!(self.version, AGENTVEC_FORMAT_VERSION);

        let bytes = self.serialize_to_vec();
        if first_time {
            let mut page = WritablePage::new(index, PageType::AgentVecMeta);
            let block = page.get_block_number();
            let offset = page.add_item(&bytes);
            page.commit();
            assert_eq!(block, META_BLOCK_NUMBER, "agentvec: meta must be block 0");
            assert_eq!(offset, META_OFFSET);
        } else {
            let mut page = WritablePage::modify(index, META_BLOCK_NUMBER);
            page.reinit(PageType::AgentVecMeta);
            page.add_item(&bytes);
            page.commit();
        }
    }

    /// Read the meta page.
    pub fn fetch(index: &PgRelation) -> AgentVecMetaPage {
        unsafe {
            let page = ReadablePage::read(index, META_BLOCK_NUMBER);
            assert!(
                page.get_type() == PageType::AgentVecMeta,
                "agentvec: block 0 is not an agentvec meta page"
            );
            let item = page.get_item_unchecked(META_OFFSET);
            let result = rkyv::from_bytes::<AgentVecMetaPage>(item.get_data_slice())
                .unwrap_or_else(|e| panic!("agentvec: meta parse failed: {:?}", e));
            assert_eq!(result.magic_number, AGENTVEC_MAGIC_NUMBER);
            assert_eq!(result.version, AGENTVEC_FORMAT_VERSION);
            result
        }
    }

    /// Read-modify-write the meta page under its exclusive content lock.
    ///
    /// The closure receives the meta parsed under the lock, so it always sees
    /// the current state (a concurrent creator's result included) and can
    /// decide to do nothing — the pattern that keeps "only one transaction
    /// performs the seal" true without an extra lock.
    pub unsafe fn update<R, F: FnOnce(&mut AgentVecMetaPage) -> R>(index: &PgRelation, f: F) -> R {
        let buffer = LockedBufferExclusive::read(index, META_BLOCK_NUMBER);
        let page = pg_sys::BufferGetPage(*buffer);
        let item_id = crate::util::ports::PageGetItemId(page, META_OFFSET);
        let item = crate::util::ports::PageGetItem(page, item_id);
        let len = (*item_id).lp_len() as usize;
        let mut meta = rkyv::from_bytes::<AgentVecMetaPage>(std::slice::from_raw_parts(
            item as *const u8,
            len,
        ))
        .unwrap_or_else(|e| panic!("agentvec: meta update parse failed: {:?}", e));
        assert_eq!(meta.magic_number, AGENTVEC_MAGIC_NUMBER);
        assert_eq!(meta.version, AGENTVEC_FORMAT_VERSION);

        let result = f(&mut meta);
        let bytes = meta.serialize_to_vec();
        page::write_single_item_page_locked(index, &buffer, PageType::AgentVecMeta, &bytes);
        result
    }

    pub fn get_num_dimensions(&self) -> u32 {
        self.num_dimensions
    }

    pub fn get_distance_type(&self) -> DistanceType {
        DistanceType::from_u16(self.distance_type)
    }

    /// Pointer to the published directory item, if any.
    pub fn get_directory_pointer(&self) -> Option<ItemPointer> {
        if self.directory.is_valid() {
            Some(self.directory)
        } else {
            None
        }
    }

    /// Blocks the published directory item occupies.
    pub fn get_directory_blocks(&self) -> u32 {
        self.directory_blocks
    }

    pub fn get_next_segment_id(&self) -> u64 {
        self.next_segment_id
    }

    pub fn get_hot_segment_id(&self) -> u64 {
        self.hot_segment_id
    }

    pub fn get_epoch(&self) -> u64 {
        self.epoch
    }

    pub fn get_generation(&self) -> u64 {
        self.generation
    }

    pub fn get_num_tuples(&self) -> u64 {
        self.num_tuples
    }

    /// Record a newly published directory item.
    pub fn set_directory(&mut self, pointer: ItemPointer, blocks: u32) {
        self.directory = pointer;
        self.directory_blocks = blocks;
    }

    /// Record the segment foreground inserts target from now on.
    pub fn set_hot_segment_id(&mut self, segment_id: u64) {
        self.hot_segment_id = segment_id;
    }

    /// Consume the next segment id.
    pub fn take_next_segment_id(&mut self) -> u64 {
        let id = self.next_segment_id;
        self.next_segment_id += 1;
        id
    }

    /// Advance the publication epoch.
    pub fn bump_epoch(&mut self) -> u64 {
        self.epoch += 1;
        self.epoch
    }

    /// Advance the index generation.
    pub fn bump_generation(&mut self) -> u64 {
        self.generation += 1;
        self.generation
    }

    /// Set the approximate indexed row count.
    pub fn set_num_tuples(&mut self, num_tuples: u64) {
        self.num_tuples = num_tuples;
    }

    /// The published directory of this index.
    pub fn load_directory(&self, index: &PgRelation) -> super::directory::AgentVecDirectory {
        match self.get_directory_pointer() {
            Some(pointer) => super::directory::AgentVecDirectory::load(index, pointer),
            None => super::directory::AgentVecDirectory::new(),
        }
    }
}
