//! hnswsq metadata management.
//!
//! The meta page (block 0) is a single-item page (item at offset 1) so it can
//! be updated atomically under its buffer content lock via
//! [`HnswMetaPage::update`] — the same publication primitive the IVF meta page
//! uses.  It stores the immutable build parameters (precision, m, dimension,
//! level multiplier), the graph entry point, the append hint page, the head of
//! the threaded free-page list, and a pointer to the SQ8 calibration chain
//! item.
//!
//! Concurrency contract (see `.design/hnswsq_concurrent_graph.md`):
//! - Readers take a share content lock for the atomic single-item read.
//! - Writers use [`HnswMetaPage::update`] (pin + exclusive content lock,
//!   parse, mutate, rewrite via GenericXLog), so concurrent updates serialize
//!   and never clobber each other.
//! - The free-page list is threaded through the freed pages themselves (each
//!   freed page stores the next free block as its only item), so push/pop need
//!   no extra chain-item bookkeeping.  Push happens only from VACUUM (one per
//!   relation); pop happens inside `update` so concurrent inserters serialize
//!   on the meta page lock.

use pgrx::*;
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{Archive, Deserialize, Serialize};
use semver::Version;

use crate::access_method::distance::DistanceType;
use crate::access_method::hnswsq::quantize::HnswPrecision;
use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::util::buffer::LockedBufferExclusive;
use crate::util::page::{self, PageType, ReadablePage, WritablePage};
use crate::util::ports::{PageGetItem, PageGetItemId, PageGetMaxOffsetNumber};
use crate::util::*;

const HNSW_MAGIC_NUMBER: u32 = 0x484E_5351; // "HNSQ"
const HNSW_VERSION: u32 = 1;

const META_BLOCK_NUMBER: pg_sys::BlockNumber = 0;
const META_OFFSET: pgrx::pg_sys::OffsetNumber = 1;

/// The link item stored as the only item of a freed page (threaded free list).
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct HnswFreePageLink {
    /// Next free page, or `InvalidBlockNumber` at the tail.
    pub next: pg_sys::BlockNumber,
}

/// hnswsq metadata about the entire index.  Stored as the only item of the
/// meta page (block 0).
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct HnswMetaPage {
    /// Magic number for sanity checks.
    magic_number: u32,
    /// On-disk format version.
    version: u32,
    /// Version of the extension when the index was built.
    extension_version_when_built: String,
    /// Distance type (Cosine=0, L2=1, InnerProduct=2).
    distance_type: u16,
    /// Number of vector dimensions.
    num_dimensions: u32,
    /// Node vector precision ([`HnswPrecision`]).
    precision: u8,
    /// Max neighbors per node per upper layer.
    m: u16,
    /// Max neighbors per node at layer 0 (= 2*m).
    m0: u16,
    /// Search width during build/insert.
    ef_construction: u32,
    /// Level multiplier 1/ln(m), frozen at build.
    ml: f32,
    /// Max level a node may occupy (bounded so the largest node fits a page).
    max_level: u8,
    /// Current graph entry point (Invalid when the index is empty).
    entry_point: ItemPointer,
    /// Level of the entry point (-1 when empty).
    entry_level: i16,
    /// Live + tombstoned node count (maintained by vacuum; estimate).
    node_count: u64,
    /// Tombstoned node count (maintained by vacuum; estimate).
    deleted_count: u64,
    /// Append hint: last node page known to have free space
    /// (InvalidBlockNumber = none).
    insert_page: pg_sys::BlockNumber,
    /// Head of the threaded free-page list (InvalidBlockNumber = empty).
    free_pages_head: pg_sys::BlockNumber,
    /// Pointer to the SQ8 calibration chain item (Invalid for other layouts).
    calibration: ItemPointer,
}

impl HnswMetaPage {
    pub fn get_distance_type(&self) -> DistanceType {
        DistanceType::from_u16(self.distance_type)
    }

    pub fn get_num_dimensions(&self) -> u32 {
        self.num_dimensions
    }

    pub fn get_precision(&self) -> HnswPrecision {
        HnswPrecision::from_u8(self.precision)
    }

    pub fn get_m(&self) -> usize {
        self.m as usize
    }

    pub fn get_m0(&self) -> usize {
        self.m0 as usize
    }

    /// Max neighbors for `layer` (layer 0 gets m0, upper layers m).
    pub fn cap_for_layer(&self, layer: usize) -> usize {
        if layer == 0 {
            self.m0 as usize
        } else {
            self.m as usize
        }
    }

    pub fn get_ef_construction(&self) -> usize {
        self.ef_construction as usize
    }

    pub fn get_ml(&self) -> f32 {
        self.ml
    }

    pub fn get_max_level(&self) -> u8 {
        self.max_level
    }

    pub fn get_entry_point(&self) -> Option<ItemPointer> {
        if self.entry_point.is_valid() {
            Some(self.entry_point)
        } else {
            None
        }
    }

    pub fn get_entry_level(&self) -> i16 {
        self.entry_level
    }

    pub fn get_node_count(&self) -> u64 {
        self.node_count
    }

    pub fn get_deleted_count(&self) -> u64 {
        self.deleted_count
    }

    pub fn get_insert_page(&self) -> Option<pg_sys::BlockNumber> {
        if self.insert_page != pg_sys::InvalidBlockNumber {
            Some(self.insert_page)
        } else {
            None
        }
    }

    pub fn get_calibration_pointer(&self) -> Option<ItemPointer> {
        if self.calibration.is_valid() {
            Some(self.calibration)
        } else {
            None
        }
    }

    /// Create a new meta page and write it to block 0 of a fresh relation.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn create(
        index: &PgRelation,
        num_dimensions: u32,
        distance_type: DistanceType,
        precision: HnswPrecision,
        m: u16,
        ef_construction: u32,
        max_level: u8,
        calibration: ItemPointer,
    ) -> HnswMetaPage {
        let version = Version::parse(env!("CARGO_PKG_VERSION")).unwrap();
        let m0 = (m as usize * 2).min(u16::MAX as usize) as u16;
        let ml = 1.0 / (m as f64).ln() as f32;

        let meta = HnswMetaPage {
            magic_number: HNSW_MAGIC_NUMBER,
            version: HNSW_VERSION,
            extension_version_when_built: version.to_string(),
            distance_type: distance_type as u16,
            num_dimensions,
            precision: precision as u8,
            m,
            m0,
            ef_construction,
            ml,
            max_level,
            entry_point: ItemPointer::new_invalid(),
            entry_level: -1,
            node_count: 0,
            deleted_count: 0,
            insert_page: pg_sys::InvalidBlockNumber,
            free_pages_head: pg_sys::InvalidBlockNumber,
            calibration,
        };

        meta.store(index, true);
        meta
    }

    /// Write the meta page.  `first_time` writes a fresh page at block 0;
    /// otherwise the existing page is rewritten in place (only for the build
    /// path — runtime mutations must use [`Self::update`]).
    pub unsafe fn store(&self, index: &PgRelation, first_time: bool) {
        assert_eq!(self.magic_number, HNSW_MAGIC_NUMBER);
        assert_eq!(self.version, HNSW_VERSION);

        let bytes = self.serialize_to_vec();
        assert!(
            bytes.len() < pg_sys::BLCKSZ as usize / 2,
            "hnswsq meta page exceeded half a page ({} bytes)",
            bytes.len()
        );
        if first_time {
            let mut page = WritablePage::new(index, PageType::HnswMeta);
            let block = page.get_block_number();
            let off = page.add_item(&bytes);
            page.commit();
            assert_eq!(block, META_BLOCK_NUMBER, "meta page must be block 0");
            assert_eq!(off, META_OFFSET);
        } else {
            let mut page = WritablePage::modify(index, META_BLOCK_NUMBER);
            page.reinit(PageType::HnswMeta);
            page.add_item(&bytes);
            page.commit();
        }
    }

    /// Read the meta page (share content lock; atomic single-item read).
    pub fn fetch(index: &PgRelation) -> HnswMetaPage {
        unsafe {
            let page = ReadablePage::read(index, META_BLOCK_NUMBER);
            assert!(
                page.get_type() == PageType::HnswMeta,
                "hnswsq: block 0 is not a meta page"
            );
            let item = page.get_item_unchecked(META_OFFSET);
            let result = rkyv::from_bytes::<HnswMetaPage>(item.get_data_slice())
                .unwrap_or_else(|e| panic!("hnswsq: meta fetch parse failed: {:?}", e));
            assert_eq!(result.magic_number, HNSW_MAGIC_NUMBER);
            assert_eq!(result.version, HNSW_VERSION);
            result
        }
    }

    /// Read-modify-write the meta page under its exclusive content lock.  The
    /// closure receives the current meta (parsed under the lock) and may do
    /// arbitrary work; the page is rewritten in place (WAL-logged via
    /// GenericXLog) after the closure returns.
    ///
    /// Lock rule: the closure must not acquire locks on pages that other
    /// backends could reach concurrently, EXCEPT free pages popped/pushed here
    /// (reachable only through `free_pages_head`, which this lock guards).
    pub unsafe fn update<R, F: FnOnce(&mut HnswMetaPage) -> R>(index: &PgRelation, f: F) -> R {
        let buffer = LockedBufferExclusive::read(index, META_BLOCK_NUMBER);
        let page = pg_sys::BufferGetPage(**&buffer);
        let item_id = PageGetItemId(page, META_OFFSET);
        let item = PageGetItem(page, item_id);
        let len = (*item_id).lp_len() as usize;
        let mut meta =
            rkyv::from_bytes::<HnswMetaPage>(std::slice::from_raw_parts(item as *const u8, len))
                .unwrap_or_else(|e| panic!("hnswsq: meta update parse failed: {:?}", e));
        assert_eq!(meta.magic_number, HNSW_MAGIC_NUMBER);
        assert_eq!(meta.version, HNSW_VERSION);

        let result = f(&mut meta);
        let bytes = meta.serialize_to_vec();
        assert!(
            bytes.len() < pg_sys::BLCKSZ as usize / 2,
            "hnswsq meta page exceeded half a page ({} bytes)",
            bytes.len()
        );
        page::write_single_item_page_locked(index, &buffer, PageType::HnswMeta, &bytes);
        result
    }

    /// Promote the entry point to `(ptr, level)` iff the current entry level is
    /// lower (or the index is empty).  Returns true when this call installed
    /// the new entry point.  Concurrent promoters serialize on the meta lock;
    /// the highest level wins.
    pub unsafe fn promote_entry_point(
        index: &PgRelation,
        ptr: ItemPointer,
        level: u8,
    ) -> bool {
        Self::update(index, |meta| {
            if (meta.entry_level as i32) < level as i32 {
                meta.entry_point = ptr;
                meta.entry_level = level as i16;
                true
            } else {
                false
            }
        })
    }

    /// Claim the entry point of an empty index.  Returns true when this call
    /// installed the entry point (loser of a first-insert race returns false).
    pub unsafe fn claim_entry_point_if_empty(
        index: &PgRelation,
        ptr: ItemPointer,
        level: u8,
    ) -> bool {
        Self::update(index, |meta| {
            if !meta.entry_point.is_valid() {
                meta.entry_point = ptr;
                meta.entry_level = level as i16;
                true
            } else {
                false
            }
        })
    }

    /// Publish a new insert-page hint (standalone meta RMW).
    pub unsafe fn set_insert_page(index: &PgRelation, block: pg_sys::BlockNumber) {
        Self::update(index, |meta| {
            meta.insert_page = block;
        });
    }

    /// Pop one page from the free list for reuse, returning its block number
    /// (None when the list is empty).  The link item is read under the meta
    /// exclusive lock: free pages are reachable only through the head pointer
    /// this lock guards, so no other backend can touch them concurrently.
    pub unsafe fn pop_free_page(index: &PgRelation) -> Option<pg_sys::BlockNumber> {
        Self::update(index, |meta| {
            let head = meta.free_pages_head;
            if head == pg_sys::InvalidBlockNumber {
                return None;
            }
            let next = {
                let page = ReadablePage::read(index, head);
                assert!(page.get_type() == PageType::HnswFreePages);
                let item = page.get_item_unchecked(1);
                let link = rkyv::from_bytes::<HnswFreePageLink>(item.get_data_slice())
                    .unwrap_or_else(|e| panic!("hnswsq: free-page link parse failed: {:?}", e));
                link.next
            };
            meta.free_pages_head = next;
            Some(head)
        })
    }

    /// Push fully-dead pages onto the free list (VACUUM only), returning the
    /// blocks ACTUALLY freed.  Each page is re-verified under its exclusive
    /// content lock (inside the meta RMW): a concurrent insert holding a stale
    /// `insert_page` hint may have appended a live node since the vacuum walk
    /// classified the page, and such a page is skipped.  Threading and
    /// verification both happen under the meta exclusive lock — free pages are
    /// reachable only through the head pointer this lock guards, so no other
    /// backend can be modifying them concurrently.
    pub unsafe fn push_free_pages(
        index: &PgRelation,
        blocks: &[pg_sys::BlockNumber],
    ) -> Vec<pg_sys::BlockNumber> {
        if blocks.is_empty() {
            return Vec::new();
        }
        Self::update(index, |meta| {
            let mut freed: Vec<pg_sys::BlockNumber> = Vec::new();
            let mut next = meta.free_pages_head;
            for &b in blocks {
                let mut page = WritablePage::modify(index, b);
                if page.get_type() != PageType::HnswNode {
                    continue; // already recycled or not a node page
                }
                let all_dead = unsafe {
                    let max_off = PageGetMaxOffsetNumber(*page);
                    let mut any_item = false;
                    let mut all = true;
                    for off in 1..=max_off {
                        let item_id = PageGetItemId(*page, off as pgrx::pg_sys::OffsetNumber);
                        if (*item_id).lp_flags() != 1 || (*item_id).lp_len() == 0 {
                            continue;
                        }
                        any_item = true;
                        let item = PageGetItem(*page, item_id) as *const u8;
                        let len = (*item_id).lp_len() as usize;
                        let node = rkyv::archived_root::<crate::access_method::hnswsq::node::HnswNode>(
                            std::slice::from_raw_parts(item, len),
                        );
                        if !node.is_deleted() {
                            all = false;
                            break;
                        }
                    }
                    any_item && all
                };
                if !all_dead {
                    continue; // a live node appeared; keep the page in service
                }
                let link = HnswFreePageLink { next };
                let bytes = link.serialize_to_vec();
                page.reinit(PageType::HnswFreePages);
                page.add_item(&bytes);
                page.commit();
                next = b;
                freed.push(b);
            }
            meta.free_pages_head = next;
            freed
        })
    }

    /// Publish vacuum's counters and fix the entry point — race-safe: the fix
    /// is applied only when the entry point is still the one vacuum inspected
    /// (`expected_entry`); a concurrent insert promotion (which changes the
    /// entry under this same lock) wins and the fix is skipped.  A live entry
    /// is only ever promoted to a strictly higher level, never demoted.
    pub unsafe fn set_counts_and_entry(
        index: &PgRelation,
        node_count: u64,
        deleted_count: u64,
        best_live: Option<(ItemPointer, u8)>,
        expected_entry: Option<ItemPointer>,
        entry_dead: bool,
    ) {
        Self::update(index, |meta| {
            meta.node_count = node_count;
            meta.deleted_count = deleted_count;
            if meta.get_entry_point() != expected_entry {
                return; // entry changed under us; the newer value wins
            }
            if entry_dead {
                match best_live {
                    Some((ptr, level)) => {
                        meta.entry_point = ptr;
                        meta.entry_level = level as i16;
                    }
                    None => {
                        meta.entry_point = ItemPointer::new_invalid();
                        meta.entry_level = -1;
                    }
                }
            } else if let Some((ptr, level)) = best_live {
                if (level as i16) > meta.entry_level {
                    meta.entry_point = ptr;
                    meta.entry_level = level as i16;
                }
            }
        });
    }

    /// Publish the result of a bulk build flush: entry point, live count, and
    /// the append hint (last written node page).  Field-only mutation, safe
    /// inside [`Self::update`].
    pub fn set_build_result(
        &mut self,
        entry_point: ItemPointer,
        entry_level: u8,
        node_count: u64,
        insert_page: pg_sys::BlockNumber,
    ) {
        self.entry_point = entry_point;
        self.entry_level = entry_level as i16;
        self.node_count = node_count;
        self.insert_page = insert_page;
    }

    /// Record the SQ8 calibration chain pointer (written after the meta page
    /// so the meta keeps block 0).  Field-only mutation for [`Self::update`].
    pub fn set_calibration_pointer(&mut self, ptr: ItemPointer) {
        self.calibration = ptr;
    }
}
