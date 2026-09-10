//! AgentVec vacuum.
//!
//! `ambulkdelete` walks every published chain, asks the vacuum callback
//! whether each entry's heap row is dead, and tombstones the dead ones in
//! place.  Three properties matter:
//!
//! * **The callback is never called while holding one of our locks.**  A
//!   chain's live entries are collected first (under the pages' share locks,
//!   which are released before the callback runs), so this AM never holds an
//!   index lock across a heap buffer access — the INSERT path takes heap then
//!   index, and inverting that order would be a deadlock risk.
//! * **Tombstoning and the segment's dead count are updated together**, under
//!   the segment header's content lock.  A scan only looks at the per-entry
//!   state byte, but the header's `dead_entries` is what the directory and
//!   `agentvec_index_info()` report, so the two must not drift apart.
//! * **Tombstoning is a single byte flip** under the page's exclusive content
//!   lock, WAL-logged, so a concurrent scan sees an entry as live or dead but
//!   never as a torn record.  Reclaiming the space those bytes occupy is a
//!   compaction concern (plan phase 10); this AM retires segments rather than
//!   truncating pages, so `pages_deleted` stays 0.

use pgrx::pg_sys::{BlockNumber, OffsetNumber};
use pgrx::*;

use crate::access_method::agentvec::directory::AgentVecSegmentHeader;
use crate::access_method::agentvec::flat;
use crate::access_method::agentvec::meta_page::AgentVecMetaPage;
use crate::util::ItemPointer;

/// Delete dead entries from the index.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambulkdelete(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    callback: pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut std::os::raw::c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let results = if stats.is_null() {
        PgBox::<pg_sys::IndexBulkDeleteResult>::alloc0().into_pg()
    } else {
        stats
    };

    let index_rel = PgRelation::from_pg((*info).index);
    let meta = AgentVecMetaPage::fetch(&index_rel);
    let directory = meta.load_directory(&index_rel);

    let mut total_live: u64 = 0;
    let mut total_dead: u64 = 0;

    for segment in directory.searchable() {
        let header = AgentVecSegmentHeader::load(&index_rel, segment.header);
        // Per chain, which bounds the TID buffer by the chain size
        // (hot_segment_max_rows) rather than by the index size.
        for chain_start in header.chain_starts() {
            let mut entries: Vec<(BlockNumber, OffsetNumber, ItemPointer)> = Vec::new();
            flat::for_each_entry(&index_rel, chain_start, |block, offset, bytes| {
                if flat::decode_state(bytes) != flat::STATE_LIVE {
                    return;
                }
                entries.push((block, offset, flat::decode_tid(bytes)));
            });

            // No AgentVec lock is held here: the callback below may touch heap
            // buffers.
            let mut to_tombstone: Vec<(BlockNumber, OffsetNumber)> = Vec::new();
            for (block, offset, tid) in entries {
                let is_dead = match callback {
                    Some(cb) => {
                        let mut tid_data = pg_sys::ItemPointerData::default();
                        tid.to_item_pointer_data(&mut tid_data);
                        cb(&mut tid_data, callback_state)
                    }
                    None => false,
                };
                if is_dead {
                    to_tombstone.push((block, offset));
                } else {
                    total_live += 1;
                }
            }

            if to_tombstone.is_empty() {
                continue;
            }
            let num_dead = to_tombstone.len() as u64;
            // Publish the tombstones and the segment's dead count in one
            // atomic header rewrite.  Taking the header lock and then the
            // page locks keeps this path's order (header -> page) the same as
            // the insert path's.
            AgentVecSegmentHeader::update(&index_rel, segment.header.block_number, |header| {
                for (block, offset) in to_tombstone {
                    flat::mark_dead(&index_rel, block, offset);
                }
                header.dead_entries += num_dead;
            });
            total_dead += num_dead;
        }
    }

    (*results).tuples_removed = total_dead as f64;
    (*results).num_index_tuples = total_live as f64;

    results
}

/// Cleanup after vacuum: nothing to reclaim yet, so just report the stats
/// accumulated by `ambulkdelete`.
#[pg_guard]
pub unsafe extern "C-unwind" fn amvacuumcleanup(
    _info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    stats
}
