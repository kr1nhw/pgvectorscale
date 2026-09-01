//! IVF index insert implementation.
//!
//! Insert is O(1) amortized: the entry is appended to the list's unpublished
//! active buffer (row-major items, one `PageAddItem` per entry), and the
//! buffer is sealed into an immutable SoA segment once it reaches
//! `ivf.seal_threshold`.  The list header's exclusive content lock is held for
//! the whole append, serializing appenders to the same list and excluding
//! concurrent compaction.

use pgrx::*;

use crate::access_method::ivf::centroid_page::IvfCentroidPage;
use crate::access_method::ivf::entry::{
    append_active_entry, read_active_entries, seal_blocks_needed, seal_entries,
    seal_entries_at, IvfEntry,
};
use crate::access_method::ivf::list_directory::IvfListDirectory;
use crate::access_method::ivf::meta_page::IvfMetaPage;
use crate::access_method::ivf::options::IVF_SEAL_THRESHOLD;
use crate::access_method::ivf::segment::{IvfFreeRange, IvfListHeader, IvfSegmentList};
use crate::access_method::ivf::simd::find_nearest_centroids;
use crate::access_method::pg_vector::PgVectorInternal;
use crate::access_method::quantization::rabitq::{padded_dim, RabitqQuantizer};
use crate::util::ItemPointer;

/// Insert a tuple into the IVF index.
///
/// Finds the nearest centroid, quantizes the vector relative to it, and
/// appends the entry to that list's active buffer (sealing on threshold).
#[pg_guard]
pub unsafe extern "C-unwind" fn aminsert(
    index: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    _heap: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    // Skip null vectors.
    if *isnull {
        return false;
    }

    let index_rel = unsafe { PgRelation::from_pg(index) };
    let meta = IvfMetaPage::fetch(&index_rel);
    let mut centroid_page = match meta.get_centroids_pointer() {
        Some(p) => IvfCentroidPage::load(&index_rel, p),
        None => IvfCentroidPage::new(Vec::new()),
    };
    let list_directory = IvfListDirectory::load(&index_rel);

    // Extract the vector.
    let datum = *values;
    let detoasted = pg_sys::pg_detoast_datum_copy(datum.cast_mut_ptr());
    let pg_vec = detoasted.cast::<PgVectorInternal>();
    let vector = (*pg_vec).to_slice().to_vec();
    pg_sys::pfree(detoasted.cast());

    // If the index has no centroids yet (built empty), seed the first centroid
    // from this vector so subsequent inserts and scans have a list to use.
    if centroid_page.centroids.is_empty() {
        centroid_page = IvfCentroidPage::new(vec![vector.clone()]);
        let existing = meta.get_centroids_pointer().map(|p| p.block_number);
        unsafe {
            centroid_page.store(&index_rel, existing);
        }
    }

    // Find the nearest centroid.
    let nearest = find_nearest_centroids(
        &vector,
        &centroid_page.centroids,
        meta.get_distance_type(),
        1,
    );
    if nearest.is_empty() {
        return false;
    }
    let list_id = nearest[0] as u16;

    // Quantize relative to the list centroid.
    let quantizer = RabitqQuantizer::new(
        meta.get_bq_num_bits_per_dimension(),
        meta.get_rotation_seed(),
        meta.get_num_dimensions() as usize,
    );
    let centroid = &centroid_page.centroids[list_id as usize];
    let code = quantizer.quantize_residual(centroid, &vector);
    let entry = IvfEntry::new(ItemPointer::with_item_pointer_data(*heap_tid), code);

    // Append the entry to the list's unpublished active buffer, sealing it
    // into an immutable segment when it reaches ivf.seal_threshold.  The
    // header's exclusive content lock is held for the whole append,
    // serializing appenders to this list and excluding concurrent compaction
    // (so no published segment is ever lost or torn).
    let Some(header_ptr) = list_directory.get_list(list_id).map(|m| m.header) else {
        return false;
    };
    let header_block = header_ptr.block_number;
    let seal_threshold = IVF_SEAL_THRESHOLD.get().max(1) as u64;
    let num_bits = meta.get_bq_num_bits_per_dimension();
    let dim_padded = padded_dim(meta.get_num_dimensions() as usize) as u32;

    // Peek the active buffer; if this insert will trigger a seal, reserve
    // reclaimed blocks from the free list BEFORE taking the header lock (the
    // ExclusiveLock must never nest inside the header content lock).  None
    // falls back to relation extension inside the closure.
    let active_count = {
        let peek = IvfListHeader::load(&index_rel, header_ptr);
        peek.active.map(|a| a.num_entries).unwrap_or(0)
    };
    let mut reserved: Option<(pg_sys::BlockNumber, u32)> = None;
    if active_count >= seal_threshold {
        let need = seal_blocks_needed(active_count as usize, num_bits, dim_padded);
        if let Some(start) = unsafe { IvfMetaPage::allocate_range(&index_rel, need) } {
            reserved = Some((start, need));
        }
    }

    // What the closure reports back for the post-lock bookkeeping (which needs
    // the ExclusiveLock again, so it cannot run inside the header lock).
    struct Outcome {
        retired: Vec<IvfFreeRange>,
        reserved_used: bool,
        unused_tail: Option<(pg_sys::BlockNumber, u32)>,
    }

    let outcome = unsafe {
        IvfListHeader::update(&index_rel, header_block, |header| {
            let mut retired: Vec<IvfFreeRange> = Vec::new();
            let mut reserved_used = false;
            let mut unused_tail = None;

            // 1. If the active buffer reached the seal threshold, seal it into
            //    a published segment now (before appending the new entry).
            if let Some(active) = header.active.as_ref() {
                if active.num_entries >= seal_threshold {
                    let entries = read_active_entries(&index_rel, active);
                    let count = entries.len() as u64;
                    let sealed = match reserved {
                        // Use the reserved reclaimed blocks, but only if the
                        // buffer size still matches the peek (a concurrent
                        // append since the peek would make the reservation too
                        // small — fall back to extension instead).
                        Some((start, need)) if count == active_count as u64 => {
                            let used = seal_entries_at(&index_rel, entries, start);
                            reserved_used = true;
                            if used < need {
                                unused_tail = Some((start + used as pg_sys::BlockNumber, need - used));
                            }
                            crate::access_method::ivf::segment::IvfSegment::new(
                                start,
                                used,
                                count,
                            )
                        }
                        _ => seal_entries(&index_rel, entries),
                    };
                    if !sealed.is_empty() {
                        let mut segments =
                            IvfSegmentList::load(&index_rel, header.segment_list).segments;
                        segments.push(sealed);
                        let new_sl = IvfSegmentList::new(segments);
                        let (new_ptr, new_blocks) = new_sl.store(&index_rel);
                        // The old segment-list item becomes garbage; retire it
                        // (the old segments stay referenced by the new item).
                        // segment_list_blocks == 0 marks the shared empty item
                        // used by never-sealed lists — never retire it.
                        if header.segment_list_blocks > 0 {
                            retired.push(IvfFreeRange {
                                start_block: header.segment_list.block_number,
                                num_blocks: header.segment_list_blocks,
                            });
                        }
                        header.segment_list = new_ptr;
                        header.segment_list_blocks = new_blocks;
                        header.version += 1;
                        // The sealed blocks must be on disk before the header
                        // swap becomes visible to smgrreadv scans.  Flush
                        // inside the closure — i.e. BEFORE the header page
                        // itself is rewritten on update() exit.
                        pg_sys::FlushRelationBuffers(index_rel.as_ptr());
                    }
                    header.active = None;
                }
            }

            // 2. Append the entry to the (possibly fresh) active buffer.
            header.active = Some(append_active_entry(
                &index_rel,
                header.active.take(),
                entry,
            ));

            Outcome {
                retired,
                reserved_used,
                unused_tail,
            }
        })
    };

    // Post-lock bookkeeping under the ExclusiveLock (never while holding the
    // header content lock): return any unused reservation and reclaim retired
    // ranges so they can be reused by later seals.
    unsafe {
        if let Some((start, need)) = reserved {
            if !outcome.reserved_used {
                IvfMetaPage::push_back_range(&index_rel, start, need);
            } else if let Some((tail_start, tail_blocks)) = outcome.unused_tail {
                IvfMetaPage::push_back_range(&index_rel, tail_start, tail_blocks);
            }
        }
        IvfMetaPage::reclaim_ranges(&index_rel, outcome.retired);
    }

    false
}
