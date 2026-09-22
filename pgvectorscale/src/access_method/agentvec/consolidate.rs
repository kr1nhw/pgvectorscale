//! CONVERT: sealed HOT (embedded hnswsq region) → immutable IVF-RaBitQ
//! segment.
//!
//! The design's whole-segment conversion, implemented as: collect the live
//! rows of the oldest sealed HOT segment (decoded inline — no heap fetch),
//! train the segment's centroids, RaBitQ-encode against them, write a NEW
//! invisible IVF payload, then publish with ONE directory republication that
//! adds the new segment and retires the source.  A crash before the swap
//! leaves orphaned pages and the source still claimable; there is no
//! per-tuple migration machinery to keep idempotent because there is no
//! per-tuple migration.

use pgrx::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::access_method::agentvec::directory::{
    AgentVecDirectory, AgentVecSegmentHeader, AgentVecSegmentMeta, SegmentAlgorithm, SegmentLevel,
    SegmentOwnership, SegmentState,
};
use crate::access_method::agentvec::meta_page::AgentVecMetaPage;
use crate::access_method::agentvec::options::TSVAgentVecOptions;
use crate::access_method::agentvec::router;
use crate::access_method::distance::{preprocess_cosine, DistanceType};
use crate::access_method::ivf::centroid::{kmeans_plus_plus_init, lloyds_algorithm};
use crate::access_method::ivf::centroid_page::IvfCentroidPage;
use crate::access_method::ivf::entry::{seal_entries, IvfEntry};
use crate::access_method::ivf::list_directory::IvfListDirectory;
use crate::access_method::ivf::meta_page::{IvfMetaPage, IVF_STANDALONE_BASE};
use crate::access_method::ivf::segment::{IvfListHeader, IvfSegmentList};
use crate::access_method::ivf::simd::find_nearest_centroids;
use crate::access_method::quantization::rabitq::RabitqQuantizer;
use crate::access_method::storage::StorageType;
use crate::util::buffer::LockRelationForExtension;
use crate::util::ItemPointer;

/// Payload format version of an immutable IVF-RaBitQ segment.
pub const SEGMENT_FORMAT_IVF_V1: u32 = 3;

/// Convert the oldest sealed HOT segment into an immutable IVF-RaBitQ
/// segment.  Returns the number of live rows converted (0 when nothing was
/// sealed or the source was empty).  Manual trigger for phase 3; the phase-4
/// maintenance worker calls the same internal function.
#[pg_extern]
fn agentvec_consolidate(index: PgRelation) -> i64 {
    unsafe { consolidate_inner(&index) }
}

pub(super) unsafe fn consolidate_inner(index: &PgRelation) -> i64 {
    let meta = AgentVecMetaPage::fetch(index);
    let options = TSVAgentVecOptions::from_relation(index);

    // Claim the oldest sealed HOT segment.  `Retiring` is also accepted so a
    // crashed claim self-heals on the next call (the segment stays searchable
    // either way).
    let claimed = AgentVecMetaPage::update(index, |m| {
        let mut directory = m.load_directory(index);
        let Some(seg) = directory.segments.iter_mut().find(|s| {
            matches!(
                s.state(),
                SegmentState::QueuedForMigration | SegmentState::Retiring
            ) && s.algorithm() == SegmentAlgorithm::Hnsw
        }) else {
            return None;
        };
        let id = seg.segment_id;
        let code_root = seg.code_root;
        if seg.state() == SegmentState::QueuedForMigration {
            seg.state = SegmentState::Retiring as u8;
        }
        let (ptr, blocks) = directory.store(index);
        m.set_directory(ptr, blocks);
        Some((id, code_root))
    });
    let Some((source_id, code_root)) = claimed else {
        return 0;
    };

    // Collect the live rows of the embedded region (decoded inline).  For
    // calibrated layouts the inline bytes clamp: the provisional incremental
    // calibration collapses every out-of-range component, so the decoded
    // vectors are unusable for re-encoding — fetch exact vectors from the
    // heap for those instead (rows that vanished concurrently are dropped).
    let base = code_root.block_number;
    let support = crate::access_method::hnswsq::utils::init_support(index.as_ptr(), base);
    let distance_type = meta.get_distance_type();
    let dim = meta.get_num_dimensions() as usize;
    let inline_rows =
        crate::access_method::hnswsq::utils::collect_elements(index.as_ptr(), base, &support);
    let rows: Vec<(pg_sys::ItemPointerData, Vec<f32>)> = if support.precision.needs_calibration() {
        inline_rows
            .into_iter()
            .filter_map(|(tid, _)| {
                super::insert::fetch_heap_vector(index, tid, dim, distance_type).map(|v| (tid, v))
            })
            .collect()
    } else {
        inline_rows
    };

    if rows.is_empty() {
        // Nothing live: just retire the source.
        AgentVecMetaPage::update(index, |m| {
            let mut directory = m.load_directory(index);
            if let Some(seg) = directory.get_mut(source_id) {
                seg.state = SegmentState::Retired as u8;
            }
            let (ptr, blocks) = directory.store(index);
            m.set_directory(ptr, blocks);
        });
        return 0;
    }

    // Train centroids on a bounded sample, assign, and RaBitQ-encode.
    let lists = options.get_ivf_lists() as usize;
    let num_bits = options.get_rabitq_bits();
    let mut rng = SmallRng::from_entropy();
    let rotation_seed: u64 = rng.gen();

    let sample = reservoir_sample(&rows, (lists * 256).min(30_000).max(16));
    let mut centroids = kmeans_plus_plus_init(&sample, lists, distance_type);
    centroids = lloyds_algorithm(&sample, &mut centroids, 25, distance_type);

    let quantizer = RabitqQuantizer::new(num_bits, rotation_seed, dim);
    let mut per_list: Vec<Vec<IvfEntry>> = (0..lists).map(|_| Vec::new()).collect();
    for (tid, vec) in &rows {
        let nearest = find_nearest_centroids(vec, &centroids, distance_type, 1);
        let list_id = nearest[0];
        let code = quantizer.quantize_residual(&centroids[list_id], vec);
        per_list[list_id].push(IvfEntry::new(ItemPointer::with_item_pointer_data(*tid), code));
    }

    let (ivf_base, list_directory_pointer) = build_ivf_segment(
        index,
        dim,
        distance_type,
        lists as u16,
        num_bits,
        rotation_seed,
        &centroids,
        per_list,
    );

    // Publish the segment's centroids to the router Vamana graph *before*
    // the directory swap: after the swap the segment is always routable,
    // while a crash before it leaves orphaned centroid nodes that the
    // router drops by id.  The id is pre-allocated under the meta lock so
    // the router nodes and the published segment agree even when two
    // consolidations interleave.
    let segment_id = AgentVecMetaPage::update(index, |m| m.take_next_segment_id());
    let router_base = router::ensure_router_region(index, dim as u32);
    router::add_segment_centroids(index, router_base, segment_id, &centroids);

    // Publish: retire the source and add the new WARM segment in ONE
    // directory republication.
    let num_rows = rows.len() as u64;
    AgentVecMetaPage::update(index, |m| {
        let mut directory = m.load_directory(index);
        let epoch = m.bump_epoch();
        if let Some(seg) = directory.get_mut(source_id) {
            seg.state = SegmentState::Retired as u8;
            seg.epoch = epoch;
        }

        let mut header =
            AgentVecSegmentHeader::new(SegmentLevel::Warm, SegmentAlgorithm::IvfRaBitQ);
        header.num_entries = num_rows;
        let header_pointer = header.store_new(index);

        let mut seg = AgentVecSegmentMeta::new(
            segment_id,
            0,
            SegmentLevel::Warm,
            SegmentState::Published,
            SegmentAlgorithm::IvfRaBitQ,
            SegmentOwnership::Owned,
            epoch,
            header_pointer,
            distance_type as u16,
            dim as u32,
            SEGMENT_FORMAT_IVF_V1,
        );
        seg.code_root = ItemPointer::new(ivf_base, 1);
        seg.posting_root = list_directory_pointer;
        seg.vector_count = num_rows;
        seg.live_count = num_rows;
        directory.segments.push(seg);

        let (ptr, blocks) = directory.store(index);
        m.set_directory(ptr, blocks);
    });

    num_rows as i64
}

/// Write the immutable IVF payload: an `IvfMetaPage` at the relation end
/// (the segment base), the centroid page, per-list sealed SoA runs with their
/// headers, and the list directory — all with the exact structures and page
/// formats the `ivf` AM uses.
#[allow(clippy::too_many_arguments)]
unsafe fn build_ivf_segment(
    index: &PgRelation,
    dim: usize,
    distance_type: DistanceType,
    lists: u16,
    num_bits: u8,
    rotation_seed: u64,
    centroids: &[Vec<f32>],
    per_list: Vec<Vec<IvfEntry>>,
) -> (pg_sys::BlockNumber, ItemPointer) {
    let base = {
        let _ext = LockRelationForExtension::new(index);
        let b = pg_sys::RelationGetNumberOfBlocksInFork(
            index.as_ptr(),
            pg_sys::ForkNumber::MAIN_FORKNUM,
        );
        IvfMetaPage::create(
            index,
            b,
            dim as u32,
            distance_type,
            lists,
            StorageType::RabitqCompression,
            num_bits,
            rotation_seed,
        );
        b
    };

    let centroid_page = IvfCentroidPage::new(centroids.to_vec());
    let centroid_pointer = centroid_page.store(index, None);

    let mut list_directory = IvfListDirectory::new(lists);
    for (list_id, entries) in per_list.into_iter().enumerate() {
        if entries.is_empty() {
            continue;
        }
        let count = entries.len() as u64;
        let segment = seal_entries(index, entries);
        let segment_list = IvfSegmentList::new(vec![segment]);
        let (segment_list_pointer, segment_list_blocks) = segment_list.store(index);
        let header = IvfListHeader::new(segment_list_pointer, segment_list_blocks);
        let header_pointer = header.store_new(index);
        let list_meta = list_directory
            .get_list_mut(list_id as u16)
            .expect("list exists");
        list_meta.header = header_pointer;
        list_meta.num_tuples = count;
    }
    let (list_directory_pointer, _blocks) = list_directory.store_new(index);

    let mut meta = IvfMetaPage::fetch(index, base);
    meta.set_centroids_pointer(centroid_pointer);
    meta.set_list_directory_pointer(list_directory_pointer);
    meta.store(index, base, false);

    // Bulk smgr scans (SoA entry reads) bypass shared buffers and read raw
    // disk blocks, so the freshly written pages must be on disk first.
    unsafe {
        pg_sys::FlushRelationBuffers(index.as_ptr());
    }

    (base, list_directory_pointer)
}

/// Reservoir sample (bounded memory; keeps every row when `rows` is smaller).
fn reservoir_sample(rows: &[(pg_sys::ItemPointerData, Vec<f32>)], want: usize) -> Vec<Vec<f32>> {
    let mut sample: Vec<Vec<f32>> = Vec::with_capacity(want);
    let mut rng = SmallRng::from_entropy();
    for (i, (_tid, vec)) in rows.iter().enumerate() {
        if sample.len() < want {
            sample.push(vec.clone());
        } else {
            let j = rng.gen_range(0..=i);
            if j < want {
                sample[j] = vec.clone();
            }
        }
    }
    sample
}

