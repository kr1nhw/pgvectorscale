use std::{cell::RefCell, iter::once, marker::PhantomData};

use pgrx::{pg_sys::AttrNumber, PgBox, PgRelation};

use crate::{
    access_method::{
        build::QUANTIZED_VECTOR_CACHE_SIZE,
        distance::{DistanceFn, DistanceType},
        graph::{
            neighbor_store::GraphNeighborStore,
            neighbor_with_distance::{DistanceWithTieBreak, NeighborWithDistance},
            ListSearchNeighbor, ListSearchResult,
        },
        labels::{LabelSet, LabelSetView, LabeledVector},
        meta_page::MetaPage,
        pg_vector::PgVector,
        quantization::rabitq::{code_hamming, RabitqQuantizer, RabitqVector},
        stats::{
            GreedySearchStats, StatsDistanceComparison, StatsHeapNodeRead, StatsNodeModify,
            StatsNodeRead, StatsNodeWrite,
        },
        storage::{NodeDistanceMeasure, Storage},
        storage_common::get_index_vector_attribute,
    },
    util::{
        page::PageType, table_slot::TableSlot, tape::Tape, HeapPointer, IndexPointer, ItemPointer,
    },
};

use super::{
    node::{ArchivedRabitqNode, RabitqNode, RabitqNodeData},
    RabitqNodeDistanceMeasure, RabitqQuantizerMetadata, RabitqSearchDistanceMeasure,
};
use super::super::rabitq::code_cos_distance;

pub struct RabitqSpeedupStorage<'a> {
    pub index: &'a PgRelation,
    pub distance_fn: DistanceFn,
    pub(crate) distance_type: DistanceType,
    quantizer: RabitqQuantizer,
    heap_rel: &'a PgRelation,
    heap_attr: AttrNumber,
    qv_cache: Option<RefCell<RabitqVectorCache>>,
    has_labels: bool,
    num_neighbors: u32,
}

impl<'a> RabitqSpeedupStorage<'a> {
    fn load_quantizer<S: StatsNodeRead>(
        index_relation: &PgRelation,
        meta_page: &MetaPage,
        stats: &mut S,
    ) -> RabitqQuantizer {
        let qip = meta_page
            .get_quantizer_metadata_pointer()
            .unwrap_or_else(|| pgrx::error!("No RaBitQ metadata pointer found in meta page"));
        unsafe { RabitqQuantizerMetadata::load(index_relation, qip, stats) }
    }

    pub unsafe fn new_for_build<S: StatsNodeRead>(
        index: &'a PgRelation,
        heap_rel: &'a PgRelation,
        meta_page: &MetaPage,
        stats: &mut S,
    ) -> Self {
        let quantizer = Self::load_quantizer(index, meta_page, stats);
        Self {
            index,
            distance_fn: meta_page.get_distance_function(),
            distance_type: meta_page.get_distance_type(),
            quantizer,
            heap_rel,
            heap_attr: get_index_vector_attribute(index),
            qv_cache: Some(RefCell::new(RabitqVectorCache::new(
                QUANTIZED_VECTOR_CACHE_SIZE,
                Self::rabitq_vec_len(meta_page),
                meta_page.get_num_neighbors() as usize,
            ))),
            has_labels: meta_page.has_labels(),
            num_neighbors: meta_page.get_num_neighbors(),
        }
    }

    pub fn load_for_insert<S: StatsNodeRead>(
        heap_rel: &'a PgRelation,
        index_relation: &'a PgRelation,
        meta_page: &MetaPage,
        stats: &mut S,
    ) -> Self {
        let quantizer = Self::load_quantizer(index_relation, meta_page, stats);
        Self {
            index: index_relation,
            distance_fn: meta_page.get_distance_function(),
            distance_type: meta_page.get_distance_type(),
            quantizer,
            heap_rel,
            heap_attr: get_index_vector_attribute(index_relation),
            qv_cache: Some(RefCell::new(RabitqVectorCache::new(
                QUANTIZED_VECTOR_CACHE_SIZE,
                Self::rabitq_vec_len(meta_page),
                meta_page.get_num_neighbors() as usize,
            ))),
            has_labels: meta_page.has_labels(),
            num_neighbors: meta_page.get_num_neighbors(),
        }
    }

    pub fn load_for_search(
        index_relation: &'a PgRelation,
        heap_relation: &'a PgRelation,
        quantizer: &RabitqQuantizer,
        meta_page: &MetaPage,
    ) -> Self {
        Self {
            index: index_relation,
            distance_fn: meta_page.get_distance_function(),
            distance_type: meta_page.get_distance_type(),
            //OPT: get rid of clone
            quantizer: quantizer.clone(),
            heap_rel: heap_relation,
            heap_attr: get_index_vector_attribute(index_relation),
            qv_cache: None,
            has_labels: meta_page.has_labels(),
            num_neighbors: meta_page.get_num_neighbors(),
        }
    }

    fn rabitq_vec_len(meta_page: &MetaPage) -> usize {
        // padded power-of-two dimension in bytes (code only)
        meta_page.get_num_dimensions().max(1).next_power_of_two() as usize / 8
    }

    fn visit_lsn_internal(
        &self,
        lsr: &mut ListSearchResult<
            <RabitqSpeedupStorage<'a> as Storage>::QueryDistanceMeasure,
            <RabitqSpeedupStorage<'a> as Storage>::LSNPrivateData,
        >,
        lsn_index_pointer: IndexPointer,
        gns: &mut GraphNeighborStore,
        no_filter: bool,
    ) {
        match gns {
            GraphNeighborStore::Disk => {
                // SAFETY: `lsn_index_pointer` addresses a live RaBitQ node item
                // written by the build path; nodes are immutable once sealed, so
                // reinterpreting the item bytes as the archived node is sound, and
                // `self.has_labels` (from the meta page) matches the variant written.
                let rn_visiting = unsafe {
                    RabitqNode::read(
                        self.index,
                        lsn_index_pointer,
                        self.has_labels,
                        &mut lsr.stats,
                    )
                };
                let node_visiting = rn_visiting.get_archived_node();
                let neighbors = node_visiting.get_index_pointer_to_neighbors();

                for &neighbor_index_pointer in neighbors.iter() {
                    if !lsr.prepare_insert(neighbor_index_pointer) {
                        continue;
                    }

                    // SAFETY: `neighbor_index_pointer` came from the stored
                    // neighbor list of a sealed node, so it addresses a live node
                    // item; same immutability/variant argument as above.
                    let rn_neighbor = unsafe {
                        RabitqNode::read(
                            self.index,
                            neighbor_index_pointer,
                            self.has_labels,
                            &mut lsr.stats,
                        )
                    };

                    let node_neighbor = rn_neighbor.get_archived_node();

                    // Skip neighbors that have no matching labels with the query
                    if let Some(labels) = lsr.sdm.as_ref().expect("sdm is Some").query.labels() {
                        if !no_filter
                            && !labels
                                .overlaps(node_neighbor.get_labels().expect("Unlabeled neighbor?"))
                        {
                            continue;
                        }
                    }

                    let distance = lsr
                        .sdm
                        .as_ref()
                        .expect("sdm is Some")
                        .calculate_bq_distance(node_neighbor.get_rabitq_data(), gns, &mut lsr.stats);

                    let lsn = ListSearchNeighbor::new(
                        neighbor_index_pointer,
                        lsr.create_distance_with_tie_break(distance, neighbor_index_pointer),
                        PhantomData::<bool>,
                        node_neighbor.get_labels().map(Into::into),
                    );

                    lsr.insert_neighbor(lsn);
                }
            }
            GraphNeighborStore::Builder(b) => {
                let neighbors = b.get_neighbors_with_full_vector_distances(
                    lsn_index_pointer,
                    self,
                    &mut lsr.prune_stats,
                );
                for neighbor in neighbors.iter() {
                    let neighbor_index_pointer = neighbor.get_index_pointer_to_neighbor();
                    if !lsr.prepare_insert(neighbor_index_pointer) {
                        continue;
                    }

                    // Skip neighbors that have no matching labels with the query
                    if let Some(labels) = lsr.sdm.as_ref().expect("lsr.sdm is None").query.labels()
                    {
                        if !no_filter && !labels.overlaps(neighbor.get_labels().unwrap()) {
                            continue;
                        }
                    }

                    let mut cache = self.qv_cache.as_ref().unwrap().borrow_mut();
                    let data = cache.get(neighbor_index_pointer, self, &mut lsr.stats);
                    let distance = lsr
                        .sdm
                        .as_ref()
                        .expect("lsr.sdm is None")
                        .calculate_bq_distance(data, gns, &mut lsr.stats);

                    let lsn = ListSearchNeighbor::new(
                        neighbor_index_pointer,
                        lsr.create_distance_with_tie_break(distance, neighbor_index_pointer),
                        PhantomData::<bool>,
                        neighbor.get_labels().cloned(),
                    );

                    lsr.insert_neighbor(lsn);
                }
            }
        }
    }

    pub fn cache(&self) -> Option<&RefCell<RabitqVectorCache>> {
        self.qv_cache.as_ref()
    }

    pub fn quantizer_num_bits(&self) -> u8 {
        self.quantizer.num_bits
    }
}

impl Storage for RabitqSpeedupStorage<'_> {
    type QueryDistanceMeasure = RabitqSearchDistanceMeasure;
    type NodeDistanceMeasure<'a>
        = RabitqNodeDistanceMeasure<'a>
    where
        Self: 'a;
    type ArchivedType<'b>
        = ArchivedRabitqNode<'b>
    where
        Self: 'b;
    type LSNPrivateData = RabitqSpeedupStorageLsnPrivateData;

    fn page_type() -> PageType {
        PageType::RabitqNode
    }

    fn create_node<S: StatsNodeWrite>(
        &self,
        full_vector: &[f32],
        labels: Option<LabelSet>,
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        tape: &mut Tape,
        stats: &mut S,
    ) -> ItemPointer {
        let qv = self.quantizer.quantize(full_vector);

        let node = RabitqNode::with_meta(
            heap_pointer,
            meta_page,
            &qv.packed_code,
            qv.l1_of_rotated,
            qv.sum_of_x2,
            qv.cent_dot,
            labels,
        );

        let index_pointer: IndexPointer = node.write(tape, stats);
        index_pointer
    }

    fn finalize_node_at_end_of_build<S: StatsNodeRead + StatsNodeModify>(
        &mut self,
        index_pointer: IndexPointer,
        neighbors: &[NeighborWithDistance],
        stats: &mut S,
    ) {
        let mut cache = self.qv_cache.as_ref().unwrap().borrow_mut();
        /* It's important to preload cache with all the items since you can run into deadlocks
        if you try to fetch a quantized vector while holding the RabitqNode::modify lock */
        let iter = neighbors
            .iter()
            .map(|n| n.get_index_pointer_to_neighbor())
            .chain(once(index_pointer));
        cache.preload(iter, self, stats);

        // SAFETY: `index_pointer` addresses a live RaBitQ node item; `modify`
        // takes the page's exclusive lock so no concurrent reader/writer can
        // race the neighbor update, and `self.has_labels` matches the stored
        // variant (both derive from the meta page).
        let mut node = unsafe { RabitqNode::modify(self.index, index_pointer, self.has_labels, stats) };
        let mut archived = node.get_archived_node();
        archived.set_neighbors(neighbors, self.num_neighbors);
        node.commit();
    }

    unsafe fn get_node_distance_measure<'b, S: StatsNodeRead + StatsNodeWrite + StatsNodeModify>(
        &'b self,
        index_pointer: IndexPointer,
        stats: &mut S,
    ) -> RabitqNodeDistanceMeasure<'b> {
        RabitqNodeDistanceMeasure::with_index_pointer(self, index_pointer, stats)
    }

    fn get_query_distance_measure(&self, query: LabeledVector) -> RabitqSearchDistanceMeasure {
        RabitqSearchDistanceMeasure::new(&self.quantizer, query, self.distance_type)
    }

    fn get_full_distance_for_resort<S: StatsHeapNodeRead + StatsDistanceComparison>(
        &self,
        scan: &PgBox<pgrx::pg_sys::IndexScanDescData>,
        qdm: &Self::QueryDistanceMeasure,
        _index_pointer: IndexPointer,
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        stats: &mut S,
    ) -> Option<f32> {
        let slot_opt = unsafe {
            TableSlot::from_index_heap_pointer(self.heap_rel, heap_pointer, scan.xs_snapshot, stats)
        };

        let slot = slot_opt?;

        let datum = unsafe {
            slot.get_attribute(self.heap_attr)
                .expect("vector attribute should exist in the heap")
        };
        let vec = unsafe { PgVector::from_datum(datum, meta_page, false, true) };
        Some(self.get_distance_function()(
            vec.to_full_slice(),
            qdm.query.vec().to_full_slice(),
        ))
    }

    fn get_neighbors_with_distances_from_disk<S: StatsNodeRead + StatsDistanceComparison>(
        &self,
        neighbors_of: ItemPointer,
        stats: &mut S,
    ) -> Vec<NeighborWithDistance> {
        // SAFETY: `neighbors_of` addresses a sealed, immutable RaBitQ node
        // (obtained from the graph's stored neighbor lists); the variant flag
        // `self.has_labels` matches what was written.
        let rn = unsafe { RabitqNode::read(self.index, neighbors_of, self.has_labels, stats) };
        let archived = rn.get_archived_node();
        let data = archived.get_rabitq_data();

        rn.get_archived_node()
            .iter_neighbors()
            .map(|n| {
                // SAFETY: `n` is a neighbor index pointer stored in a sealed
                // node; same liveness/immutability/variant argument as above.
                let rn1 = unsafe { RabitqNode::read(self.index, n, self.has_labels, stats) };
                let arch = rn1.get_archived_node();
                let other = arch.get_rabitq_data();
                stats.record_quantized_distance_comparison();
                let dist = match self.quantizer_num_bits() {
                    4 | 8 => code_cos_distance(
                        &data.code,
                        &other.code,
                        self.quantizer_num_bits(),
                    ),
                    _ => {
                        // code-to-code cosine estimate via the arcsin identity
                        let d = data.code.len() as f32 * 8.0;
                        let m12 = d - 2.0 * code_hamming(data.code, other.code) as f32;
                        let cos = (std::f32::consts::FRAC_PI_2 * (m12 / d)).sin();
                        RabitqVector::distance_from_cos(
                            cos,
                            data.sum_of_x2,
                            other.sum_of_x2,
                            self.distance_type,
                        )
                    }
                };
                NeighborWithDistance::new(
                    n,
                    DistanceWithTieBreak::new(dist, neighbors_of, n),
                    arch.get_labels().map(Into::into),
                )
            })
            .collect()
    }

    fn create_lsn_for_start_node(
        &self,
        lsr: &mut ListSearchResult<Self::QueryDistanceMeasure, Self::LSNPrivateData>,
        index_pointer: ItemPointer,
        gns: &mut GraphNeighborStore,
    ) -> Option<ListSearchNeighbor<Self::LSNPrivateData>> {
        if !lsr.prepare_insert(index_pointer) {
            // Already processed this start node
            return None;
        }

        // SAFETY: `index_pointer` is the graph's entry/start node pointer
        // (written by the build path and live while the scan runs); the node is
        // sealed and immutable, and `self.has_labels` matches the stored variant.
        let rn = unsafe { RabitqNode::read(self.index, index_pointer, self.has_labels, &mut lsr.stats) };
        let node = rn.get_archived_node();
        let distance = lsr.sdm.as_ref().unwrap().calculate_bq_distance(
            node.get_rabitq_data(),
            gns,
            &mut lsr.stats,
        );

        Some(ListSearchNeighbor::new(
            index_pointer,
            lsr.create_distance_with_tie_break(distance, index_pointer),
            PhantomData::<bool>,
            node.get_labels().map(Into::into),
        ))
    }

    fn visit_lsn(
        &self,
        lsr: &mut ListSearchResult<Self::QueryDistanceMeasure, Self::LSNPrivateData>,
        lsn_idx: usize,
        gns: &mut GraphNeighborStore,
        no_filter: bool,
    ) {
        let lsn_index_pointer = lsr.get_lsn_by_idx(lsn_idx).index_pointer;
        self.visit_lsn_internal(lsr, lsn_index_pointer, gns, no_filter);
    }

    fn return_lsn(
        &self,
        lsn: &ListSearchNeighbor<Self::LSNPrivateData>,
        stats: &mut GreedySearchStats,
    ) -> HeapPointer {
        let lsn_index_pointer = lsn.index_pointer;
        // SAFETY: `lsn_index_pointer` comes from the LSN store, which only holds
        // pointers to live, sealed nodes; same argument as the other read sites.
        let rn = unsafe { RabitqNode::read(self.index, lsn_index_pointer, self.has_labels, stats) };
        let node = rn.get_archived_node();

        node.get_heap_item_pointer()
    }

    fn set_neighbors_on_disk<S: StatsNodeModify + StatsNodeRead>(
        &self,
        index_pointer: IndexPointer,
        neighbors: &[NeighborWithDistance],
        stats: &mut S,
    ) {
        let mut cache = self.cache().as_ref().unwrap().borrow_mut();

        /* It's important to preload cache with all the items since you can run into deadlocks
        if you try to fetch a quantized vector while holding the RabitqNode::modify lock */
        let iter = neighbors
            .iter()
            .map(|n| n.get_index_pointer_to_neighbor())
            .chain(once(index_pointer));
        cache.preload(iter, self, stats);

        // SAFETY: `index_pointer` addresses a live RaBitQ node item; `modify`
        // takes the page's exclusive lock so no concurrent reader/writer can
        // race the neighbor update, and `self.has_labels` matches the stored
        // variant (both derive from the meta page).
        let mut node = unsafe { RabitqNode::modify(self.index, index_pointer, self.has_labels, stats) };
        let mut archived = node.get_archived_node();
        archived.set_neighbors(neighbors, self.num_neighbors);
        node.commit();
    }

    fn get_distance_function(&self) -> DistanceFn {
        self.distance_fn
    }

    fn get_labels<S: StatsNodeRead>(
        &self,
        index_pointer: IndexPointer,
        stats: &mut S,
    ) -> Option<LabelSet> {
        if !self.has_labels {
            return None;
        }
        // SAFETY: caller passes a live node pointer; `self.has_labels` is true
        // here (checked above), so parsing the stored node as the Labeled
        // variant matches what was written.
        let rn = unsafe { RabitqNode::read(self.index, index_pointer, true, stats) };
        let node = rn.get_archived_node();
        node.get_labels().map(Into::into)
    }

    fn get_has_labels(&self) -> bool {
        self.has_labels
    }
}

pub type RabitqSpeedupStorageLsnPrivateData = PhantomData<bool>;

/// Build-time cache of quantized node payloads (code + norm metadata).
pub struct RabitqVectorCache {
    cache: crate::util::lru::LruCacheWithStats<ItemPointer, RabitqCacheEntry>,
}

#[derive(Clone)]
pub struct RabitqCacheEntry {
    pub code: Vec<u8>,
    pub l1_of_rotated: f32,
    pub sum_of_x2: f32,
    pub cent_dot: f32,
}

impl RabitqVectorCache {
    pub fn new(memory_budget: f64, code_len: usize, min_capacity: usize) -> Self {
        let total_memory = crate::access_method::build::maintenance_work_mem_bytes() as f64;
        let memory_budget = (total_memory * memory_budget).ceil() as usize;
        let entry_size = std::mem::size_of::<ItemPointer>()
            + std::mem::size_of::<Vec<u8>>()
            + code_len
            + std::mem::size_of::<f32>() * 3;
        let capacity = std::cmp::max(memory_budget / entry_size, min_capacity);

        Self {
            cache: crate::util::lru::LruCacheWithStats::new(
                std::num::NonZero::new(capacity).unwrap(),
                "RaBitQ vector",
            ),
        }
    }

    pub fn get<S: StatsNodeRead + StatsNodeWrite + StatsNodeModify>(
        &mut self,
        index_pointer: IndexPointer,
        storage: &RabitqSpeedupStorage,
        stats: &mut S,
    ) -> RabitqNodeData<'_> {
        if !self.cache.contains(&index_pointer) {
            // SAFETY: callers only pass pointers to live, sealed RaBitQ nodes
            // (neighbor lists of the node being visited); `storage.get_has_labels()`
            // matches the variant written to disk.
            let node = unsafe {
                RabitqNode::read(
                    storage.index,
                    index_pointer,
                    storage.get_has_labels(),
                    stats,
                )
            };
            let archived = node.get_archived_node();
            let data = archived.get_rabitq_data();
            self.cache.push(
                index_pointer,
                RabitqCacheEntry {
                    code: data.code.to_vec(),
                    l1_of_rotated: data.l1_of_rotated,
                    sum_of_x2: data.sum_of_x2,
                    cent_dot: data.cent_dot,
                },
            );
        }
        let entry = self.cache.get(&index_pointer).unwrap();
        RabitqNodeData {
            code: &entry.code,
            l1_of_rotated: entry.l1_of_rotated,
            sum_of_x2: entry.sum_of_x2,
            cent_dot: entry.cent_dot,
        }
    }

    pub fn preload<I: Iterator<Item = IndexPointer>, S: StatsNodeRead>(
        &mut self,
        index_pointers: I,
        storage: &RabitqSpeedupStorage,
        stats: &mut S,
    ) {
        for index_pointer in index_pointers {
            let item_pointer = ItemPointer::new(index_pointer.block_number, index_pointer.offset);
            if !self.cache.contains(&item_pointer) {
                // SAFETY: preload iterates pointers to live, sealed RaBitQ nodes
                // (the neighbors being written); the variant flag matches the
                // disk representation via `storage.get_has_labels()`.
                let node = unsafe {
                    RabitqNode::read(storage.index, item_pointer, storage.get_has_labels(), stats)
                };
                let archived = node.get_archived_node();
                let data = archived.get_rabitq_data();
                self.cache.push(
                    item_pointer,
                    RabitqCacheEntry {
                        code: data.code.to_vec(),
                        l1_of_rotated: data.l1_of_rotated,
                        sum_of_x2: data.sum_of_x2,
                        cent_dot: data.cent_dot,
                    },
                );
            }
        }
    }
}
