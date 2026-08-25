mod cache;
pub mod node;
pub mod quantize;
pub(crate) mod rotation;
pub mod storage;
#[cfg(any(test, feature = "pg_test"))]
mod tests;

use super::{
    distance::distance_xor_optimized,
    graph::neighbor_store::GraphNeighborStore,
    labels::LabeledVector,
    stats::{StatsDistanceComparison, StatsNodeModify, StatsNodeRead, StatsNodeWrite},
    storage::NodeDistanceMeasure,
};

use quantize::{RabitqCode, RabitqQuantizer};

use pgrx::PgRelation;
use rkyv::{Archive, Deserialize, Serialize};
use storage::RabitqStorage;

use super::meta_page::MetaPage;
use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::util::{
    chain::{ChainItemReader, ChainTapeWriter},
    page::PageType,
    ItemPointer, ReadableBuffer, WritableBuffer,
};
use pgvectorscale_derive::{Readable, Writeable};

pub type RabitqVectorElement = u64;

/// Persisted RaBitQ quantizer metadata: global-mean centroid + rotation signs.
#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
#[repr(C)]
pub struct RabitqMetadata {
    count: u64,
    mean: Vec<f32>,
    rotation_signs: Vec<u8>,
    num_bits: u8,
}

impl RabitqMetadata {
    pub unsafe fn load<S: StatsNodeRead>(
        index: &PgRelation,
        meta_page: &MetaPage,
        stats: &mut S,
    ) -> RabitqQuantizer {
        let mut quantizer = RabitqQuantizer::new(meta_page, 1); // num_bits overwritten below
        let qip = meta_page
            .get_quantizer_metadata_pointer()
            .unwrap_or_else(|| pgrx::error!("No RaBitQ pointer found in meta page"));

        let mut tape_reader = ChainItemReader::new(index, PageType::RabitqMetadata, stats);
        let mut buf: Vec<u8> = Vec::new();
        for item in tape_reader.read(qip) {
            buf.extend_from_slice(item.get_data_slice());
        }

        let metadata = rkyv::from_bytes::<RabitqMetadata>(buf.as_slice()).unwrap();
        quantizer.load(
            metadata.count,
            metadata.mean,
            metadata.rotation_signs,
            metadata.num_bits,
        );
        quantizer
    }

    pub unsafe fn store<S: StatsNodeWrite>(
        index: &PgRelation,
        quantizer: &RabitqQuantizer,
        stats: &mut S,
    ) -> ItemPointer {
        let metadata = RabitqMetadata {
            count: quantizer.count,
            mean: quantizer.mean.clone(),
            rotation_signs: quantizer.rotation_signs.clone(),
            num_bits: quantizer.num_bits,
        };
        let mut tape = ChainTapeWriter::new(index, PageType::RabitqMetadata, stats);
        let buf = rkyv::to_bytes::<_, 1024>(&metadata).unwrap();
        tape.write(&buf)
    }
}

pub struct RabitqSearchDistanceMeasure {
    quantizer: RabitqQuantizer,
    query: LabeledVector,
    measure: quantize::RabitqQueryMeasure,
}

impl RabitqSearchDistanceMeasure {
    pub fn new(quantizer: &RabitqQuantizer, query: LabeledVector) -> Self {
        let measure = quantizer.query_measure(query.vec().to_index_slice());
        Self {
            quantizer: quantizer.clone(),
            query,
            measure,
        }
    }

    pub fn calculate_rabitq_distance<S: StatsDistanceComparison>(
        &self,
        code: &RabitqCode,
        _gns: &GraphNeighborStore,
        stats: &mut S,
    ) -> f32 {
        stats.record_quantized_distance_comparison();
        self.quantizer.estimate_distance(&self.measure, code)
    }
}

pub struct RabitqNodeDistanceMeasure<'a> {
    vec: Vec<RabitqVectorElement>,
    storage: &'a RabitqStorage<'a>,
}

impl<'a> RabitqNodeDistanceMeasure<'a> {
    pub unsafe fn with_index_pointer<T: StatsNodeRead + StatsNodeWrite + StatsNodeModify>(
        storage: &'a RabitqStorage<'a>,
        index_pointer: crate::util::IndexPointer,
        stats: &mut T,
    ) -> Self {
        let mut cache = storage.cache().as_ref().unwrap().borrow_mut();
        let code = cache.get(index_pointer, storage, stats);
        Self {
            vec: code.code.clone(),
            storage,
        }
    }
}

impl NodeDistanceMeasure for RabitqNodeDistanceMeasure<'_> {
    unsafe fn get_distance<
        T: StatsNodeRead + StatsDistanceComparison + StatsNodeWrite + StatsNodeModify,
    >(
        &self,
        index_pointer: crate::util::IndexPointer,
        stats: &mut T,
    ) -> f32 {
        let mut cache = self.storage.cache().as_ref().unwrap().borrow_mut();
        let code = cache.get(index_pointer, self.storage, stats);
        // Node-vs-node distance uses the Hamming distance between sign codes
        // (as in SBQ); search uses the more accurate continuous-query estimator.
        distance_xor_optimized(&code.code, self.vec.as_slice()) as f32
    }
}
