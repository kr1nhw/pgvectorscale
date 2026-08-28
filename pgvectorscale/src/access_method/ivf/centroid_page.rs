//! IVF centroid page management.
//!
//! The centroid page (Page 2) stores all centroids for the IVF index.

use pgrx::*;
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{Archive, Deserialize, Serialize};

use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::util::chain::{ChainItemReader, ChainTapeWriter};
use crate::util::page::PageType;
use crate::util::*;

/// IVF centroid page containing all centroids, stored as a chained item at a
/// dynamically-assigned block recorded in the meta page.
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct IvfCentroidPage {
    /// All centroids in the index
    pub centroids: Vec<Vec<f32>>,
}

impl IvfCentroidPage {
    /// Create a new centroid page with the given centroids.
    pub fn new(centroids: Vec<Vec<f32>>) -> Self {
        Self { centroids }
    }

    /// Get the number of centroids.
    pub fn num_centroids(&self) -> usize {
        self.centroids.len()
    }

    /// Get a centroid by index.
    pub fn get_centroid(&self, index: usize) -> Option<&Vec<f32>> {
        self.centroids.get(index)
    }

    /// Store the centroid page and return the `ItemPointer` to its first chunk.
    /// `existing_block` reinitializes an already-allocated chain (insert path);
    /// `None` allocates a fresh chain (build path).
    pub unsafe fn store(
        &self,
        index: &PgRelation,
        existing_block: Option<pg_sys::BlockNumber>,
    ) -> ItemPointer {
        let mut stats = crate::access_method::stats::WriteStats::default();
        let mut tape = match existing_block {
            None => ChainTapeWriter::new(index, PageType::IvfCentroids, &mut stats),
            Some(b) => ChainTapeWriter::reinit(index, PageType::IvfCentroids, &mut stats, b),
        };

        let bytes = self.serialize_to_vec();
        tape.write(&bytes)
    }

    /// Load the centroid page from the given pointer.
    pub fn load(index: &PgRelation, pointer: ItemPointer) -> IvfCentroidPage {
        unsafe {
            let mut stats = crate::access_method::stats::WriteStats::default();
            let mut tape = ChainItemReader::new(index, PageType::IvfCentroids, &mut stats);

            let mut buf: Vec<u8> = Vec::new();
            for item in tape.read(pointer) {
                buf.extend_from_slice(item.get_data_slice());
            }
            rkyv::from_bytes::<IvfCentroidPage>(&buf).unwrap()
        }
    }
}
