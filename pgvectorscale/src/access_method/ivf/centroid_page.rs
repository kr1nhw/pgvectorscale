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

const CENTROID_PAGE_BLOCK_NUMBER: pg_sys::BlockNumber = 2;
const CENTROID_PAGE_OFFSET: pgrx::pg_sys::OffsetNumber = 1;

/// IVF centroid page containing all centroids.
/// Stored on Page 2.
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

    /// Store the centroid page to the index.
    pub unsafe fn store(&self, index: &PgRelation, first_time: bool) {
        let mut stats = crate::access_method::stats::WriteStats::default();
        let mut tape = if first_time {
            ChainTapeWriter::new(index, PageType::IvfCentroids, &mut stats)
        } else {
            ChainTapeWriter::reinit(
                index,
                PageType::IvfCentroids,
                &mut stats,
                CENTROID_PAGE_BLOCK_NUMBER,
            )
        };

        let bytes = self.serialize_to_vec();
        let off = tape.write(&bytes);
        assert_eq!(
            off,
            ItemPointer::new(CENTROID_PAGE_BLOCK_NUMBER, CENTROID_PAGE_OFFSET)
        );
    }

    /// Load the centroid page from the index.
    pub fn load(index: &PgRelation) -> IvfCentroidPage {
        unsafe {
            let mut stats = crate::access_method::stats::WriteStats::default();
            let mut tape = ChainItemReader::new(index, PageType::IvfCentroids, &mut stats);

            let mut buf: Vec<u8> = Vec::new();
            for item in tape.read(ItemPointer::new(
                CENTROID_PAGE_BLOCK_NUMBER,
                CENTROID_PAGE_OFFSET,
            )) {
                buf.extend_from_slice(item.get_data_slice());
            }
            rkyv::from_bytes::<IvfCentroidPage>(&buf).unwrap()
        }
    }
}
