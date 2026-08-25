use std::num::NonZero;

use pgrx::debug1;

use crate::access_method::storage::Storage;
use crate::util::lru::LruCacheWithStats;

use crate::{
    access_method::{
        build::maintenance_work_mem_bytes,
        stats::{StatsNodeModify, StatsNodeRead, StatsNodeWrite},
    },
    util::{IndexPointer, ItemPointer},
};

use super::{node::RabitqNode, quantize::RabitqCode, RabitqStorage};

pub struct RabitqCodeCache {
    cache: LruCacheWithStats<ItemPointer, RabitqCode>,
}

impl RabitqCodeCache {
    pub fn new(memory_budget: f64, rabitq_vec_len: usize, min_capacity: usize) -> Self {
        let total_memory = maintenance_work_mem_bytes() as f64;
        let memory_budget = (total_memory * memory_budget).ceil() as usize;
        let capacity = std::cmp::max(
            memory_budget / Self::entry_size(rabitq_vec_len),
            min_capacity,
        );

        Self {
            cache: LruCacheWithStats::new(NonZero::new(capacity).unwrap(), "Rabitq code"),
        }
    }

    /// Estimate of the size of an entry in the cache in bytes.
    /// `rabitq_vec_len` is the number of u64 sign-code words; ex-code bytes are
    /// approximated by the same length (upper bound).
    pub fn entry_size(rabitq_vec_len: usize) -> usize {
        std::mem::size_of::<ItemPointer>()
            + std::mem::size_of::<Vec<u64>>()
            + std::mem::size_of::<Vec<u8>>()
            + 4 * std::mem::size_of::<f32>()
            + (std::mem::size_of::<u64>() * rabitq_vec_len)
            + (std::mem::size_of::<u8>() * rabitq_vec_len * 8)
    }

    pub fn get<S: StatsNodeRead + StatsNodeWrite + StatsNodeModify>(
        &mut self,
        index_pointer: IndexPointer,
        storage: &RabitqStorage,
        stats: &mut S,
    ) -> &RabitqCode {
        if !self.cache.contains(&index_pointer) {
            let node = unsafe {
                RabitqNode::read(
                    storage.index,
                    index_pointer,
                    storage.get_has_labels(),
                    stats,
                )
            };
            let code = node.get_archived_node().get_rabitq_code();
            self.cache.push(index_pointer, code);
        }

        self.cache.get(&index_pointer).unwrap()
    }

    pub fn preload<I: Iterator<Item = IndexPointer>, S: StatsNodeRead>(
        &mut self,
        index_pointers: I,
        storage: &RabitqStorage,
        stats: &mut S,
    ) {
        for index_pointer in index_pointers {
            let item_pointer = ItemPointer::new(index_pointer.block_number, index_pointer.offset);
            if !self.cache.contains(&item_pointer) {
                let node = unsafe {
                    RabitqNode::read(storage.index, item_pointer, storage.get_has_labels(), stats)
                };
                let code = node.get_archived_node().get_rabitq_code();
                self.cache.push(item_pointer, code);
            }
        }
    }
}

impl Drop for RabitqCodeCache {
    fn drop(&mut self) {
        debug1!(
            "Rabitq code cache teardown: capacity {}, stats: {:?}",
            self.cache.cap(),
            self.cache.stats()
        );
    }
}
