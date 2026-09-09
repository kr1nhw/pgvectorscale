use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::access_method::PgRelation;
use crate::util::tape::Tape;
use crate::util::{ArchivedItemPointer, HeapPointer, ItemPointer, ReadableBuffer, WritableBuffer};
use pgrx::pg_sys::{InvalidBlockNumber, InvalidOffsetNumber};
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{vec::ArchivedVec, Archive, Deserialize, Serialize};
use std::fmt::Debug;
use std::pin::Pin;

use crate::access_method::{
    graph::neighbor_with_distance::NeighborWithDistance,
    labels::{ArchivedLabelSet, LabelSet},
    meta_page::MetaPage,
    stats::{StatsNodeModify, StatsNodeRead, StatsNodeWrite},
    storage::{ArchivedData, NodeVacuum},
};

/// The quantized payload of a RaBitQ node: packed sign-bit code plus the
/// norm metadata needed by the estimator.
#[derive(Clone, Copy, Debug)]
pub struct RabitqNodeData<'a> {
    pub code: &'a [u8],
    pub l1_of_rotated: f32,
    pub sum_of_x2: f32,
    /// ⟨rot_c, code⟩ — the rotated-center correction, precomputed at build.
    pub cent_dot: f32,
}

/// A node in a RaBitQ-compressed index.
pub enum RabitqNode {
    Classic(ClassicRabitqNode),
    Labeled(LabeledRabitqNode),
}

#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
#[repr(C)]
pub struct ClassicRabitqNode {
    pub heap_item_pointer: HeapPointer,
    pub code: Vec<u8>,
    pub l1_of_rotated: f32,
    pub sum_of_x2: f32,
    pub cent_dot: f32,
    neighbor_index_pointers: Vec<ItemPointer>,
}

#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
#[repr(C)]
pub struct LabeledRabitqNode {
    heap_item_pointer: HeapPointer,
    code: Vec<u8>,
    l1_of_rotated: f32,
    sum_of_x2: f32,
    cent_dot: f32,
    neighbor_index_pointers: Vec<ItemPointer>,
    labels: LabelSet,
}

impl RabitqNode {
    pub fn with_meta(
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        code: &[u8],
        l1_of_rotated: f32,
        sum_of_x2: f32,
        cent_dot: f32,
        labels: Option<LabelSet>,
    ) -> Self {
        Self::new(
            heap_pointer,
            meta_page.get_num_neighbors() as usize,
            meta_page.has_labels(),
            code,
            l1_of_rotated,
            sum_of_x2,
            cent_dot,
            labels,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        heap_pointer: HeapPointer,
        num_neighbors: usize,
        has_labels: bool,
        code: &[u8],
        l1_of_rotated: f32,
        sum_of_x2: f32,
        cent_dot: f32,
        labels: Option<LabelSet>,
    ) -> Self {
        // always use vectors of num_neighbors in length because we never want the serialized size of a Node to change
        let neighbor_index_pointers: Vec<_> = (0..num_neighbors)
            .map(|_| ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber))
            .collect();

        if has_labels {
            RabitqNode::Labeled(LabeledRabitqNode {
                heap_item_pointer: heap_pointer,
                code: code.to_vec(),
                l1_of_rotated,
                sum_of_x2,
                cent_dot,
                neighbor_index_pointers,
                labels: labels.unwrap_or_default(),
            })
        } else {
            RabitqNode::Classic(ClassicRabitqNode {
                heap_item_pointer: heap_pointer,
                code: code.to_vec(),
                l1_of_rotated,
                sum_of_x2,
                cent_dot,
                neighbor_index_pointers,
            })
        }
    }

    pub unsafe fn read<'a, S: StatsNodeRead>(
        index: &'a PgRelation,
        index_pointer: ItemPointer,
        has_labels: bool,
        stats: &mut S,
    ) -> ReadableRabitqNode<'a> {
        if has_labels {
            ReadableRabitqNode::Labeled(LabeledRabitqNode::read(index, index_pointer, stats))
        } else {
            ReadableRabitqNode::Classic(ClassicRabitqNode::read(index, index_pointer, stats))
        }
    }

    pub unsafe fn modify<'a, S: StatsNodeModify>(
        index: &'a PgRelation,
        index_pointer: ItemPointer,
        has_labels: bool,
        stats: &mut S,
    ) -> WritableRabitqNode<'a> {
        if has_labels {
            WritableRabitqNode::Labeled(LabeledRabitqNode::modify(index, index_pointer, stats))
        } else {
            WritableRabitqNode::Classic(ClassicRabitqNode::modify(index, index_pointer, stats))
        }
    }

    pub fn write<S: StatsNodeWrite>(&self, tape: &mut Tape, stats: &mut S) -> ItemPointer {
        match self {
            RabitqNode::Classic(node) => node.write(tape, stats),
            RabitqNode::Labeled(node) => node.write(tape, stats),
        }
    }
}

impl NodeVacuum for ArchivedClassicRabitqNode {
    fn with_data(data: &mut [u8]) -> Pin<&mut Self> {
        ArchivedClassicRabitqNode::with_data(data)
    }

    fn delete(self: Pin<&mut Self>) {
        //TODO: actually optimize the deletes by removing index tuples. For now just mark it.
        // SAFETY: `with_data` hands out a Pin<&mut Self> over the item's bytes;
        // projecting the first field keeps the pinned-struct invariants (the
        // returned field stays inside the same allocation and is unique).
        let mut heap_pointer = unsafe { self.map_unchecked_mut(|s| &mut s.heap_item_pointer) };
        heap_pointer.offset = InvalidOffsetNumber;
        heap_pointer.block_number = InvalidBlockNumber;
    }
}

impl NodeVacuum for ArchivedLabeledRabitqNode {
    fn with_data(data: &mut [u8]) -> Pin<&mut Self> {
        ArchivedLabeledRabitqNode::with_data(data)
    }

    fn delete(self: Pin<&mut Self>) {
        // SAFETY: same field projection as `ArchivedClassicRabitqNode::delete`;
        // the pinned struct is only borrowed structurally, never moved.
        let mut heap_pointer = unsafe { self.map_unchecked_mut(|s| &mut s.heap_item_pointer) };
        heap_pointer.offset = InvalidOffsetNumber;
        heap_pointer.block_number = InvalidBlockNumber;
    }
}

impl ArchivedData for ArchivedClassicRabitqNode {
    fn is_deleted(&self) -> bool {
        self.heap_item_pointer.offset == InvalidOffsetNumber
    }

    fn get_heap_item_pointer(&self) -> HeapPointer {
        self.heap_item_pointer.deserialize_item_pointer()
    }

    fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer> {
        self.neighbor_index_pointers
            .iter()
            .map(|p| p.deserialize_item_pointer())
            .collect()
    }
}

impl ArchivedData for ArchivedLabeledRabitqNode {
    fn is_deleted(&self) -> bool {
        self.heap_item_pointer.offset == InvalidOffsetNumber
    }

    fn get_heap_item_pointer(&self) -> HeapPointer {
        self.heap_item_pointer.deserialize_item_pointer()
    }

    fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer> {
        self.neighbor_index_pointers
            .iter()
            .map(|p| p.deserialize_item_pointer())
            .collect()
    }
}

pub enum ReadableRabitqNode<'a> {
    Classic(ReadableClassicRabitqNode<'a>),
    Labeled(ReadableLabeledRabitqNode<'a>),
}

impl<'a> ReadableRabitqNode<'a> {
    pub fn get_archived_node(&'a self) -> ArchivedRabitqNode<'a> {
        match self {
            ReadableRabitqNode::Classic(node) => {
                ArchivedRabitqNode::Classic(node.get_archived_node())
            }
            ReadableRabitqNode::Labeled(node) => {
                ArchivedRabitqNode::Labeled(node.get_archived_node())
            }
        }
    }
}

pub enum WritableRabitqNode<'a> {
    Classic(WritableClassicRabitqNode<'a>),
    Labeled(WritableLabeledRabitqNode<'a>),
}

impl WritableRabitqNode<'_> {
    pub fn get_archived_node(&mut self) -> ArchivedMutRabitqNode<'_> {
        match self {
            WritableRabitqNode::Classic(node) => {
                ArchivedMutRabitqNode::Classic(node.get_archived_node())
            }
            WritableRabitqNode::Labeled(node) => {
                ArchivedMutRabitqNode::Labeled(node.get_archived_node())
            }
        }
    }

    pub fn commit(self) {
        match self {
            WritableRabitqNode::Classic(node) => node.commit(),
            WritableRabitqNode::Labeled(node) => node.commit(),
        }
    }
}

pub enum ArchivedMutRabitqNode<'a> {
    Classic(Pin<&'a mut ArchivedClassicRabitqNode>),
    Labeled(Pin<&'a mut ArchivedLabeledRabitqNode>),
}

pub enum ArchivedRabitqNode<'a> {
    Classic(&'a ArchivedClassicRabitqNode),
    Labeled(&'a ArchivedLabeledRabitqNode),
}

impl ArchivedData for ArchivedRabitqNode<'_> {
    fn is_deleted(&self) -> bool {
        match self {
            ArchivedRabitqNode::Classic(node) => node.is_deleted(),
            ArchivedRabitqNode::Labeled(node) => node.is_deleted(),
        }
    }

    fn get_heap_item_pointer(&self) -> HeapPointer {
        match self {
            ArchivedRabitqNode::Classic(node) => node.get_heap_item_pointer(),
            ArchivedRabitqNode::Labeled(node) => node.get_heap_item_pointer(),
        }
    }

    fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer> {
        match self {
            ArchivedRabitqNode::Classic(node) => node.get_index_pointer_to_neighbors(),
            ArchivedRabitqNode::Labeled(node) => node.get_index_pointer_to_neighbors(),
        }
    }
}

impl Debug for ArchivedRabitqNode<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArchivedRabitqNode::Classic(node) => node.fmt(f),
            ArchivedRabitqNode::Labeled(node) => node.fmt(f),
        }
    }
}

impl ArchivedRabitqNode<'_> {
    pub fn num_neighbors(&self) -> usize {
        match self {
            ArchivedRabitqNode::Classic(node) => node
                .neighbor_index_pointers
                .iter()
                .position(|f| f.block_number == InvalidBlockNumber)
                .unwrap_or(node.neighbor_index_pointers.len()),
            ArchivedRabitqNode::Labeled(node) => node
                .neighbor_index_pointers
                .iter()
                .position(|f| f.block_number == InvalidBlockNumber)
                .unwrap_or(node.neighbor_index_pointers.len()),
        }
    }

    pub fn iter_neighbors(&self) -> impl Iterator<Item = ItemPointer> + '_ {
        let neighbor_index_pointers = match self {
            ArchivedRabitqNode::Classic(node) => &node.neighbor_index_pointers,
            ArchivedRabitqNode::Labeled(node) => &node.neighbor_index_pointers,
        };
        neighbor_index_pointers
            .iter()
            .take(self.num_neighbors())
            .map(|ip| ip.deserialize_item_pointer())
    }

    pub fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer> {
        self.iter_neighbors().collect()
    }

    pub fn get_rabitq_data(&self) -> RabitqNodeData<'_> {
        match self {
            ArchivedRabitqNode::Classic(node) => RabitqNodeData {
                code: &node.code,
                l1_of_rotated: node.l1_of_rotated,
                sum_of_x2: node.sum_of_x2,
                cent_dot: node.cent_dot,
            },
            ArchivedRabitqNode::Labeled(node) => RabitqNodeData {
                code: &node.code,
                l1_of_rotated: node.l1_of_rotated,
                sum_of_x2: node.sum_of_x2,
                cent_dot: node.cent_dot,
            },
        }
    }

    pub fn get_heap_item_pointer(&self) -> HeapPointer {
        match self {
            ArchivedRabitqNode::Classic(node) => node.heap_item_pointer.deserialize_item_pointer(),
            ArchivedRabitqNode::Labeled(node) => node.heap_item_pointer.deserialize_item_pointer(),
        }
    }

    pub fn get_labels(&self) -> Option<&ArchivedLabelSet> {
        match self {
            ArchivedRabitqNode::Classic(_) => None,
            ArchivedRabitqNode::Labeled(node) => Some(&node.labels),
        }
    }
}

impl ArchivedData for ArchivedMutRabitqNode<'_> {
    fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer> {
        self.iter_neighbors().collect()
    }

    fn is_deleted(&self) -> bool {
        self.get_heap_item_pointer().offset == InvalidOffsetNumber
    }

    fn get_heap_item_pointer(&self) -> HeapPointer {
        let hip = match self {
            ArchivedMutRabitqNode::Classic(node) => &node.heap_item_pointer,
            ArchivedMutRabitqNode::Labeled(node) => &node.heap_item_pointer,
        };
        hip.deserialize_item_pointer()
    }
}

impl Debug for ArchivedClassicRabitqNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArchivedClassicRabitqNode")
            .field(
                "heap_item_pointer.block_number",
                &self.heap_item_pointer.block_number,
            )
            .field("heap_item_pointer.offset", &self.heap_item_pointer.offset)
            .field("code.len()", &self.code.len())
            .field("l1_of_rotated", &self.l1_of_rotated)
            .field("sum_of_x2", &self.sum_of_x2)
            .field(
                "neighbor_index_pointers.len()",
                &self.neighbor_index_pointers.len(),
            )
            .finish()
    }
}

impl Debug for ArchivedLabeledRabitqNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArchivedLabeledRabitqNode")
            .field(
                "heap_item_pointer.block_number",
                &self.heap_item_pointer.block_number,
            )
            .field("heap_item_pointer.offset", &self.heap_item_pointer.offset)
            .field("code.len()", &self.code.len())
            .field("l1_of_rotated", &self.l1_of_rotated)
            .field("sum_of_x2", &self.sum_of_x2)
            .field(
                "neighbor_index_pointers.len()",
                &self.neighbor_index_pointers.len(),
            )
            .field("labels", &self.labels)
            .finish()
    }
}

impl<'a> ArchivedMutRabitqNode<'a> {
    fn neighbor_index_pointer(&'a mut self) -> Pin<&'a mut ArchivedVec<ArchivedItemPointer>> {
        match self {
            // SAFETY: structural field projection of a pinned archived node;
            // the returned field borrows the same allocation and is never moved.
            ArchivedMutRabitqNode::Classic(node) => unsafe {
                node.as_mut()
                    .map_unchecked_mut(|s| &mut s.neighbor_index_pointers)
            },
            ArchivedMutRabitqNode::Labeled(node) => unsafe {
                node.as_mut()
                    .map_unchecked_mut(|s| &mut s.neighbor_index_pointers)
            },
        }
    }

    pub fn set_neighbors(&'a mut self, neighbors: &[NeighborWithDistance], num_neighbors: u32) {
        let mut neighbor_index_pointer = self.neighbor_index_pointer();
        for (i, new_neighbor) in neighbors.iter().enumerate() {
            let mut a_index_pointer = neighbor_index_pointer.as_mut().index_pin(i);
            let ip = new_neighbor.get_index_pointer_to_neighbor();
            a_index_pointer.block_number = ip.block_number;
            a_index_pointer.offset = ip.offset;
        }
        //set the marker that the list ended
        if neighbors.len() < num_neighbors as _ {
            let mut past_last_index_pointers = neighbor_index_pointer.index_pin(neighbors.len());
            past_last_index_pointers.block_number = InvalidBlockNumber;
            past_last_index_pointers.offset = InvalidOffsetNumber;
        }
    }

    pub fn num_neighbors(&self) -> usize {
        match self {
            ArchivedMutRabitqNode::Classic(node) => node
                .neighbor_index_pointers
                .iter()
                .position(|f| f.block_number == InvalidBlockNumber)
                .unwrap_or(node.neighbor_index_pointers.len()),
            ArchivedMutRabitqNode::Labeled(node) => node
                .neighbor_index_pointers
                .iter()
                .position(|f| f.block_number == InvalidBlockNumber)
                .unwrap_or(node.neighbor_index_pointers.len()),
        }
    }

    pub fn iter_neighbors(&self) -> impl Iterator<Item = ItemPointer> + '_ {
        let neighbor_index_pointers = match self {
            ArchivedMutRabitqNode::Classic(node) => &node.neighbor_index_pointers,
            ArchivedMutRabitqNode::Labeled(node) => &node.neighbor_index_pointers,
        };

        neighbor_index_pointers
            .iter()
            .take(self.num_neighbors())
            .map(|ip| ip.deserialize_item_pointer())
    }
}
