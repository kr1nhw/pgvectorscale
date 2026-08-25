pub mod node;
pub mod storage;

use pgrx::PgRelation;
use rkyv::{Archive, Deserialize, Serialize};

use crate::access_method::{
    distance::DistanceType,
    graph::neighbor_store::GraphNeighborStore,
    labels::LabeledVector,
    node::{ReadableNode, WriteableNode},
    quantization::rabitq::{code_dot_with_rotated, code_hamming, RabitqQuantizer, RabitqVector},
    stats::{StatsDistanceComparison, StatsNodeModify, StatsNodeRead, StatsNodeWrite},
    storage::NodeDistanceMeasure,
};
use crate::util::{
    page::PageType,
    tape::Tape,
    IndexPointer, ItemPointer, ReadableBuffer, WritableBuffer,
};
use pgvectorscale_derive::{Readable, Writeable};

pub use node::{ArchivedRabitqNode, RabitqNode, RabitqNodeData};
pub use storage::{
    RabitqCacheEntry, RabitqSpeedupStorage, RabitqSpeedupStorageLsnPrivateData, RabitqVectorCache,
};

/// Persisted RaBitQ quantizer configuration.  Stored in its own page (like
/// SbqMeans) referenced from the meta page's quantizer metadata pointer, so
/// the meta page format itself does not need to change.
#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
#[repr(C)]
pub struct RabitqQuantizerMetadata {
    pub rotation_seed: u64,
    pub num_bits: u8,
    pub dim: u32,
    /// Optional global center (dataset mean) subtracted before rotation.
    /// Empty for distance types that do not use centering.
    pub center: Vec<f32>,
}

impl RabitqQuantizerMetadata {
    pub unsafe fn store<S: StatsNodeWrite>(
        index: &PgRelation,
        quantizer: &RabitqQuantizer,
        stats: &mut S,
    ) -> ItemPointer {
        let mut tape = Tape::new(index, PageType::RabitqMetadata);
        let node = RabitqQuantizerMetadata {
            rotation_seed: quantizer.rotation_seed,
            num_bits: quantizer.num_bits,
            dim: quantizer.dim() as u32,
            center: quantizer.center.clone(),
        };
        let ptr = node.write(&mut tape, stats);
        tape.close();
        ptr
    }

    pub unsafe fn load<S: StatsNodeRead>(
        index: &PgRelation,
        qip: ItemPointer,
        stats: &mut S,
    ) -> RabitqQuantizer {
        let node = RabitqQuantizerMetadata::read(index, qip, stats);
        let archived = node.get_archived_node();
        RabitqQuantizer::new(archived.num_bits, archived.rotation_seed, archived.dim as usize)
            .with_center(archived.center.to_vec())
    }
}

/// Search-time distance measure: holds the full-precision rotated query and
/// estimates distances to quantized node payloads.
pub struct RabitqSearchDistanceMeasure {
    pub query: LabeledVector,
    rotated_query: Vec<f32>,
    sum_q: f32,
    query_sum_of_x2: f32,
    distance_type: DistanceType,
    num_bits: u8,
}

/// byte -> signed (lo, hi) nibble pair as f32 (low nibble first).
/// Level ∈ [-8, 7]; nibble 0x08..0x0F decodes to -8..-1.
pub const NIBBLE_LUT: [(f32, f32); 256] = {
    let mut lut = [(0.0f32, 0.0f32); 256];
    let mut b = 0usize;
    while b < 256 {
        let lo = ((b & 0x0F) as i8) - if b & 0x08 != 0 { 16 } else { 0 };
        let hi = (((b >> 4) & 0x0F) as i8) - if b & 0x80 != 0 { 16 } else { 0 };
        lut[b] = (lo as f32, hi as f32);
        b += 1;
    }
    lut
};

/// Code L1 distance.  For 4-bit codes the nibbles must be unpacked: the raw
/// byte difference cancels low/high nibble contributions (lo diff -1, hi
/// diff +1 yields byte diff 15 while the true L1 is 2).
pub fn code_l1(a: &[u8], b: &[u8], num_bits: u8) -> f32 {
    if num_bits == 4 {
        let mut d = 0.0f32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let (lo_a, hi_a) = NIBBLE_LUT[x as usize];
            let (lo_b, hi_b) = NIBBLE_LUT[y as usize];
            d += (lo_a - lo_b).abs() + (hi_a - hi_b).abs();
        }
        d
    } else {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as i8 as f32 - y as i8 as f32).abs())
            .sum()
    }
}

/// Code-to-code cosine ordering for the Lance codes: the per-dim factor
/// `f = sign·code_scale + ex + code_bias` reconstructs the rotated residual,
/// so `cos ≈ ⟨f_a,f_b⟩/(‖f_a‖·‖f_b‖)` is the right similarity for the beam
/// and the prune.  Returns a pseudo-distance `1 − cos ∈ [0,2]`.
pub fn code_cos_distance(code_a: &[u8], code_b: &[u8], num_bits: u8) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    match num_bits {
        4 => {
            for (&x, &y) in code_a.iter().zip(code_b.iter()) {
                let (fa0, fa1) = NIBBLE4_FACTOR_LUT[x as usize];
                let (fb0, fb1) = NIBBLE4_FACTOR_LUT[y as usize];
                dot += fa0 * fb0 + fa1 * fb1;
                na += fa0 * fa0 + fa1 * fa1;
                nb += fb0 * fb0 + fb1 * fb1;
            }
        }
        _ => {
            for (&x, &y) in code_a.iter().zip(code_b.iter()) {
                let fa = BYTE8_FACTOR_LUT[x as usize];
                let fb = BYTE8_FACTOR_LUT[y as usize];
                dot += fa * fb;
                na += fa * fa;
                nb += fb * fb;
            }
        }
    }
    let cos = (dot / (na.sqrt() * nb.sqrt()).max(1e-9)).clamp(-1.0, 1.0);
    1.0 - cos
}

/// Lance-style `full_dot` of a stored code with a full-precision rotated
/// vector `rot`: for 1-bit `m = Σ sign(code)·rotᵢ`; for 4/8-bit
/// `code_scale·binary_ip + ex_dist + code_bias·Σrot`.
pub fn code_full_dot(code: &[u8], num_bits: u8, rot: &[f32], sum_rot: f32) -> f32 {
    match num_bits {
        4 => {
            let mut binary_ip = 0.0f32;
            let mut ex_dist = 0.0f32;
            for (bi, &b) in code.iter().enumerate() {
                let (ls, le, hs, he) = NIBBLE4_DOT_LUT[b as usize];
                let r0 = rot[bi * 2];
                let r1 = rot[bi * 2 + 1];
                binary_ip += ls * r0 + hs * r1;
                ex_dist += le * r0 + he * r1;
            }
            8.0 * binary_ip + ex_dist - 7.5 * sum_rot
        }
        8 => {
            let mut binary_ip = 0.0f32;
            let mut ex_dist = 0.0f32;
            for (i, &b) in code.iter().enumerate() {
                let r = rot[i];
                binary_ip += if b & 0x80 != 0 { r } else { -r };
                ex_dist += (b & 0x7F) as f32 * r;
            }
            128.0 * binary_ip + ex_dist - 127.5 * sum_rot
        }
        _ => code_dot_with_rotated(code, rot, sum_rot),
    }
}

/// byte -> (lo, hi) signed factor pair (4-bit Lance code).
/// f = sign·8 + ex − 7.5 with sign = bit3, ex = bits 0-2.
static NIBBLE4_FACTOR_LUT: [(f32, f32); 256] = {
    let mut lut = [(0.0f32, 0.0f32); 256];
    let mut b = 0usize;
    while b < 256 {
        let lo = b & 0x0F;
        let hi = b >> 4;
        let f_lo = if lo & 0x08 != 0 { 8.0 } else { -8.0 } + (lo & 0x07) as f32 - 7.5;
        let f_hi = if hi & 0x08 != 0 { 8.0 } else { -8.0 } + (hi & 0x07) as f32 - 7.5;
        lut[b] = (f_lo, f_hi);
        b += 1;
    }
    lut
};

/// byte -> (lo_sign, lo_ex, hi_sign, hi_ex) as f32 (4-bit Lance code),
/// for the query-side full_dot: binary_ip += sign·rot, ex_dist += ex·rot.
static NIBBLE4_DOT_LUT: [(f32, f32, f32, f32); 256] = {
    let mut lut = [(0.0f32, 0.0f32, 0.0f32, 0.0f32); 256];
    let mut b = 0usize;
    while b < 256 {
        let lo = b & 0x0F;
        let hi = b >> 4;
        lut[b] = (
            if lo & 0x08 != 0 { 1.0 } else { -1.0 },
            (lo & 0x07) as f32,
            if hi & 0x08 != 0 { 1.0 } else { -1.0 },
            (hi & 0x07) as f32,
        );
        b += 1;
    }
    lut
};

/// byte -> signed factor (8-bit Lance code): f = sign·128 + ex − 127.5.
static BYTE8_FACTOR_LUT: [f32; 256] = {
    let mut lut = [0.0f32; 256];
    let mut b = 0usize;
    while b < 256 {
        lut[b] = if b & 0x80 != 0 { 128.0 } else { -128.0 } + (b & 0x7F) as f32 - 127.5;
        b += 1;
    }
    lut
};

impl RabitqSearchDistanceMeasure {
    pub fn new(
        quantizer: &RabitqQuantizer,
        query: LabeledVector,
        distance_type: crate::access_method::distance::DistanceType,
    ) -> Self {
        let rq = quantizer.rotate_query(query.vec().to_index_slice());
        let sum_q = rq.rotated.iter().sum::<f32>();
        Self {
            query,
            rotated_query: rq.rotated,
            sum_q,
            query_sum_of_x2: rq.sum_of_x2,
            distance_type,
            num_bits: quantizer.num_bits,
        }
    }

    /// Estimate the distance between the query and a quantized node payload
    /// (Lance-style raw-query L2 estimator):
    ///   distance = full_dot·scale_factor + add_factor + ‖q−c‖²
    /// with full_dot from `code_full_dot`, scale = −2‖r‖²/⟨r,code⟩ and
    /// add = ‖r‖² + 2·‖r‖²·⟨c,code⟩/⟨r,code⟩.
    pub fn calculate_bq_distance<S: StatsDistanceComparison>(
        &self,
        data: RabitqNodeData,
        _gns: &GraphNeighborStore,
        stats: &mut S,
    ) -> f32 {
        stats.record_quantized_distance_comparison();
        let full_dot = code_full_dot(&data.code, self.num_bits, &self.rotated_query, self.sum_q);
        let cent_dot = data.cent_dot;
        let res_dot = data.l1_of_rotated.max(1e-9);
        let norm_sq = data.sum_of_x2;
        let scale = -2.0 * norm_sq / res_dot;
        let add = norm_sq + 2.0 * norm_sq * cent_dot / res_dot;
        // The affine estimator is unbiased around the true distance but can
        // go slightly negative for very close vectors; clamp so the graph's
        // non-negative distance invariant holds (ordering is preserved).
        (full_dot * scale + add + self.query_sum_of_x2).max(0.0)
    }
}

/// Node-to-node distance measure (used during graph construction/pruning).
pub struct RabitqNodeDistanceMeasure<'a> {
    data: RabitqCacheEntry,
    storage: &'a RabitqSpeedupStorage<'a>,
}

impl<'a> RabitqNodeDistanceMeasure<'a> {
    pub unsafe fn with_index_pointer<T: StatsNodeRead + StatsNodeWrite + StatsNodeModify>(
        storage: &'a RabitqSpeedupStorage<'a>,
        index_pointer: IndexPointer,
        stats: &mut T,
    ) -> Self {
        let mut cache = storage.cache().as_ref().unwrap().borrow_mut();
        let data = cache.get(index_pointer, storage, stats);
        Self {
            data: RabitqCacheEntry {
                code: data.code.to_vec(),
                l1_of_rotated: data.l1_of_rotated,
                sum_of_x2: data.sum_of_x2,
                cent_dot: data.cent_dot,
            },
            storage,
        }
    }
}

impl NodeDistanceMeasure for RabitqNodeDistanceMeasure<'_> {
    unsafe fn get_distance<
        T: StatsNodeRead + StatsDistanceComparison + StatsNodeWrite + StatsNodeModify,
    >(
        &self,
        index_pointer: IndexPointer,
        stats: &mut T,
    ) -> f32 {
        let mut cache = self.storage.cache().as_ref().unwrap().borrow_mut();
        let other = cache.get(index_pointer, self.storage, stats);
        match self.storage.quantizer_num_bits() {
            4 | 8 => code_cos_distance(
                &self.data.code,
                &other.code,
                self.storage.quantizer_num_bits(),
            ),
            _ => {
                // code-to-code cosine estimate via the arcsin identity
                // (monotone with hamming distance).
                let d = self.data.code.len() as f32 * 8.0;
                let m12 = d - 2.0 * code_hamming(&self.data.code, other.code) as f32;
                let cos = (std::f32::consts::FRAC_PI_2 * (m12 / d)).sin();
                RabitqVector::distance_from_cos(
                    cos,
                    self.data.sum_of_x2,
                    other.sum_of_x2,
                    self.storage.distance_type,
                )
            }
        }
    }
}
