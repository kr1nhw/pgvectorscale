//! hnswsq precision codecs.
//!
//! Four storage layouts for node vectors:
//!
//! | layout       | format                                   | bytes/dim | training |
//! |--------------|------------------------------------------|-----------|----------|
//! | `plain`      | IEEE f32 verbatim                        | 4         | none     |
//! | `ieeefp16`   | IEEE 754 binary16 (`half::f16`)          | 2         | none     |
//! | `ieeefp8`    | OCP FP8 E4M3 (`half::f8e4m3`)            | 1         | none     |
//! | `f8` (sq8)   | Lance-style per-dimension min/max linear | 1         | build-time calibration |
//!
//! The IEEE layouts are *stateless* casts: they need no index-level training
//! artifact, which makes them the right choice for incremental index building
//! (empty-start index, rows appended over time).  SQ8 freezes a per-dimension
//! `(min, max)` range at build; inserts outside the range clamp (accuracy loss
//! only, never a correctness issue), and an empty-start SQ8 index gets a
//! provisional `[-1, 1]` range until `REINDEX` retrains from real data.
//!
//! Distances are always computed by decoding into an f32 scratch buffer and
//! calling the existing SIMD kernels ([`crate::access_method::distance`]).

use half::f16;
use pgrx::*;
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{Archive, Deserialize, Serialize};

use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::util::chain::{ChainItemReader, ChainTapeWriter};
use crate::util::page::PageType;
use crate::util::*;

/// FP8 E4M3 maximum finite magnitude (OCP FP8 standard).  Encodes pre-clamp
/// f32 inputs to `±FP8_E4M3_MAX` so saturation semantics never depend on the
/// converter's overflow behavior.
pub const FP8_E4M3_MAX: f32 = 448.0;

/// IEEE binary16 maximum finite magnitude.  Encodes pre-clamp to keep stored
/// values (and therefore distances) finite.
pub const FP16_MAX: f32 = 65504.0;

/// Identifies which reduced-precision layout an hnswsq index stores node
/// vectors in.  Persisted as a `u8` in the meta page; do not renumber.

/// Fixed-range int8 SQ (training-free): `round(clamp(x, 0, 255))` stored as
/// an unsigned byte, scale 1.0 — byte-valued vectors (BIGANN/SIFT-style
/// uint8 data) encode EXACTLY, so the integer pairwise distance
/// `SUM (qhat - code)^2` reproduces the decoded-domain L2 ranking (of the
/// quantized query) exactly on the stored side.
pub const SQ8_FIXED_SCALE: f32 = 1.0;
/// Fixed-range int16 SQ (training-free): `round(clamp(x, 0, 255) * 128)`,
/// scale 2^-7 — 128 sub-steps per unit over the same [0, 255] range.
pub const SQ16_FIXED_SCALE: f32 = 0.007_812_5; // 2^-7
/// Encoded range of the fixed SQ layouts (`[0, 255]` — the uint8 domain;
/// byte-valued vectors encode losslessly).
pub const SQ_FIXED_MIN: f32 = 0.0;
pub const SQ_FIXED_MAX: f32 = 255.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HnswPrecision {
    /// IEEE f32 verbatim (4 bytes/dim).
    Plain = 0,
    /// IEEE 754 binary16 (2 bytes/dim), training-free.
    IeeeFp16 = 1,
    /// Lance-style SQ8: per-dimension min/max linear quantization
    /// (1 byte/dim), calibrated at build.
    Sq8 = 2,
    /// OCP FP8 E4M3 (1 byte/dim), training-free.
    IeeeFp8 = 3,
    /// Fixed-range int8 SQ (1 byte/dim), training-free.  The range is a
    /// global `[0, 255]` constant (no calibration): byte-valued vectors
    /// encode losslessly, and distances are computable directly from the
    /// codes.
    Sq8Fixed = 4,
    /// Fixed-range int16 SQ (2 bytes/dim), training-free.  Same global range
    /// as [`HnswPrecision::Sq8Fixed`] at 2^-7 sub-step precision.
    Sq16Fixed = 5,
}

/// Per-query SQ distance state (see [`Codec::sq8_query_state`]).
pub enum Sq8QueryState {
    /// The pairwise form: the query is quantized ONCE into code space
    /// (`qhat`), and every candidate distance is then pure integer
    /// arithmetic: `d = SUM (qhat - code)^2` (scale-weighted for the
    /// calibrated `f8` layout, unweighted — and therefore exact — for the
    /// fixed-range `sq8`/`sq16` layouts).
    Pairwise(Vec<i16>),
}

impl HnswPrecision {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => HnswPrecision::Plain,
            1 => HnswPrecision::IeeeFp16,
            2 => HnswPrecision::Sq8,
            3 => HnswPrecision::IeeeFp8,
            4 => HnswPrecision::Sq8Fixed,
            5 => HnswPrecision::Sq16Fixed,
            _ => pgrx::error!("Invalid hnswsq precision: {}", value),
        }
    }

    /// Parse a `storage_layout` reloption value.  Accepts the canonical names
    /// plus the `f16` alias for `ieeefp16`, case-insensitively.
    pub fn parse(value: &str) -> Self {
        match value.to_lowercase().as_str() {
            "plain" => HnswPrecision::Plain,
            "ieeefp16" | "f16" => HnswPrecision::IeeeFp16,
            "ieeefp8" => HnswPrecision::IeeeFp8,
            "f8" => HnswPrecision::Sq8,
            "sq8" => HnswPrecision::Sq8Fixed,
            "sq16" => HnswPrecision::Sq16Fixed,
            _ => pgrx::error!(
                "Invalid storage_layout '{}'. Must be one of 'plain', 'ieeefp16' (f16), \
                 'ieeefp8', 'f8', 'sq8', or 'sq16'",
                value
            ),
        }
    }

    /// Canonical reloption spelling.
    pub fn as_str(&self) -> &'static str {
        match self {
            HnswPrecision::Plain => "plain",
            HnswPrecision::IeeeFp16 => "ieeefp16",
            HnswPrecision::IeeeFp8 => "ieeefp8",
            HnswPrecision::Sq8 => "f8",
            HnswPrecision::Sq8Fixed => "sq8",
            HnswPrecision::Sq16Fixed => "sq16",
        }
    }

    /// Encoded bytes per dimension.
    pub fn elem_bytes(&self) -> usize {
        match self {
            HnswPrecision::Plain => 4,
            HnswPrecision::IeeeFp16 => 2,
            HnswPrecision::Sq8 => 1,
            HnswPrecision::IeeeFp8 => 1,
            HnswPrecision::Sq8Fixed => 1,
            HnswPrecision::Sq16Fixed => 2,
        }
    }

    /// Whether the layout needs a build-time calibration artifact.  Only the
    /// calibrated `f8` does; the IEEE layouts and the fixed-range `sq8`/`sq16`
    /// are stateless (incremental-build friendly).
    pub fn needs_calibration(&self) -> bool {
        matches!(self, HnswPrecision::Sq8)
    }

    /// Whether the layout uses the per-query integer pairwise distance
    /// (`sq8_query_state` + `encoded_distance`).
    pub fn is_sq(&self) -> bool {
        matches!(
            self,
            HnswPrecision::Sq8 | HnswPrecision::Sq8Fixed | HnswPrecision::Sq16Fixed
        )
    }
}

/// Per-dimension min/max calibration for the SQ8 layout, stored as a chained
/// item referenced from the meta page.
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct Sq8Calibration {
    /// Number of dimensions calibrated.
    pub dim: u32,
    /// True when this is the placeholder `[-1, 1]` range written for an
    /// empty-start index (cleared by a REINDEX that trains on real data).
    pub provisional: bool,
    /// Per-dimension minimum.
    pub mins: Vec<f32>,
    /// Per-dimension maximum.
    pub maxs: Vec<f32>,
}

impl Sq8Calibration {
    /// Train per-dimension min/max from a sample of (already cosine-normalized
    /// when applicable) vectors.  Empty sample → provisional range.
    pub fn train(samples: &[Vec<f32>], dim: usize) -> Self {
        if samples.is_empty() {
            return Self::provisional(dim);
        }
        let mut mins = vec![f32::MAX; dim];
        let mut maxs = vec![f32::MIN; dim];
        for s in samples {
            for (d, &x) in s.iter().take(dim).enumerate() {
                if x < mins[d] {
                    mins[d] = x;
                }
                if x > maxs[d] {
                    maxs[d] = x;
                }
            }
        }
        Self {
            dim: dim as u32,
            provisional: false,
            mins,
            maxs,
        }
    }

    /// Placeholder calibration for an empty-start SQ8 index: `[-1, 1]` per
    /// dimension (cosine-normalized embeddings land in this range naturally).
    pub fn provisional(dim: usize) -> Self {
        Self {
            dim: dim as u32,
            provisional: true,
            mins: vec![-1.0; dim],
            maxs: vec![1.0; dim],
        }
    }

    /// Store as a chained item, returning the pointer to its first chunk.
    pub unsafe fn store(&self, index: &PgRelation) -> ItemPointer {
        let mut stats = crate::access_method::stats::WriteStats::default();
        let mut tape = ChainTapeWriter::new(index, PageType::HnswCalibration, &mut stats);
        let bytes = self.serialize_to_vec();
        tape.write(&bytes)
    }

    /// Load from a chained-item pointer.
    pub fn load(index: &PgRelation, pointer: ItemPointer) -> Sq8Calibration {
        let mut stats = crate::access_method::stats::WriteStats::default();
        let mut tape = ChainItemReader::new(index, PageType::HnswCalibration, &mut stats);

        let mut buf: Vec<u8> = Vec::new();
        for item in tape.read(pointer) {
            buf.extend_from_slice(item.get_data_slice());
        }
        rkyv::from_bytes::<Sq8Calibration>(&buf).unwrap_or_else(|e| {
            panic!(
                "hnswsq: calibration parse failed ({} bytes): {:?}",
                buf.len(),
                e
            )
        })
    }
}

/// Encode/decode dispatcher for one index's precision layout.
///
/// For SQ8 the per-dimension scale factors are precomputed once (per insert /
/// per scan state) so the hot loop is a multiply + round + clamp.
#[derive(Clone, Debug)]
pub struct Codec {
    precision: HnswPrecision,
    dim: usize,
    /// SQ8 only: per-dimension minimum.
    sq8_mins: Vec<f32>,
    /// SQ8 only: per-dimension maximum.
    sq8_maxs: Vec<f32>,
    /// SQ8 only: per-dimension `255 / (max - min)`; 0.0 marks a degenerate
    /// (constant) dimension, which encodes as 0 and decodes to `min`.
    sq8_inv_scales: Vec<f32>,
    /// SQ8 only: per-dimension `(max - min) / 255`.
    sq8_scales: Vec<f32>,
    /// SQ8 only: `sq8_scales` squared — the per-dimension weights of the
    /// weighted pairwise distance.
    sq8_scales2: Vec<f32>,
}

impl Codec {
    /// Codec for a training-free layout (`plain`, `ieeefp16`, `ieeefp8`).
    pub fn new(precision: HnswPrecision, dim: usize) -> Self {
        assert!(
            !precision.needs_calibration(),
            "SQ8 codec must be constructed from its calibration"
        );
        Self {
            precision,
            dim,
            sq8_mins: Vec::new(),
            sq8_maxs: Vec::new(),
            sq8_inv_scales: Vec::new(),
            sq8_scales: Vec::new(),
            sq8_scales2: Vec::new(),
        }
    }

    /// Codec for the SQ8 layout from its calibration.
    pub fn new_sq8(calib: &Sq8Calibration) -> Self {
        let dim = calib.dim as usize;
        let mut inv_scales = vec![0.0f32; dim];
        let mut scales = vec![0.0f32; dim];
        for d in 0..dim {
            let range = calib.maxs[d] - calib.mins[d];
            if range > f32::EPSILON {
                inv_scales[d] = 255.0 / range;
                scales[d] = range / 255.0;
            }
            // else: degenerate constant dimension → inv_scale 0 → q = 0 →
            // decode to mins[d].
        }

        let scales2: Vec<f32> = scales.iter().map(|s| s * s).collect();
        Self {
            precision: HnswPrecision::Sq8,
            dim,
            sq8_mins: calib.mins.clone(),
            sq8_maxs: calib.maxs.clone(),
            sq8_inv_scales: inv_scales,
            sq8_scales: scales,
            sq8_scales2: scales2,
        }
    }

    pub fn precision(&self) -> HnswPrecision {
        self.precision
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Encoded length of one vector.
    pub fn vector_bytes(&self) -> usize {
        self.dim * self.precision.elem_bytes()
    }

    /// Encode `v` (exactly `dim` values) into a fresh byte vector.
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.vector_bytes());
        self.encode_into(v, &mut out);
        out
    }

    /// Encode `v` into `out` (cleared first; exact length guaranteed).
    /// Returns true when any component was clamped to the layout's
    /// representable range (the distance lower bound is then unknowable).
    pub fn encode_into(&self, v: &[f32], out: &mut Vec<u8>) -> bool {
        debug_assert_eq!(v.len(), self.dim);
        out.clear();
        let mut clamped = false;
        match self.precision {
            HnswPrecision::Plain => {
                for &x in v.iter().take(self.dim) {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            HnswPrecision::IeeeFp16 => {
                for &x in v.iter().take(self.dim) {
                    let x = sanitize(x);
                    if x.abs() > FP16_MAX {
                        clamped = true;
                    }
                    let x = x.clamp(-FP16_MAX, FP16_MAX);
                    out.extend_from_slice(&f32_to_f16(x).to_le_bytes());
                }
            }
            HnswPrecision::IeeeFp8 => {
                for &x in v.iter().take(self.dim) {
                    let x = sanitize(x);
                    if x.abs() > FP8_E4M3_MAX {
                        clamped = true;
                    }
                    out.push(f32_to_e4m3(x.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)));
                }
            }
            HnswPrecision::Sq8 => {
                for (d, &x) in v.iter().take(self.dim).enumerate() {
                    let x = sanitize(x);
                    if x < self.sq8_mins[d] || x > self.sq8_maxs[d] {
                        clamped = true;
                    }
                    let q = if self.sq8_inv_scales[d] == 0.0 {
                        0.0
                    } else {
                        ((x - self.sq8_mins[d]) * self.sq8_inv_scales[d]).round()
                    };
                    out.push(q.clamp(0.0, 255.0) as u8);
                }
            }
            HnswPrecision::Sq8Fixed => {
                for &x in v.iter().take(self.dim) {
                    let x = sanitize(x);
                    if !(SQ_FIXED_MIN..=SQ_FIXED_MAX).contains(&x) {
                        clamped = true;
                    }
                    // Truncate toward zero (the native float->int cast; no
                    // rounding instruction): byte-valued data is exact
                    // either way, and truncation is the cheapest correct
                    // quantization for this fixed range.
                    out.push(x.clamp(SQ_FIXED_MIN, SQ_FIXED_MAX) as u8);
                }
            }
            HnswPrecision::Sq16Fixed => {
                for &x in v.iter().take(self.dim) {
                    let x = sanitize(x);
                    if !(SQ_FIXED_MIN..=SQ_FIXED_MAX).contains(&x) {
                        clamped = true;
                    }
                    let q = (x.clamp(SQ_FIXED_MIN, SQ_FIXED_MAX) * 128.0) as u16;
                    out.extend_from_slice(&q.to_le_bytes());
                }
            }
        }
        debug_assert_eq!(out.len(), self.vector_bytes());
        clamped
    }

    /// Decode `bytes` (exactly `vector_bytes()` long) into `out` (exactly
    /// `dim` long).
    pub fn decode_into(&self, bytes: &[u8], out: &mut [f32]) {
        debug_assert_eq!(bytes.len(), self.vector_bytes());
        debug_assert_eq!(out.len(), self.dim);
        match self.precision {
            HnswPrecision::Plain => {
                for (d, chunk) in bytes.chunks_exact(4).enumerate() {
                    out[d] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            }
            HnswPrecision::IeeeFp16 => {
                for (d, chunk) in bytes.chunks_exact(2).enumerate() {
                    out[d] = crate::access_method::distance::f16_to_f32(u16::from_le_bytes([
                        chunk[0], chunk[1],
                    ]));
                }
            }
            HnswPrecision::IeeeFp8 => {
                for (d, &b) in bytes.iter().enumerate() {
                    out[d] = e4m3_to_f32(b);
                }
            }
            HnswPrecision::Sq8 => {
                for (d, &q) in bytes.iter().enumerate() {
                    out[d] = self.sq8_mins[d] + q as f32 * self.sq8_scales[d];
                }
            }
            HnswPrecision::Sq8Fixed => {
                for (d, &q) in bytes.iter().enumerate() {
                    out[d] = q as f32; // scale 1.0
                }
            }
            HnswPrecision::Sq16Fixed => {
                for (d, chunk) in bytes.chunks_exact(2).enumerate() {
                    out[d] = u16::from_le_bytes([chunk[0], chunk[1]]) as f32 * SQ16_FIXED_SCALE;
                }
            }
        }
    }

    /// Decode into a fresh vector.
    pub fn decode(&self, bytes: &[u8]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim];
        self.decode_into(bytes, &mut out);
        out
    }

    // -----------------------------------------------------------------------
    // SQ integer distances (the smoke-gun winners; see smoke.rs)
    // -----------------------------------------------------------------------

    /// Build the per-query state for the integer pairwise distance: the query
    /// is quantized ONCE into code space; every candidate distance is then
    /// pure integer arithmetic.  Applies to every SQ layout (`f8`, `sq8`,
    /// `sq16`).
    pub fn sq8_query_state(&self, q: &[f32], _pairwise: bool) -> Sq8QueryState {
        debug_assert_eq!(q.len(), self.dim);
        let mut qhat = Vec::with_capacity(self.dim);
        match self.precision {
            HnswPrecision::Sq8 => {
                for i in 0..self.dim {
                    let s = (q[i] - self.sq8_mins[i]) * self.sq8_inv_scales[i];
                    qhat.push(s.round().clamp(0.0, 255.0) as i16);
                }
            }
            HnswPrecision::Sq8Fixed => {
                for &x in q.iter().take(self.dim) {
                    qhat.push(x.clamp(SQ_FIXED_MIN, SQ_FIXED_MAX) as i16);
                }
            }
            HnswPrecision::Sq16Fixed => {
                for &x in q.iter().take(self.dim) {
                    qhat.push((x.clamp(SQ_FIXED_MIN, SQ_FIXED_MAX) * 128.0) as i16);
                }
            }
            _ => unreachable!("integer query state only for SQ layouts"),
        }
        Sq8QueryState::Pairwise(qhat)
    }

    /// The pairwise form, scale-weighted: `SUM scale_i^2 (qhat - code)^2`.
    /// `scale_i^2 * (qhat - code)^2` is the squared difference of the DECODED
    /// values, so this reproduces the scalar decode distance's ranking while
    /// keeping the query-quantized integer form (Lance's stored-norm dot and
    /// Milvus's per-dimension LUTs preserve the same scale weights; the
    /// unweighted variant measured ~4 pts recall loss on uneven-scale data).
    /// Dispatches to the vectorized kernels where available.
    #[inline]
    pub fn distance_l2_sq8_pairwise(&self, qhat: &[i16], bytes: &[u8]) -> f32 {
        debug_assert_eq!(qhat.len(), self.dim);
        debug_assert_eq!(bytes.len(), self.dim);
        crate::access_method::distance::distance_l2_sq8_pairwise(
            qhat,
            bytes,
            &self.sq8_scales2,
        )
    }

    /// Fixed-range `sq8` pairwise: `SUM (qhat - code)^2` over the unsigned
    /// byte codes, i32 accumulation (max per-lane sum ~1.04e9 at the
    /// 16000-dim limit, no overflow).  The scale is 1.0, so for in-range
    /// (byte-valued) vectors this equals the decoded-domain L2 of the
    /// quantized query exactly on the stored side (no per-dimension weights,
    /// no calibration); the query carries only the standard sub-step
    /// truncation error.
    #[inline]
    pub fn distance_l2_sq8_fixed_pairwise(&self, qhat: &[i16], bytes: &[u8]) -> f32 {
        debug_assert_eq!(qhat.len(), self.dim);
        debug_assert_eq!(bytes.len(), self.dim);
        crate::access_method::distance::distance_l2_sq8_fixed_pairwise(qhat, bytes)
    }

    /// Fixed-range `sq16` pairwise: `SUM (qhat - code)^2` over the u16
    /// codes, i64 accumulation.  Equals the decoded-domain L2 of the
    /// quantized query times `128^2`.
    #[inline]
    pub fn distance_l2_sq16_fixed_pairwise(&self, qhat: &[i16], bytes: &[u8]) -> f32 {
        debug_assert_eq!(qhat.len(), self.dim);
        debug_assert_eq!(bytes.len(), self.dim * 2);
        crate::access_method::distance::distance_l2_sq16_fixed_pairwise(qhat, bytes)
    }



    /// Compute `distance(query, bytes)` directly over the encoded byte
    /// representation, without the decode-into-scratch copy.  Matches the
    /// semantics of the [`crate::access_method::distance`] kernels exactly:
    /// L2 is the sum of squared differences (no sqrt — ordering only), cosine is
    /// `1 − Σ q·v` clamped at zero (inputs are normalized) and inner product is
    /// the NEGATED dot (`<#>` ordering).  This is the hottest routine of index
    /// builds and scans.
    ///
    /// Every branch is a counted `for i in 0..dim` loop: the previous
    /// `chunks_exact(..).map(..)`/`enumerate()` iterator chains blocked
    /// vectorization, and profiling a dim-128 build put ~66% of samples in those
    /// adapters against 3.6% in the SIMD kernel they fed.  For the lossless
    /// `plain` layout the stored bytes ARE little-endian IEEE f32, so when the
    /// slice is 4-byte aligned (page items always are; `Vec<u8>` copies are not
    /// guaranteed to be) the SIMD kernels consume them with no decode at all.
    #[inline]
    pub fn distance_encoded_direct(
        &self,
        dist_type: crate::access_method::distance::DistanceType,
        query: &[f32],
        bytes: &[u8],
    ) -> f32 {
        use crate::access_method::distance as kernels;
        use crate::access_method::distance::DistanceType;
        debug_assert_eq!(query.len(), self.dim);
        debug_assert_eq!(bytes.len(), self.vector_bytes());

        /// Apply the operator conventions shared by every layout: cosine is
        /// `1 − dot` clamped at zero, inner product is the negated dot.
        #[inline(always)]
        fn finish(acc: f32, dist_type: DistanceType) -> f32 {
            match dist_type {
                DistanceType::Cosine => (1.0 - acc).max(0.0),
                DistanceType::InnerProduct => -acc,
                _ => acc,
            }
        }

        let dim = self.dim;

        if self.precision == HnswPrecision::Plain
            && (bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>())
        {
            // SAFETY: the slice is `dim * 4` bytes (debug-asserted) and 4-byte
            // aligned; every bit pattern is a valid `f32`.
            let v: &[f32] =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), dim) };
            return match dist_type {
                DistanceType::L2 => kernels::distance_l2(query, v),
                DistanceType::Cosine => kernels::distance_cosine(query, v),
                DistanceType::InnerProduct => kernels::distance_inner_product(query, v),
                _ => kernels::distance_l2(query, v),
            };
        }

        let mut acc = 0.0f32;
        match self.precision {
            HnswPrecision::Plain => {
                // Unaligned copy (e.g. a `Vec<u8>` node buffer): unaligned loads
                // in a counted loop, which vectorizes the same way.
                let ptr = bytes.as_ptr().cast::<f32>();
                match dist_type {
                    DistanceType::L2 => {
                        for i in 0..dim {
                            // SAFETY: `i < dim` and the slice is `dim * 4` long.
                            let x = unsafe { std::ptr::read_unaligned(ptr.add(i)) };
                            let d = query[i] - x;
                            acc += d * d;
                        }
                    }
                    _ => {
                        for i in 0..dim {
                            // SAFETY: as above.
                            let x = unsafe { std::ptr::read_unaligned(ptr.add(i)) };
                            acc += query[i] * x;
                        }
                    }
                }
            }
            HnswPrecision::IeeeFp16 => {
                // The stored pattern is order-preserving: convert with bit
                // moves (vectorized in the kernels), not per-element IEEE
                // decode.  Cosine/IP share the dot-product kernel.
                return match dist_type {
                    DistanceType::L2 => kernels::distance_l2_f16(query, bytes),
                    _ => finish(kernels::distance_inner_product_f16(query, bytes), dist_type),
                };
            }
            HnswPrecision::IeeeFp8 => match dist_type {
                DistanceType::L2 => {
                    for i in 0..dim {
                        let d = query[i] - e4m3_to_f32(bytes[i]);
                        acc += d * d;
                    }
                }
                _ => {
                    for i in 0..dim {
                        acc += query[i] * e4m3_to_f32(bytes[i]);
                    }
                }
            },
            HnswPrecision::Sq8 => match dist_type {
                DistanceType::L2 => {
                    for i in 0..dim {
                        let x = self.sq8_mins[i] + bytes[i] as f32 * self.sq8_scales[i];
                        let d = query[i] - x;
                        acc += d * d;
                    }
                }
                _ => {
                    for i in 0..dim {
                        let x = self.sq8_mins[i] + bytes[i] as f32 * self.sq8_scales[i];
                        acc += query[i] * x;
                    }
                }
            },
            HnswPrecision::Sq8Fixed => match dist_type {
                DistanceType::L2 => {
                    return crate::access_method::distance::distance_l2_sq8_fixed_decode(
                        query, bytes,
                    );
                }
                _ => {
                    for i in 0..dim {
                        acc += query[i] * bytes[i] as f32;
                    }
                }
            },
            HnswPrecision::Sq16Fixed => match dist_type {
                DistanceType::L2 => {
                    return crate::access_method::distance::distance_l2_sq16_fixed_decode(
                        query, bytes,
                    );
                }
                _ => {
                    for i in 0..dim {
                        let x = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]) as f32
                            * SQ16_FIXED_SCALE;
                        acc += query[i] * x;
                    }
                }
            },
        }
        finish(acc, dist_type)
    }

    /// SQ layouts only: the L2 norm of one vector's worst-case quantization
    /// error `‖v̂ − v‖`, which the scan turns into provable distance lower
    /// bounds.  For round-to-nearest (the calibrated `f8`) the per-dimension
    /// error is half a step, so the norm is `√(Σ scale_d²) / 2`; for the
    /// fixed-range layouts the encode TRUNCATES (one full step per dim), so
    /// the norm is `scale · √dim`.  Zero for the other layouts.
    pub fn quant_error_norm(&self) -> f32 {
        match self.precision {
            HnswPrecision::Sq8 => {
                self.sq8_scales.iter().map(|s| s * s).sum::<f32>().sqrt() / 2.0
            }
            HnswPrecision::Sq8Fixed => SQ8_FIXED_SCALE * (self.dim as f32).sqrt(),
            HnswPrecision::Sq16Fixed => SQ16_FIXED_SCALE * (self.dim as f32).sqrt(),
            _ => 0.0,
        }
    }
}

/// Maximum relative per-element error of the IEEE layouts (round-to-nearest):
/// binary16 has 10 explicit mantissa bits (half-ulp 2^-11), E4M3 has 3
/// (half-ulp 2^-4).  Zero for plain/SQ layouts (SQ error is absolute; see
/// [`Codec::quant_error_norm`]).
pub fn relative_element_error(precision: HnswPrecision) -> f32 {
    match precision {
        HnswPrecision::IeeeFp16 => 2.0f32.powi(-11),
        HnswPrecision::IeeeFp8 => 2.0f32.powi(-4),
        _ => 0.0,
    }
}

/// pgvector's `vector` type rejects NaN/±Inf inputs, but guard anyway so a
/// corrupted datum can never poison stored codes or distances.
#[inline]
fn sanitize(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// OCP FP8 E4M3 (S.1111.111 = NaN, no infinities, max finite 448, min normal
// 2^-6, subnormal step 2^-9).  Conversions are round-to-nearest-even; the
// encoder expects pre-clamped finite input (|x| <= 448).
// ---------------------------------------------------------------------------

/// Round `t` (a non-negative exact quotient, t <= 16) to nearest even integer.
#[inline]
fn round_half_even(t: f32) -> u32 {
    let f = t.floor();
    let r = t - f;
    let up = r > 0.5 || (r == 0.5 && (f as u32) % 2 == 1);
    (f + if up { 1.0 } else { 0.0 }) as u32
}

/// Encode a finite, pre-clamped (|x| <= 65504) f32 as a binary16.  The f16
/// pattern is order-preserving, so the normal case is bit moves (round the
/// 13 dropped fraction bits to nearest-even; a mantissa carry propagates
/// into the exponent field through bit 10).  Values in the f16 subnormal
/// range take the exact slow path.
#[inline]
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x7FFFFF;
    if !(113..=142).contains(&exp) {
        // f16 subnormal range (|x| < 2^-14) or zero — exact slow path
        return half::f16::from_f32(x).to_bits();
    }
    let kept = (mant >> 13) & 0x3FF;
    let rest = mant & 0x1FFF;
    let round_up = (rest > 0x1000) || (rest == 0x1000 && (kept & 1) == 1);
    let m = kept + round_up as u32; // 0..=0x400; the carry hits exponent bit 10
    sign | (((exp - 112) as u16) << 10) | (m as u16)
}

/// Encode a finite, pre-clamped (|x| <= 448) f32 as an E4M3 byte.  The E4M3
/// pattern is order-preserving, so the encode is bit moves on the f32 fields
/// (no log2/powi); subnormals take the slow path.
pub fn f32_to_e4m3(x: f32) -> u8 {
    if x == 0.0 {
        // Preserve -0.0 (harmless; keeps the sign bit semantics of the format).
        return if x.is_sign_negative() { 0x80 } else { 0x00 };
    }
    let bits = x.to_bits();
    let sign = ((bits >> 24) & 0x80) as u8;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    let e = exp - 127;

    if e < -6 {
        // Subnormal: value = q * 2^-9, q in 0..=8 (q = 8 is the minimal
        // normal 2^-6, whose encoding 0b0000_1000 is contiguous).
        let a = x.abs();
        let q = round_half_even(a * 512.0);
        return sign | (q.min(8) as u8);
    }

    // Normal: value = (1 + m/8) * 2^e; the E4M3 exponent field is e + 7 and
    // the mantissa keeps 3 of the 23 fraction bits (round-to-nearest-even).
    let m3_top = ((mant >> 20) & 0x7) as i32;
    let rest = mant & 0xFFFFF;
    let round_up = (rest > 0x80000) || (rest == 0x80000 && (m3_top & 1) == 1);
    let mut m3 = m3_top + round_up as i32; // 0..=8
    let mut ef = e + 7; // 1..=15
    if m3 == 8 {
        // Rounded up into the next binade: 1 + 8/8 = 2
        m3 = 0;
        ef += 1;
    }
    if ef > 15 || (ef == 15 && m3 >= 7) {
        // Saturate to max finite 0x7E (the caller pre-clamps; only reachable
        // via rounding at the top of the range).
        return sign | 0x7E;
    }
    sign | ((ef as u8) << 3) | (m3 as u8)
}

/// Decode an E4M3 byte to f32.  The NaN encoding (0x7F/0xFF) decodes to 0.0
/// defensively: our encoder never produces it (inputs are sanitized finite),
/// and a finite fallback keeps distances well-defined on corrupted data.
/// E4M3 (OCP FP8) -> f32.  The E4M3 pattern is order-preserving too, so the
/// normal case is bit moves (sign / exp+bias / mantissa fields); subnormals
/// and the NaN encoding take the exact slow path.
/// Defensively, the NaN encoding returns 0.0: our encoder never produces it
/// (inputs are sanitized), and a finite fallback keeps distances
/// well-defined on corrupted data.
#[inline]
pub fn e4m3_to_f32(b: u8) -> f32 {
    let bits = b as u32;
    let sign = (bits & 0x80) << 24;
    let exp = (bits >> 3) & 0xF;
    let man = bits & 0x7;
    if exp == 0 || (exp == 0xF && man == 7) {
        // subnormal (0x00/0x80 -> ±0) or NaN encoding -> defensive 0
        return if exp == 0 {
            let v = man as f32 * 2.0f32.powi(-9);
            if sign == 0 {
                v
            } else {
                -v
            }
        } else {
            0.0
        };
    }
    // normal: (1 + man/8) * 2^(exp - 7) -> f32 exp field = exp + 120
    f32::from_bits(sign | ((exp + 120) << 23) | (man << 20))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn sample_vector(dim: usize, seed: usize) -> Vec<f32> {
        (0..dim)
            .map(|d| ((seed * 31 + d * 17) % 2000) as f32 / 1000.0 - 1.0)
            .collect()
    }

    /// The decode-free distance path must agree with a plain scalar reference
    /// for every layout and distance type, and the `plain` fast path (SIMD
    /// kernels over the stored bytes) must agree with the unaligned fallback
    /// (counted loop) — they differ only in summation order.
    #[test]
    fn test_distance_encoded_direct_matches_reference() {
        use crate::access_method::distance::DistanceType;
        let dim = 40usize;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        let q: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

        let reference = |dt: DistanceType| -> f32 {
            match dt {
                DistanceType::L2 => v
                    .iter()
                    .zip(q.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>(),
                DistanceType::Cosine => {
                    (1.0 - v.iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f32>()).max(0.0)
                }
                DistanceType::InnerProduct => {
                    -v.iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f32>()
                }
            }
        };

        for precision in [
            HnswPrecision::Plain,
            HnswPrecision::IeeeFp16,
            HnswPrecision::IeeeFp8,
            HnswPrecision::Sq8,
            HnswPrecision::Sq8Fixed,
            HnswPrecision::Sq16Fixed,
        ] {
            let codec = match precision {
                HnswPrecision::Sq8 => {
                    let sample: Vec<Vec<f32>> = (0..64)
                        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
                        .collect();
                    Codec::new_sq8(&Sq8Calibration::train(&sample, dim))
                }
                _ => Codec::new(precision, dim),
            };
            let enc = codec.encode(&v);
            let decoded: Vec<f32> = codec.decode(&enc);
            for dt in [
                DistanceType::L2,
                DistanceType::Cosine,
                DistanceType::InnerProduct,
            ] {
                let got = codec.distance_encoded_direct(dt, &q, &enc);
                // Reference over the *decoded* vector: the codec may have
                // changed it (quantization), and the distance is defined on the
                // stored representation.
                let want = match dt {
                    DistanceType::L2 => decoded
                        .iter()
                        .zip(q.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>(),
                    DistanceType::Cosine => (1.0
                        - decoded.iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f32>())
                    .max(0.0),
                    DistanceType::InnerProduct => {
                        -decoded.iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f32>()
                    }
                };
                let tol = 1e-5 * (1.0 + want.abs());
                assert!(
                    (got - want).abs() <= tol,
                    "{:?} {:?}: got {} want {} (decoded {:?})",
                    precision,
                    dt,
                    got,
                    want,
                    &decoded[..3]
                );
            }
        }

        // plain: the aligned SIMD path and the unaligned counted loop agree.
        let codec = Codec::new(HnswPrecision::Plain, dim);
        let mut enc = codec.encode(&v);
        for dt in [
            DistanceType::L2,
            DistanceType::Cosine,
            DistanceType::InnerProduct,
        ] {
            let aligned = codec.distance_encoded_direct(dt, &q, &enc);
            // Force the fallback: shift the slice so it cannot be 4-byte aligned.
            let mut shifted = vec![0u8; enc.len() + 1];
            shifted[1..].copy_from_slice(&enc);
            assert_ne!(shifted[1..].as_ptr() as usize % 4, 0);
            let unaligned = codec.distance_encoded_direct(dt, &q, &shifted[1..]);
            assert!(
                (aligned - unaligned).abs() <= 1e-5 * (1.0 + aligned.abs()),
                "{:?}: aligned {} vs unaligned {}",
                dt,
                aligned,
                unaligned
            );
        }
        enc.clear();
        let _ = reference; // the closure documents the reference formula
    }

    #[test]
    fn test_precision_parse_aliases() {
        assert_eq!(HnswPrecision::parse("plain"), HnswPrecision::Plain);
        assert_eq!(HnswPrecision::parse("ieeefp16"), HnswPrecision::IeeeFp16);
        assert_eq!(HnswPrecision::parse("F16"), HnswPrecision::IeeeFp16);
        assert_eq!(HnswPrecision::parse("ieeefp8"), HnswPrecision::IeeeFp8);
        assert_eq!(HnswPrecision::parse("f8"), HnswPrecision::Sq8);
        assert_eq!(HnswPrecision::parse("sq8"), HnswPrecision::Sq8Fixed);
        assert_eq!(HnswPrecision::parse("SQ8"), HnswPrecision::Sq8Fixed);
        assert_eq!(HnswPrecision::parse("sq16"), HnswPrecision::Sq16Fixed);
        assert_eq!(HnswPrecision::parse("Sq16"), HnswPrecision::Sq16Fixed);
    }

    #[test]
    fn test_elem_bytes_roundtrip_u8() {
        for p in [
            HnswPrecision::Plain,
            HnswPrecision::IeeeFp16,
            HnswPrecision::Sq8,
            HnswPrecision::IeeeFp8,
            HnswPrecision::Sq8Fixed,
            HnswPrecision::Sq16Fixed,
        ] {
            assert_eq!(HnswPrecision::from_u8(p as u8), p);
            assert!(p.elem_bytes() >= 1 && p.elem_bytes() <= 4);
        }
        // Storage-footprint expectations per layout.
        assert_eq!(HnswPrecision::Sq8Fixed.elem_bytes(), 1);
        assert_eq!(HnswPrecision::Sq16Fixed.elem_bytes(), 2);
        // The fixed layouts are training-free.
        assert!(!HnswPrecision::Sq8Fixed.needs_calibration());
        assert!(!HnswPrecision::Sq16Fixed.needs_calibration());
        assert!(HnswPrecision::Sq8.needs_calibration());
    }

    #[test]
    fn test_plain_roundtrip_exact() {
        let dim = 16;
        let codec = Codec::new(HnswPrecision::Plain, dim);
        let v = sample_vector(dim, 3);
        let enc = codec.encode(&v);
        assert_eq!(enc.len(), dim * 4);
        let dec = codec.decode(&enc);
        assert_eq!(dec, v);
    }

    #[test]
    fn test_f16_roundtrip_relative_error() {
        let dim = 64;
        let codec = Codec::new(HnswPrecision::IeeeFp16, dim);
        for seed in 0..50 {
            let v = sample_vector(dim, seed);
            let enc = codec.encode(&v);
            assert_eq!(enc.len(), dim * 2);
            let dec = codec.decode(&enc);
            for (a, b) in v.iter().zip(dec.iter()) {
                // binary16: 10 explicit mantissa bits → rel err ≤ 2^-11 (RTNE)
                let err = (a - b).abs();
                assert!(
                    err <= a.abs() * 2.0f32.powi(-11) + 1e-8,
                    "f16 error too large: {} vs {}",
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn test_fp8_saturation_and_error_bound() {
        let dim = 4;
        let codec = Codec::new(HnswPrecision::IeeeFp8, dim);
        // Out-of-range values clamp to ±448, never NaN/Inf.
        let v = vec![1000.0, -1000.0, 448.0, -448.0];
        let enc = codec.encode(&v);
        let dec = codec.decode(&enc);
        assert_eq!(dec[0], FP8_E4M3_MAX);
        assert_eq!(dec[1], -FP8_E4M3_MAX);
        assert_eq!(dec[2], FP8_E4M3_MAX);
        assert_eq!(dec[3], -FP8_E4M3_MAX);
        assert!(dec.iter().all(|x| x.is_finite()));

        // E4M3: 3 explicit mantissa bits → rel err ≤ 2^-4 within range.
        for seed in 0..50 {
            let v = sample_vector(dim, seed);
            let dec = codec.decode(&codec.encode(&v));
            for (a, b) in v.iter().zip(dec.iter()) {
                let err = (a - b).abs();
                assert!(
                    err <= a.abs() * 2.0f32.powi(-4) + 2.0f32.powi(-9),
                    "fp8 error too large: {} vs {}",
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn test_sq8_roundtrip_and_degenerate_dim() {
        let dim = 8;
        let samples: Vec<Vec<f32>> = (0..100).map(|s| sample_vector(dim, s)).collect();
        let calib = Sq8Calibration::train(&samples, dim);
        assert!(!calib.provisional);
        let codec = Codec::new_sq8(&calib);

        for s in 0..20 {
            let v = sample_vector(dim, s);
            let enc = codec.encode(&v);
            assert_eq!(enc.len(), dim);
            let dec = codec.decode(&enc);
            for (d, (a, b)) in v.iter().zip(dec.iter()).enumerate() {
                let range = calib.maxs[d] - calib.mins[d];
                // abs err ≤ half a quantization step + rounding
                assert!(
                    (a - b).abs() <= range / 255.0 / 2.0 + 1e-6,
                    "sq8 error too large at dim {}: {} vs {}",
                    d,
                    a,
                    b
                );
            }
        }

        // Constant dimension → degenerate scale guard: q=0, decode = min.
        let constant: Vec<Vec<f32>> = (0..10).map(|_| vec![0.5; dim]).collect();
        let calib2 = Sq8Calibration::train(&constant, dim);
        let codec2 = Codec::new_sq8(&calib2);
        let enc = codec2.encode(&vec![0.5; dim]);
        assert!(enc.iter().all(|&q| q == 0));
        let dec = codec2.decode(&enc);
        assert!(dec.iter().all(|&x| (x - 0.5).abs() < 1e-6));
        // Out-of-range values clamp into [0, 255].
        let enc_oob = codec2.encode(&vec![100.0, -100.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        assert!(enc_oob.iter().all(|&q| q == 0)); // degenerate dims → 0
    }

    #[test]
    fn test_sq8_clamping_to_range() {
        let dim = 2;
        let samples = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let calib = Sq8Calibration::train(&samples, dim);
        let codec = Codec::new_sq8(&calib);
        let enc = codec.encode(&vec![5.0, -5.0]);
        assert_eq!(enc[0], 255);
        assert_eq!(enc[1], 0);
        let dec = codec.decode(&enc);
        assert!((dec[0] - 1.0).abs() < 1e-6);
        assert!(dec[1].abs() < 1e-6);
    }

    #[test]
    fn test_provisional_calibration() {
        let calib = Sq8Calibration::provisional(4);
        assert!(calib.provisional);
        assert!(calib.mins.iter().all(|&m| m == -1.0));
        assert!(calib.maxs.iter().all(|&m| m == 1.0));
        let codec = Codec::new_sq8(&calib);
        // Within [-1, 1]: fine-grained codes.
        let enc = codec.encode(&vec![0.0, 1.0, -1.0, 0.5]);
        let dec = codec.decode(&enc);
        for (a, b) in [0.0f32, 1.0, -1.0, 0.5].iter().zip(dec.iter()) {
            assert!((a - b).abs() <= 2.0 / 255.0 / 2.0 + 1e-6);
        }
    }

    #[test]
    fn test_nan_guard() {
        let dim = 2;
        let codec = Codec::new(HnswPrecision::IeeeFp16, dim);
        let enc = codec.encode(&[f32::NAN, f32::INFINITY]);
        let dec = codec.decode(&enc);
        assert!(dec.iter().all(|x| x.is_finite()));
        let codec8 = Codec::new(HnswPrecision::IeeeFp8, dim);
        let dec8 = codec8.decode(&codec8.encode(&[f32::NAN, f32::NEG_INFINITY]));
        assert!(dec8.iter().all(|x| x.is_finite()));
        for p in [HnswPrecision::Sq8Fixed, HnswPrecision::Sq16Fixed] {
            let codec_sq = Codec::new(p, dim);
            let dec_sq = codec_sq.decode(&codec_sq.encode(&[f32::NAN, f32::NEG_INFINITY]));
            assert!(dec_sq.iter().all(|x| x.is_finite()), "{:?}", p);
        }
    }

    /// The fixed-range SQ layouts: training-free, global [0, 255] range, one
    /// global quantization step, and `clamped` flags on out-of-range values.
    /// Byte-valued vectors must roundtrip exactly for `sq8` (scale 1.0).
    #[test]
    fn test_fixed_sq_roundtrip_and_clamp() {
        let dim = 16;
        for (p, scale, max_code) in [
            (HnswPrecision::Sq8Fixed, SQ8_FIXED_SCALE, 1.0f32),
            (HnswPrecision::Sq16Fixed, SQ16_FIXED_SCALE, 128.0f32),
        ] {
            let codec = Codec::new(p, dim);
            // Training-free.
            assert!(!p.needs_calibration());
            // Byte-domain samples: in-range, fractional values.
            let byte_sample = |s: usize| -> Vec<f32> {
                (0..dim)
                    .map(|d| ((s * 31 + d * 17) % 2000) as f32 / 7.84)
                    .collect()
            };
            for s in 0..30 {
                let v = byte_sample(s);
                let mut enc = Vec::new();
                let clamped = codec.encode_into(&v, &mut enc);
                assert_eq!(enc.len(), dim * p.elem_bytes());
                assert!(!clamped, "sample data lies within [0, 255]");
                let dec = codec.decode(&enc);
                for (a, b) in v.iter().zip(dec.iter()) {
                    // truncation: error in [0, 1) per dimension
                    assert!(
                        (a - b).abs() < scale + 1e-9,
                        "{:?}: error too large: {} vs {}",
                        p,
                        a,
                        b
                    );
                }
            }
            // Integer byte values encode EXACTLY for both layouts.
            let ints: Vec<f32> = (0..dim).map(|d| (d * 3) as f32).collect();
            let dec_ints = codec.decode(&codec.encode(&ints));
            for (a, b) in ints.iter().zip(dec_ints.iter()) {
                assert_eq!(a, b, "{:?}: integer byte value must roundtrip", p);
            }
            // Out-of-range: clamps to the range endpoints and flags it.
            let codec2 = Codec::new(p, 2);
            let mut enc = Vec::new();
            let clamped = codec2.encode_into(&vec![500.0, -500.0], &mut enc);
            assert!(clamped);
            let dec = codec2.decode(&enc);
            assert!((dec[0] - 255.0).abs() < 1e-4, "{:?}", p);
            assert!(dec[1].abs() < 1e-4, "{:?}", p);
            // The query state quantizes identically to encode.
            let q = vec![64.25, 128.75];
            let Sq8QueryState::Pairwise(qhat) = codec2.sq8_query_state(&q, true);
            assert_eq!(qhat.len(), 2);
            for (&qh, &x) in qhat.iter().zip(q.iter()) {
                let want = (x * max_code) as i16; // truncating cast, like encode
                assert_eq!(qh, want, "{:?}", p);
            }
        }
    }

    /// THE fixed-range property: the code-pairwise distance equals the
    /// decoded-domain L2 distance of the quantized query times a global
    /// constant (the code side is exact — no per-dimension weights, no
    /// calibration; the query side carries the standard sub-step
    /// truncation error).
    #[test]
    fn test_fixed_sq_pairwise_is_exact_decoded_l2() {
        use crate::access_method::distance::DistanceType;
        let dim = 41; // odd tail exercises the scalar remainder paths
        let mut rng = rand::rngs::SmallRng::seed_from_u64(99);
        for (p, scale, k2) in [
            (HnswPrecision::Sq8Fixed, SQ8_FIXED_SCALE, 1.0f32),
            (HnswPrecision::Sq16Fixed, SQ16_FIXED_SCALE, 128.0f32 * 128.0f32),
        ] {
            let codec = Codec::new(p, dim);
            for _ in 0..30 {
                // Byte-domain data (the layout's range).
                let q: Vec<f32> = (0..dim).map(|_| rng.gen_range(0.0f32..255.0)).collect();
                let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(0.0f32..255.0)).collect();
                let enc = codec.encode(&v);
                let Sq8QueryState::Pairwise(qhat) = codec.sq8_query_state(&q, true);
                let got = match p {
                    HnswPrecision::Sq8Fixed => {
                        codec.distance_l2_sq8_fixed_pairwise(&qhat, &enc)
                    }
                    HnswPrecision::Sq16Fixed => {
                        codec.distance_l2_sq16_fixed_pairwise(&qhat, &enc)
                    }
                    _ => unreachable!(),
                };
                // Reference: decoded-domain L2 of the QUANTIZED query —
                // `SUM (qhat*scale - code*scale)^2 = pairwise * scale^2`.
                let decoded: Vec<f32> = codec.decode(&enc);
                let want = qhat
                    .iter()
                    .zip(decoded.iter())
                    .map(|(&qh, &x)| {
                        let d = qh as f32 * scale - x;
                        d * d
                    })
                    .sum::<f32>();
                // Integer sums are exact; only the f32 reference accumulates.
                let tol = 1e-5 * (1.0 + want);
                assert!(
                    (got - want * k2).abs() <= tol * k2,
                    "{:?}: pairwise {} vs decoded(qhat)*K^2 {}",
                    p,
                    got,
                    want * k2
                );
                // And the raw-query decoded distance is within the query
                // quantization error: ‖q̂·scale − q‖ ≤ q_err, so
                // |d(q̂,v̂) − d(q,v̂)| ≤ q_err·(2‖q−v̂‖ + q_err).
                let decoded_raw = codec.distance_encoded_direct(DistanceType::L2, &q, &enc);
                let q_err = (dim as f32).sqrt() * scale;
                let bound = q_err * (2.0 * decoded_raw.sqrt() + q_err);
                assert!(
                    (got * scale * scale - decoded_raw).abs() <= bound * 1.001 + 1e-3,
                    "{:?}: pairwise {} vs raw decoded {}",
                    p,
                    got * scale * scale,
                    decoded_raw
                );
                // And both kernels (x86 SIMD + scalar fallback) agree.
                let scalar = match p {
                    HnswPrecision::Sq8Fixed => crate::access_method::distance::distance_l2_sq8_fixed_pairwise_scalar(&qhat, &enc),
                    HnswPrecision::Sq16Fixed => crate::access_method::distance::distance_l2_sq16_fixed_pairwise_scalar(&qhat, &enc),
                    _ => unreachable!(),
                };
                assert!(
                    (got - scalar).abs() <= 1e-3 * (1.0 + scalar.abs()),
                    "{:?}: x86 {} vs scalar {}",
                    p,
                    got,
                    scalar
                );
            }
        }
    }

    /// Exhaustive oracle: every finite E4M3 value, as (bits, f32).
    fn e4m3_finite_values() -> Vec<(u8, f32)> {
        (0u16..256)
            .map(|b| b as u8)
            .filter(|&b| b & 0x7F != 0x7F) // exclude NaN encodings
            .map(|b| (b, e4m3_to_f32(b)))
            .collect()
    }

    /// Round-to-nearest-even over the exhaustive value table (ties to even
    /// significand bits).
    fn oracle_encode(x: f32) -> u8 {
        let vals = e4m3_finite_values();
        let mut best = vals[0];
        let mut best_key = (f64::INFINITY, 1u32);
        for &(b, v) in &vals {
            let d = (x - v).abs() as f64;
            // tie-break: even mantissa LSB preferred
            let parity = (b & 1) as u32;
            let key = (d, parity);
            if key < best_key {
                best_key = key;
                best = (b, v);
            }
        }
        best.0
    }

    #[test]
    fn test_e4m3_decode_roundtrip_exhaustive() {
        for &(b, v) in &e4m3_finite_values() {
            // Re-encoding a decoded value must return the canonical bits
            // (including the sign bit: -0 decodes to -0.0 and re-encodes to
            // 0x80).
            let re = f32_to_e4m3(v.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX));
            assert_eq!(re, b, "bits {:02x} value {} re-encoded {:02x}", b, v, re);
        }
    }

    #[test]
    fn test_e4m3_encode_matches_oracle() {
        // Structured sweep across binades + tie midpoints + random values.
        let mut inputs: Vec<f32> = Vec::new();
        let mut e = -12i32;
        while e <= 9 {
            let base = 2.0f32.powi(e);
            for k in 0..64 {
                inputs.push(base * (1.0 + k as f32 / 64.0));
                inputs.push(-base * (1.0 + k as f32 / 64.0));
            }
            // exact midpoints between adjacent e4m3 values in this binade
            let step = 2.0f32.powi(e - 3);
            for m in 0..16 {
                let v = (8.0 + m as f32) * step;
                inputs.push(v + step / 2.0);
                inputs.push(v);
                inputs.push(v - step / 2.0);
            }
            e += 1;
        }
        inputs.extend([0.0, -0.0, 448.0, -448.0, 447.9, 2e-9, 1e-30, 1e-38]);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        for _ in 0..5000 {
            let x: f32 = rng.gen_range(-500.0f32..500.0);
            inputs.push(x);
        }
        for x in inputs {
            let xc = x.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX);
            let got = f32_to_e4m3(xc);
            let want = oracle_encode(xc);
            assert_eq!(
                got & 0x7F,
                want & 0x7F,
                "x={} got {:02x} ({}) want {:02x} ({})",
                x,
                got,
                e4m3_to_f32(got),
                want,
                e4m3_to_f32(want)
            );
        }
    }

    #[test]
    fn test_e4m3_special_encodings() {
        assert_eq!(f32_to_e4m3(0.0), 0x00);
        assert_eq!(f32_to_e4m3(1.0), 0x38); // exp=7 (2^0), man=0
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(f32_to_e4m3(448.0), 0x7E); // max finite
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert_eq!(e4m3_to_f32(0x7F), 0.0); // NaN encoding → defensive 0
        assert_eq!(f32_to_e4m3(2.0f32.powi(-9)), 0x01); // min subnormal
        assert_eq!(e4m3_to_f32(0x01), 2.0f32.powi(-9));
        assert_eq!(f32_to_e4m3(-3.0), 0x80 | 0x44); // -3 = (8+4)*2^-3+... verify by decode
        assert_eq!(e4m3_to_f32(f32_to_e4m3(-3.0)), -3.0);
    }

    /// Informational micro-benchmark: encode cost, round vs truncation
    /// (~140 vs ~103 ns per dim-128 vector on this machine).
    #[test]
    #[ignore]
    fn bench_encode_round_vs_trunc() {
        let dim = 128;
        let v: Vec<f32> = (0..dim).map(|i| i as f32 * 1.7 + 0.3).collect();
        let mut out = Vec::with_capacity(dim);
        let n = 500_000usize;

        // Round-to-nearest variant (the pre-truncation encode).
        for _ in 0..10_000 {
            out.clear();
            for &x in &v {
                let x = super::sanitize(x);
                out.push(x.clamp(0.0f32, 255.0f32).round() as u8);
            }
        }
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            out.clear();
            for &x in &v {
                let x = super::sanitize(x);
                out.push(x.clamp(0.0f32, 255.0f32).round() as u8);
            }
        }
        println!("round-based: {} ns/vec", t0.elapsed().as_nanos() / n as u128);

        // Truncation variant: clamp + native truncating cast, no round().
        for _ in 0..10_000 {
            out.clear();
            for &x in &v {
                let x = super::sanitize(x);
                out.push(x.clamp(0.0f32, 255.0f32) as u8);
            }
        }
        let t1 = std::time::Instant::now();
        for _ in 0..n {
            out.clear();
            for &x in &v {
                let x = super::sanitize(x);
                out.push(x.clamp(0.0f32, 255.0f32) as u8);
            }
        }
        println!("trunc-based: {} ns/vec", t1.elapsed().as_nanos() / n as u128);
    }
}
