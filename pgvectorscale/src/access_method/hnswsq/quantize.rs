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
}

impl HnswPrecision {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => HnswPrecision::Plain,
            1 => HnswPrecision::IeeeFp16,
            2 => HnswPrecision::Sq8,
            3 => HnswPrecision::IeeeFp8,
            _ => pgrx::error!("Invalid hnswsq precision: {}", value),
        }
    }

    /// Parse a `storage_layout` reloption value.  Accepts the canonical names
    /// plus aliases (`f16` → `ieeefp16`, `sq8` → `f8`), case-insensitively.
    pub fn parse(value: &str) -> Self {
        match value.to_lowercase().as_str() {
            "plain" => HnswPrecision::Plain,
            "ieeefp16" | "f16" => HnswPrecision::IeeeFp16,
            "ieeefp8" => HnswPrecision::IeeeFp8,
            "f8" | "sq8" => HnswPrecision::Sq8,
            _ => pgrx::error!(
                "Invalid storage_layout '{}'. Must be one of 'plain', 'ieeefp16' (f16), \
                 'ieeefp8', or 'f8' (sq8)",
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
        }
    }

    /// Encoded bytes per dimension.
    pub fn elem_bytes(&self) -> usize {
        match self {
            HnswPrecision::Plain => 4,
            HnswPrecision::IeeeFp16 => 2,
            HnswPrecision::Sq8 => 1,
            HnswPrecision::IeeeFp8 => 1,
        }
    }

    /// Whether the layout needs a build-time calibration artifact.  Only SQ8
    /// does; the IEEE layouts are stateless casts (incremental-build friendly).
    pub fn needs_calibration(&self) -> bool {
        matches!(self, HnswPrecision::Sq8)
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
        Self {
            precision: HnswPrecision::Sq8,
            dim,
            sq8_mins: calib.mins.clone(),
            sq8_maxs: calib.maxs.clone(),
            sq8_inv_scales: inv_scales,
            sq8_scales: scales,
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
                    out.extend_from_slice(&f16::from_f32(x.clamp(-FP16_MAX, FP16_MAX)).to_le_bytes());
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
                    out[d] = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
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
        }
    }

    /// Decode into a fresh vector.
    pub fn decode(&self, bytes: &[u8]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim];
        self.decode_into(bytes, &mut out);
        out
    }

    /// Compute `distance(query, bytes)` directly over the encoded byte
    /// representation, without the decode-into-scratch copy.  Matches the
    /// semantics of the [`crate::access_method::distance`] kernels exactly:
    /// L2 is the sum of squared differences (no sqrt — ordering only), inner
    /// product is `Σ q·v`, cosine is `1 − Σ q·v` clamped at zero (inputs are
    /// normalized).  This is the hottest routine of index builds and scans.
    #[inline]
    pub fn distance_encoded_direct(
        &self,
        dist_type: crate::access_method::distance::DistanceType,
        query: &[f32],
        bytes: &[u8],
    ) -> f32 {
        use crate::access_method::distance::DistanceType;
        debug_assert_eq!(query.len(), self.dim);
        debug_assert_eq!(bytes.len(), self.vector_bytes());

        // Decode one element at a time via `get` (per-layout closure).
        macro_rules! elem_loop {
            ($get:expr) => {{
                let mut acc = 0.0f32;
                match dist_type {
                    DistanceType::L2 => {
                        for (i, v) in ($get).enumerate() {
                            let d = query[i] - v;
                            acc += d * d;
                        }
                    }
                    DistanceType::InnerProduct | DistanceType::Cosine => {
                        for (i, v) in ($get).enumerate() {
                            acc += query[i] * v;
                        }
                    }
                }
                match dist_type {
                    // `<#>` orders by the NEGATIVE inner product, matching
                    // distance_inner_product.
                    DistanceType::Cosine => (1.0 - acc).max(0.0),
                    DistanceType::InnerProduct => -acc,
                    _ => acc,
                }
            }};
        }

        match self.precision {
            HnswPrecision::Plain => elem_loop!(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
            HnswPrecision::IeeeFp16 => elem_loop!(bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())),
            HnswPrecision::IeeeFp8 => elem_loop!(bytes.iter().map(|&b| e4m3_to_f32(b))),
            HnswPrecision::Sq8 => elem_loop!(bytes
                .iter()
                .enumerate()
                .map(|(d, &q)| self.sq8_mins[d] + q as f32 * self.sq8_scales[d])),
        }
    }

    /// SQ8 only: `sqrt(Σ scale_d²)` — the L2 norm of the per-dimension
    /// quantization steps.  Half of it bounds the L2 norm of one vector's
    /// quantization error `‖v̂ − v‖ ≤ scale_norm / 2` (round-to-nearest),
    /// which the scan turns into provable distance lower bounds.  Zero for
    /// the other layouts.
    pub fn sq8_scale_norm(&self) -> f32 {
        if self.precision != HnswPrecision::Sq8 {
            return 0.0;
        }
        self.sq8_scales.iter().map(|s| s * s).sum::<f32>().sqrt()
    }
}

/// Maximum relative per-element error of the IEEE layouts (round-to-nearest):
/// binary16 has 10 explicit mantissa bits (half-ulp 2^-11), E4M3 has 3
/// (half-ulp 2^-4).  Zero for plain/SQ8 (SQ8 error is absolute; see
/// [`Codec::sq8_scale_norm`]).
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

/// Encode a finite, pre-clamped (|x| <= 448) f32 as an E4M3 byte.
pub fn f32_to_e4m3(x: f32) -> u8 {
    if x == 0.0 {
        // Preserve -0.0 (harmless; keeps the sign bit semantics of the format).
        return if x.is_sign_negative() { 0x80 } else { 0x00 };
    }
    let sign: u8 = if x < 0.0 { 1 } else { 0 };
    let a = x.abs().min(FP8_E4M3_MAX);

    // Binade from log2; an off-by-one floor is self-correcting because the
    // mantissa quotient then hits q = 16 and re-promotes to the next binade.
    let mut e = a.log2().floor() as i32;
    if e < -6 {
        // Subnormal: value = q * 2^-9, q in 0..=8 (q = 8 is the minimal
        // normal 2^-6, whose encoding 0b0000_1000 is contiguous).
        let q = round_half_even(a * 512.0);
        return (sign << 7) | q.min(8) as u8;
    }
    // Normal: value = q * 2^(e-3), q in 8..=15 (16 promotes).
    let step = 2f64.powi(e - 3) as f32; // exact power of two
    let mut q = round_half_even(a / step);
    if q >= 16 {
        e += 1;
        q = 8;
    }
    if e > 8 {
        // Saturate to max finite (only reachable via rounding at the top).
        return (sign << 7) | 0x7E;
    }
    let exp_field = (e + 7) as u8; // bias 7
    (sign << 7) | (exp_field << 3) | (q - 8) as u8
}

/// Decode an E4M3 byte to f32.  The NaN encoding (0x7F/0xFF) decodes to 0.0
/// defensively: our encoder never produces it (inputs are sanitized finite),
/// and a finite fallback keeps distances well-defined on corrupted data.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0xF) as i32;
    let man = (b & 0x7) as i32;
    let v = if exp == 0 {
        man as f32 * 2.0f32.powi(-9) // subnormal (0x00/0x80 → ±0)
    } else if exp == 0xF && man == 7 {
        return 0.0; // NaN encoding → defensive 0
    } else {
        (8 + man) as f32 * 2.0f32.powi(exp - 10)
    };
    sign * v
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

    #[test]
    fn test_precision_parse_aliases() {
        assert_eq!(HnswPrecision::parse("plain"), HnswPrecision::Plain);
        assert_eq!(HnswPrecision::parse("ieeefp16"), HnswPrecision::IeeeFp16);
        assert_eq!(HnswPrecision::parse("F16"), HnswPrecision::IeeeFp16);
        assert_eq!(HnswPrecision::parse("ieeefp8"), HnswPrecision::IeeeFp8);
        assert_eq!(HnswPrecision::parse("f8"), HnswPrecision::Sq8);
        assert_eq!(HnswPrecision::parse("SQ8"), HnswPrecision::Sq8);
    }

    #[test]
    fn test_elem_bytes_roundtrip_u8() {
        for p in [
            HnswPrecision::Plain,
            HnswPrecision::IeeeFp16,
            HnswPrecision::Sq8,
            HnswPrecision::IeeeFp8,
        ] {
            assert_eq!(HnswPrecision::from_u8(p as u8), p);
            assert!(p.elem_bytes() >= 1 && p.elem_bytes() <= 4);
        }
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
}
