//! RaBitQ quantization.
//!
//! Implements the RaBitQ algorithm (Lin et al., "RaBitQ: Quantizing
//! High-Dimensional Vectors with a Theoretical Error Bound for Approximate
//! Nearest Neighbor Search", SIGMOD 2024, arXiv:2405.12497).
//!
//! ## Algorithm
//!
//! 1. **Fast random rotation** (Fast Johnson-Lindenstrauss transform): the
//!    vector is multiplied by random ±1 signs (seeded, deterministic) and then
//!    by the (symmetric) Walsh-Hadamard transform scaled by 1/√D, with the
//!    dimension padded to the next power of two.  After rotation the
//!    coordinates behave approximately like independent Gaussians.
//!
//! 2. **1-bit code**: `code[i] = sign(rotated[i])`, packed LSB-first into
//!    bytes.  Only the data vectors are stored in quantized form; the query is
//!    rotated in full `f32` precision (the query is quantized once per search,
//!    so keeping it exact costs nothing per-node).
//!
//! 3. **Norm-aware estimator** (Theorem 3.2 of the paper): for a data vector
//!    `o` and query `q` (unit-normalized by the rotation),
//!
//!    ```text
//!    <o, q> ≈ <ō, q> / <ō, o>
//!    ```
//!
//!    where `ō = P·sign(P⁻¹o)/√D` is the quantized data vector.  With
//!    `m = Σᵢ sign(P⁻¹o)ᵢ · (P⁻¹q)ᵢ` and `l1 = Σᵢ |(P⁻¹o)ᵢ|`:
//!
//!    ```text
//!    cos(o,q) = m / l1          (unbiased, error O(1/√D))
//!    dot(o,q) = ‖o‖·‖q‖·cos
//!    ‖o-q‖²   = ‖o‖² + ‖q‖² − 2·dot
//!    ```
//!
//!    The estimator is unbiased because the rotation makes the coordinate
//!    pairs approximately Gaussian with correlation `cos(o,q)`, and
//!    `E[sign(X)·Y] = E[|X|]·corr(X,Y)` for jointly Gaussian `X, Y`.
//!
//! Per data vector we therefore store only the packed sign bits plus two
//! `f32` values: `sum_of_x2 = ‖o‖²` and `l1_of_rotated = Σ|P⁻¹o|`.

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use rkyv::{Archive, Deserialize, Serialize};

use crate::access_method::distance::DistanceType;

/// Dimension padding: RaBitQ rotates in a power-of-two space.
pub fn padded_dim(dim: usize) -> usize {
    dim.max(1).next_power_of_two()
}

/// In-place fast Walsh-Hadamard transform (unscaled, symmetric: H = Hᵀ,
/// H² = n·I).  `values.len()` must be a power of two.
#[inline]
pub fn fwht_inplace(values: &mut [f32]) {
    debug_assert!(values.len().is_power_of_two());
    let mut half = 1usize;
    while half < values.len() {
        let step = half * 2;
        for block in values.chunks_exact_mut(step) {
            let (left, right) = block.split_at_mut(half);
            for (x, y) in left.iter_mut().zip(right.iter_mut()) {
                let lx = *x;
                let ry = *y;
                *x = lx + ry;
                *y = lx - ry;
            }
        }
        half = step;
    }
}

/// Apply random ±1 signs (one bit per element) by toggling the f32 sign bit.
#[inline]
fn flip_signs(values: &mut [f32], signs: &[u8]) {
    for (byte_idx, &mask) in signs.iter().enumerate() {
        let start = byte_idx * 8;
        if start >= values.len() {
            break;
        }
        let end = (start + 8).min(values.len());
        for (bit_idx, value) in values[start..end].iter_mut().enumerate() {
            let sign_mask = (((mask >> bit_idx) & 1) as u32) << 31;
            *value = f32::from_bits(value.to_bits() ^ sign_mask);
        }
    }
}

/// Fast random rotation (FJLT): random signs then normalized FWHT.
///
/// `values` has length `padded_dim(dim)` and only the first `dim` entries
/// carry data (the rest are zero and get mixed in by the Hadamard transform).
/// The transform `P = H·S/√D` is orthogonal and symmetric, so the same
/// function implements both "forward" and "inverse" rotation.  Deterministic
/// for a given `seed`.
pub fn rotate_inplace(values: &mut [f32], dim: usize, seed: u64) {
    debug_assert_eq!(values.len(), padded_dim(dim));
    let mut rng = StdRng::seed_from_u64(seed);
    let mut signs = vec![0u8; dim.div_ceil(8)];
    rng.fill_bytes(&mut signs);
    flip_signs(&mut values[..dim], &signs);
    fwht_inplace(values);
    let scale = 1.0 / (values.len() as f32).sqrt();
    for v in values.iter_mut() {
        *v *= scale;
    }
}

/// m = Σᵢ sign(codeᵢ) · rotᵢ  where `rot` is a full-precision rotated
/// vector (typically the query) and `sum_rot = Σ rot` (precomputed once per
/// query).  Uses the identity m = 2·Σ_set rot − Σ_rot so only the set bits
/// of the code are touched per call.
#[inline]
pub fn code_dot_with_rotated(code: &[u8], rot: &[f32], sum_rot: f32) -> f32 {
    debug_assert!(rot.len() >= code.len() * 8);
    let mut sum_set = 0.0f32;
    for (word_idx, chunk) in code.chunks_exact(8).enumerate() {
        let mut word = u64::from_le_bytes(chunk.try_into().unwrap());
        let base = word_idx * 64;
        while word != 0 {
            let t = word.trailing_zeros() as usize;
            sum_set += rot[base + t];
            word &= word - 1;
        }
    }
    2.0 * sum_set - sum_rot
}

/// Hamming distance between two packed codes (in bits).
#[inline]
pub fn code_hamming(a: &[u8], b: &[u8]) -> usize {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones() as usize)
        .sum()
}

/// A RaBitQ-quantized vector.
///
/// Two modes, selected by `num_bits`:
/// - `num_bits == 1`: sign-bit code.  `packed_code` has `dim / 8` bytes;
///   `l1_of_rotated = Σ|P⁻¹x|` (the estimator denominator).
/// - `num_bits == 8`: signed 8-bit scalar code of the rotated vector.
///   `packed_code` has `dim` bytes (i8 values); `l1_of_rotated` holds the
///   per-vector estimator gain `gamma = scale/⟨ō,o⟩` (see `quantize`).
///
/// `dim` is the *padded* power-of-two dimension; `sum_of_x2 = ‖x‖²`.
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
pub struct RabitqVector {
    pub dim: u32,
    pub sum_of_x2: f32,
    pub l1_of_rotated: f32,
    pub packed_code: Vec<u8>,
    pub num_bits: u8,
    /// ⟨rot_c, code⟩ — the rotated-center correction (Lance add-factor term),
    /// precomputed at build time and stored on the node.
    pub cent_dot: f32,
}

impl RabitqVector {
    pub fn quantized_size_bytes(&self) -> usize {
        self.packed_code.len() + std::mem::size_of::<f32>() * 2
    }

    /// Lance-style raw dot with a full-precision rotated vector `rot`
    /// (typically the query residual).  For 1-bit: `m = Σ sign(code)·rotᵢ`.
    /// For 4/8-bit: `full_dot = code_scale·binary_ip + ex_dist + code_bias·Σrot`.
    #[inline]
    pub fn dot_with_rotated(&self, rot: &[f32]) -> f32 {
        match self.num_bits {
            4 => {
                let code_scale = 8.0;
                let code_bias = -7.5;
                let mut binary_ip = 0.0f32;
                let mut ex_dist = 0.0f32;
                for (bi, &b) in self.packed_code.iter().enumerate() {
                    let lo = b & 0x0F;
                    let hi = b >> 4;
                    for (k, nib) in [lo, hi].into_iter().enumerate() {
                        let r = rot[bi * 2 + k];
                        binary_ip += if nib & 0x08 != 0 { r } else { -r };
                        ex_dist += (nib & 0x07) as f32 * r;
                    }
                }
                code_scale * binary_ip
                    + ex_dist
                    + code_bias * rot[..self.dim as usize].iter().sum::<f32>()
            }
            8 => {
                let code_scale = 128.0;
                let code_bias = -127.5;
                let mut binary_ip = 0.0f32;
                let mut ex_dist = 0.0f32;
                for (i, &b) in self.packed_code.iter().enumerate() {
                    let r = rot[i];
                    binary_ip += if b & 0x80 != 0 { r } else { -r };
                    ex_dist += (b & 0x7F) as f32 * r;
                }
                code_scale * binary_ip
                    + ex_dist
                    + code_bias * rot[..self.dim as usize].iter().sum::<f32>()
            }
            _ => code_dot_with_rotated(
                &self.packed_code,
                rot,
                rot[..self.dim as usize].iter().sum(),
            ),
        }
    }

    /// Code-to-code signed agreement count `m12 = Σ sign(c1)ᵢ·sign(c2)ᵢ`
    /// (equals `dim − 2·hamming`).
    #[inline]
    pub fn code_agreement(&self, other: &Self) -> f32 {
        debug_assert_eq!(self.dim, other.dim);
        self.dim as f32 - 2.0 * code_hamming(&self.packed_code, &other.packed_code) as f32
    }

    /// Estimated distance between two *stored* (code-only) vectors.
    ///
    /// Only sign bits are available on both sides, so the unbiased
    /// search-style estimator (which needs the full-precision rotated query)
    /// cannot be used.  Instead we use the classic binary-code cosine
    /// estimator: `m12/D` estimates `(2/π)·arcsin(ρ)`, hence
    /// `cos ≈ sin((π/2)·m12/D)`, which is exact at ρ ∈ {-1, 0, 1} and has
    /// error O(1/√D) elsewhere.
    pub fn estimated_distance(&self, other: &Self, distance_type: DistanceType) -> f32 {
        let d = self.dim as f32;
        let m12 = self.code_agreement(other);
        let cos = (std::f32::consts::FRAC_PI_2 * (m12 / d)).sin();
        Self::distance_from_cos(cos, self.sum_of_x2, other.sum_of_x2, distance_type)
    }

    /// Combine a cosine estimate with the stored norms into a distance for
    /// the given distance type.  `cos` is clamped to [-1, 1].
    #[inline]
    pub fn distance_from_cos(cos: f32, sx2_a: f32, sx2_b: f32, distance_type: DistanceType) -> f32 {
        let cos = cos.clamp(-1.0, 1.0);
        let dot = sx2_a.max(0.0).sqrt() * sx2_b.max(0.0).sqrt() * cos;
        match distance_type {
            DistanceType::L2 => (sx2_a + sx2_b - 2.0 * dot).max(0.0),
            DistanceType::InnerProduct => -dot,
            DistanceType::Cosine => 1.0 - cos,
        }
    }
}

/// The RaBitQ quantizer.  Data-independent: no training pass needed, only a
/// rotation seed.  For L2 datasets with a large DC component (e.g. BIGANN's
/// u8 vectors), a global center (dataset mean) is subtracted before rotation
/// so the sign bits capture the discriminative deviation structure; the L2
/// distance is invariant under this translation.
#[derive(Clone)]
pub struct RabitqQuantizer {
    pub num_bits: u8,
    pub rotation_seed: u64,
    pub dim: usize,
    /// Optional global center subtracted before rotation (paper's `o − c`).
    pub center: Vec<f32>,
}

impl RabitqQuantizer {
    pub fn new(num_bits: u8, rotation_seed: u64, dim: usize) -> Self {
        Self {
            num_bits,
            rotation_seed,
            dim: padded_dim(dim),
            center: Vec::new(),
        }
    }

    pub fn with_center(mut self, center: Vec<f32>) -> Self {
        self.center = center;
        self
    }

    pub fn rotation_seed(&self) -> u64 {
        self.rotation_seed
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Rotated global center (same rotation as `quantize`/`rotate_query`),
    /// used for the centroid-correction term ⟨c, code⟩.
    pub fn rotate_center(&self) -> (Vec<f32>, f32) {
        self.rotate_center_of(&self.center.clone())
    }

    /// Rotate an arbitrary center/centroid vector (padded to the rotation dim),
    /// returning the rotated vector and its element sum.
    pub fn rotate_center_of(&self, center: &[f32]) -> (Vec<f32>, f32) {
        if center.is_empty() {
            return (vec![0.0; self.dim], 0.0);
        }
        let mut rotated = vec![0f32; self.dim];
        rotated[..center.len()].copy_from_slice(center);
        rotate_inplace(&mut rotated, center.len(), self.rotation_seed);
        let sum = rotated.iter().sum::<f32>();
        (rotated, sum)
    }

const EX_QUANTIZATION_EPSILON: f32 = 1.0e-5;
const EX_TIGHT_START: [f32; 9] = [0.0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81];

/// Lance's `best_ex_rescale_factor`: the scale t for the ex-code magnitude
/// quantization that maximizes ⟨|r̂|, code⟩/‖code‖ (tight quantization of the
/// unit-direction magnitude shape).
fn best_ex_rescale_factor(abs_normalized: &[f32], ex_bits: u32) -> f32 {
    let max_value = abs_normalized.iter().copied().fold(0.0f32, f32::max);
    if max_value <= 0.0 {
        return 0.0;
    }
    let max_code = (1usize << ex_bits) - 1;
    let t_end = ((max_code + 10) as f32) / max_value;
    let t_start = t_end * Self::EX_TIGHT_START[ex_bits as usize];

    let mut current_codes = Vec::with_capacity(abs_normalized.len());
    let mut squared_denominator = abs_normalized.len() as f32 * 0.25;
    let mut numerator = 0.0f32;
    let mut thresholds = Vec::with_capacity(abs_normalized.len() * max_code);

    for (idx, &value) in abs_normalized.iter().enumerate() {
        if value <= 0.0 || !value.is_finite() {
            current_codes.push(0usize);
            continue;
        }
        let current = ((t_start * value) + Self::EX_QUANTIZATION_EPSILON)
            .floor()
            .clamp(0.0, max_code as f32) as usize;
        current_codes.push(current);
        squared_denominator += (current * current + current) as f32;
        numerator += (current as f32 + 0.5) * value;

        let mut next = current + 1;
        while next <= max_code {
            let threshold = next as f32 / value;
            if threshold < t_end {
                thresholds.push((threshold, idx));
            }
            next += 1;
        }
    }

    thresholds.sort_unstable_by(|(left, _), (right, _)| left.total_cmp(right));

    let mut best_inner_product = numerator / squared_denominator.sqrt();
    let mut best_t = t_start;
    for (threshold, idx) in thresholds {
        current_codes[idx] += 1;
        let updated = current_codes[idx];
        squared_denominator += (2 * updated) as f32;
        numerator += abs_normalized[idx];

        let current_inner_product = numerator / squared_denominator.sqrt();
        if current_inner_product > best_inner_product {
            best_inner_product = current_inner_product;
            best_t = threshold;
        }
    }
    best_t
}

pub fn quantize(&self, full_vector: &[f32]) -> RabitqVector {
        self.quantize_residual(&self.center.clone(), full_vector)
    }

    /// Quantize a vector relative to an explicit centroid (IVF list centroid),
    /// rather than the global center.  The centroid is subtracted before
    /// rotation; `cent_dot` stores ⟨rot(centroid), code⟩ for the L2 estimator.
    pub fn quantize_residual(&self, centroid: &[f32], full_vector: &[f32]) -> RabitqVector {
        let mut centered = full_vector.to_vec();
        if !centroid.is_empty() {
            debug_assert_eq!(centroid.len(), full_vector.len());
            for (x, c) in centered.iter_mut().zip(centroid.iter()) {
                *x -= c;
            }
        }
        // Lance-style residual RaBitQ: the vector is centered (residual to the
        // centroid), NOT normalized — the per-vector scale/add factors absorb
        // the norms.  sum_of_x2 = ‖r‖² is the residual norm squared.
        let sum_of_x2 = centered.iter().map(|v| v * v).sum::<f32>();
        let mut rotated = vec![0f32; self.dim];
        rotated[..full_vector.len()].copy_from_slice(&centered);
        rotate_inplace(&mut rotated, full_vector.len(), self.rotation_seed);
        let (rot_c, sum_rc) = self.rotate_center_of(centroid);

        match self.num_bits {
            4 | 8 => {
                // Lance sign + unsigned ex-bits code: per dimension the top
                // bit is the sign, the low (num_bits-1) bits are the magnitude
                // |rot| quantized with Lance's optimized normalized scale t
                // (the ex code for negative dims is bitwise-inverted).
                //   l1_of_rotated = ⟨rot, code⟩ (reconstruction dot)
                //   sum_of_x2     = ‖r‖²
                let ex_bits = (self.num_bits - 1) as u32;
                let max_code = ((1u16 << ex_bits) - 1) as u8;
                let mask = max_code;
                let code_scale = (1u32 << ex_bits) as f32;
                let code_bias = -(code_scale - 0.5);
                let norm = sum_of_x2.sqrt().max(1e-9);
                let abs_normalized: Vec<f32> =
                    rotated.iter().map(|v| v.abs() / norm).collect();
                let t = Self::best_ex_rescale_factor(&abs_normalized, ex_bits);
                let mut code = vec![0u8; self.dim.div_ceil(2 / (self.num_bits / 4) as usize)];
                let mut res_dot = 0.0f32; // ⟨rot, code⟩
                let mut cent_dot = 0.0f32; // ⟨rot_c, code⟩
                for (i, &v) in rotated.iter().enumerate() {
                    let mut ex = ((t * abs_normalized[i]) + Self::EX_QUANTIZATION_EPSILON)
                        .floor()
                        .clamp(0.0, max_code as f32) as u8;
                    if v.is_sign_negative() {
                        ex = (!ex) & mask;
                    }
                    let sign_bit = u8::from(v.is_sign_positive());
                    let full_code = ((sign_bit as u32) << ex_bits) + ex as u32;
                    let factor = full_code as f32 + code_bias;
                    res_dot += v * factor;
                    cent_dot += factor * rot_c[i];
                    if self.num_bits == 4 {
                        // nibble: bit3 = sign, bits 0-2 = ex; 2 dims per byte
                        let nib = (sign_bit << 3) | (ex & 0x07);
                        let byte = &mut code[i / 2];
                        if i % 2 == 0 {
                            *byte |= nib;
                        } else {
                            *byte |= nib << 4;
                        }
                    } else {
                        // byte: bit7 = sign, bits 0-6 = ex
                        code[i] = (sign_bit << 7) | (ex & 0x7F);
                    }
                }
                RabitqVector {
                    dim: self.dim as u32,
                    sum_of_x2,
                    l1_of_rotated: res_dot,
                    packed_code: code,
                    num_bits: self.num_bits,
                    cent_dot,
                }
            }
            _ => {
                let mut code = vec![0u8; self.dim.div_ceil(8)];
                let mut l1 = 0f32; // Σ|rot| = ⟨rot, sign(code)⟩
                for (i, &val) in rotated.iter().enumerate() {
                    if val > 0.0 {
                        code[i / 8] |= 1 << (i % 8);
                    }
                    l1 += val.abs();
                }
                let cent_dot = code_dot_with_rotated(&code, &rot_c, sum_rc);
                RabitqVector {
                    dim: self.dim as u32,
                    sum_of_x2,
                    l1_of_rotated: l1,
                    packed_code: code,
                    num_bits: self.num_bits,
                    cent_dot,
                }
            }
        }
    }

    /// Rotate a full-precision query vector (kept in `f32` — the query is
    /// only quantized once per search, so precision is free).
    pub fn rotate_query(&self, query: &[f32]) -> RabitqQuery {
        self.rotate_query_residual(&self.center.clone(), query)
    }

    /// Rotate a query relative to an explicit centroid (IVF list centroid).
    pub fn rotate_query_residual(&self, centroid: &[f32], query: &[f32]) -> RabitqQuery {
        let mut centered = query.to_vec();
        if !centroid.is_empty() {
            debug_assert_eq!(centroid.len(), query.len());
            for (x, c) in centered.iter_mut().zip(centroid.iter()) {
                *x -= c;
            }
        }
        let sum_of_x2 = centered.iter().map(|v| v * v).sum::<f32>();
        let mut rotated = vec![0f32; self.dim];
        rotated[..query.len()].copy_from_slice(&centered);
        rotate_inplace(&mut rotated, query.len(), self.rotation_seed);
        let l1 = rotated.iter().map(|v| v.abs()).sum::<f32>();
        RabitqQuery {
            sum_q: rotated.iter().sum(),
            rotated,
            sum_of_x2,
            l1,
        }
    }

    /// Lance-style L2 distance estimate between a quantized residual and a
    /// rotated full-precision query residual (asymmetric estimator: data
    /// vector quantized, query kept in f32).  Clamped to >= 0.
    ///
    /// L2 is translation-invariant, so the residual estimate is
    /// `‖ro‖² + ‖rq‖² - 2·(‖ro‖²/⟨ro,code⟩)·⟨code,rq⟩`; no centroid
    /// correction term is needed (that term belongs to the inner-product
    /// estimator).
    #[inline]
    pub fn estimate_l2(&self, qv: &RabitqVector, rq: &RabitqQuery) -> f32 {
        let full_dot = qv.dot_with_rotated(&rq.rotated);
        let res_dot = qv.l1_of_rotated.max(1e-9);
        let scale = -2.0 * qv.sum_of_x2 / res_dot;
        (full_dot * scale + qv.sum_of_x2 + rq.sum_of_x2).max(0.0)
    }
}

/// A rotated, full-precision query plus its precomputed aggregates.
pub struct RabitqQuery {
    pub rotated: Vec<f32>,
    /// Σ rotated — precomputed for the fast m = 2·Σ_set − Σ_all trick.
    pub sum_q: f32,
    pub sum_of_x2: f32,
    pub l1: f32,
}

#[cfg(test)]
/// Lance-style L2 estimate of (o, q) used by the unit tests: mirrors the
/// search measure's `calculate_bq_distance`.
fn lance_dist(qv: &RabitqVector, rq: &RabitqQuery, _quantizer: &RabitqQuantizer) -> f32 {
    let full_dot = qv.dot_with_rotated(&rq.rotated);
    let cent_dot = qv.cent_dot;
    let res_dot = qv.l1_of_rotated.max(1e-9);
    let scale = -2.0 * qv.sum_of_x2 / res_dot;
    let add = qv.sum_of_x2 + 2.0 * qv.sum_of_x2 * cent_dot / res_dot;
    // raw (unclamped) estimate — the clamp is only for the graph invariant
    full_dot * scale + add + rq.sum_of_x2
}

mod tests {
    use super::*;
    use rand::Rng;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn rotation_preserves_norm() {
        let dim = 128;
        let v: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01 + 0.5).collect();
        let sx2: f32 = v.iter().map(|x| x * x).sum();
        let mut rot = vec![0f32; padded_dim(dim)];
        rot[..dim].copy_from_slice(&v);
        rotate_inplace(&mut rot, dim, 42);
        let rot_sx2: f32 = rot.iter().map(|x| x * x).sum();
        assert!(approx(rot_sx2, sx2, sx2 * 1e-3), "{} vs {}", rot_sx2, sx2);
    }

    #[test]
    fn rotation_is_deterministic() {
        let dim = 64;
        let v: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let mut a = vec![0f32; padded_dim(dim)];
        let mut b = vec![0f32; padded_dim(dim)];
        a[..dim].copy_from_slice(&v);
        b[..dim].copy_from_slice(&v);
        rotate_inplace(&mut a, dim, 7);
        rotate_inplace(&mut b, dim, 7);
        assert_eq!(a, b);
    }

    #[test]
    fn code_packing_roundtrip() {
        let q = RabitqQuantizer::new(1, 1, 16);
        let input =
            vec![1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0];
        let v = q.quantize(&input);
        // code bits must equal sign of the rotated vector
        let mut rot = vec![0f32; padded_dim(16)];
        rot[..16].copy_from_slice(&input);
        rotate_inplace(&mut rot, 16, 1);
        for (i, &val) in rot.iter().enumerate() {
            let bit_set = v.packed_code[i / 8] & (1 << (i % 8)) != 0;
            assert_eq!(bit_set, val > 0.0, "bit {} mismatch (rot val {})", i, val);
        }
        assert_eq!(v.packed_code.len(), 2); // 16 dims / 8
        assert!(approx(v.sum_of_x2, 16.0, 1e-5));
    }

    #[test]
    fn quantized_size_matches_layout() {
        let q = RabitqQuantizer::new(1, 1, 128);
        let v = q.quantize(&vec![0.5f32; 128]);
        assert_eq!(v.quantized_size_bytes(), 16 + 8);
        assert_eq!(v.dim, 128);
    }

    #[test]
    fn identical_vectors_estimate_zero_l2() {
        let q = RabitqQuantizer::new(1, 3, 128);
        let v: Vec<f32> = (0..128).map(|i| ((i % 7) as f32) - 3.0).collect();
        let qv = q.quantize(&v);
        let d = qv.estimated_distance(&qv, DistanceType::L2);
        assert!(d < 1e-3, "self L2 distance should be ~0, got {}", d);
    }

    #[test]
    fn cosine_estimation_accuracy() {
        // Estimate cosine between two random vectors.  Single-rotation
        // variance is O(1/sqrt(D)) ~ 0.09 for D=128, so allow 0.25 per draw.
        let q = RabitqQuantizer::new(1, 99, 128);
        let mut rng = StdRng::seed_from_u64(5);
        let mut buf = vec![0f32; 128];
        let mut max_err = 0.0f32;
        for _ in 0..10 {
            rng.fill(&mut buf[..]);
            let a: Vec<f32> = buf.clone();
            rng.fill(&mut buf[..]);
            let b: Vec<f32> = buf.clone();
            let qa = q.quantize(&a);

            let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let exact: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>() / (na * nb);

            // search-style: full-precision rotated query
            let rq = q.rotate_query(&b);
            let m = qa.dot_with_rotated(&rq.rotated);
            let est = m / qa.l1_of_rotated;
            max_err = max_err.max((est - exact).abs());
        }
        assert!(
            max_err < 0.25,
            "max cosine estimation error {} exceeds 0.25",
            max_err
        );
    }

    #[test]
    fn eight_bit_l2_accuracy() {
        // 8-bit mode should be much more accurate than 1-bit (error ~1/128).
        let q = RabitqQuantizer::new(8, 42, 128);
        let mut rng = StdRng::seed_from_u64(9);
        let mut buf = vec![0f32; 128];
        let a: Vec<f32> = (0..128).map(|i| ((i % 11) as f32) - 5.0).collect();
        let qa = q.quantize(&a);
        assert_eq!(qa.packed_code.len(), 128, "8-bit code is 1 byte per dim");
        assert_eq!(qa.num_bits, 8);
        let mut max_rel = 0.0f32;
        for _ in 0..10 {
            rng.fill(&mut buf[..]);
            let b: Vec<f32> = buf.clone();
            let exact: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>();
            let rq = q.rotate_query(&b);
            let est = lance_dist(&qa, &rq, &q);
            let rel = (est - exact).abs() / exact.max(1.0);
            max_rel = max_rel.max(rel);
        }
        assert!(
            max_rel < 0.08,
            "8-bit max L2 rel err {} exceeds 0.08",
            max_rel
        );
    }

    #[test]
    fn bigann_like_estimation_accuracy() {
        // BIGANN vectors are uint8 0-255: large norm, large positive mean,
        // all components positive.  Verify the estimator on this
        // distribution (unit tests used small centered vectors).
        let q = RabitqQuantizer::new(1, 42, 128);
        let mut rng = StdRng::seed_from_u64(11);
        let mut buf = vec![0u8; 128];
        let mut a = vec![0f32; 128];
        for (i, x) in a.iter_mut().enumerate() {
            rng.fill(&mut buf[..]);
            *x = buf[i % 128] as f32;
        }
        let qa = q.quantize(&a);
        let mut worst = 0.0f32;
        let mut mean_err = 0.0f32;
        let n = 50;
        for _ in 0..n {
            rng.fill(&mut buf[..]);
            let b: Vec<f32> = buf.iter().map(|&v| v as f32).collect();
            let exact: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>();
            let rq = q.rotate_query(&b);
            let m = qa.dot_with_rotated(&rq.rotated);
            let cos = m / qa.l1_of_rotated;
            let true_cos: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
                / (qa.sum_of_x2 * rq.sum_of_x2).sqrt();
            let est =
                RabitqVector::distance_from_cos(cos, qa.sum_of_x2, rq.sum_of_x2, DistanceType::L2);
            let rel = (est - exact).abs() / exact.max(1.0);
            worst = worst.max(rel);
            mean_err += rel;
        }
        for k in 0..5 {
            rng.fill(&mut buf[..]);
            let b: Vec<f32> = buf.iter().map(|&v| v as f32).collect();
            let exact: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>();
            let rq = q.rotate_query(&b);
            let m = qa.dot_with_rotated(&rq.rotated);
            let cos = m / qa.l1_of_rotated;
            let true_cos: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| x * y)
                .sum::<f32>()
                / (qa.sum_of_x2 * rq.sum_of_x2).sqrt();
            let est = RabitqVector::distance_from_cos(
                cos,
                qa.sum_of_x2,
                rq.sum_of_x2,
                DistanceType::L2,
            );
            eprintln!(
                "pair{}: exact_l2={:.0} est_l2={:.0} est_cos={:.3} true_cos={:.3} sx2b={:.0}",
                k, exact, est, cos, true_cos, rq.sum_of_x2
            );
        }
        let mean_err = mean_err / n as f32;
        eprintln!(
            "bigann-like diag: norm_a={:.1} l1={:.1} sum_of_x2={:.1} mean_err={:.3} worst={:.3}",
            a.iter().map(|v| v * v).sum::<f32>().sqrt(),
            qa.l1_of_rotated,
            qa.sum_of_x2,
            mean_err,
            worst
        );
        assert!(
            mean_err < 0.3,
            "bigann-like mean L2 rel err {} exceeds 0.3 (worst {})",
            mean_err,
            worst
        );
    }

    #[test]
    fn eight_bit_identical_is_zero() {
        let q = RabitqQuantizer::new(8, 7, 128);
        let v: Vec<f32> = (0..128).map(|i| ((i % 7) as f32) - 3.0).collect();
        let qv = q.quantize(&v);
        let rq = q.rotate_query(&v);
        let cos = qv.dot_with_rotated(&rq.rotated);
        let d = RabitqVector::distance_from_cos(cos, qv.sum_of_x2, rq.sum_of_x2, DistanceType::L2);
        assert!(d < 1e-3, "8-bit self L2 should be ~0, got {}", d);
    }
}

#[cfg(test)]
mod centered_tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn centered_bigann_accuracy() {
        // BIGANN-like u8 data WITH a trained global center subtracted.
        let mut rng = StdRng::seed_from_u64(21);
        let mut buf = vec![0u8; 128];
        // train center on 10000 samples
        let mut sums = vec![0f64; 128];
        for _ in 0..10000 {
            rng.fill(&mut buf[..]);
            for (s, &v) in sums.iter_mut().zip(buf.iter()) {
                *s += v as f64;
            }
        }
        let center: Vec<f32> = sums.iter().map(|s| (*s / 10000.0) as f32).collect();
        let q = RabitqQuantizer::new(1, 42, 128).with_center(center);
        let mut a = vec![0f32; 128];
        rng.fill(&mut buf[..]);
        for (x, &v) in a.iter_mut().zip(buf.iter()) {
            *x = v as f32;
        }
        let qa = q.quantize(&a);
        let mut mean_err = 0.0f32;
        let n = 50;
        for _ in 0..n {
            rng.fill(&mut buf[..]);
            let b: Vec<f32> = buf.iter().map(|&v| v as f32).collect();
            let exact: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>();
            let rq = q.rotate_query(&b);
            let m = qa.dot_with_rotated(&rq.rotated);
            let cos = m / qa.l1_of_rotated;
            let est =
                RabitqVector::distance_from_cos(cos, qa.sum_of_x2, rq.sum_of_x2, DistanceType::L2);
            mean_err += (est - exact).abs() / exact.max(1.0);
        }
        let mean_err = mean_err / n as f32;
        assert!(
            mean_err < 0.15,
            "centered bigann mean L2 rel err {} exceeds 0.15",
            mean_err
        );
    }
}

#[cfg(test)]
mod search_path_tests {
    use super::*;

    #[test]
    fn self_match_with_center_is_zero() {
        // Search-path check: quantize a vector with a trained center, then
        // "query" with the same vector; the estimated L2 must be ~0.
        let dim = 128;
        let center: Vec<f32> = (0..dim).map(|i| 127.5 + (i % 7) as f32).collect();
        let q = RabitqQuantizer::new(1, 42, dim).with_center(center.clone());
        let v: Vec<f32> = (0..dim).map(|i| ((i * 37) % 256) as f32).collect();
        let qv = q.quantize(&v);
        let rq = q.rotate_query(&v);
        let dist = lance_dist(&qv, &rq, &q);
        assert!(dist < 1e-3, "self-match distance {dist}");
    }
}

#[cfg(test)]
mod four_bit_tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn four_bit_bigann_accuracy() {
        // 4-bit scalar mode on BIGANN-like data: error between 1-bit and 8-bit.
        let q = RabitqQuantizer::new(4, 42, 128);
        let center: Vec<f32> = vec![127.5; 128];
        let q = q.with_center(center);
        let mut rng = StdRng::seed_from_u64(9);
        let mut buf = vec![0u8; 128];
        rng.fill(&mut buf[..]);
        let a: Vec<f32> = buf.iter().map(|&v| v as f32).collect();
        let qa = q.quantize(&a);
        let mut mean_err = 0.0f32;
        for _ in 0..20 {
            rng.fill(&mut buf[..]);
            let b: Vec<f32> = buf.iter().map(|&v| v as f32).collect();
            let rq = q.rotate_query(&b);
            let exact: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
            let est = lance_dist(&qa, &rq, &q);
            mean_err += ((est - exact) / exact.max(1.0)).abs();
        }
        let mean_err = mean_err / 20.0;
        assert!(mean_err < 0.22, "4-bit mean L2 rel err {} too high", mean_err);
    }

    #[test]
    fn real_bigann_8bit_ranking() {
        // Uses /tmp/est_test.csv (real bigann vectors: query 0 = id -1).
        let raw = std::fs::read_to_string("/tmp/est_test.csv").expect("est_test.csv");
        let mut vectors: Vec<(i32, Vec<f32>)> = Vec::new();
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let (id, rest) = line.split_once(',').expect("csv split");
            let id: i32 = id.trim().parse().expect("id parse");
            let vals: Vec<f32> = rest
                .trim_matches('"')
                .split(',')
                .filter_map(|v| v.trim().parse::<f32>().ok())
                .collect();
            vectors.push((id, vals));
        }
        let query = vectors
            .iter()
            .find(|(id, _)| *id == -1)
            .expect("query")
            .1
            .clone();
        let q = RabitqQuantizer::new(8, 42, 128);
        let mut exact: Vec<(i32, f32)> = Vec::new();
        let mut est: Vec<(i32, f32)> = Vec::new();
        for (id, v) in &vectors {
            if *id == -1 {
                continue;
            }
            let exact_d: f32 = v
                .iter()
                .zip(query.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let qv = q.quantize(v);
            let rq = q.rotate_query(&query);
            let est_d = lance_dist(&qv, &rq, &q);
            exact.push((*id, exact_d));
            est.push((*id, est_d));
        }
        exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        est.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        eprintln!("exact top5: {:?}", &exact[..5]);
        eprintln!("est   top5: {:?}", &est[..5]);
        assert_eq!(
            est[0].0, exact[0].0,
            "8-bit estimator must rank the exact NN first"
        );
    }
}
