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

/// Dot of a stored code with a full-precision rotated vector `rot`, given the
/// primitive fields directly.
///
/// This mirrors `RabitqVector::dot_with_rotated` but operates on borrowed
/// slices so the zero-copy (archived) scan path never allocates an owned
/// `RabitqVector` per entry.
#[inline]
pub fn dot_with_rotated_fields(num_bits: u8, dim: u32, packed_code: &[u8], rot: &[f32]) -> f32 {
    match num_bits {
        2 => {
            // 1 sign bit + 1 ex bit per dim, 4 dims per byte (bits 2j = sign,
            // 2j+1 = ex).  Lance full_dot: 2·Σ sign·rot + Σ ex·rot − 1.5·Σrot,
            // with the sign bit unsigned 0/1.
            let code_scale = 2.0;
            let code_bias = -1.5;
            let mut binary_ip = 0.0f32;
            let mut ex_dist = 0.0f32;
            for (bi, &b) in packed_code.iter().enumerate() {
                for j in 0..4u32 {
                    let dim_idx = bi as u32 * 4 + j;
                    if dim_idx as usize >= rot.len() {
                        break;
                    }
                    let r = rot[dim_idx as usize];
                    if b & (1 << (2 * j)) != 0 {
                        binary_ip += r;
                    }
                    ex_dist += ((b >> (2 * j + 1)) & 1) as f32 * r;
                }
            }
            code_scale * binary_ip
                + ex_dist
                + code_bias * rot[..dim as usize].iter().sum::<f32>()
        }
        4 => {
            let code_scale = 8.0;
            let code_bias = -7.5;
            let mut binary_ip = 0.0f32;
            let mut ex_dist = 0.0f32;
            for (bi, &b) in packed_code.iter().enumerate() {
                let lo = b & 0x0F;
                let hi = b >> 4;
                for (k, nib) in [lo, hi].into_iter().enumerate() {
                    let r = rot[bi * 2 + k];
                    // The sign bit is unsigned 0/1 in the Lance full_dot
                    // formula: `2^ex_bits * Σ sign·rot + Σ ex·rot + bias·Σrot`.
                    binary_ip += if nib & 0x08 != 0 { r } else { 0.0 };
                    ex_dist += (nib & 0x07) as f32 * r;
                }
            }
            code_scale * binary_ip
                + ex_dist
                + code_bias * rot[..dim as usize].iter().sum::<f32>()
        }
        8 => {
            let code_scale = 128.0;
            let code_bias = -127.5;
            let mut binary_ip = 0.0f32;
            let mut ex_dist = 0.0f32;
            for (i, &b) in packed_code.iter().enumerate() {
                let r = rot[i];
                // Sign bit is unsigned 0/1 (see 4-bit branch comment).
                binary_ip += if b & 0x80 != 0 { r } else { 0.0 };
                ex_dist += (b & 0x7F) as f32 * r;
            }
            code_scale * binary_ip
                + ex_dist
                + code_bias * rot[..dim as usize].iter().sum::<f32>()
        }
        _ => code_dot_with_rotated(packed_code, rot, rot[..dim as usize].iter().sum()),
    }
}

/// `Σ full_code·rot` for multi-bit codes (our sequential layout: 8-bit is one
/// byte per dim, 4-bit is two nibbles per byte).  The caller reconstructs the
/// full dot as `dot_full_code + code_bias·Σrot`.
#[inline]
pub fn dot_full_code(num_bits: u8, code: &[u8], rot: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: only selected when AVX2 and FMA were detected.
            return unsafe { ex_dot_simd::dot_full_code_avx2(num_bits, code, rot) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is part of the aarch64 baseline (no runtime detection needed).
        // SAFETY: NEON is guaranteed on aarch64.
        return unsafe { ex_dot_neon::dot_full_code_neon(num_bits, code, rot) };
    }
    dot_full_code_scalar(num_bits, code, rot)
}

#[inline]
fn dot_full_code_scalar(num_bits: u8, code: &[u8], rot: &[f32]) -> f32 {
    match num_bits {
        4 => {
            let mut s = 0.0f32;
            for (i, &b) in code.iter().enumerate() {
                s += (b & 0x0F) as f32 * rot[i * 2];
                s += (b >> 4) as f32 * rot[i * 2 + 1];
            }
            s
        }
        8 => {
            let mut s = 0.0f32;
            for (i, &b) in code.iter().enumerate() {
                s += b as f32 * rot[i];
            }
            s
        }
        _ => 0.0,
    }
}

#[cfg(target_arch = "x86_64")]
mod ex_dot_simd {
    use std::arch::x86_64::*;

    /// Dispatch to the per-width kernel.
    #[inline]
    pub(super) unsafe fn dot_full_code_avx2(num_bits: u8, code: &[u8], rot: &[f32]) -> f32 {
        match num_bits {
            4 => dot_u4_full_avx2(code, rot),
            8 => dot_u8_full_avx2(code, rot),
            _ => super::dot_full_code_scalar(num_bits, code, rot),
        }
    }

    /// Unpack 8 bytes (16 sequential 4-bit codes, dims 2i and 2i+1 per byte)
    /// into 16 bytes in natural dim order.
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn unpack_u4_sequential(ptr: *const u8) -> __m128i {
        let word = (ptr as *const u64).read_unaligned();
        let mask = 0x0f0f_0f0f_0f0f_0f0fu64;
        let lo = word & mask;
        let hi = (word >> 4) & mask;
        _mm_unpacklo_epi8(_mm_set_epi64x(0, lo as i64), _mm_set_epi64x(0, hi as i64))
    }

    /// FMA 16 u8 codes against 16 query floats (AVX2: two 8-float halves).
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn fma16_avx2(codes: __m128i, query: *const f32, acc: &mut [__m256; 2]) {
        let lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(codes));
        acc[0] = _mm256_fmadd_ps(lo, _mm256_loadu_ps(query), acc[0]);
        let hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(codes)));
        acc[1] = _mm256_fmadd_ps(hi, _mm256_loadu_ps(query.add(8)), acc[1]);
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn reduce_add_avx2(v: __m256) -> f32 {
        let halves = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let pairs = _mm_add_ps(halves, _mm_movehl_ps(halves, halves));
        let total = _mm_add_ss(pairs, _mm_shuffle_ps::<1>(pairs, pairs));
        _mm_cvtss_f32(total)
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn dot_u8_full_avx2(code: &[u8], rot: &[f32]) -> f32 {
        let mut acc = [_mm256_setzero_ps(); 2];
        let n = code.len();
        let full = n - (n % 16);
        let mut i = 0;
        while i < full {
            let codes = _mm_loadu_si128(code.as_ptr().add(i) as *const __m128i);
            fma16_avx2(codes, rot.as_ptr().add(i), &mut acc);
            i += 16;
        }
        let mut sum = reduce_add_avx2(_mm256_add_ps(acc[0], acc[1]));
        for j in i..n {
            sum += code[j] as f32 * rot[j];
        }
        sum
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn dot_u4_full_avx2(code: &[u8], rot: &[f32]) -> f32 {
        let mut acc = [_mm256_setzero_ps(); 2];
        let n_bytes = code.len();
        let full_bytes = n_bytes - (n_bytes % 8);
        let mut i = 0;
        while i < full_bytes {
            let unpacked = unpack_u4_sequential(code.as_ptr().add(i));
            fma16_avx2(unpacked, rot.as_ptr().add(i * 2), &mut acc);
            i += 8;
        }
        let mut sum = reduce_add_avx2(_mm256_add_ps(acc[0], acc[1]));
        for j in i..n_bytes {
            sum += (code[j] & 0x0F) as f32 * rot[j * 2];
            sum += (code[j] >> 4) as f32 * rot[j * 2 + 1];
        }
        sum
    }
}

#[cfg(target_arch = "aarch64")]
mod ex_dot_neon {
    use std::arch::aarch64::*;

    /// Dispatch to the per-width NEON kernel (NEON is baseline on aarch64).
    #[inline]
    pub(super) unsafe fn dot_full_code_neon(num_bits: u8, code: &[u8], rot: &[f32]) -> f32 {
        match num_bits {
            4 => dot_u4_full_neon(code, rot),
            8 => dot_u8_full_neon(code, rot),
            _ => super::dot_full_code_scalar(num_bits, code, rot),
        }
    }

    /// FMA 16 u8 codes against 16 query floats over four 4-float lanes.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn fma16_neon(codes: uint8x16_t, query: *const f32, acc: &mut [float32x4_t; 4]) {
        let lo = vmovl_u8(vget_low_u8(codes));
        let hi = vmovl_u8(vget_high_u8(codes));
        let c0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
        let c1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo)));
        let c2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
        let c3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi)));
        acc[0] = vfmaq_f32(acc[0], c0, vld1q_f32(query));
        acc[1] = vfmaq_f32(acc[1], c1, vld1q_f32(query.add(4)));
        acc[2] = vfmaq_f32(acc[2], c2, vld1q_f32(query.add(8)));
        acc[3] = vfmaq_f32(acc[3], c3, vld1q_f32(query.add(12)));
    }

    #[inline]
    unsafe fn reduce_add_neon(acc: [float32x4_t; 4]) -> f32 {
        vaddvq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3])))
    }

    #[target_feature(enable = "neon")]
    unsafe fn dot_u8_full_neon(code: &[u8], rot: &[f32]) -> f32 {
        let mut acc = [vdupq_n_f32(0.0); 4];
        let n = code.len();
        let full = n - (n % 16);
        let mut i = 0;
        while i < full {
            let codes = vld1q_u8(code.as_ptr().add(i));
            fma16_neon(codes, rot.as_ptr().add(i), &mut acc);
            i += 16;
        }
        let mut sum = reduce_add_neon(acc);
        for j in i..n {
            sum += code[j] as f32 * rot[j];
        }
        sum
    }

    #[target_feature(enable = "neon")]
    unsafe fn dot_u4_full_neon(code: &[u8], rot: &[f32]) -> f32 {
        let mut acc = [vdupq_n_f32(0.0); 4];
        let n_bytes = code.len();
        let full_bytes = n_bytes - (n_bytes % 16);
        let mask = vdupq_n_u8(0x0f);
        let mut i = 0;
        while i < full_bytes {
            let raw = vld1q_u8(code.as_ptr().add(i));
            let lo = vandq_u8(raw, mask);
            let hi = vshrq_n_u8::<4>(raw);
            // Interleave lo/hi nibbles into natural dim order (byte i holds
            // dims 2i and 2i+1).
            let zipped = vzipq_u8(lo, hi);
            fma16_neon(zipped.0, rot.as_ptr().add(i * 2), &mut acc);
            fma16_neon(zipped.1, rot.as_ptr().add(i * 2 + 16), &mut acc);
            i += 16;
        }
        let mut sum = reduce_add_neon(acc);
        for j in i..n_bytes {
            sum += (code[j] & 0x0F) as f32 * rot[j * 2];
            sum += (code[j] >> 4) as f32 * rot[j * 2 + 1];
        }
        sum
    }
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
        dot_with_rotated_fields(self.num_bits, self.dim, &self.packed_code, rot)
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
            2 => {
                // 1 sign bit + 1 ex bit per dim, packed 4 dims per byte
                // (bits 2j = sign, 2j+1 = ex).  Same Lance norm-aware ex
                // quantization as the 4/8-bit arms with ex_bits = 1:
                //   l1_of_rotated = ⟨rot, code⟩ (reconstruction dot)
                //   sum_of_x2     = ‖r‖²
                let ex_bits = 1u32;
                let max_code: u8 = 1;
                let mask: u8 = 0x1;
                let code_scale = 2.0f32;
                let code_bias = -1.5f32;
                let norm = sum_of_x2.sqrt().max(1e-9);
                let abs_normalized: Vec<f32> =
                    rotated.iter().map(|v| v.abs() / norm).collect();
                let t = Self::best_ex_rescale_factor(&abs_normalized, ex_bits);
                let mut code = vec![0u8; self.dim.div_ceil(4)];
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
                    // byte i/4: bit 2j = sign, bit 2j+1 = ex.
                    let j = (i % 4) as u8;
                    code[i / 4] |= (sign_bit << (2 * j)) | ((ex & mask) << (2 * j + 1));
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
    /// vector quantized, query kept in f32), returned as a lower bound so the
    /// executor's recheck (`xs_recheckorderby = true`) never sees the estimate
    /// overshoot the exact distance.
    ///
    /// L2 is translation-invariant, so the residual estimate is
    /// `‖ro‖² + ‖rq‖² - 2·(‖ro‖²/⟨ro,code⟩)·⟨code,rq⟩`; a conservative error
    /// margin `2·‖ro‖·‖rq‖/√D` is subtracted to make it a lower bound.
    #[inline]
    pub fn estimate_l2(&self, qv: &RabitqVector, rq: &RabitqQuery) -> f32 {
        self.estimate_l2_fields(
            qv.num_bits,
            qv.dim,
            &qv.packed_code,
            qv.sum_of_x2,
            qv.l1_of_rotated,
            rq,
        )
    }

    /// Lance-style L2 estimate given the primitive fields directly (used by the
    /// zero-copy archived scan path).  See `estimate_l2` for the derivation.
    #[inline]
    pub fn estimate_l2_fields(
        &self,
        num_bits: u8,
        dim: u32,
        packed_code: &[u8],
        sum_of_x2: f32,
        l1_of_rotated: f32,
        rq: &RabitqQuery,
    ) -> f32 {
        let full_dot = dot_with_rotated_fields(num_bits, dim, packed_code, &rq.rotated);
        let res_dot = l1_of_rotated.max(1e-9);
        let scale = -2.0 * sum_of_x2 / res_dot;
        let est = full_dot * scale + sum_of_x2 + rq.sum_of_x2;
        let margin = 2.0 * sum_of_x2.max(0.0).sqrt() * rq.sum_of_x2.max(0.0).sqrt()
            / (self.dim as f32).sqrt().max(1.0);
        (est - margin).max(0.0)
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

/// Precomputed per-(query, centroid) scan state.
///
/// For the 1-bit path it builds a FastScan distance table (`d/4 × 16`) so each
/// candidate's binary inner product is `d/4` 4-bit table lookups instead of a
/// scalar set-bit walk.  It also carries the query-side constants so the
/// per-entry L2 estimate (`estimate`) uses the build-time-precomputed `scale`
/// and `margin_factor` and performs no per-entry division or sqrt.
pub struct RabitqFastScan<'a> {
    rq: &'a RabitqQuery,
    num_bits: u8,
    dim: u32,
    /// u8-quantized `d/4 × 16` table for 1-bit FastScan (flat `d/4·16` bytes).
    table_u8: Vec<u8>,
    /// Dequantization: `sum_f32 = sum_u16·range_scale + num_chunks·qmin`.
    qmin: f32,
    range_scale: f32,
    num_chunks: usize,
    sum_rot: f32,
    rq_sum_of_x2: f32,
    /// `sqrt(max(rq.sum_of_x2, 0))` — the query-side half of the error margin.
    rq_margin: f32,
    /// Fused dequantize constants: `full_dot = a_full·sum + b_full`.
    a_full: f32,
    b_full: f32,
    /// 2-bit affine constants: `full_dot = a_2bit·(2·sum0 + sum1) + b_2bit`.
    a_2bit: f32,
    b_2bit: f32,
}

impl<'a> RabitqFastScan<'a> {
    pub fn new(rq: &'a RabitqQuery, num_bits: u8, dim: usize) -> Self {
        let (table_u8, qmin, range_scale, num_chunks) = if num_bits == 1 || num_bits == 2 {
            let f32_table = Self::build_table(&rq.rotated);
            let (q, qmin, rs) = crate::access_method::quantization::rabitq_fastscan::quantize_table(
                &f32_table,
            );
            (q, qmin, rs, f32_table.len() / 16)
        } else {
            (Vec::new(), 0.0, 0.0, 0)
        };
        // `full_dot = 2·(sum_u16·range_scale + num_chunks·qmin) − sum_rot`
        //        = (2·range_scale)·sum + (2·num_chunks·qmin − sum_rot).
        let a_full = 2.0 * range_scale;
        let b_full = 2.0 * num_chunks as f32 * qmin - rq.sum_q;
        // 2-bit: dot = 2·m + e − 1.5·Σrot with m/e the two plane sums, each
        // dequantized as sum·range_scale + num_chunks·qmin:
        //   = (2·s0 + s1)·range_scale + 3·num_chunks·qmin − 1.5·Σrot.
        let a_2bit = range_scale;
        let b_2bit = 3.0 * num_chunks as f32 * qmin - 1.5 * rq.sum_q;
        Self {
            rq,
            num_bits,
            dim: dim as u32,
            table_u8,
            qmin,
            range_scale,
            num_chunks,
            sum_rot: rq.sum_q,
            rq_sum_of_x2: rq.sum_of_x2,
            rq_margin: rq.sum_of_x2.max(0.0).sqrt(),
            a_full,
            b_full,
            a_2bit,
            b_2bit,
        }
    }

    fn build_table(rot: &[f32]) -> Vec<f32> {
        let n_chunks = rot.len() / 4;
        let mut table = vec![0.0f32; n_chunks * 16];
        for i in 0..n_chunks {
            let base = i * 16;
            for j in 0..16u32 {
                let mut s = 0.0f32;
                for k in 0..4 {
                    if j & (1 << k) != 0 {
                        s += rot[i * 4 + k];
                    }
                }
                table[base + j as usize] = s;
            }
        }
        table
    }

    /// Bytes per 1-bit code (`dim / 8`).
    #[inline]
    pub fn code_len(&self) -> usize {
        self.dim as usize / 8
    }

    /// Sum the quantized table for one 32-row transposed batch (1-bit).
    #[inline]
    pub fn sum_batch(&self, batch_codes: &[u8], out: &mut [u16]) {
        crate::access_method::quantization::rabitq_fastscan::sum_batch(
            batch_codes,
            self.code_len(),
            &self.table_u8,
            out,
        );
    }

    /// Fused estimate for a batch of 1-bit rows (dequantize + full_dot + L2),
    /// SIMD over 4/8 rows.
    #[inline]
    pub fn estimate_batch(
        &self,
        sums: &[u16],
        scales: &[f32],
        sx2: &[f32],
        mf: &[f32],
        out: &mut [f32],
        n: usize,
    ) {
        crate::access_method::quantization::rabitq_fastscan::estimate_batch(
            sums,
            scales,
            sx2,
            mf,
            self.a_full,
            self.b_full,
            self.rq_sum_of_x2,
            self.rq_margin,
            out,
            n,
        );
    }

    /// Fused estimate for a batch of 2-bit rows from the two plane sums
    /// (dequantize + combine + L2), SIMD over 4/8 rows.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn estimate_batch_2bit(
        &self,
        sums0: &[u16],
        sums1: &[u16],
        scales: &[f32],
        sx2: &[f32],
        mf: &[f32],
        out: &mut [f32],
        n: usize,
    ) {
        crate::access_method::quantization::rabitq_fastscan::estimate_batch_2bit(
            sums0,
            sums1,
            scales,
            sx2,
            mf,
            self.a_2bit,
            self.b_2bit,
            self.rq_sum_of_x2,
            self.rq_margin,
            out,
            n,
        );
    }

    /// Dequantize a u16 FastScan sum to the f32 `Σ_{set bits} rot`.
    #[inline]
    pub fn dequantize_sum(&self, q: u16) -> f32 {
        q as f32 * self.range_scale + self.num_chunks as f32 * self.qmin
    }

    /// Full binary dot for a dequantized 1-bit `sum_set`.
    #[inline]
    pub fn full_dot_1bit(&self, sum_set: f32) -> f32 {
        2.0 * sum_set - self.sum_rot
    }

    /// Full binary dot for a 4/8-bit code (row-major).
    #[inline]
    pub fn full_dot_multi(&self, code: &[u8]) -> f32 {
        let code_bias = -((1u32 << (self.num_bits - 1)) as f32 - 0.5);
        dot_full_code(self.num_bits, code, &self.rq.rotated) + code_bias * self.sum_rot
    }

    /// Lower-bounded L2 estimate from a full dot and the precomputed per-entry
    /// factors.  No division or sqrt in the hot loop.
    #[inline]
    pub fn estimate_from_full_dot(
        &self,
        full_dot: f32,
        sum_of_x2: f32,
        scale: f32,
        margin_factor: f32,
    ) -> f32 {
        let est = full_dot * scale + sum_of_x2 + self.rq_sum_of_x2;
        (est - margin_factor * self.rq_margin).max(0.0)
    }
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
    fn fastscan_matches_estimate_l2_fields() {
        let q = RabitqQuantizer::new(1, 42, 128);
        let v: Vec<f32> = (0..128).map(|i| ((i % 7) as f32) - 3.0).collect();
        let qv = q.quantize(&v);
        let rq = q.rotate_query(&v);
        let fastscan = RabitqFastScan::new(&rq, 1, q.dim());

        // The 1-bit FastScan (transposed batch sum + dequantize) full_dot must
        // equal the scalar set-bit walk.
        let transposed = crate::access_method::quantization::rabitq_fastscan::transpose_1bit(
            &qv.packed_code,
            1,
            fastscan.code_len(),
        );
        let mut sums = [0u16; 32];
        fastscan.sum_batch(&transposed, &mut sums);
        let sum_set = fastscan.dequantize_sum(sums[0]);
        let expected_dot = qv.dot_with_rotated(&rq.rotated);
        let actual_dot = fastscan.full_dot_1bit(sum_set);
        // The u8 table quantization introduces a small sum error (bounded by
        // num_chunks·range/255/2), so use an absolute tolerance.
        assert!(
            (actual_dot - expected_dot).abs() <= 0.5,
            "dot {} vs {}",
            actual_dot,
            expected_dot
        );

        // The precomputed-factors estimate must equal estimate_l2_fields.
        let scale = -2.0 * qv.sum_of_x2 / qv.l1_of_rotated.max(1e-9);
        let margin_factor = 2.0 * qv.sum_of_x2.max(0.0).sqrt() / (q.dim() as f32).sqrt().max(1.0);
        let expected = q.estimate_l2(&qv, &rq);
        let actual = fastscan.estimate_from_full_dot(
            actual_dot,
            qv.sum_of_x2,
            scale,
            margin_factor,
        );
        assert!(
            (actual - expected).abs() <= 1.0,
            "estimate {} vs {}",
            actual,
            expected
        );
    }

    /// The multi-bit full_dot must equal Σ (unpacked_code − bias) · rot, i.e. the
    /// binary sign bit is unsigned 0/1 (not ±1).  This guards the bug that made
    /// 4/8-bit ranking collapse.
    #[test]
    fn multi_bit_dot_matches_unpacked_reference() {
        for num_bits in [4u8, 8u8] {
            let ex_bits = (num_bits - 1) as u32;
            let code_bias = -((1u32 << ex_bits) as f32 - 0.5);
            let dim = 128usize;
            let q = RabitqQuantizer::new(num_bits, 7, dim);
            let v: Vec<f32> = (0..dim).map(|i| ((i % 11) as f32) - 5.0).collect();
            let qv = q.quantize(&v);
            let rot: Vec<f32> = (0..dim).map(|i| ((i % 9) as f32) - 4.0).collect();

            let full_dot = dot_with_rotated_fields(num_bits, dim as u32, &qv.packed_code, &rot);

            // Unpack the code and compute the reference Σ (full_code + bias)·rot.
            let mut reference = 0.0f32;
            for (i, &b) in qv.packed_code.iter().enumerate() {
                if num_bits == 4 {
                    let lo = (b & 0x0F) as f32;
                    let hi = (b >> 4) as f32;
                    reference += (lo + code_bias) * rot[i * 2];
                    reference += (hi + code_bias) * rot[i * 2 + 1];
                } else {
                    // 8-bit: one dim per byte, full_code = b (0..255).
                    reference += (b as f32 + code_bias) * rot[i];
                }
            }
            assert!(
                (full_dot - reference).abs() <= 1e-3 * reference.abs().max(1.0),
                "num_bits={} dot {} vs {}",
                num_bits,
                full_dot,
                reference
            );
        }
    }

    /// The SIMD/dispatched `dot_full_code` (via `RabitqFastScan::full_dot`)
    /// must agree with the scalar `dot_with_rotated_fields` for multi-bit codes.
    #[test]
    fn fastscan_multi_bit_full_dot_matches_scalar() {
        for num_bits in [4u8, 8u8] {
            let dim = 128usize;
            let q = RabitqQuantizer::new(num_bits, 7, dim);
            let v: Vec<f32> = (0..dim).map(|i| ((i % 11) as f32) - 5.0).collect();
            let qv = q.quantize(&v);
            let rq = q.rotate_query(&v);
            let fastscan = RabitqFastScan::new(&rq, num_bits, q.dim());
            let actual = fastscan.full_dot_multi(&qv.packed_code);
            let expected =
                dot_with_rotated_fields(num_bits, dim as u32, &qv.packed_code, &rq.rotated);
            assert!(
                (actual - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                "num_bits={} {} vs {}",
                num_bits,
                actual,
                expected
            );
        }
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

#[cfg(test)]
mod two_bit_tests {
    use super::*;

    #[test]
    fn two_bit_dot_matches_unpacked_reference() {
        // Unpack the 2-bit code and compare dot_with_rotated_fields against
        // Σ (full_code − 1.5)·rot (guards the sign-0/1 convention).
        let dim = 128usize;
        let q = RabitqQuantizer::new(2, 7, dim);
        let v: Vec<f32> = (0..dim).map(|i| ((i % 11) as f32) - 5.0).collect();
        let qv = q.quantize(&v);
        assert_eq!(qv.packed_code.len(), dim / 4, "2-bit packs 4 dims per byte");
        let rot: Vec<f32> = (0..dim).map(|i| ((i % 9) as f32) - 4.0).collect();

        let full_dot = dot_with_rotated_fields(2, dim as u32, &qv.packed_code, &rot);

        let mut reference = 0.0f32;
        for (byte_idx, &b) in qv.packed_code.iter().enumerate() {
            for j in 0..4usize {
                let d = byte_idx * 4 + j;
                let sign = (b >> (2 * j)) & 1;
                let ex = (b >> (2 * j + 1)) & 1;
                let full_code = ((sign as u32) << 1) + ex as u32;
                reference += (full_code as f32 - 1.5) * rot[d];
            }
        }
        assert!(
            (full_dot - reference).abs() <= 1e-3 * reference.abs().max(1.0),
            "dot {} vs {}",
            full_dot,
            reference
        );
    }

    #[test]
    fn two_bit_fastscan_matches_estimate_l2_fields() {
        // The two-plane FastScan estimate must agree with the scalar
        // estimate_l2_fields path (within the u8-table tolerance).
        let dim = 128usize;
        let q = RabitqQuantizer::new(2, 42, dim);
        let v: Vec<f32> = (0..dim).map(|i| ((i % 7) as f32) - 3.0).collect();
        let qv = q.quantize(&v);
        let rq = q.rotate_query(&v);
        let fastscan = RabitqFastScan::new(&rq, 2, q.dim());

        // Split into planes (mirroring serialize_entries) and transpose one row.
        let mut sign_plane = vec![0u8; dim / 8];
        let mut ex_plane = vec![0u8; dim / 8];
        for (byte_idx, &b) in qv.packed_code.iter().enumerate() {
            for j in 0..4usize {
                let d = byte_idx * 4 + j;
                if b & (1 << (2 * j)) != 0 {
                    sign_plane[d / 8] |= 1 << (d % 8);
                }
                if b & (1 << (2 * j + 1)) != 0 {
                    ex_plane[d / 8] |= 1 << (d % 8);
                }
            }
        }
        let p0 = crate::access_method::quantization::rabitq_fastscan::transpose_1bit(
            &sign_plane, 1, dim / 8,
        );
        let p1 = crate::access_method::quantization::rabitq_fastscan::transpose_1bit(
            &ex_plane, 1, dim / 8,
        );
        let mut s0 = [0u16; 32];
        let mut s1 = [0u16; 32];
        fastscan.sum_batch(&p0, &mut s0);
        fastscan.sum_batch(&p1, &mut s1);
        let scale = -2.0 * qv.sum_of_x2 / qv.l1_of_rotated.max(1e-9);
        let margin_factor =
            2.0 * qv.sum_of_x2.max(0.0).sqrt() / (q.dim() as f32).sqrt().max(1.0);
        let mut out = [0f32; 32];
        fastscan.estimate_batch_2bit(&s0, &s1, &[scale], &[qv.sum_of_x2], &[margin_factor], &mut out, 1);

        let expected = q.estimate_l2(&qv, &rq);
        assert!(
            (out[0] - expected).abs() <= 1.0,
            "fastscan 2-bit estimate {} vs scalar {}",
            out[0],
            expected
        );
    }
}
