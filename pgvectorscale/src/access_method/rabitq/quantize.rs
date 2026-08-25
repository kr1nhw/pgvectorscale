use crate::access_method::{distance::DistanceType, meta_page::MetaPage};

use super::rotation::apply_fast_rotation;
use super::RabitqVectorElement;

/// Number of bits per packed sign-code word.
pub const BITS_STORE_TYPE_SIZE: usize = 64;

const EX_QUANTIZATION_EPSILON: f64 = 1e-5;
const EX_TIGHT_START: [f64; 9] = [0.0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81];

/// The compact per-node RaBitQ representation.
#[derive(Clone)]
pub struct RabitqCode {
    /// 1-bit sign codes, packed LSB-first into u64 words (bit i = dimension i).
    pub code: Vec<RabitqVectorElement>,
    /// Error-correction codes. Empty for 1-bit RaBitQ; nibble-packed (2 dims/byte)
    /// for 4-bit (ex_bits=3) and byte-packed for 8-bit (ex_bits=7).
    pub ex_code: Vec<u8>,
    /// Affine add factor for the 1-bit estimator.
    pub f_add: f32,
    /// Affine scale factor for the 1-bit estimator.
    pub f_rescale: f32,
    /// Affine add factor for the multi-bit (ex-code) estimator.
    pub f_add_ex: f32,
    /// Affine scale factor for the multi-bit (ex-code) estimator.
    pub f_rescale_ex: f32,
}

/// Per-query RaBitQ state, computed once at query time.
pub struct RabitqQueryMeasure {
    /// Continuous rotated query residual `R(q - c)`.
    pub rotated_query: Vec<f32>,
    /// Sum of the rotated query residual.
    pub sum_q: f32,
    /// Query-dependent add factor (`‖q-c‖²` for L2, `0.5·‖q-c‖²` for cosine,
    /// `-⟨q-c, c⟩` for inner product).
    pub g_add: f32,
}

/// RaBitQ quantizer: global-mean centering + fast random rotation + sign/ex-code
/// quantization, with the Lance/RaBitQ affine distance estimator.
#[derive(Clone)]
pub struct RabitqQuantizer {
    distance_type: DistanceType,
    training: bool,
    pub count: u64,
    pub mean: Vec<f32>,
    pub rotation_signs: Vec<u8>,
    /// Total bits per dimension: 1, 4, or 8. `ex_bits = num_bits - 1`.
    pub num_bits: u8,
    ex_bits: u8,
}

impl RabitqQuantizer {
    pub fn new(meta_page: &MetaPage, num_bits: u8) -> Self {
        Self::new_with_distance_type(meta_page.get_distance_type(), num_bits)
    }

    pub fn new_with_distance_type(distance_type: DistanceType, num_bits: u8) -> Self {
        assert!(
            matches!(num_bits, 1 | 4 | 8),
            "RaBitQ num_bits must be 1, 4 or 8"
        );
        Self {
            distance_type,
            training: false,
            count: 0,
            mean: vec![],
            rotation_signs: vec![],
            num_bits,
            ex_bits: num_bits - 1,
        }
    }

    pub fn load(&mut self, count: u64, mean: Vec<f32>, rotation_signs: Vec<u8>, num_bits: u8) {
        self.count = count;
        self.mean = mean;
        self.rotation_signs = rotation_signs;
        self.num_bits = num_bits;
        self.ex_bits = num_bits - 1;
    }

    pub fn set_rotation_signs(&mut self, signs: Vec<u8>) {
        self.rotation_signs = signs;
    }

    pub fn get_ex_bits(&self) -> u8 {
        self.ex_bits
    }

    pub fn start_training(&mut self, meta_page: &MetaPage) {
        self.training = true;
        self.count = 0;
        self.mean = vec![0.0; meta_page.get_num_dimensions_to_index() as usize];
    }

    pub fn add_sample(&mut self, sample: &[f32]) {
        self.count += 1;
        assert!(self.mean.len() == sample.len());
        self.mean
            .iter_mut()
            .zip(sample.iter())
            .for_each(|(m, s)| *m += (s - *m) / self.count as f32);
    }

    pub fn finish_training(&mut self) {
        self.training = false;
    }

    /// Quantize a full (indexed) vector into its compact node representation.
    pub fn quantize(&self, full_vector: &[f32]) -> RabitqCode {
        assert!(!self.training);
        let dim = full_vector.len();
        assert!(self.mean.len() == dim);

        let residual: Vec<f32> = full_vector
            .iter()
            .zip(self.mean.iter())
            .map(|(v, m)| v - m)
            .collect();
        let l2_sqr: f32 = residual.iter().map(|v| v * v).sum();

        let mut rotated = vec![0.0f32; dim];
        apply_fast_rotation(&residual, &mut rotated, &self.rotation_signs);

        let code = pack_sign_bits(&rotated);
        let l1_rot: f32 = rotated.iter().map(|v| v.abs()).sum();
        let denom = 0.5 * l1_rot;

        let (ex_code, ipnorm_inv) = if self.ex_bits > 0 {
            quantize_ex_code(&rotated, self.ex_bits)
        } else {
            (Vec::new(), 1.0f32)
        };

        let (f_add, f_rescale) = self.one_bit_factors(l2_sqr, denom);
        let (f_add_ex, f_rescale_ex) = self.ex_factors(l2_sqr, ipnorm_inv);

        RabitqCode {
            code,
            ex_code,
            f_add,
            f_rescale,
            f_add_ex,
            f_rescale_ex,
        }
    }

    /// Build the per-query measure from a query vector (already pre-processed,
    /// e.g. normalized for cosine, by the caller).
    pub fn query_measure(&self, query: &[f32]) -> RabitqQueryMeasure {
        let dim = query.len();
        assert!(self.mean.len() == dim);

        let residual: Vec<f32> = query
            .iter()
            .zip(self.mean.iter())
            .map(|(q, m)| q - m)
            .collect();
        let l2_sqr_q: f32 = residual.iter().map(|v| v * v).sum();

        let mut rotated = vec![0.0f32; dim];
        apply_fast_rotation(&residual, &mut rotated, &self.rotation_signs);
        let sum_q: f32 = rotated.iter().sum();

        // For graph traversal, always use a non-negative L2-style estimate.
        // Inner-product uses the L2 estimate as a surrogate (the exact IP
        // distance is computed in the resort phase); cosine is 0.5 * L2.
        let g_add = match self.distance_type {
            DistanceType::L2 | DistanceType::InnerProduct => l2_sqr_q,
            DistanceType::Cosine => 0.5 * l2_sqr_q,
        };

        RabitqQueryMeasure {
            rotated_query: rotated,
            sum_q,
            g_add,
        }
    }

    /// Estimate the distance between a query and a node from their quantized codes.
    /// The result is clamped to be non-negative (estimates can undershoot slightly
    /// below zero for near-identical vectors).
    #[inline]
    pub fn estimate_distance(&self, qm: &RabitqQueryMeasure, code: &RabitqCode) -> f32 {
        let binary_dot = binary_dot(&code.code, &qm.rotated_query);
        let dist = if self.ex_bits == 0 {
            let binary_term = binary_dot + (-0.5) * qm.sum_q;
            qm.g_add + code.f_add + code.f_rescale * binary_term
        } else {
            let ex_dot = ex_dot(&code.ex_code, &qm.rotated_query, self.ex_bits);
            let cb = -((1u32 << self.ex_bits) as f32 - 0.5);
            let code_scale = (1u32 << self.ex_bits) as f32;
            let total_term = code_scale * binary_dot + ex_dot + cb * qm.sum_q;
            qm.g_add + code.f_add_ex + code.f_rescale_ex * total_term
        };
        dist.max(0.0)
    }

    fn one_bit_factors(&self, l2_sqr: f32, denom: f32) -> (f32, f32) {
        let rescale_denom = if denom.abs() <= f32::EPSILON {
            f32::INFINITY
        } else {
            denom
        };
        match self.distance_type {
            DistanceType::L2 | DistanceType::InnerProduct => {
                (l2_sqr, -2.0 * l2_sqr / rescale_denom)
            }
            DistanceType::Cosine => (0.5 * l2_sqr, -l2_sqr / rescale_denom),
        }
    }

    fn ex_factors(&self, l2_sqr: f32, ipnorm_inv: f32) -> (f32, f32) {
        let l2_norm = l2_sqr.sqrt();
        match self.distance_type {
            DistanceType::L2 | DistanceType::InnerProduct => (l2_sqr, -2.0 * l2_norm * ipnorm_inv),
            DistanceType::Cosine => (0.5 * l2_sqr, -l2_norm * ipnorm_inv),
        }
    }
}

/// Pack the sign bits of a rotated vector into u64 words, LSB-first.
#[inline]
pub fn pack_sign_bits(rotated: &[f32]) -> Vec<u64> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                return pack_sign_bits_avx2(rotated);
            }
        }
    }
    pack_sign_bits_scalar(rotated)
}

#[inline]
fn pack_sign_bits_scalar(rotated: &[f32]) -> Vec<u64> {
    let num_words = rotated.len().div_ceil(BITS_STORE_TYPE_SIZE);
    let mut code = vec![0u64; num_words];
    for (i, &v) in rotated.iter().enumerate() {
        if v >= 0.0 {
            code[i / BITS_STORE_TYPE_SIZE] |= 1u64 << (i % BITS_STORE_TYPE_SIZE);
        }
    }
    code
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn pack_sign_bits_avx2(rotated: &[f32]) -> Vec<u64> {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let num_words = rotated.len().div_ceil(BITS_STORE_TYPE_SIZE);
    let mut code = vec![0u64; num_words];
    let full_chunks = rotated.len() / 8;

    for chunk in 0..full_chunks {
        let ptr = rotated.as_ptr().add(chunk * 8);
        let v = _mm256_loadu_ps(ptr);
        // `movemask_ps` sets bit i for negative lanes; invert for non-negative.
        let mask = (!_mm256_movemask_ps(v)) & 0xFF;
        code[chunk / 8] |= (mask as u64) << ((chunk % 8) * 8);
    }

    for i in (full_chunks * 8)..rotated.len() {
        if rotated[i] >= 0.0 {
            code[i / BITS_STORE_TYPE_SIZE] |= 1u64 << (i % BITS_STORE_TYPE_SIZE);
        }
    }
    code
}

/// `Σ bit[i] * q[i]` — the dot product of a node's binary sign code with the
/// continuous rotated query.
#[inline]
pub fn binary_dot(code: &[u64], q: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                return binary_dot_avx2(code, q);
            }
        }
    }
    binary_dot_scalar(code, q)
}

#[inline]
fn binary_dot_scalar(code: &[u64], q: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (word_idx, &word) in code.iter().enumerate() {
        let base = word_idx * BITS_STORE_TYPE_SIZE;
        let mut w = word;
        while w != 0 {
            let i = w.trailing_zeros() as usize;
            let idx = base + i;
            if idx < q.len() {
                sum += q[idx];
            }
            w &= w - 1;
        }
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn binary_dot_avx2(code: &[u64], q: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let sel = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    let mut accum = _mm256_setzero_ps();

    for (word_idx, &word) in code.iter().enumerate() {
        for chunk in 0..8 {
            let base = word_idx * 64 + chunk * 8;
            if base >= q.len() {
                break;
            }
            let bits = ((word >> (chunk * 8)) & 0xFF) as i32;
            let v = _mm256_set1_epi32(bits);
            let t = _mm256_and_si256(v, sel);
            let cmp = _mm256_cmpeq_epi32(t, sel); // all-ones where the bit is set
            let qv = _mm256_loadu_ps(q.as_ptr().add(base));
            let masked = _mm256_and_ps(qv, _mm256_castsi256_ps(cmp));
            accum = _mm256_add_ps(accum, masked);
        }
    }

    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(out.as_mut_ptr(), accum);
    out.iter().sum()
}

/// `Σ ex[i] * q[i]` for packed ex codes.
#[inline]
pub fn ex_dot(ex_code: &[u8], q: &[f32], ex_bits: u8) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if ex_bits == 7 && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                return ex_dot_7bit_avx2(ex_code, q);
            }
        }
    }
    ex_dot_scalar(ex_code, q, ex_bits)
}

#[inline]
fn ex_dot_scalar(ex_code: &[u8], q: &[f32], ex_bits: u8) -> f32 {
    let mut sum = 0.0f32;
    match ex_bits {
        7 => {
            for i in 0..q.len() {
                sum += ex_code[i] as f32 * q[i];
            }
        }
        3 => {
            for i in 0..q.len() {
                let byte = ex_code[i / 2];
                let val = (byte >> ((i % 2) * 4)) & 0xF;
                sum += val as f32 * q[i];
            }
        }
        _ => unreachable!("only 3 or 7 ex bits are supported"),
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn ex_dot_7bit_avx2(ex_code: &[u8], q: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut accum = _mm256_setzero_ps();
    let mut i = 0usize;
    let n = q.len();

    while i + 8 <= n {
        let b = _mm_loadl_epi64(ex_code.as_ptr().add(i) as *const _);
        let b32 = _mm256_cvtepu8_epi32(b);
        let bf = _mm256_cvtepi32_ps(b32);
        let qv = _mm256_loadu_ps(q.as_ptr().add(i));
        accum = _mm256_fmadd_ps(bf, qv, accum);
        i += 8;
    }

    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(out.as_mut_ptr(), accum);
    let mut sum: f32 = out.iter().sum();
    for j in i..n {
        sum += ex_code[j] as f32 * q[j];
    }
    sum
}

fn ex_code_packed_len(dim: usize, ex_bits: u8) -> usize {
    match ex_bits {
        3 => dim.div_ceil(2),
        7 => dim,
        _ => unreachable!("only 3 or 7 ex bits are supported"),
    }
}

fn pack_ex_code(ex_code: &[u8], ex_bits: u8) -> Vec<u8> {
    match ex_bits {
        3 => {
            let mut out = vec![0u8; ex_code.len().div_ceil(2)];
            for (i, &c) in ex_code.iter().enumerate() {
                out[i / 2] |= c << ((i % 2) * 4);
            }
            out
        }
        7 => ex_code.to_vec(),
        _ => unreachable!("only 3 or 7 ex bits are supported"),
    }
}

/// Quantize the residual magnitudes into `2^ex_bits` error-correction levels,
/// folding the sign into the low bits (Lance / RaBitQ `quantize_ex_code`).
/// Returns `(packed_ex_code, ipnorm_inv)`.
fn quantize_ex_code(rotated: &[f32], ex_bits: u8) -> (Vec<u8>, f32) {
    let dim = rotated.len();
    let norm_sq: f32 = rotated.iter().map(|v| v * v).sum();
    let norm = norm_sq.sqrt();
    if norm <= f32::EPSILON || !norm.is_finite() {
        return (vec![0u8; ex_code_packed_len(dim, ex_bits)], 1.0);
    }

    let abs_normalized: Vec<f32> = rotated.iter().map(|v| v.abs() / norm).collect();
    let t = best_rescale_factor(&abs_normalized, ex_bits);

    let max_val = ((1u16 << ex_bits) - 1) as u8;
    let mask = max_val;
    let mut ex_code = vec![0u8; dim];
    let mut ipnorm = 0.0f64;

    for i in 0..dim {
        let mut cur = (t * abs_normalized[i] as f64 + EX_QUANTIZATION_EPSILON) as i32;
        if cur > max_val as i32 {
            cur = max_val as i32;
        }
        let mut code = cur as u8;
        ipnorm += (cur as f64 + 0.5) * abs_normalized[i] as f64;
        if rotated[i] < 0.0 {
            code = (!code) & mask;
        }
        ex_code[i] = code;
    }

    let ipnorm_inv = if ipnorm.is_finite() && ipnorm > 0.0 {
        (1.0 / ipnorm) as f32
    } else {
        1.0
    };

    (pack_ex_code(&ex_code, ex_bits), ipnorm_inv)
}

/// Find the rescale factor `t` that maximizes the alignment between the residual
/// and its quantized code (ported from rabitq-rs / Lance `best_ex_rescale_factor`).
fn best_rescale_factor(abs_normalized: &[f32], ex_bits: u8) -> f64 {
    let dim = abs_normalized.len();
    let max_value = abs_normalized
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0f32, f32::max) as f64;
    if max_value <= f64::EPSILON {
        return 1.0;
    }

    let table_idx = (ex_bits as usize).min(EX_TIGHT_START.len() - 1);
    let t_end = (((1usize << ex_bits) - 1) as f64 + 10.0) / max_value;
    let t_start = t_end * EX_TIGHT_START[table_idx];

    let mut cur_o_bar = vec![0i32; dim];
    let mut sqr_denominator = dim as f64 * 0.25;
    let mut numerator = 0.0f64;

    let mut heap = std::collections::BinaryHeap::<std::cmp::Reverse<(u64, usize)>>::new();
    for (idx, &val) in abs_normalized.iter().enumerate() {
        let cur = ((t_start * val as f64) + EX_QUANTIZATION_EPSILON) as i32;
        cur_o_bar[idx] = cur;
        sqr_denominator += (cur * cur + cur) as f64;
        numerator += (cur as f64 + 0.5) * val as f64;
        if val > 0.0 {
            let next_t = (cur + 1) as f64 / val as f64;
            // encode the (f64) threshold into a sortable u64 key for the min-heap
            heap.push(std::cmp::Reverse((next_t.to_bits(), idx)));
        }
    }

    let mut max_ip = 0.0f64;
    let mut best_t = t_start;

    while let Some(std::cmp::Reverse((t_bits, idx))) = heap.pop() {
        let cur_t = f64::from_bits(t_bits);
        if cur_t >= t_end {
            continue;
        }
        cur_o_bar[idx] += 1;
        let update = cur_o_bar[idx];
        sqr_denominator += 2.0 * update as f64;
        numerator += abs_normalized[idx] as f64;

        let cur_ip = numerator / sqr_denominator.sqrt();
        if cur_ip > max_ip {
            max_ip = cur_ip;
            best_t = cur_t;
        }

        if update < (1i32 << ex_bits) - 1 && abs_normalized[idx] > 0.0 {
            let t_next = (update + 1) as f64 / abs_normalized[idx] as f64;
            if t_next < t_end {
                heap.push(std::cmp::Reverse((t_next.to_bits(), idx)));
            }
        }
    }

    if best_t <= 0.0 {
        t_start.max(f64::EPSILON)
    } else {
        best_t
    }
}
