//! Lance-style FastScan for 1-bit RaBitQ codes.
//!
//! The 1-bit binary inner product is `Σ_d sign(code_d)·rot_d`.  Instead of
//! walking set bits per candidate, we precompute a `d/4 × 16` distance table
//! (each 4-bit nibble → the partial sum over that chunk's 4 rotated-query
//! components) and reduce each candidate to `d/4` table lookups.
//!
//! To make those lookups SIMD (NEON `vqtbl`/AVX2 `pshufb`), the 1-bit codes are
//! stored *transposed*: batched by 32 rows, chunk-major, two rows per byte.
//! One 16-entry table lookup then serves 16 rows at once.  The f32 table is
//! quantized to u8 and the per-row sums dequantized affinely.

/// Rows processed per transposed batch.
pub const BATCH_SIZE: usize = 32;

/// Maps a byte position (0..16) in a transposed chunk to the row index whose
/// code occupies that byte's low nibble (the +16 row is in the high nibble).
pub const PERM0: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];

/// Quantize a flat `d/4 × 16` f32 distance table to u8.
///
/// Returns `(quantized, qmin, range_scale)` so that a per-row u16 sum of
/// `num_chunks` lookups dequantizes as `sum·range_scale + num_chunks·qmin`.
pub fn quantize_table(table: &[f32]) -> (Vec<u8>, f32, f32) {
    let qmin = table.iter().copied().fold(f32::INFINITY, f32::min);
    let qmax = table
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let range = qmax - qmin;
    let factor = if range > 0.0 { 255.0 / range } else { 0.0 };
    let q = table
        .iter()
        .map(|&v| ((v - qmin) * factor).round() as u8)
        .collect();
    (q, qmin, range / 255.0)
}

/// Transpose row-major 1-bit codes (`n_rows × code_len` bytes) into the
/// batched/transposed layout, zero-padding the final partial batch.
pub fn transpose_1bit(codes: &[u8], n_rows: usize, code_len: usize) -> Vec<u8> {
    let n_batches = n_rows.div_ceil(BATCH_SIZE);
    let mut out = vec![0u8; n_batches * BATCH_SIZE * code_len];
    for batch in 0..n_batches {
        let row_base = batch * BATCH_SIZE;
        for c in 0..code_len {
            let out_base = (batch * code_len + c) * BATCH_SIZE;
            for j in 0..16 {
                let lo_row = row_base + PERM0[j];
                let hi_row = row_base + PERM0[j] + 16;
                let lo_byte = if lo_row < n_rows {
                    codes[lo_row * code_len + c]
                } else {
                    0
                };
                let hi_byte = if hi_row < n_rows {
                    codes[hi_row * code_len + c]
                } else {
                    0
                };
                // Byte j: chunk 2c in low nibble, chunk 2c+1 in the next 16 bytes.
                out[out_base + j] = (lo_byte & 0x0F) | ((hi_byte & 0x0F) << 4);
                out[out_base + 16 + j] = (lo_byte >> 4) | ((hi_byte >> 4) << 4);
            }
        }
    }
    out
}

/// Inverse of [`transpose_1bit`]: recover row-major codes from the transposed
/// layout (used by the insert/vacuum rewrite path).
pub fn untranspose_1bit(transposed: &[u8], n_rows: usize, code_len: usize) -> Vec<u8> {
    let n_batches = n_rows.div_ceil(BATCH_SIZE);
    let mut out = vec![0u8; n_rows * code_len];
    for batch in 0..n_batches {
        let row_base = batch * BATCH_SIZE;
        for c in 0..code_len {
            let in_base = (batch * code_len + c) * BATCH_SIZE;
            for j in 0..16 {
                let current = transposed[in_base + j]; // chunk 2c
                let next = transposed[in_base + 16 + j]; // chunk 2c+1
                let lo_row = row_base + PERM0[j];
                let hi_row = row_base + PERM0[j] + 16;
                let lo_byte = (current & 0x0F) | ((next & 0x0F) << 4);
                let hi_byte = (current >> 4) | ((next >> 4) << 4);
                if lo_row < n_rows {
                    out[lo_row * code_len + c] = lo_byte;
                }
                if hi_row < n_rows {
                    out[hi_row * code_len + c] = hi_byte;
                }
            }
        }
    }
    out
}

/// Fused L2 estimate for a batch of rows, from the precomputed FastScan sums.
///
/// With `full_dot = a_full·sum + b_full`, the lower-bounded estimate is
/// `(a_full·scale·sum + b_full·scale + sum_of_x2 + rq_sum − margin_factor·rq_margin).max(0)`.
/// `sums` is `n` u16 FastScan sums; `scales`/`sx2`/`mf` are the per-entry
/// precomputed factors; `out` receives `n` estimates.
#[inline]
pub fn estimate_batch(
    sums: &[u16],
    scales: &[f32],
    sx2: &[f32],
    mf: &[f32],
    a_full: f32,
    b_full: f32,
    rq_sum: f32,
    rq_margin: f32,
    out: &mut [f32],
    n: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            // SAFETY: only selected when the AVX-512 families were detected.
            unsafe {
                estimate_batch_avx512(sums, scales, sx2, mf, a_full, b_full, rq_sum, rq_margin, out, n);
                return;
            }
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: only selected when AVX2 was detected.
            unsafe {
                estimate_batch_avx2(sums, scales, sx2, mf, a_full, b_full, rq_sum, rq_margin, out, n);
                return;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is baseline on aarch64.
        unsafe {
            estimate_batch_neon(sums, scales, sx2, mf, a_full, b_full, rq_sum, rq_margin, out, n);
            return;
        }
    }
    estimate_batch_scalar(sums, scales, sx2, mf, a_full, b_full, rq_sum, rq_margin, out, n);
}

#[inline]
fn estimate_batch_scalar(
    sums: &[u16],
    scales: &[f32],
    sx2: &[f32],
    mf: &[f32],
    a_full: f32,
    b_full: f32,
    rq_sum: f32,
    rq_margin: f32,
    out: &mut [f32],
    n: usize,
) {
    for i in 0..n {
        let sum = sums[i] as f32;
        let scale = scales[i];
        let d = (a_full * scale * sum + b_full * scale + rq_sum + sx2[i] - mf[i] * rq_margin)
            .max(0.0);
        out[i] = d;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn estimate_batch_neon(
    sums: &[u16],
    scales: &[f32],
    sx2: &[f32],
    mf: &[f32],
    a_full: f32,
    b_full: f32,
    rq_sum: f32,
    rq_margin: f32,
    out: &mut [f32],
    n: usize,
) {
    use std::arch::aarch64::*;
    let rq_sum_v = vdupq_n_f32(rq_sum);
    let rq_margin_v = vdupq_n_f32(rq_margin);
    let a_v = vdupq_n_f32(a_full);
    let b_v = vdupq_n_f32(b_full);
    let zero = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= n {
        let sums_v = vcvtq_f32_u32(vmovl_u16(vld1_u16(sums.as_ptr().add(i))));
        let scale_v = vld1q_f32(scales.as_ptr().add(i));
        let sx2_v = vld1q_f32(sx2.as_ptr().add(i));
        let mf_v = vld1q_f32(mf.as_ptr().add(i));
        let mut acc = vmlaq_n_f32(rq_sum_v, scale_v, b_full);
        acc = vaddq_f32(acc, sx2_v);
        acc = vmlsq_n_f32(acc, mf_v, rq_margin);
        acc = vmlaq_f32(acc, vmulq_f32(sums_v, scale_v), a_v);
        vst1q_f32(out.as_mut_ptr().add(i), vmaxq_f32(acc, zero));
        i += 4;
    }
    estimate_batch_scalar(&sums[i..], &scales[i..], &sx2[i..], &mf[i..], a_full, b_full, rq_sum, rq_margin, &mut out[i..], n - i);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn estimate_batch_avx2(
    sums: &[u16],
    scales: &[f32],
    sx2: &[f32],
    mf: &[f32],
    a_full: f32,
    b_full: f32,
    rq_sum: f32,
    rq_margin: f32,
    out: &mut [f32],
    n: usize,
) {
    use std::arch::x86_64::*;
    let rq_sum_v = _mm256_set1_ps(rq_sum);
    let rq_margin_v = _mm256_set1_ps(rq_margin);
    let a_v = _mm256_set1_ps(a_full);
    let b_v = _mm256_set1_ps(b_full);
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let sums_v = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(
            sums.as_ptr().add(i) as *const __m128i,
        )));
        let scale_v = _mm256_loadu_ps(scales.as_ptr().add(i));
        let sx2_v = _mm256_loadu_ps(sx2.as_ptr().add(i));
        let mf_v = _mm256_loadu_ps(mf.as_ptr().add(i));
        let mut acc = _mm256_fmadd_ps(scale_v, b_v, rq_sum_v);
        acc = _mm256_add_ps(acc, sx2_v);
        acc = _mm256_fnmadd_ps(mf_v, rq_margin_v, acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(sums_v, scale_v), a_v, acc);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_max_ps(acc, zero));
        i += 8;
    }
    estimate_batch_scalar(&sums[i..], &scales[i..], &sx2[i..], &mf[i..], a_full, b_full, rq_sum, rq_margin, &mut out[i..], n - i);
}

/// Fused L2 estimate for 2-bit rows from the two bit-plane sums.
///
/// Per row: `dot = (2·sum0 + sum1)·a2 + b2` with `a2 = range_scale` and
/// `b2 = 3·num_chunks·qmin − 1.5·Σrot` (Lance: `full_dot = 2·m + e − 1.5·Σrot`
/// where `m`/`e` are the sign/ex plane sums); the estimate then uses the same
/// fused formula as [`estimate_batch`].  u32 intermediates avoid the u16
/// overflow that `2·sum0 + sum1` would hit at very high dimensions.
#[inline]
pub fn estimate_batch_2bit(
    sums0: &[u16],
    sums1: &[u16],
    scales: &[f32],
    sx2: &[f32],
    mf: &[f32],
    a2: f32,
    b2: f32,
    rq_sum: f32,
    rq_margin: f32,
    out: &mut [f32],
    n: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            // SAFETY: only selected when the AVX-512 families were detected.
            unsafe {
                estimate_batch_2bit_avx512(sums0, sums1, scales, sx2, mf, a2, b2, rq_sum, rq_margin, out, n);
                return;
            }
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: only selected when AVX2 was detected.
            unsafe {
                estimate_batch_2bit_avx2(sums0, sums1, scales, sx2, mf, a2, b2, rq_sum, rq_margin, out, n);
                return;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is baseline on aarch64.
        unsafe {
            estimate_batch_2bit_neon(sums0, sums1, scales, sx2, mf, a2, b2, rq_sum, rq_margin, out, n);
            return;
        }
    }
    estimate_batch_2bit_scalar(sums0, sums1, scales, sx2, mf, a2, b2, rq_sum, rq_margin, out, n);
}

#[inline]
fn estimate_batch_2bit_scalar(
    sums0: &[u16],
    sums1: &[u16],
    scales: &[f32],
    sx2: &[f32],
    mf: &[f32],
    a2: f32,
    b2: f32,
    rq_sum: f32,
    rq_margin: f32,
    out: &mut [f32],
    n: usize,
) {
    for i in 0..n {
        let s = (2 * sums0[i] as u32 + sums1[i] as u32) as f32;
        let dot = s * a2 + b2;
        out[i] = (dot * scales[i] + rq_sum + sx2[i] - mf[i] * rq_margin).max(0.0);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn estimate_batch_2bit_neon(
    sums0: &[u16],
    sums1: &[u16],
    scales: &[f32],
    sx2: &[f32],
    mf: &[f32],
    a2: f32,
    b2: f32,
    rq_sum: f32,
    rq_margin: f32,
    out: &mut [f32],
    n: usize,
) {
    use std::arch::aarch64::*;
    let rq_sum_v = vdupq_n_f32(rq_sum);
    let rq_margin_v = vdupq_n_f32(rq_margin);
    let a2_v = vdupq_n_f32(a2);
    let b2_v = vdupq_n_f32(b2);
    let zero = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= n {
        let s0_32 = vmovl_u16(vld1_u16(sums0.as_ptr().add(i)));
        let s1_32 = vmovl_u16(vld1_u16(sums1.as_ptr().add(i)));
        let s_32 = vmlaq_n_u32(s1_32, s0_32, 2); // 2·sum0 + sum1 (u32)
        let s_f = vcvtq_f32_u32(s_32);
        let dot = vmlaq_n_f32(b2_v, s_f, a2);
        let scale_v = vld1q_f32(scales.as_ptr().add(i));
        let sx2_v = vld1q_f32(sx2.as_ptr().add(i));
        let mf_v = vld1q_f32(mf.as_ptr().add(i));
        let mut acc = vaddq_f32(rq_sum_v, sx2_v);
        acc = vmlaq_f32(acc, dot, scale_v);
        acc = vmlsq_n_f32(acc, mf_v, rq_margin);
        vst1q_f32(out.as_mut_ptr().add(i), vmaxq_f32(acc, zero));
        i += 4;
    }
    estimate_batch_2bit_scalar(&sums0[i..], &sums1[i..], &scales[i..], &sx2[i..], &mf[i..], a2, b2, rq_sum, rq_margin, &mut out[i..], n - i);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn estimate_batch_2bit_avx2(
    sums0: &[u16],
    sums1: &[u16],
    scales: &[f32],
    sx2: &[f32],
    mf: &[f32],
    a2: f32,
    b2: f32,
    rq_sum: f32,
    rq_margin: f32,
    out: &mut [f32],
    n: usize,
) {
    use std::arch::x86_64::*;
    let rq_sum_v = _mm256_set1_ps(rq_sum);
    let rq_margin_v = _mm256_set1_ps(rq_margin);
    let a2_v = _mm256_set1_ps(a2);
    let b2_v = _mm256_set1_ps(b2);
    let two = _mm256_set1_ps(2.0);
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let s0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(
            sums0.as_ptr().add(i) as *const __m128i,
        )));
        let s1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(
            sums1.as_ptr().add(i) as *const __m128i,
        )));
        let s = _mm256_fmadd_ps(s0, two, s1); // 2·sum0 + sum1
        let dot = _mm256_fmadd_ps(s, a2_v, b2_v);
        let scale_v = _mm256_loadu_ps(scales.as_ptr().add(i));
        let sx2_v = _mm256_loadu_ps(sx2.as_ptr().add(i));
        let mf_v = _mm256_loadu_ps(mf.as_ptr().add(i));
        let mut acc = _mm256_add_ps(rq_sum_v, sx2_v);
        acc = _mm256_fmadd_ps(dot, scale_v, acc);
        acc = _mm256_fnmadd_ps(mf_v, rq_margin_v, acc);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_max_ps(acc, zero));
        i += 8;
    }
    estimate_batch_2bit_scalar(&sums0[i..], &sums1[i..], &scales[i..], &sx2[i..], &mf[i..], a2, b2, rq_sum, rq_margin, &mut out[i..], n - i);
}

/// Sum the quantized table for one 32-row transposed batch.
///
/// `codes` is `code_len × 32` transposed bytes, `table` is the flat
/// `code_len × 32` quantized table, `out` receives 32 u16 sums.  Returns 32.
#[inline]
pub fn sum_batch(codes: &[u8], code_len: usize, table: &[u8], out: &mut [u16]) -> usize {
    debug_assert_eq!(codes.len(), code_len * BATCH_SIZE);
    debug_assert_eq!(table.len(), code_len * BATCH_SIZE);
    debug_assert!(out.len() >= BATCH_SIZE);

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vbmi")
        {
            // SAFETY: only selected when the AVX-512 families were detected.
            unsafe {
                sum_batch_avx512(codes, code_len, table, out);
                return BATCH_SIZE;
            }
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: only selected when AVX2 was detected.
            unsafe {
                sum_batch_avx2(codes, code_len, table, out);
                return BATCH_SIZE;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is part of the aarch64 baseline.
        unsafe {
            sum_batch_neon(codes, code_len, table, out);
            return BATCH_SIZE;
        }
    }
    sum_batch_scalar(codes, code_len, table, out);
    BATCH_SIZE
}

#[inline]
pub fn sum_batch_scalar(codes: &[u8], code_len: usize, table: &[u8], out: &mut [u16]) {
    out[..BATCH_SIZE].fill(0);
    for c in 0..code_len {
        let block = &codes[c * BATCH_SIZE..(c + 1) * BATCH_SIZE];
        let current = &table[c * BATCH_SIZE..c * BATCH_SIZE + 16];
        let next = &table[c * BATCH_SIZE + 16..(c + 1) * BATCH_SIZE];
        for j in 0..16 {
            let low_cur = (block[j] & 0x0F) as usize;
            let high_cur = (block[j] >> 4) as usize;
            let low_next = (block[j + 16] & 0x0F) as usize;
            let high_next = (block[j + 16] >> 4) as usize;
            let lo = PERM0[j];
            let hi = PERM0[j] + 16;
            out[lo] = out[lo].saturating_add(current[low_cur] as u16 + next[low_next] as u16);
            out[hi] =
                out[hi].saturating_add(current[high_cur] as u16 + next[high_next] as u16);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_batch_neon(codes: &[u8], code_len: usize, table: &[u8], out: &mut [u16]) {
    use std::arch::aarch64::*;

    let low_mask = vdupq_n_u8(0x0f);
    let mut acc0_lo = vdupq_n_u16(0);
    let mut acc1_lo = vdupq_n_u16(0);
    let mut acc2_lo = vdupq_n_u16(0);
    let mut acc3_lo = vdupq_n_u16(0);
    let mut acc0_hi = vdupq_n_u16(0);
    let mut acc1_hi = vdupq_n_u16(0);
    let mut acc2_hi = vdupq_n_u16(0);
    let mut acc3_hi = vdupq_n_u16(0);

    for c in 0..code_len {
        let codes = codes.as_ptr().add(c * BATCH_SIZE);
        let dt = table.as_ptr().add(c * BATCH_SIZE);

        let c_lo = vld1q_u8(codes);
        let lut_lo = vld1q_u8(dt);
        let lo_lo = vandq_u8(c_lo, low_mask);
        let hi_lo = vshrq_n_u8::<4>(c_lo);
        let res_lo_lo = vqtbl1q_u8(lut_lo, lo_lo);
        let res_hi_lo = vqtbl1q_u8(lut_lo, hi_lo);
        acc0_lo = vaddq_u16(acc0_lo, vreinterpretq_u16_u8(res_lo_lo));
        acc1_lo = vaddq_u16(acc1_lo, vshrq_n_u16::<8>(vreinterpretq_u16_u8(res_lo_lo)));
        acc2_lo = vaddq_u16(acc2_lo, vreinterpretq_u16_u8(res_hi_lo));
        acc3_lo = vaddq_u16(acc3_lo, vshrq_n_u16::<8>(vreinterpretq_u16_u8(res_hi_lo)));

        let c_hi = vld1q_u8(codes.add(16));
        let lut_hi = vld1q_u8(dt.add(16));
        let lo_hi = vandq_u8(c_hi, low_mask);
        let hi_hi = vshrq_n_u8::<4>(c_hi);
        let res_lo_hi = vqtbl1q_u8(lut_hi, lo_hi);
        let res_hi_hi = vqtbl1q_u8(lut_hi, hi_hi);
        acc0_hi = vaddq_u16(acc0_hi, vreinterpretq_u16_u8(res_lo_hi));
        acc1_hi = vaddq_u16(acc1_hi, vshrq_n_u16::<8>(vreinterpretq_u16_u8(res_lo_hi)));
        acc2_hi = vaddq_u16(acc2_hi, vreinterpretq_u16_u8(res_hi_hi));
        acc3_hi = vaddq_u16(acc3_hi, vshrq_n_u16::<8>(vreinterpretq_u16_u8(res_hi_hi)));
    }

    acc0_lo = vsubq_u16(acc0_lo, vshlq_n_u16::<8>(acc1_lo));
    acc0_hi = vsubq_u16(acc0_hi, vshlq_n_u16::<8>(acc1_hi));
    let dis0_even = vaddq_u16(acc0_lo, acc0_hi);
    let dis0_odd = vaddq_u16(acc1_lo, acc1_hi);
    vst1q_u16(out.as_mut_ptr(), dis0_even);
    vst1q_u16(out.as_mut_ptr().add(8), dis0_odd);

    acc2_lo = vsubq_u16(acc2_lo, vshlq_n_u16::<8>(acc3_lo));
    acc2_hi = vsubq_u16(acc2_hi, vshlq_n_u16::<8>(acc3_hi));
    let dis1_even = vaddq_u16(acc2_lo, acc2_hi);
    let dis1_odd = vaddq_u16(acc3_lo, acc3_hi);
    vst1q_u16(out.as_mut_ptr().add(16), dis1_even);
    vst1q_u16(out.as_mut_ptr().add(24), dis1_odd);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_batch_avx2(codes: &[u8], code_len: usize, table: &[u8], out: &mut [u16]) {
    use std::arch::x86_64::*;

    let low_mask = _mm256_set1_epi8(0x0f);
    let mut accu0 = _mm256_setzero_si256();
    let mut accu1 = _mm256_setzero_si256();
    let mut accu2 = _mm256_setzero_si256();
    let mut accu3 = _mm256_setzero_si256();

    for c in 0..code_len {
        let codes = codes.as_ptr().add(c * BATCH_SIZE);
        let dt = table.as_ptr().add(c * BATCH_SIZE);

        let c_lo = _mm256_loadu_si256(codes as *const __m256i);
        let lut_lo = _mm256_loadu_si256(dt as *const __m256i);
        let lo = _mm256_and_si256(c_lo, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(c_lo, 4), low_mask);
        let res_lo = _mm256_shuffle_epi8(lut_lo, lo);
        let res_hi = _mm256_shuffle_epi8(lut_lo, hi);
        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
    }

    accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
    let dis0 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu0, accu1, 0x21),
        _mm256_blend_epi32(accu0, accu1, 0xF0),
    );
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, dis0);

    accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
    let dis1 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu2, accu3, 0x21),
        _mm256_blend_epi32(accu2, accu3, 0xF0),
    );
    _mm256_storeu_si256(out.as_mut_ptr().add(16) as *mut __m256i, dis1);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi")]
unsafe fn sum_batch_avx512(codes: &[u8], code_len: usize, table: &[u8], out: &mut [u16]) {
    use std::arch::x86_64::*;

    // 64-byte LUT lookups (vpermb) process TWO chunks per iteration: chunk
    // 2k's 32 code bytes and chunk 2k+1's 32 code bytes are loaded as one
    // zmm, the two 32-byte LUTs are concatenated, and each byte position's
    // nibble index is offset by its LUT half: pos&0x30 (0/16/32/48).  This
    // reproduces the per-128-bit-lane behavior of the AVX2 vpshufb kernel at
    // twice the width.
    let low_mask = _mm512_set1_epi8(0x0f);
    #[rustfmt::skip]
    static OFFSETS: [i8; 64] = [
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
        32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
        48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,
    ];
    let offsets = _mm512_loadu_si512(OFFSETS.as_ptr() as *const _);

    let mut accu0 = _mm512_setzero_si512();
    let mut accu1 = _mm512_setzero_si512();
    let mut accu2 = _mm512_setzero_si512();
    let mut accu3 = _mm512_setzero_si512();

    // Padded scratch for an odd trailing chunk: its LUT/code halves are
    // zeroed, so the pair iteration below accumulates +0 for it.
    let padded: [u8; 128];
    let (codes_ptr, lut_ptr, n_pairs) = if code_len % 2 == 1 {
        padded = [0u8; 128];
        std::ptr::copy_nonoverlapping(
            codes.as_ptr().add((code_len - 1) * BATCH_SIZE),
            padded.as_ptr() as *mut u8,
            BATCH_SIZE,
        );
        std::ptr::copy_nonoverlapping(
            table.as_ptr().add((code_len - 1) * BATCH_SIZE),
            padded.as_ptr().add(BATCH_SIZE) as *mut u8,
            BATCH_SIZE,
        );
        (padded.as_ptr(), padded.as_ptr().add(BATCH_SIZE), code_len / 2 + 1)
    } else {
        (codes.as_ptr(), table.as_ptr(), code_len / 2)
    };

    for k in 0..n_pairs {
        let src = if code_len % 2 == 1 && k == n_pairs - 1 {
            codes_ptr
        } else {
            codes.as_ptr().add(2 * k * BATCH_SIZE)
        };
        let lut_src = if code_len % 2 == 1 && k == n_pairs - 1 {
            lut_ptr
        } else {
            table.as_ptr().add(2 * k * BATCH_SIZE)
        };

        let c64 = _mm512_loadu_si512(src as *const _);
        let lut64 = _mm512_loadu_si512(lut_src as *const _);
        let lo = _mm512_or_si512(_mm512_and_si512(c64, low_mask), offsets);
        let hi = _mm512_or_si512(
            _mm512_and_si512(_mm512_srli_epi16(c64, 4), low_mask),
            offsets,
        );
        let res_lo = _mm512_permutexvar_epi8(lo, lut64);
        let res_hi = _mm512_permutexvar_epi8(hi, lut64);
        accu0 = _mm512_add_epi16(accu0, res_lo);
        accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
        accu2 = _mm512_add_epi16(accu2, res_hi);
        accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));
    }

    // Lanes 0..31 and 32..63 of each accumulator hold the same rows for the
    // two chunks of a pair: fold them together, then apply the same epilogue
    // as the AVX2 kernel.
    let fold = |v: __m512i| -> __m256i {
        let lo = _mm512_castsi512_si256(v);
        let hi = _mm512_extracti64x4_epi64::<1>(v);
        _mm256_add_epi16(lo, hi)
    };
    let mut a0 = fold(accu0);
    let mut a1 = fold(accu1);
    let mut a2 = fold(accu2);
    let mut a3 = fold(accu3);

    a0 = _mm256_sub_epi16(a0, _mm256_slli_epi16(a1, 8));
    let dis0 = _mm256_add_epi16(
        _mm256_permute2f128_si256(a0, a1, 0x21),
        _mm256_blend_epi32(a0, a1, 0xF0),
    );
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, dis0);

    a2 = _mm256_sub_epi16(a2, _mm256_slli_epi16(a3, 8));
    let dis1 = _mm256_add_epi16(
        _mm256_permute2f128_si256(a2, a3, 0x21),
        _mm256_blend_epi32(a2, a3, 0xF0),
    );
    _mm256_storeu_si256(out.as_mut_ptr().add(16) as *mut __m256i, dis1);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn estimate_batch_avx512(
    sums: &[u16],
    scales: &[f32],
    sx2: &[f32],
    mf: &[f32],
    a_full: f32,
    b_full: f32,
    rq_sum: f32,
    rq_margin: f32,
    out: &mut [f32],
    n: usize,
) {
    use std::arch::x86_64::*;
    let rq_sum_v = _mm512_set1_ps(rq_sum);
    let rq_margin_v = _mm512_set1_ps(rq_margin);
    let a_v = _mm512_set1_ps(a_full);
    let b_v = _mm512_set1_ps(b_full);
    let zero = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        let sums_v = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
            sums.as_ptr().add(i) as *const __m256i,
        )));
        let scale_v = _mm512_loadu_ps(scales.as_ptr().add(i));
        let sx2_v = _mm512_loadu_ps(sx2.as_ptr().add(i));
        let mf_v = _mm512_loadu_ps(mf.as_ptr().add(i));
        let mut acc = _mm512_fmadd_ps(scale_v, b_v, rq_sum_v);
        acc = _mm512_add_ps(acc, sx2_v);
        acc = _mm512_fnmadd_ps(mf_v, rq_margin_v, acc);
        acc = _mm512_fmadd_ps(_mm512_mul_ps(sums_v, scale_v), a_v, acc);
        _mm512_storeu_ps(out.as_mut_ptr().add(i), _mm512_max_ps(acc, zero));
        i += 16;
    }
    estimate_batch_scalar(&sums[i..], &scales[i..], &sx2[i..], &mf[i..], a_full, b_full, rq_sum, rq_margin, &mut out[i..], n - i);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn estimate_batch_2bit_avx512(
    sums0: &[u16],
    sums1: &[u16],
    scales: &[f32],
    sx2: &[f32],
    mf: &[f32],
    a2: f32,
    b2: f32,
    rq_sum: f32,
    rq_margin: f32,
    out: &mut [f32],
    n: usize,
) {
    use std::arch::x86_64::*;
    let rq_sum_v = _mm512_set1_ps(rq_sum);
    let rq_margin_v = _mm512_set1_ps(rq_margin);
    let a2_v = _mm512_set1_ps(a2);
    let b2_v = _mm512_set1_ps(b2);
    let two = _mm512_set1_ps(2.0);
    let zero = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        let s0 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
            sums0.as_ptr().add(i) as *const __m256i,
        )));
        let s1 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
            sums1.as_ptr().add(i) as *const __m256i,
        )));
        let s = _mm512_fmadd_ps(s0, two, s1);
        let dot = _mm512_fmadd_ps(s, a2_v, b2_v);
        let scale_v = _mm512_loadu_ps(scales.as_ptr().add(i));
        let sx2_v = _mm512_loadu_ps(sx2.as_ptr().add(i));
        let mf_v = _mm512_loadu_ps(mf.as_ptr().add(i));
        let mut acc = _mm512_add_ps(rq_sum_v, sx2_v);
        acc = _mm512_fmadd_ps(dot, scale_v, acc);
        acc = _mm512_fnmadd_ps(mf_v, rq_margin_v, acc);
        _mm512_storeu_ps(out.as_mut_ptr().add(i), _mm512_max_ps(acc, zero));
        i += 16;
    }
    estimate_batch_2bit_scalar(&sums0[i..], &sums1[i..], &scales[i..], &sx2[i..], &mf[i..], a2, b2, rq_sum, rq_margin, &mut out[i..], n - i);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_roundtrip_matches_row_major_sum() {
        // Build a random f32 table (code_len*32 entries) and random row-major
        // 1-bit codes; verify the transposed sum equals the row-major sum.
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(42);
        let code_len = 16usize;
        let n_rows = 100usize;

        let table_f32: Vec<f32> = (0..code_len * 32).map(|_| rng.gen_range(-3.0..3.0)).collect();
        let (table_u8, qmin, range_scale) = quantize_table(&table_f32);

        let codes: Vec<u8> = (0..n_rows * code_len).map(|_| rng.gen::<u8>()).collect();
        let transposed = transpose_1bit(&codes, n_rows, code_len);

        let n_chunks = code_len * 2;
        for row in 0..n_rows {
            // Reference: sum_set over the f32 table (row-major).
            let mut ref_sum = 0.0f32;
            for b in 0..code_len {
                let byte = codes[row * code_len + b];
                let lo = (byte & 0x0F) as usize;
                let hi = (byte >> 4) as usize;
                ref_sum += table_f32[b * 32 + lo] + table_f32[b * 32 + 16 + hi];
            }
            // Quantized: sum the u8 lookups + dequantize.
            let mut q_sum = 0u16;
            for b in 0..code_len {
                let byte = codes[row * code_len + b];
                let lo = (byte & 0x0F) as usize;
                let hi = (byte >> 4) as usize;
                q_sum += table_u8[b * 32 + lo] as u16 + table_u8[b * 32 + 16 + hi] as u16;
            }
            let deq = q_sum as f32 * range_scale + n_chunks as f32 * qmin;
            assert!(
                (deq - ref_sum).abs() <= 0.1 * ref_sum.abs().max(1.0),
                "row {row}: deq {deq} vs {ref_sum}"
            );
        }

        // Round-trip: transpose then untranspose must reproduce the input.
        let recovered = untranspose_1bit(&transposed, n_rows, code_len);
        assert_eq!(recovered, codes, "untranspose must invert transpose");

        // Fused SIMD estimate must match the scalar estimate bit-exactly
        // (the formula is a fixed linear combination, no FMA reordering across
        // the accumulation).  Tolerance guards the 1-ulp u16->f32 conversion.
        {
            use rand::rngs::SmallRng;
            use rand::{Rng, SeedableRng};
            let mut rng = SmallRng::seed_from_u64(7);
            let n = 100usize;
            let sums: Vec<u16> = (0..n).map(|_| rng.gen::<u16>()).collect();
            let scales: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
            let sx2: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0..1e6)).collect();
            let mf: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0..1e3)).collect();
            let (a_full, b_full, rq_sum, rq_margin) = (0.01, -2.5, 5e5, 700.0);
            let mut out_simd = vec![0.0f32; n];
            let mut out_scalar = vec![0.0f32; n];
            estimate_batch(&sums, &scales, &sx2, &mf, a_full, b_full, rq_sum, rq_margin, &mut out_simd, n);
            estimate_batch_scalar(&sums, &scales, &sx2, &mf, a_full, b_full, rq_sum, rq_margin, &mut out_scalar, n);
            for i in 0..n {
                assert!(
                    (out_simd[i] - out_scalar[i]).abs() <= 1e-3 * out_scalar[i].abs().max(1.0),
                    "i={i} simd {} vs scalar {}",
                    out_simd[i],
                    out_scalar[i]
                );
            }
        }

        // Now verify the batched SIMD/scalar sum matches, batch by batch.
        let n_batches = n_rows.div_ceil(BATCH_SIZE);
        for batch in 0..n_batches {
            let mut out = [0u16; BATCH_SIZE];
            let cbatch = &transposed[batch * code_len * BATCH_SIZE..(batch + 1) * code_len * BATCH_SIZE];
            sum_batch(cbatch, code_len, &table_u8, &mut out);
            for r in 0..BATCH_SIZE {
                let row = batch * BATCH_SIZE + r;
                if row >= n_rows {
                    continue;
                }
                let mut q_sum = 0u16;
                for b in 0..code_len {
                    let byte = codes[row * code_len + b];
                    let lo = (byte & 0x0F) as usize;
                    let hi = (byte >> 4) as usize;
                    q_sum += table_u8[b * 32 + lo] as u16 + table_u8[b * 32 + 16 + hi] as u16;
                }
                assert_eq!(out[r], q_sum, "batch {batch} row {r}");
            }
        }
    }
}
