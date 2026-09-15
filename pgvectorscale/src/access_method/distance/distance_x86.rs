//! Calculate the distance by vector arithmetic optimized for x86

use simdeez::avx2::*;
use simdeez::scalar::*;
use simdeez::sse2::*;
use simdeez::sse41::*;

#[cfg(not(target_feature = "avx2"))]
#[cfg(not(doc))]
compile_error!(
    "On x86, the AVX2 feature must be enabled. Set RUSTFLAGS=\"-C target-feature=+avx2,+fma\""
);

//note: without fmadd, the performance degrades pretty badly. Benchmark before disbaling
#[cfg(not(target_feature = "fma"))]
#[cfg(not(doc))]
compile_error!(
    "On x86, the fma feature must be enabled. Set RUSTFLAGS=\"-C target-feature=+avx2,+fma\""
);

simdeez::simd_runtime_generate!(
    pub fn distance_l2_x86(x: &[f32], y: &[f32]) -> f32 {
        super::distance_l2_simd_body!(x, y)
    }
);

simdeez::simd_runtime_generate!(
    pub fn inner_product_x86(x: &[f32], y: &[f32]) -> f32 {
        super::inner_product_simd_body!(x, y)
    }
);

/// Calculate the cosine distance between two normal vectors
pub unsafe fn distance_cosine_x86_avx2(x: &[f32], y: &[f32]) -> f32 {
    (1.0 - inner_product_x86_avx2(x, y)).max(0.0)
}

// ---------------------------------------------------------------------------
// fp16 (binary16) stored vectors: the encoded bit pattern is ORDER-preserving,
// so the f16 -> f32 conversion is pure bit math on the pattern (sign/exp/mant
// field moves) — no IEEE decode machinery, no F16C dependency — and the L2/dot
// accumulation is the same FMA as the f32 kernels.  A lane with a subnormal or
// inf/nan exponent falls back to the exact scalar conversion for its whole
// 8-element block (essentially never taken on real data).
// ---------------------------------------------------------------------------

/// L2 squared between the f32 query and an f16 vector (8 lanes/step).
pub unsafe fn distance_l2_f16_x86(q: &[f32], v: &[u8]) -> f32 {
    if std::arch::is_x86_feature_detected!("f16c") {
        distance_l2_f16_x86_f16c(q, v)
    } else {
        distance_l2_f16_x86_int(q, v)
    }
}

/// Dot product between the f32 query and an f16 vector (8 lanes/step).
pub unsafe fn distance_inner_product_f16_x86(q: &[f32], v: &[u8]) -> f32 {
    if std::arch::is_x86_feature_detected!("f16c") {
        distance_inner_product_f16_x86_f16c(q, v)
    } else {
        distance_inner_product_f16_x86_int(q, v)
    }
}

/// F16C path: `vcvtph2ps` IS the ordered-pattern conversion as one hardware
/// instruction.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn distance_l2_f16_x86_f16c(q: &[f32], v: &[u8]) -> f32 {
    use core::arch::x86_64::*;

    let n = q.len();
    debug_assert_eq!(n * 2, v.len());
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm_loadu_si128(v.as_ptr().add(i * 2).cast());
        let vf = _mm256_cvtph_ps(h);
        let qa = _mm256_loadu_ps(q.as_ptr().add(i));
        let d = _mm256_sub_ps(qa, vf);
        acc = _mm256_fmadd_ps(d, d, acc);
        i += 8;
    }
    let mut total = horizontal_sum_8(acc);
    while i < n {
        let x = super::f16_to_f32(u16::from_le_bytes([v[2 * i], v[2 * i + 1]]));
        let d = q[i] - x;
        total += d * d;
        i += 1;
    }
    total
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn distance_inner_product_f16_x86_f16c(q: &[f32], v: &[u8]) -> f32 {
    use core::arch::x86_64::*;

    let n = q.len();
    debug_assert_eq!(n * 2, v.len());
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm_loadu_si128(v.as_ptr().add(i * 2).cast());
        let vf = _mm256_cvtph_ps(h);
        let qa = _mm256_loadu_ps(q.as_ptr().add(i));
        acc = _mm256_fmadd_ps(qa, vf, acc);
        i += 8;
    }
    let mut total = horizontal_sum_8(acc);
    while i < n {
        let x = super::f16_to_f32(u16::from_le_bytes([v[2 * i], v[2 * i + 1]]));
        total += q[i] * x;
        i += 1;
    }
    total
}

/// Integer bit-math path (no F16C): the same ordered-pattern conversion as
/// explicit integer SIMD.
#[target_feature(enable = "avx2,fma")]
unsafe fn distance_l2_f16_x86_int(q: &[f32], v: &[u8]) -> f32 {
    use core::arch::x86_64::*;

    let n = q.len();
    debug_assert_eq!(n * 2, v.len());
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm_loadu_si128(v.as_ptr().add(i * 2).cast());
        let vf = f16x8_to_f32(h);
        let qa = _mm256_loadu_ps(q.as_ptr().add(i));
        let d = _mm256_sub_ps(qa, vf);
        acc = _mm256_fmadd_ps(d, d, acc);
        i += 8;
    }
    let mut total = horizontal_sum_8(acc);
    while i < n {
        let x = super::f16_to_f32(u16::from_le_bytes([v[2 * i], v[2 * i + 1]]));
        let d = q[i] - x;
        total += d * d;
        i += 1;
    }
    total
}

#[target_feature(enable = "avx2,fma")]
unsafe fn distance_inner_product_f16_x86_int(q: &[f32], v: &[u8]) -> f32 {
    use core::arch::x86_64::*;

    let n = q.len();
    debug_assert_eq!(n * 2, v.len());
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm_loadu_si128(v.as_ptr().add(i * 2).cast());
        let vf = f16x8_to_f32(h);
        let qa = _mm256_loadu_ps(q.as_ptr().add(i));
        acc = _mm256_fmadd_ps(qa, vf, acc);
        i += 8;
    }
    let mut total = horizontal_sum_8(acc);
    while i < n {
        let x = super::f16_to_f32(u16::from_le_bytes([v[2 * i], v[2 * i + 1]]));
        total += q[i] * x;
        i += 1;
    }
    total
}

/// Convert 8 f16 lanes to f32 with integer bit math; falls back to the exact
/// scalar conversion when any lane is subnormal/inf/nan.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn f16x8_to_f32(h: core::arch::x86_64::__m128i) -> core::arch::x86_64::__m256 {
    use core::arch::x86_64::*;

    // All lanes normal?  x' = h ^ 0x8000 (signed mirror); normal iff
    // 0x8400 <= x' < 0xFC00, i.e. x' - 0x8400 < 0x7800 (signed).
    let flipped = _mm_xor_si128(h, _mm_set1_epi16(-0x8000));
    let shifted = _mm_sub_epi16(flipped, _mm_set1_epi16(-0x7C00)); // 0x8400
    let cmp = _mm_cmpgt_epi16(_mm_set1_epi16(0x7800), shifted);
    if _mm_movemask_epi8(cmp) != 0xFFFF {
        // Rare: subnormal or inf/nan present — exact scalar conversion.
        let mut tmp = [0u8; 16];
        _mm_storeu_si128(tmp.as_mut_ptr().cast(), h);
        let mut out = [0.0f32; 8];
        for (o, pair) in out.iter_mut().zip(tmp.chunks_exact(2)) {
            *o = super::f16_to_f32(u16::from_le_bytes([pair[0], pair[1]]));
        }
        return _mm256_loadu_ps(out.as_ptr());
    }

    let h32 = _mm256_cvtepu16_epi32(h);
    let sign = _mm256_slli_epi32(_mm256_and_si256(h32, _mm256_set1_epi32(0x8000)), 16);
    let abs = _mm256_and_si256(h32, _mm256_set1_epi32(0x7FFF));
    let exp = _mm256_srli_epi32(abs, 10);
    // 127 - 15 = 112 exponent bias
    let exp32 = _mm256_slli_epi32(_mm256_add_epi32(exp, _mm256_set1_epi32(112)), 23);
    let mant = _mm256_slli_epi32(_mm256_and_si256(abs, _mm256_set1_epi32(0x3FF)), 13);
    let bits = _mm256_or_si256(sign, _mm256_or_si256(exp32, mant));
    _mm256_castsi256_ps(bits)
}

/// Scale-weighted pairwise SQ8: `SUM w * (qhat - code)^2`, 8 lanes/step
/// (i16 widening + integer sub + f32 square/FMA).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn distance_l2_sq8_pairwise_x86(qhat: &[i16], code: &[u8], w: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let n = qhat.len();
    debug_assert_eq!(n, code.len());
    debug_assert_eq!(n, w.len());
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let qh = _mm_loadu_si128(qhat.as_ptr().add(i).cast()); // 8 x i16
        let qh32 = _mm256_cvtepi16_epi32(qh);
        let c = _mm_cvtsi64_si128(u64::from_le_bytes([
            code[i],
            code[i + 1],
            code[i + 2],
            code[i + 3],
            code[i + 4],
            code[i + 5],
            code[i + 6],
            code[i + 7],
        ]) as i64);
        let c32 = _mm256_cvtepu8_epi32(c);
        let d = _mm256_sub_epi32(qh32, c32);
        let df = _mm256_cvtepi32_ps(d);
        let wv = _mm256_loadu_ps(w.as_ptr().add(i));
        acc = _mm256_fmadd_ps(wv, _mm256_mul_ps(df, df), acc);
        i += 8;
    }
    let mut total = horizontal_sum_8(acc);
    while i < n {
        let d = (qhat[i] - code[i] as i16) as f32;
        total += w[i] * d * d;
        i += 1;
    }
    total
}

/// Sum the 8 lanes of an AVX register.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_8(acc: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum = _mm_add_ps(lo, hi);
    let sum = _mm_hadd_ps(sum, sum);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}

#[cfg(test)]
mod tests {
    #[test]
    fn distances_equal() {
        let r: Vec<f32> = (0..2000).map(|v| v as f32 + 1.0).collect();
        let l: Vec<f32> = (0..2000).map(|v| v as f32 + 2.0).collect();

        let r_size = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        let l_size = l.iter().map(|v| v * v).sum::<f32>().sqrt();

        let r: Vec<f32> = r.iter().map(|v| v / r_size).collect();
        let l: Vec<f32> = l.iter().map(|v| v / l_size).collect();

        assert!(
            (unsafe { super::distance_cosine_x86_avx2(&r, &l) }
                - super::super::distance_cosine_unoptimized(&r, &l))
            .abs()
                < 0.000001
        );
        assert!(
            (unsafe { super::distance_l2_x86_avx2(&r, &l) }
                - super::super::distance_l2_unoptimized(&r, &l))
            .abs()
                < 0.000001
        );
    }
}
