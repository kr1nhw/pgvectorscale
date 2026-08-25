//! Fast random rotation for RaBitQ, ported from Lance's `vector/bq/rotation.rs`.
//!
//! The transform is a composition of:
//! 1) random diagonal sign flips (Rademacher variables),
//! 2) FWHT-style mixing on a power-of-two window,
//! 3) a Kac-style pairwise mixing step for non-power-of-two dimensions.
//!
//! The only state that needs to be persisted is the random sign bytes, so the
//! rotation is "matrix-free" and cheap to apply per-vector.
use rand::RngCore;

const FAST_ROTATION_ROUNDS: usize = 4;

/// In-place Fast Walsh-Hadamard transform (butterfly network).
/// Complexity: O(n log n), no heap allocation.
#[inline]
fn fwht_in_place(values: &mut [f32]) {
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

/// Apply a random diagonal matrix with +/-1 entries by toggling the f32 sign bit.
/// One bit in `signs` controls one element in `values`.
#[inline]
fn flip_signs_scalar(values: &mut [f32], signs: &[u8]) {
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn flip_signs_avx2(values: &mut [f32], signs: &[u8]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // Vectorized variant of `flip_signs_scalar`: consume 8 f32 values per AVX2 lane.
    let full_chunks = values.len() / 8;
    let bit_select = _mm256_setr_epi32(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
    let sign_flip = _mm256_set1_epi32(0x80000000u32 as i32);

    for (chunk_idx, &mask) in signs.iter().take(full_chunks).enumerate() {
        let mask = mask as i32;
        let mask_bits = _mm256_set1_epi32(mask);
        let test = _mm256_and_si256(mask_bits, bit_select);
        let cmp = _mm256_cmpeq_epi32(test, bit_select);
        let xor_mask = _mm256_and_si256(cmp, sign_flip);

        let ptr = unsafe { values.as_mut_ptr().add(chunk_idx * 8) };
        let vec = unsafe { _mm256_loadu_ps(ptr) };
        let out = _mm256_xor_ps(vec, _mm256_castsi256_ps(xor_mask));
        unsafe { _mm256_storeu_ps(ptr, out) };
    }

    if full_chunks * 8 < values.len() {
        flip_signs_scalar(&mut values[full_chunks * 8..], &signs[full_chunks..]);
    }
}

#[inline]
fn flip_signs(values: &mut [f32], signs: &[u8]) {
    debug_assert!(signs.len() * 8 >= values.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                flip_signs_avx2(values, signs);
            }
            return;
        }
    }
    flip_signs_scalar(values, signs);
}

/// A fixed-angle (pi/4) plane-rotation-like sweep over paired coordinates:
/// (x, y) -> (x + y, x - y). One Kac-style mixing step.
#[inline]
fn kacs_walk(values: &mut [f32]) {
    let half = values.len() / 2;
    let (left, right) = values.split_at_mut(half);
    for (x, y) in left.iter_mut().zip(right.iter_mut()) {
        let lx = *x;
        let ry = *y;
        *x = lx + ry;
        *y = lx - ry;
    }
}

/// Keep the transform numerically stable and approximately orthonormal.
#[inline]
fn rescale(values: &mut [f32], factor: f32) {
    for value in values.iter_mut() {
        *value *= factor;
    }
}

#[inline]
fn sign_bytes_per_round(dim: usize) -> usize {
    dim.div_ceil(8)
}

pub fn fast_rotation_signs_len(dim: usize) -> usize {
    FAST_ROTATION_ROUNDS * sign_bytes_per_round(dim)
}

/// Generate the random sign bytes for a rotation of the given dimension.
pub fn random_fast_rotation_signs(dim: usize, rng: &mut impl RngCore) -> Vec<u8> {
    let mut signs = vec![0u8; fast_rotation_signs_len(dim)];
    rng.fill_bytes(&mut signs);
    signs
}

/// Apply the fast random rotation. `input` may be shorter than `dim` (the rest is
/// treated as zero); `output` must have length `dim`.
#[inline]
pub fn apply_fast_rotation(input: &[f32], output: &mut [f32], signs: &[u8]) {
    let dim = output.len();
    let input_len = input.len().min(dim);
    output[..input_len]
        .iter_mut()
        .zip(input[..input_len].iter())
        .for_each(|(dst, src)| *dst = *src);
    if input_len < dim {
        output[input_len..].fill(0.0);
    }

    apply_fast_rotation_in_place(output, signs);
}

#[inline]
pub fn apply_fast_rotation_in_place(output: &mut [f32], signs: &[u8]) {
    let dim = output.len();
    let bytes_per_round = sign_bytes_per_round(dim);
    debug_assert_eq!(signs.len(), FAST_ROTATION_ROUNDS * bytes_per_round);
    if dim == 0 {
        return;
    }

    let trunc_dim = 1usize << dim.ilog2();
    let scale = 1.0f32 / (trunc_dim as f32).sqrt();
    if trunc_dim == dim {
        for round in 0..FAST_ROTATION_ROUNDS {
            let offset = round * bytes_per_round;
            flip_signs(output, &signs[offset..offset + bytes_per_round]);
            fwht_in_place(output);
            rescale(output, scale);
        }
        return;
    }

    let start = dim - trunc_dim;
    for round in 0..FAST_ROTATION_ROUNDS {
        let offset = round * bytes_per_round;
        flip_signs(output, &signs[offset..offset + bytes_per_round]);

        if round % 2 == 0 {
            let head = &mut output[..trunc_dim];
            fwht_in_place(head);
            rescale(head, scale);
        } else {
            let tail = &mut output[start..];
            fwht_in_place(tail);
            rescale(tail, scale);
        }

        kacs_walk(output);
    }

    // Matches RaBitQ-Library FhtKacRotator behavior for non-power-of-two dimensions.
    rescale(output, 0.25);
}

#[cfg(test)]
mod tests {
    use rand::{rngs::SmallRng, SeedableRng};

    use super::*;

    #[test]
    fn test_fast_rotation_sign_bytes() {
        let mut rng = SmallRng::seed_from_u64(42);
        assert_eq!(random_fast_rotation_signs(128, &mut rng).len(), 64);
        assert_eq!(random_fast_rotation_signs(130, &mut rng).len(), 68);
    }

    #[test]
    fn test_fast_rotation_preserves_shape() {
        let mut rng = SmallRng::seed_from_u64(7);
        let input = vec![1.0f32; 129];
        let mut output = vec![0.0f32; 129];
        let signs = random_fast_rotation_signs(129, &mut rng);
        apply_fast_rotation(&input, &mut output, &signs);
        assert_eq!(output.len(), 129);
    }

    #[test]
    fn test_rotation_preserves_inner_product_power_of_two() {
        let mut rng = SmallRng::seed_from_u64(1);
        let dim = 1024;
        let signs = random_fast_rotation_signs(dim, &mut rng);

        let x: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let y: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
        let mut rx = vec![0.0; dim];
        let mut ry = vec![0.0; dim];
        apply_fast_rotation(&x, &mut rx, &signs);
        apply_fast_rotation(&y, &mut ry, &signs);

        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
        let (dx, dy) = (dot(&rx, &rx), dot(&ry, &ry));
        let (ox, oy) = (dot(&x, &x), dot(&y, &y));
        // Orthonormality is approximate due to rounding; allow a loose relative tolerance.
        assert!((dx - ox).abs() / ox < 0.1);
        assert!((dy - oy).abs() / oy < 0.1);
    }
}
