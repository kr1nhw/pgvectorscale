//! SQ8 distance smoke gun: the user-suggested pairwise-quantized approaches
//! vs the current scalar decode, plus an E4M3 pattern-compare micro-bench.
//!
//! Run (release, no postgres needed):
//!   cargo test --release sq8_distance_smoke_gun -- --ignored --nocapture
//!
//! Approaches (dim 128, corpus 50k, 100 queries):
//!   1. `scalar`  — the current codec: per-element min + code*scale, f32 FMA.
//!   2. `integer` — the user's idea: quantize q into code space ONCE
//!      (q_hat = round((q-min)*inv_scale)), then pairwise
//!      SUM (q_hat - code)^2 in i32 (pure integer, auto-vectorizes).
//!   3. `dot`     — Lance-style: SUM q*v_hat via u8 x i16 multiply-add with
//!      the precomputed v_hat norm (SUM v_hat^2 stored per vector),
//!      d = SUM q^2 + SUM v_hat^2 - 2 SUM q*v_hat.
//!
//! Precision: recall@10 vs the exact f32 top-10 and mean rank displacement.
#![cfg(test)]

use crate::access_method::hnswsq::quantize::{Codec, HnswPrecision, Sq8Calibration};
use crate::access_method::distance::DistanceType;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const DIM: usize = 128;
const CORPUS: usize = 50_000;
const QUERIES: usize = 100;

fn gen(rng: &mut SmallRng, n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| (0..DIM).map(|_| rng.gen_range(0.0f32..255.0)).collect())
        .collect()
}

/// Current codec path: decode each stored byte to f32, then FMA.
fn l2_scalar(q: &[f32], code: &[u8], mins: &[f32], scales: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..DIM {
        let x = mins[i] + code[i] as f32 * scales[i];
        let d = q[i] - x;
        acc += d * d;
    }
    acc
}

/// The user's pairwise-quantized distance: query pre-quantized once, then
/// pure integer squared differences over the codes.
fn l2_integer(qhat: &[i16], code: &[u8]) -> i32 {
    let mut acc = 0i32;
    for i in 0..DIM {
        let d = qhat[i] - code[i] as i16;
        acc += d as i32 * d as i32;
    }
    acc
}

/// Lance-style dot with precomputed stored norms.
fn l2_dot(q: &[f32], qnorm2: f32, code: &[u8], vnorm2: f32, mins: &[f32], scales: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    for i in 0..DIM {
        dot += q[i] * (mins[i] + code[i] as f32 * scales[i]);
    }
    (qnorm2 + vnorm2 - 2.0 * dot).max(0.0)
}

/// Lance-style dot in pure integer: per-query i16 weights w_i =
/// round(q_i * scale_i * 2^SHIFT), stored norm precomputed, d = qnorm2 +
/// vnorm2 - 2 * (SUM w_i * code_i) >> SHIFT.  The i16 x u8 multiply-add
/// auto-vectorizes (pmaddwd after widening); exact ranking preserved up to
/// the weight quantization.
const DOT_SHIFT: i32 = 5;

fn l2_dot_integer(qnorm2: f32, w: &[i16], code: &[u8], vnorm2: f32) -> f32 {
    let mut dot = 0i32;
    for i in 0..DIM {
        dot += w[i] as i32 * code[i] as i32;
    }
    (qnorm2 + vnorm2 - 2.0 * (dot >> DOT_SHIFT) as f32).max(0.0)
}

fn top10(dists: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..dists.len()).collect();
    idx.sort_by(|&a, &b| dists[a].partial_cmp(&dists[b]).unwrap());
    idx.truncate(10);
    idx
}

fn recall10(a: &[usize], b: &[usize]) -> f32 {
    let mut hit = 0;
    for &x in b {
        if a.contains(&x) {
            hit += 1;
        }
    }
    hit as f32 / 10.0
}

#[test]
#[ignore]
fn sq8_distance_smoke_gun() {
    let mut rng = SmallRng::seed_from_u64(42);
    let corpus = gen(&mut rng, CORPUS);
    let queries = gen(&mut rng, QUERIES);

    // Calibration from a sample (like the build's reservoir)
    let sample: Vec<Vec<f32>> = corpus[..30000].to_vec();
    let calib = Sq8Calibration::train(&sample, DIM);
    let codec = Codec::new_sq8(&calib);

    // Encode the corpus once
    let codes: Vec<Vec<u8>> = corpus.iter().map(|v| codec.encode(v)).collect();
    let mins = calib.mins.clone();
    let scales: Vec<f32> = (0..DIM)
        .map(|d| (calib.maxs[d] - calib.mins[d]) / 255.0)
        .collect();
    let vnorm2: Vec<f32> = codes
        .iter()
        .map(|c| {
            (0..DIM)
                .map(|i| mins[i] + c[i] as f32 * scales[i])
                .map(|x| x * x)
                .sum()
        })
        .collect();

    // Per-query state
    let mut qhats: Vec<Vec<i16>> = Vec::new();
    let mut qnorm2s = Vec::new();
    let mut qweights: Vec<Vec<i16>> = Vec::new();
    for q in &queries {
        let mut qhat = Vec::with_capacity(DIM);
        let mut qw = Vec::with_capacity(DIM);
        for i in 0..DIM {
            let s = (q[i] - mins[i]) * (255.0 / (calib.maxs[i] - calib.mins[i]));
            qhat.push(s.round().clamp(0.0, 255.0) as i16);
            // w = q * scale * 2^SHIFT; |q*scale| <= 230 << 5 fits i16 easily
            qw.push(((q[i] * scales[i]) * (1 << DOT_SHIFT) as f32).round() as i16);
        }
        qhats.push(qhat);
        qweights.push(qw);
        qnorm2s.push(q.iter().map(|x| x * x).sum::<f32>());
    }

    // Exact reference distances (f32 over the raw vectors)
    let mut exact_dists: Vec<Vec<f32>> = Vec::new();
    for q in &queries {
        let d: Vec<f32> = corpus
            .iter()
            .map(|v| {
                let mut acc = 0.0f32;
                for i in 0..DIM {
                    let d = q[i] - v[i];
                    acc += d * d;
                }
                acc
            })
            .collect();
        exact_dists.push(d);
    }

    // --- timings (min of 5 sweeps over the whole matrix) ---
    let sweep = |f: &dyn Fn(usize) -> f32| -> f64 {
        let t0 = Instant::now();
        let mut sink = 0.0f32;
        for qi in 0..QUERIES {
            for ci in 0..CORPUS {
                sink += f(qi * CORPUS + ci);
            }
        }
        let el = t0.elapsed().as_secs_f64();
        assert!(sink >= 0.0);
        el
    };

    let t_scalar = (0..5)
        .map(|_| {
            sweep(&|k| {
                let qi = k / CORPUS;
                let ci = k % CORPUS;
                l2_scalar(&queries[qi], &codes[ci], &mins, &scales)
            })
        })
        .fold(f64::INFINITY, f64::min);
    let t_integer = (0..5)
        .map(|_| {
            sweep(&|k| {
                let qi = k / CORPUS;
                let ci = k % CORPUS;
                l2_integer(&qhats[qi], &codes[ci]) as f32
            })
        })
        .fold(f64::INFINITY, f64::min);
    let t_dot = (0..5)
        .map(|_| {
            sweep(&|k| {
                let qi = k / CORPUS;
                let ci = k % CORPUS;
                l2_dot(&queries[qi], qnorm2s[qi], &codes[ci], vnorm2[ci], &mins, &scales)
            })
        })
        .fold(f64::INFINITY, f64::min);
    let t_dot_int = (0..5)
        .map(|_| {
            sweep(&|k| {
                let qi = k / CORPUS;
                let ci = k % CORPUS;
                l2_dot_integer(qnorm2s[qi], &qweights[qi], &codes[ci], vnorm2[ci])
            })
        })
        .fold(f64::INFINITY, f64::min);

    let pairs = (QUERIES * CORPUS) as f64;
    println!(
        "timing ns/pair: scalar={:.1} integer={:.1} dot={:.1} dot_integer={:.1}",
        t_scalar / pairs * 1e9,
        t_integer / pairs * 1e9,
        t_dot / pairs * 1e9,
        t_dot_int / pairs * 1e9,
    );

    // --- precision: recall@10 + mean rank displacement vs exact ---
    let mut recall_scalar = 0.0f32;
    let mut recall_integer = 0.0f32;
    let mut recall_dot = 0.0f32;
    let mut recall_dot_int = 0.0f32;
    for qi in 0..QUERIES {
        let exact = top10(&exact_dists[qi]);
        let d_scalar: Vec<f32> = (0..CORPUS)
            .map(|ci| l2_scalar(&queries[qi], &codes[ci], &mins, &scales))
            .collect();
        let d_integer: Vec<f32> = (0..CORPUS)
            .map(|ci| l2_integer(&qhats[qi], &codes[ci]) as f32)
            .collect();
        let d_dot: Vec<f32> = (0..CORPUS)
            .map(|ci| l2_dot(&queries[qi], qnorm2s[qi], &codes[ci], vnorm2[ci], &mins, &scales))
            .collect();
        let d_dot_int: Vec<f32> = (0..CORPUS)
            .map(|ci| l2_dot_integer(qnorm2s[qi], &qweights[qi], &codes[ci], vnorm2[ci]))
            .collect();
        recall_scalar += recall10(&exact, &top10(&d_scalar));
        recall_integer += recall10(&exact, &top10(&d_integer));
        recall_dot += recall10(&exact, &top10(&d_dot));
        recall_dot_int += recall10(&exact, &top10(&d_dot_int));
    }
    println!(
        "recall@10: scalar={:.3} integer={:.3} dot={:.3} dot_integer={:.3}",
        recall_scalar / QUERIES as f32,
        recall_integer / QUERIES as f32,
        recall_dot / QUERIES as f32,
        recall_dot_int / QUERIES as f32,
    );

    // --- e4m3 pattern-compare micro-bench (the user's E/M pruning idea) ---
    // The E4M3 bit pattern is ORDER-preserving for non-negative values, so a
    // pairwise "which vector is closer" decision needs only a byte compare
    // per element until the first difference.  Measure how early the first
    // differing byte appears (how many full elements need comparing) and the
    // cost of the byte-compare loop vs the float path.
    let encode_e4m3 = |x: f32| crate::access_method::hnswsq::quantize::f32_to_e4m3(x.clamp(-448.0, 448.0));
    let a: Vec<Vec<u8>> = (0..10_000).map(|_| (0..DIM).map(|_| encode_e4m3(rng.gen_range(0.0..255.0))).collect()).collect();
    let b: Vec<Vec<u8>> = (0..10_000).map(|_| (0..DIM).map(|_| encode_e4m3(rng.gen_range(0.0..255.0))).collect()).collect();
    let mut first_diff = vec![0usize; DIM + 1];
    for (x, y) in a.iter().zip(b.iter()) {
        let mut d = DIM;
        for i in 0..DIM {
            if x[i] != y[i] {
                d = i;
                break;
            }
        }
        first_diff[d] += 1;
    }
    println!(
        "e4m3 first-differing element: median~{} (histogram head: {:?})",
        (0..=DIM)
            .find(|&i| {
                first_diff[..=i].iter().sum::<usize>() >= 5000
            })
            .unwrap(),
        &first_diff[..6],
    );
    let t0 = Instant::now();
    let mut sink = 0u32;
    for (x, y) in a.iter().zip(b.iter()) {
        for i in 0..DIM {
            if x[i] != y[i] {
                sink += x[i] as u32;
                break;
            }
        }
    }
    let t_bytecmp = t0.elapsed().as_secs_f64();
    println!(
        "e4m3 byte-compare: {} ns/pair (branchy, first-diff early-out)",
        t_bytecmp / 10_000.0 * 1e9
    );
    assert!(sink > 0);
}
