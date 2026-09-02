# Vanilla PG17 — instruction promotion (AVX-512 / VNNI / VBMI) observation & plan

Status: **observation recorded for planning** — not implemented.
Applies to the compute-bound path: vanilla PG17, and any LFC-hit Neon reads
(Neon as a whole is I/O-bound — see `.design/neon/OPTIMIZATION-PLAN.md` §0).

## Evidence (2026-09-02, perf on the x86 box)

Workload: ivf/ivfrq probes=64, BIGANN-10M, 100-query loop, vanilla PG17 on
5432. Backend profile (`perf record -F 199 -g`, 20 s):

| hotspot | share |
|---|---|
| `ivf::scan::amgettuple_inner` (our scan: FastScan + rescore + heap) | 28.6% |
| kernel `copy_user_enhanced_fast_string` (smgrreadv page copies — fixed) | 23.5% |
| libc memcpy | 8.5% |
| `PinBuffer` / `find_get_pages_contig` (buffer/page-cache machinery) | 12.0% |

Inside `amgettuple_inner` (perf annotate): the single hottest instruction is
`vpaddw %ymm` (AVX2 16-bit FastScan sum accumulation), surrounded by a large
block of **scalar** f32 ops (`vaddss/vmaxss/vminss/vblendvps`) from the
rescore/top-K heap.

Current SIMD state: **AVX2 only**, runtime-detected
(`distance/distance_x86.rs`, `quantization/rabitq.rs`,
`quantization/rabitq_fastscan.rs`). The host exposes
`avx512f/bw/vl`, `avx512_vnni`, `avx512_vbmi/vbmi2`, `avx512_bitalg`,
`avx512_vpopcntdq`, `avx_vnni`, `avx512_bf16` (QEMU guest on a modern Xeon).

## Planned instruction promotions (ranked by expected effect)

1. **1-bit `sum_batch`**: 512-bit XOR + `vpopcntb` (`AVX512_BITALG`) +
   `vpternlogd` — replaces the LUT/popcount path; hottest inner loop,
   expect 2–4× on it.
2. **2-bit `sum_batch`**: 512-bit `vpternlogd`/shift accumulation (~2× width).
3. **8-bit sums**: `vpdpbusd` (`AVX512-VNNI`) u8 dot-accumulate — 4×
   per-instruction throughput vs the AVX2 workaround.
4. **u4 nibble LUT**: `vpermb` (`AVX512-VBMI`) — 64-byte permutes (2×
   `vpshufb` width).
5. **f32 rescore `dot_u*_full`**: AVX-512F FMA (16-wide vs 8-wide) — ~2× on
   rescore math.

All behind `is_x86_feature_detected!` gates mirroring the existing AVX2
dispatch; keep AVX2 as the fallback tier (and NEON for aarch64).

## Expected impact & caveats

- SIMD-relevant CPU fraction ≈ 20–30% of a vanilla query; the 23.5% kernel
  copy is fixed by design. Realistic overall gain: **~10–25% query latency**
  on the compute-bound path, best at large `probes`/`num_bits=1,8`.
- QEMU guest: AVX-512 can trigger frequency throttling (observed 4.07 GHz
  under AVX2); must A/B measure, not assume.
- Neon cluster: no wall-clock effect until the storage path is fixed
  (LFC/pageserver cache sizing — see the Neon optimization plan).

## When to do it

After the Neon read-path work lands (LFC sizing + pipelining), implement
promotion #1 first, re-run the vanilla sweep + a k8s sweep, and compare the
recall@10/latency curves against the current `bench/RESULTS.md` numbers.
