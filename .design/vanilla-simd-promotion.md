# Vanilla PG17 — instruction promotion (AVX-512 / VNNI / VBMI) observation & plan

Status: **x86 tiers implemented (B1)**; aarch64 mirror analyzed (B2) — see below.

## B1/B3 result: implemented, verified, benchmarked — NO GAIN, reverted

Implemented `c0ccf64`, unit-test-verified bit-exact on x86, then A/B'd on the
vanilla BIGANN-10M ivfrq sweep.  Result vs the AVX2 baseline (p50 ms):

| probes | 1 | 8 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| AVX2 baseline | 1.636 | 3.071 | 6.421 | 9.402 | 15.157 |
| AVX-512 tiers | 1.525 | 3.034 | 6.543 | 9.925 | 16.285 |

No downclocking (4.074 GHz, same as AVX2's 4.072) but **IPC dropped
1.87 → 1.73**: the 512-bit loop's bookkeeping (offset-OR, lane fold,
epilogue) offsets the doubled LUT width, and the FastScan ALU fraction is too
small in this profile (memory/recheck-dominated).  The acceptance criterion
(≥10% p50 gain) was not met → **reverted (`8ca12d8`)**; kernels remain in git
history (`c0ccf64`) for resurrection on real (non-emulated) AVX-512 hardware
or workloads where `sum_batch` dominates (very high dims / num_bits=8 heavy
scans).  The fallback policy stands: scalar/SSE2/AVX2/NEON are the shipping
tiers.

## Implemented (commit `c0ccf64`, reverted `8ca12d8`) — x86 AVX-512 tiers, additive with fallbacks

- `sum_batch_avx512` (AVX512F+BW+VBMI): 64-byte `vpermb` LUT processes two
  chunks per iteration; per-byte offsets (pos&0x30) reproduce the AVX2
  kernel's per-128-bit-lane `vpshufb` semantics. **Verified bit-exact vs the
  scalar reference** by the existing unit test on x86 (an earlier flat-offset
  version failed the test — the lane semantics matter).
- `estimate_batch_avx512` / `estimate_batch_2bit_avx512` (F+BW): 16-wide.
- `dot_u8/u4_full_avx512` (F+BW): 16-wide FMA rescore.
- All scalar/SSE2/AVX2/NEON paths remain untouched as runtime fallbacks
  (per the non-negotiable fallback policy).

## aarch64 mirror (B2) — analysis result: SVE2, deferred to ARM hardware

Our aarch64 hot loops are NEON `vqtbl1q_u8` (16-byte LUT) and `vmlaq_f32`
(4-wide FMA). The VNNI-analog families do NOT apply: DotProd `udot`/I8MM
`usmmla` are int8-dot instructions, while our scheme mixes a u8 LUT and
u8×f32 FMAs. The true analogs are **SVE2 `tbl`** (scalable LUT width) and
**SVE FMA** (scalable f32 width). SVE/SVE2 target features are Linux-only and
cannot be compile-verified on the current Apple/x86 toolchains, so they stay
a documented sketch (below) until ARM hardware (Kunpeng/Graviton) is
available; the NEON baseline remains the fallback tier in the meantime.
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
