# hnswsq vs pgvector-hnsw: same-host 1M A/B (121.37.117.106, 32 vCPU, PG 17.11)

Dataset: BIGANN 1M rows, dim 128, first 100 queries, exact top-10 ground truth
(`items_1m` / `gt_1m`).  Both engines: `m=16`, `ef_construction=64`,
`maintenance_work_mem=8GB`.  hnswsq index: `storage_layout=plain`.

## Release-PostgreSQL results (2026-09-16, same box)

The box's default cluster runs pgrx's DEBUG PostgreSQL
(`--enable-cassert -DRANDOMIZE_ALLOCATED_MEMORY=1`), which taxes every
palloc and distorts absolute numbers.  A **release PostgreSQL 17.11** was
built from the same source tree on the box (`/root/pg17-rel-src`,
prefix `/root/pg17-release`, port 54331), with pgvector 0.8.6 rebuilt
against it and the pgrx-built release extension installed; the dataset
was restored from the scratch cluster (`items_1m` 1M + `bench_queries` +
`gt_1m`).  All rows below are one 2026-09-16 pass (see the settings
matrix).

| config | build s | qms ef10/40/160/640 (ms) | recall@10 ef160/640 | ins rps (ms/row) |
|---|---|---|---|---|
| pgvector hnsw | 64.7 | 0.545 / 0.913 / 2.196 / 6.291 | 0.991 / 0.999 | 691 (1.45) |
| hnswsq plain | 66.9 | 0.572 / 0.990 / 2.317 / 6.252 | 0.994 / 0.999 | 696 (1.44) |
| hnswsq f8 (weighted pairwise, default) | 113.8 | 0.492 / 0.902 / 2.033 / 5.345 | 0.991 / 0.998 | 457 (2.19) |
| hnswsq sq8 (fixed [0,255] int8, training-free) | 114.4 | 0.508 / 0.976 / 2.278 / 5.448 | 0.994 / 0.999 | 504 (1.99) |
| hnswsq sq16 (fixed [0,255] int16, training-free) | 149.2 | 0.628 / 1.103 / 2.245 / 5.657 | 0.994 / 0.999 | 404 (2.48) |

Recall@10 across ef (release PG, `recall_sweep.sh`, all layouts on the
1M-row table):

| ef | pgvector | plain | ieeefp16 | ieeefp8 | f8 | sq8 | sq16 |
|----|----------|-------|----------|---------|----|-----|------|
| 10 | 0.795 | 0.782 | 0.782 | 0.709 | 0.782 | 0.783 | 0.782 |
| 20 | 0.886 | 0.883 | 0.883 | 0.876 | 0.878 | 0.884 | 0.884 |
| 40 | 0.941 | 0.939 | 0.939 | 0.928 | 0.943 | 0.940 | 0.941 |
| 80 | 0.975 | 0.978 | 0.979 | 0.971 | 0.979 | 0.978 | 0.979 |
| 160 | 0.991 | 0.994 | 0.994 | 0.994 | 0.991 | 0.994 | 0.994 |
| 320 | 0.999 | 0.999 | 0.999 | 1.000 | 0.998 | 0.999 | 0.999 |
| 640 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The fixed-range layouts track plain within ±0.2pt over the whole range —
the byte-exact codes give them plain-class ranking on uint8 data with a
quarter of the vector bytes.

Findings:

- **The quantized layouts' queries beat pgvector at every ef on release
  PG** (f8 0.492–5.345, sq8 0.508–5.448, sq16 0.628–5.657 vs pgvector
  0.545–6.291 ms), with recall within 0.4pt (0.991–0.994 vs 0.991–0.999)
  and 45–56% of the index size.
- **The weighted pairwise distance is the f8 default**: the query
  is quantized once into code space and candidate distances are
  `SUM scale_i^2 (qhat - code)^2` — the squared difference of the DECODED
  values, i.e. the scalar decode's exact ranking with the
  query-quantized integer form (the scale weights Lance preserves via
  stored per-vector norms and Milvus via per-dimension 256-entry LUTs),
  with AVX2/AVX-512 kernels.  Measured pairwise-vs-scalar on the same
  data: recall identical, queries ~2.2x the scalar form, build ~1.15x,
  and a smaller graph (376 vs 450 MB).  The UNWEIGHTED pairwise was
  rejected (4 pt recall loss on uneven scales); the Lance-dot variant
  needs stored norms (format change).
- **pgvector's build is faster** (64.7s vs 113.8s for f8; plain reaches
  parity at 66.9s): the quantized builds are distance/load-bound (see
  the perf attribution below).  Build parity for the quantized layouts
  remains the open item.
- **`sq8`/`sq16` — the training-free fixed-range SQ layouts**: a global
  `[0, 255]` range, so byte-valued BIGANN vectors encode losslessly
  (`sq8` stores the byte itself, scale 1.0; `sq16` adds 128 sub-steps
  per unit at 2⁻⁷) and distances come straight from the integer codes
  (`Σ(q̂ − code)²`).  **Recall tracks plain exactly** (0.994/0.999/1.0 at
  ef 160/320/640 — equal to plain/fp16, ahead of f8's 0.991/0.998 and
  pgvector's 0.991/0.999), queries beat pgvector at every ef (0.508–5.448
  ms for sq8; 0.628–5.657 for sq16 vs 0.545–6.291), inserts are the
  fastest quantized layout (504 rows/s, 1.99 ms/row), and the size is
  the plain ratio expected of 1-byte / 2-byte vectors (376 vs 537 B/row
  — sq8 matches f8 per row).  Graph construction uses the integer
  pairwise for sq8 (value-identical to the decoded form — its build's
  query is always integral) and the decoded form for sq16; the scan
  always uses the GUC-selected form.

## Settings matrix (build / query latency / insert throughput)

Measured on the **release PostgreSQL 17.11** cluster on the same box
(port 54331; see the top section for how it was built), `items_1m`,
m=16, efc=64, mwm 8GB, 4 workers, 50k fresh inserts per config (batches
of 1000, one index at a time).  Each config indexes the table as grown
by the previous configs' inserts (1M for pgvector, +50k each step), so
per-row costs are the comparable column.  2026-09-16 run — all seven
configs in one pass, current code (bit-math fp conversions,
weighted-pairwise f8 default, the fixed-range `sq8`/`sq16` layouts):

| config | build s | size_bytes | qms ef10/40/160/640 (LIMIT 10) | ins rows/s | ins row_ms mean/p50/p99 |
|---|---|---|---|---|---|
| pgvector | 64.7 | 832,200,704 | 0.545 / 0.913 / 2.196 / 6.291 | 691 | 1.448 / 1.449 / 1.642 |
| hnswsq plain | 66.9 | 860,176,384 | 0.572 / 0.990 / 2.317 / 6.252 | 696 | 1.436 / 1.420 / 1.620 |
| hnswsq ieeefp16 | 108.5 | 564,215,808 | 0.487 / 0.933 / 2.068 / 5.215 | 336 | 2.972 / 3.099 / 3.420 |
| hnswsq ieeefp8 | 188.0 | 431,980,544 | 0.567 / 1.156 / 2.828 / 7.740 | 338 | 2.962 / 2.980 / 3.392 |
| hnswsq f8 (weighted-pairwise default) | 113.8 | 450,797,568 | 0.492 / 0.902 / 2.033 / 5.345 | 457 | 2.190 / 2.178 / 2.467 |
| hnswsq sq8 (fixed [0,255] int8) | **114.4** | 469,581,824 | 0.508 / 0.976 / 2.278 / 5.448 | **504** | 1.985 / 1.971 / 2.268 |
| hnswsq sq16 (fixed [0,255] int16) | 149.2 | 666,836,992 | 0.628 / 1.103 / 2.245 / 5.657 | 404 | 2.477 / 2.482 / 2.766 |

Notes:

- **plain reaches parity everywhere**: build 66.9 vs 64.7s, queries
  equal, inserts 696 vs 691 rows/s (1.44 vs 1.45 ms/row).
- **fp16's halved memory traffic shows up**: queries beat plain at
  ef >= 40 (e.g. 5.215 vs 6.252 ms at ef640) at 65% of the size.
- **f8 and the fixed-range sq8 tie on queries** (0.492–5.345 vs
  0.508–5.448 ms — the same code space, both ahead of plain's
  0.572–6.252) at 52% of plain's size; **sq8's inserts are the fastest
  quantized layout** (504 rows/s, 1.99 ms/row, vs f8's 457) and its
  per-row size matches f8's (376 B/row).  sq16 costs the expected
  +~50% size (513 B/row) and ~10% more query time for near-plain recall.
- Quantized builds stay slower than plain (fp16 1.6x, fp8 2.8x, f8 1.7x,
  sq8 1.7x, sq16 2.2x) — the per-dimension conversion cost in the
  build's distance kernel; f8/sq8/sq16's insert rates are the best of
  the quantized layouts (buffer/backlink-bound, not distance-bound).

### perf attribution (2026-09-16, `perf record` on the release build)

Single-backend sq8 build (90s sample): **`distance_encoded_direct` 69.7%**,
then visited-table insert (4.8%), candidate loading (5.3%),
`select_neighbors` sort (1.8%) and heap pops (1.7%) — the build is
distance-bound.  ef-640 scan (20s sample): pairwise distance kernel
27.4%, `load_element_impl` 19.3%, `PinBuffer` 18.0%, `LWLockRelease`
8.8%, `LWLockAttemptLock` 6.9% — the scan is buffer-manager-bound with
the distance as the largest single symbol.

Two changes followed:

- **sq8 graph mutation now uses the integer pairwise distance** — for
  `sq8` the build's query is the decoded vector, which is always
  integral (decode = the codes themselves), so the integer pairwise is
  value-identical to the f32 scalar form and the graph is unchanged.
  Matrix: sq8 build 125.2 → **114.4s**, inserts 461 → **504 rows/s**,
  recall identical (0.993/0.999/1.0 at ef 160/320/640).  `sq16` keeps
  the scalar mutation path: its decoded domain is fractional
  (code/128), where the integer form deviates from the f32 form at the
  ulp level and those near-tie flips fragmented the graph (recall
  collapsed to ~0.72 in the local suite — the same signature the old
  ±1-range pairwise build showed).
- **AVX-512 distance kernels** (the box has avx512f/bw/vl): 16-lane
  versions of the weighted f8 pairwise, the fixed sq8/sq16 pairwise and
  the fixed decode distances, with runtime dispatch.  The distance
  share of an ef-640 scan dropped from 27% to **~2.5%**, leaving the
  scan ~60% PostgreSQL buffer management (PinBuffer/LWLock/UnpinBuffer)
  and ~20% element loading — the distance math is no longer a visible
  scan cost.  The buffer-manager path (pgvector-identical) is the next
  frontier.
