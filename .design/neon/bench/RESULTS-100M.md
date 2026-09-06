# BIGANN-100M benchmark — ivfrq recall@10 / p50 / p99 (real ground truth)

Date: 2026-09-06 — Server: `root@113.44.106.182` (x86_64, 16 vCPU / 60 GB).
Vanilla PostgreSQL 17.11 (5432), database `bench100m`, pgvectorscale 0.9.0
(`ivf-rabitq` HEAD incl. the streaming two-pass build, `750af1c`).

## Setup

| item | value |
|---|---|
| dataset | BIGANN-100M, `vector(128)` (uint8 source parsed as floats), table `items_10m` (51 GB) |
| load | text COPY of `/data1/bigann_100m_vectors.txt` (bracket literals) — **12m53s** |
| ground truth | `/data1/bigann_ground_truth.txt` (10k × 100, `qid rank vid`) → `gt_10m` top-10 arrays (vid 0-based → id 1-based) |
| queries | first 100 of `/data1/bigann_queries.txt` |
| index | `ivf (embedding vector_l2_ops) WITH (lists=1000, num_bits=1)` — **3.3 GB** (vs 51 GB table ≈ 15.5×) |
| build | streaming two-pass (reservoir sample 30k + batched seal) — **30.2 min** (bounded memory; the old all-in-RAM builder OOMs at this size) |
| protocol | 100 queries/point, LIMIT 10, real-GT recall@10 in SQL, warmup + timed pass, p50/p99 over 100 wall times |

## Results

| probes | recall@10 | p50 (ms) | p99 (ms) |
|---|---|---|---|
| 1 | 54.48 | 2.151 | 4.664 |
| 2 | 69.00 | 2.581 | 5.461 |
| 4 | 82.20 | 3.547 | 7.250 |
| 8 | 90.40 | 5.551 | 12.462 |
| 16 | 95.60 | 9.293 | 19.728 |
| 32 | 97.80 | 16.636 | 31.553 |
| 64 | 98.30 | 29.253 | 49.517 |
| 128 | 98.30 | 56.864 | 81.626 |
| 256 | 98.30 | 109.595 | 142.004 |

## Notes

- Recall **caps at 98.3%** from probes=64: 1000 lists on 100M rows is a coarse
  partition (100k entries/list); the 30k-sample k-means also limits centroid
  quality. Raising `lists` (up to 32768) and/or the sample size should push
  recall toward 99%+ — next tuning step, at the cost of slower low-probes
  points (larger directory/meta scans).
- Latency scales roughly linearly with probes (~0.43 ms/probe p50) — the
  FastScan estimate pass dominates; the per-query fixed cost is ~2 ms.
- Comparison anchors: 10M run was 99.3% @ 6.4 ms p50 (p64); Lance IVF-RQ on
  10M (1 segment) reported 0.9974 recall @ p99 3.91 ms (np=160 + refine) —
  different dataset sizes, but our 100M p64 = 98.3% @ 49.5 ms p99 with zero
  refinement on a 3.3 GB index.
- The streaming build (`750af1c`) is what makes 100M feasible on 60 GB:
  peak = 1000 lists × 10k-entry buffers ≈ 600 MB + the 30k-vector sample.
