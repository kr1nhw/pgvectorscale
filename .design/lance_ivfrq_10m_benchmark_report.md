# Lance IVF-RQ (1 segment) Benchmark Report — BIGANN 10M

Date: 2026-08-27
Server: `root@113.44.106.182` (Huawei Cloud EulerOS 2.0, 16 vCPU / 60 GB RAM)
Scope: Lance vector index `ivfrq` (IVF-RQ), **1 segment**, recall@10 + latency (p50/p95/p99) + build time on BIGANN 10M.

## 1. Summary

| Metric | Value |
|---|---|
| Dataset | BIGANN 10M (first 10M of the 100M set), 128-dim float32, L2 |
| Table write (10M rows, **1 fragment / 1 segment**) | **2.62 s** (5.148 GB) |
| IVF-RQ build, `num_partitions=3162`, `num_bits=1` | **42.9 s** (+310 MB index) |
| IVF-RQ build, `num_partitions=3162`, `num_bits=8` | **79.3 s** (+1.82 GB index) |
| Best recall@10 (b=8, nprobes=160, refine=4) | **0.9974** @ p99 **3.91 ms** |
| Good default (b=8, nprobes=20, no refine) | 0.899 @ p99 2.57 ms |
| Flat / no-index baseline | p50 **937 ms** / p99 **949 ms** (~230–450× slower) |

Key findings:

- A true **1-segment** dataset requires bypassing `lancedb.add()`: the high-level API splits large
  writes into ~1M-row fragments (10M rows → 10 fragments). Writing with
  `lance.write_dataset(..., max_rows_per_file=11M)` yields exactly **1 fragment / 1 data file**,
  verified post-index (`num_indexed_fragments = 1`).
- **1-bit RabitQ distance estimates are too coarse to rank top-10 without refinement**: recall caps
  at ~0.46 regardless of `nprobes`. `refine_factor` (exact re-scoring) is mandatory for `num_bits=1`
  and lifts recall to 0.97 (np=160, rf=16).
- **`num_bits=8` is the accuracy/speed sweet spot**: ~0.90 recall with **no** refinement at only
  20 probes (~2.6 ms p99), and 0.997 recall at np=160 + rf=4 in <4 ms p99.

## 2. Environment

| Component | Version / detail |
|---|---|
| OS | Huawei Cloud EulerOS 2.0 (x86_64, glibc 2.28) |
| CPU / RAM | 16 vCPU (2×HT, 8 cores), 60 GB RAM, ~530 GB free on `/data1` |
| Python | Miniconda 3.14.6 (`/root/miniconda3`) |
| `lancedb` | **0.37.1** (cp310-abi3 manylinux x86_64 wheel) |
| `pylance` (lance core) | **10.0.0** — i.e. `lance-format/lance` **v10.0.0** (cp310-abi3 wheel) |
| `numpy` / `pyarrow` | 2.5.2 / 25.0.1 |

Notes on packaging:

- `lancedb==10.0.0` does **not** exist on PyPI. "lance 10.0.0" maps to **`pylance==10.0.0`**, the
  Python bindings for the `lance-format/lance` core release **v10.0.0**; `lancedb 0.37.1` is its
  matched high-level API (its test extra pins `pylance==10.0.0`).
- `pylance 10.0.0` requires Python ≥ 3.10; the box only had Python 3.9 (Miniconda was installed).
- PyPI (`files.pythonhosted.org`) is slow from this China-region host (~36 KB/s) and pip's own
  downloader stalls on >10 MB files; all wheels were fetched with `curl` from the Aliyun PyPI
  mirror and installed offline (`pip install --no-index --find-links ...`).

## 3. Dataset and ground truth

- Source files on the box: `/data1/bigann_100m_vectors.txt`, `/data1/bigann_queries.txt`,
  `/data1/bigann_ground_truth.txt`, and `/data1/bench/items10m.bin` (PostgreSQL COPY binary of
  `(id int4, vector(128) float32 BE)` for rows 0..9,999,999).
- Vectors were parsed from `items10m.bin` (row layout: 2 B nfields, 4 B len, 4 B id, 4 B len,
  4 B dim, 2 B pad, 512 B f32 data; 2 B trailer) into `vectors.npy` (10M×128 f32) + `ids.npy`.
- Queries parsed from `bigann_queries.txt` (10k queries, 128-dim).
- **Gotcha**: `bigann_ground_truth.txt` on the box is the **100M** ground truth (contains ids ≥ 10M),
  so it cannot be used directly for the 10M subset. Exact top-10 for the 10M set was computed from
  scratch in numpy (10 workers × BLAS matvec, `d = ‖x‖² − 2·x·q + ‖q‖²`, `argpartition`), ~137 s for
  1,000 queries, and **verified 10/10 per query against the existing PostgreSQL exact scan**
  (`gt_10m` table, computed by pgvectorscale sequential scan).
- Benchmark protocol: 1,000 queries, k=10, recall@10 = mean fraction of the 10 exact nearest
  neighbors returned; latencies measured per query (wall clock of `search().limit(10).nprobes(n)
  .refine_factor(r)`) after a 50-query warmup; p50/p95/p99 reported.

## 4. Method: single-segment write and index configuration

```python
# 1) single-fragment write (1 segment)
lance.write_dataset(tbl, uri, mode="create", max_rows_per_file=11_000_000)

# 2) IVF-RQ index
table.create_index("vector", config=IvfRq(num_partitions=3162, num_bits=1))  # or num_bits=8
```

- `max_rows_per_file` default is 1,048,576 → 10M rows would land in 10 fragments; raised to 11M to
  force **1 fragment / 1 data file** (= 1 segment). Verified: `len(ds.get_fragments()) == 1`,
  `num_indexed_fragments == 1` in index statistics.
- `IvfRq` options used: `num_partitions = 3162` (= √10M, the default), `num_bits ∈ {1, 8}`,
  `distance_type = "l2"` (default), all other params default
  (`sample_rate=256`, `max_iterations=50`, `target_partition_size=8192`).
- Query knobs: `nprobes` (partitions visited) and `refine_factor` (exact re-scoring factor).

## 5. Build results

| Stage | Time | Size |
|---|---|---|
| Write 10M rows (1 fragment) | 2.62 s | 5,148,081,150 B (4.79 GiB) table |
| IVF-RQ build, b=1 | 42.87 s | +309,986,604 B (~296 MiB) index; table 5.458 GB |
| IVF-RQ build, b=8 | 79.34 s | +1,820,446,631 B (~1.70 GiB) index; table 6.5 GB |

Index statistics (b=8): `num_indexed_rows = 10,000,000`, `num_indexed_fragments = 1`,
`num_unindexed_rows = 0`, index file version V3, final kmeans loss reported by
`index_statistics()`.

## 6. Query results — recall@10 and latency (1,000 queries, k=10)

### 6.1 IVF-RQ, num_bits=1 (needs refinement)

| nprobes | refine | recall@10 | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
|---|---|---|---|---|---|---|
| 5 | 1 | 0.4155 | 1.99 | 2.25 | 2.42 | 1.99 |
| 5 | 4 | 0.6445 | 1.91 | 2.10 | 2.26 | 1.96 |
| 5 | 16 | 0.6991 | 2.26 | 2.45 | 2.60 | 2.27 |
| 10 | 1 | 0.4460 | 1.91 | 2.16 | 2.32 | 1.92 |
| 10 | 4 | 0.7325 | 1.96 | 2.14 | 2.33 | 1.97 |
| 10 | 16 | 0.8224 | 2.34 | 2.51 | 2.68 | 2.34 |
| 20 | 1 | 0.4613 | 1.95 | 2.18 | 2.31 | 1.96 |
| 20 | 4 | 0.7865 | 2.06 | 2.27 | 2.43 | 2.08 |
| 20 | 16 | 0.9071 | 2.42 | 2.63 | 2.80 | 2.43 |
| 40 | 1 | 0.4609 | 2.11 | 2.34 | 2.55 | 2.12 |
| 40 | 4 | 0.8022 | 2.27 | 2.48 | 2.69 | 2.28 |
| 40 | 16 | 0.9455 | 2.62 | 2.84 | 3.03 | 2.63 |
| 80 | 1 | 0.4586 | 2.44 | 2.69 | 2.85 | 2.46 |
| 80 | 4 | 0.8071 | 2.61 | 2.84 | 3.09 | 2.63 |
| 80 | 16 | 0.9624 | 2.99 | 3.26 | 3.47 | 3.01 |
| 160 | 1 | 0.4575 | 3.15 | 3.47 | 3.64 | 3.17 |
| 160 | 4 | 0.8062 | 3.33 | 3.59 | 3.86 | 3.35 |
| 160 | 16 | 0.9672 | 3.68 | 3.98 | 4.16 | 3.70 |

Without refinement the recall saturates at ~0.46 regardless of `nprobes` — the 1-bit quantized
ranking is the bottleneck, not partition coverage. `refine_factor=16` recovers 0.97 recall.

### 6.2 IVF-RQ, num_bits=8

| nprobes | refine | recall@10 | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
|---|---|---|---|---|---|---|
| 5 | 1 | 0.6930 | 2.34 | 2.86 | 3.07 | 2.32 |
| 5 | 4 | 0.7034 | 2.01 | 2.21 | 2.40 | 2.06 |
| 10 | 1 | 0.8135 | 2.00 | 2.45 | 2.61 | 2.06 |
| 10 | 4 | 0.8298 | 2.09 | 2.28 | 2.45 | 2.10 |
| 20 | 1 | 0.8988 | 2.04 | 2.45 | 2.57 | 2.08 |
| 20 | 4 | 0.9223 | 2.21 | 2.42 | 2.61 | 2.22 |
| 40 | 1 | 0.9376 | 2.15 | 2.38 | 2.59 | 2.17 |
| 40 | 4 | 0.9685 | 2.39 | 2.65 | 2.81 | 2.41 |
| 80 | 1 | 0.9560 | 2.44 | 2.69 | 2.90 | 2.46 |
| 80 | 4 | 0.9899 | 2.69 | 3.00 | 3.24 | 2.71 |
| 160 | 1 | 0.9619 | 3.00 | 3.26 | 3.55 | 3.01 |
| 160 | 4 | **0.9974** | 3.28 | 3.63 | **3.91** | 3.31 |

### 6.3 Flat (no index) baseline

Exact scan over the 10M×128 table, k=10, 100 queries:

| p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
|---|---|---|---|
| 937.1 | 947.1 | 949.2 | 935.6 |

## 7. Observations / conclusions

1. **1 segment is achievable and verified.** Use `lance.write_dataset(max_rows_per_file ≥ N)` (or
   equivalent low-level write) rather than `lancedb.add()`; the latter splits 10M rows into
   10 × 1M-row fragments. Single-fragment build+query has identical recall to the 10-fragment
   layout at slightly lower latency.
2. **IVF-RQ 1-bit requires exact re-scoring** (`refine_factor`) for usable accuracy on BIGANN 10M;
   without it recall plateaus at ~0.46. This mirrors the pgvectorscale RaBitQ design, where the
   final top-k is always re-ranked by exact full-vector distance.
3. **8-bit IVF-RQ hits ~0.90 recall with no refinement** at only 20 probes (~2.6 ms p99) and
   0.9974 at np=160 + rf=4 (~3.9 ms p99). Build time 79 s (vs 43 s for 1-bit), index 1.82 GB
   (vs 310 MB).
4. **Speedup vs flat**: ~230–450× (0.9–4 ms vs ~940 ms per query), consistent with an
   IVF-partitioned, quantized+rescored index over 10M×128 float32.

## 8. Artifacts

- Results (JSON): `/data1/lance_bench/FINAL_RESULTS.json` on the server
  (`bench_results_s1.json`, `sweep_results.json`, `bench_results_b8_flat.json`,
  `flat_baseline.json`, logs `bench_s1.log` / `bench_b8.log` / `sweep.log` / `flat.log`)
- Tables (Lance, 1 segment): `/data1/lance_bench/lancedb_s1/` (`bigann10m_s1.lance`,
  `flat_table.lance`)
- Scripts on the server: `/root/prep_bigann10m.py` (data prep), `/root/exact_gt.py` (exact GT),
  `/root/lance_bench.py` / `/root/bench_s1.py` (1-segment bench), `/root/bench_b8.py` (b=8 + flat),
  `/root/sweep.py` (nprobes × refine sweep)
- Prepared data: `/data1/lance_bench/{ids,vectors,queries,gt_top10_exact}.npy`
- Reproducibility commands:

```bash
/root/miniconda3/bin/python3 /root/prep_bigann10m.py /data1/lance_bench
/root/miniconda3/bin/python3 /root/exact_gt.py /data1/lance_bench 1000 10
/root/miniconda3/bin/python3 /root/bench_s1.py /data1/lance_bench 1000 "5,10,20,40,80,160" "1,4,16" 3162 1
/root/miniconda3/bin/python3 /root/bench_b8.py /data1/lance_bench 1000 "5,10,20,40,80,160" "1,4" 8
```
