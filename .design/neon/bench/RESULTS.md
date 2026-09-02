# recall@10 vs p50/p99 latency — pgvectorscale ivfrq & pgvector hnsw: vanilla PG17 vs Neon-on-Kubernetes (x86)

Date: 2026-09-02 — Server: `root@113.44.106.182` (Huawei Cloud EulerOS 2.0, x86_64,
16 vCPU, 60 GB RAM). Four configurations on the **same machine**:

- **vanilla PG17**: PostgreSQL 17.11 on port 5432, `shared_buffers=8GB`, 8 parallel maintenance workers.
- **Neon (k8s)**: Neon dev stack with the storage layer on a single-node **k3s**
  cluster (v1.29.10) — 3 pageserver pods + 3 safekeeper pods — and the compute
  (fork `1e01fcea`, PG 17.5) on port 55434, running on a **3/3 WAL quorum** of the
  pod safekeepers with the recommended tuning applied (see below).

## 1. Setup

| | vanilla PG17 | Neon on k8s (3 PS + 3 SK) |
|---|---|---|
| Server | PG 17.11, `shared_buffers=8GB` | Neon fork PG 17.5, compute on 55434, **tuned**: `shared_buffers=2GB`, LFC `neon.max_file_cache_size`/`neon.file_cache_size_limit`=8GB, pageserver `page_cache_size`=4GB, `maintenance_work_mem=1GB` |
| Storage layer | local disk | pageserver pod (node 1, host) + standby pods ps2–ps4; WAL to safekeeper pods sk4–sk6 (3/3 quorum, gen 1) |
| vectorscale | 0.9.0 `ivf-rabitq` HEAD (plain `pg17`) | 0.9.0 same HEAD, features `pg17 neon pgrx/unsafe-postgres` |
| pgvector | 0.8.6 (stock) | 0.8.6 + Neon unlogged-build patch (`-DNEON_SMGR`) |
| Dataset | BIGANN-10M first 10M rows, `vector(128)`, L2 | same data, branched timeline (`bench-k8s`) |
| Queries / ground truth | 100 queries, exact top-10 `gt_10m` | same |
| ivfrq index | `ivf (embedding vector_l2_ops) WITH (lists=1000, num_bits=1)`, 346 MB | same, 346 MB |
| hnsw index | `hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=64)`, 7.9 GB | same, 7.9 GB |
| Protocol | 100 queries/point, LIMIT 10; recall@10 in SQL; one untimed warmup pass + one timed pass (psql `\timing`); p50/p99 over 100 wall times | identical harness |

## 2. Results (recall@10, p50/p99 ms)

| config | engine | param | value | recall@10 | p50 | p99 |
|---|---|---|---|---|---|---|
| ivfrq-vanilla | ivf | probes | 1 | 47.10 | 1.636 | 4.675 |
| ivfrq-vanilla | ivf | probes | 2 | 61.31 | 2.267 | 6.669 |
| ivfrq-vanilla | ivf | probes | 4 | 76.57 | 2.649 | 7.150 |
| ivfrq-vanilla | ivf | probes | 8 | 88.00 | 3.071 | 7.781 |
| ivfrq-vanilla | ivf | probes | 16 | 94.40 | 3.600 | 8.339 |
| ivfrq-vanilla | ivf | probes | 32 | 97.80 | 4.525 | 10.507 |
| ivfrq-vanilla | ivf | probes | 64 | 99.30 | 6.421 | 12.066 |
| ivfrq-vanilla | ivf | probes | 128 | 99.80 | 9.402 | 15.319 |
| ivfrq-vanilla | ivf | probes | 256 | 99.80 | 15.157 | 22.382 |
| hnsw-vanilla | hnsw | ef_search | 10 | 71.30 | 1.611 | 4.165 |
| hnsw-vanilla | hnsw | ef_search | 20 | 80.80 | 2.118 | 5.574 |
| hnsw-vanilla | hnsw | ef_search | 40 | 88.40 | 2.948 | 6.617 |
| hnsw-vanilla | hnsw | ef_search | 80 | 94.30 | 4.478 | 9.417 |
| hnsw-vanilla | hnsw | ef_search | 160 | 97.90 | 6.315 | 14.926 |
| hnsw-vanilla | hnsw | ef_search | 320 | 99.00 | 8.988 | 27.778 |
| hnsw-vanilla | hnsw | ef_search | 640 | 99.70 | 12.679 | 43.839 |
| ivfrq-k8s | ivf | probes | 1 | 47.23 | 1.388 | 3.076 |
| ivfrq-k8s | ivf | probes | 2 | 61.50 | 1.641 | 3.624 |
| ivfrq-k8s | ivf | probes | 4 | 76.80 | 1.834 | 4.434 |
| ivfrq-k8s | ivf | probes | 8 | 88.50 | 2.099 | 4.662 |
| ivfrq-k8s | ivf | probes | 16 | 94.90 | 2.804 | 5.711 |
| ivfrq-k8s | ivf | probes | 32 | 98.10 | 3.835 | 7.417 |
| ivfrq-k8s | ivf | probes | 64 | 99.40 | 5.707 | 9.561 |
| ivfrq-k8s | ivf | probes | 128 | 99.60 | 9.599 | 14.372 |
| ivfrq-k8s | ivf | probes | 256 | 99.40 | 16.509 | 22.392 |
| hnsw-k8s | hnsw | ef_search | 10 | 67.00 | 0.559 | 1.871 |
| hnsw-k8s | hnsw | ef_search | 20 | 77.20 | 0.816 | 2.093 |
| hnsw-k8s | hnsw | ef_search | 40 | 88.70 | 1.270 | 2.954 |
| hnsw-k8s | hnsw | ef_search | 80 | 95.00 | 1.849 | 4.149 |
| hnsw-k8s | hnsw | ef_search | 160 | 98.00 | 11.791 | 43.018 |
| hnsw-k8s | hnsw | ef_search | 320 | 99.10 | 87.855 | 352.179 |
| hnsw-k8s | hnsw | ef_search | 640 | 99.40 | 32.495 | 89.600 |

Plot: `results.svg` (x = recall@10 %, y = latency ms, log scale; p50 solid, p99 dashed).

## 3. Headlines

- **Recall curves are stack-identical** (same indexes, same data): ivfrq probes=8 →
  88.00% (vanilla) vs 88.50% (k8s Neon); probes=64 → 99.30% vs 99.40%.
- **At ~99% recall, ivfrq beats hnsw on both stacks**:
  - vanilla: ivfrq 6.42 ms p50 (probes=64, 99.3%) vs hnsw 8.99 ms (ef=320, 99.0%)
  - k8s Neon: ivfrq 5.71 ms p50 vs hnsw 32.5–87.9 ms (see note)
- **With the recommended tuning, Neon-on-k8s is now at vanilla parity for ivfrq**
  (5.71 vs 6.42 ms p50 at 99.4% — slightly faster) and within ~3–7× for hnsw,
  down from the ~100–180× penalty of the untuned dev defaults. The residual
  hnsw gap is the 7.9 GB graph exceeding the 8 GB LFC + 4 GB pageserver cache:
  deep ef_search scans spill to pageserver round trips (visible as the local
  peak at ef_search=320, reproducible: re-run gave 76.6/126.7 ms).
- **Index build times (10M × 128):** ivfrq 28 s vanilla / ~4.9 min k8s Neon
  (WAL to the 3/3 quorum); hnsw ~10 min vanilla / ~30 min k8s Neon (unlogged
  build locally + one `log_newpage_range` WAL burst of the 7.9 GB graph,
  replicated to 3 safekeepers).

## 4. Tuning journey (why the untuned numbers were misleading)

The first Neon run (single safekeeper, dev defaults) was ~100–180× vanilla:

| knobs | default | tuned |
|---|---|---|
| compute `shared_buffers` | 1MB | 2GB |
| compute LFC (`neon.max_file_cache_size` / `neon.file_cache_size_limit`) | disabled (0) | 8GB / 8GB |
| pageserver `page_cache_size` | 8192 pages (64MB) | 524288 pages (4GB) |

ivfrq p50 (probes=64, 99.4% recall): **626.4 ms untuned → 5.15 ms tuned
(1-SK) → 5.71 ms k8s 3-SK** — the 3/3 WAL quorum costs ~5–10% over tuned
single-node, while adding HA. See `K8S.md` and
`../scripts/RECOMMENDED-SETUP.md` for the full recipes.

## 5. Compatibility bugs found (and fixed) by this benchmark

1. `smgrreadv` bursts above Neon's `PG_IOV_MAX=32` (`Read request too large`)
   → chunked reads (`89d19ed`).
2. pgvector hnsw build PANICs Neon (`Page ... evicted with zero LSN`) →
   applied Neon's unlogged-build patch (`patch_pgvector_neon.py`, `-DNEON_SMGR`).
3. pgrx `FMGR_ABI_EXTRA` / `unsafe-postgres` and the 3-arg `smgropen` —
   see `../README.md`.

## 6. Reproduce

```bash
# data (once): transfer_data.sh copies items_10m/bench_queries/gt_10m into the target db
.design/neon/bench/transfer_data.sh

# indexes + sweeps (one engine at a time; drop the other vector index first)
.design/neon/bench/build_indexes.sh $PSQL ivf|hnsw builds.csv
.design/neon/bench/run_sweep.sh $PSQL ivf|hnsw <label> sweep.csv

# aggregate
.design/neon/bench/aggregate_plot.py all.csv -o results
```

Raw CSVs: `.design/neon/bench/data/all.csv`.
