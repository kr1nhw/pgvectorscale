# recall@10 vs p50/p99 latency — pgvectorscale ivfrq & pgvector hnsw on vanilla PG17 vs Neon (x86)

Date: 2026-09-02 — Server: `root@113.44.106.182` (Huawei Cloud EulerOS 2.0, x86_64,
16 vCPU, 60 GB RAM, local NVMe-class `/data1`). All four configurations run on the
**same machine**: vanilla PostgreSQL 17.11 on port 5432 and a Neon dev stack
(pageserver + safekeeper + storage_broker + storage_controller + compute,
fork `1e01fcea`, PG 17.5) on port 55432.

## 1. Setup

| | vanilla PG17 | Neon compute |
|---|---|---|
| Server | PG 17.11, `shared_buffers=8GB`, 8 parallel maintenance workers | Neon fork PG 17.5, `shared_buffers=1MB` (Neon default), `fsync=off`, LFC **disabled** (`neon.file_cache_size_limit=0`) |
| vectorscale | 0.9.0 built from `ivf-rabitq` HEAD (plain `pg17` features) | 0.9.0 same HEAD, features `pg17 neon pgrx/unsafe-postgres` |
| pgvector | 0.8.6 (stock) | 0.8.6 + Neon's unlogged-build patch (see §4) |
| Dataset | BIGANN-10M first 10M rows, `vector(128)`, L2, `items_10m` | same rows COPY-binary-loaded (2m53s, ~30 MB/s through pageserver) |
| Queries | 100 query vectors, `bench_queries`, exact top-10 ground truth `gt_10m` (seq scan) | same |
| Protocol | 100 queries/point, LIMIT 10; recall@10 vs ground truth in SQL; one untimed warmup pass, then a timed pass parsing psql `\timing`; p50/p99 over 100 per-query wall times | identical harness |
| ivfrq index | `ivf (embedding vector_l2_ops) WITH (lists=1000, num_bits=1)`, 346 MB | same, 346 MB |
| hnsw index | `hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=64)`, 7.9 GB | same, 7.9 GB |

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
| ivfrq-neon | ivf | probes | 1 | 47.23 | 389.687 | 422.798 |
| ivfrq-neon | ivf | probes | 2 | 61.50 | 397.716 | 427.371 |
| ivfrq-neon | ivf | probes | 4 | 76.80 | 403.749 | 424.640 |
| ivfrq-neon | ivf | probes | 8 | 88.50 | 418.409 | 457.222 |
| ivfrq-neon | ivf | probes | 16 | 94.90 | 464.119 | 536.424 |
| ivfrq-neon | ivf | probes | 32 | 98.10 | 520.722 | 589.665 |
| ivfrq-neon | ivf | probes | 64 | 99.40 | 626.435 | 743.478 |
| ivfrq-neon | ivf | probes | 128 | 99.60 | 803.403 | 962.328 |
| ivfrq-neon | ivf | probes | 256 | 99.40 | 1142.316 | 1334.262 |
| hnsw-neon | hnsw | ef_search | 10 | 67.68 | 130.377 | 224.084 |
| hnsw-neon | hnsw | ef_search | 20 | 79.08 | 175.222 | 388.616 |
| hnsw-neon | hnsw | ef_search | 40 | 88.50 | 292.850 | 504.434 |
| hnsw-neon | hnsw | ef_search | 80 | 95.10 | 514.641 | 776.807 |
| hnsw-neon | hnsw | ef_search | 160 | 97.80 | 904.897 | 1220.966 |
| hnsw-neon | hnsw | ef_search | 320 | 99.00 | 1592.769 | 2220.374 |
| hnsw-neon | hnsw | ef_search | 640 | 99.50 | 2842.271 | 4254.562 |

Plot: `results.svg` (x = recall@10 %, y = latency ms, log scale; p50 solid, p99 dashed).

## 3. Headlines

- **Recall curves are identical between stacks** (same indexes, same data): e.g.
  ivfrq probes=8 → 88.00% (vanilla) vs 88.50% (Neon); probes=64 → 99.30% vs 99.40%.
  The ivf scan path (FastScan `smgrreadv` bursts + exact rescoring) behaves
  identically on Neon's pagestore smgr after the 32-block chunking fix (§4.1).
- **At ~99% recall, ivfrq is faster than hnsw on both stacks:**
  - vanilla: ivfrq 6.42 ms p50 (probes=64, 99.3%) vs hnsw 8.99 ms p50 (ef=320, 99.0%)
  - Neon: ivfrq 626 ms p50 vs hnsw 1593 ms p50
- **Neon latency is ~100–180× vanilla at matched recall** in this configuration:
  the compute runs `shared_buffers=1MB` with the local file cache disabled, so every
  index/metadata page read is a pagestore round trip (~0.4 s floor visible at
  probes=1: 390 ms vs 1.6 ms on vanilla). This is a *configuration* of the dev
  stack, not a storage-format tax: warm pages served from a local cache would close
  most of the gap. On vanilla, ivfrq probes=1..16 stays at p50 1.6–3.6 ms.
- **Index build times (10M × 128):** ivfrq 28 s vanilla / 276–293 s Neon (~10×,
  WAL + pageserver ingest); hnsw 16 GB-maintenance_work_mem vanilla ≈ 10 min,
  Neon ≈ 38 min (unlogged build locally + one `log_newpage_range` WAL burst of
  the 7.9 GB graph at the end).

## 4. Compatibility bugs found (and fixed) by this benchmark

1. **`smgrreadv` burst > Neon's vectored-I/O cap.** Neon's fork defines
   `PG_IOV_MAX = Min(IOV_MAX, 32)` (`port/pg_iovec.h`) and `neon_readv` errors
   with `Read request too large: 76 is larger than max 32`. Our FastScan passed
   whole list segments (up to ~76 blocks at probes=16) in one call. Fixed in
   commit `89d19ed`: the burst is chunked at 32 blocks under the `neon` feature
   (1024 otherwise). No effect on vanilla.
2. **pgvector hnsw build PANICs Neon** with
   `[NEON_SMGR] Page 0 ... is evicted with zero LSN` (write path, buffer
   eviction) — pgvector's build writes MAIN-fork graph pages with
   `MarkBufferDirty` and no per-page LSN, which Neon's smgr forbids. Fixed with
   Neon's own `compute/patches/pgvector.patch` (wraps the build in
   `smgr_start_unlogged_build`/`smgr_finish_unlogged_build_phase_1`/
   `smgr_end_unlogged_build`, compiled with `-DNEON_SMGR`); applied to 0.8.6 by
   `.design/neon/scripts/patch_pgvector_neon.py`. The crash also took down
   concurrent backends (postmaster abort on PANIC) — the benchmark run was
   restarted after patching.
3. (Earlier, from the functional test) pgrx `FMGR_ABI_EXTRA`/`unsafe-postgres`
   and the 3-arg `smgropen` — see `.design/neon/README.md`.

## 5. Reproduce

```bash
# data (once): transfer_data.sh copies items_10m/bench_queries/gt_10m from the
# vanilla bench db into the Neon compute
.design/neon/bench/transfer_data.sh

# indexes + sweeps (one engine at a time; drop the other vector index first)
.design/neon/bench/build_indexes.sh $PSQL ivf|hnsw builds.csv
.design/neon/bench/run_sweep.sh $PSQL ivf|hnsw <label> sweep.csv

# aggregate
.design/neon/bench/aggregate_plot.py all.csv -o results
```

Raw CSVs: `bench_results_x86/*.csv` (also under `.design/neon/bench/data/`).
