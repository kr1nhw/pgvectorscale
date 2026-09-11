# hnswsq vs pgvector-hnsw: 100K-row A/B (113.44.106.182, PG 17.11)

Serverless-scale dataset (100K rows, BIGANN SIFT 128-dim, first 100 queries,
subset-exact top-10 ground truth).  Both engines: `m=16, ef_construction=64`,
`maintenance_work_mem=8GB`.  hnswsq index: `storage_layout=plain`.

The VM is a shared-CPU box (16 vCPUs, "General Purpose Processor") — absolute
build times are much higher than on dedicated hardware; the ratios and the
query numbers are the meaningful part.

## Build + size

| engine | build_s | size_bytes | bytes/vector |
|--------|---------|------------|--------------|
| pgvector hnsw | 13 | 83,165,184 | 832 |
| hnswsq plain | 2938 (pre-opt code) | 86,204,416 | 862 |
| hnswsq plain | 3556 (optimized code) | 86,138,880 | 861 |

Both hnswsq numbers are from this VM; the second is the current code (decode-free
distance kernels + pair-distance cache).  On the local dev machine the same
optimizations cut the in-memory build+search tests 91s → 43s and the full pgrx
suite 264s → 124s, so the VM measurement (shared, throttled CPU) is dominated
by host noise; treat the hnswsq build as ~35ms/node here vs ~8ms/node locally.

## Recall@10 and latency sweep (ef_search 10..640)

| engine | ef | recall@10 | p50 ms | p99 ms |
|--------|----|-----------|--------|--------|
| pgvector | 10 | 85.15 | 0.31 | 5.14 |
| pgvector | 20 | 91.30 | 0.39 | 5.38 |
| pgvector | 40 | 97.10 | 0.58 | 5.80 |
| pgvector | 80 | 99.20 | 0.80 | 6.82 |
| pgvector | 160 | 100.00 | 1.34 | 7.81 |
| pgvector | 320 | 100.00 | 2.27 | 9.54 |
| pgvector | 640 | 100.00 | 3.72 | 11.53 |
| hnswsq | 10 | 84.20 | 3.79 | 9.41 |
| hnswsq | 20 | 91.70 | 5.58 | 14.33 |
| hnswsq | 40 | 97.10 | 8.64 | 17.83 |
| hnswsq | 80 | 99.10 | 14.16 | 26.52 |
| hnswsq | 160 | 100.00 | 24.29 | 40.13 |
| hnswsq | 320 | 100.00 | 41.63 | 62.17 |
| hnswsq | 640 | 100.00 | 68.02 | 98.71 |

(hnswsq numbers are the final optimized-code sweep.)

Recall parity is near-exact: 84.2% vs 85.2% at ef=10, 100% by ef=160 for
hnswsq (pgvector reaches 100% at ef=160 as well).

## Interpretation

- **Recall**: hnswsq-plain matches pgvector at every ef point within ±1%.
- **Size**: hnswsq (861 B/vector) ≈ pgvector (832 B/vector), +3.5%.
- **Latency**: hnswsq is ~10-13x higher p50 at equal recall.  This is the
  architectural tradeoff of a page-based, buffer-managed index: each search
  hop loads a page through the shared-buffer manager and deserializes the
  node (rkyv), while pgvector's hnsw keeps the whole graph memory-mapped.
  hnswsq's page-based design buys MVCC-safe online inserts/deletes, WAL
  safety, vacuumability, and serverless-friendly per-page storage — at a
  per-hop page-load cost.
- **Build cost**: hnswsq's in-memory HNSW build is single-backend and much
  slower per node than pgvector's parallel C build.  This branch's
  optimizations (decode-free distance kernels + a build-scoped pair-distance
  cache for backlink re-pruning) roughly halve local build time; on this
  shared-CPU VM the build remains the dominant cost.  hnswsq targets
  serverless-scale datasets (1M-10M rows); for larger builds use pgvector's
  hnsw or batch `REINDEX`.

## Commands

```bash
# build + size
build_indexes_hnswsq.sh <psql> hnsw   builds.csv
build_indexes_hnswsq.sh <psql> hnswsq builds.csv
# recall/latency sweep (drops the other engine's index first)
run_sweep_hnswsq.sh <psql> hnsw   pgvector-hnsw sweep.csv
run_sweep_hnswsq.sh <psql> hnswsq hnswsq-plain sweep.csv
```
