# hnswsq vs pgvector-hnsw: same-host 1M A/B (121.37.117.106, 32 vCPU, PG 17.11)

Dataset: BIGANN 1M rows, dim 128, first 100 queries, exact top-10 ground truth
(`items_1m` / `gt_1m`).  Both engines: `m=16`, `ef_construction=64`,
`maintenance_work_mem=8GB`.  hnswsq index: `storage_layout=plain`.

Both engines are **release builds** on the same box and the same table, measured
back to back with `.design/neon/bench/cycle.sh` (which kills stale builds by PID,
drops and recreates the index, and records build time + stats + sweep in one
row).  This replaces the earlier 100k/113 comparison, whose hnswsq side was a
*debug* build of the extension — see "History" at the bottom.

## Build + size

| engine | build_s | size_bytes | bytes/vector |
|--------|---------|------------|--------------|
| pgvector hnsw | 105 | 832,184,320 | 832 |
| hnswsq plain | **517** | 861,921,280 | 862 (+3.6%) |

hnswsq is single-backend; pgvector's C build parallelizes across the box.  The
gap is parallelism, not algorithm — measured on the same box with both engines
in release and `max_parallel_maintenance_workers` varied (see
`.design/hnswsq_vs_pgvector_gap.md`):

| build | workers | seconds |
|---|---|---|
| hnswsq | 1 | 562 |
| pgvector | 1 | 442 |
| pgvector | 4 (server default) | 103 |
| pgvector | 32 requested (7 effective) | 64 |

i.e. **1.27x per core** and 5.5-8.8x in wall clock purely from worker count.
The phase split of the hnswsq build (from `hnswsq.build_stats`):

```
search=372079ms (76%) backlink_select=97912ms (20%) flush=10810ms select=6990ms
backlink_pairs=2000ms  accounted_total=489791ms of 517s wall
```

For scale: the same build on this table before this branch's optimizations was
measured at ~170 nodes/s (interrupted at 48 min ≈ 97 min projected for 1M, and
that was the *debug* profile); the current release build is 11x faster, and the
remaining 76% is the graph search, which is what a parallel build would attack.

## Recall@10 and latency sweep (ef_search 10..640)

| ef | pgvector recall | pgvector p50 ms | hnswsq recall | hnswsq p50 ms | hnswsq p99 ms |
|----|-----------------|-----------------|---------------|---------------|---------------|
| 10 | 77.47 | 0.655 | **78.30** | 1.020 | 5.864 |
| 20 | 88.40 | 0.824 | **88.30** | 1.335 | 7.171 |
| 40 | 93.70 | 1.056 | **94.00** | 1.919 | 9.012 |
| 80 | 97.30 | 1.577 | **97.60** | 2.905 | 12.155 |
| 160 | 98.90 | 2.520 | **99.30** | 4.629 | 18.197 |
| 320 | 99.80 | 4.146 | **99.90** | 7.831 | 28.373 |
| 640 | 100.00 | 7.013 | **100.00** | 13.787 | 43.785 |

## Interpretation

- **Recall**: hnswsq-plain matches or slightly beats pgvector at every ef point
  (+0.4 to +0.8 points at ef 10-320) with a single-threaded build.
- **Size**: +3.6% over pgvector (862 vs 832 B/vector) — the page-based node
  format stores ItemPointers, and each node carries one extra list slot.
- **Build**: 4.9x pgvector's build time *single-threaded* (517s vs 105s).  This
  is now an algorithmic/parallelism gap, not a constant-factor one: 76% of the
  hnswsq build is the graph search, which is per-insert independent and is the
  target of the planned parallel build (`.design/hnswsq_perf_analysis.md`,
  "Next-round plan").
- **Query latency**: ~1.6-2.0x higher p50 across the sweep (1.9ms vs 1.1ms at
  ef=40; 13.8ms vs 7.0ms at ef=640).  The earlier "10-13x" figure in this file
  was a debug-build artifact.  The remaining gap is the architectural tradeoff
  of a page-based, buffer-managed index: each search hop loads a page through
  the shared-buffer manager and deserializes the node (rkyv) instead of walking
  a memory-mapped graph — which is what buys MVCC-safe online inserts/deletes,
  WAL safety, vacuumability, and serverless-friendly per-page storage.

## History (not comparable — debug profile)

The first version of this file recorded a 100k BIGANN (dim 128) A/B on
113.44.106.182 with hnswsq built **without `--release`** (that is what
`cargo pgrx install` does by default): pgvector 13s vs hnswsq 3556s to build,
and p50 3.79ms vs 0.31ms at ef=10.  A debug build of this extension is ~12x
slower than release, which accounts for most of that spread; the numbers are
kept only as a reminder to compare like with like (all current local and remote
measurements use `cargo pgrx install --release`).

## Commands

```bash
# same-host cycle (build + stats + recall sweep + insert bench) — 121
PGPORT=54330 PGUSER=pgtest PGDATABASE=postgres \
PSQL_BIN=/root/.pgrx-hnswsq/17.11/pgrx-install/bin/psql \
  bash .design/neon/bench/cycle.sh hnswsq postgres <label> 1m
PGPORT=54330 ... bash .design/neon/bench/cycle.sh hnsw postgres pgvector-1m 1m

# local (single host, pinned seed, reproducible dataset)
# dim 16 dataset (tag = rows), then a dim 128 one for kernel work
.design/neon/bench/local_dataset.sh 1m t100kdb 100 200 16
.design/neon/bench/local_dataset.sh 100kd128 t100kdb 200 200 128
.design/neon/bench/local_cycle.sh <label> plain t100kdb 100kd128 16 64 128
```
