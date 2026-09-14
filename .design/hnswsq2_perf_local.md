# hnswsq (Rust port) vs pgvector — local A/B, plain layout

Measured on the manual dev cluster (PostgreSQL 18.4, Homebrew, aarch64 macOS,
Apple M4 Pro), `postgres -D /tmp/hnswsq2db -p 54329 -k /tmp/pgtestk`, extension
installed at `/opt/homebrew/lib/postgresql@18/vectorscale-0.9.0.dylib`,
pgvector 0.8.5.

Both engines are **release** builds. pgvector ships a release dylib via
Homebrew; ours is `cargo build --release --features "pg18 pg_test"
--no-default-features` copied over the installed dylib.

> **Trap**: `cargo pgrx test pg18` (and any pgrx install step) overwrites the
> installed dylib with the **debug** build (~7.6 MB vs ~2.5 MB release) —
> queries and builds then measure 15-40x slow. Always `cp
> target/release/libvectorscale.dylib /opt/homebrew/lib/postgresql@18/vectorscale-0.9.0.dylib`
> (and verify the checksum) before benchmarking.

Dataset: `ab2`, 100,000 rows, dim 128, components i.i.d. uniform
(`generate_series` cross join), all distinct. Queries: 100 vectors (`ab2_q`,
every 1000th row). Index parameters identical: `m=16, ef_construction=64`,
`maintenance_work_mem=2GB`. One index at a time — the planner picks the
surviving one, so each engine was benchmarked with the other's index dropped.

## 1. Build

| | hnswsq | pgvector | ratio |
|---|---|---|---|
| build 100k × 128, mwm 2GB | 6.94-7.59 s | 7.54-7.69 s | **0.92-0.99x** |

## 2. Query (warm, release builds)

Batch shape (10,000 index scans per statement; 2-3 rounds each):
`SELECT sum((SELECT count(*) FROM (SELECT id FROM ab2 ORDER BY embedding <-> q.q LIMIT 10) t)) FROM ab2_q q, generate_series(1,100) g;`

| ef_search | hnswsq | pgvector | ratio |
|---|---|---|---|
| 160 | 12.34-12.94 s | 12.28-12.44 s | **1.00-1.05x** |
| 640 | 43.77-44.70 s | 43.12-43.39 s | **1.01-1.03x** |

Collect shape (100 index scans, same lateral): hnswsq 134.3 / 476.6 ms at
ef 160 / 640 vs pgvector 131.4 / 453.0 ms — same per-query cost as the batch
shape for both engines (a shape difference in earlier measurements was the
debug dylib, not the code).

## 3. Recall@10 (100 queries, exact top-10 via seq scan)

| ef_search | hnswsq | pgvector |
|---|---|---|
| 160 | 65.4 % | 66.9 % |
| 640 | 87.2 % | 86.8 % |

Parity (random data is recall-hostile; the deltas are within noise of
1000 draws).

## 4. What the search costs (sample, ef=640)

Flat profile of one warm backend, 12 s:

| work | share |
|---|---|
| `load_element_impl` (ReadBuffer + LockBuffer + distance + copy) | ~87% |
| `distance_l2_aarch64_neon` | ~11% |
| `load_unvisited_from_disk` (visited hashing per neighbor) | ~7% |
| `Visited::insert_key_hash` | ~4% |
| `BinaryHeap::pop` | ~3% |

The buffer-pin machinery is the cost for both engines; per-pin costs match
(~120-150 ns). The fixes that got us from ~1.6x to parity:

1. **Visited sizing** — `Visited::new(1000 * m * 2)` at beginscan instead of
   256. A too-small table grew + rehashed per query; the churn was ~27% of the
   profile. (pgvector: `InitVisited(ef * m * 2)`.)
2. **ElementArena** — the search materialized one `Box<Element>` per admitted
   candidate (malloc + free pair per candidate, ~4000/query at ef=640). A
   chunked bump allocator (512 × Element per 64 KiB chunk, address-stable
   pointers) removes the per-candidate malloc.
3. **Heap preallocation** — `BinaryHeap::with_capacity(ef + 1)` for C/W.

## 5. Storage layouts (IO efficiency)

Same `ab2` dataset, `m=16, efc=64`, `build_seed=20240912`, release build:

| layout | index size | bytes/vec | vs plain | query ms (ef=160, warm) |
|---|---|---|---|---|
| plain | 81,928,192 | 819 | 1.00x | 1.35 |
| ieeefp16 | 51,306,496 | 513 | 0.63x | 1.65 |
| ieeefp8 | 37,584,896 | 376 | 0.46x | 2.52 |
| f8 (sq8) | 37,593,088 | 376 | 0.46x | 1.47 |

Quantized queries cost 1.1-1.9x plain at ef=160 (the scan materializes the
encoded vectors of admitted candidates for the lower-bound emission — the
sanctioned divergence; the pgvector reference never materializes in a scan —
plus the `xs_recheckorderby` path pulls a few extra tuples). The graph also
differs per layout (neighbor selection runs on quantized distances), which
moves the expansion count a bit. The vector-byte ratios approach 0.5x/0.25x
as dim grows; at dim 128 the fixed per-node overhead (tuple headers + ~192 B
of layer-0 neighbor TIDs) dominates the remainder.

## 6. Caveats

- **Degenerate tables**: a table of 100k *identical* vectors (the original
  `ab` scratch table) is a pure tie-class pathology; both engines' behavior
  there is build-seed-sensitive noise (pgvector's own hit count varied 12x
  between builds). Use i.i.d. data for A/B.
- **Host-121 study still pending** (`.design/neon/bench/gap_study.sh`): the
  remote 1M-row BIGANN A/B at 1/2/4/8 workers against pgvector has not been
  re-run since the retirement; the old-engine numbers in
  `hnswsq_vs_pgvector_gap.md` predate the port.
