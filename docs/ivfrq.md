# ivfrq — the IVF + RaBitQ index access method

`ivfrq` is this repository's IVF (inverted file) index with **RaBitQ**
quantization: vectors are partitioned into `lists` clusters (k-means
centroids), and each vector's residual is quantized to `num_bits` bits per
dimension. Searches probe the nearest lists using a SIMD "FastScan" over the
packed codes (Lance-style LUT sums), then re-rank candidates with the stored
quantized codes. The index is ~15× smaller than the raw table at 1 bit
(3.3 GB for BIGANN-100M's 51 GB) and its scans bypass the buffer manager with
vectored `smgrreadv` reads.

## 1. Setup

```sql
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector provides the vector type
CREATE EXTENSION IF NOT EXISTS vectorscale; -- registers the ivf access method
```

## 2. Creating an index

```sql
CREATE INDEX items_idx ON items
  USING ivf (embedding vector_l2_ops)
  WITH (lists = 1000, num_bits = 1);
```

Notes:

- The operator class must be named explicitly — the ivf opclasses are not
  `DEFAULT` (pgvector convention): `vector_l2_ops` (L2, `<->`),
  `vector_cosine_ops` (cosine, `<=>`), `vector_ip_ops` (inner product, `<#>`).
- The build is a **streaming two-pass** pipeline with bounded memory
  (reservoir sampling + batched segment sealing), so very large tables build
  on modest hardware (BIGANN-100M: ~30 min on 16 vCPU / 60 GB).

### Options (WITH clause)

| option | range | default | meaning |
|---|---|---|---|
| `lists` | 1 – 32768 | 100 | number of inverted lists (k-means centroids). More lists → finer partition → higher recall ceiling, larger directory; ~100–1000 is a good start for ≤10M rows. |
| `num_bits` | 1 – 8 | 1 | RaBitQ bits per dimension: **1** (smallest/fastest), **2**, **4**, **8** (highest accuracy, largest). Size scales linearly with bits. Values 3/5/6/7 fall back to the 1-bit path — use 1, 2, 4, or 8. |
| `sample_size` | 0 – 1000000 | 0 (= auto, 30000) | vectors reservoir-sampled for k-means training. `0` uses the build default (30000); set it explicitly to trade centroid quality against build time (e.g. `sample_size = 100000` for ≥10M rows). Tables smaller than the value are fully sampled. |
| `storage_layout` | — | `rabitq_compression` | the ivf access method is a RaBitQ index; this is its only layout. |

```sql
-- example: 100M-scale settings
CREATE INDEX items_idx ON items
  USING ivf (embedding vector_l2_ops)
  WITH (lists = 1000, num_bits = 1, sample_size = 100000);
```

## 3. Querying

The index serves `ORDER BY embedding <-> query LIMIT k` plans. Useful GUCs
(all `SET`-able per session):

| GUC | range | default | meaning |
|---|---|---|---|
| `ivf.probes` | 1 – 32768 | 1 | lists probed per query — the main recall/speed dial |
| `ivf.top_k` | 1 – 1000000 | 1000 | candidate heap size kept per query before re-ranking; must be ≥ your `LIMIT` |
| `ivf.max_probes` | 1 – 32768 | 32768 | cap for `ivf.iterative_scan` probe growth |
| `ivf.iterative_scan` | 0 / 1 | 0 | grow probes iteratively until enough results pass re-check |
| `ivf.seal_threshold` | 1 – 1000000 | 4096 | insert-buffer size before a list segment is sealed (DML path) |

```sql
SET enable_seqscan = off;   -- force the index on small tables
SET ivf.probes = 64;
SET ivf.top_k = 1000;

SELECT id FROM items
ORDER BY embedding <-> '[0.1, 0.2, ...]'::vector
LIMIT 10;

EXPLAIN (ANALYZE, BUFFERS) SELECT id FROM items
ORDER BY embedding <-> '[0.1, 0.2, ...]'::vector LIMIT 10;
```

## 4. Tuning: recall vs latency

The recall curve is controlled almost entirely by `probes` (and `lists` at
build time). Measured anchors (BIGANN, 100 queries, real ground truth,
vanilla PG 17, x86):

| | probes=1 | probes=8 | probes=64 | probes=256 |
|---|---|---|---|---|
| 10M rows, lists=1000 | 47.1% / 1.6 ms | 88.0% / 3.1 ms | 99.3% / 6.4 ms | 99.8% / 15.2 ms |
| 100M rows, lists=1000 | 54.5% / 2.2 ms | 90.4% / 5.6 ms | **98.3% / 29.3 ms** | 98.3% / 109.6 ms |

Rules of thumb:

- Start with `probes` ≈ 10–20 for interactive use; raise toward 64–128 when
  recall matters.
- If recall plateaus below your target (as at 98.3% on 100M with 1000 lists),
  increase `lists` and/or `sample_size` — the partition granularity is the
  ceiling, not `probes`.
- `num_bits` trades index size and estimate quality: 1-bit is the size/speed
  sweet spot for large datasets; 4/8-bit help smaller, accuracy-critical sets.
- `ivf.top_k` only needs to exceed your `LIMIT` by a margin; very large values
  slow the candidate heap.

## 5. Maintenance notes

- `VACUUM` retires dead entries and seals/merges segments; a freshly built
  index still has rows in its active (unsealed) buffer, which is scanned
  through the slower buffer-manager path — run `VACUUM` after bulk loads
  before measuring steady-state query performance.
- On Neon, run pageserver compaction after large builds/loads (pages live in
  WAL-only layers until imaged) and size the local file cache generously —
  see `.design/neon/OPTIMIZATION-PLAN.md` and `.design/neon/scripts/RECOMMENDED-SETUP.md`.
- Index size scales with `num_bits`: BIGANN-100M at 1 bit ≈ 3.3 GB
  (~15.5× smaller than the table).

## 6. Benchmark harness

`.design/neon/bench/run_sweep.sh` drives recall@10 + p50/p99 sweeps
(`ivf.probes` 1..256) against `items_10m`/`bench_queries`/`gt_10m` tables;
see `.design/neon/bench/RESULTS.md` (10M, vanilla vs Neon-on-k8s) and
`.design/neon/bench/RESULTS-100M.md` (100M with the real BIGANN ground truth).
