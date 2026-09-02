# Neon cluster — coherence design & optimization plan

Evidence base: BIGANN-10M (128-dim, L2) benchmarks on the k8s-based Neon dev
stack (`root@113.44.106.182`, 16 vCPU / 60 GB, QEMU guest) + `perf` profiling.
See `bench/RESULTS.md`, `bench/K8S.md`, `scripts/RECOMMENDED-SETUP.md`.

## 0. The measured baseline (what we are optimizing against)

| config | ivfrq p50 @99.4% recall | hnsw p50 @~99% recall | build (10M×128) |
|---|---|---|---|
| vanilla PG17 | 6.42 ms | 9.0 ms | ivf 28 s / hnsw ~10 min |
| Neon dev defaults (1MB shared_buffers, LFC off, 64MB PS cache) | 626 ms | ~1.6 s | ivf 4.9 min / hnsw ~30 min |
| Neon tuned (2GB sb, 8GB LFC, 4GB PS cache) | 5.15 ms | 32.5–87.9 ms | unchanged |
| Neon k8s 3-SK tuned | 5.71 ms | 32.5–87.9 ms | +5–10% WAL-sync |

`perf stat` on a probes=64 workload backend: **7.3% CPU utilized on Neon vs
100% on vanilla** (IPC ~1.9 on both). The Neon backend waits on pagestore
reads ~93% of wall time. **Every Neon-side optimization below is therefore
about the read path, not ALU.** (ALU/instruction promotion is a *vanilla*
topic — see `../vanilla-simd-promotion.md`.)

## 1. LFC (local file cache) — the primary lever

Facts from this session:

- LFC defaults to **disabled** (`neon.max_file_cache_size` = 0, PGC_POSTMASTER;
  `neon.file_cache_size_limit` = 0, PGC_SIGHUP, capped by the max).
- Enabling 8GB took ivfrq from 626 ms → 5.15 ms p50 (~120×): the 346 MB index +
  hot heap pages fit entirely in LFC, so warm scans never touch the pageserver.
- hnsw stayed ~3–7× vanilla because the working set (7.9 GB graph + 5.3 GB
  table) **exceeds 8 GB LFC + 4 GB pageserver cache**: deep ef_search walks
  spill to pageserver round trips (the reproducible ef_search=320 spike,
  76.6 ms on re-run, is a working-set boundary).

Design rules:

1. **Size LFC to the index + hot heap pages, not to shared_buffers.**
   `LFC ≈ index_size + hot_fraction × table_size + headroom`. For this box:
   8 GB for ivfrq (measured parity), **16 GB for hnsw@10M** (predicted parity;
   to be verified — the box has RAM headroom: 60 GB total, ~25–30 GB used).
2. Keep `shared_buffers` modest (2 GB is fine): LFC is the cache that matters;
   oversized shared_buffers just duplicate pages the LFC already serves.
3. **Chunk size**: `neon.file_cache_chunk_size=256` (2 MB) suits our
   contiguous segment bursts; consider 128–256 for mixed workloads. Larger
   chunks = fewer metadata lookups, more read amplification for point reads.
4. `neon.file_cache_path` may point at a **raw device** (bypasses FS overhead);
   on this VM it stays on the virtio disk — an NVMe-backed path is the next
   hardware step.
5. Bump the **pageserver page cache** too (currently 4 GB): it is the second
   tier for LFC misses. 8–16 GB is affordable here; ×N with sharding (§3).
6. `autoprewarm` (compute spec) primes LFC at endpoint start — avoids the
   cold-start latency cliff in production-like operation.

## 2. Prefetch / pipelining (hide the round trip)

The read path is: ivf scan issues `smgrreadv` segment bursts → libpagestore →
LFC hit (local pread) or pageserver fetch (0.4–1 ms localhost RTT). Today the
scan is **synchronous**: burst N+1 is issued only after burst N is consumed.

1. **Our side (vectorscale scan): software pipelining.** Issue the next
   segment's readv before scoring the current one (double-buffer the raw
   buffers in `entry.rs::read_bytes`). On LFC hits this overlaps compute with
   pread; on misses it overlaps FastScan compute with the pageserver RTT.
   Expected: hides most of the residual Neon gap for multi-segment scans
   (large `probes`), little effect on single-segment scans.
2. **Neon's `smgr_prefetch` hook.** The fork's smgr vtable has
   `smgrprefetch()` → `neon_prefetch` (libpagestore). Our scan can issue
   prefetches for the blocks of the next list(s) before `smgrreadv` — the
   pageserver/LFC fetch overlaps with scoring. This is the "coherence" win
   between our access pattern (predictable segment lists) and Neon's smgr.
3. **Deeper**: libpagestore already supports vectored page requests; verify
   whether readv bursts are sent as one pageserver RPC (they are — up to
   `PG_IOV_MAX`=32 blocks on Neon) and whether raising the compute's
   `neon.max_pageserver_parallel_requests`-style concurrency helps. Check the
   actual GUC surface in `pgxn/neon` before claiming; at minimum batch =
   1 RPC per 32-block burst.

## 3. Multi-pageserver / sharding — the scale-out coherence

Current state: 3 pageserver pods run, but the tenant's single shard sits on
node 1 — pods 2–4 are standby (re-attached, 0 tenants). Real multi-PS gains
come from **tenant sharding** (`shard_count` / `shard_stripe_size` in the
compute spec + a controller-driven split):

- **Effective cache scales out**: each pageserver caches its stripe (page
  cache 4 GB × N + LFC stays per-compute). For hnsw@10M with 3 shards the
  per-PS working set drops below the 4 GB cache → deep scans stop spilling.
- **Aggregate read bandwidth × N** (interleaved stripe reads), while single-
  query latency still pays one round trip (possibly less, if the hot stripe
  is cached).
- Requires: controller-side shard split of the tenant (dev controller does
  not currently orchestrate it — needs a shard split op), spec
  `shard_stripe_size` set, and compute pageserver_connstring with the shard
  map. Documented as the next infrastructure step, not yet measured.

Also coherent with §1: **1 big pageserver cache vs N small ones** — prefer N
shards only if the workload parallelizes; otherwise one PS with a bigger
page cache is simpler and equally effective for single-stream latency.

## 4. Other optimizing points found while profiling

1. **io_uring unavailable in this VM**: pageserver logs
   `auto-detected IO engine StdFs; tokio-epoll-uring fails: Operation not
   supported`. On a host with io_uring, the pageserver's virtual-file layer
   (`virtual_file_io_engine`) would cut syscall overhead on its local files.
2. **WAL/commit path**: the 3/3 quorum costs ~5–10% (synchronous walproposer
   standby). `max_replication_flush_lag=10GB` is already loose; for
   read-heavy benchmarks this is not the binding constraint.
3. **Bulk ingest**: 10M rows imported at ~30 MB/s (2m53s) — WAL ingest into
   the pageserver is the limiter; `fsync=off` + `wal_log_hints=off` already
   applied. Consider disabling the compute's WAL proposer sync lag during
   bulk load only.
4. **Parallelism**: scans are single-backend; `max_parallel_maintenance_workers=8`
   accelerates builds only. A parallel ivf scan (per-list workers) would
   amortize round trips across cores — high value on Neon (latency-bound),
   moderate on vanilla (already CPU-bound per backend). Candidate: parallel
   index scan support (`amcanbuildparallels`/parallel-aware scan).
5. **Network/compression** (real clusters): libpagestore protocol v3 supports
   zstd; for remote pageservers prefer `compression` in the connstring; in
   this all-localhost k8s setup hostNetwork+loopback is already optimal.
6. **k8s niceties**: pin storage pods to nodes colocated with their disk;
   use `hostPath`→local PV with the right FS; avoid pod eviction during
   benchmarks (the pageserver pods are stateful).

## 5. Coherent recommendation matrix

| workload | LFC | PS page cache | shards | prefetch | expected |
|---|---|---|---|---|---|
| ivf/ivfrq ≤ 1 GB index | 8 GB | 4 GB | 1 | burst pipelining (§2.1) | = vanilla (achieved) |
| hnsw ≥ 8 GB graph | 16 GB | 8 GB | 1–3 | §2.1 + §2.2 | → vanilla parity (to verify) |
| many tenants / throughput | 8 GB+ | 8 GB × N | N (stripe) | §2.2 | cache capacity × N |
| write-heavy | — | — | any | — | WAL/ingest-bound (§4.3) |

## 6. Next actions (priority order)

1. hnsw@10M with 16 GB LFC + 8 GB PS cache → verify parity (30 min bench).
2. Implement §2.1 (double-buffered readv pipelining) in the ivf scan; A/B on
   k8s Neon (expect large-probes p50 to drop toward LFC-hit floor).
3. Try `smgrprefetch` (§2.2) for the next list's blocks; measure RTT hiding.
4. Investigate shard split feasibility in this controller revision (§3).
5. `../vanilla-simd-promotion.md` for the ALU side.
