# Neon cluster — coherence design & optimization plan

Evidence base: BIGANN-10M (128-dim, L2) benchmarks on the k8s-based Neon dev
stack (`root@113.44.106.182`, 16 vCPU / 60 GB, QEMU guest) + `perf` profiling.
See `bench/RESULTS.md`, `bench/K8S.md`, `scripts/RECOMMENDED-SETUP.md`.

**Design constraint (production): the serverless compute is memory-bound.**
Neon compute size units (CU): **1 CU = 1 vCPU + 2 GB RAM**; typical service
sizes are 2 CU (2 vCPU / 4 GB) to 8 CU (8 vCPU / 16 GB). Everything below
must fit that budget — the dev-box numbers (8–16 GB caches) are upper-bound
references, not production settings.

## 0. The measured baseline (what we are optimizing against)

| config | ivfrq p50 @99.4% recall | hnsw p50 @~99% recall | build (10M×128) |
|---|---|---|---|
| vanilla PG17 | 6.42 ms | 9.0 ms | ivf 28 s / hnsw ~10 min |
| Neon dev defaults (1MB shared_buffers, LFC off, 64MB PS cache) | 626 ms | ~1.6 s | ivf 4.9 min / hnsw ~30 min |
| Neon tuned (2GB sb, 8GB LFC, 4GB PS cache) | 5.15 ms | 32.5–87.9 ms | unchanged |
| Neon k8s 3-SK tuned | 5.71 ms | 32.5–87.9 ms | +5–10% WAL-sync |

`perf stat` on a probes=64 workload backend: **7.3% CPU utilized on Neon vs
100% on vanilla** (IPC ~1.9 on both). The Neon backend waits on pagestore
reads ~93% of wall time. **Every Neon-side optimization below is about the
read path, not ALU.** (ALU/instruction promotion is a *vanilla* topic — see
`../vanilla-simd-promotion.md`.)

## 1. Memory budget per CU (the coherence envelope)

What a CU's RAM must cover: postgres shared_buffers + per-backend work_mem +
connections + **LFC** (compute-local page cache) + the extension's own
allocations (our scan allocates the readv burst buffers + candidate heaps).

| CU | RAM | suggested split (tunable) | fits (BIGANN-10M) |
|---|---|---|---|
| 2 CU | 4 GB | shared_buffers 256–512 MB, LFC **2–2.5 GB**, rest for PG/conns | ivfrq (346 MB index + hot segments) ✔ |
| 4 CU | 8 GB | shared_buffers 512 MB–1 GB, LFC **4–6 GB** | ivfrq ✔, hnsw graph *partial* (7.9 GB → thrash) |
| 8 CU | 16 GB | shared_buffers 1–2 GB, LFC **10–12 GB** | ivfrq ✔, hnsw graph + hot heap *mostly* fits |

Notes:

- The production control plane auto-sizes LFC from the CU count; our spec has
  `disable_lfc_resizing` — keep auto-sizing ON and tune within it. Our dev
  numbers (8 GB LFC) correspond to roughly an 8 CU compute.
- **Index size is now a first-class design constraint.** On 2–4 CU the LFC
  cannot hold a 7.9 GB hnsw graph, so hnsw deep scans are pageserver-bound no
  matter what. Our RaBitQ index is **~23× smaller** (346 MB at num_bits=1 for
  10M×128; ~600 MB at num_bits=4; ~1.2 GB at num_bits=8) — it *fits* the 2 CU
  budget, which is the architectural argument for ivfrq on small serverless
  computes. The num_bits knob becomes a memory-budget knob, not just a
  recall/speed one.
- **Build-time memory is also CU-constrained.** pgvector hnsw needs ~10–12 GB
  `maintenance_work_mem` to build 10M×128 without spilling — impossible on
  2–8 CU computes (16 GB total at 8 CU); spilled builds are dramatically
  slower. Our ivf build runs in a few minutes within small-CU memory
  (parallel maintenance workers only). Co-design consequence: **recommend
  ivfrq for large datasets on small CUs; hnsw only with reduced m/ef_construction
  or smaller datasets.**

## 1b. A1 measurements (2 CU proxy, measured 2026-09-02)

2 CU proxy = `shared_buffers` 256/512MB, LFC 2GB, postmaster pinned to 2
vCPUs; ivf built in-place (160 s), 9-point sweep:

- ivfrq p50 @99.7% recall: **~430–700 ms** (vs 6.4 ms vanilla, ~5.7 ms at the
  8 CU tier). The parity target is NOT met at 2 CU with current code.
- Three cost mechanisms identified (perf trace + EXPLAIN BUFFERS):
  1. **WAL-redo on first touch**: freshly built pages live in WAL-only
     layers; every first read pays `neon_walredo` per page (~0.4 ms/page,
     serialized). Fixed by aggressive pageserver compaction
     (`compaction_period=2s, compaction_threshold=1` in pageserver.toml +
     restart): new-qid fetches dropped from 13–326 ms to ~2–5 ms.
  2. **Per-query first-touch working set**: each distinct query touches
     ~1000 buffer-manager pages (heap recheck + active-segment entries);
     on Neon a miss = one pageserver round trip (~0.4 ms), on vanilla it is
     a local read. The 100-query sweep working set (~800 MB) exceeds the
     2 CU buffer budget, so the timed pass re-misses (~370 ms p50 even
     after warmup). Same-qid repeats: ~2 ms.
  3. **Unsealed active segments**: a freshly built (never VACUUMed) index
     serves entries through the buffer-manager row path instead of the
     `smgrreadv` FastScan path (EXPLAIN: read=676–990, dirtied≈written≈
     600–780 per query). VACUUM did not remove the cost in this build —
     seal/merge behavior needs review (`vacuum.rs`).
- Actions: (a) benchmark recipe = compact after build + VACUUM + warm; (b)
  investigate the executor recheck heap-fetch amplification (top_k=1000
  candidates ⇒ up to ~1000 heap fetches/query) — reducing it is the
  highest-leverage 2 CU fix; (c) review active-segment sealing.

## 2. LFC (local file cache) — the primary lever, sized to the CU

Facts from this session:

- LFC defaults to **disabled** in dev (`neon.max_file_cache_size` = 0,
  PGC_POSTMASTER; `neon.file_cache_size_limit` = 0, PGC_SIGHUP, capped by max).
- 8 GB LFC took ivfrq from 626 ms → 5.15 ms p50 (~120×): index + hot heap
  pages fit entirely in LFC, so warm scans never touch the pageserver.
- hnsw stayed ~3–7× vanilla because the working set (7.9 GB graph + 5.3 GB
  table) exceeded 8 GB LFC + 4 GB pageserver cache; the reproducible
  ef_search=320 spike (76.6 ms on re-run) is a working-set boundary.

CU-aware design rules:

1. **Size LFC inside the CU budget**: `LFC ≈ min(index + hot_heap_pages,
   ~60% of CU RAM)`. ivfrq 10M fits 2 CU (2–2.5 GB LFC); hnsw 10M needs the
   pageserver cache as the real second tier on any small CU.
2. Keep `shared_buffers` modest (256 MB–1 GB by CU): LFC is the cache that
   matters; oversized shared_buffers duplicate what LFC already serves.
3. **Chunk size**: `neon.file_cache_chunk_size=256` (2 MB) suits our
   contiguous segment bursts; 128–256 for mixed workloads. Fewer, larger
   chunks amortize metadata lookups within a small cache.
4. `neon.file_cache_path` may point at a **raw device**; on NVMe-backed
   compute nodes this removes FS overhead — most valuable exactly when the
   LFC budget is small.
5. Pageserver page cache is tier-2 and **scales independently of CU RAM** —
   this is the right place to spend the big-memory budget (see §4 sharding).
6. `autoprewarm` (compute spec) primes LFC at endpoint start — hides the
   cold-start latency cliff that otherwise dominates small-CU P99s.

## 3. Prefetch / pipelining (hide the round trip — memory-cheap)

The read path: ivf scan issues `smgrreadv` segment bursts → libpagestore →
LFC hit (local pread) or pageserver fetch (0.4–1 ms localhost RTT). Today the
scan is **synchronous**: burst N+1 is issued only after burst N is consumed.

1. **Measured (A2/A3, 2026-09-02): implemented and REVERTED — no gain.**
   A double-buffered `smgrreadv` pipeline + `smgrprefetch`
   (`neon_prefetch`, feature-gated, commit `560e8cb`) was A/B'd on the 8 CU
   tier (warm LFC): probes=1/8/64 p50 1.39→1.58 / 2.10→2.21 / 5.71→5.83 ms —
   noise-level regression. On LFC-hit reads there is nothing to hide, and the
   prefetch request overhead adds up. On LFC-miss bursts the win would be
   bounded by libpagestore's own serialized per-page processing anyway
   (perf trace: one recv→LFC-write→send cycle per page). Keep the revert
   (`d67741e`); revisit only if a cold-path workload demands it.
2. **The real read-path levers (measured, higher priority than prefetch):**
   post-build pageserver compaction (walredo-per-page removal, §1b) and the
   executor recheck heap-fetch amplification (top_k=1000 candidates ⇒ up to
   ~1000 buffer-manager fetches per query — the dominant 2 CU cost).

## 4. Multi-pageserver / sharding — scale cache outside the CU

Current state: 3 pageserver pods run, but the tenant's single shard sits on
node 1 — pods 2–4 are standby. Real multi-PS gains come from **tenant
sharding** (`shard_count` / `shard_stripe_size` + a controller-driven split):

- **Cache capacity scales out**: each pageserver caches its stripe
  (4 GB page cache × N). For hnsw@10M with 3 shards the per-PS working set
  drops below 4 GB → deep scans stop spilling — *without* growing the
  compute's LFC (which the CU budget forbids).
- **Aggregate read bandwidth × N**; single-query latency still pays one round
  trip (less if the hot stripe is cached).
- Under the CU constraint this is the *only* knob that grows cache for
  big-graph workloads — the compute-side LFC is capped by RAM.
- Requires: controller-side shard split (dev controller does not orchestrate
  it today — next infra step), spec `shard_stripe_size`, compute
  pageserver_connstring with the shard map.

## 5. Other optimizing points found while profiling

1. **io_uring unavailable in this VM** (pageserver: `tokio-epoll-uring
   fails: Operation not supported` → StdFs fallback). On hosts with io_uring,
   the pageserver virtual-file layer cuts syscall overhead on its local files.
2. **WAL/commit path**: the 3/3 quorum costs ~5–10% (synchronous walproposer
   standby); for read-heavy workloads not the binding constraint.
3. **Bulk ingest**: ~30 MB/s (2m53s for 10M rows) — WAL ingest limited;
   `fsync=off` + `wal_log_hints=off` already applied. Consider relaxing the
   compute's WAL sync lag only during bulk load.
4. **Parallelism**: scans are single-backend; on a 2–8 CU compute the other
   vCPUs are idle during a single scan. A parallel per-list ivf scan
   amortizes round trips across cores — high value on Neon (latency-bound),
   and it fits the CU model by construction (uses the vCPUs the CU provides).
5. **Network/compression** (real clusters): libpagestore protocol v3 supports
   zstd; for remote pageservers prefer `compression` in the connstring; on
   all-localhost k8s, hostNetwork+loopback is already optimal.
6. **k8s niceties**: colocate storage pods with their disks; use local PVs;
   request/limit CPU to match the CU model in the demo manifests.

## 6. Coherent recommendation matrix (CU-aware)

| workload | CU | LFC | PS page cache | shards | prefetch | expected |
|---|---|---|---|---|---|---|
| ivf/ivfrq ≤ 1 GB index | 2–4 | 2–3 GB | 4 GB | 1 | §3.1 | = vanilla (projected from 8 CU parity) |
| ivfrq ≥ 1 GB index (num_bits=8) | 4–8 | 4–10 GB | 4–8 GB | 1 | §3.1+§3.2 | → vanilla parity |
| hnsw ≥ 8 GB graph | 2–8 | — | — | — | — | **use ivfrq instead** (mainline recommendation; do not tune hnsw) |
| many tenants / throughput | any | per-CU auto-size | 8 GB × N | N | §3.2 | cache capacity × N |
| write-heavy | any | — | — | any | — | WAL/ingest-bound (§5.3) |

## 7. Next actions (priority order)

1. **Validate under a CU memory cap**: re-run the ivfrq sweep with the compute
   limited to 4 GB (2 CU) and LFC 2 GB → confirm the parity projection holds
   in-budget; repeat at 16 GB (8 CU) for hnsw.
2. Implement §3.1 (double-buffered readv pipelining) — RAM-cheap, A/B on k8s
   Neon (expect large-probes p50 to drop toward the LFC-hit floor).
3. Try `smgrprefetch` (§3.2) for the next list's blocks; measure RTT hiding.
4. Investigate shard split feasibility in this controller revision (§4) —
   the only cache-growth path that respects the CU memory envelope.
5. `../vanilla-simd-promotion.md` for the ALU side (vanilla-only for now).
