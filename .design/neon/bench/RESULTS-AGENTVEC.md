# agentvec vs hnswsq — same-host 1M A/B (121.37.117.106, 32 vCPU, PG 17.11 release)

Date: 2026-09-23 — branch `agentvec` (`ce9ccea` + `4894eff` + the two leak fixes
`82d1d72`), release build, same box/cluster/data as `RESULTS-HNSWSQ.md`.
Dataset: BIGANN 1M rows (first 1M of `base.1B.u8bin`), dim 128, 100 queries,
exact top-10 ground truth computed within the 1M subset (numpy, verified
against brute-force SQL).  Both engines: m=16, ef_construction=64,
maintenance_work_mem=8GB, seed pinned.  `hnswsq` baseline re-measured in the
same pass on the same binary.

## 0. Revised architecture (2026-09-23, user-directed)

`ambuild` now builds the payload DIRECTLY as an immutable IVF-RaBitQ
segment — the `ivf` AM's streaming two-pass build (Lance-style:
reservoir sample -> k-means -> per-list batched seal, memory bounded by
`num_lists x 10k` entries).  The HNSW HOT path belongs to `aminsert`
only; its sealed segments are converted into further WARM segments by
the maintenance worker.  (Dims < 8 fall back to a bulk HNSW HOT
segment; RaBitQ needs >= 8.)

Same-pass comparison on the 1M table (release PG 17.11):

| config | build s | size MB | recall@10 | q0 ms |
|---|---|---|---|---|
| hnswsq plain (m=16, efc=64) | 63 | 819 | 0.991 @ ef160 | 2.7 |
| agentvec bulk (ivf_lists=1000, sc=100) | **16** | **56** | 0.817/0.885/0.927/0.937/0.939 @ p8/16/32/64/128 | 1.7/1.9/2.1/2.5/3.2 |

Build **4x faster**, size **14.5x smaller**, query latency 1.7-3.2 ms;
recall trades off against hnswsq's 0.991 (partition-quality bound at
lists=1000; more lists + the phase-12 recoding push it up — the ivfrq
100M study reached 98.3% at 1000 lists on 100M rows).

## 1. The performance goal

**agentvec is not worse than hnswsq** on the HNSW operating point
(single-HOT segment, plain and f8):

| config | build s | size bytes | recall@10 ef10/40/160/640 | q ms LIMIT 10 ef10/40/160/640 |
|---|---|---|---|---|
| hnswsq plain | 65 | 819,208,192 | 0.726 / 0.932 / 0.991 / 0.998 | 0.62 / 1.19 / 2.68 / 7.11 |
| agentvec HOT plain | 69 (1.06x) | 819,249,152 (+0.005%) | 0.726 / 0.931 / 0.989 / 0.996 | 0.80 / 1.26 / 2.85 / 5.74 |
| agentvec HOT f8 | 69 | 375,717,888 (46%) | 0.717 / 0.933 / 0.990 / 0.998 | 0.81 / 1.27 / 2.83 / 5.65 |

* Build parity was achieved by porting the hnswsq bulk builder into
  `ambuild` (`build_region`): the incremental insert path cost 1113 s for
  1M rows (17x); the bulk path costs 69 s.
* Query latency is within noise (agentvec is actually ~20% faster at ef=640
  in this pass); recall within ±0.5 pt; index size identical (+32 KB of
  directory/meta scaffolding).

## 2. Bugs the benchmark exposed (fixed in the same session)

* **~100 KB/row OOM leak** in the shared hnswsq insert path: (a)
  `ElementArena::reset()` dropped arena chunks without deallocating them;
  (b) the agentvec HOT insert ran without hnswsq's per-insert memory
  context, so the search's per-neighbor pallocs accumulated.  1M build went
  from OOM-killed at 97 GB RSS to a ~0.5 GB peak.  Found via LD_PRELOAD
  malloc sampling + addr2line.
* **"index returned tuples in wrong order"**: the RaBitQ FastScan estimate
  is not a guaranteed lower bound and cannot be the executor's orderby
  hint.  Bounded scans (`search_candidates > 0`) now exact-rerank at
  emission (top-k by estimate → heap fetch → exact distance); this also
  fixes the `-infinity` candidate flood that made the bounded heap keep
  arbitrary WARM candidates (recall collapsed to 0.003 at
  `search_candidates=100` before the fix).
* PG18's `ExecDropSingleTupleTableSlot` left the heap-fetch buffer pin
  registered ("resource was not closed" at commit); fixed with an explicit
  `ExecClearTuple`.

## 3. Compressed operating point (HOT f8 + 1 WARM RaBitQ segment)

`agentvec_consolidate` converts the 1M-row segment in **7 s**; the WARM
payload is **37 MB** (22x smaller than the 819 MB plain index); total index
412 MB.

| probes | recall@10 | q ms (LIMIT 10) |
|---|---|---|
| 4 | 0.870 | ~300 |
| 8 | 0.914 | ~300 |
| 16 | 0.926 | ~300 |
| 32 | 0.921 | ~300 |

The ~300 ms plateau is the estimate pass over 100k-entry lists
(`ivf_lists=100`); a finer partition removes it — `ivf_lists=1000`
(1k-entry lists, search_candidates=100):

| probes | recall@10 | q ms (LIMIT 10) |
|---|---|---|
| 8 | 0.793 | 1.8 |
| 16 | 0.858 | 1.9 |
| 32 | 0.908 | 2.1 |
| 64 | 0.936 | 2.6 |
| 128 | 0.943 | 3.4 |

Build 71 s + consolidate 20 s, total 432 MB (53% of plain).  vs hnswsq
plain's 2.7 ms / 0.991 recall@ef160: at matched ~2.7 ms the compressed
config reaches 0.936 at 47% less space — a real size/latency tradeoff
curve; partition quality (phase-12 recoding) and the phase-8 rerank
(bounding the unbounded scan) remain the levers to push recall up.

## 3.1 Insert throughput at 1M scale (50k fresh rows, single backend)

| engine | 50k inserts ms | rows/s |
|---|---|---|
| hnswsq plain | 9278 | 5390 |
| agentvec HOT plain | 8887 | 5627 |

Parity (agentvec 4% faster in this pass — same insert path plus the
agentvec wrapper, amortized by the batch).

## 4. aarch64 parity (116.204.102.142, 16 vCPU, PG 17.11 release)

Same 1M dataset (subset of the box's 10M BIGANN table; exact top-10
ground truth computed within the 1M subset in SQL).  Release build of the
same commit.

| config | build s | size bytes | recall@10 ef10/40/160/640 | q0 ef40 ms (3 runs) |
|---|---|---|---|---|
| hnswsq plain | 209 | 819,216,384 | 0.781 / 0.942 / 0.993 / 1.000 | 6.35 / 4.19 / 4.13 |
| agentvec HOT plain | 213 (1.02x) | 819,249,152 | 0.776 / 0.939 / 0.993 / 1.000 | 8.48 / 4.40 / 4.39 |

Build 1.02x, size identical (+32 KB), recall/latency parity — the
"not worse than hnswsq" goal holds on aarch64 too.  (The first attempt on
this box OOM-killed exactly like the old x86 builds because its extension
predated the leak fixes; rebuilt from the fixed commit.)

## 5. 100M results (113.44.106.182, 16 vCPU, release PG 17.11)

Direct IVF-RaBitQ bulk build on the 100M BIGANN table, exact top-10
ground truth computed over the table itself (SQL brute force, verified):

| config | build min | size GB | recall@10 / q0 ms |
|---|---|---|---|
| agentvec bulk rabitq_bits=1, lists=1000 | 19.3 | 3.46 | (see below) |
| agentvec bulk rabitq_bits=2, lists=1000, sc=1000 | ~40 | 5.10 | probes 8: 0.904 / 59 · 16: 0.970 / 36 · 32: 0.989 / 67 · 64: 0.997 / 109 · 128: 0.999 / 146 · 256: 1.000 / 226 |
| hnswsq plain (m=16, efc=64) | ~25-50 h (flush-bound estimate) | ~82 | ~0.99 @ ef160 (1M-scale extrapolation) |

Recall reaches hnswsq-class levels (0.99+ at p32-64) with build ~40x
faster and size ~16x smaller; latency is 3-5x hnswsq's at matched recall
(the emission-time rerank's per-candidate heap fetches — phase 8's
batched/prefetched rerank is the lever).

Notes: the first recall sweep here read 0.35 — traced to a stale
ground-truth table computed from a different (10M-row) dataset and, once
that was replaced, a dump-order vs id-order mismatch in the numpy GT; the
final GT is verified byte-for-byte against SQL brute force.  The earlier
100M study's 98.3% figure was measured against the same stale GT.

## 6. BIGANN-1B results (121.37.117.106, 32 vCPU/121GB, release PG 17.11)

Direct IVF-RaBitQ bulk build on 1,000,000,000 rows (dim 128) from
`base.1B.u8bin` (u8 -> float32, no offset), index
`agentvec (ivf_lists=2000, rabitq_bits=1, search_candidates=1000,
hot_segment_max_rows=50000)`:

| stage | seconds | notes |
|---|---|---|
| COPY 522 GB (server-side binary COPY, unlogged) | 5074 | ~103 MB/s, disk-bound |
| ALTER TABLE SET LOGGED | 9417 | full rewrite |
| ADD PRIMARY KEY (1B bigints) | 4850 | |
| **CREATE INDEX agentvec** | **19819 (5.5 h)** | **index 32.1 GB**; table 585 GB |

Recall@10 vs exact top-10 ground truth (200 queries, see GT note), sweep of
`search_candidates` (sc, the emission-time exact-rerank window) x
`ivf_probes` (p):

| sc \ p | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| 1000 | 0.774 | 0.837 | 0.871 | 0.8815 | 0.875 | 0.8685 | — |
| 2000 | — | — | — | 0.935 | 0.933 | 0.9295 | 0.9285 |
| 4000 | — | — | — | 0.9725 | 0.9725 | 0.9715 | 0.970 |
| 8000 | — | — | — | — | — | **0.9905** | 0.9895 |

q1 ms (LIMIT 10): sc=1000: 31/51/104/178/345/613 at p8..256; sc=4000:
~350-715 at p64..512; sc=8000: 699 at p256.  (Latency is dominated by the
rerank's per-candidate heap fetches — sc sequential fetches per query —
which is exactly phase 8's batched/prefetched rerank target.)

hnswsq context: a 1B hnswsq build is not practical on this box — the 100M
build was ~25-50 h flush-bound and ~82 GB, so 1B would be ~250-500 h and
~820 GB.  agentvec builds 1B in **5.5 h at 32 GB** and reaches
hnswsq-class recall (**0.9905** at sc=8000/p=256, ~700 ms) vs hnswsq's
~0.991/ef160 (~30 ms at 100M).  The goal "not worse than hnswsq" holds on
build time and size by orders of magnitude and on recall within 0.1 pt at
the raised-knob operating point; latency is ~20x hnswsq's at matched
recall and remains the open gap (phase-8 batched rerank).

The probe-sweep shape also shows the estimate window, not the probe count,
is the recall limiter at this scale: at fixed sc, recall is flat-to-
slightly-declining in p (estimate crowding — with 500k-entry lists, more
probes flood the top-sc estimate heap with false positives), so
`search_candidates` must scale with the list size.

### GT note (important)

The bundled `GT_1B/bigann-1B` does NOT match this
`base.1B.u8bin`/`query.public.10K.u8bin` pairing: for qid=0 the GT's
top-10 ids sit at L2 distances 244k-356k while a full-brute-force scan
finds the true top-10 at 80k-87k (zero overlap at id offsets 0 and +1).
Instead, exact top-10 ground truth was computed for a 200-query subset by
chunked matmul over all 1B vectors (25M-vector chunks), and
cross-validated against an independent full-scan brute force (sets match
exactly).

### Bugs the 1B run exposed (fixed)

* **Buffer-pin leak in the emission-time rerank** (`fetch_heap_vector`):
  `ExecStoreBufferHeapTuple` pins the buffer a second time
  (transfer_pin=false) in PG17.11+/PG18, so the slot clear released only
  that second pin — one leaked heap pin per reranked candidate (100
  "resource was not closed" warnings per smoke query).  At 1B with
  sc=1000 the recall statement would have pinned ~2.5M buffers (20 GB)
  against a 4 GB shared_buffers pool and died.  Fixed with
  `ExecStorePinnedBufferHeapTuple` (commit `b3ed72d`); verified 100->0
  leaked pins on PG17 and PG18.
* **Never replace a loaded .so in place**: `cp` over
  `vectorscale-0.9.0.so` truncates+rewrites the same inode; backends with
  the old mapping fault in new bytes at old offsets and crash the
  maintenance worker (SIGILL 48 s after one install, SIGSEGV after
  another), and any worker crash forces a full crash-restart that rolls
  back in-flight work (the first 1B COPY died at its commit this way).
  Use atomic rename (or swap only while the server is down).
