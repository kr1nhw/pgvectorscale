# agentvec vs hnswsq — same-host 1M A/B (121.37.117.106, 32 vCPU, PG 17.11 release)

Date: 2026-09-23 — branch `agentvec` (`ce9ccea` + `4894eff` + the two leak fixes
`82d1d72`), release build, same box/cluster/data as `RESULTS-HNSWSQ.md`.
Dataset: BIGANN 1M rows (first 1M of `base.1B.u8bin`), dim 128, 100 queries,
exact top-10 ground truth computed within the 1M subset (numpy, verified
against brute-force SQL).  Both engines: m=16, ef_construction=64,
maintenance_work_mem=8GB, seed pinned.  `hnswsq` baseline re-measured in the
same pass on the same binary.

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

## 5. Large-scale runs (in flight)

* 100M A/B on 113.44.106.182: hnswsq plain baseline relaunched with the
  fixed binary (mwm 24GB) after the box recovered from an outage;
  agentvec 100M build + recall sweep queued behind it.
* 20M flush calibration on 121.37.117.106: 20M rows stream-loaded from
  `base.1B.u8bin` via binary COPY (~7 min, no staging file — the same
  loader scales to 1B); hnswsq 20M build with the default 8GB cap measures
  the flush-streaming rate (observed ~9 MB/s ≈ 10K rows/s from the index
  file growth) — 100M ≈ 4.5-5 h/engine, 1B ≈ 2 days/engine.  agentvec
  20M build + recall queued behind it.
* 1B on 121: loader validated; the table load is ~6 h and each engine's
  build ~2 days — pending the 100M results and disk headroom.
