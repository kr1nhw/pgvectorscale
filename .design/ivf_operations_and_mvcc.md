# IVF (IVF-RaBitQ) index — operations, parameters, and MVCC

The `ivf` access method is an inverted-file + RaBitQ index over `vector`
columns. It is an **approximate** nearest-neighbor index: it returns a bounded
set of candidates ranked by a quantized distance estimate, and the executor
re-checks the *exact* distance on the heap tuple (so the final ranking is
exact over the probed candidates, but recall < 1 is possible).

## 1. Operators / operator classes

| operator class | operator | distance |
|---|---|---|
| `vector_l2_ops` | `<->` | squared L2 (Euclidean) |
| `vector_ip_ops` | `<#>` | negative inner product |
| `vector_cosine_ops` | `<=>` | cosine distance |

## 2. DDL operations

```sql
-- Create (takes ACCESS EXCLUSIVE lock; parallel-safe via maintenance_work_mem)
CREATE INDEX idx ON tbl USING ivf (embedding vector_l2_ops)
  WITH (lists = 100, num_bits = 1);

-- Drop (plain DROP INDEX semantics)
DROP INDEX idx;

-- Reindex (rebuild; full table + index scan)
REINDEX INDEX idx;            -- single index
REINDEX TABLE tbl;            -- all indexes on the table
```

Build notes:

- The build runs k-means on a reservoir sample (`DEFAULT_SAMPLE_SIZE`), assigns
  every row to its nearest centroid, quantizes each residual with RaBitQ, and
  writes the meta / list-directory / centroid / entry pages.
- `maintenance_work_mem` is honored (`amusemaintenanceworkmem = true`).
- `ambuildempty` exists, but PostgreSQL 17 calls `ambuild` even for an empty
  table, so `ambuild` also writes the minimal empty structure.

## 3. Query

```sql
-- k-NN; the LIMIT is pushed into the index scan and drives the recheck queue.
SELECT id FROM tbl ORDER BY embedding <-> '[...]'::vector LIMIT 10;

-- parameterized
SELECT id FROM tbl, (SELECT q FROM queries WHERE qid = 0) q
ORDER BY embedding <-> q.q LIMIT 10;
```

The index is used for `ORDER BY <op>` with a `vector` expression; it has no
search-key (WHERE) strategies (`amstrategies = 0`, `amcanorder = false`,
`amcanorderbyop = true`).

## 4. Parameters

### 4.1 `WITH (...)` reloptions (CREATE INDEX)

| option | type | default | range | meaning |
|---|---|---|---|---|
| `lists` | int | 100 | 1–32768 | number of inverted lists (k-means centroids). More lists = fewer candidates/query and finer granularity (higher recall at the same probe count). |
| `num_bits` | int | 1 | 1–8 | RaBitQ bits/dimension: `1` = sign bit; `4` = 1 sign + 3 magnitude; `8` = 1 sign + 7 magnitude. |
| `storage_layout` | string | `plain` | — | **Parsed but ignored** — the build always uses `RabbitqCompression`. |

> `storage_layout` is a leftover diskann option; it does not change behavior.
> `num_bits ∈ {1,4,8}` are the tested values; other values use the scalar
> fallback code path.

### 4.2 GUCs (session-level)

| GUC | default | range | used? | meaning |
|---|---|---|---|---|
| `ivf.probes` | 1 | 1–32768 | ✅ | number of lists to probe per query. |
| `ivf.top_k` | 1000 | 1–1000000 | ✅ | number of top candidates (by estimate) kept before the executor's exact recheck. Must be ≥ the query LIMIT. |
| `ivf.iterative_scan` | 0 | 0–1 | ❌ (registered, unused) | — |
| `ivf.max_probes` | 32768 | 1–32768 | ❌ (registered, unused) | — |

```sql
SET ivf.probes = 40;
SET ivf.top_k  = 1000;
```

### 4.3 Recall / latency knobs

- **`lists`** and **`ivf.probes`** together control how many candidates are
  scanned (`probes × rows/list`). Raising `lists` (with more `probes` to keep
  recall) is the dominant latency lever.
- **`ivf.top_k`** bounds how many candidates the executor re-checks exactly;
  smaller = faster but lower recall.

## 5. MVCC support

### 5.1 What works

- **Visibility filtering** — the index stores only `(heap TID, quantized code)`.
  `amgettuple` returns the heap TID (`xs_heaptid`) with `xs_recheckorderby =
  true`; the **executor** fetches each heap tuple and applies the ordinary MVCC
  snapshot check, plus re-checks the exact distance. Uncommitted/dead tuples are
  therefore filtered by the executor, not by the index. Read-only MVCC (queries
  over committed data) is correct.
- **Vacuum** — `ambulkdelete` walks each list, drops entries whose heap TID the
  vacuum callback marks dead, and rewrites the survivors; `amvacuumcleanup` is a
  no-op. This is the standard MVCC dead-tuple cleanup.

### 5.2 What is NOT supported (concurrent DML during scans)

The index is **not safe for concurrent INSERT/VACUUM while a scan is running**:

1. **No locking** — `aminsert`/`ambulkdelete` rewrite a whole inverted list
   (read all entries → write new contiguous blocks → update the list directory
   in place) with no lock serializing writers against readers. There is no
   `acquire_index_lock` / `LockRelationForExtension` on this path.
2. **`smgrread` bypasses shared_buffers** — the scan reads entry blocks directly
   from the storage manager (OS page cache), which is only correct because the
   writer calls `FlushRelationBuffers` after rewriting. During an in-flight
   rewrite, a concurrent scan can observe a torn state: the list directory may
   already point at the new blocks while the new blocks are not yet on disk, so
   the scan reads stale/zero pages (garbage distances, or missed results).

### 5.3 Practical contract

- **Safe:** build → flush → read-only queries; `CREATE/DROP/REINDEX` (which take
  ACCESS EXCLUSIVE / table locks); `VACUUM` on an otherwise-idle table.
- **Unsafe:** concurrent `INSERT`/`DELETE`/`UPDATE`/`VACUUM` overlapping with
  `SELECT ... ORDER BY <->`. To make it MVCC-safe for concurrent writes you
  would need (a) a relation/advisory lock around list rewrites vs scans, and/or
  (b) to read entry blocks through the buffer manager instead of `smgrread`.

So the short answer: **the query/visibility path is MVCC-correct, but concurrent
writers during scans are not yet supported** — the index assumes a read-mostly /
append-then-flush workload.

## 6. Making concurrent writes MVCC-safe — options compared

Our IVF currently uses (a) a **full list rewrite** on insert/vacuum (read all
entries → write new contiguous blocks → update the directory in place) and
(b) **`smgrread`** for entry blocks (bypasses shared_buffers). Both contribute
to the concurrent-write unsafety.

### Option A — coarse lock around rewrites vs scans

Serialize writers against readers with a transaction-level lock (e.g.
`pg_advisory_xact_lock` — the diskann AM here already has `acquire_index_lock`).

| | |
|---|---|
| change size | small (take an exclusive lock in `aminsert`/`ambulkdelete`, a share lock in `ambeginscan`) |
| read path | keeps the fast `smgrread` + contiguous SIMD FastScan |
| concurrency | **serializes** scans vs rewrites (a scan blocks during a rewrite, and vice versa); no overlap |
| still O(n) rewrite | yes — the per-insert full-list rewrite is unchanged |

### Option B — read entry blocks via the buffer manager

Replace `smgrread` with `ReadBufferExtended` + `LockBuffer(BUFFER_LOCK_SHARE)`
(or pin-only `PinnedBufferShare`).

| | |
|---|---|
| change size | medium (rework `IvfEntryReader::read_bytes`) |
| read path | **slower** — brings back per-page `PinBuffer`/`LWLock` (the ~38% hotspot we removed), and breaks the single-contiguous bulk read |
| consistency | buffer manager + WAL make reads see committed, flushed-or-buffered state (no stale reads) |
| torn rewrite | **not fixed by itself** — the writer still rewrites non-atomically, so a reader can interleave between page commits unless the writer *also* holds a lock across the rewrite (i.e. you still need A) |

### What pgvector does (ivfflat / ivfrq)

pgvector avoids the problem structurally rather than locking around it:

1. **Append-only insert** — each list is a *linked chain of pages* with an
   `insertPage` pointer. `ivfinsert.c` reads the tail page
   (`ReadBuffer` + `LockBuffer(BUFFER_LOCK_EXCLUSIVE)` + `GenericXLog`), appends
   the `IndexTuple` via `PageAddItem`, and only `LockRelationForExtension`s when
   it has to allocate a new page. No full-list rewrite → no torn mid-rewrite
   state. (O(1) amortized per insert.)
2. **Buffer manager everywhere** — scans (`ivfscan.c`) and vacuum
   (`ivfvacuum.c`) read with `ReadBuffer` + `LockBuffer(BUFFER_LOCK_SHARE)`.
   Vacuum deletes *in place* (`PageIndexTupleDelete` under
   `LockBufferForCleanup`) — again no rewrite.
3. **Visibility** is the ordinary executor MVCC snapshot check on the returned
   heap TIDs.

So pgvector's concurrency safety = *append-only storage* + *buffer-manager
reads*, not a coarse lock and not a rewrite-then-flush design.

### Recommendation

- **Shortest path to MVCC-safety (keep our SoA/transposed SIMD layout): Option A**
  — a transaction-level advisory lock (`EXCLUSIVE` for writers, `SHARE` for
  scans). Simple, preserves the fast bulk read; the cost is serializing reads
  vs. the (already O(n)) rewrites.
- **The "right" long-term fix is pgvector's model**: switch entry storage to an
  append-only chain (or a segmented/log-structured layout) so inserts don't
  rewrite the whole list, and read through the buffer manager (or keep `smgrread`
  but only rewrite/free on a locked VACUUM compaction). That removes both the
  O(n) rewrite and the stale-read window without serializing every query.

Option B alone is the weakest choice: it pays the buffer-manager read cost
without fixing the non-atomic rewrite.
