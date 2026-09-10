# AgentVec — Phase 1 implementation notes

Status: **implemented and verified** (2026-09-10, branch `agentvec`, PG 18.4 arm64).
Target milestone: **M1 — PostgreSQL integration** (`agentvec` creates an index and survives
INSERT / UPDATE / DELETE / VACUUM / restart).

Phase 0 record: `.design/agentvec_phase0_technology_decision_record.md`.
Design: `.design/agentvec_system_design_and_implementation_plan.md`.

---

## 1. What phase 1 delivers

```sql
CREATE INDEX idx ON t USING agentvec (embedding vector_l2_ops);          -- works
CREATE INDEX idx ON t USING agentvec (embedding);                        -- l2 is the default opclass
CREATE INDEX idx ON t USING agentvec (embedding vector_cosine_ops)
    WITH (hot_segment_max_rows = 5000, search_candidates = 1000);        -- reloptions work
SELECT * FROM agentvec_index_info('idx');                                -- directory introspection
```

* a real `agentvec` access method registered with `CREATE ACCESS METHOD`, with the
  `vector_l2_ops` (default), `vector_cosine_ops` and `vector_ip_ops` operator classes;
* the on-disk skeleton the later phases build on: **meta page → segment directory → segment
  headers → append-only payload chains**;
* the HOT segment lifecycle of plan §10 in its simplest form: inserts append to the current HOT
  segment, and the transaction that observes `hot_segment_max_rows` performs a **metadata-only
  seal** plus the creation of the next HOT segment;
* one owned segment algorithm, `FLAT`: exact (exhaustive) search over stored `f32` vectors, with
  tombstoning on VACUUM. `FLAT` is the correctness baseline every later ANN algorithm is
  measured against, and it is what makes a committed row immediately searchable with no
  background step;
* correct PostgreSQL lifecycle handling for build, insert, scan, bulkdelete, vacuum cleanup,
  options, validation, cost estimation, REINDEX and crash recovery.

Not in phase 1 (by design): no HNSW, no IVF-RaBitQ, no router, no maintenance worker, no
migration, no reclamation, no parallel scan/build. See §7.

### Files

| File | Purpose |
|---|---|
| `pgvectorscale/src/access_method/agentvec/mod.rs` | AM handler + AM/opclass registration, `amvalidate`, `amcostestimate`, `agentvec_index_info()` |
| `.../agentvec/options.rs` | the plan §30 reloptions (13 knobs) |
| `.../agentvec/meta_page.rs` | `AgentVecMetaPage` (block 0), atomic read-modify-write |
| `.../agentvec/directory.rs` | segment directory (copy-on-write chained item), segment metadata, segment header, level/state/algorithm/ownership enums |
| `.../agentvec/flat.rs` | the FLAT payload: page-chain format, append, seal helper, tombstoning, iteration |
| `.../agentvec/insert.rs` | `aminsert` + the shared lifecycle (`current_hot_segment`, `seal_hot_and_open_new`, `insert_entry`) used by both build and DML |
| `.../agentvec/build.rs` | `ambuild` (streaming single pass) and `ambuildempty` |
| `.../agentvec/scan.rs` | `ambeginscan`/`amrescan`/`amgettuple`/`amendscan` |
| `.../agentvec/vacuum.rs` | `ambulkdelete` (tombstones) / `amvacuumcleanup` |
| `.../agentvec/tests.rs` | behavioural + structural test suite |
| `src/util/page.rs` | four new `PageType` variants (`AgentVecMeta/Directory/SegmentHeader/FlatPage`) |

---

## 2. On-disk layout (format version 1)

```text
block 0 : AgentVecMetaPage          single-item page, atomic RMW under content lock
            magic, format version, extension version, distance type, dimensions,
            directory pointer + block count, next_segment_id, hot_segment_id,
            epoch, generation, num_tuples

block 1+ : AgentVecDirectory        chained item, COPY-ON-WRITE, immutable once published
            Vec<AgentVecSegmentMeta {
              segment_id, group_id, level, state, algorithm, ownership,
              generation, epoch, header, code_root, posting_root,
              vector_count, live_count, dead_count, metric, dimension,
              format_version, physical_index_oid, access_method_oid, opclass_oid }>

dynamic : AgentVecSegmentHeader     one page per segment, atomic RMW — the publication point
            version, generation, level, state, algorithm,
            sealed: Vec<FlatRun{first_page, num_entries}>,
            active: Option<FlatActive{first_page, last_page, num_entries}>,
            num_entries, dead_entries

dynamic : AgentVecFlatPage          payload chain pages
            offset 1 : next block (u32, 0xFFFFFFFF = end)
            offset 2+: [heap block u32][heap offset u16][state u8][vector f32 x dim]
```

### Invariants

1. **A published item is immutable.** The directory item and every payload page it reaches are
   never rewritten; a change publishes a *new* item and repoints the single page that owns the
   reference (the meta page, or the segment header).
2. **Publication is one page.** The segment header is a single-item page rewritten under its
   buffer content lock (WAL-logged), so a concurrent reader holding the share lock sees either
   the old or the new version, never a mix; concurrent writers serialize their read-modify-write.
3. **Only published state is searched**, and a scan captures the directory once, so nothing a
   reader walks can move under it.
4. **Sealing is metadata-only.** Freezing the active chain moves its descriptor into the
   segment's `sealed` list; the entries are already durable where they are, nothing is copied,
   and the seal therefore costs one header page write plus one directory republication.
5. **Chains link strictly forward** (a new page always comes from extending the relation), which
   lets a scan walk a chain to its end from its first block alone; the reader asserts progress
   against the relation size to catch corruption.
6. **A partially built segment is never published** — a segment exists in the directory only
   after its header page is durable, and the directory item that exposes it is written after.

### Lock protocol

```text
plain INSERT :                              segment header (content lock, RMW)
                                              └─ may extend the relation for a new chain page

seal path    :  meta page (content lock, RMW)
                  ├─ segment header (content lock, RMW)   ← seal the old HOT chain
                  └─ relation extension (new header page, new directory item)
```

The order is **meta → header** and is never inverted (no path takes the meta lock while holding
a header lock), which is what keeps it deadlock-free. A plain insert therefore never touches the
meta page: only the segment header's content lock serializes appenders, so inserts do not
serialize globally. A row appended by a transaction that raced the seal lands in the sealed
segment's fresh active chain — still searchable, and drained by migration in phase 5.

---

## 3. Search contract

```text
directory (one snapshot) → searchable segments
   → per segment: sealed runs + active chain
      → exact distance per live (non-tombstoned) entry
         → bounded top-N per segment (or exhaustive) → ascending distance order
```

* The query vector is normalized for cosine at `amrescan`, exactly as stored vectors are
  normalized at insert, so `distance_cosine` is applied to unit vectors and the value equals what
  the `<=>` operator computes.
* `search_candidates = 0` (the default) means **exhaustive**: every live entry is returned in
  distance order, so `LIMIT k` semantics never depend on a candidate bound. A positive value
  bounds the per-segment heap instead (the ANN behaviour phase 3 will use).
* `xs_recheckorderby = false` because FLAT distances are exact and identical to the ordering
  operator's; the executor can stop as soon as LIMIT is satisfied.
* `xs_recheck = (nkeys > 0)`: this AM evaluates no index quals itself, so anything the planner
  passed as a scan key is rechecked against the heap.
* **The planner must never use this index without ORDER BY keys.** `agentvec_amcostestimate`
  returns infinite costs *and* (on PG 18) marks the path with `disabled_nodes = 2`; PG 18 compares
  `disabled_nodes` before cost, so without that mark `SET enable_seqscan = off` made the planner
  choose an Index Only Scan for `count(*)` and return no rows. See the phase 0 record §0.7.1 —
  the pre-existing `ivf` AM has that bug; `agentvec` has a regression test.

---

## 4. DELETE / VACUUM

* A deleted row stops being returned as soon as the deleting transaction commits, because the
  executor's heap fetch applies the snapshot — index entries are hints, not truth.
* `ambulkdelete` walks every live entry, collects a chain's TIDs **before** calling the vacuum
  callback (so no AgentVec lock is ever held across a heap buffer access — the INSERT path
  takes heap then index, and inverting that order would be a deadlock risk), then flips the
  per-entry state byte to `DEAD` and increments the segment header's `dead_entries` counter in
  the *same* header rewrite, so the bytes scanned and the counters reported by the directory and
  `agentvec_index_info()` cannot drift apart.
* `tuples_removed` reports the tombstoned entry count; `pages_deleted` stays 0 because this AM
  retires segments rather than truncating pages (physical compaction is phase 10).
* Tombstones matter for correctness of *bounded* searches: a dead entry would otherwise consume
  a candidate slot and leave the query short of its LIMIT. The test suite asserts exactly that.
* Whether a given `VACUUM` can tombstone a row depends on PostgreSQL's vacuum cutoff (another
  backend holding an older snapshot keeps the deleted tuple "recently dead"), which is why the
  tombstone tests are written to be independent of it.

---

## 5. Options (`WITH (...)`)

| Option | Default | Phase-1 effect |
|---|---|---|
| `hot_segment_max_rows` | 50000 | rows a HOT segment absorbs before sealing |
| `search_candidates` | 0 | per-segment candidate bound; 0 = exhaustive (exact) |
| `rerank_k` | 100 | reserved (phase 8) |
| `warm_segment_target_rows` | 1000000 | reserved (phase 6) |
| `cold_segment_target_rows` | 10000000 | reserved (phase 6) |
| `rabitq_bits` | 1 | reserved (phase 3) |
| `ivf_lists` / `ivf_probes` | 100 / 10 | reserved (phase 3) |
| `router_top_m` / `router_group_top_m` | 8 / 2 | reserved (phase 7) |
| `migration_batch_rows` | 5000 | reserved (phase 4/5) |
| `maintenance_interval` | 60000 | reserved (phase 4) |
| `maintenance_max_bytes` | 67108864 | reserved (phase 4) |

The reserved options are declared now so the SQL surface and the format do not change when the
owning phase lands; each is documented as reserved in its SQL description.

---

## 6. Verification performed

Environment: PostgreSQL 18.4 (Homebrew, arm64), `cargo-pgrx 0.16.1`, extension built with
`--features pg18`.

### 6.1 Test suite

`PGRX_HOME=.pgrx-home cargo pgrx test pg18 agentvec` → **15 passed, 0 failed**:

behavioural — exact L2 order; a row inserted after the index exists is immediately searchable;
UPDATE reveals the new vector; DELETE removes the row; cosine ignores magnitude; inner product
orders by descending dot product; `search_candidates` bounds the stream; REINDEX reproduces the
same answers; the planner uses the index only for ORDER BY queries (and `count(*)` returns the
true count, including under `enable_seqscan = off`);

tombstones — a tombstoned entry stops consuming candidate slots (applied at the byte level, so
the assertion does not depend on PostgreSQL's vacuum cutoff), and `ambulkdelete` (driven directly
with a synthetic callback) tombstones exactly the reported entries, reports `tuples_removed` /
`num_index_tuples`, and keeps the segment's `dead_entries` counter in step;

structural — a fresh index is meta + one published empty HOT segment; the seal lifecycle at
`hot_segment_max_rows = 2` produces four HOT segments (three `queued_for_migration` with one
frozen run each, one current), and both the Rust-side directory walks and the SQL
`agentvec_index_info()` view agree on the counts;

integration — `VACUUM` runs through a second connection and leaves results correct.

Whole-crate run: `cargo pgrx test pg18` → 164 passed, 10 ignored, **2 pre-existing `ivf`
failures** (`pg_test_ivf_options_defaults` / `_custom`: `data type vector has no default operator
class for access method "ivf"`, because `ivf`'s operator classes are not declared `DEFAULT`).
`ivf`'s sources are untouched by this work.

### 6.2 Lifecycle checks beyond the test framework

Run against a scratch PG 18 instance (`initdb` in a workspace directory, extension installed
from this branch):

| Check | Result |
|---|---|
| 50 rows inserted into an index with `hot_segment_max_rows = 10`, then `pg_ctl restart` | index returns all 50 rows and the correct nearest neighbours |
| insert 5 more rows after restart | index returns 55 rows; the new rows are searchable |
| `pg_ctl stop -m immediate` (crash) then start | recovery replays cleanly: 55 rows, correct order, new insert works, no panic/corruption in the server log |
| 4 concurrent sessions × 500 inserts (=2000 rows) with `hot_segment_max_rows = 50` (seal races) | index reports exactly 2000 rows — no lost or duplicated entries under the seal race |
| concurrent INSERT + DELETE + VACUUM while searching | heap and index agree at 2300 live rows; deleted ids are gone from results; no deadlock/panic |

### 6.3 M1 checklist

| M1 requirement | Status |
|---|---|
| `agentvec` creates an index | ✅ `CREATE INDEX ... USING agentvec` |
| INSERT | ✅ immediate visibility, bounded foreground work |
| UPDATE | ✅ new version searchable, old version filtered by the heap |
| DELETE | ✅ tombstoned by VACUUM; correct before that via heap visibility |
| VACUUM | ✅ `ambulkdelete` reports `tuples_removed`; cleanup is a no-op |
| restart (clean) | ✅ |
| restart (crash recovery) | ✅ |
| REINDEX / rebuild | ✅ |

### 6.4 Reproducing this

```bash
# build and unit tests
cd pgvectorscale
cargo build --features pg18
PGRX_HOME=$PWD/../.pgrx-home cargo pgrx test pg18 agentvec     # the agentvec suite
PGRX_HOME=$PWD/../.pgrx-home cargo pgrx test pg18              # the whole crate
```

`cargo pgrx test` installs the extension into the target PostgreSQL installation, so the run
needs write access to that installation's `sharedir`/`pkglibdir`. For manual checking of the
lifecycle beyond the test framework, a scratch instance works and keeps the workspace clean:

```bash
mkdir -p .avtest && initdb -D .avtest/data -U $USER --encoding=UTF8 --locale=C
mkdir -p .avtest/sock
pg_ctl -D .avtest/data -l .avtest/server.log \
       -o "-p 55432 -k $PWD/.avtest/sock -c listen_addresses=''" start
cargo pgrx install --features pg18        # installs into the target PG
psql -h $PWD/.avtest/sock -p 55432 -U $USER -d postgres \
     -c 'CREATE EXTENSION vector; CREATE EXTENSION vectorscale;'
```

`.avtest/` is gitignored.

---

## 7. Known limitations and what the next phase must handle

1. **`ambuildempty` writes the main fork, not the init fork.** This matches the IVF AM on this
   branch, but a fully correct implementation must also write the `INIT_FORKNUM` image that crash
   recovery copies for **unlogged** relations; that needs a fork-aware page writer. Logged tables
   (everything tested) are unaffected.
2. **No reclamation.** Superseded directory items and retired chains are not freed yet — one
   directory item leaks per seal (bounded, one per `hot_segment_max_rows` inserts). Old-generation
   reclamation is plan Phase 10; the directory already records `generation`/`epoch` and the meta
   page has room for a free list (the `ivf` free-list allocator is directly reusable).
3. **FLAT is exhaustive**, so a scan materialises `O(live rows)` candidates and reads every
   segment. That is correct but not the target performance: the bounded `search_candidates` path
   and the router (phase 7/8) are what make search sublinear.
4. **Sealed segments can still hold an active chain** (a row appended by a transaction that raced
   the seal). Scans read it; the migration planner must drain it along with the frozen runs.
5. **The directory's counts are publication-time snapshots**, refreshed when the directory is
   republished (which today happens on every seal); the segment headers are authoritative and are
   what `agentvec_index_info()` reports. Phase 4 may refresh them on its own republications too.
6. **No parallel scan/build** (`amcanparallel = amcanbuildparallel = false`). Feasibility is
   established for PG 18 (phase 0 record §0.3) but the AM-side partitioning is phase 9 work.
7. **`USING agentvec` on a relation whose opclass is not one of the three** is rejected by
   PG itself; `agentvec_validate` accepts anything else it is handed.

### Recommended next step (Phase 2 — HOT HNSW)

1. Spike the external-adapter route (H1 in the phase 0 record): register a pgvector `hnsw`
   relation as an `External` segment and drive it with
   `index_open`/`index_beginscan`/`index_rescan`/`index_getnext_slot` from a `SegmentExecutor`.
   Success criterion: a `SegmentExecutor` implementation that returns candidates for a query
   vector without touching AgentVec's directory from inside the scan.
2. If the spike fails or is too fragile, port the `hnswsq` branch's page-based HNSW as an
   `Owned` `Hnsw` segment behind the same trait.
3. Either way, phase 2 adds the second `SegmentAlgorithm`, which is the first real test of
   whether the segment abstraction in `directory.rs` holds.
