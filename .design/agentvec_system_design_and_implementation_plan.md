# AgentVec System Design and Implementation Plan

## 1. Executive Summary

`AgentVec` is a PostgreSQL extension for agent/session RAG memory databases.

The design is based on a **segmented mixture-of-index architecture**:

```text
Logical Vector Index
        │
        ├── HOT  → small mutable HNSW cache
        │
        ├── WARM → IVF-RaBitQ segments
        │
        └── COLD → larger IVF-RaBitQ segments
```

The foreground transaction path is intentionally simple:

```text
INSERT
   ↓
HOT HNSW cache
   ↓
commit
```

When a HOT cache is full, asynchronous maintenance migrates its vectors into existing WARM/COLD segments or creates new segments.

The migration path is:

```text
HOT HNSW
   ↓
migration planner
   ↓
destination segment selection
   ↓
destination centroid assignment
   ↓
centroid-aware RaBitQ encoding
   ↓
posting-list construction
   ↓
new segment generation
   ↓
atomic publication
   ↓
old HOT segment retired
```

The design does **not** require physical vector data to remain immutable. Instead:

> **Logical `vector_id` identity is stable; physical vector representation is versioned and asynchronously replaceable.**

This allows RaBitQ codes to be regenerated using destination-local centroids, improving quantization efficiency and precision.

The implementation should be based on **pgvectorscale/PGRX as the extension foundation**, while preserving pgvector-compatible SQL, vector types, and operator semantics. The project should reuse useful pgvectorscale/pgvector implementation patterns without treating StreamingDiskANN as the core AgentVec architecture.

---

# 2. Design Goals

## 2.1 Functional goals

1. Support small-batch and row-level INSERT/UPDATE/DELETE through PostgreSQL Index AM interfaces.
2. Make newly committed writes immediately searchable.
3. Keep the foreground INSERT path bounded by a small mutable HOT HNSW cache.
4. Move accumulated vectors asynchronously into durable ANN segments.
5. Support multiple physical segments for one logical topic/group.
6. Support multiple ANN algorithms by segment level.
7. Use centroid-aware RaBitQ for compact and accurate WARM/COLD storage.
8. Support copy-on-write segment generation and atomic publication.
9. Support PostgreSQL MVCC, WAL, crash recovery, and Neon compute/storage separation.
10. Provide a storage layout that can later support parallel segment search without redesigning the data model.

## 2.2 Performance goals

| Metric | Target |
|---|---:|
| Index space reduction | ≥ 50% |
| Incremental insertion p99 | < 100 ms |
| Search p99 | < 50 ms |
| Knowledge-base Top-100 | < 500 ms |
| Large-scale index | 100M–1B vectors |
| Large build target | < 30 min, hardware-dependent |

Performance targets must always be specified together with:

- vector dimension
- recall target
- CPU/compute capacity
- storage throughput
- number of workers
- quantization configuration

---

# 3. Core Architectural Principle

Do not build one global HNSW.

Instead:

```text
                         AgentVec
                            │
                 ┌──────────┼──────────┐
                 │          │          │
                HOT        WARM       COLD
                 │          │          │
                HNSW      IVF-RQ     IVF-RQ
                 │          │          │
                 └──────────┼──────────┘
                            │
                     Segment Directory
                            │
                     Vector + postings
```

HNSW is the **write cache**.

IVF-RaBitQ is the **durable large-scale ANN representation**.

This avoids the memory-first architecture that causes large HNSW builds to spill after exceeding `maintenance_work_mem`.

---

# 4. HOT / WARM / COLD Leveling

## 4.1 HOT

Purpose:

- absorb new INSERTs
- provide immediate visibility
- handle small incremental updates
- minimize foreground latency

Properties:

```text
HOT
 ├── mutable
 ├── HNSW
 ├── small
 ├── high write rate
 └── short lifetime
```

The HOT segment is not expected to grow indefinitely.

When it reaches its configured threshold:

```text
HOT S100
   ↓
SEALED
   ↓
new HOT S101
```

Foreground INSERTs immediately continue against S101.

S100 is handed to asynchronous maintenance.

## 4.2 WARM

Purpose:

- recently consolidated vectors
- medium-sized segments
- relatively active search population

Properties:

```text
WARM
 ├── IVF-RaBitQ
 ├── moderate size
 ├── searchable
 ├── can receive migrated vectors through new generations
 └── periodically optimized
```

## 4.3 COLD

Purpose:

- stable, large knowledge-base data
- minimal mutation
- maximum compression and efficient disk search

Properties:

```text
COLD
 ├── large IVF-RaBitQ
 ├── mostly immutable generations
 ├── aggressively compressed
 └── optimized for high-throughput search
```

COLD may eventually use a disk-oriented graph algorithm if needed, but that is future work.

---

# 5. Logical Groups and Physical Segments

A logical topic/group can contain multiple physical segments.

```text
Logical Group: PostgreSQL
    ├── S21
    ├── S31
    ├── S42
    └── S53
```

Do not require:

```text
one topic = one physical segment
```

Instead:

> A logical group is a semantic routing/storage family; a physical segment is an executable ANN unit.

This allows a topic to grow without requiring continuous migration.

Example:

```text
PostgreSQL group

S21 = 50K
S31 = 70K
S42 = 80K
S53 = 60K
```

A query may activate multiple segments from the same group.

---

# 6. Stable Vector Identity

Every logical index entry has a stable `vector_id`.

```text
vector_id = 812937
```

The physical representation may change:

```text
version 1:
    HOT HNSW

version 2:
    S31 / centroid C7 / RaBitQ

version 3:
    S52 / centroid C11 / RaBitQ
```

The stable entity is:

```text
vector_id
```

The following are replaceable:

- segment membership
- IVF list
- centroid assignment
- RaBitQ code
- code version
- ANN topology

---

# 7. Storage Architecture

The index should be physically separated into three major layers.

```text
AgentVec Relation
│
├── SegmentDirectory
│
├── VectorStore
│
└── PostingStore
```

## 7.1 SegmentDirectory

Stores:

```text
segment_id
group_id
level             # HOT/WARM/COLD
state
generation
algorithm
centroid metadata
quantizer metadata
vector_count
live_count
posting_root
code_root
statistics
epoch
```

## 7.2 VectorStore

Stores physical vector representations:

```text
vector_id
code_version
centroid_id
TID / heap reference
RaBitQ code
norm / scale / correction factors
quantizer version
state
```

A vector can have multiple physical versions during asynchronous recoding.

Only one is current.

## 7.3 PostingStore

Stores ANN membership:

```text
segment_id
generation
list_id
vector_id
state
epoch
```

Posting membership is reorganizable independently from vector payload.

---

# 8. Centroid-Aware RaBitQ

A global centroid is not a hard requirement.

For IVF-RaBitQ, the preferred representation is:

```text
vector
   ↓
destination segment
   ↓
local IVF centroid
   ↓
residual / centroid-relative representation
   ↓
RaBitQ code
```

Each physical representation stores:

```text
vector_id
centroid_id
code_version
RaBitQ code
```

This allows asynchronous recoding when vectors move to a different segment or a segment's centroid structure changes.

## Important invariant

> RaBitQ code is a physical optimization artifact, not the logical identity of the vector.

---

# 9. INSERT Path

The foreground path must be extremely small.

```text
aminsert()
    │
    ▼
get current HOT segment
    │
    ▼
HNSW insert
    │
    ▼
mark HOT maintenance-needed if threshold reached
    │
    ▼
return
```

Do not perform:

- large migration
- centroid retraining
- old-vector re-encoding
- segment split
- segment merge
- large posting-list rewrite

inside the user INSERT transaction.

The objective is:

> `aminsert()` performs bounded local work and lets asynchronous maintenance handle large-scale organization.

---

# 10. HOT Cache Lifecycle

When a HOT cache becomes full:

```text
HOT S100
   │
   ├── seal
   │
   └── create HOT S101
```

The transaction that observes the threshold should only perform the small metadata changes required to ensure future INSERTs use S101.

S100 becomes a maintenance input.

```text
S100 → QUEUED_FOR_MIGRATION
```

Foreground INSERTs continue on S101.

---

# 11. Asynchronous Migration Planner

The migration planner examines a sealed HOT segment.

For each vector batch:

```text
HOT vector batch
      ↓
determine logical group
      ↓
find suitable existing destination segment
      │
      ├── suitable → migrate to existing segment
      │
      └── unsuitable → create new segment
```

Destination selection considers:

- semantic/group affinity
- centroid distance
- segment size
- current level
- deletion ratio
- search frequency
- estimated build cost
- available capacity

---

# 12. Migration Into Existing Segments

Example:

```text
HOT S100
    │
    ├── 20K vectors → existing S21
    ├── 15K vectors → existing S31
    └── 15K vectors → new S52
```

Do not mutate an actively published segment in place.

Instead build a new generation:

```text
S21 generation 7
        ↓
build
        ↓
S21 generation 8
        ↓
atomic publication
```

The old generation remains available to readers using an older snapshot until safe reclamation.

---

# 13. Creating New Segments

If no existing segment is suitable:

```text
HOT S100
    ↓
new segment S150
    ↓
train/choose destination centroids
    ↓
assign vectors to centroids
    ↓
RaBitQ encode
    ↓
build posting lists
    ↓
seal S150
    ↓
publish S150
```

This gives a clean copy-on-write construction path.

---

# 14. Physical Migration and Recoding

Migration has two separate concepts:

### Logical migration

```text
vector_id
    ↓
destination segment/generation
```

### Physical recoding

```text
original vector
    ↓
destination centroid
    ↓
new RaBitQ code
```

They should not be confused.

The preferred publication sequence is:

```text
read source
  ↓
construct destination generation
  ↓
encode using destination centroids
  ↓
build postings
  ↓
validate
  ↓
publish
  ↓
retire source
```

Thus a newly published generation is internally centroid/code consistent from the first moment it becomes visible.

---

# 15. Batch Migration

Migration must be batch-oriented.

Example:

```text
S100 = 200K vectors

batch 1 = 5K
batch 2 = 5K
...
batch N = 5K
```

Each batch performs:

```text
read heap vectors
   ↓
destination assignment
   ↓
centroid assignment
   ↓
RaBitQ batch encode
   ↓
posting construction
   ↓
write pages
```

Benefits:

- bounded memory
- bounded WAL
- bounded transaction size
- easier retry
- easier progress reporting
- better storage locality

---

# 16. Vector Code Versioning

Use versioned physical representations.

Conceptually:

```text
vector_id = 812937

Code v14
    centroid = C2
    state = OBSOLETE

Code v15
    centroid = C7
    state = CURRENT
```

The current code can be switched with a small metadata update.

Old code versions can be reclaimed asynchronously after they are no longer visible to any active reader.

---

# 17. Search Path

For a query:

```text
q
```

search:

```text
HOT HNSW
   +
activated WARM IVF-RaBitQ segments
   +
activated COLD IVF-RaBitQ segments
```

Conceptually:

```text
q
 │
 ▼
Segment Router
 │
 ├── HOT
 ├── S21
 ├── S31
 └── S42
 │
 ▼
local ANN search
 │
 ▼
candidate merge
 │
 ▼
optional exact rerank
 │
 ▼
Top-K
```

---

# 18. Deterministic Router — MVP

No learned model in the initial system.

The router should use inexpensive metadata:

```text
query embedding
     ↓
group/segment prototype similarity
     ↓
top-M segment activation
```

The activated list becomes immutable scan state.

Later, the router can evolve into:

```text
query
  ↓
small learned model
  ↓
latent representation
  ↓
segment prototype matching
```

but the storage and search architecture do not depend on this future feature.

---

# 19. Scan State Design

The most important lock-avoidance decision is to make the activated segment list part of scan state.

Single-threaded:

```c
typedef struct VecScanOpaqueData
{
    Vector query;

    uint32 segment_count;
    VecSegmentRef *segments;

    uint32 current_segment;

    VecSegmentScan *segment_scan;

    CandidateHeap *candidate_heap;

    uint32 top_k;
} VecScanOpaqueData;
```

`amrescan()` computes:

```text
segments[]
```

and that list remains immutable for the lifetime of the scan.

No worker needs to consult the live SegmentDirectory during ANN execution.

---

# 20. Future Parallel Scan State

Design the above structure so it can become:

```text
Shared:
    segment_refs[]
    generation
    next_segment

Worker-local:
    current_segment
    local ANN state
    local candidate heap
```

The segment list is immutable.

Only the task cursor is shared.

Workers claim work through:

```c
task = pg_atomic_fetch_add_u32(
    &shared->next_segment, 1);
```

This avoids normal-path locks.

---

# 21. PostgreSQL Index AM Integration

Use the standard PostgreSQL Index AM interfaces.

### `amestimateparallelscan()`

Reserve DSM space for:

```text
segment references
generation
next task
task metadata
```

### `aminitparallelscan()`

Initialize:

```text
generation
next_segment
segment_count
```

### `amrescan()`

Query-specific setup:

```text
extract query vector
    ↓
run deterministic router
    ↓
materialize segment references
    ↓
publish scan state
```

### `amgettuple()`

```text
claim/execute segment
    ↓
local ANN candidates
    ↓
return candidates
```

### `amparallelrescan()`

Reset:

```text
generation
next_segment
worker-local state
```

A target-version verification step is required because executor-level parallel index scan support depends on the PostgreSQL/Neon version and is not guaranteed merely by implementing AM callbacks.

---

# 22. Parallel Scheduling

Do not statically assign segments to workers.

Bad:

```text
worker 0 → S1
worker 1 → S2
worker 2 → S3
```

Better:

```text
segments sorted by estimated cost
        ↓
atomic next_segment
        ↓
workers dynamically claim tasks
```

Example:

```text
worker 0: S2 → S7
worker 1: S1 → S4 → S9
worker 2: S3 → S5
```

This handles the fact that:

- HNSW cache segments
- small IVF segments
- large IVF segments

can have very different costs.

---

# 23. Search Result Merge

For `Top-K = 100`:

```text
Router
   ↓
M segments
   ↓
local candidate generation
   ↓
Gather
   ↓
bounded Top-N
   ↓
optional exact rerank
   ↓
Top-100
```

The first parallel implementation should avoid a shared global Top-K heap.

Use:

```text
parallel local candidates
→ Gather
→ bounded Top-N
```

This minimizes synchronization.

---

# 24. Maintenance Worker Architecture

The preferred foundation is:

```text
PostgreSQL Background Worker
        ↓
AgentVec maintenance queue
        ↓
PostgreSQL transaction
        ↓
migration/build/recode
        ↓
WAL
        ↓
atomic publication
```

Rust/PGRX should improve implementation complexity, but it must not bypass PostgreSQL's transaction, buffer, WAL, or relation-management semantics.

Do not use:

```text
arbitrary Rust thread
    ↓
direct PostgreSQL page mutation
```

---

# 25. Maintenance Job Model

Define explicit jobs:

```text
MIGRATE_HOT
BUILD_SEGMENT
SPLIT_SEGMENT
MERGE_SEGMENTS
RECODE_SEGMENT
COMPACT_SEGMENT
```

A job contains:

```text
job_id
source_segment
target_segment
group_id
job_type
state
progress
source_epoch
target_generation
```

State machine:

```text
QUEUED
  ↓
CLAIMED
  ↓
READING
  ↓
ENCODING
  ↓
BUILDING
  ↓
READY
  ↓
PUBLISHING
  ↓
DONE
```

Failures return to a retryable state without exposing a partial destination segment.

---

# 26. Why pgvectorscale/PGRX Is the Preferred Foundation

The implementation should use pgvectorscale as the primary extension/runtime foundation because the project requires substantial asynchronous and stateful maintenance infrastructure.

Useful capabilities/patterns to leverage include:

- Rust/PGRX PostgreSQL extension structure
- disk-oriented ANN implementation patterns
- streaming build concepts
- quantization integration
- candidate generation/rescoring
- existing PostgreSQL extension packaging

StreamingDiskANN should be treated as a source of engineering patterns, not as the final AgentVec index structure.

The AgentVec logical design remains:

```text
HOT HNSW
    ↓
async migration
    ↓
WARM/COLD IVF-RaBitQ
    ↓
Segment Router
```

---

# 27. What to Reuse vs. What to Build

## Reuse / study

### From pgvector

- vector/halfvec data types
- operator classes
- distance semantics
- HNSW implementation concepts
- PostgreSQL Index AM integration
- SQL compatibility

### From pgvectorscale

- Rust/PGRX extension organization
- streaming/disk-oriented implementation patterns
- quantization infrastructure
- build/search operational patterns
- background maintenance architecture where reusable

## Build

- SegmentDirectory
- HOT/WARM/COLD lifecycle
- Segment Router
- Migration Planner
- migration generations
- centroid-aware segment recoding
- PostingStore abstraction
- versioned VectorStore
- Segment MoI execution
- AgentVec-specific maintenance jobs
- eventual segment-parallel scheduler

---

# 28. Module Structure

Recommended implementation layout:

```text
agentvec/
├── src/
│   ├── am/
│   │   ├── mod.rs
│   │   ├── build.rs
│   │   ├── insert.rs
│   │   ├── scan.rs
│   │   └── parallel.rs
│   │
│   ├── segment/
│   │   ├── directory.rs
│   │   ├── lifecycle.rs
│   │   ├── generation.rs
│   │   └── publish.rs
│   │
│   ├── storage/
│   │   ├── vector_store.rs
│   │   ├── posting_store.rs
│   │   └── page_format.rs
│   │
│   ├── ann/
│   │   ├── flat.rs
│   │   ├── hnsw.rs
│   │   └── ivf_rabitq.rs
│   │
│   ├── routing/
│   │   └── centroid_router.rs
│   │
│   ├── maintenance/
│   │   ├── worker.rs
│   │   ├── queue.rs
│   │   ├── migrate.rs
│   │   ├── split.rs
│   │   ├── merge.rs
│   │   ├── recode.rs
│   │   └── vacuum.rs
│   │
│   └── wal/
│       └── records.rs
│
├── sql/
├── tests/
└── benchmarks/
```

The exact PGRX layout can be adapted to the selected pgvectorscale version.

---

# 29. Implementation Phases

## Phase 0 — Technology validation

Before writing the new index:

1. Check the target pgvectorscale version and license.
2. Validate PGRX support for the target PostgreSQL version.
3. Determine how pgvectorscale implements background workers/maintenance.
4. Determine which pieces can be reused without coupling AgentVec to StreamingDiskANN.
5. Confirm exact Neon/PostgreSQL support for custom parallel index scans.

Deliverable:

```text
technology decision record
```

---

## Phase 1 — Extension skeleton

Implement:

- new PGRX extension
- SQL extension creation
- pgvector-compatible vector operators
- custom Index AM skeleton
- basic relation metadata
- SegmentDirectory

No advanced ANN yet.

Deliverable:

```text
CREATE INDEX ... USING agentvec
```

with correct PostgreSQL lifecycle handling.

---

## Phase 2 — HOT HNSW

Implement:

```text
HOT segment
    ↓
HNSW insertion
```

Support:

- INSERT
- UPDATE
- DELETE tombstones
- immediate search
- segment sealing

Measure:

```text
single insert
small-batch insert
concurrent insert
p99 latency
```

This is the first real production-critical path.

---

## Phase 3 — WARM IVF-RaBitQ

Implement:

- IVF segment
- local centroids
- centroid-aware RaBitQ
- posting lists
- local search
- exact rerank

At this stage, conversion can initially be manually triggered.

Deliverable:

```text
HOT HNSW + WARM IVF-RaBitQ
```

---

## Phase 4 — Maintenance Worker

Implement PostgreSQL background worker infrastructure.

Start with one worker.

Worker loop:

```text
load maintenance queue
      ↓
claim job
      ↓
open transaction
      ↓
execute bounded batch
      ↓
commit
      ↓
repeat
```

Do not introduce parallel maintenance yet.

---

## Phase 5 — HOT Migration

Implement:

```text
HOT sealed
   ↓
route vectors
   ↓
existing segment or new segment
   ↓
destination centroid assignment
   ↓
RaBitQ encoding
   ↓
build destination generation
   ↓
publish
```

This is the first complete end-to-end version.

---

## Phase 6 — WARM/COLD Leveling

Implement automatic promotion:

```text
HOT
 ↓
WARM
 ↓
COLD
```

Possible policies:

```text
HOT:
small + high write rate

WARM:
recently consolidated + moderate mutation/search

COLD:
large + stable + optimized
```

The exact thresholds should be benchmark-driven.

---

## Phase 7 — Segment Router

Initial deterministic version:

```text
query
 ↓
group/segment prototype similarity
 ↓
top-M segments
```

No machine learning.

The router must produce the immutable `VecScanState.segment_refs[]`.

---

## Phase 8 — Single-Threaded Full Search

Implement:

```text
HOT HNSW
+
selected WARM
+
selected COLD
+
candidate merge
+
exact rerank
```

At this point we should have the complete single-threaded logical system before adding parallel execution.

---

## Phase 9 — Parallel Segment Search

Add:

```text
DSM state
generation
segment_refs[]
atomic next_segment
worker-local ANN state
```

Then implement AM parallel callbacks.

Validate against the actual target PostgreSQL/Neon executor behavior.

---

## Phase 10 — Segment Split/Merge and Reorganization

Implement:

- split
- merge
- recode
- centroid refresh
- old-generation reclamation

All via copy-on-write generation publication.

---

## Phase 11 — Scale and Optimization

Target:

```text
10M
100M
1B
```

Tune:

- segment size
- HOT size
- WARM/COLD thresholds
- IVF list count
- probes
- RaBitQ bit width
- migration batch size
- maintenance frequency
- search segment activation
- parallel workers

---

## Phase 12 — Future Learned Router

Only after deterministic routing works:

```text
query history
   ↓
true-result segment labels
   ↓
tiny learned router
   ↓
prototype activation
```

The learned model should remain optional and must not modify the underlying storage model.

---

# 30. Initial Configuration Knobs

Example index parameters:

```text
hot_segment_max_rows
warm_segment_target_rows
cold_segment_target_rows

rabitq_bits
ivf_lists
ivf_probes

router_top_m
router_group_top_m

migration_batch_rows
maintenance_interval
maintenance_max_bytes

rerank_k
search_candidates
```

Future:

```text
learned_router = off
```

The MVP should default to the deterministic router.

---

# 31. Maintenance Policies

Example starting policy:

```text
HOT:
≤ 50K–100K vectors

WARM:
~100K–5M vectors

COLD:
> 5M vectors
```

These are initial benchmark hypotheses, not fixed product requirements.

Maintenance triggers can combine:

```text
row count
write rate
dead ratio
search frequency
segment size
centroid quality
migration cost
storage utilization
```

---

# 32. Important Correctness Rules

1. New committed vectors must immediately be searchable.
2. A partially built segment is never published.
3. Every published segment generation is internally consistent.
4. Segment publication is an atomic metadata operation.
5. Old generations remain valid for readers that already captured them.
6. Vector identity is stable.
7. Physical vector representations may be replaced asynchronously.
8. Foreground INSERT does not perform large migrations.
9. Background maintenance operates in bounded transactions.
10. All durable index state is PostgreSQL storage/WAL managed.
11. No search worker mutates ANN structures.
12. Learned routing never becomes a correctness dependency.

---

# 33. Critical Performance Principles

## Foreground path

```text
INSERT
 → HOT HNSW
 → WAL/commit
```

Must stay bounded.

## Maintenance path

```text
read
 → batch
 → encode
 → write
 → publish
```

Can consume available compute/storage asynchronously.

## Search path

```text
route
 → selected segments
 → local ANN
 → merge
 → rerank
```

Avoid global search.

## Parallel path

```text
immutable segment list
+
atomic task claiming
```

Avoid global locks.

---

# 34. Benchmark Plan

## Dataset sizes

```text
100K
1M
10M
100M
1B
```

## Dimensions

```text
768D
1024D
1536D
```

## Workloads

### Write-heavy

```text
70% INSERT
20% SEARCH
5% UPDATE
5% DELETE
```

### Search-heavy

```text
10% INSERT
80% SEARCH
5% UPDATE
5% DELETE
```

### Migration-heavy

```text
continuous HOT sealing
large background consolidation
```

## Metrics

```text
insert p50/p95/p99/p999
search p50/p95/p99/p999
recall@10
recall@100
bytes/vector
peak RSS
WAL volume
migration throughput
recode throughput
segment publication latency
build time
maintenance backlog
```

---

# 35. Key Milestones

### M1 — PostgreSQL Integration

`agentvec` creates an index and survives:

- INSERT
- UPDATE
- DELETE
- VACUUM
- restart

### M2 — HOT HNSW

Immediate transactional search with bounded INSERT latency.

### M3 — WARM IVF-RaBitQ

Successful asynchronous HOT → IVF-RaBitQ conversion.

### M4 — Migration Generations

Existing segments can receive migrated vectors through new generations.

### M5 — HOT/WARM/COLD

Automatic level management works.

### M6 — Deterministic Segment Router

Search avoids unnecessary segments.

### M7 — Single-threaded Full System

Complete end-to-end system with:

```text
HOT
+
WARM
+
COLD
+
router
+
migration
+
rerank
```

### M8 — Parallel Segment Search

Dynamic segment scheduling with AM parallel-scan infrastructure.

### M9 — 100M/1B Benchmark

Large-scale build/search/migration evaluation.

### M10 — Learned Router

Future optimization only.

---

# 36. Final Architecture

```text
                                SQL
                                 │
                                 ▼
                         PostgreSQL Index AM
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
        Foreground Path                       Maintenance Path
              │                                     │
          aminsert()                           Worker/Queue
              │                                     │
              ▼                                     ▼
         HOT HNSW Cache                    Migration Planner
              │                                     │
        cache reaches limit                 ┌───────┼────────┐
              │                              │       │        │
              ▼                              ▼       ▼        ▼
        Seal + new HOT                    existing  new    recode
              │                           segment segment  / merge
              │                              │       │        │
              └──────────────┬───────────────┴───────┴────────┘
                             ▼
                       Destination Build
                             │
                  centroid assignment
                             │
                         RaBitQ encode
                             │
                     posting construction
                             │
                       new generation
                             │
                     atomic publication
                             │
                             ▼
                    Segment Directory
                             │
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
              HOT           WARM          COLD
             HNSW          IVF-RQ         IVF-RQ
               │             │             │
               └─────────────┼─────────────┘
                             ▼
                       Segment Router
                             │
                  immutable scan state
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
             serial search      future parallel search
                                      │
                              atomic segment claiming
                                      │
                                      ▼
                                  Top-K merge
                                      │
                                      ▼
                                  rerank
                                      │
                                      ▼
                                    result
```

---

# 37. Final Engineering Principle

The complete system should be understood as:

> **An LSM-style, multi-level ANN index implemented as a PostgreSQL extension.**

The database semantics remain PostgreSQL's responsibility.

The vector system is responsible for:

```text
HNSW write cache
+
segment lifecycle
+
centroid-aware RaBitQ
+
IVF storage
+
segment routing
+
asynchronous consolidation
+
eventual parallel segment execution
```

The core invariant is:

```text
Stable:
    vector_id
    logical row identity

Replaceable:
    HOT/WARM/COLD membership
    segment generation
    centroid assignment
    RaBitQ code
    IVF posting
    ANN topology
```

This gives AgentVec the freedom to optimize its physical representation continuously without compromising transaction semantics or the foreground INSERT path.
