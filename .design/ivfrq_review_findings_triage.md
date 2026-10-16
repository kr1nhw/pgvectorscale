# Review-findings triage: page bounds, corruption errors, amvalidate

Ten findings from an automated review of the `agentvec` / `ivf` / `hnswsq` code,
triaged against `7ee69e0` and then either fixed or refuted.  Everything marked
*validated* was reproduced on the 1B box (32 vCPU, PG 17.11, `items_1b` with its
32 GB `agentvec` index and the exact 200-query ground truth); the fixes are in
`16743c4`, whose tree rebuilds byte-identically (`a1f5cec6…`) to the binary the
measurements below were taken with.

| # | Finding | Verdict | Outcome |
|---|---------|---------|---------|
| 1 | `parse_buffer` uses `lp_len` as a slice length | valid, fix as suggested is insufficient | `PageGetItemChecked` |
| 1b | not in the review: `IvfEntryReader::read_burst` compacts with an unbounded `lp_len` | valid, worse (a heap *write*) | bounded against the scratch |
| 2 | per-chain `Vec::new()` in `ambulkdelete` | partially valid | hoisted (legacy path, see below) |
| 3 | `hnswsq_validate` never checks support proc 1 | valid | shared validator, used by ivf/agentvec too |
| 4 | `Sq8Calibration::load` panics | partially valid | `corrupt_data` + the real gap: `dim` never checked |
| 5 | advisory-lock leak in the test scaffolds | **invalid** | no change |
| 6 | `debug_assert_ne!(key, 0)` compiled out in release | **invalid** (0 is unreachable) | comment records the invariant |
| 7 | raw pointer arithmetic over neighbor TIDs | valid as hardening | `neighbor_tids` bounds `count` against the stored length |
| 8 | `CurrentMemoryContext` restored only on success | partially valid | restore moved into `PgTryBuilder::finally` |
| 9 | `IvfListDirectory::load_at` panics | partially valid (no abort) | `corrupt_data` → `ERRCODE_DATA_CORRUPTED` |
| 10 | 50 000-element `Vec` per iteration | already fixed | the file was `hnswsq/smoke.rs`, deleted in `8e7c5ab` |

## Where the review was wrong

* **The suggested bound is not sufficient (1).** `lp_off` and `lp_len` are 15-bit
  fields read straight off the page, so `len <= BLCKSZ` passes for
  `lp_off = 8000, lp_len = 8192` and the slice still runs 8 KiB past the page.
  The invariant that matters is `lp_off >= SizeOfPageHeaderData && lp_off + lp_len
  <= BLCKSZ`, which is what `PageGetItemChecked` enforces.
* **A `panic!` here does not abort the backend (4, 9).** `panic = "unwind"` (both
  profiles) plus `#[pg_guard]` turns it into an `ereport(ERROR)`, so the review's
  stated risk is wrong.  The real defect is the SQLSTATE: corruption surfaced as
  `INTERNAL_ERROR`.  Every on-disk parse failure now goes through
  `util::corrupt_data`, which raises `ERRCODE_DATA_CORRUPTED` and still unwinds
  Rust frames so RAII buffer locks are released first.
* **`debug_assert_ne!` is not a latent bug (6).** Key 0 is unreachable for all
  three producers — a packed TID would have to be block 0 / offset 0, a
  shared-relptr key starts at `MAXALIGN(1)` because 0 is the relptr NULL
  encoding, and pointer keys come from `palloc`.  Promoting it to a hard assert
  would buy nothing and sit in the innermost ANN loop, so the invariant is
  documented instead.
* **The advisory lock cannot leak (5).** `pgrx_tests::client()` opens a new
  connection per call and `postgres::Client::drop` closes it when a panic unwinds
  the test thread, so the session-level lock dies with the session.

## Findings the review missed

* `Sq8Calibration::load` never checked `dim` against `mins`/`maxs`, so
  `Codec::new_sq8`'s `maxs[d]` indexed out of bounds for a truncated or foreign
  calibration (`init_support` now also cross-checks the metapage's dimensions).
* `read_burst` copied a page-supplied length straight into the compaction
  scratch — a single corrupt page was an out-of-bounds **write**.

## Measurement on the 1B index

A/B/A with the HEAD build bracketing the patched one (100 queries × 3 reps,
`items_1b_av` = 1 000 000 512 rows, 32 GB):

| config | HEAD-1 | patched | HEAD-2 |
|---|---|---|---|
| p=256 sc=8000 | 535 / 546 / 539 ms | 544 / 537 / 558 ms | 537 / 553 / 541 ms |
| p=256 sc=1000 | 463 / 470 / 470 ms | 464 / 527 / 463 ms | 463 / 470 / 471 ms |

Recall@10 is identical on both builds: 0.9905 (sc=8000) and 0.8685 (sc=1000).
The insert path was exercised at 1B scale too: 5 000 rows appended at
4 202 rows/s, all visible through the index, the HOT segment growing
19 859 → 24 859 entries.

## Error paths, with a deliberately corrupted index

Same binaries and `.so` on a scratch cluster (so `items_1b` is never touched);
each case asserts the SQLSTATE and that the backend survives:

```
oversized lp_len      XX001: item 1 lies outside the page (lp_off = 8112, lp_len = 30000, page size = 8192)
segment header 0xFF   XX001: agentvec: segment header parse failed: CheckBytesError(…)
neighbor tuple count  XX001: hnswsq: neighbor tuple at offset 2 records 3000 tids (18004 bytes) but stores 200 bytes
calibration 0xFF      XX001: hnswsq: calibration parse failed (280 bytes): CheckBytesError(…)
```

`amvalidate` (finding 3) is proven by A/B: an opclass created without
`FUNCTION 1` reports `true` on the HEAD build and `false` with the fix, while all
nine extension opclasses stay accepted — including the `agentvec vector_l2_ops`
that 17 indexes (and `items_1b_av`) use.

## Finding 2: covering a branch nothing produces any more

Finding 2's loop lives in `ambulkdelete`'s `SegmentAlgorithm::Flat` branch, and
Flat segments have not been produced since `1a85350` ("embed hnswsq as the HOT
segment") — no index on the box has one, so the rewrite would otherwise be code
that never runs.  `test_agentvec_flat_segment_bulkdelete_across_chains` covers it:

* it turns the empty HOT segment `CREATE INDEX` publishes into a legacy Flat
  segment (the flag lives in the directory entry and the header), so the
  **product's own insert path** appends real entries to it;
* it seals the active chain between four batches (2000 + 1500 + 1000 + 500 rows)
  so the segment ends up with **four chains** at ~130 pages each — the sweep has
  to carry its buffers across chains, which is exactly what the rewrite changed;
* it drives `ambulkdelete` directly with a synthetic dead-verdict callback (the
  idiom `hnswsq_bulkdelete_tombstones_direct` already uses, because VACUUM's own
  verdict depends on the cutoff concurrent backends can pin) and then asserts the
  sweep's stats, the header counters, that exactly the four killed entries are
  tombstoned, and that a scan agrees with the heap.  A second sweep must find
  nothing, which is what proves no entry is tombstoned twice.

The test is sensitive to the failure mode the rewrite could introduce, which is
the point: dropping either `clear()` from the loop makes it fail —

| variant | result |
|---|---|
| patched loop | ok |
| patched loop − `to_tombstone.clear()` | FAILED: `agentvec: entry 72/11 tombstoned twice` |
| patched loop − `entries.clear()` | FAILED: `agentvec: entry 72/11 tombstoned twice` |
| HEAD's per-chain `Vec::new()` loop + the same test | ok |

The last row is the useful control: the test is a regression test for the branch
itself, not something written to fit the rewrite.
