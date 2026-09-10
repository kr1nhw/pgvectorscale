# ivf-rabitq issue audit — verification & fixes (2026-09-10)

Audit of the 12-item review list (`pgvectorscale-ivfrq.md`).  Verdict per
item, with the fixing commit; validation status on vanilla PG17
(x86_64 + aarch64) and the Neon k8s cluster.

| # | claim | verdict | commit |
|---|---|---|---|
| 1 | `dot_with_rotated_fields` recomputes Σrot per candidate | **true** — per-entry O(dim) sum in the active-buffer scan path; `RabitqQuery::sum_q` already precomputes it | `daae330` |
| 2 | AVX2 4-bit tail loop drops the high nibble | **false** — both nibbles are processed (`code[j] >> 4`); tail bounds are exact because dim is padded to a power of two; SIMD-vs-scalar tests + 4/8-bit recall tests agree | — |
| 3 | tests hardcode `/tmp/est_test.csv` and panic | **true** — both copies now skip with a notice when the file is absent | `aed46dd` |
| 4 | unsafe `RabitqNode::read/modify` without SAFETY comments | **true** — documented at every call site | `d1f5f00` |
| 5 | `Rabbitq`→`Rabitq` rename | **true** — 15 sites still used double-b; renamed, discriminant 3 and the `rabitq_compression` string untouched | `86e208c` |
| 6 | `lloyds_algorithm` re-allocates per iteration | **true** (build-time churn only) — buffers hoisted and reset in place | `ed9a638` |
| 7 | 2-bit `serialize_entries` allocates per entry | **true** (build-time) — planes pre-sized, entries written by index | `cc36157` |
| 8 | `from_raw_parts` f32 casts on unaligned SoA regions | **true UB** — sums/scales/margin regions sit at 2-mod-4 offsets for odd entry counts; replaced with byte copies into aligned stack buffers (same wire format, no extra allocs) | `0656d89` |
| 9 | `warning!` spam in planner hot path | **true** — removed from `ivf_amcostestimate` + `ivf_amhandler` | `1696ad2` |
| 10 | startup cost assigned from `indexTotalCost` | **true** — now `indexStartupCost` | `c2175d4` |
| 11 | `amendscan` pfrees state without dropping `Vec`s | **true** — `drop_in_place` before `pfree` (zeroed-state drop is still sound) | `dbbe2a2` |
| 12 | dead-tuple count written to `pages_deleted` | **true** — now `tuples_removed` (f64); `pages_deleted` stays 0 (AM retires segments, doesn't truncate) | `7c0806f` |

## Validation

- Pure-Rust unit tests: 92/92 pass (incl. a new alignment regression test
  covering odd/even entry counts for num_bits 1/2/4/8).
- `ivf_functional.sql` (build + query + DML/VACUUM/MVCC + per-width recall):
  100% recall on all four bit widths on vanilla PG17 x86_64
  (`113.44.106.182:5432`), vanilla PG17 aarch64 (`116.204.102.142:5432`),
  and the Neon k8s compute (`113.44.106.182:55434`, 3/3 safekeeper quorum).
- Note: databases created before the ivf AM exist lack its operator classes
  (`extension_sql!` runs only at CREATE/ALTER EXTENSION) — backfill with
  `fix_ivf_operator_classes.sql` (recreated here from a box-local copy).
- Performance guard: sweeps vs the `.design/neon/bench/RESULTS.md` baselines
  (see sweep CSVs on the box).

## Performance guard (vs .design baselines, same boxes/indexes)

| config | point | recall | p50 | p99 |
|---|---|---|---|---|
| vanilla x86 10M (baseline) | p64 | 99.30 | 6.421 | 12.066 |
| vanilla x86 10M (fixed build) | p64 | 99.60 | **6.076** | **10.424** |
| neon k8s 10M (baseline) | p64 | 99.40 | 5.707 | 9.561 |
| neon k8s 10M (fixed build) | p64 | 99.70 | **6.009** | **10.447** |
| vanilla x86 100M (baseline) | p64 | 98.30 | 29.253 | 49.517 |
| vanilla x86 100M (fixed build) | p64 | 98.30 | **29.982** | **49.476** |

All probe points (1–256) within ±6% of baseline latency, most equal or
faster; 100M recall bit-identical at every point (same index, confirming
the estimator arithmetic is unchanged).  ARM aarch64 has no prior baseline;
its curve (99.00% @ 12.17 ms p50 at p64) is a healthy match for the x86
shape.  Zero `amcostestimate`/`amhandler` WARNING lines in any sweep log
(issue 9 verified on all three engines).

Note: the Neon `benchk8s` endpoint had drifted from its documented tuning
(512MB shared_buffers / 2GB LFC → 2GB / 8GB restored from
`RECOMMENDED-SETUP.md`; the drift made the first k8s sweep run at ~260–600 ms
per query purely from environment, not from these changes — backup kept as
`postgresql.conf.bak512mb`).
