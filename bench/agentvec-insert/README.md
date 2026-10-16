# agentvec insert-throughput harness

Test tooling for measuring and validating `agentvec` foreground insert throughput.
The results it produced are written up in
[`.design/agentvec_insert_throughput_and_hot_pool.md`](../../.design/agentvec_insert_throughput_and_hot_pool.md);
this directory is the reproduction kit.

Deployed on the x86 box as `/data1/avbench`, with results appended to
`/data1/avbench/results.csv` and each run echoed to `run.log`.

## Prerequisites

* A release build of the extension installed in the cluster under test.  A
  **debug** build makes everything ~25x slower and silently invalidates every
  number here.
* `pgbench` and `psql` from the same PostgreSQL installation as the server.
* A table with 128-dim vectors to sample (`src_table`, default `items_1b`) and a
  `bench_queries(qid, q)` query table — both are overridable through
  `00_setup.sql`'s `-v` variables.

## Configuration

Everything is read from `env.sh`, which the other scripts source.  Override with
environment variables instead of editing it:

| variable | default | meaning |
|---|---|---|
| `PGBIN` | `/root/pg17-release/bin` | PostgreSQL `bin` directory |
| `PGHOST_` / `PGPORT_` / `PGUSER_` / `PGDATABASE_` | `/tmp` / `54331` / `pgtest` / `postgres` | connection target |
| `AVB_DIR` | `/data1/avbench` | where scripts and logs live |
| `AVB_RESULTS` | `$AVB_DIR/results.csv` | the measurement log |

## Walking through it

```bash
# 1. Baseline: the reported ~4k rows/s, and where it comes from.
psql -v pool_rows=200000 -f 00_setup.sql
./gen_scripts.sh                                   # per-target pgbench scripts
for t in avb_t_heap avb_t_btree avb_t_pgv avb_t_av; do
    ./reset.sh $t; ./prefill.sh $t 20000
    ./bench_insert_fixed.sh $t 8 40000 200000 "sweep-c8"
done

# 2. Is it CPU-bound or contention-bound?
./bench_insert_cpu.sh avb_t_av 32 15 200000      # CPU us/row + cores busy
./bench_insert_waits.sh avb_t_av 32 15 200000    # wait-event distribution

# 3. Does splitting the write path across K graphs help?  (It does not.)
./setup_multi.sh 8 20000 f8
for K in 1 2 4 8; do ./bench_multi.sh $K 32 15 200000 "K$K-c32"; done

# 4. Where to spend the CPU: layout, ef_construction, WAL, HOT sizing.
./wal_per_row.sh avb_t_av 5000
./fill_curve.sh avb_t_av 8 5000 13 200000
./recall_check.sh avb_t_f8 200

# 5. Insertion + query together (the production risk).
./bench_mixed.sh items_1b 8 items_1b 8 15 200000 10000 "mixed-8-8"

# 6. Correctness of the HOT pool, including the transaction guarantee.
./smoke_multi_hot.sh 54330
```

## Scripts

**Setup / state**

| script | purpose |
|---|---|
| `00_setup.sql` | pools sampled from real BIGANN data plus the four insert targets |
| `env.sh` | shared paths, connection, `PGOPTIONS`, result logging |
| `gen_scripts.sh` | bakes the target table into `ins_<tbl>.sql` / `q_<tbl>.sql` |
| `reset.sh` | truncate + reindex a target back to a pristine state |
| `prefill.sh` | fill a target to a given HOT-segment occupancy |
| `setup_multi.sh` | create K independent targets for the multi-hot emulation |

**Insert / query measurement**

| script | purpose |
|---|---|
| `bench_insert.sh` | fixed-duration insert run (rows/s = tps) |
| `bench_insert_fixed.sh` | fixed row count, so every client count grows the HOT graph identically |
| `bench_insert_cpu.sh` | insert run + CPU sampling |
| `bench_insert_waits.sh` | insert run + `pg_stat_activity` wait-event sampling |
| `bench_query.sh` | query-only throughput |
| `bench_mixed.sh` | insert and query concurrently, both throughputs + CPU |
| `bench_multi.sh` | K targets, clients split across them, aggregate rows/s — the multi-hot emulation, paired with `setup_multi.sh` |

**Diagnostics**

| script | purpose |
|---|---|
| `cpu_per_row.sh` | CPU microseconds per committed row, and cores busy |
| `cpu_sample.sh` | busy/user/sys/iowait/steal + runnable threads over a window |
| `waits.sh` | top wait events across active backends |
| `wal_per_row.sh` | WAL bytes per inserted row, per variant |
| `fill_curve.sh` | insert throughput as the HOT segment fills, and across the seal |
| `recall_check.sh` | recall@10 of the index against an exact seqscan top-10 |

**Build and correctness**

| script | purpose |
|---|---|
| `build.sh` | release build (AVX2/FMA) and optional install; logs the full compiler output |
| `smoke_multi_hot.sh` | end-to-end HOT-pool test: pool size, insert spread, scan merge, same-transaction visibility, rollback, seal, adaptive mode |

## Notes that cost time to learn

* `pgbench`'s extended protocol sends `:variables` as bind parameters, so a
  **table name cannot come from `-D`** — that is what `gen_scripts.sh` works
  around, and why the scripts use `--protocol=prepared`.
* Comparing client counts with a fixed *duration* confounds the result with the
  HOT graph size, which grows differently in each run.  Use
  `bench_insert_fixed.sh` for scaling curves.
* Reconciling `pgbench`'s steady-state tps needs pgbench's own progress line
  (`--progress=1`); its summary average includes the ramp-up.
* `hot_segment_count` needs the build from this change; older binaries reject
  the reloption outright, and `00_setup.sql` will fail at `CREATE INDEX` with
  "unrecognized parameter".
