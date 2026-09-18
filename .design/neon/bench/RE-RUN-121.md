# Re-running the host-121 gap study (post-retirement engine)

The 1M-row BIGANN A/B on `121.37.117.106` (`items_1m`, dim 128, m=16, efc=64)
must be re-run against the **ported** hnswsq: the numbers in
`RESULTS-HNSWSQ.md` and `.design/hnswsq_vs_pgvector_gap.md` are from the
retired engine. The scripts now match the current GUC surface — the retired
engine's `hnswsq.build_stats / build_backlink_mode / build_engine /
build_backfill` GUCs are gone and no longer referenced.

## 0. Reach the box

`ssh root@121.37.117.106` (Huawei Cloud; from the office network — plain
internet port 22 is filtered). The study runs as root with `sudo -u pgtest`
against the server on `127.0.0.1:54330`.

## 1. Sync the repo

```sh
cd /data1/pgvectorscale-hnswsq
git fetch origin && git checkout <study commit>
# the .design/neon/bench scripts are part of the repo
```

## 2. Rebuild + install the extension in RELEASE

The pgrx toolchain lives in `/root/.pgrx-hnswsq`:

```sh
export PGRX_HOME=/root/.pgrx-hnswsq
cd /data1/pgvectorscale-hnswsq/pgvectorscale
cargo pgrx install --release --pg-config \
  /root/.pgrx-hnswsq/17.11/pgrx-install/bin/pg_config \
  --no-default-features --features pg17
```

## 3. Restart the server and guard the build

```sh
# restart the 54330 server (see the box's notes; systemd or pg_ctl)
sudo -u pgtest /root/.pgrx-hnswsq/17.11/pgrx-install/bin/pg_ctl \
  -D <data dir> restart -w
stat -c %s /root/.pgrx-hnswsq/17.11/pgrx-install/lib/postgresql/vectorscale-0.9.0.so
# release is ~2.3 MB; anything > 8 MB is the DEBUG build from `cargo pgrx test`
# — every number measured with it is ~25x slow and worthless.
```

Sanity: `SHOW hnswsq.ef_search; SET hnswsq.build_seed = 20240912;` must both
work, and a scratch `CREATE INDEX ... USING hnswsq ... WITH (storage_layout =
'plain', m = 16, ef_construction = 64)` must build.

## 4. Run the study

```sh
cd /data1/pgvectorscale-hnswsq/.design/neon/bench
./gap_study.sh          # ~15 min: builds + sweeps + perf profiles, /tmp/gap_study.log
```

Notes:

- The phase-split line (`hnswsq build stats: ...`) only logs when the
  extension is built **with the pg_test feature** — optional, never fatal for
  the scripts.
- `gap_sweep.py` expects the `bench_queries(qid, q)` table and
  `items_1m(embedding vector(128))`; both exist on the box.
- Single-backend vs parallel pgvector builds are `max_parallel_maintenance_workers`
  = 0 / server default / 32, all inside the script.
- Copy the CSV + log back and update `RESULTS-HNSWSQ.md` (the pre-retirement
  numbers stay in the History section).
