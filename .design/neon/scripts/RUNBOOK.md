# Neon cluster — start & manual-test runbook

Reference box: `root@113.44.106.182` (Huawei Cloud EulerOS, x86_64). All Neon
components run as user **`pg17test`** (uid 1001) — files owned by root crash
the pageserver ("Permission denied: crashsafe_overwrite"). For background
startups use `setsid nohup ... < /dev/null &` so processes survive SSH exit.

## 0. The pieces (what you are starting)

| component | how it runs | port |
|---|---|---|
| controller metadata PG | host `pg_ctl -D /data1/pg17test/data` | 1235 |
| storage_broker | host binary | 50051 |
| storage_controller | host binary (`--dev`) | 1234 |
| pageserver node 1 | host binary, `-D .neon/pageserver_1` | pg 64000 / http 9898 / grpc 51051 |
| pageservers 2–4 | **k3s pods** ps2/ps3/ps4 (image `neon-storage:latest`) | 64001–3 / 9890–2 |
| safekeeper 1 | host binary, `.neon/safekeepers/sk1` | 5454 / http 7676 |
| safekeepers 4–6 | **k3s pods** sk4/sk5/sk6 (serve branch `bench-k8s` as a 3/3 quorum) | 5457–9 / 7679–81 |
| compute `main` | `neon_local endpoint start main` (compute_ctl + postgres) | 55432 |
| compute `benchk8s` | `neon_local endpoint start benchk8s --safekeepers 4,5,6 --safekeepers-generation 1` | 55434 |

(k3s = single-node v1.29.10; pods use hostNetwork + hostPath under
`/data1/neon-test/.neon`.)

## 1. Cold start, in order

```bash
# 1) controller's metadata database
su pg17test -s /bin/bash -c "/data1/pg17test/install/bin/pg_ctl -D /data1/pg17test/data -o '-p 1235' start"

# 2) host services (broker, pageserver_1, safekeeper sk1, controller)
su pg17test -s /bin/bash -c "cd /data1/neon-test && bash start_neon.sh"
#   (equivalently, the four nohup lines inside start_neon.sh)

# 3) k3s storage pods (3 pageservers + 3 safekeepers)
kubectl get nodes                        # k3s must be Ready
kubectl apply -f .design/neon/scripts/k8s-pods-demo.yaml
#   first time only: .design/neon/scripts/deploy-k8s-storage.sh
#   (stages rootfs -> build_oci_image.py -> k3s ctr -n k8s.io images import -> dirs -> apply)

# 4) compute endpoints
su pg17test -s /bin/bash -c "export PATH=/home/pg17test/.cargo/bin:/data1/neon-test/target/release:\$PATH; cd /data1/neon-test \
  && neon_local endpoint start main --start-timeout 600s \
  && neon_local endpoint start benchk8s --start-timeout 600s --safekeepers 4,5,6 --safekeepers-generation 1"
```

### Verify (each line should return something)

```bash
ss -tlnp | grep -E '1234|1235|5454|9898|55432|55434'
kubectl get pods -o wide                    # ps2..ps4, sk4..sk6 Running
curl -s http://127.0.0.1:9898/v1/status | head -c 200        # pageserver alive
su pg17test -s /bin/bash -c "/data1/neon-test/pg_install/v17/bin/psql \
  -h 127.0.0.1 -p 55432 -U cloud_admin -d postgres -At -c 'select version();'"
```

## 2. Manual tests

The branch `main` (55432) and branch `bench-k8s` (55434) both carry the
BIGANN-10M benchmark data: `items_10m` (10M rows, `vector(128)`),
`bench_queries` (100 queries), `gt_10m` (exact top-10 ground truth), plus the
`vectorscale` + `vector` extensions.

### 2.1 Smoke test (fresh cluster, no benchmark data needed)

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale;
CREATE TABLE t (id int PRIMARY KEY, v vector(8) NOT NULL);
INSERT INTO t SELECT i, (ARRAY[random(),random(),random(),random(),
                              random(),random(),random(),random()]::real[])::vector
FROM generate_series(1, 1000) i;
CREATE INDEX ON t USING ivf (v vector_l2_ops) WITH (lists = 8, num_bits = 1);
SET enable_seqscan = off;
SELECT id FROM t ORDER BY v <-> '[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]'::vector LIMIT 5;
```

### 2.2 One manual top-k query on the 10M set (with plan + timing)

```sql
SET enable_seqscan = off;
SET ivf.probes = 64;
SET ivf.top_k = 1000;
\timing on
SELECT id, embedding <-> (SELECT q FROM bench_queries WHERE qid = 0) AS dist
FROM items_10m ORDER BY embedding <-> (SELECT q FROM bench_queries WHERE qid = 0) LIMIT 10;
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
SELECT id FROM items_10m ORDER BY embedding <-> (SELECT q FROM bench_queries WHERE qid = 0) LIMIT 10;
```

### 2.3 Single-query recall vs ground truth

```sql
SELECT count(*) AS hits FROM gt_10m g WHERE g.qid = 0 AND g.ids @> ARRAY(
  SELECT id FROM items_10m ORDER BY embedding <-> (SELECT q FROM bench_queries WHERE qid = 0) LIMIT 10);
```

### 2.4 The full benchmark sweep (one engine at a time)

```bash
# drop the other vector index first so the planner has no choice
su pg17test -s /bin/bash -c "export PGHOST=127.0.0.1 PGPORT=55434 PGUSER=cloud_admin PGDATABASE=postgres; \
  setsid nohup bash /tmp/run_sweep.sh /data1/neon-test/pg_install/v17/bin/psql ivf ivfrq-k8s /tmp/sweep_k8s_ivf.csv \
  > /tmp/sweep.log 2>&1 < /dev/null &"
#   ivf: probes 1..256 | hnsw: ef_search 10..640
#   output CSV: label,engine,param,value,recall_at_10,p50_ms,p99_ms
```

## 3. Gotchas that cost us hours (quick reference)

| symptom | fix |
|---|---|
| postgres reaches "ready" then dies; `WAL proposer ... signal 11` | pageserver's sticky in-memory `corruption_detected` flag (from an earlier walredo failure) rides in feedback and neon HEAD NULL-derefs on it → **restart the pageserver** |
| `could not access file "neon"/"neon_rmgr"/"neon_walredo"` | pgxn extensions missing/mismatched in `/data1/neon-test/pg_install/v17/lib/postgresql` — rebuild against fork `1e01fcea` |
| `failed to get basebackup@0/...: invalid basebackup lsn` | endpoint pgdata poisoned → `rm -rf .neon/endpoints/<id>/pgdata` and restart (fresh basebackup) |
| `PANIC: Page ... evicted with zero LSN` | pgvector must be the Neon-patched build (`patch_pgvector_neon.py` + `-DNEON_SMGR`) |
| first reads of a freshly built index are ~100× slow | pages live in WAL-only layers (walredo per page) → aggressive pageserver compaction in `pageserver.toml` (`compaction_period="2s"`, `compaction_threshold=1`) + restart, then `VACUUM` |
| `Read request too large: 76 > 32` | old `.so` without the readv chunking (`89d19ed`) — rebuild/install current branch |
| `VACUUM cannot run inside a transaction block` | psql `-c` wraps in a txn → run VACUUM via `-f` |
| endpoints silently use old settings | the REAL spec source is `.neon/endpoints/<id>/postgresql.conf` (neon_local regenerates config.json from it); structured safekeeper fields need `endpoint start --safekeepers 1,2,3 --safekeepers-generation N` |

See also: `.design/neon/OPTIMIZATION-PLAN.md` (CU-aware tuning), `.design/neon/bench/K8S.md`,
`.design/neon/bench/RESULTS.md`.
