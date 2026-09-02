# Recommended Neon setup (self-hosted dev stack on x86)

Recommendations distilled from the BIGANN-10M benchmark session. The full
deployment narrative is in `../bench/K8S.md` and `../bench/RESULTS.md`; this
file is the runnable recipe.

## 1. Compute tuning (biggest win, ~120–280× warm latency)

Neon dev computes default to `shared_buffers=1MB` and a **disabled** local
file cache, so every page read round-trips to the pageserver. Append
`recommended-neon-compute.conf` to the endpoint's `postgresql.conf`:

```bash
cat .design/neon/scripts/recommended-neon-compute.conf \
    >> /data1/neon-test/.neon/endpoints/main/postgresql.conf
neon_local endpoint stop main && neon_local endpoint start main
```

Verify: `SHOW shared_buffers; SHOW neon.max_file_cache_size; SHOW neon.file_cache_size_limit;`

CU-aware scaling (production): 1 CU = 1 vCPU + 2 GB RAM; typical computes are
2 CU (4 GB) to 8 CU (16 GB). The 8 GB LFC above matches an ~8 CU dev box —
scale down per size: 2 CU → LFC 2–2.5 GB / shared_buffers 256–512 MB;
4 CU → LFC 4–6 GB; keep `neon.file_cache_size_limit <= ~60%` of CU RAM.
See ../bench/../OPTIMIZATION-PLAN.md for the full memory-budget design.

## 2. Pageserver page cache (64MB default → 4GB)

```toml
# append to .neon/pageserver_1/pageserver.toml, then restart the pageserver
page_cache_size = 524288   # pages; 8192 = 64MB default
```

## 3. Kubernetes storage layer (3 pageservers + 3 safekeepers)

Prereqs: k3s on the node (see below), binaries built (`target/release`),
broker on 50051 and storage controller on 1234 already running.

```bash
.design/neon/scripts/deploy-k8s-storage.sh   # builds image, imports, applies pods
kubectl get pods -o wide                       # expect ps2/ps3/ps4 + sk4/sk5/sk6 Running
```

Pod manifests (demo, hostNetwork + hostPath): `k8s-pods-demo.yaml`.

k3s install on a China/GFW-affected or cgroup-v1 host:

```bash
curl -sfL https://rancher-mirror.rancher.cn/k3s/k3s-install.sh -o /tmp/k3s-install.sh
INSTALL_K3S_MIRROR=cn INSTALL_K3S_SKIP_SELINUX_RPM=true \
  INSTALL_K3S_VERSION=v1.29.10+k3s1 sh /tmp/k3s-install.sh   # v1.29 = cgroup v1 support
cat > /etc/rancher/k3s/registries.yaml <<'EOF'
mirrors:
  docker.io:
    endpoint:
      - "https://docker.m.daocloud.io"
EOF
systemctl restart k3s
```

Image without docker: `build_oci_image.py` writes a docker-archive tarball
(gzipped layer + correct `diff_ids`); import with
`k3s ctr -n k8s.io images import <tarball>` (must be the **k8s.io**
namespace), and keep `imagePullPolicy: IfNotPresent` in pods (tag `latest`
otherwise forces a registry pull).

## 4. 3/3 WAL quorum on the pod safekeepers

Join by creating a **fresh branch** served by the three pod safekeepers
(joining an existing timeline by hand is not orchestrated by the dev control
plane and panics safekeepers — see K8S.md):

```bash
neon_local timeline branch --branch-name bench-k8s --ancestor-branch-name main
# → timeline <TLID> at LSN <LSN>

for p in 7679 7680 7681; do
  curl -s -X POST -H 'Content-Type: application/json' \
    -d "{\"tenant_id\":\"7c1458844321f0c7c9cd0c468f21ff93\",
         \"timeline_id\":\"<TLID>\",
         \"mconf\":{\"generation\":1,
                    \"members\":[{\"id\":4,\"host\":\"127.0.0.1\",\"pg_port\":5457},
                                {\"id\":5,\"host\":\"127.0.0.1\",\"pg_port\":5458},
                                {\"id\":6,\"host\":\"127.0.0.1\",\"pg_port\":5459}],
                    \"new_members\":null},
         \"pg_version\":170004,\"system_id\":<SYSID>,\"wal_seg_size\":16777216,
         \"start_lsn\":\"<LSN>\",\"commit_lsn\":\"<LSN>\"}" \
    http://127.0.0.1:$p/v1/tenant/timeline
done

neon_local endpoint create benchk8s --branch-name bench-k8s --pg-port 55434
neon_local endpoint start benchk8s --safekeepers 4,5,6 --safekeepers-generation 1
```

(`<SYSID>` from the main compute log's ProposerGreeting, e.g. 7660721711221088489.)
Verify in `endpoints/benchk8s/compute.log`: `walproposer connected to quorum
of safekeepers ... 3/3 total`, and all three safekeepers report the identical
`flush_lsn` after a write.

## 5. Expected result (BIGANN-10M, ivfrq lists=1000 num_bits=1)

| probes | recall | Neon untuned | Neon tuned | k8s 3-SK tuned | vanilla PG17 |
|---|---|---|---|---|---|
| 1 | 47.23% | 389.7 ms | 1.37 ms | 1.44 ms | 1.64 ms |
| 8 | 88.50% | 418.4 ms | 2.02 ms | 2.15 ms | 3.07 ms |
| 64 | 99.40% | 626.4 ms | 5.15 ms | 5.64 ms | 6.42 ms |
