# Neon storage layer on Kubernetes (k3s) — 3 pageservers + 3 safekeepers

Date: 2026-09-02 — Server: `root@113.44.106.182` (single-node k3s v1.29.10, cgroup v1 host).

## 1. Cluster

- Installed with the Rancher China mirror (GitHub is GFW-blocked from this box):

  ```bash
  curl -sfL https://rancher-mirror.rancher.cn/k3s/k3s-install.sh -o /tmp/k3s-install.sh
  INSTALL_K3S_MIRROR=cn INSTALL_K3S_SKIP_SELINUX_RPM=true \
    INSTALL_K3S_VERSION=v1.29.10+k3s1 sh /tmp/k3s-install.sh   # v1.29: host uses cgroup v1
  ```

- Registry mirror for docker.io (pause image etc.):
  `/etc/rancher/k3s/registries.yaml` → `mirrors.docker.io.endpoint = https://docker.m.daocloud.io`, then `systemctl restart k3s`.

## 2. Container image (no docker needed)

The storage binaries are dynamically linked against glibc only, so a scratch
image suffices. `.design/neon/scripts/build_oci_image.py` builds a
docker-archive tarball directly (rootfs → layer.tar.gz + config.json +
manifest.json with correct `diff_ids`), which is imported into k3s's
containerd:

```bash
mkdir -p /tmp/imgbuild/usr/local/bin /tmp/imgbuild/lib64
cp /data1/neon-test/target/release/{pageserver,safekeeper,storage_broker,storage_controller,neon_local} /tmp/imgbuild/usr/local/bin/
cp /usr/lib64/{ld-linux-x86-64.so.2,libc.so.6,libm.so.6,libgcc_s.so.1} /tmp/imgbuild/lib64/
python3 build_oci_image.py /tmp/imgbuild /tmp/neon-storage.tar neon-storage:latest /usr/local/bin/safekeeper
k3s ctr -n k8s.io images import /tmp/neon-storage.tar      # must be the k8s.io namespace (CRI)
```

Gotchas solved along the way: layer must be gzipped and `rootfs.diff_ids`
must be the sha256 of the *uncompressed* tar; pods must set
`imagePullPolicy: IfNotPresent` (tag `latest` forces `Always`).

## 3. Pods

All pods run `hostNetwork: true` with hostPath volumes into
`/data1/neon-test/.neon`, so they use the existing broker (50051) and
controller (1234) exactly like the host processes.

| pod | component | ports (pg / http) | notes |
|---|---|---|---|
| ps2, ps3, ps4 | pageserver (ids 2–4) | 64001/9890, 64002/9891, 64003/9892 | standby nodes: re-attached to the controller with 0 tenants; tenant shard stays on the host pageserver (node 1) |
| sk4, sk5, sk6 | safekeeper (ids 4–6) | 5457/7679, 5458/7680, 5459/7681 | **serve the `bench-k8s` branch as a 3/3 WAL quorum** |

Example pod manifest (safekeeper):

```yaml
apiVersion: v1
kind: Pod
metadata: {name: sk4, namespace: default}
spec:
  hostNetwork: true
  restartPolicy: Always
  containers:
  - name: sk4
    image: neon-storage:latest
    imagePullPolicy: IfNotPresent
    command: ["/usr/local/bin/safekeeper"]
    args: ["-D","/data1/neon-test/.neon/safekeepers/sk4","--id","4",
           "--listen-pg","127.0.0.1:5457","--listen-http","127.0.0.1:7679"]
    volumeMounts: [{name: data, mountPath: /data1/neon-test/.neon/safekeepers/sk4}]
  volumes: [{name: data, hostPath: {path: /data1/neon-test/.neon/safekeepers/sk4}}]
```

Pageserver pods are identical but run `pageserver -D /data1/neon-test/.neon/pageserver_N`
with `identity.toml` (`id=N`) and a `pageserver.toml` cloned from
`pageserver_1` with shifted ports (pg 6400N-1, http 989N-2, grpc 5105N).

## 4. The 3/3 WAL quorum (fresh branch served by the pod safekeepers)

The clean way to add safekeepers is a **fresh timeline** (joining an
existing timeline by hand requires a WAL bootstrap the dev control plane
does not orchestrate — attempted and documented in the session log):

```bash
neon_local timeline branch --branch-name bench-k8s --ancestor-branch-name main
# → timeline 3bc62070... at LSN 3/7FE0D388

# create the timeline on all three pod safekeepers with a 3-member mconf
for p in 7679 7680 7681; do
  curl -s -X POST -H 'Content-Type: application/json' \
    -d '{"tenant_id":"7c1458844321f0c7c9cd0c468f21ff93",
         "timeline_id":"3bc6207063fb2fedda213f44a5ca9607",
         "mconf":{"generation":1,"members":[{"id":4,"host":"127.0.0.1","pg_port":5457},
                                          {"id":5,"host":"127.0.0.1","pg_port":5458},
                                          {"id":6,"host":"127.0.0.1","pg_port":5459}],
                  "new_members":null},
         "pg_version":170004,"system_id":7660721711221088489,
         "wal_seg_size":16777216,"start_lsn":"3/7FE0D388","commit_lsn":"3/7FE0D388"}' \
    http://127.0.0.1:$p/v1/tenant/timeline
done

# safekeepers 4-6 must also be in .neon/config so neon_local resolves the ids
neon_local endpoint create benchk8s --branch-name bench-k8s --pg-port 55434
neon_local endpoint start benchk8s --safekeepers 4,5,6 --safekeepers-generation 1
```

Verified in the compute log: `walproposer connected to quorum of
safekeepers ... 3/3 total`, elected with voters {4,5,6}; after a write, all
three safekeepers report the identical `flush_lsn` (e.g. 3/7FE26920).

## 5. Benchmark impact

Tuning (LFC + shared_buffers + pageserver page cache) is what fixes latency;
the quorum adds only a small per-commit WAL-sync cost. BIGANN-10M, 100
queries, ivfrq (lists=1000, num_bits=1), p50/p99:

| probes | recall | 1-SK untuned | 1-SK tuned | **k8s 3-SK tuned** |
|---|---|---|---|---|
| 1 | 47.23% | 389.7 / 422.8 ms | 1.37 / 2.89 ms | **1.44 / 3.06 ms** |
| 8 | 88.50% | 418.4 / 457.2 ms | 2.02 / 4.49 ms | **2.15 / 4.78 ms** |
| 64 | 99.40% | 626.4 / 743.5 ms | 5.15 / 9.01 ms | **5.64 / 9.34 ms** |

So: **~120–280× from tuning**, then the 3-safekeeper k8s quorum keeps within
~5–10% of the tuned single-node (and matches vanilla PG17's 6.42 ms p50 at
99.3% recall).
