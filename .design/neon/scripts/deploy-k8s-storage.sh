#!/usr/bin/env bash
# Deploy the Neon storage layer (3 pageservers + 3 safekeepers) into a k3s
# cluster using the locally built neon binaries.  Run as root on the node.
#
# Environment (defaults match the reference box):
#   NEON_BIN_DIR   dir with pageserver/safekeeper/storage_broker/
#                  storage_controller/neon_local (default /data1/neon-test/target/release)
#   NEON_DATA      neon_local state dir (default /data1/neon-test/.neon)
#   IMAGE_TAG      image tag (default neon-storage:latest)
#   K8S_NS         namespace (default default)
#
# Steps: stage rootfs -> build OCI image (no docker) -> import into k3s
# containerd (k8s.io namespace) -> create pageserver/safekeeper data dirs
# with shifted ports -> kubectl apply the demo pods.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEON_BIN_DIR="${NEON_BIN_DIR:-/data1/neon-test/target/release}"
NEON_DATA="${NEON_DATA:-/data1/neon-test/.neon}"
IMAGE_TAG="${IMAGE_TAG:-neon-storage:latest}"
K8S_NS="${K8S_NS:-default}"

# 1. stage rootfs (scratch image: binaries + glibc bits)
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/usr/local/bin" "$STAGE/lib64"
for b in pageserver safekeeper storage_broker storage_controller neon_local; do
    [ -x "$NEON_BIN_DIR/$b" ] || { echo "missing binary: $NEON_BIN_DIR/$b" >&2; exit 1; }
    cp "$NEON_BIN_DIR/$b" "$STAGE/usr/local/bin/"
done
cp /usr/lib64/ld-linux-x86-64.so.2 /usr/lib64/libc.so.6 \
   /usr/lib64/libm.so.6 /usr/lib64/libgcc_s.so.1 "$STAGE/lib64/" 2>/dev/null \
   || cp /lib64/ld-linux-x86-64.so.2 /lib64/libc.so.6 /lib64/libm.so.6 /lib64/libgcc_s.so.1 "$STAGE/lib64/"

# 2. build + import image
TARBALL="/tmp/${IMAGE_TAG//[:\/]/_}.tar"
python3 "$HERE/build_oci_image.py" "$STAGE" "$TARBALL" "$IMAGE_TAG" /usr/local/bin/safekeeper
k3s ctr -n k8s.io images rm "docker.io/library/$IMAGE_TAG" 2>/dev/null || true
k3s ctr -n k8s.io images import "$TARBALL"

# 3. prepare data dirs (pageserver tomls cloned from pageserver_1 with
#    shifted ports: pg 6400(N-1), http 989(N-2), grpc 5105N)
for i in 2 3 4; do
    d="$NEON_DATA/pageserver_$i"
    mkdir -p "$d"
    echo "id=$i" > "$d/identity.toml"
    if [ -f "$NEON_DATA/pageserver_1/pageserver.toml" ]; then
        sed -e "s/64000/6400$((i-1))/" \
            -e "s/9898/989$((i-2))/" \
            -e "s/51051/5105$i/" \
            "$NEON_DATA/pageserver_1/pageserver.toml" > "$d/pageserver.toml"
    fi
done
mkdir -p "$NEON_DATA/safekeepers"/sk{4,5,6}

# 4. apply pods
if [ -n "$(command -v kubectl)" ] && kubectl cluster-info >/dev/null 2>&1; then
    kubectl apply -f "$HERE/k8s-pods-demo.yaml"
    echo "pods applied; check with: kubectl get pods -o wide"
else
    echo "kubectl not available or cluster not reachable; apply $HERE/k8s-pods-demo.yaml manually" >&2
    exit 1
fi
