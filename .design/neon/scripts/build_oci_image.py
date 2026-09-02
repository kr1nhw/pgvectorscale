#!/usr/bin/env python3
"""Build a minimal docker-archive OCI image tarball from a rootfs dir (no docker needed)."""
import hashlib, io, json, os, tarfile, sys

root = sys.argv[1]          # e.g. /tmp/imgbuild
out = sys.argv[2]           # e.g. /tmp/neon-storage.tar
tag = sys.argv[3] if len(sys.argv) > 3 else "neon-storage:latest"
entrypoint = sys.argv[4].split(",") if len(sys.argv) > 4 else ["/usr/local/bin/safekeeper"]

# layer.tar = the rootfs (relative paths only)
raw_buf = io.BytesIO()
with tarfile.open(fileobj=raw_buf, mode="w") as t:
    for base, _, files in os.walk(root):
        for f in files:
            full = os.path.join(base, f)
            rel = os.path.relpath(full, root)
            t.add(full, arcname=rel, recursive=False)
raw = raw_buf.getvalue()
import gzip
layer_data = gzip.compress(raw)

config = {
    "architecture": "amd64",
    "os": "linux",
    "config": {
        "Env": ["PATH=/usr/local/bin:/usr/bin:/bin"],
        "Entrypoint": entrypoint,
        "WorkingDir": "/data",
    },
    "rootfs": {
        "type": "layers",
        "diff_ids": ["sha256:" + hashlib.sha256(raw).hexdigest()],
    },
    "history": [{"created_by": "python-oci-builder"}],
}
manifest = [{
    "Config": "config.json",
    "RepoTags": [tag],
    "Layers": ["layer.tar.gz"],
}]

with tarfile.open(out, "w") as t:
    def add_bytes(name, data):
        ti = tarfile.TarInfo(name)
        ti.size = len(data)
        t.addfile(ti, io.BytesIO(data))
    add_bytes("config.json", json.dumps(config).encode())
    add_bytes("manifest.json", json.dumps(manifest).encode())
    add_bytes("layer.tar.gz", layer_data)

print(f"wrote {out} ({os.path.getsize(out)//(1024*1024)} MB), tag={tag}")
